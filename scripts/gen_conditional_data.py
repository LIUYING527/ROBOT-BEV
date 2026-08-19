"""造【语义条件 -> 轨迹】训练数据(sim) —— Phase 2 可控性的数据端。

思路(见 docs/架构与现状_实时.md §A): 巡检机器人无标准 benchmark, 故自建。
在重建走廊里, 对每个位姿 + 每种语义条件, 脚本生成一条 ego 系 GT 轨迹(8 路点),
沿轨迹渲 FPV(discoverse) + 记录条件 + GT, 打成训练样本。快脑微调后学会"换条件->换轨迹"。

⚠️条件是【行为模板】级(cruise/inspect_left/inspect_right/exit/pause), 非学长 4 运动模式。
"靠近设备巡检"按设备在左/右实例化成 inspect_left/right(可控性对比最直观的就是 左vs右vs直行)。

两个模式:
  --viz_only  纯 numpy: 把各条件模板叠在占据图上出 PNG(秒出, 给人确认抽象对不对), 不渲染。
  (默认)      full: 沿参考轨迹采位姿 -> 渲 FPV + 存 dataset(需 discoverse_venv, 见文件尾用法)。

坐标: ego x前 y左(同 scripts/eval_diffusiondrive_nonreactive.py)。世界变换与 unroll 一致。
"""
import os, sys, argparse, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- 语义条件集(行为模板级) ----------------
# 4 个行为清晰的条件(两轴: 横向 左/直/右 + 停走)。"exit接近出口"轨迹等同 cruise, 作语义别名不单列。
CONDITIONS = ["cruise", "inspect_left", "inspect_right", "strafe_left", "strafe_right",
              "turn_left", "turn_right", "pause", "stop"]
COND_ZH = {"cruise": "直行巡航", "inspect_left": "靠近左侧设备巡检",
           "inspect_right": "靠近右侧设备巡检", "exit": "接近出口(直行加速)", "pause": "降质暂停"}
COND_COLOR = {"cruise": (0, 200, 255), "inspect_left": (0, 255, 0), "inspect_right": (255, 128, 0),
              "exit": (255, 0, 255), "pause": (128, 128, 128)}  # BGR

T = 8            # 未来路点数(同 NAVSIM/DiffusionDrive)
DT = 0.5         # s/步 -> 4s 视界


def make_traj(cond):
    """条件 -> ego 系 GT 轨迹 (T,3)=(x前, y左, yaw)。机器人尺度(慢)。"""
    i = np.arange(1, T + 1)
    if cond == "cruise":
        x = 0.40 * i; y = np.zeros(T)                       # 0.8 m/s 直行 -> 3.2m
    elif cond == "exit":
        x = 0.55 * i; y = np.zeros(T)                       # 1.1 m/s 直行加速 -> 4.4m
    elif cond == "inspect_left":
        x = 0.28 * i; y = 0.9 * (i / T) ** 2                # 慢 + 向左飘 <=0.9m
    elif cond == "inspect_right":
        x = 0.28 * i; y = -0.9 * (i / T) ** 2               # 向右飘
    elif cond == "pause":
        x = 0.05 * i; y = np.zeros(T)                       # 近乎停 -> 0.4m
    else:
        raise ValueError(cond)
    # yaw = 沿路径切线方向
    dx = np.diff(np.concatenate([[0.0], x])); dy = np.diff(np.concatenate([[0.0], y]))
    yaw = np.arctan2(dy, dx)
    return np.stack([x, y, yaw], 1).astype(np.float32)


def ego_to_world(traj, pose):
    """ego (T,3) -> 世界 (T,2)。pose=(x,y,th)。"""
    x, y, th = pose; c, s = np.cos(th), np.sin(th)
    wx = x + c * traj[:, 0] - s * traj[:, 1]
    wy = y + s * traj[:, 0] + c * traj[:, 1]
    return np.stack([wx, wy], 1)


# ---------------- 占据(复用 stage_b_eval / eval_nonreactive 同配方) ----------------
def read_ply_xyz(path, max_pts=1_500_000):
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.readline()
        lines = head.decode("ascii", "replace").splitlines()
        n = next(int(l.split()[-1]) for l in lines if l.startswith("element vertex"))
        props = [l.split()[-1] for l in lines if l.startswith("property")]
        assert props[:3] == ["x", "y", "z"], props[:3]
        stride = len(props)
        buf = np.frombuffer(f.read(n * stride * 4), dtype="<f4").reshape(n, stride)
    xyz = buf[:, :3].astype(np.float64)
    if len(xyz) > max_pts:
        xyz = xyz[np.random.RandomState(0).choice(len(xyz), max_pts, replace=False)]
    return xyz


def build_occ(xyz, res=0.08, robot_r=0.3):
    from scipy import ndimage
    z = xyz[:, 2]; floor = np.percentile(z, 2)
    m = (z - floor > 0.3) & (z - floor < 2.0)
    p = xyz[m][:, :2]
    lo = p.min(0) - 1.5; hi = p.max(0) + 1.5
    nx = int((hi[0] - lo[0]) / res) + 1; ny = int((hi[1] - lo[1]) / res) + 1
    grid = np.zeros((ny, nx), np.int32)
    ix = ((p[:, 0] - lo[0]) / res).astype(int); iy = ((p[:, 1] - lo[1]) / res).astype(int)
    np.add.at(grid, (iy, ix), 1)
    occ = grid >= 3
    occ = ndimage.binary_dilation(occ, iterations=max(1, int(robot_r / res)))
    return occ, lo, res, floor


def occupied(occ, lo, res, x, y):
    ix = int((x - lo[0]) / res); iy = int((y - lo[1]) / res)
    if ix < 0 or iy < 0 or iy >= occ.shape[0] or ix >= occ.shape[1]:
        return True
    return bool(occ[iy, ix])


def traj_feasible(world_xy, occ, lo, res, max_bad=1):
    bad = sum(occupied(occ, lo, res, wx, wy) for wx, wy in world_xy)
    return bad <= max_bad


# ---------------- 参考位姿(去程) ----------------
def forward_leg_poses(poses):
    """参考轨迹是去了又回; 只取起点->最远点的去程(避免掉头段)。"""
    d = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    far = int(d.argmax())
    return poses[:far + 1]


# ---------------- 可视化: 每位姿一个 ego 局部 panel(forward朝上) ----------------
def _occ_world_pts(occ, lo, res):
    """占据格 -> 世界坐标点(格中心)。"""
    iy, ix = np.where(occ)
    return np.stack([lo[0] + (ix + 0.5) * res, lo[1] + (iy + 0.5) * res], 1)


def corridor_halfwidth(pose, occ_world, default=1.4):
    """从局部占据稳健估走廊半宽(前方两侧最近墙)。噪点多->取较近分位, 兜底 default。"""
    x, y, th = pose; c, s = np.cos(th), np.sin(th)
    dx = occ_world[:, 0] - x; dy = occ_world[:, 1] - y
    ex = c * dx + s * dy; ey = -s * dx + c * dy
    ahead = (ex > 0.3) & (ex < 4.0) & (np.abs(ey) < 3.0)
    if ahead.sum() < 20:
        return default
    L = ey[ahead & (ey > 0)]; Rr = -ey[ahead & (ey < 0)]
    hw = np.median([np.percentile(L, 25) if len(L) else default,
                    np.percentile(Rr, 25) if len(Rr) else default])
    return float(np.clip(hw, 0.9, 2.2))


def ego_panel(pose, occ_world, S=480, ppm=95.0):
    """干净示意图(游戏感): 白底 + 规整走廊墙 + 机器人 + 5 粗彩轨迹。ego x前=上, y左=左。"""
    import cv2
    img = np.full((S, S, 3), 248, np.uint8)
    cx, cy = S // 2, int(S * 0.82)                            # 机器人靠下, 前方留空间
    def e2px(ex, ey):
        return int(cx - ey * ppm), int(cy - ex * ppm)
    # 规整走廊墙(浅灰粗线, 示意)
    hw = corridor_halfwidth(pose, occ_world)
    for side in (+1, -1):
        p0 = e2px(-0.3, side * hw); p1 = e2px(4.6, side * hw)
        cv2.line(img, p0, p1, (205, 205, 205), 10)
    cv2.putText(img, f"corridor ~{2*hw:.1f}m", (10, S - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    # 机器人(黑三角朝上)
    tri = np.array([e2px(0.32, 0), e2px(-0.12, 0.16), e2px(-0.12, -0.16)], np.int32)
    cv2.fillPoly(img, [tri], (0, 0, 0))
    # 5 条件轨迹(平滑粗线 + 终点标签)
    for cond in CONDITIONS:
        tr = make_traj(cond); col = COND_COLOR[cond]
        pts = np.array([(cx, cy)] + [e2px(px_, py_) for px_, py_ in tr[:, :2]], np.int32)
        cv2.polylines(img, [pts], False, col, 4, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 5, col, -1)
        lbl = cond if cond != "pause" else "pause(stop)"
        ox = 6 if "right" in cond or cond in ("cruise", "exit") else -110
        cv2.putText(img, lbl, (pts[-1][0] + ox, pts[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (S - 1, S - 1), (60, 60, 60), 1)
    return img


def viz(poses, occ, lo, res, sample_idx, out_png):
    import cv2
    occ_world = _occ_world_pts(occ, lo, res)
    panels = []
    for si in sample_idx:
        p = ego_panel(poses[si], occ_world)
        cv2.putText(p, f"pose#{si}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        panels.append(p)
    grid = np.hstack(panels)
    title = np.full((44, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(title, "same pose, different semantic condition -> different trajectory",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(out_png, np.vstack([title, grid]))
    print(f"[viz] {len(sample_idx)} 位姿 x {len(CONDITIONS)} 条件 (clean) -> {out_png}")


# ---------------- FPV 叠加: 把 ego 地面轨迹投到真实前视图(mujoco相机) ----------------
def project_ground(cam_pos, cam_mat, fovy_deg, W, H, world_pts_xy):
    """世界地面点(z=0) -> 像素。mujoco 相机看向 -z, +x右 +y上。返回 [(u,v)|None]。"""
    R = np.asarray(cam_mat, float).reshape(3, 3)          # 列=相机轴(世界系)
    f = (H / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    out = []
    for wx, wy in world_pts_xy:
        Pw = np.array([wx, wy, 0.0]) - np.asarray(cam_pos, float)
        Pc = R.T @ Pw                                     # 世界->相机
        z = -Pc[2]
        if z <= 0.05:
            out.append(None); continue
        u = W / 2.0 + f * Pc[0] / z
        v = H / 2.0 - f * Pc[1] / z
        out.append((int(round(u)), int(round(v))))
    return out


def fpv_overlay(session, sample_idx, out_png, width=960, height=540, fovy=75.0, model_trajs=None):
    """model_trajs: None=画模板; 否则 (Npose,ncond,8,3) 画【模型预测】轨迹。"""
    import cv2, mujoco
    from sim_walk_common import make_env
    env, W = make_env(session, width=width, height=height, fovy=fovy)
    poses = np.load(os.path.join(ROOT, "outputs", f"obs_seq_{session}", "poses.npz"),
                    allow_pickle=True)["poses"].astype(np.float64)
    fwd = forward_leg_poses(poses)
    cam_id = env.mj_model.camera("fpv").id
    src = "MODEL pred" if model_trajs is not None else "template"
    panels = []
    for si in sample_idx:
        pose = fwd[si]
        env.set_pose(*pose); env.render()
        img = env.frame()[:, :, ::-1].copy()              # RGB->BGR
        cpos = env.mj_data.cam_xpos[cam_id]; cmat = env.mj_data.cam_xmat[cam_id]
        for ci, cond in enumerate(CONDITIONS):
            tr = model_trajs[si, ci] if model_trajs is not None else make_traj(cond)
            col = COND_COLOR[cond]
            wxy = ego_to_world(tr, pose)
            uv = project_ground(cpos, cmat, fovy, width, height, wxy)
            # 起点=机器人脚下(ego原点地面)
            uv0 = project_ground(cpos, cmat, fovy, width, height, [ego_to_world(np.zeros((1, 3)), pose)[0]])[0]
            chain = [uv0] + uv
            for a, b in zip(chain[:-1], chain[1:]):
                if a and b:
                    cv2.line(img, a, b, col, 4, cv2.LINE_AA)
            if uv[-1]:
                cv2.circle(img, uv[-1], 6, col, -1)
                cv2.putText(img, cond, (uv[-1][0] + 6, uv[-1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"pose#{si}  [{src}]", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(img)
    cv2.imwrite(out_png, np.vstack(panels))
    print(f"[fpv] {len(sample_idx)} 帧真实FPV叠4条件轨迹({src}) -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--viz_only", action="store_true", help="只出模板叠占据图(纯numpy,不渲染)")
    ap.add_argument("--fpv_overlay", action="store_true", help="真实FPV上叠4条件【模板】轨迹(需discoverse_venv)")
    ap.add_argument("--fpv_model", action="store_true", help="真实FPV上叠4条件【模型预测】轨迹(读 model_trajs npz)")
    ap.add_argument("--idx", default="8,17,20", help="展示哪几个去程帧(逗号分隔)")
    ap.add_argument("--n_viz", type=int, default=6, help="viz 采样几个位姿")
    args = ap.parse_args()

    idx = [int(x) for x in args.idx.split(",")]
    if args.fpv_overlay or args.fpv_model:
        mt = None
        if args.fpv_model:
            npz = os.path.join(ROOT, "arch_e2e", "runs", f"model_trajs_{args.session}.npz")
            mt = np.load(npz, allow_pickle=True)["trajs"]
            print(f"[fpv] 载入模型预测轨迹 {mt.shape} <- {npz}")
        tag = "model" if args.fpv_model else "template"
        out = os.path.join(ROOT, "outputs", f"cond_fpv_{tag}_{args.session}.png")
        fpv_overlay(args.session, idx, out, model_trajs=mt)
        return

    seqdir = os.path.join(ROOT, "outputs", f"obs_seq_{args.session}")
    P = np.load(os.path.join(seqdir, "poses.npz"), allow_pickle=True)
    poses = P["poses"].astype(np.float64); merged_ply = str(P["merged_ply"])
    fwd = forward_leg_poses(poses)
    print(f"[data] 参考 {len(poses)} 位姿, 去程 {len(fwd)}; ply {merged_ply}")

    xyz = read_ply_xyz(merged_ply)
    occ, lo, res, floor = build_occ(xyz)
    print(f"  ply {len(xyz)}点 占据 {int(occ.sum())}格 floor={floor:.2f}")

    if args.viz_only:
        idx = np.linspace(1, len(fwd) - 1, args.n_viz).astype(int)
        out = os.path.join(ROOT, "outputs", f"cond_templates_{args.session}.png")
        viz(fwd, occ, lo, res, idx, out)
        return

    print("[data] full 模式(渲染)未实现于本次 —— 先看 viz 确认模板, 再放量。")


if __name__ == "__main__":
    main()
