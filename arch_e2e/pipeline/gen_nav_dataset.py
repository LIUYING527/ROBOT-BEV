"""造 DiffusionDrive 导航训练集(模仿学习)。
专家示范 = 我们那条真实走完全程的参考轨迹(env.ref, 到goal, 穿窄口, 居中)。这是【训练标签】不是运行时输入。
沿 ref 密集采位姿 + 横向/朝向扰动(教 recovery: 从偏离态回到走廊中线并朝目标走, 治蹭墙)。
每样本: 渲 FPV(卡7 3DGS) + 记 pose + goal方向(ego指向前方~5m的ref点) + 专家轨迹标签(ref续延到ego)。
BEV 在训练时按 pose 几何投影(便宜), 不存。

产物: outputs/nav_ds_<session>/NNN.jpg + meta.npz(poses,goal_dirs,base_ego,merged_ply)
用法(卡7):
  CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python arch_e2e/pipeline/gen_nav_dataset.py --session colmapjoint_all
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

T = 8


def _advance(ref, i0, dist):
    """沿ref从i0向前走dist米, 返回到达的索引。"""
    acc = 0.0; j = i0
    while j + 1 < len(ref) and acc < dist:
        acc += float(np.linalg.norm(ref[j + 1] - ref[j])); j += 1
    return j


def resample_ref_from(ref, x, y, yaw, horizon, T=T, merge_ahead=1.0):
    """从位姿(x,y,yaw)沿ref续延到ego系(T,2)。
    ⭐recovery=前向斜切merge: 瞄准前方~merge_ahead米的ref点(不是垂直最近点), 故从偏离态是
       【平滑斜向并入】而非【先横跳后前进】的L形。"""
    d = np.hypot(ref[:, 0] - x, ref[:, 1] - y); i0 = int(d.argmin())
    i_start = _advance(ref, i0, merge_ahead)                     # 前向lookahead: 目标在前方非侧方
    pts = [ref[i_start]]; acc = 0.0; j = i_start
    while j + 1 < len(ref) and acc < horizon:
        acc += float(np.linalg.norm(ref[j + 1] - ref[j])); pts.append(ref[j + 1]); j += 1
    pts = np.array(pts)
    c, s = np.cos(yaw), np.sin(yaw); dd = pts - np.array([x, y])
    ego = np.stack([c * dd[:, 0] + s * dd[:, 1], -s * dd[:, 0] + c * dd[:, 1]], 1)
    ego = np.vstack([[0.0, 0.0], ego])                          # 含当前点
    seg = np.linalg.norm(np.diff(ego, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
    if arc[-1] < 1e-3:
        return np.zeros((T, 2), np.float32)
    tgt = np.linspace(arc[-1] / T, arc[-1], T)
    return np.stack([np.interp(tgt, arc, ego[:, 0]), np.interp(tgt, arc, ego[:, 1])], 1).astype(np.float32)


def goal_dir_ego(ref, x, y, yaw, ahead=5.0):
    d = np.hypot(ref[:, 0] - x, ref[:, 1] - y); i0 = int(d.argmin())
    acc = 0.0; j = i0
    while j + 1 < len(ref) and acc < ahead:
        acc += float(np.linalg.norm(ref[j + 1] - ref[j])); j += 1
    tx, ty = ref[j]; c, s = np.cos(yaw), np.sin(yaw); dx, dy = tx - x, ty - y
    ex, ey = c * dx + s * dy, -s * dx + c * dy; n = (ex * ex + ey * ey) ** 0.5 + 1e-6
    return np.array([ex / n, ey / n], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--stride_m", type=float, default=0.4, help="沿ref采位姿间距")
    ap.add_argument("--lat", type=float, nargs="+", default=[0.0, 0.4, -0.4, 0.7, -0.7], help="横向扰动(m,温和)")
    ap.add_argument("--dyaw", type=float, nargs="+", default=[0.0, 0.15, -0.15], help="朝向扰动(rad,温和)")
    args = ap.parse_args()

    from test_env import CorridorTestEnv
    env = CorridorTestEnv(skin_session=args.session)
    ref = env.ref.astype(np.float64)                            # 世界参考路径(专家示范)
    outdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}"); os.makedirs(outdir, exist_ok=True)
    P = np.load(os.path.join(ROOT, "outputs", f"obs_seq_{args.session}", "poses.npz"), allow_pickle=True)
    merged_ply = str(P["merged_ply"])

    # 沿 ref 按弧长采基准点
    seg = np.linalg.norm(np.diff(ref, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
    base_s = np.arange(0, arc[-1] - 0.5, args.stride_m)
    poses, goals, labels, fpv_paths = [], [], [], []
    n = 0
    for si in base_s:
        i = int(np.searchsorted(arc, si))
        i = min(i, len(ref) - 2)
        th = float(np.arctan2(ref[i + 1, 1] - ref[i, 1], ref[i + 1, 0] - ref[i, 0]))  # 路径切向
        px = th + np.pi / 2                                     # 横向垂线
        for off in args.lat:
            for dy in args.dyaw:
                x = ref[i, 0] + off * np.cos(px); y = ref[i, 1] + off * np.sin(px); yaw = th + dy
                if env._occupied(x, y):
                    continue
                lab = resample_ref_from(ref, x, y, yaw, horizon=1.6)   # 专家ego轨迹(T,2)
                # ⭐前向单调过滤: 丢掉倒退/不推进的样本(治GT脏), 且强制 x 非递减
                if lab[-1, 0] < 0.25:
                    continue
                lab[:, 0] = np.maximum.accumulate(lab[:, 0])           # x 前向单调(去微倒退)
                env.env.set_pose(x, y, yaw); env.env.render()
                fpv = env.env.frame()[:, :, ::-1].copy()        # RGB->BGR 存jpg
                fp = os.path.join(outdir, f"{n:04d}.jpg"); cv2.imwrite(fp, fpv)
                poses.append([x, y, yaw]); goals.append(goal_dir_ego(ref, x, y, yaw))
                labels.append(lab); fpv_paths.append(f"{n:04d}.jpg"); n += 1
        if n and (len(poses) % 50 < (len(args.lat) * len(args.dyaw))):
            print(f"  已生成 {n} 样本 (base_s={si:.1f}/{arc[-1]:.1f})", flush=True)
    np.savez(os.path.join(outdir, "meta.npz"),
             poses=np.array(poses, np.float32), goal_dirs=np.array(goals, np.float32),
             base_ego=np.array(labels, np.float32), fpv=np.array(fpv_paths), merged_ply=merged_ply)
    print(f"[done] {n} 样本 -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
