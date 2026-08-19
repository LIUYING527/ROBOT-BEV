"""Phase2 微调 + 可控性验证 —— 一个脚本跑通"造数据用现有obs_seq -> 接cond -> 微调 -> 对比图"。

思路: 复用现有 obs_seq 的 FPV(去程~33位姿, 不重渲) 当观测; 每位姿 x 4 条件 = 训练样本;
GT 轨迹 = scripts/gen_conditional_data.make_traj(cond)(ego 8路点模板); cond_id 走 features
-> 模型 cond_embed 加到 time_embed -> 每层 FiLM 被条件调制 -> 扩散 loss 微调。
微调前所有条件输出相同(cond_embed零初始化); 微调后应"换cond->换轨迹"=可控性。

在【模型 env】跑(artifixer, 用 arch_e2e/vendor 已注入cond的模型):
  CUDA_VISIBLE_DEVICES=N /home/DongBaorong/micromamba/envs/artifixer/bin/python \
    arch_e2e/pipeline/train_conditional.py --session colmapjoint_all --epochs 60
产物: runs/cond_finetuned.pt, runs/controllability_<session>.png (微调前/后对比)
"""
import os, sys, argparse
import numpy as np
import cv2

from _paths import inject_vendor, WEIGHTS, RUNS, REPO
inject_vendor()
sys.path.insert(0, os.path.join(REPO, "scripts"))          # 借 make_traj / CONDITIONS
import torch
import torch.nn as nn
import stage_b_eval as sb                                   # 复用 load_model / 特征 / ply / occ
from gen_conditional_data import CONDITIONS, make_traj, ego_to_world, forward_leg_poses

COND_COLOR = {"cruise": (0, 200, 255), "inspect_left": (0, 255, 0), "inspect_right": (255, 128, 0),
              "strafe_left": (0, 255, 180), "strafe_right": (255, 0, 180),
              "turn_left": (180, 255, 0), "turn_right": (180, 0, 255),
              "pause": (150, 150, 150), "stop": (0, 0, 255)}  # BGR


ANCHOR_SCALE = 0.10                                            # 车尺度kmeans -> 机器人尺度(x最大47.7m*0.1=4.8m)


def robot_anchors():
    """⭐用学长/NAVSIM 的 kmeans 聚类轨迹(20锚点), 整体缩放到机器人尺度(不再手搓)。
    保留聚类出的 直行/左转/右转/短 形状分布, 只改尺度。见 compute_robot_plan_anchors 同思路。"""
    A = np.load(os.path.join(WEIGHTS, "kmeans_navsim_traj_20.npy")).astype(np.float32)  # (20,8,2)
    return (A * ANCHOR_SCALE).astype(np.float32)


def condition_anchor_sets(A):
    """把 20 个机器人尺度锚点按行为分到 4 个条件子集(任务驱动选择的'可选项')。"""
    end = A[:, -1, :]; reach = np.hypot(end[:, 0], end[:, 1])
    sets = {
        "pause":         np.where(reach < 0.6)[0],
        "inspect_left":  np.where(end[:, 1] > 0.45)[0],
        "inspect_right": np.where(end[:, 1] < -0.45)[0],
        "cruise":        np.where((np.abs(end[:, 1]) <= 0.45) & (reach >= 1.0))[0],
    }
    for k, v in sets.items():                                  # 兜底: 空集给最接近的
        if len(v) == 0:
            sets[k] = np.array([int(reach.argmin())] if k == "pause" else [int(reach.argmax())])
    return sets


def anchor_to_traj3(a2):
    """锚点 (8,2) -> GT (8,3), heading=沿路径切线。"""
    dx = np.diff(np.concatenate([[0.0], a2[:, 0]])); dy = np.diff(np.concatenate([[0.0], a2[:, 1]]))
    yaw = np.arctan2(dy, dx)
    return np.concatenate([a2, yaw[:, None]], 1).astype(np.float32)


# ---------------- ⭐真实路径 GT(模仿真实重建轨迹, 取代"编的"锚点子集) ----------------
def smooth_path_xy(poses, win=5):
    """对位姿 xy 做居中滑动平均(端点收窄), 去掉 VGGT 位姿噪声导致的锯齿。
    只平滑位置(GT 用), 不改朝向/不改缓存的 FPV/BEV。win<=1 时原样返回。"""
    if win is None or win <= 1:
        return poses
    p = poses.copy().astype(np.float64)
    xy = p[:, :2]
    n = len(xy); k = win // 2
    out = xy.copy()
    for i in range(n):
        a, b = max(0, i - k), min(n, i + k + 1)
        out[i] = xy[a:b].mean(0)
    p[:, :2] = out
    return p


def realpath_ego(poses, i, horizon=1.6, T=8):
    """第 i 帧位姿起, 未来真实路径转 ego(x前,y左), 按弧长重采到 horizon 米 T 点。
    这样 cruise 学的是"跟着真实走廊走"(走廊拐它也拐), 尺度天然合理, 不再死直线。"""
    x0, y0, th0 = poses[i]
    pts, acc, j = [poses[i, :2].astype(np.float64)], 0.0, i
    while j + 1 < len(poses) and acc < horizon:
        acc += float(np.linalg.norm(poses[j + 1, :2] - poses[j, :2]))
        pts.append(poses[j + 1, :2].astype(np.float64)); j += 1
    pts = np.array(pts)
    c, s = np.cos(th0), np.sin(th0)
    d = pts - np.array([x0, y0])
    ego = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)   # x前 y左
    seg = np.linalg.norm(np.diff(ego, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] < 1e-3:
        return np.zeros((T, 2), np.float32)
    tgt = np.linspace(arc[-1] / T, arc[-1], T)
    return np.stack([np.interp(tgt, arc, ego[:, 0]), np.interp(tgt, arc, ego[:, 1])], 1).astype(np.float32)


def _with_yaw(xy):
    dx = np.diff(np.concatenate([[0.0], xy[:, 0]])); dy = np.diff(np.concatenate([[0.0], xy[:, 1]]))
    return np.concatenate([xy, np.arctan2(dy, dx)[:, None]], 1).astype(np.float32)


def _yaw0(xy):
    """yaw 强制=0(面朝前): 给 strafe/stop 用——4轮横移时身体不转。"""
    return np.concatenate([xy, np.zeros((len(xy), 1))], 1).astype(np.float32)


def cond_gts_from_realpath(base_ego, lat=0.6, strafe=0.85):
    """真实未来路径(T,2) -> 7条件 GT (7,T,3), 顺序同 CONDITIONS。
    cruise=真实路径; inspect_L/R=路径+渐进斜移(边走边靠); strafe_L/R=纯横移(几乎不前进,面朝前滑过去);
    pause=缩短近停(慢爬); stop=零(停下读表)。"""
    T = len(base_ego)
    i1 = np.linspace(0, 1, T)
    off = lat * (i1 ** 2)                                       # inspect 渐进斜移
    tiny = 0.10 * np.arange(1, T + 1) / T                       # strafe 的极小前向
    parts = {
        "cruise":        _with_yaw(base_ego.copy()),
        "inspect_left":  _with_yaw(base_ego + np.stack([np.zeros(T), off], 1)),
        "inspect_right": _with_yaw(base_ego + np.stack([np.zeros(T), -off], 1)),
        "strafe_left":   _yaw0(np.stack([tiny, strafe * i1], 1)),     # 纯左横移
        "strafe_right":  _yaw0(np.stack([tiny, -strafe * i1], 1)),    # 纯右横移
        # 原地转(转视角): 位置≈0, yaw 渐变到 ±1.2rad(~70°)
        "turn_left":     np.concatenate([np.stack([tiny * 0.3, np.zeros(T)], 1), (1.2 * i1)[:, None]], 1).astype(np.float32),
        "turn_right":    np.concatenate([np.stack([tiny * 0.3, np.zeros(T)], 1), (-1.2 * i1)[:, None]], 1).astype(np.float32),
        "pause":         _with_yaw(base_ego * 0.15),
        "stop":          _yaw0(np.stack([0.02 * np.arange(1, T + 1) / T, np.zeros(T)], 1)),  # ~0
    }
    return np.stack([parts[c] for c in CONDITIONS])


def load_dataset(session, device):
    """返回 poses(去程), FPV camera_feature 缓存, BEV lidar_feature 缓存(每位姿一次)。"""
    seqdir = os.path.join(REPO, "outputs", f"obs_seq_{session}")
    P = np.load(os.path.join(seqdir, "poses.npz"), allow_pickle=True)
    poses = P["poses"].astype(np.float64); merged_ply = str(P["merged_ply"])
    fwd_mask = forward_leg_poses(poses)                      # 去程位姿(值)
    nfwd = len(fwd_mask)
    xyz = sb.read_ply_xyz(merged_ply)
    _, _, _, floor = sb.build_occ(xyz)
    cams, bevs, kept = [], [], []
    for i in range(nfwd):
        fp = os.path.join(seqdir, f"{i:03d}.jpg")
        fpv = cv2.imread(fp)
        if fpv is None:
            continue
        cams.append(sb.camera_feature(fpv, device))         # (1,3,256,1024)
        bevs.append(sb.lidar_feature_fov(xyz, poses[i], floor, device))  # (1,1,256,256)
        kept.append(poses[i])
    print(f"[data] 去程 {nfwd} 位姿, 有效FPV {len(kept)}; 每位姿 x {len(CONDITIONS)} 条件")
    return np.array(kept), cams, bevs


def gt_traj_tensor(device):
    """4 条件的 GT 轨迹模板 (ncond,8,3)。"""
    return torch.tensor(np.stack([make_traj(c) for c in CONDITIONS]), dtype=torch.float32, device=device)


def status_neutral(bs, device, vx=0.4):
    s = np.zeros((bs, 8), np.float32); s[:, 4] = vx          # 中性 status(不当条件用)
    return torch.from_numpy(s).to(device)


def predict(model, cam, bev, device, cond_id=None):
    """单位姿单条件 -> 预测轨迹 (8,3) numpy。"""
    feats = {"camera_feature": cam, "lidar_feature": bev, "status_feature": status_neutral(1, device)}
    if cond_id is not None:
        feats["cond_id"] = torch.tensor([cond_id], device=device)
    with torch.no_grad():
        out = model(feats)
    return out["trajectory"][0].cpu().numpy()


def controllability_panel(poses, cams, bevs, model, device, sample_idx, tag, hw=1.5):
    """每位姿一个干净 ego 示意图: 走廊墙 + 机器人 + 4 条件的【模型预测】轨迹。"""
    S, ppm = 460, 88.0
    panels = []
    for si in sample_idx:
        img = np.full((S, S, 3), 248, np.uint8)
        cx, cy = S // 2, int(S * 0.82)
        def e2px(ex, ey):
            return int(cx - ey * ppm), int(cy - ex * ppm)
        for side in (+1, -1):                                   # 规整走廊墙
            cv2.line(img, e2px(-0.3, side * hw), e2px(4.6, side * hw), (205, 205, 205), 10)
        tri = np.array([e2px(0.32, 0), e2px(-0.12, 0.16), e2px(-0.12, -0.16)], np.int32)
        cv2.fillPoly(img, [tri], (0, 0, 0))
        for ci, cond in enumerate(CONDITIONS):
            tr = predict(model, cams[si], bevs[si], device, cond_id=ci)
            col = COND_COLOR[cond]
            pts = np.array([(cx, cy)] + [e2px(x_, y_) for x_, y_ in tr[:, :2]], np.int32)
            cv2.polylines(img, [pts], False, col, 4, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[-1]), 5, col, -1)
            cv2.putText(img, cond, (pts[-1][0] + (6 if "right" in cond or cond == "cruise" else -108),
                        pts[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"pose#{si}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(img, (0, 0), (S - 1, S - 1), (60, 60, 60), 1)
        panels.append(img)
    grid = np.hstack(panels)
    band = np.full((34, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(band, f"[{tag}] model trajectory per condition (SAME obs, different condition)",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    return np.vstack([band, grid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16, help="每步喂多少帧(×4条件); 大=占满卡训得快")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smooth_gt", type=int, default=1, help="真实路径GT位姿滑动平均窗(消抖); 1=不平滑")
    ap.add_argument("--dump_trajs", action="store_true",
                    help="不训练: 载入 cond_finetuned.pt, 对所有位姿×4条件出预测轨迹, 存npz给FPV叠加")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dump_trajs:
        poses, cams, bevs = load_dataset(args.session, device)
        model = sb.load_model(device)
        model._trajectory_head.plan_anchor.data = torch.tensor(robot_anchors(), device=device)
        model._trajectory_head.load_state_dict(
            torch.load(os.path.join(RUNS, "cond_finetuned.pt"))["trajectory_head"])
        model.eval()
        nc = len(CONDITIONS)
        trajs = np.zeros((len(poses), nc, 8, 3), np.float32)
        for i in range(len(poses)):
            for ci in range(nc):
                trajs[i, ci] = predict(model, cams[i], bevs[i], device, cond_id=ci)
        outp = os.path.join(RUNS, f"model_trajs_{args.session}.npz")
        np.savez(outp, trajs=trajs, poses=poses, conds=np.array(CONDITIONS))
        print(f"[dump] {len(poses)}位姿×{nc}条件 模型预测轨迹 -> {outp}")
        return

    poses, cams, bevs = load_dataset(args.session, device)
    N = len(poses)
    ncond = len(CONDITIONS)

    print("[model] 加载 DiffusionDrive(+cond注入) 暖启...")
    model = sb.load_model(device)                           # V2 直连 + best-val-loss.ckpt(cond_embed为新增,零初始化)
    # 锚点仍用 kmeans×0.1 当扩散先验; ⭐GT 改用【真实重建路径】(模仿), 不再用编的锚点子集
    A_np = robot_anchors()
    model._trajectory_head.plan_anchor.data = torch.tensor(A_np, device=device)
    poses_gt = smooth_path_xy(poses, args.smooth_gt)             # 平滑位姿只用于造GT(消抖), 不动缓存FPV/BEV
    if args.smooth_gt > 1:
        print(f"  GT位姿滑动平均 win={args.smooth_gt} (消 VGGT 位姿锯齿)")
    gt_all = torch.tensor(np.stack([cond_gts_from_realpath(realpath_ego(poses_gt, i)) for i in range(N)]),
                          dtype=torch.float32, device=device)   # (N,9,T,3)
    print(f"  GT=真实路径(cruise跟走廊)+横移(inspect_L/R)+近停(pause); "
          f"cruise平均前向={float(gt_all[:,1,-1,0].mean()):.2f}m 平均横移={float(gt_all[:,1,-1,1].abs().mean()):.2f}m")

    sample_idx = list(np.linspace(0, N - 1, 3).astype(int))
    # 微调前对比图(应4条几乎重合=cond还没学)
    model.eval()
    before = controllability_panel(poses, cams, bevs, model, device, sample_idx, "BEFORE finetune")

    # 只训 trajectory head(含 cond_embed), 冻结 backbone 省显存
    for p in model.parameters():
        p.requires_grad = False
    for p in model._trajectory_head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    B = args.batch
    print(f"[train] {N}位姿 x {ncond}条件, batch={B}帧/步(={B*ncond}样本), {args.epochs} epochs ...")
    model.train()
    for ep in range(args.epochs):
        order = np.random.permutation(N)
        tot, nb = 0.0, 0
        for s in range(0, N, B):
            bi = order[s:s + B]                                # 一批帧
            cam = torch.cat([cams[i].repeat(ncond, 1, 1, 1) for i in bi])   # (ncond*b,3,256,1024)
            bev = torch.cat([bevs[i].repeat(ncond, 1, 1, 1) for i in bi])
            gt = torch.cat([gt_all[i] for i in bi])                          # (ncond*b,T,3)
            cond = torch.arange(ncond, device=device).repeat(len(bi))        # [0,1,2,3, 0,1,2,3, ...]
            feats = {"camera_feature": cam, "lidar_feature": bev,
                     "status_feature": status_neutral(ncond * len(bi), device),
                     "cond_id": cond}
            out = model(feats, {"trajectory": gt})
            loss = out["trajectory_loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:03d} loss={tot / nb:.4f}", flush=True)

    torch.save({"trajectory_head": model._trajectory_head.state_dict()},
               os.path.join(RUNS, "cond_finetuned.pt"))
    model.eval()
    after = controllability_panel(poses, cams, bevs, model, device, sample_idx, "AFTER finetune")

    out_png = os.path.join(RUNS, f"controllability_{args.session}.png")
    cv2.imwrite(out_png, np.vstack([before, after]))
    print(f"[done] 微调前/后可控性对比 -> {out_png}\n  权重 -> {os.path.join(RUNS, 'cond_finetuned.pt')}")


if __name__ == "__main__":
    main()
