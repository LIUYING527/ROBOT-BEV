"""goal-conditioned DiffusionDrive(干净版, 按官方机制).
⭐"往哪走"走官方原生通路: status_feature 的 command 槽(前4维)改放【ego系目标点】(gx,gy)。
   不用运动模式cond, 不用goal_mlp bolt-on。 status = [gx/Sx, gy/Sy, 0, 0, vx, vy, 0, 0]。
   机器人尺度锚点 + 在我们走廊数据(含扰动recovery)上训 head+_status_encoding 学"朝目标点、沿走廊避墙规划"。
数据: outputs/nav_ds_<session>/ (poses, goal_dirs, base_ego专家轨迹, FPV)。
用法(卡7):
  CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python goalnav/pipeline/train_goalnav.py --epochs 300 --batch 24
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
import stage_b_eval as sb


def robot_kmeans_anchors():
    """⭐24锚点=原20车锚点缩到机器人尺度(前向扇) + 2原地转 + 2纯横移。形状干净、覆盖全。"""
    p = os.path.join(ROOT, "outputs", "anchors_car20_plus4_robot.npy")
    return np.load(p).astype(np.float32)


def smooth_traj(xy, k=1):
    """轻平滑专家轨迹(消VGGT位姿抖导致的锯齿)。"""
    out = xy.copy()
    for i in range(len(xy)):
        a, b = max(0, i - k), min(len(xy), i + k + 1)
        out[i] = xy[a:b].mean(0)
    return out

GOALNAV = os.path.join(ROOT, "goalnav", "runs"); os.makedirs(GOALNAV, exist_ok=True)
SX, SY = 5.0, 2.0          # 目标点归一化尺度(前向~5m, 横向~2m)
GOAL_AHEAD = 4.0           # ego目标点取前方~4m


def with_yaw(xy):
    dx = np.diff(np.concatenate([[0.0], xy[:, 0]])); dy = np.diff(np.concatenate([[0.0], xy[:, 1]]))
    return np.concatenate([xy, np.arctan2(dy, dx)[:, None]], 1).astype(np.float32)


def status_with_goal(goal_ego, bs_vx=0.4, device="cuda"):
    """status_feature(8): command槽[0:2]=归一化目标点, [2:4]=0, [4]=vx, 其余0。"""
    g = goal_ego.reshape(-1, 2)
    st = np.zeros((len(g), 8), np.float32)
    st[:, 0] = np.clip(g[:, 0] / SX, -1, 1); st[:, 1] = np.clip(g[:, 1] / SY, -1, 1)
    st[:, 4] = bs_vx
    return torch.tensor(st, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dsdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}")
    M = np.load(os.path.join(dsdir, "meta.npz"), allow_pickle=True)
    poses = M["poses"].astype(np.float64); gdir = M["goal_dirs"].astype(np.float32)
    base = M["base_ego"].astype(np.float32); fpvs = [str(x) for x in M["fpv"]]
    xyz = sb.read_ply_xyz(str(M["merged_ply"])); _, _, _, floor = sb.build_occ(xyz)
    N = len(poses)
    print(f"[data] {N} 样本; featurize + 水平镜像增广(治系统左偏/平衡左右) ...", flush=True)
    cams, bevs, gt_list, goal_list = [], [], [], []
    for i in range(N):
        img = cv2.imread(os.path.join(dsdir, fpvs[i]))
        bev_i = sb.lidar_feature_fov(xyz, poses[i], floor, device)
        for flip in (False, True):                              # ⭐每样本 原始 + 左右镜像
            im2 = cv2.flip(img, 1) if flip else img
            cams.append(sb.camera_feature(im2, device))
            bevs.append(torch.flip(bev_i, dims=[-1]) if flip else bev_i)
            b = base[i].copy(); gd = gdir[i].copy()
            if flip: b[:, 1] = -b[:, 1]; gd[1] = -gd[1]         # 镜像: 横向取反
            gt_list.append(with_yaw(smooth_traj(b)))
            goal_list.append(gd * GOAL_AHEAD)
        if i % 200 == 0: print(f"  {i}/{N}", flush=True)
    st_all = status_with_goal(np.stack(goal_list), device=device)   # (2N,8)
    gts = torch.tensor(np.stack(gt_list), dtype=torch.float32, device=device)  # (2N,T,3)
    N = len(cams)                                               # 增广后样本数=2N
    print(f"[data] 增广后 {N} 样本", flush=True)

    print("[model] pristine DiffusionDrive + 机器人聚类锚点 (不用cond/goal_mlp) ...", flush=True)
    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    for p in model.parameters(): p.requires_grad = False
    for p in model._trajectory_head.parameters(): p.requires_grad = True
    for p in model._status_encoding.parameters(): p.requires_grad = True   # goal进status,需重训这层
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    B = args.batch
    print(f"[train] {N}样本 batch={B} {args.epochs}ep (目标点走status/command槽, 无cond) ...", flush=True)
    model.train()
    for ep in range(args.epochs):
        order = np.random.permutation(N); tot = 0.0; nb = 0
        for s in range(0, N, B):
            bi = order[s:s + B]
            cam = torch.cat([cams[i] for i in bi]); bev = torch.cat([bevs[i] for i in bi])
            st = st_all[bi]; gt = gts[bi]
            feats = {"camera_feature": cam, "lidar_feature": bev, "status_feature": st}  # 无cond_id/goal_dir
            out = model(feats, {"trajectory": gt})
            loss = out["trajectory_loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:03d} loss={tot / nb:.4f}", flush=True)
    torch.save({"trajectory_head": model._trajectory_head.state_dict(),
                "status_encoding": model._status_encoding.state_dict()},
               os.path.join(GOALNAV, "goalnav_ft.pt"))
    print(f"[done] -> {os.path.join(GOALNAV, 'goalnav_ft.pt')}", flush=True)


if __name__ == "__main__":
    main()
