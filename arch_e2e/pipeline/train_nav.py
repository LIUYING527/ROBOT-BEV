"""训练 DiffusionDrive 成【导航策略】(模仿学习 on 导航训练集)。
条件: BEV几何 + goal方向(ego) + cond(cruise/inspect±) -> 学"朝goal沿走廊居中走 + inspect时veer"。
暖启 cond_finetuned.pt(保留已学的9条件, goal_mlp新增零初始化), 训 trajectory_head(含goal_mlp)。
用法(卡7):
  CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python arch_e2e/pipeline/train_nav.py --epochs 300 --batch 24
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
import stage_b_eval as sb
from train_conditional import robot_anchors, status_neutral

RUNS = os.path.join(ROOT, "arch_e2e", "runs")
CONDS = ["cruise", "inspect_left", "inspect_right"]              # 导航训练聚焦这3个(其余保留自cond_finetuned)
CONABS = ["cruise", "inspect_left", "inspect_right", "strafe_left", "strafe_right",
          "turn_left", "turn_right", "pause", "stop"]           # 全表(cond_id索引)


def with_yaw(xy):
    dx = np.diff(np.concatenate([[0.0], xy[:, 0]])); dy = np.diff(np.concatenate([[0.0], xy[:, 1]]))
    return np.concatenate([xy, np.arctan2(dy, dx)[:, None]], 1).astype(np.float32)


def label_for(base_ego, cond, lat=0.55):
    """base_ego(T,2)=朝goal沿走廊的专家轨迹。cruise=原样; inspect_L/R=+渐进侧移(边走边贴设备)。"""
    T = len(base_ego); i1 = np.linspace(0, 1, T)
    off = lat * (i1 ** 2)
    if cond == "inspect_left":
        xy = base_ego + np.stack([np.zeros(T), off], 1)
    elif cond == "inspect_right":
        xy = base_ego + np.stack([np.zeros(T), -off], 1)
    else:
        xy = base_ego.copy()
    return with_yaw(xy)


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
    poses = M["poses"].astype(np.float64); goals = M["goal_dirs"].astype(np.float32)
    base = M["base_ego"].astype(np.float32); fpvs = [str(x) for x in M["fpv"]]
    xyz = sb.read_ply_xyz(str(M["merged_ply"])); _, _, _, floor = sb.build_occ(xyz)
    N = len(poses)
    print(f"[data] {N} 样本; 预算 BEV + 缓存 cam ...", flush=True)

    cams, bevs = [], []                                         # 预计算(N小, 缓存省训练时间)
    for i in range(N):
        fpv = cv2.imread(os.path.join(dsdir, fpvs[i]))
        cams.append(sb.camera_feature(fpv, device))
        bevs.append(sb.lidar_feature_fov(xyz, poses[i], floor, device))
        if i % 200 == 0: print(f"  featurize {i}/{N}", flush=True)
    goals_t = torch.tensor(goals, device=device)
    # 标签: 每样本 x 3 cond
    gts = torch.tensor(np.stack([[label_for(base[i], c) for c in CONDS] for i in range(N)]),
                       dtype=torch.float32, device=device)      # (N,3,T,3)

    print("[model] 加载 DiffusionDrive + 暖启 cond_finetuned.pt ...", flush=True)
    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_anchors(), device=device)
    sd = torch.load(os.path.join(RUNS, "cond_finetuned.pt"))["trajectory_head"]
    model._trajectory_head.load_state_dict(sd, strict=False)    # goal_mlp新增=保持零初始化
    for p in model.parameters(): p.requires_grad = False
    for p in model._trajectory_head.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    ncond = len(CONDS); B = args.batch
    cond_ids = torch.tensor([CONABS.index(c) for c in CONDS], device=device)
    print(f"[train] {N}样本x{ncond}cond, batch={B}, {args.epochs}ep ...", flush=True)
    model.train()
    for ep in range(args.epochs):
        order = np.random.permutation(N); tot = 0.0; nb = 0
        for s in range(0, N, B):
            bi = order[s:s + B]; nb_ = len(bi)
            cam = torch.cat([cams[i].repeat(ncond, 1, 1, 1) for i in bi])
            bev = torch.cat([bevs[i].repeat(ncond, 1, 1, 1) for i in bi])
            goal = torch.cat([goals_t[i].view(1, 2).repeat(ncond, 1) for i in bi])
            gt = torch.cat([gts[i] for i in bi])                # (ncond*b,T,3)
            cid = cond_ids.repeat(nb_)
            feats = {"camera_feature": cam, "lidar_feature": bev,
                     "status_feature": status_neutral(ncond * nb_, device),
                     "cond_id": cid, "goal_dir": goal}
            out = model(feats, {"trajectory": gt})
            loss = out["trajectory_loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:03d} loss={tot / nb:.4f}", flush=True)
    torch.save({"trajectory_head": model._trajectory_head.state_dict()},
               os.path.join(RUNS, "nav_finetuned.pt"))
    print(f"[done] -> {os.path.join(RUNS, 'nav_finetuned.pt')}", flush=True)


if __name__ == "__main__":
    main()
