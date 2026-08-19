"""让 DD 产出【所有轨迹, 含原地转】的重训(试验版)。
⭐核心洞察: 原地转是"图像无关"的——机器人转身是因为被命令朝某方向, 不是因为看到啥。
   所以给 status 加【目标朝向 target_heading(ego系 cos/sin)】信号, 并加入【原地转样本】
   (GT: x≈0,y≈0, yaw从0爬到±90°), 让 DD 学会"命令朝向变→输出转向轨迹"。
   yaw 是 head 独立回归通道((8,3)), (0,0,yaw) 表示上本就成立; 缺的只是训练样本+信号。

status(8): [0:2]=归一化目标点(平移去哪), [2:4]=目标朝向(cos,sin 该朝哪/转多少), [4]=vx。
用法(空卡):
  CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python goalnav/pipeline/train_goalnav_turn.py --epochs 240
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
import stage_b_eval as sb
from train_goalnav import robot_kmeans_anchors, smooth_traj, with_yaw, GOAL_AHEAD, SX, SY

GOALNAV = os.path.join(ROOT, "goalnav", "runs")
T = 8


def status_gh(goal_ego, head_ego, bs_vx=0.4, device="cuda"):
    """status(8): [0:2]=目标点(平移), [2:4]=目标朝向ego(cos,sin=该朝哪), [4]=vx。"""
    g = np.asarray(goal_ego, np.float32).reshape(-1, 2)
    h = np.asarray(head_ego, np.float32).reshape(-1, 2)
    st = np.zeros((len(g), 8), np.float32)
    st[:, 0] = np.clip(g[:, 0] / SX, -1, 1); st[:, 1] = np.clip(g[:, 1] / SY, -1, 1)
    st[:, 2] = np.clip(h[:, 0], -1, 1); st[:, 3] = np.clip(h[:, 1], -1, 1)
    st[:, 4] = bs_vx
    return torch.tensor(st, device=device)


def turn_gt(dyaw):
    """原地转GT: 8点 x=y=0, yaw从0线性爬到dyaw。"""
    ys = np.linspace(0, dyaw, T, dtype=np.float32)
    return np.stack([np.zeros(T), np.zeros(T), ys], 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_turn", type=int, default=260, help="合成原地转样本数(左右各半)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dsdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}")
    M = np.load(os.path.join(dsdir, "meta.npz"), allow_pickle=True)
    poses = M["poses"].astype(np.float64); gdir = M["goal_dirs"].astype(np.float32)
    base = M["base_ego"].astype(np.float32); fpvs = [str(x) for x in M["fpv"]]
    xyz = sb.read_ply_xyz(str(M["merged_ply"])); _, _, _, floor = sb.build_occ(xyz)
    N = len(poses)
    print(f"[data] {N} 平移样本 featurize(+镜像) + 合成{args.n_turn}原地转样本 ...", flush=True)

    cams, bevs, gts, heads = [], [], [], []
    # ---- 平移样本(cruise/strafe): 目标朝向=前向(1,0), GT=真实ego轨迹(平移) ----
    for i in range(N):
        img = cv2.imread(os.path.join(dsdir, fpvs[i]))
        bev_i = sb.lidar_feature_fov(xyz, poses[i], floor, device)
        for flip in (False, True):
            im2 = cv2.flip(img, 1) if flip else img
            cams.append(sb.camera_feature(im2, device))
            bevs.append(torch.flip(bev_i, dims=[-1]) if flip else bev_i)
            b = base[i].copy(); gd = gdir[i].copy()
            if flip: b[:, 1] = -b[:, 1]; gd[1] = -gd[1]
            gts.append(with_yaw(smooth_traj(b)))
            goal = gd * GOAL_AHEAD
            heads.append((goal, np.array([1.0, 0.0], np.float32)))   # 目标朝向=前向
        if i % 300 == 0: print(f"  featurize {i}/{N}", flush=True)
    n_trans = len(cams)

    # ---- 原地转样本: 复用真实观测(随机取帧的cam/bev), GT=yaw爬±90°, 目标朝向=侧向 ----
    rng = np.random.RandomState(0)
    for t in range(args.n_turn):
        src = int(rng.randint(0, n_trans))
        cams.append(cams[src]); bevs.append(bevs[src])
        left = (t % 2 == 0)
        dyaw = (np.pi / 2) * (1 if left else -1) * rng.uniform(0.6, 1.0)   # 转60~90°
        gts.append(turn_gt(dyaw))
        h = np.array([np.cos(dyaw), np.sin(dyaw)], np.float32)             # 目标朝向=转到的方向
        heads.append((np.zeros(2, np.float32), h))                        # 原地转: 目标点=0(不平移)
    N2 = len(cams)
    st_all = status_gh(np.stack([h[0] for h in heads]), np.stack([h[1] for h in heads]), device=device)
    gts = torch.tensor(np.stack(gts), dtype=torch.float32, device=device)
    print(f"[data] 共{N2}样本 (平移{n_trans} + 原地转{args.n_turn})", flush=True)

    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    for p in model.parameters(): p.requires_grad = False
    for p in model._trajectory_head.parameters(): p.requires_grad = True
    for p in model._status_encoding.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    B = args.batch
    print(f"[train] {N2}样本 batch={B} {args.epochs}ep (status加目标朝向 + 原地转样本) ...", flush=True)
    model.train()
    for ep in range(args.epochs):
        order = np.random.permutation(N2); tot = 0.0; nb = 0
        for s in range(0, N2, B):
            bi = order[s:s + B]
            cam = torch.cat([cams[i] for i in bi]); bev = torch.cat([bevs[i] for i in bi])
            st = st_all[bi]; gt = gts[bi]
            out = model({"camera_feature": cam, "lidar_feature": bev, "status_feature": st},
                        {"trajectory": gt})
            loss = out["trajectory_loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:03d} loss={tot / nb:.4f}", flush=True)
        # ⭐每40ep存中途ckpt(防重启前功尽弃)
        if ep >= 40 and ep % 40 == 0:
            torch.save({"trajectory_head": model._trajectory_head.state_dict(),
                        "status_encoding": model._status_encoding.state_dict()},
                       os.path.join(GOALNAV, "goalnav_turn.pt"))
            print(f"  [ckpt] ep{ep} -> goalnav_turn.pt", flush=True)
    torch.save({"trajectory_head": model._trajectory_head.state_dict(),
                "status_encoding": model._status_encoding.state_dict()},
               os.path.join(GOALNAV, "goalnav_turn.pt"))
    print(f"[done] -> {os.path.join(GOALNAV, 'goalnav_turn.pt')}", flush=True)


if __name__ == "__main__":
    main()
