"""train_goalnav_gate v2 (0731): Qwen-RobotNav训练经验落地 P0-1+P0-2。

P0-1 数据再平衡(论文§4.1.2: 前进帧只留45%, 转向/停止全保留, 关键决策点过采样):
  - 按GT末点偏航角分类: 直线(<10°)只留45%, 中间(10-20°)全留, 转向(≥20°)全留
  - 决策点(≥35°大转向)复制×2 = 过采样
  - 选择先于featurize计算(省BEV/相机特征计算)
P0-2 条件dropout(论文§4.1.5: 同轨迹多先验变体):
  - 直线样本30%概率追加"目标点置零"变体(同obs同GT, 仅status目标=0)
  - 直线时走廊跟随≈目标跟随, 监督不冲突; 让模型学会条件缺失时退回场景驱动

用法: CUDA_VISIBLE_DEVICES=? MUJOCO_GL=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python goalnav/pipeline/train_goalnav_gate_v2.py --session colmapjoint_110744s2
产出: goalnav/runs/goalnav_gate_lab_v2.pt (不覆盖旧版)
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch
import torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
import stage_b_eval as sb
from train_goalnav import robot_kmeans_anchors, smooth_traj, with_yaw, GOAL_AHEAD, SX, SY
from train_goalnav_turn import status_gh, turn_gt

GOALNAV = os.path.join(ROOT, "goalnav", "runs")


def gate_report(model, device):
    layer = model._trajectory_head.diff_decoder.layers[0]
    with torch.no_grad():
        st_fwd = status_gh(np.array([GOAL_AHEAD, 0], np.float32), np.array([1.0, 0.0], np.float32), device=device)
        st_turn = status_gh(np.zeros(2, np.float32), np.array([0.0, 1.0], np.float32), device=device)
        se_f = model._status_encoding(st_fwd)[:, None]; se_t = model._status_encoding(st_turn)[:, None]
        gf = torch.sigmoid(layer.scene_gate(se_f) + layer.scene_gate_bias).item()
        gt_ = torch.sigmoid(layer.scene_gate(se_t) + layer.scene_gate_bias).item()
    return gf, gt_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--ds_suffix", default="", help="数据集目录后缀(如_v3=连续条件+减速GT+vx场)")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_turn", type=int, default=2000)
    ap.add_argument("--warm", default="goalnav_gate_lab.pt")
    ap.add_argument("--out", default="goalnav_gate_lab_v2.pt")
    ap.add_argument("--gate_reg", type=float, default=0.3)
    ap.add_argument("--keep_straight", type=float, default=0.45, help="直线样本保留比例(论文45%)")
    ap.add_argument("--decision_deg", type=float, default=35.0, help="决策点阈值(度), ≥此复制×2")
    ap.add_argument("--p_goal_drop", type=float, default=0.3, help="直线样本目标置零变体概率")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(0); torch.manual_seed(0)

    dsdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}{args.ds_suffix}")
    M = np.load(os.path.join(dsdir, "meta.npz"), allow_pickle=True)
    poses = M["poses"].astype(np.float64); gdir = M["goal_dirs"].astype(np.float32)
    base = M["base_ego"].astype(np.float32); fpvs = [str(x) for x in M["fpv"]]
    vx_arr = M["vx"].astype(np.float32) if "vx" in M else None          # v3速度场(论文speed aug)
    xyz = sb.read_ply_xyz(str(M["merged_ply"])); _, _, _, floor = sb.build_occ(xyz)
    N = len(poses)

    # ---- P0-1: 先按GT动作分布做样本选择(直线45%/转向全留/决策点×2) ----
    turn_deg = np.degrees(np.abs(np.arctan2(base[:, -1, 1], base[:, -1, 0])))
    cls = np.where(turn_deg < 10, 0, np.where(turn_deg < 20, 1, 2))     # 0直 1中 2转
    rng = np.random.RandomState(0)
    sel = []                                                            # (样本i, 是否目标置零变体)
    for i in range(N):
        if cls[i] == 0:
            if rng.rand() < args.keep_straight:
                sel.append((i, False))
                if rng.rand() < args.p_goal_drop:                       # P0-2: 目标置零变体
                    sel.append((i, True))
        else:
            sel.append((i, False))
            if turn_deg[i] >= args.decision_deg:
                sel.append((i, False))                                  # 决策点过采样×2
    n_str = int((cls == 0).sum())
    print(f"[data] 原{N} (直{n_str} 中{(cls==1).sum()} 转{(cls==2).sum()}) -> 选{len(sel)}"
          f" (直线留{args.keep_straight:.0%}, 决策≥{args.decision_deg:.0f}°×2, 目标置零{args.p_goal_drop:.0%})", flush=True)

    # ---- featurize(选中样本, 含镜像) ----
    cams, bevs, gts, heads, vxl = [], [], [], [], []
    for k, (i, gdrop) in enumerate(sel):
        img = cv2.imread(os.path.join(dsdir, fpvs[i]))
        bev_i = sb.lidar_feature_fov(xyz, poses[i], floor, device)
        for flip in (False, True):
            im2 = cv2.flip(img, 1) if flip else img
            cams.append(sb.camera_feature(im2, device))
            bevs.append(torch.flip(bev_i, dims=[-1]) if flip else bev_i)
            b = base[i].copy(); gd = gdir[i].copy()
            if flip: b[:, 1] = -b[:, 1]; gd[1] = -gd[1]
            gts.append(with_yaw(smooth_traj(b)))
            heads.append((np.zeros(2, np.float32) if gdrop else gd * GOAL_AHEAD,
                          np.array([1.0, 0.0], np.float32)))
            vxl.append(float(vx_arr[i]) if vx_arr is not None else 0.4)
        if k % 200 == 0: print(f"  featurize {k}/{len(sel)}", flush=True)
    n_trans = len(cams)

    # ---- 合成原地转(同v1) ----
    rng2 = np.random.RandomState(0)
    for t in range(args.n_turn):
        src = int(rng2.randint(0, n_trans))
        cams.append(cams[src]); bevs.append(bevs[src])
        left = (t % 2 == 0)
        dyaw = (np.pi / 2) * (1 if left else -1) * rng2.uniform(0.6, 1.0)
        gts.append(turn_gt(dyaw))
        heads.append((np.zeros(2, np.float32), np.array([np.cos(dyaw), np.sin(dyaw)], np.float32)))
        vxl.append(0.0)                                                  # 原地转vx=0
    N2 = len(cams)
    st_all = status_gh(np.stack([h[0] for h in heads]), np.stack([h[1] for h in heads]), device=device)
    st_all[:, 4] = torch.tensor(vxl, device=device)                      # 每样本速度(默认0.4兼容旧数据)
    gts = torch.tensor(np.stack(gts), dtype=torch.float32, device=device)
    print(f"[data] 共{N2} (平移{n_trans} + 原地转{args.n_turn})", flush=True)
    gate_tgt = torch.zeros(N2, device=device); gate_tgt[:n_trans] = 1.0

    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    wp = os.path.join(GOALNAV, args.warm)
    if os.path.exists(wp):
        ck = torch.load(wp)
        model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
        model._status_encoding.load_state_dict(ck["status_encoding"])
        print(f"[warm] 载 {args.warm}", flush=True)

    for p in model.parameters(): p.requires_grad = False
    for p in model._trajectory_head.parameters(): p.requires_grad = True
    for p in model._status_encoding.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    B = args.batch
    gf, gt_ = gate_report(model, device)
    print(f"[gate] 初始: 前向{gf:.3f} 转向{gt_:.3f}", flush=True)
    print(f"[train] {N2}样本 batch={B} {args.epochs}ep ...", flush=True)
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
            if args.gate_reg > 0:
                se = model._status_encoding(st)[:, None]
                tgt = gate_tgt[bi][:, None, None]
                greg = 0.0
                for lyr in model._trajectory_head.diff_decoder.layers:
                    g = torch.sigmoid(lyr.scene_gate(se) + lyr.scene_gate_bias)
                    greg = greg + F.binary_cross_entropy(g, tgt.expand_as(g))
                loss = loss + args.gate_reg * greg
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            gf, gt_ = gate_report(model, device)
            print(f"  ep{ep:03d} loss={tot / nb:.4f}  门[前向{gf:.3f} 转向{gt_:.3f}]", flush=True)
        if ep >= 40 and ep % 40 == 0:
            torch.save({"trajectory_head": model._trajectory_head.state_dict(),
                        "status_encoding": model._status_encoding.state_dict()},
                       os.path.join(GOALNAV, args.out))
    torch.save({"trajectory_head": model._trajectory_head.state_dict(),
                "status_encoding": model._status_encoding.state_dict()},
               os.path.join(GOALNAV, args.out))
    gf, gt_ = gate_report(model, device)
    print(f"[done] -> {args.out}  终门[前向{gf:.3f} 转向{gt_:.3f}]", flush=True)


if __name__ == "__main__":
    main()
