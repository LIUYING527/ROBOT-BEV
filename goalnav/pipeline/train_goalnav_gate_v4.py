"""train_goalnav_gate v4 (0803): mode条件 + 骨架搜索行为 — 论文§4.1.3落地(周报清单⑨)。

相对v3(0802观察策略参数化版)的升级(数据再平衡/条件dropout/门正则/时序历史全保留):
  ⭐status[7]=mode标志(0=goal-seek/1=search, v4及以前恒0=旧行为100%兼容):
    search样本(goal=(0,0))教"无目标时覆盖式探索+死胡同回退"——A*最短路GT教不出的行为。
  ⭐search样本heading也置零(无目标=无朝向约束); 条件dropout不再对mode=1复制(本来就零目标)。
  ⭐合成转向行继承源样本mode(转向行为mode无关, 条件分布要一致)。

用法: CUDA_VISIBLE_DEVICES=? MUJOCO_GL=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python goalnav/pipeline/train_goalnav_gate_v4.py --session colmapjoint_110744s2
产出: goalnav/runs/goalnav_gate_lab_v5.pt (不覆盖v4)
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
from train_goalnav_gate_v2 import gate_report

GOALNAV = os.path.join(ROOT, "goalnav", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--ds_suffix", default="_v5", help="数据集目录后缀(v5=mode条件+搜索行为)")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_turn", type=int, default=2000)
    ap.add_argument("--warm", default="goalnav_gate_lab_v4.pt")
    ap.add_argument("--out", default="goalnav_gate_lab_v5.pt")
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
    has_hist = "hist_poses" in M                                        # v4历史场(无则退v3单帧)
    hposes = M["hist_poses"].astype(np.float64) if has_hist else None
    hw_arr = M["hist_w"].astype(np.int32) if has_hist else np.zeros(len(poses), np.int32)
    hg_arr = M["hist_gamma"].astype(np.float32) if has_hist else np.ones(len(poses), np.float32)
    mode_arr = M["mode"].astype(np.int32) if "mode" in M else np.zeros(len(poses), np.int32)  # ⭐v5: 0=goal-seek 1=search
    xyz = sb.read_ply_xyz(str(M["merged_ply"])); _, _, _, floor = sb.build_occ(xyz)
    N = len(poses)
    print(f"[data] 历史场={'有' if has_hist else '无(退单帧)'} mode: goal-seek{(mode_arr==0).sum()}/search{(mode_arr==1).sum()}", flush=True)

    # ---- P0-1: 先按GT动作分布做样本选择(直线45%/转向全留/决策点×2) ----
    turn_deg = np.degrees(np.abs(np.arctan2(base[:, -1, 1], base[:, -1, 0])))
    cls = np.where(turn_deg < 10, 0, np.where(turn_deg < 20, 1, 2))     # 0直 1中 2转
    rng = np.random.RandomState(0)
    sel = []                                                            # (样本i, 是否目标置零变体)
    for i in range(N):
        if mode_arr[i] == 1:                                            # ⭐search全留+×3过采样:
            sel.extend([(i, False)] * 3)                                #   mode=1仅260条(13%)→位死(status[7]被无视,
            continue                                                    #   ep120探针mode0/1输出差0.003m); ×3→~1/3行数逼模型用mode位
        if cls[i] == 0:
            if rng.rand() < args.keep_straight:
                sel.append((i, False))
                if mode_arr[i] == 0 and rng.rand() < args.p_goal_drop:    # P0-2: 目标置零变体(⭐mode=1本来就零目标, 不复制)
                    sel.append((i, True))
        else:
            sel.append((i, False))
            if turn_deg[i] >= args.decision_deg:
                sel.append((i, False))                                  # 决策点过采样×2
    n_str = int((cls == 0).sum())
    print(f"[data] 原{N} (直{n_str} 中{(cls==1).sum()} 转{(cls==2).sum()}) -> 选{len(sel)}"
          f" (直线留{args.keep_straight:.0%}, 决策≥{args.decision_deg:.0f}°×2, 目标置零{args.p_goal_drop:.0%})", flush=True)

    # ---- ⭐ref位姿FOV选点缓存(历史位姿大量重复) ----
    _cache = {}
    def hist_world_pts(hp):
        key = (round(float(hp[0]), 3), round(float(hp[1]), 3), round(float(hp[2]), 3))
        if key not in _cache:
            _cache[key] = sb.fov_select_ego(xyz, hp, floor)[1]
        return _cache[key]

    # ---- featurize(选中样本, 含镜像; BEV=历史累积版) ----
    cams, bevs, gts, heads, vxl, wl, gl, ml = [], [], [], [], [], [], [], []
    for k, (i, gdrop) in enumerate(sel):
        img = cv2.imread(os.path.join(dsdir, fpvs[i]))
        w_i, g_i = int(hw_arr[i]), float(hg_arr[i])
        if has_hist and w_i > 0:
            ego_cur, _ = sb.fov_select_ego(xyz, poses[i], floor)
            hwl = [hist_world_pts(hposes[i][lag]) for lag in range(w_i)]
            ego = sb.hist_merge_ego(ego_cur, hwl, poses[i], floor, g_i, seed=i)
            bev_np = sb.sim_bev.official_lidar_feature(ego)
            bev_i = torch.from_numpy(bev_np).unsqueeze(0).to(device)
        else:
            bev_i = sb.lidar_feature_fov(xyz, poses[i], floor, device)
        for flip in (False, True):
            im2 = cv2.flip(img, 1) if flip else img
            cams.append(sb.camera_feature(im2, device))
            bevs.append(torch.flip(bev_i, dims=[-1]) if flip else bev_i)
            b = base[i].copy(); gd = gdir[i].copy()
            if flip: b[:, 1] = -b[:, 1]; gd[1] = -gd[1]
            gts.append(with_yaw(smooth_traj(b)))
            if mode_arr[i] == 1:                                     # ⭐search: goal+heading全零(无目标语义)
                heads.append((np.zeros(2, np.float32), np.zeros(2, np.float32)))
            else:
                heads.append((np.zeros(2, np.float32) if gdrop else gd * GOAL_AHEAD,
                              np.array([1.0, 0.0], np.float32)))
            vxl.append(float(vx_arr[i]) if vx_arr is not None else 0.4)
            wl.append(float(w_i)); gl.append(float(g_i)); ml.append(int(mode_arr[i]))
        if k % 200 == 0: print(f"  featurize {k}/{len(sel)} (缓存{len(_cache)})", flush=True)
    n_trans = len(cams)

    # ---- 合成原地转(同v2; ⭐转向行带源样本(W,γ)) ----
    rng2 = np.random.RandomState(0)
    for t in range(args.n_turn):
        src = int(rng2.randint(0, n_trans))
        cams.append(cams[src]); bevs.append(bevs[src])
        left = (t % 2 == 0)
        dyaw = (np.pi / 2) * (1 if left else -1) * rng2.uniform(0.6, 1.0)
        gts.append(turn_gt(dyaw))
        heads.append((np.zeros(2, np.float32), np.array([np.cos(dyaw), np.sin(dyaw)], np.float32)))
        vxl.append(0.0)                                                  # 原地转vx=0
        wl.append(wl[src]); gl.append(gl[src])                           # ⭐源样本的观察参数
        ml.append(ml[src])                                               # ⭐源样本的mode(转向行条件分布一致)
    N2 = len(cams)
    st_all = status_gh(np.stack([h[0] for h in heads]), np.stack([h[1] for h in heads]), device=device)
    st_all[:, 4] = torch.tensor(vxl, device=device)                      # 每样本速度(默认0.4兼容旧数据)
    st_all[:, 5] = torch.tensor(wl, device=device) / 8.0                 # ⭐观察窗口W(归一化)
    st_all[:, 6] = torch.tensor(gl, device=device)                       # ⭐衰减γ
    st_all[:, 7] = torch.tensor(ml, dtype=torch.float32, device=device)  # ⭐mode: 0=goal-seek 1=search(v4及以前恒0=兼容)
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
