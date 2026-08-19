"""mode条件可控性探针 (0803, 论文⑨配套证据): 同一观测, 只翻status[7](0=goal-seek/1=search) → 行为分叉。

三组对照(=论文"mode-conditioned behavior"的我们版证据):
  A) goal-seek样本: v4 ckpt vs v5 ckpt 同输入(mode=0) → 回归检验(训练没训坏旧行为)
  B) search样本: v5预测 vs GT(骨架搜索路径) → 搜索行为学会没
  C) search样本同观测: mode=0(零目标=走廊跟随, dropout行为) vs mode=1 → 只翻标志位行为分叉
用法: CUDA_VISIBLE_DEVICES=? MUJOCO_GL=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python scripts/probe_mode.py [--idx_search N] [--idx_goal N]
产出: outputs/mode_probe_lab_v5.png
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
import stage_b_eval as sb
from train_goalnav import robot_kmeans_anchors, GOAL_AHEAD
from train_goalnav_turn import status_gh

RUNS = os.path.join(ROOT, "goalnav", "runs")


def load_ckpt(name, device):
    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    ck = torch.load(os.path.join(RUNS, name))
    model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
    model._status_encoding.load_state_dict(ck["status_encoding"]); model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--idx_search", type=int, default=-1, help="search样本索引(默认取mode=1中段)")
    ap.add_argument("--idx_goal", type=int, default=300, help="goal-seek样本索引")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dsdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}_v5")
    M = np.load(os.path.join(dsdir, "meta.npz"), allow_pickle=True)
    poses = M["poses"].astype(np.float64); gdir = M["goal_dirs"].astype(np.float32)
    base = M["base_ego"].astype(np.float32); fpvs = [str(x) for x in M["fpv"]]
    hposes = M["hist_poses"].astype(np.float64)
    hw = M["hist_w"].astype(np.int32); hg = M["hist_gamma"].astype(np.float32)
    mode = M["mode"].astype(np.int32)
    xyz = sb.read_ply_xyz(str(M["merged_ply"])); _, _, _, floor = sb.build_occ(xyz)

    idx_s = args.idx_search
    if idx_s < 0:                                            # 默认选搜索GT转向最大的样本(走廊直线段各mode轨迹趋同, 测不出分叉)
        si = np.where(mode == 1)[0]
        tdeg = np.degrees(np.abs(np.arctan2(base[si, -1, 1], base[si, -1, 0])))
        idx_s = int(si[int(np.argmax(tdeg))])
        print(f"[probe] 自动选search样本#{idx_s} (GT末端转向{tdeg.max():.0f}°)", flush=True)
    idx_g = min(args.idx_goal, int((mode == 0).sum()) - 1)
    m4 = load_ckpt("goalnav_gate_lab_v4.pt", device)
    m5 = load_ckpt("goalnav_gate_lab_v5.pt", device)

    def bev_of(idx, w, g):
        if w > 0:
            return sb.lidar_feature_fov_hist(xyz, poses[idx], floor, device, hposes[idx], W=w, gamma=g, seed=idx)
        return sb.lidar_feature_fov(xyz, poses[idx], floor, device)

    def run(model, idx, goal, head, vx, w, g, md):
        img = cv2.imread(os.path.join(dsdir, fpvs[idx]))
        cam = sb.camera_feature(img, device)
        st = status_gh(np.asarray(goal, np.float32), np.asarray(head, np.float32), bs_vx=vx, device=device)
        st[0, 5] = w / 8.0; st[0, 6] = g; st[0, 7] = float(md)
        torch.manual_seed(0)
        with torch.no_grad():
            t = model({"camera_feature": cam, "lidar_feature": bev_of(idx, w, g),
                       "status_feature": st})["trajectory"][0].cpu().numpy()
        return t, bev_of(idx, w, g)[0, 0].cpu().numpy()

    # A) 回归: v4 vs v5, goal-seek样本同输入
    gd = gdir[idx_g] * GOAL_AHEAD
    tA4, bA = run(m4, idx_g, gd, (1.0, 0.0), 0.5, int(hw[idx_g]), float(hg[idx_g]), 0)
    tA5, _ = run(m5, idx_g, gd, (1.0, 0.0), 0.5, int(hw[idx_g]), float(hg[idx_g]), 0)
    dev = float(np.linalg.norm(tA4[:, :2] - tA5[:, :2], axis=1).mean())
    print(f"[A回归] idx{idx_g}: v4 vs v5 平均逐点偏差={dev:.3f}m", flush=True)
    # B) 搜索学会没: v5预测 vs GT
    tB, bB = run(m5, idx_s, np.zeros(2), np.zeros(2), 0.45, int(hw[idx_s]), float(hg[idx_s]), 1)
    gtB = base[idx_s]
    dgt = float(np.linalg.norm(tB[:, :2] - gtB, axis=1).mean())
    print(f"[B搜索] idx{idx_s}(W{hw[idx_s]}γ{hg[idx_s]:.2f}): 预测vs搜索GT 平均偏差={dgt:.3f}m", flush=True)
    # C) 只翻mode: 同观测零目标, mode0(走廊跟随) vs mode1(搜索)
    tC0, _ = run(m5, idx_s, np.zeros(2), np.zeros(2), 0.45, int(hw[idx_s]), float(hg[idx_s]), 0)
    tC1, _ = run(m5, idx_s, np.zeros(2), np.zeros(2), 0.45, int(hw[idx_s]), float(hg[idx_s]), 1)
    fork = float(np.linalg.norm(tC0[:, :2] - tC1[:, :2], axis=1).mean())
    e0, e1 = tC0[-1, :2], tC1[-1, :2]
    print(f"[C分叉] idx{idx_s}: mode0末点=({e0[0]:+.2f},{e0[1]:+.2f}) mode1末点=({e1[0]:+.2f},{e1[1]:+.2f}) 平均分叉={fork:.3f}m", flush=True)
    # D) ⭐产品语义对照: 同观测, "有目标mode0" vs "零目标mode1" —— 慢脑切换搜索/寻求的真实开关
    gd_s = gdir[idx_g] * GOAL_AHEAD
    tD0, bD = run(m5, idx_g, gd_s, (1.0, 0.0), 0.5, int(hw[idx_g]), float(hg[idx_g]), 0)
    tD1, _ = run(m5, idx_g, np.zeros(2), np.zeros(2), 0.45, 8, 0.9, 1)
    forkD = float(np.linalg.norm(tD0[:, :2] - tD1[:, :2], axis=1).mean())
    print(f"[D开关] idx{idx_g}: 有目标末点=({tD0[-1,0]:+.2f},{tD0[-1,1]:+.2f}) 零目标搜索末点=({tD1[-1,0]:+.2f},{tD1[-1,1]:+.2f}) 分叉={forkD:.3f}m", flush=True)
    # E) ⭐三点门探针(实战条件, 非gate_report的默认status): 巡航门应~1 / 任务转向门(vx=0,W8)应~0 / 搜索门应~1(搜索要吃场景)
    layer = m5._trajectory_head.diff_decoder.layers[0]
    def gate_of(goal, head, vx, w, g, md):
        st = status_gh(np.asarray(goal, np.float32), np.asarray(head, np.float32), bs_vx=vx, device=device)
        st[0, 5] = w / 8.0; st[0, 6] = g; st[0, 7] = float(md)
        se = m5._status_encoding(st)[:, None]
        return float(torch.sigmoid(layer.scene_gate(se) + layer.scene_gate_bias).item())
    g_cruise = gate_of((4.0, 0.0), (1.0, 0.0), 0.5, 2, 0.7, 0)
    g_turn = gate_of((0.0, 0.0), (0.0, 1.0), 0.0, 8, 0.85, 0)
    g_search = gate_of((0.0, 0.0), (0.0, 0.0), 0.45, 8, 0.9, 1)
    print(f"[E门] 巡航={g_cruise:.3f}(应~1) 任务转向={g_turn:.3f}(应~0) 搜索={g_search:.3f}(应~1)", flush=True)

    # ---- 画板: 3组各一个面板(BEV底+轨迹) ----
    ZP = 48; PS = 300; SC = PS / (2 * ZP)
    def to_px(fx, fy):
        return (int((ZP - fy / 32.0 * 128) * SC), int((ZP - fx / 32.0 * 128) * SC))
    def panel(bev, trajs, title):
        bg = cv2.cvtColor((bev * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        p = cv2.resize(bg[128 - ZP:128 + ZP, 128 - ZP:128 + ZP], (PS, PS), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(p, to_px(0, 0), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        cols = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
        for k, (t, lb) in enumerate(trajs):
            pts = np.array([to_px(fx, fy) for fx, fy, _ in t], np.int32)
            cv2.polylines(p, [pts], False, cols[k % 4], 2)
            cv2.circle(p, tuple(pts[-1]), 4, cols[k % 4], -1)
            cv2.putText(p, lb, (8, 44 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cols[k % 4], 1)
        cv2.putText(p, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return p
    pA = panel(bA, [(tA4, "v4 mode0"), (tA5, "v5 mode0")], f"A regress dev={dev:.2f}m")
    pB = panel(bB, [(tB, "v5 pred(zero-goal)"), (np.hstack([gtB, np.zeros((len(gtB), 1), np.float32)]), "search GT")], f"B search vsGT={dgt:.2f}m")
    pC = panel(bB, [(tC0, "mode0"), (tC1, "mode1")], f"C modeflip fork={fork:.2f}m")
    pD = panel(bD, [(tD0, "goal-seek"), (tD1, "zero-goal search")], f"D switch fork={forkD:.2f}m")
    grid = np.hstack([pA, pB, pC, pD])
    cv2.putText(grid, "red/green/magenta=trajs  same obs, only status[7] differs (paper mode-conditioned behavior)",
                (10, grid.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    out = os.path.join(ROOT, "outputs", "mode_probe_lab_v5.png")
    cv2.imwrite(out, grid)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
