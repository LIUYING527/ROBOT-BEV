"""观察策略参数化可控性探针 (0802): 同一观测, 只换观察参数(W窗口/γ衰减) → 轨迹分布不同。
= Qwen-RobotNav核心claim"多任务导航差异本质是怎么看历史"的我们版证据图。

固定: 数据集中段一个样本(位姿/FPV/目标点), torch.manual_seed(0)固定扩散噪声;
扫: W∈{0,2,4,8} × γ∈{0.5,0.85} → 8条预测轨迹叠在该样本W=8累积BEV上。
用法: CUDA_VISIBLE_DEVICES=? MUJOCO_GL=egl PYTHONPATH=third_party/discoverse \
    CKPT=goalnav_gate_lab_v4.pt ~/discoverse_venv/bin/python scripts/probe_obs_params.py [--idx N]
产出: outputs/obs_param_probe_<ckpt>.png
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--idx", type=int, default=-1, help="样本索引(默认自动选W=8且历史足的中段样本)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dsdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}_v4")
    M = np.load(os.path.join(dsdir, "meta.npz"), allow_pickle=True)
    poses = M["poses"].astype(np.float64); gdir = M["goal_dirs"].astype(np.float32)
    fpvs = [str(x) for x in M["fpv"]]
    hposes = M["hist_poses"].astype(np.float64)
    xyz = sb.read_ply_xyz(str(M["merged_ply"])); _, _, _, floor = sb.build_occ(xyz)

    idx = args.idx
    if idx < 0:
        idx = int(len(poses) * 2 // 3)                        # 中后段样本历史一般充足
    print(f"[probe] 样本#{idx} pose=({poses[idx,0]:.2f},{poses[idx,1]:.2f},{np.degrees(poses[idx,2]):.0f}°)", flush=True)

    ck_name = os.environ.get("CKPT", "goalnav_gate_lab_v4.pt")
    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    ck = torch.load(os.path.join(RUNS, ck_name))
    model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
    model._status_encoding.load_state_dict(ck["status_encoding"]); model.eval()
    print(f"[probe] ckpt={ck_name}", flush=True)

    img = cv2.imread(os.path.join(dsdir, fpvs[idx]))
    cam = sb.camera_feature(img, device)
    gd = gdir[idx] * GOAL_AHEAD
    st = status_gh(gd, np.array([1.0, 0.0], np.float32), device=device)

    combos = [(w, g) for w in (0, 2, 4, 8) for g in (0.5, 0.85)]
    res = []
    for ci, (w, g) in enumerate(combos):
        st_w = st.clone(); st_w[0, 5] = w / 8.0; st_w[0, 6] = g
        if w > 0:
            bev = sb.lidar_feature_fov_hist(xyz, poses[idx], floor, device, hposes[idx], W=w, gamma=g, seed=idx)
        else:
            bev = sb.lidar_feature_fov(xyz, poses[idx], floor, device)
        torch.manual_seed(0)                                  # 固定扩散噪声=只看参数影响
        with torch.no_grad():
            traj = model({"camera_feature": cam, "lidar_feature": bev, "status_feature": st_w})["trajectory"][0].cpu().numpy()
        res.append(((w, g), traj, bev[0, 0].cpu().numpy()))
        end = traj[-1]
        print(f"  W={w} γ={g}: 末点ego=({end[0]:+.2f},{end[1]:+.2f}) 进度={np.hypot(end[0],end[1]):.2f}m", flush=True)
    # ⭐±6m视窗(48px@0.25m/px); 每条轨迹单独小面板(横向分叉小, 全叠一起互相盖=看不清)
    # 布局: 行=γ(0.5上/0.85下), 列=W(0/2/4/8) → 同列比γ同行比W
    ZP = 48; PS = 256; SC = PS / (2 * ZP)
    def to_px(fx, fy):                                        # ego(x前y左) -> 面板坐标(前=上)
        return (int((ZP - fy / 32.0 * 128) * SC), int((ZP - fx / 32.0 * 128) * SC))
    td = dict(((w, g), (t, bv)) for (w, g), t, bv in res)
    rows = []
    for g in (0.5, 0.85):
        panels = []
        for w in (0, 2, 4, 8):
            traj_w, bev_w = td[(w, g)]
            bg = cv2.cvtColor((bev_w * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            p = cv2.resize(bg[128 - ZP:128 + ZP, 128 - ZP:128 + ZP], (PS, PS), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(p, to_px(0, 0), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
            cv2.drawMarker(p, to_px(gd[0], gd[1]), (0, 255, 255), cv2.MARKER_STAR, 18, 2)
            pts = np.array([to_px(fx, fy) for fx, fy, _ in traj_w], np.int32)
            cv2.polylines(p, [pts], False, (0, 0, 255), 2)
            cv2.circle(p, tuple(pts[-1]), 4, (0, 0, 255), -1)
            cv2.putText(p, f"W={w} gamma={g}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            panels.append(p)
        rows.append(np.hstack(panels))
    grid = np.vstack(rows)
    cv2.putText(grid, "star=goal  red=DD traj  (same obs, only obs-params differ)", (10, grid.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    tag = os.path.splitext(ck_name)[0].replace("goalnav_gate_", "")
    out = os.path.join(ROOT, "outputs", f"obs_param_probe_{tag}.png")
    cv2.imwrite(out, grid)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
