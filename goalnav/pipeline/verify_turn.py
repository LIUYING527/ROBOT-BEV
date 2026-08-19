"""验证 goalnav_turn.pt: 同一真实观测, 改【目标朝向】看轨迹是否随命令变。
   期望: 目标朝向=前向(0°)→直行(x前进,yaw≈0); 目标朝向=±90°→原地转(x≈y≈0, yaw爬到±90°)。
用法(空卡):
  CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python goalnav/pipeline/verify_turn.py
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
import stage_b_eval as sb
from train_goalnav import robot_kmeans_anchors, GOAL_AHEAD
from train_goalnav_turn import status_gh

RUNS = os.path.join(ROOT, "goalnav", "runs")
SESSION = "colmapjoint_all"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 载模型 + ckpt(可 env CKPT 覆盖, 默认 turn; 验门模型传 goalnav_gate.pt) ---
CKPT = os.environ.get("CKPT", "goalnav_turn.pt")
model = sb.load_model(device)
model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
ck = torch.load(os.path.join(RUNS, CKPT))
model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
model._status_encoding.load_state_dict(ck["status_encoding"]); model.eval()
print(f"[verify] 载 {CKPT}", flush=True)

# --- 取几帧真实观测 ---
dsdir = os.path.join(ROOT, "outputs", f"nav_ds_{SESSION}")
M = np.load(os.path.join(dsdir, "meta.npz"), allow_pickle=True)
poses = M["poses"].astype(np.float64); fpvs = [str(x) for x in M["fpv"]]
xyz = sb.read_ply_xyz(str(M["merged_ply"])); _, _, _, floor = sb.build_occ(xyz)

frame_idxs = [len(poses) // 4, len(poses) // 2, 3 * len(poses) // 4]
# 命令的目标朝向(度): 前向0, +45, +90(左转), -45, -90(右转)
headings_deg = [0, 45, 90, -45, -90]

fig, axes = plt.subplots(1, len(frame_idxs), figsize=(5 * len(frame_idxs), 5))
if len(frame_idxs) == 1: axes = [axes]

for ax, fi in zip(axes, frame_idxs):
    img = cv2.imread(os.path.join(dsdir, fpvs[fi]))
    cam = sb.camera_feature(img, device)
    bev = sb.lidar_feature_fov(xyz, poses[fi], floor, device)
    print(f"\n=== frame {fi} ({fpvs[fi]}) ===")
    for hd in headings_deg:
        rad = np.deg2rad(hd)
        h = np.array([np.cos(rad), np.sin(rad)], np.float32)
        # 前向命令给前方目标点; 转向命令目标点=0(不平移)
        goal = np.array([GOAL_AHEAD, 0.0], np.float32) if hd == 0 else np.zeros(2, np.float32)
        st = status_gh(goal, h, device=device)
        with torch.no_grad():
            traj = model({"camera_feature": cam, "lidar_feature": bev,
                          "status_feature": st})["trajectory"][0].cpu().numpy()  # (8,3)=x,y,yaw
        end_yaw = np.rad2deg(traj[-1, 2]); dx = traj[-1, 0]; dy = traj[-1, 1]
        reach = np.hypot(dx, dy)
        print(f"  命令朝向{hd:+4d}°  -> 末点(x={dx:+.2f},y={dy:+.2f}) 位移{reach:.2f}m  末yaw={end_yaw:+6.1f}°")
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=3, label=f"cmd {hd:+d}° (yaw→{end_yaw:+.0f}°)")
    ax.plot(0, 0, "ks", ms=8); ax.set_title(f"frame {fi}")
    ax.set_xlabel("x fwd (m)"); ax.set_ylabel("y left (m)")
    ax.axis("equal"); ax.grid(alpha=.3); ax.legend(fontsize=7)

out = os.environ.get("OUT", os.path.join(ROOT, "outputs", "verify_turn_demo.png"))
fig.suptitle(f"{CKPT}: 同一观测, 改目标朝向 -> 轨迹/末端朝向随命令变", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=120)
print(f"\n[verify] 图 -> {out}")
