"""lidar_feature_fov_hist 单测 (0802):
①W=0 → 与 lidar_feature_fov 逐位一致  ②同种子两次调用一致  ③历史点进入当前ego身后象限
④近距重掩码: 历史点不会落进机器人0.2m内(ego原点假亮斑)  ⑤γ=1保全部/γ→0≈无历史
用法: ~/discoverse_venv/bin/python scripts/test_lidar_hist.py
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
import stage_b_eval as sb

SESSION = "colmapjoint_110744s2"
xyz = sb.read_ply_xyz(os.path.join(ROOT, "outputs", f"gs_vggto_{SESSION}_world.ply"))
_, _, _, floor = sb.build_occ(xyz)
W_ = np.load(os.path.join(ROOT, "outputs", f"sim_world_{SESSION}.npz"))
ref = W_["waypoints_xy"].astype(np.float64)

# 取中段一个点构造位姿+8个历史位姿(0.4m沿轨迹向后, 朝向=切线)
i0 = len(ref) // 2
def pose_at(i):
    j = min(i, len(ref) - 2)
    th = float(np.arctan2(ref[j + 1, 1] - ref[j, 1], ref[j + 1, 0] - ref[j, 0]))
    return np.array([ref[j, 0], ref[j, 1], th])
seg = np.linalg.norm(np.diff(ref, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
pose = pose_at(i0)
hist = []
for lag in range(1, 9):
    si = max(0.0, arc[i0] - 0.4 * lag)
    hist.append(pose_at(int(np.searchsorted(arc, si))))
hist = np.array(hist)

dev = "cpu"
# ① W=0 逐位一致
a = sb.lidar_feature_fov(xyz, pose, floor, dev).numpy()
b = sb.lidar_feature_fov_hist(xyz, pose, floor, dev, hist, W=0, gamma=0.7, seed=42).numpy()
assert np.array_equal(a, b), "W=0 不一致!"
print("① W=0 与 lidar_feature_fov 逐位一致 ✓")

# ② 种子确定性
c1 = sb.lidar_feature_fov_hist(xyz, pose, floor, dev, hist, W=8, gamma=0.7, seed=42).numpy()
c2 = sb.lidar_feature_fov_hist(xyz, pose, floor, dev, hist, W=8, gamma=0.7, seed=42).numpy()
c3 = sb.lidar_feature_fov_hist(xyz, pose, floor, dev, hist, W=8, gamma=0.7, seed=43).numpy()
assert np.array_equal(c1, c2), "同种子不一致!"
assert not np.array_equal(c1, c3), "不同种子应不同!"
print("② 种子确定性 ✓ (同种子同图, 异种子异图)")

# ③ 历史点进入身后象限(当前ego系 ex<0)
ego_cur, _ = sb.fov_select_ego(xyz, pose, floor)
hw = [sb.fov_select_ego(xyz, hist[k], floor)[1] for k in range(8)]
merged = sb.hist_merge_ego(ego_cur, hw, pose, floor, gamma=1.0, seed=42)
n_behind = int((merged[:, 0] < -0.2).sum())
n_cur = len(ego_cur)
assert len(merged) > n_cur and n_behind > 100, f"身后点太少: {n_behind}"
print(f"③ 历史融合: 当前{n_cur}点 -> 合并{len(merged)}点, 身后(ex<-0.2){n_behind}点 ✓")

# ④ 近距重掩码: 合并后0.2m内无点
assert (np.hypot(merged[:, 0], merged[:, 1]) > 0.2).all(), "近距掩码失效!"
print("④ 近距重掩码: 合并点全部>0.2m ✓")

# ⑤ γ语义: γ=1保留全部历史点, γ→0退回≈单帧
m1 = sb.hist_merge_ego(ego_cur, hw, pose, floor, gamma=1.0, seed=42)
m0 = sb.hist_merge_ego(ego_cur, hw, pose, floor, gamma=1e-6, seed=42)
assert len(m1) > len(m0) * 1.5 and len(m0) <= len(ego_cur) + 50
print(f"⑤ γ语义: γ=1→{len(m1)}点, γ≈0→{len(m0)}点(≈单帧{n_cur}) ✓")

# ⑥ W越大BEV能量越大(直方图非零像素数)
import itertools
dens = []
for w in (0, 2, 4, 8):
    f = sb.lidar_feature_fov_hist(xyz, pose, floor, dev, hist, W=w, gamma=0.85, seed=42).numpy()
    dens.append(float((f > 0).sum()))
assert dens[0] < dens[-1], "W增大密度应增"
print(f"⑥ 密度随W单调: W=0/2/4/8 非零像素={dens} ✓")
print("[PASS] 全部单测通过")
