"""
演示 NAVSIM/DiffusionDrive 真实的 image->BEV 到底怎么回事。
注意：模型吃的那张 BEV 不是从图像来的，是 LiDAR 点云俯视直方图(splat_points)。
本脚本用合成点云 + NAVSIM 原样参数与逻辑复现，仅为看效果(SYNTHETIC)。
参数取自 transfuser_config.py。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- NAVSIM 原样参数 ---
lidar_min_x, lidar_max_x = -32.0, 32.0
lidar_min_y, lidar_max_y = -32.0, 32.0
pixels_per_meter = 4.0
hist_max_per_pixel = 5
max_height_lidar = 100.0
lidar_split_height = 0.2
use_ground_plane = False

rng = np.random.default_rng(0)

# --- 合成一个"城市路口"点云: 车道边墙/两侧建筑/几辆车/路面 ---
pts = []
# 地面稀疏点(z~0)
g = rng.uniform([-32,-32],[32,32], size=(40000,2)); gz=rng.normal(0.02,0.03,(40000,1)); pts.append(np.hstack([g,gz]))
# 左右两道"墙/建筑立面" (y=±10 附近, 高)
for yc in (-10, 10):
    w = np.column_stack([rng.uniform(-32,32,8000), rng.normal(yc,0.4,8000), rng.uniform(0.3,6,8000)]); pts.append(w)
# 前方横向建筑
w = np.column_stack([rng.normal(26,0.4,6000), rng.uniform(-10,10,6000), rng.uniform(0.3,8,6000)]); pts.append(w)
# 几辆车(盒子, 高~1.5)
for cx,cy in [(8,-3),(-12,4),(18,2),(-4,-6)]:
    b = np.column_stack([rng.normal(cx,1.5,1500), rng.normal(cy,0.8,1500), rng.uniform(0.3,1.6,1500)]); pts.append(b)
lidar_pc = np.vstack(pts)

# --- NAVSIM splat_points 原样逻辑 ---
def splat_points(point_cloud):
    xbins = np.linspace(lidar_min_x, lidar_max_x, int((lidar_max_x-lidar_min_x)*pixels_per_meter)+1)
    ybins = np.linspace(lidar_min_y, lidar_max_y, int((lidar_max_y-lidar_min_y)*pixels_per_meter)+1)
    hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    return hist / hist_max_per_pixel

pc = lidar_pc[lidar_pc[..., 2] < max_height_lidar]
below = pc[pc[..., 2] <= lidar_split_height]
above = pc[pc[..., 2] >  lidar_split_height]
above_features = splat_points(above)
if use_ground_plane:
    feat = np.stack([splat_points(below), above_features], axis=-1)
else:
    feat = np.stack([above_features], axis=-1)
feat = np.transpose(feat, (2,0,1)).astype(np.float32)
print("BEV tensor shape (C,H,W):", feat.shape, " 数值范围:", feat.min(), feat.max())

# --- 画图 ---
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
ax[0].scatter(lidar_pc[::15,0], lidar_pc[::15,1], s=0.3, c=lidar_pc[::15,2], cmap="viridis")
ax[0].set_title("(1) LiDAR point cloud  (top-down x-y, color=height z)")
ax[0].set_xlim(-32,32); ax[0].set_ylim(-32,32); ax[0].set_aspect("equal")
ax[0].set_xlabel("x fwd (m)"); ax[0].set_ylabel("y left (m)")

im1 = ax[1].imshow(above_features.T, origin="lower", extent=[-32,32,-32,32], cmap="magma")
ax[1].set_title("(2) splat_points histogram\n>0.2m points  ->  256x256 density")
ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("y (m)")
plt.colorbar(im1, ax=ax[1], fraction=0.046)

ax[2].imshow(above_features.T, origin="lower", cmap="gray")
ax[2].set_title("(3) The actual BEV tensor fed to model\nshape (1,256,256), values 0..1")
ax[2].set_xlabel("pixel col"); ax[2].set_ylabel("pixel row")

fig.suptitle("NAVSIM / DiffusionDrive  BEV input = LiDAR top-down histogram  (SYNTHETIC demo, NAVSIM params & code)", fontsize=13)
plt.tight_layout()
out = "/data/DongBaorong/BEVLOOK/_navsim_bev_demo.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved:", out)
