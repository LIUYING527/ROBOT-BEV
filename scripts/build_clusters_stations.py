"""为线性巡检场景放【等距巡逻站位】device_clusters (K,5)=[x,y,side,x_lo,x_hi]。
⭐用于室外111450: 点云无干净周期立柱(3DGS对远处细立柱重建薄), 检测不可靠 → 沿路径等距设站,
   每站转向 side 侧(此场景铁网/电线杆在行进右侧=side<0)由 VLM 面向分类"是不是目标"(geo_vlm)。
   这仍是"几何候选定在哪转→无空转", 只是候选来自路径均分而非凸出物检测(连续铁网本就该分段巡)。
用法: ~/discoverse_venv/bin/python scripts/build_clusters_stations.py <session> <n_stations> <side> [x_start] [x_end] [half_win]
  side: +1=行进左侧 / -1=行进右侧
产出: outputs/device_clusters_<session>.npy
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sess = sys.argv[1] if len(sys.argv) > 1 else "111450"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
side = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
W = np.load(os.path.join(ROOT, "outputs", f"sim_world_colmapjoint_{sess}.npz"))
wp = W["waypoints_xy"]
# 用出程段(x 单调增到峰值)拟合 y(x), 避免去而复返的返程点
xpk = int(np.argmax(wp[:, 0]))
out_wp = wp[:xpk + 1]
xmin_all, xmax_all = float(out_wp[:, 0].min()), float(out_wp[:, 0].max())
x0 = float(sys.argv[4]) if len(sys.argv) > 4 else xmin_all + 0.15 * (xmax_all - xmin_all)
x1 = float(sys.argv[5]) if len(sys.argv) > 5 else xmax_all - 0.12 * (xmax_all - xmin_all)
half = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

# y(x) 用出程waypoints插值(排序去重)
order = np.argsort(out_wp[:, 0])
xs, ys = out_wp[order, 0], out_wp[order, 1]
_, uniq = np.unique(xs, return_index=True)
xs, ys = xs[uniq], ys[uniq]

sx = np.linspace(x0, x1, n)
arr = []
for x in sx:
    y = float(np.interp(x, xs, ys))
    arr.append([round(float(x), 2), round(y, 2), side, round(float(x) - half, 2), round(float(x) + half, 2)])
arr = np.array(arr, np.float32)
out = os.path.join(ROOT, "outputs", f"device_clusters_{sess}.npy")
np.save(out, arr)
print(f"[ok] {out}  {arr.shape}  side={'左' if side>0 else '右'}({side:+.0f})")
for a in arr:
    print(f"  站位 x={a[0]:5.1f} y={a[1]:5.1f} side={a[2]:+.0f} 触发窗[{a[3]:.1f},{a[4]:.1f}]")
