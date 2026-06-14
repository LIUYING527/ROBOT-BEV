"""点云 → BEV 占据栅格。

读取 outputs/scene.ply（多帧拼接结果），沿 Z 轴投影成 2D 占据栅格，
保存 bev_map.npy（供仿真加载）和 bev_map.png（供汇报截图）。

用法：
    cd robot_bev_sim
    python scripts/04_pcd_to_bev.py [input.ply]
"""
import os
import sys

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import camera as cfg  # noqa: E402


def pcd_to_bev(points):
    x_min, x_max = cfg.X_RANGE
    y_min, y_max = cfg.Y_RANGE
    z_min, z_max = cfg.Z_RANGE
    res = cfg.RESOLUTION

    mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    points = points[mask]

    W = int((x_max - x_min) / res)
    H = int((y_max - y_min) / res)
    grid = np.zeros((H, W), dtype=np.uint8)

    ix = ((points[:, 0] - x_min) / res).astype(int)
    iy = ((points[:, 1] - y_min) / res).astype(int)
    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    grid[iy[valid], ix[valid]] = 1
    return grid


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/scene.ply"
    pcd = o3d.io.read_point_cloud(in_path)
    points = np.asarray(pcd.points)
    print(f"[INFO] 输入点数: {len(points)}")

    grid = pcd_to_bev(points)
    occ = int(grid.sum())
    print(f"[INFO] 占据栅格: {grid.shape}  占据格数: {occ} ({occ / grid.size:.1%})")

    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/bev_map.npy", grid)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="gray_r", origin="lower")
    plt.title("BEV Occupancy Grid")
    plt.tight_layout()
    plt.savefig("outputs/bev_map.png", dpi=150)
    print("[OK] 已保存 outputs/bev_map.npy 和 outputs/bev_map.png")


if __name__ == "__main__":
    main()
