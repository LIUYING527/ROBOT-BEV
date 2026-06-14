"""单帧 RGB-D → 点云（验证 pipeline）。

依赖：configs/camera.py 里的相机内参确认后才能得到正确尺度。

用法：
    cd robot_bev_sim
    python scripts/02_single_frame_pcd.py [frame_id]
"""
import os
import sys

import open3d as o3d

# 允许从 scripts/ 直接运行时导入 configs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import camera as cfg  # noqa: E402


def build_point_cloud(color_path, depth_path):
    color_raw = o3d.io.read_image(color_path)
    depth_raw = o3d.io.read_image(depth_path)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        depth_scale=cfg.DEPTH_SCALE,
        depth_trunc=cfg.DEPTH_TRUNC,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cfg.open3d_intrinsic())
    # 相机坐标系 → 习惯的世界坐标系（翻转 y/z）
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    return pcd


def main():
    frame_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    name = f"{frame_id:06d}.png"
    color_path = os.path.join("data/color", name)
    depth_path = os.path.join("data/depth", name)

    pcd = build_point_cloud(color_path, depth_path)
    print(f"[INFO] 点数: {len(pcd.points)}")

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/single_frame.ply"
    o3d.io.write_point_cloud(out, pcd)
    print(f"[OK] 已保存 {out}")

    # 无显示环境下注释掉下面这行
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
