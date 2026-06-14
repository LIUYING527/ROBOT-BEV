"""多帧拼接成完整场地点云。

两种位姿来源：
  A. 有 SLAM/里程计给出的每帧位姿 (poses.txt) → 直接按位姿变换叠加（首选）。
  B. 无位姿 → 用 Open3D 的逐帧 ICP 配准估计相对位姿（兜底，易漂移）。

⚠️ 待确认：是否有 poses.txt？格式如何（4x4 矩阵 / 平移+四元数 / TUM 格式）？

用法：
    cd robot_bev_sim
    python scripts/03_merge_pcd.py [step]
"""
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import camera as cfg  # noqa: E402

import importlib.util  # noqa: E402

# 复用 02 里的点云构建函数
_spec = importlib.util.spec_from_file_location(
    "single_frame", os.path.join(os.path.dirname(__file__), "02_single_frame_pcd.py")
)
_single = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_single)
build_point_cloud = _single.build_point_cloud

POSES_FILE = "data/poses.txt"
VOXEL = cfg.RESOLUTION  # 下采样体素，控制点数


def load_poses(path):
    """读取每帧 4x4 位姿。支持两种格式：
    - 每行 16 个数 (展平的 4x4)
    - TUM: timestamp tx ty tz qx qy qz qw  (需要 scipy)
    返回 list[np.ndarray(4,4)] 或 None。
    """
    if not os.path.exists(path):
        return None
    poses = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
            elif len(vals) == 8:  # TUM
                from scipy.spatial.transform import Rotation
                _, tx, ty, tz, qx, qy, qz, qw = vals
                T = np.eye(4)
                T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                T[:3, 3] = [tx, ty, tz]
                poses.append(T)
    return poses or None


def merge_with_poses(frame_ids, poses):
    merged = o3d.geometry.PointCloud()
    for fid in frame_ids:
        name = f"{fid:06d}.png"
        pcd = build_point_cloud(
            os.path.join("data/color", name),
            os.path.join("data/depth", name),
        )
        pcd.transform(poses[fid])
        merged += pcd
        merged = merged.voxel_down_sample(VOXEL)
    return merged


def merge_with_icp(frame_ids):
    """无位姿兜底：逐帧 ICP。仅供 demo，长序列会漂移。"""
    merged = None
    prev = None
    cur_T = np.eye(4)
    for fid in frame_ids:
        name = f"{fid:06d}.png"
        pcd = build_point_cloud(
            os.path.join("data/color", name),
            os.path.join("data/depth", name),
        ).voxel_down_sample(VOXEL)
        if prev is None:
            merged = pcd
        else:
            reg = o3d.pipelines.registration.registration_icp(
                pcd, prev, VOXEL * 2, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            cur_T = cur_T @ reg.transformation
            pcd.transform(cur_T)
            merged += pcd
            merged = merged.voxel_down_sample(VOXEL)
        prev = pcd
    return merged


def main():
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_color = len(os.listdir("data/color"))
    frame_ids = list(range(0, n_color, step))
    print(f"[INFO] 拼接帧: {len(frame_ids)} 帧 (step={step})")

    poses = load_poses(POSES_FILE)
    if poses is not None:
        print(f"[INFO] 使用 poses.txt ({len(poses)} 个位姿)")
        merged = merge_with_poses(frame_ids, poses)
    else:
        print("[WARN] 未找到 poses.txt，回退到逐帧 ICP（可能漂移）")
        merged = merge_with_icp(frame_ids)

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/scene.ply"
    o3d.io.write_point_cloud(out, merged)
    print(f"[OK] 已保存 {out}  点数: {len(merged.points)}")


if __name__ == "__main__":
    main()
