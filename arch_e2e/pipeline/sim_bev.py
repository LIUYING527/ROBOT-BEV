"""仿真器 BEV 生成件 —— 喂 DiffusionDrive 的 lidar_feature。（arch_e2e 解耦版）

⭐原则(用户要求): 优先官方代码, 不手搓。
主路径 official_lidar_feature() 直接调官方直方图逻辑
(navsim/agents/diffusiondrive/transfuser_features.py::_get_lidar_feature,
 学长 robot_data_prep/bev.py::points_to_bev_tensor 是其逐行拷贝),
输出格式与原始 DiffusionDrive 完全一致 [1,256,256]。

与 repo 版唯一区别: 从本地 vendor/ 引入 robot_data_prep.bev, 不再 sys.path 到 robot_orig_code。
"""
import numpy as np
from _paths import inject_vendor
inject_vendor()
from robot_data_prep.bev import BEVParams, points_to_bev_tensor   # ⭐官方直方图(vendored)


# ---- 官方参数(= transfuser_config.py, ±32m 车端默认; 室内重训可改 range) ----
def default_bev_params(lidar_range=32.0, pixels_per_meter=None, split_height=0.2):
    if pixels_per_meter is None:
        pixels_per_meter = 256.0 / (2 * lidar_range)   # 保持 256×256
    return BEVParams(-lidar_range, lidar_range, -lidar_range, lidar_range,
                     pixels_per_meter, hist_max_per_pixel=5, max_height_lidar=100.0,
                     lidar_split_height=split_height, use_ground_plane=False)


def official_lidar_feature(ego_xyz, lidar_range=32.0):
    """⭐主路径: ego点云 -> 官方 DiffusionDrive lidar_feature [1,256,256] float 0~1.
    直接用官方 points_to_bev_tensor。lidar_range=32 对齐原始; 室内重训用 4~6。"""
    p = default_bev_params(lidar_range)
    return points_to_bev_tensor(np.asarray(ego_xyz, np.float64), p).astype(np.float32)
