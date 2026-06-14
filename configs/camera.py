"""相机内参与 BEV 地图参数。

⚠️ 待确认项（P0 阻塞）：相机型号、内参 fx/fy/cx/cy、分辨率、深度单位。
   拿到真实标定值后替换下面的 PLACEHOLDER。
"""

# === 相机内参（待替换为真实标定值） ===
# 下面是 RealSense / TUM 数据集的常见默认值，仅供 pipeline 验证，
# 真实重建前必须替换。
WIDTH, HEIGHT = 640, 480
FX, FY = 525.0, 525.0          # 焦距（像素）
CX, CY = 319.5, 239.5          # 主点

# === 深度图参数 ===
DEPTH_SCALE = 1000.0           # 深度值 → 米的缩放（uint16 毫米图常用 1000）
DEPTH_TRUNC = 10.0             # 截断距离（米），超过丢弃

# === BEV 占据栅格参数 ===
RESOLUTION = 0.05              # 每格边长（米），5cm
X_RANGE = (-10.0, 10.0)        # 米
Y_RANGE = (-10.0, 10.0)
Z_RANGE = (0.1, 1.5)           # 只保留车体高度范围内的点

# === 运动学（差速模型） ===
DT = 0.1                       # 仿真步长（秒）


def intrinsic_matrix():
    """返回 3x3 内参矩阵 K。"""
    import numpy as np
    return np.array([[FX, 0.0, CX],
                     [0.0, FY, CY],
                     [0.0, 0.0, 1.0]])


def open3d_intrinsic():
    """返回 Open3D PinholeCameraIntrinsic 对象。"""
    import open3d as o3d
    return o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)
