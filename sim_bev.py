"""仿真器 BEV 生成件 —— 喂 DiffusionDrive 的 lidar_feature。

⭐原则(用户要求): 优先官方代码,不手搓。
主路径 official_lidar_feature() 直接调官方直方图逻辑
(navsim/agents/diffusiondrive/transfuser_features.py::_get_lidar_feature,
 学长 robot_data_prep/bev.py::points_to_bev_tensor 是其逐行拷贝),
输出格式与原始 DiffusionDrive 完全一致 [1,256,256]。

log-odds 占据(OccBEV)只作可选的"世界底图可视化",非模型输入,别用于对齐原始模型。
"""
import sys, numpy as np
sys.path.insert(0, '/data/DongBaorong/BEVLOOK/robot_orig_code')  # 官方 bev.py
from robot_data_prep.bev import BEVParams, points_to_bev_tensor   # ⭐官方直方图


# ---- 官方参数(= transfuser_config.py, ±32m 车端默认; 室内重训可改 range) ----
def default_bev_params(lidar_range=32.0, pixels_per_meter=None, split_height=0.2):
    if pixels_per_meter is None:
        pixels_per_meter = 256.0 / (2 * lidar_range)   # 保持 256×256
    return BEVParams(-lidar_range, lidar_range, -lidar_range, lidar_range,
                     pixels_per_meter, hist_max_per_pixel=5, max_height_lidar=100.0,
                     lidar_split_height=split_height, use_ground_plane=False)


def depth_npz_to_ego(npz_path):
    """ZED 深度点云 npz -> ego 系点云 (x前 y左 z上, 米). 相机系 remap 同上车 _depth_to_xyz_m。"""
    a = np.load(npz_path, allow_pickle=True); k = a.files[0]
    p = np.asarray(a[k]).reshape(-1, np.asarray(a[k]).shape[-1])[:, :3].astype(np.float64)
    p = p[np.isfinite(p).all(1)] / 1000.0
    return np.stack([p[:, 2], -p[:, 0], -p[:, 1]], axis=1)


def official_lidar_feature(ego_xyz, lidar_range=32.0):
    """⭐主路径: ego点云 -> 官方 DiffusionDrive lidar_feature [1,256,256] float 0~1.
    直接用官方 points_to_bev_tensor。lidar_range=32 对齐原始;室内重训用 4~6。"""
    p = default_bev_params(lidar_range)
    return points_to_bev_tensor(np.asarray(ego_xyz, np.float64), p).astype(np.float32)


def lidar_feature_from_depth_npz(npz_path, lidar_range=32.0):
    return official_lidar_feature(depth_npz_to_ego(npz_path), lidar_range)


# ---------------- 以下为可选: log-odds 占据世界底图(非模型输入,勿用于对齐原始) ----------------
class OccBEV:
    """可选: 从 log-odds 多帧占据全局图 ego 裁窗。仅作可视化/规划底图。"""
    def __init__(self, npz='outputs/bev_logodds_114830.npz'):
        g = np.load(npz, allow_pickle=True)
        self.prob = g['prob']; self.obsv = g['obsv']
        self.xr = tuple(g['xr']); self.yr = tuple(g['yr']); self.res = float(g['res'])
        self.cams = g['cams']; self.nx, self.ny = self.prob.shape

    def ego_bev(self, pos_xy, yaw, R=6.0, N=256):
        soft = np.where(self.obsv, self.prob, 0.5).astype(np.float32)
        ff = np.linspace(-R, R, N); ll = np.linspace(R, -R, N)
        FL, FF = np.meshgrid(ll, ff); c, s = np.cos(yaw), np.sin(yaw)
        wx = pos_xy[0] + FF*c - FL*s; wy = pos_xy[1] + FF*s + FL*c
        ix = ((wx - self.xr[0])/self.res).astype(np.int32); iy = ((wy - self.yr[0])/self.res).astype(np.int32)
        ok = (ix>=0)&(ix<self.nx)&(iy>=0)&(iy<self.ny)
        out = np.full((N, N), 0.5, np.float32); out[ok] = soft[ix[ok], iy[ok]]
        return out[None]


if __name__ == '__main__':
    import glob
    pcs = sorted(glob.glob('data/114830/pointcloud/zed/*.npz'))
    for R in (32.0, 6.0):
        bev = lidar_feature_from_depth_npz(pcs[len(pcs)//2], lidar_range=R)
        print(f'official lidar_feature +/-{R}m: shape={bev.shape} dtype={bev.dtype} '
              f'range[{bev.min():.2f},{bev.max():.2f}] nonzero={int((bev>0).sum())}')
