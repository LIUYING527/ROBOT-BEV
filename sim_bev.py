"""仿真器 BEV 生成件: log-odds 全局占据图 -> 按机器人位姿 ego 裁窗 -> [1,256,256]
用于喂 (原始)DiffusionDrive 的 lidar_feature。语义: 0=可通行 0.5=未知 1=障碍。
"""
import numpy as np

class OccBEV:
    def __init__(self, npz='outputs/bev_logodds_114830.npz'):
        g = np.load(npz, allow_pickle=True)
        self.prob = g['prob']; self.obsv = g['obsv']
        self.xr = tuple(g['xr']); self.yr = tuple(g['yr']); self.res = float(g['res'])
        self.cams = g['cams']
        self.nx, self.ny = self.prob.shape
        # 预算三态图: 0 free / 0.5 unknown / 1 occ
        tri = np.full_like(self.prob, 0.5, np.float32)
        tri[self.obsv & (self.prob < 0.35)] = 0.0
        tri[self.obsv & (self.prob > 0.6)] = 1.0
        self.tri = tri

    def ego_bev(self, pos_xy, yaw, R=6.0, N=256, source='tri'):
        """返回 [1,N,N]: ego居中(下=后 上=前), 左在左. source: 'tri'三态 / 'prob'连续占据概率"""
        M = self.tri if source == 'tri' else self.prob
        ff = np.linspace(-R, R, N)            # forward 轴(行)
        ll = np.linspace(R, -R, N)            # left 轴(列): 左在左
        FL, FF = np.meshgrid(ll, ff)
        c, s = np.cos(yaw), np.sin(yaw)
        wx = pos_xy[0] + FF*c - FL*s          # forward沿heading
        wy = pos_xy[1] + FF*s + FL*c
        ix = ((wx - self.xr[0])/self.res).astype(np.int32)
        iy = ((wy - self.yr[0])/self.res).astype(np.int32)
        ok = (ix>=0)&(ix<self.nx)&(iy>=0)&(iy<self.ny)
        out = np.full((N, N), 0.5 if source=='tri' else 0.0, np.float32)  # 界外=未知
        out[ok] = M[ix[ok], iy[ok]]
        return out[None]                       # [1,N,N]

    def yaw_at(self, i, k=5):
        j = min(i+k, len(self.cams)-1)
        d = self.cams[j][:2] - self.cams[i][:2]
        return float(np.arctan2(d[1], d[0]))

if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    b = OccBEV()
    cmap = ListedColormap(['white', '#8a8a8a', 'black'])
    poses = [60, 140, 210]
    fig, ax = plt.subplots(1, len(poses), figsize=(15, 5.5))
    for j, pi in enumerate(poses):
        pos = b.cams[pi][:2]; yaw = b.yaw_at(pi)
        ego = b.ego_bev(pos, yaw, R=6.0)[0]
        ax[j].imshow(ego, origin='lower', extent=[6,-6,-6,6], cmap=cmap, vmin=0, vmax=1)
        ax[j].plot(0,0,'r^',ms=13)
        ax[j].set_title(f'EGO BEV pose#{pi}  256x256 +/-6m\nfeed DiffusionDrive lidar_feature')
        ax[j].set_xlabel('left(m)'); ax[j].set_ylabel('forward(m)')
        ax[j].text(5.3,5.2,'L',color='b'); ax[j].text(-5.6,5.2,'R',color='b')
    plt.tight_layout(); plt.savefig('outputs/bev_ego_logodds_0702.png', dpi=120, bbox_inches='tight')
    print('SAVED outputs/bev_ego_logodds_0702.png')
