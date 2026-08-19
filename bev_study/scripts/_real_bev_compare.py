"""真实数据对比: NAVSIM LiDAR vs 我们 ZED 深度 —— 都走同一官方 points_to_bev_tensor。
展示域差: NAVSIM=360°环视填满 ±32m; 我们=前向锥, ±32m 只占中间一条缝。
2×2: 上=NAVSIM(点云俯视+BEV), 下=我们ZED(点云俯视+BEV)。输出 bev_study/outputs + docs/assets。
"""
import sys, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
# --- 中文字体 ---
_fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_fp); plt.rcParams["font.family"]=fm.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"]=False
sys.path.insert(0,'/data/DongBaorong/BEVLOOK/robot_orig_code')
from robot_data_prep.bev import BEVParams, points_to_bev_tensor
REPO="/data/DongBaorong/BEVLOOK/robot_bev_sim"

def bev_params(rng, split=0.2):
    return BEVParams(-rng,rng,-rng,rng,256.0/(2*rng),hist_max_per_pixel=5,
                     max_height_lidar=100.0,lidar_split_height=split,use_ground_plane=False)

# --- NAVSIM pcd ---
pcd=glob.glob(f'{REPO}/bev_study/data/navsim_sample/**/*.pcd',recursive=True)[0]
raw=open(pcd,'rb').read(); he=raw.index(b'DATA binary\n')+len(b'DATA binary\n')
n=int([l for l in raw[:he].split(b'\n') if l.startswith(b'POINTS')][0].split()[1])
dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('i','u1'),('li','u1'),('ring','u1')])
pc=np.frombuffer(raw[he:he+n*dt.itemsize],dtype=dt)
nav=np.stack([pc['x'],pc['y'],pc['z']],1).astype(np.float64)
nav=nav[(np.abs(nav[:,0])<32)&(np.abs(nav[:,1])<32)]            # 裁到±32m显示
nav_bev=points_to_bev_tensor(nav,bev_params(32.0))[0]

# --- 我们 ZED 深度 (mm, cam系→ego[z,-x,-y]) ---
zn=sorted(glob.glob(f'{REPO}/../robot_orig_code/20260421_101905/pointcloud/zed/*.npz'))
a=np.load(zn[len(zn)//2],allow_pickle=True);k=a.files[0]
p=np.asarray(a[k]).reshape(-1,np.asarray(a[k]).shape[-1])[:,:3].astype(np.float64)
p=p[np.isfinite(p).all(1)]/1000.0
zed=np.stack([p[:,2],-p[:,0],-p[:,1]],1)
zed_bev=points_to_bev_tensor(zed,bev_params(32.0))[0]           # 同±32m
zed_clip=zed[(zed[:,0]<15)&(np.abs(zed[:,1])<12)]              # 裁远立体噪(仅显示点云用)

# --- 画 2×2 ---
fig,ax=plt.subplots(2,2,figsize=(12,12))
hs=dict(s=0.2,cmap='turbo',vmin=-2,vmax=6)
# NAVSIM
sc=ax[0,0].scatter(nav[::4,0],nav[::4,1],c=nav[::4,2],**hs)
ax[0,0].set_title(f"① NAVSIM 真实 LiDAR 点云俯视\n{len(nav)}点  360°环视  色=高度z")
ax[0,1].imshow(nav_bev.T,origin='lower',extent=[-32,32,-32,32],cmap='magma')
ax[0,1].set_title("② NAVSIM BEV (±32m, 256²)\n四周填满")
# OURS
ax[1,0].scatter(zed_clip[::4,0],zed_clip[::4,1],c=zed_clip[::4,2],**hs)
ax[1,0].set_title(f"③ 我们 ZED 深度点云俯视\n{len(zed)}点  只有前向锥  色=高度z")
ax[1,1].imshow(zed_bev.T,origin='lower',extent=[-32,32,-32,32],cmap='magma')
ax[1,1].set_title("④ 我们 ZED BEV (同±32m, 同函数)\n只占中间一条缝 → 域差")
for a_ in (ax[0,0],ax[1,0]):
    a_.plot(0,0,'^',c='k',ms=11); a_.set_xlim(-32,32); a_.set_ylim(-32,32); a_.set_aspect('equal')
    a_.set_xlabel("前 x(m)"); a_.set_ylabel("左 y(m)")
for a_ in (ax[0,1],ax[1,1]):
    a_.set_xlabel("前 x(m)"); a_.set_ylabel("左 y(m)")
cb=fig.colorbar(sc,ax=ax[:,0].tolist(),shrink=0.5,location='left',pad=0.08); cb.set_label("高度 z (m)")
fig.suptitle("同一官方函数 points_to_bev_tensor,两种点源\nNAVSIM=360°环视稠密LiDAR(填满) | 我们=ZED前视深度锥(一条缝) —— 域差=视场覆盖,非算法",fontsize=13)
out=f"{REPO}/bev_study/outputs/_real_bev_compare.png"; plt.savefig(out,dpi=110,bbox_inches='tight'); print("saved",out)
import shutil; shutil.copy(out, f"{REPO}/docs/assets/_real_bev_compare.png"); print("copied to docs/assets")
