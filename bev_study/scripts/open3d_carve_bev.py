"""单帧 ZED 深度 → 局部占据 BEV，用 Open3D 官方空间雕刻 (VoxelGrid.carve_depth_map)。
正确解(官方库,非手搓): 稠密体素 → 用深度图雕掉视线前方的自由空间 → 剩下=表面/遮挡。
free = 被雕掉(视线畅通)  occupied = 障碍高度带的点  unknown = 其余(视场外/墙后)。
输出: bev_study/outputs/open3d_carve_bev_114830.png
"""
import glob, numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO="/data/DongBaorong/BEVLOOK/robot_bev_sim"
f=sorted(glob.glob(f"{REPO}/data/114830/pointcloud/zed/*.npz"))[315]
a=np.load(f,allow_pickle=True); arr=np.asarray(a[a.files[0]]).astype(np.float64)  # (720,1280,4) mm, 相机系
H,W=arr.shape[:2]
X,Y,Z=arr[...,0]/1000., arr[...,1]/1000., arr[...,2]/1000.   # m, 相机系(x右 y下 z前)
valid=np.isfinite(X)&np.isfinite(Y)&np.isfinite(Z)&(Z>0.1)&(Z<10)
uu,vv=np.meshgrid(np.arange(W),np.arange(H))

# --- 从有序点云反解内参 (X/Z = u/fx - cx/fx) ---
m=valid
sx,ix=np.polyfit(uu[m], (X[m]/Z[m]), 1); fx=1/sx; cx=-ix*fx
sy,iy=np.polyfit(vv[m], (Y[m]/Z[m]), 1); fy=1/sy; cy=-iy*fy
print(f"内参拟合: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

# --- Open3D 深度图 + 相机参数 ---
depth=np.where(valid, Z, 0).astype(np.float32)
depth_o3d=o3d.geometry.Image(depth)
intr=o3d.camera.PinholeCameraIntrinsic(W,H,fx,fy,cx,cy)
cam=o3d.camera.PinholeCameraParameters(); cam.intrinsic=intr; cam.extrinsic=np.eye(4)

# --- 稠密体素(相机系: x∈[-5,5] y(下)∈[-2,1.5] z(前)∈[0.2,8]) ---
vs=0.1
dense=o3d.geometry.VoxelGrid.create_dense(np.array([-5.,-2.,0.2]),np.array([.5,.5,.5]),vs,10.,3.5,7.8)
di=set(map(tuple,np.array([v.grid_index for v in dense.get_voxels()])))   # ⚠️carve就地改dense,必须先取
carved=dense.carve_depth_map(depth_o3d,cam,keep_voxels_outside_image=True)
ci=set(map(tuple,np.array([v.grid_index for v in carved.get_voxels()])))
free_idx=np.array(list(di-ci))              # 被雕掉=自由
print(f"稠密体素{len(di)} 雕后{len(ci)} 自由{len(free_idx)}")
org=np.array([-5.,-2.,0.2])
free_ctr=org+(free_idx+0.5)*vs              # 相机系自由体素中心 (x右,y下,z前)

# --- 转 ego(前x,左y,上z) 并投影 BEV ±6m ---
def cam2ego(P): return np.stack([P[:,2],-P[:,0],-P[:,1]],axis=1)
free_ego=cam2ego(free_ctr)
pts=np.stack([X[valid],Y[valid],Z[valid]],axis=1); pts_ego=cam2ego(pts)
obst=pts_ego[(pts_ego[:,2]>0.2)&(pts_ego[:,2]<2.5)]   # 障碍高度带

R,N=6.0,256; res=2*R/N; UNK,FREE,OCC=0.5,0.,1.
def cells(P):
    i=((P[:,0]+R)/res).astype(int); j=((P[:,1]+R)/res).astype(int)
    ok=(i>=0)&(i<N)&(j>=0)&(j<N); return i[ok],j[ok]
from scipy.ndimage import binary_closing,binary_opening,binary_dilation
grid=np.full((N,N),UNK,np.float32)
fm=np.zeros((N,N),bool); fi,fj=cells(free_ego); fm[fi,fj]=True
fm=binary_dilation(fm,iterations=int(round(vs/res)))   # 每个0.1m自由体素填满其footprint(消网点)
grid[fm]=FREE
om0=np.zeros((N,N),bool); oi,oj=cells(obst); om0[oi,oj]=True
om=binary_closing(binary_opening(om0,iterations=1),iterations=2)
grid[om]=OCC                                       # 障碍盖在自由之上

# --- 画 ---
fig,ax=plt.subplots(1,2,figsize=(12,6))
ax[0].scatter(pts_ego[:,0],pts_ego[:,1],s=0.1,c='0.75')
ax[0].scatter(obst[:,0],obst[:,1],s=0.15,c=obst[:,2],cmap='viridis')
ax[0].plot(0,0,'^',c='lime',ms=13);ax[0].set_xlim(0,R);ax[0].set_ylim(-R,R);ax[0].set_aspect('equal')
ax[0].set_title("(1) 单帧ZED点云俯视");ax[0].set_xlabel("前(m)");ax[0].set_ylabel("左(m)")
disp=np.full(grid.shape,0.6,np.float32);disp[grid==FREE]=1.0;disp[grid==OCC]=0.0
ax[1].imshow(disp.T,origin='lower',extent=[-R,R,-R,R],cmap='gray',vmin=0,vmax=1,aspect='equal')
ax[1].plot(0,0,'^',c='lime',ms=13,zorder=5)
ax[1].set_title("(2) Open3D空间雕刻占据BEV\n白=可通行 黑=墙 灰=未知");ax[1].set_xlabel("前x(m)");ax[1].set_ylabel("左y(m)")
fig.suptitle("单帧 ZED深度 → Open3D官方空间雕刻(carve_depth_map) 占据BEV (走廊114830, ±6m)",fontsize=12)
plt.tight_layout()
out=f"{REPO}/bev_study/outputs/open3d_carve_bev_114830.png"; plt.savefig(out,dpi=120,bbox_inches='tight'); print("saved",out)
