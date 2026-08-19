"""短窗融合 BEV: 一个时刻附近~15帧 ZED深度用位姿拼成局部点云 → ego对齐局部BEV。
比单帧强得多(填满前向锥),又还是"一瞬间"(~2s窗)。位姿来自 feeder_poses(Rot=cam→world重力对齐, Tr=世界相机中心)。
world = Rot @ p_cam(米) + Tr。 输出 bev_study/outputs/short_window_bev_114830.png
"""
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_opening

REPO="/data/DongBaorong/BEVLOOK/robot_bev_sim"
g=np.load(f"{REPO}/outputs/feeder_poses_114830.npz",allow_pickle=True)
names=[os.path.splitext(str(n))[0] for n in g["names"]]; Rot=g["Rot"]; Tr=g["Trans"]
NP=len(names)

def load_cam(stem, stride=3):
    p=f"{REPO}/data/114830/pointcloud/zed/{stem}.npz"
    if not os.path.exists(p): return None
    xyz=np.load(p)["xyzrgba"].astype(np.float32)[...,:3].reshape(-1,3)/1000.0
    xyz=xyz[np.isfinite(xyz).all(1)]
    return xyz[::stride]

C=NP//2; W=7                       # 中心帧 ± W (共~15帧)
cen=Tr[C][:2]
dirv=Tr[min(C+3,NP-1)][:2]-Tr[max(C-3,0)][:2]; yaw=np.arctan2(dirv[1],dirv[0])
c,s=np.cos(yaw),np.sin(yaw)
allw=[]
for i in range(max(0,C-W),min(NP,C+W+1)):
    pc=load_cam(names[i])
    if pc is None: continue
    wpt=(Rot[i]@pc.T).T+Tr[i]                     # → 世界(重力对齐,z上,floor≈0)
    allw.append(wpt)
allw=np.concatenate(allw,0)
print("融合点数",len(allw),"帧",2*W+1,"中心",names[C])

# 世界 → ego(前x 左y), 以中心相机为原点, 朝前=+x
d=allw[:,:2]-cen
ex= c*d[:,0]+s*d[:,1]; ey=-s*d[:,0]+c*d[:,1]; ez=allw[:,2]
ego=np.stack([ex,ey,ez],1)
R,N=6.0,240; res=2*R/N
inb=(np.abs(ex)<R)&(np.abs(ey)<R); ego=ego[inb]
obst=ego[(ego[:,2]>0.2)&(ego[:,2]<2.2)]           # 障碍高度带

def cell(P):
    i=((P[:,0]+R)/res).astype(int); j=((P[:,1]+R)/res).astype(int)
    ok=(i>=0)&(i<N)&(j>=0)&(j<N); return i[ok],j[ok]
occ=np.zeros((N,N),bool); oi,oj=cell(obst); occ[oi,oj]=True
occ=binary_closing(binary_opening(occ,iterations=1),iterations=1)

fig,ax=plt.subplots(1,2,figsize=(13,6.5))
ax[0].scatter(ego[::2,0],ego[::2,1],s=0.15,c=ego[::2,2],cmap='turbo',vmin=-0.2,vmax=2.2)
ax[0].plot(0,0,'^',c='k',ms=13);ax[0].set_xlim(-R,R);ax[0].set_ylim(-R,R);ax[0].set_aspect('equal')
ax[0].set_title(f"(1) 短窗融合点云({2*W+1}帧, ego对齐)\n色=高度");ax[0].set_xlabel("前x(m)");ax[0].set_ylabel("左y(m)")
disp=np.where(occ,0.0,1.0)
ax[1].imshow(disp.T,origin='lower',extent=[-R,R,-R,R],cmap='gray',vmin=0,vmax=1,aspect='equal')
ax[1].plot(0,0,'^',c='lime',ms=14,zorder=5)
ax[1].set_title("(2) 局部占据BEV(黑=墙/障碍 白=空)");ax[1].set_xlabel("前x(m)");ax[1].set_ylabel("左y(m)")
fig.suptitle(f"ZED深度 短窗融合(~{2*W+1}帧≈2s) 局部BEV — 走廊114830 (±{R}m, ego对齐)",fontsize=12)
plt.tight_layout()
out=f"{REPO}/bev_study/outputs/short_window_bev_114830.png"; plt.savefig(out,dpi=120,bbox_inches='tight'); print("saved",out)
