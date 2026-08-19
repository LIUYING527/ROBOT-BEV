import glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_opening, grey_closing

def load_ego(f):
    a=np.load(f,allow_pickle=True);k=a.files[0]
    p=np.asarray(a[k]);p=p.reshape(-1,p.shape[-1])[:,:3].astype(np.float64)
    p=p[np.isfinite(p).all(1)]/1000.0
    ego=np.stack([p[:,2],-p[:,0],-p[:,1]],axis=1)
    return ego[(ego[:,0]>0.1)&(ego[:,0]<12)&(np.abs(ego[:,1])<10)]

def ism_bev(ego,R=6.0,N=256,NB=900,hmin=0.2,hmax=2.5,min_hits=6,near_pct=8):
    res=2*R/N
    gx,gy,gz=ego[:,0],ego[:,1],ego[:,2]
    rng=np.hypot(gx,gy); az=np.arctan2(gy,gx)
    Hband=(gz>hmin)&(gz<hmax)
    edges=np.linspace(-np.pi,np.pi,NB+1)
    bi=np.clip(np.digitize(az,edges)-1,0,NB-1)
    r_stop=np.full(NB,np.inf); hit=np.zeros(NB,bool); has=np.zeros(NB,bool)
    for b in range(NB):
        m=bi==b
        if not m.any(): continue
        has[b]=True
        ob=m&Hband
        if ob.sum()>=min_hits:
            r_stop[b]=np.percentile(rng[ob],near_pct); hit[b]=True   # 稳健最近墙
        else:
            r_stop[b]=np.percentile(rng[m],95)                        # 无墙:自由到最远可见
    # 逐格分类(向量化)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    cx=(ii+0.5)*res-R; cy=(jj+0.5)*res-R
    cr=np.hypot(cx,cy); ca=np.arctan2(cy,cx)
    cb=np.clip(np.digitize(ca,edges)-1,0,NB-1)
    RS=r_stop[cb]; HIT=hit[cb]; HAS=has[cb]
    grid=np.full((N,N),0.5,np.float32)          # 未知
    free = HAS & (cr < RS-res)
    grid[free]=0.0
    occ = HAS & HIT & (np.abs(cr-RS)<=1.5*res)
    grid[occ]=1.0
    # 障碍点直接盖 + 形态学
    occm=(grid==1.0)
    for x,y in ego[Hband][:,:2]:
        i=int((x+R)/res); j=int((y+R)/res)
        if 0<=i<N and 0<=j<N: occm[i,j]=True
    occm=binary_closing(binary_opening(occm,iterations=1),iterations=2)
    grid[occm]=1.0
    return grid

files=sorted(glob.glob('data/114830/pointcloud/zed/*.npz'))
# 挑一帧走廊直段
f=files[315]; ego=load_ego(f); grid=ism_bev(ego)
print("frame",f.split('/')[-1],"pts",len(ego))
R=6.0
fig,ax=plt.subplots(1,2,figsize=(12,6))
ob=ego[(ego[:,2]>0.2)&(ego[:,2]<2.5)]
ax[0].scatter(ego[:,0],ego[:,1],s=0.12,c='0.75')
ax[0].scatter(ob[:,0],ob[:,1],s=0.15,c=ob[:,2],cmap='viridis')
ax[0].plot(0,0,'^',c='lime',ms=13);ax[0].set_xlim(0,R);ax[0].set_ylim(-R,R);ax[0].set_aspect('equal')
ax[0].set_title("(1) 单帧ZED点云俯视\n彩=障碍高度带0.2~2.5m");ax[0].set_xlabel("前(m)");ax[0].set_ylabel("左(m)")
disp=np.full(grid.shape,0.6,np.float32);disp[grid<0.25]=1.0;disp[grid>0.75]=0.0
ax[1].imshow(disp.T,origin='lower',extent=[-R,R,-R,R],cmap='gray',vmin=0,vmax=1,aspect='equal')
ax[1].plot(0,0,'^',c='lime',ms=13,zorder=5)
ax[1].set_title("(2) 单帧BEV: 逆传感器模型+光线投射(清理版)\n白=可通行 黑=墙 灰=未知");ax[1].set_xlabel("前x(m)");ax[1].set_ylabel("左y(m)")
fig.suptitle("最佳单帧效果: ZED深度 → 逆传感器模型BEV (走廊114830一帧, ±6m)",fontsize=13)
plt.tight_layout();plt.savefig('_ism_best_114830.png',dpi=120,bbox_inches='tight');print("saved")
