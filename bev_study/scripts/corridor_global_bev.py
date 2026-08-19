import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,'.'); from sim_bev import OccBEV
from scipy.ndimage import binary_closing, binary_opening
occ=OccBEV('outputs/bev_logodds_114830.npz')
xr,yr,res=occ.xr,occ.yr,occ.res; prob,obsv=occ.prob,occ.obsv
txy=occ.cams[:,:2]
occ_m=binary_closing(binary_opening(obsv&(prob>0.60)),iterations=1)
disp=np.full(prob.shape,0.6,np.float32); disp[obsv&(prob<0.40)]=1.0; disp[occ_m]=0.0
fig,ax=plt.subplots(figsize=(6.5,9))
ax.imshow(disp.T,origin='lower',extent=[xr[0],xr[1],yr[0],yr[1]],cmap='gray',vmin=0,vmax=1,aspect='equal')
ax.plot(txy[:,0],txy[:,1],'-',c='red',lw=1.6,label='行进轨迹(边走边建图)')
ax.plot(txy[0,0],txy[0,1],'o',c='deepskyblue',ms=9,label='起点')
ax.plot(txy[-1,0],txy[-1,1],'s',c='orange',ms=9,label='终点')
ax.set_title("走廊 114830 —— 边走边建图 融合占据栅格 (世界坐标, 已对齐)\n黑=墙/占据  白=可通行  灰=未观测  |  网格0.05m, ~38×84m",fontsize=11)
ax.set_xlabel("world x (m)");ax.set_ylabel("world y (m)");ax.legend(loc='upper right',fontsize=9)
ax.grid(True,alpha=0.15)
plt.tight_layout();plt.savefig('_corridor_bev_114830.png',dpi=120,bbox_inches='tight');print("saved")
