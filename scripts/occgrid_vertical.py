"""标准占据图 —— 按"每格竖直高度跨度"判障碍(保留真实两侧立面,滤掉地面/路面)。

octomap 把地面也投成占据→一坨。地面是平的(格内 z 跨度小)、墙/建筑是竖直的(跨度大),
按格内 z_max-z_min 阈值分离,鲁棒于室外地面倾斜。输出标准 ROS .pgm/.yaml(可喂 nav2/Stage)。

用法: OPENBLAS_NUM_THREADS=1 python3 scripts/occgrid_vertical.py <session>
输出: outputs/occ_vert_<s>.pgm/.yaml + occ_vert_<s>_view.png
"""
import os, sys, numpy as np, open3d as o3d, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"]=["Noto Sans CJK JP"]; matplotlib.rcParams["axes.unicode_minus"]=False
s2=importlib.util.spec_from_file_location("cp","scripts/colmap_postprocess.py"); cp=importlib.util.module_from_spec(s2); s2.loader.exec_module(cp)

SESS=sys.argv[1] if len(sys.argv)>1 else "111450"
RES=0.2; VSPAN=1.2; MINPTS=3; ROAD_SPAN=0.4

P=np.asarray(o3d.io.read_point_cloud(f"outputs/tsdf_{SESS}.ply").points)
Rg=cp.align_gravity(P); P=(Rg@P.T).T; P[:,2]-=np.median(P[:,2])
# 轨迹(用于标自由空间/可行驶区)
tr=None
fp=f"outputs/feeder_poses_{SESS}.npz"
if os.path.exists(fp):
    d=np.load(fp); tr=d["Trans"][:,:2]   # 已重力对齐+地面归零的相机中心

x0,y0=P[:,0].min()-2,P[:,1].min()-2; x1,y1=P[:,0].max()+2,P[:,1].max()+2
W=int(np.ceil((x1-x0)/RES)); H=int(np.ceil((y1-y0)/RES))
ix=np.clip(((P[:,0]-x0)/RES).astype(int),0,W-1); iy=np.clip(((P[:,1]-y0)/RES).astype(int),0,H-1)
cell=iy*W+ix
zmax=np.full(W*H,-1e9); zmin=np.full(W*H,1e9); cnt=np.zeros(W*H,int)
np.maximum.at(zmax,cell,P[:,2]); np.minimum.at(zmin,cell,P[:,2]); np.add.at(cnt,cell,1)
span=zmax-zmin
occ=np.full(W*H,205,np.uint8)            # 205 未知
has=cnt>=MINPTS
occ[has&(span<ROAD_SPAN)]=254            # 平坦=可行驶(自由)
occ[has&(span>=VSPAN)]=0                 # 竖直结构=占据(墙/立面)
# 轨迹经过的格子强制自由(机器人真走过=可行驶)
if tr is not None:
    tix=np.clip(((tr[:,0]-x0)/RES).astype(int),0,W-1); tiy=np.clip(((tr[:,1]-y0)/RES).astype(int),0,H-1)
    occ[tiy*W+tix]=254
occ=occ.reshape(H,W)
nocc=int((occ==0).sum()); nfree=int((occ==254).sum())
print(f"[INFO] {SESS} {W}x{H}@{RES}m 占据{nocc} 自由{nfree} (VSPAN={VSPAN}m)",flush=True)

# 存标准 .pgm/.yaml(原点左下,y向上→pgm行需翻转)
from PIL import Image
Image.fromarray(np.flipud(occ)).save(f"outputs/occ_vert_{SESS}.pgm")
with open(f"outputs/occ_vert_{SESS}.yaml","w") as f:
    f.write(f"image: occ_vert_{SESS}.pgm\nmode: trinary\nresolution: {RES}\n"
            f"origin: [{x0:.3f}, {y0:.3f}, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
print(f"[OK] occ_vert_{SESS}.pgm/.yaml",flush=True)

# 可视化
ext=[x0,x1,y0,y1]
fig,axs=plt.subplots(1,2,figsize=(22,9))
oct_pgm=f"outputs/octomap_{SESS}.pgm"
if os.path.exists(oct_pgm):
    import re
    om=np.array(Image.open(oct_pgm)); oy=open(f"outputs/octomap_{SESS}.yaml").read()
    oox,ooy=[float(v) for v in re.search(r"origin: \[([-0-9.]+), ([-0-9.]+)",oy).groups()]
    oe=[oox,oox+om.shape[1]*RES,ooy,ooy+om.shape[0]*RES]
    axs[0].imshow(om,cmap="gray",origin="lower",extent=oe,vmin=0,vmax=255)
axs[0].set_aspect("equal"); axs[0].set_title("octomap 原始(地面也投→一坨)")
cmap=matplotlib.colors.ListedColormap(["#222222","#bdbdbd","#ffffff"])  # 0占据黑,205未知灰,254自由白
disp=np.zeros_like(occ); disp[occ==205]=1; disp[occ==254]=2
axs[1].imshow(disp,cmap=cmap,origin="lower",extent=ext,vmin=0,vmax=2)
if tr is not None: axs[1].plot(tr[:,0],tr[:,1],"-",color="#1b6cff",lw=1.2,alpha=0.7)
axs[1].set_aspect("equal"); axs[1].set_title(f"竖直跨度占据图(黑=两侧立面 白=可行驶 灰=未知)")
plt.tight_layout(); plt.savefig(f"outputs/occ_vert_{SESS}_view.png",dpi=110)
print(f"[OK] occ_vert_{SESS}_view.png",flush=True)
