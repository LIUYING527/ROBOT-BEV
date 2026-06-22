"""抽象2D仿真——墙=TSDF几何提两侧开放facade(室外开阔不闭口),轨迹=清理后真实轨迹。

改进:
- 轨迹:COLMAP中心(按时间序)→空间+主线去离群→时间一致性去鬼畜→平滑。
- 墙:障碍按"轨迹左右"分两侧→每侧沿弧长分箱取中位→两道**开放折线facade**(不闭环)。
- 渲染:棋盘+红墙(开放)+箭头(走清理后轨迹)+黄激光+目标。

用法: python scripts/sim2d_tsdf.py [session]
输出: sim2d_<session>_map.png + sim2d_<session>.gif + walls_<session>.json
"""
import os, sys, json, numpy as np, pycolmap, open3d as o3d
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrow
from matplotlib.animation import FuncAnimation, PillowWriter
import importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.render2d import FLOOR_A, FLOOR_B, WALL_C
s2=importlib.util.spec_from_file_location("cp","scripts/colmap_postprocess.py"); cp=importlib.util.module_from_spec(s2); s2.loader.exec_module(cp)
matplotlib.rcParams["font.sans-serif"]=["Noto Sans CJK JP","Droid Sans Fallback"]; matplotlib.rcParams["axes.unicode_minus"]=False

SESS=sys.argv[1] if len(sys.argv)>1 else "111450"
H_GND=0.5; H_TOP=8.0; LAT_MAX=22.0; BIN=2.0
N_RAYS=40; FOV=200; LIDAR_MAX=18.0


def metric_scale(rec):
    """ZED米制深度 vs COLMAP相机深度 → 真实尺度 m/单位。"""
    import glob, os as _os
    p3=rec.points3D; rs=[]
    for im in list(rec.images.values())[::20]:
        npz=f"data/{SESS}/pointcloud/zed/{_os.path.splitext(im.name)[0]}.npz"
        if not _os.path.exists(npz): continue
        Z=np.load(npz)["xyzrgba"].astype(np.float32)[...,2]/1000.0
        R=np.array(im.cam_from_world().rotation.matrix()); t=np.array(im.cam_from_world().translation)
        for p2 in im.points2D:
            if not p2.has_point3D(): continue
            dc=(R@np.array(p3[p2.point3D_id].xyz)+t)[2]
            c,r=int(p2.xy[0]),int(p2.xy[1])
            if 0<=r<720 and 0<=c<1280 and dc>0.1:
                zd=Z[r,c]
                if 0.3<zd<18 and np.isfinite(zd): rs.append(zd/dc)
    return float(np.median(rs)) if rs else 1.0


def clean_traj(C, win=7, tol=3.0):
    """时间一致性去鬼畜:与局部中位差>tol(米)的点剔除,再平滑。C:Nx2按时间序。"""
    keep=np.ones(len(C),bool)
    for _ in range(2):
        idx=np.where(keep)[0]; Ck=C[idx]
        med=np.array([np.median(Ck[max(0,i-win):i+win+1],0) for i in range(len(Ck))])
        keep[idx]=np.linalg.norm(Ck-med,axis=1)<tol
    Cc=C[keep]
    # 移动平均平滑
    k=5; pad=np.pad(Cc,((k,k),(0,0)),mode="edge")
    Cs=np.stack([np.convolve(pad[:,d],np.ones(2*k+1)/(2*k+1),"valid") for d in range(2)],1)
    return Cs


def open_walls(B, traj):
    """轨迹是同一走廊来回多趟→用**全局走廊主轴**的垂直偏移分两侧(不受掉头翻转影响),
    沿主轴分箱取中位偏移→平滑→两道开放facade折线。"""
    ctr=traj.mean(0); _,_,vt=np.linalg.svd(traj-ctr); ax=vt[0]; nrm=np.array([-ax[1],ax[0]])
    strj=(traj-ctr)@ax; s0,s1=strj.min(),strj.max()
    s=(B-ctr)@ax; d=(B-ctr)@nrm
    keep=(np.abs(d)<LAT_MAX)&(s>s0-2)&(s<s1+2); s,d=s[keep],d[keep]
    bins=np.arange(s0,s1+BIN,BIN); walls=[]
    for sgn in (1,-1):
        m=np.sign(d)==sgn
        if m.sum()<8: continue
        ss,dd=s[m],d[m]; off=[]; cs=[]
        for b in bins:
            sel=(ss>=b)&(ss<b+BIN)
            if sel.sum()>=3: off.append(np.median(dd[sel])); cs.append(b+BIN/2)
        if len(off)<2: continue
        off=np.array(off); cs=np.array(cs)
        k=2; off=np.convolve(np.pad(off,k,mode="edge"),np.ones(2*k+1)/(2*k+1),"valid")  # 平滑去锯齿
        pts=ctr[None,:]+cs[:,None]*ax[None,:]+off[:,None]*nrm[None,:]
        for i in range(len(pts)-1):
            if cs[i+1]-cs[i]<3*BIN:   # 不连过大间隙(开口)
                walls.append([pts[i,0],pts[i,1],pts[i+1,0],pts[i+1,1]])
    return walls


def build():
    rec=pycolmap.Reconstruction(f"outputs/_colmap_{SESS}/sparse/1")
    SCALE=metric_scale(rec)
    ims=sorted(rec.images.values(),key=lambda im:im.name)
    cc=np.array([-(np.array(im.cam_from_world().rotation.matrix()).T@np.array(im.cam_from_world().translation)) for im in ims])
    cmed=np.median(cc,0); inl=np.linalg.norm(cc-cmed,axis=1)<5*np.median(np.linalg.norm(cc-cmed,axis=1))
    km=cp.clean_camera_mask(cc[inl]*SCALE); keep=inl.copy(); keep[np.where(inl)[0]]=km
    traj=cc[keep]*SCALE
    P=np.asarray(o3d.io.read_point_cloud(f"outputs/tsdf_{SESS}.ply").points)
    Rg=cp.align_gravity(P); P=(Rg@P.T).T; traj=(Rg@traj.T).T
    g=np.median(P[:,2]); P[:,2]-=g; traj=traj[:,:2]
    traj=clean_traj(traj)
    B=P[(P[:,2]>H_GND)&(P[:,2]<H_TOP)][:,:2]
    walls=open_walls(B,traj)
    allx=np.concatenate([np.array(walls)[:,[0,2]].ravel(),traj[:,0]]) if walls else traj[:,0]
    ally=np.concatenate([np.array(walls)[:,[1,3]].ravel(),traj[:,1]]) if walls else traj[:,1]
    bb=(allx.min()-4,ally.min()-4,allx.max()+4,ally.max()+4)
    print(f"[INFO] {SESS} 尺度{SCALE:.2f} 轨迹{len(traj)}点 墙段{len(walls)} 场景{bb[2]-bb[0]:.0f}×{bb[3]-bb[1]:.0f}m",flush=True)
    return walls,traj,bb


def lidar(px,py,th,Wn):
    out=[]
    for a in th+np.radians(np.linspace(-FOV/2,FOV/2,N_RAYS)):
        dx,dy=np.cos(a),np.sin(a); best=LIDAR_MAX
        for x1,y1,x2,y2 in Wn:
            ex,ey=x2-x1,y2-y1; den=dx*ey-dy*ex
            if abs(den)<1e-9: continue
            t=((x1-px)*ey-(y1-py)*ex)/den; u=((x1-px)*dy-(y1-py)*dx)/den
            if t>0 and 0<=u<=1 and t<best: best=t
        out.append((px+dx*best,py+dy*best))
    return out


def main():
    walls,traj,(x0,y0,x1,y1)=build()
    Wn=np.array(walls); th=np.arctan2(np.gradient(traj[:,1]),np.gradient(traj[:,0]))
    json.dump({"bounds":[x0,y0,x1,y1],"walls":walls},open(f"outputs/walls_{SESS}.json","w"))
    def setup(ax):
        nx=int(np.ceil(x1-x0));ny=int(np.ceil(y1-y0))
        ax.imshow(np.indices((ny,nx)).sum(0)%2,cmap=mcolors.ListedColormap([FLOOR_A,FLOOR_B]),extent=[x0,x1,y0,y1],origin="lower",zorder=0,alpha=0.9)
        ax.add_collection(LineCollection([[(w[0],w[1]),(w[2],w[3])] for w in walls],colors=WALL_C,linewidths=3,zorder=3))
        ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect("equal");ax.set_xlabel("x(m)");ax.set_ylabel("y(m)")
    fig,ax=plt.subplots(figsize=(13,9)); setup(ax)
    ax.plot(traj[:,0],traj[:,1],"-",color="#1b6cff",lw=1.8,alpha=0.8)
    ax.plot(traj[0,0],traj[0,1],"o",color="lime",ms=11,mec="k"); ax.plot(traj[-1,0],traj[-1,1],"s",color="red",ms=10,mec="k")
    ax.set_title(f"{SESS} 抽象2D仿真(开放墙=两侧facade, 轨迹已清理, 米制)")
    fig.tight_layout(); fig.savefig(f"outputs/sim2d_{SESS}_map.png",dpi=130); plt.close(fig)
    print(f"[OK] sim2d_{SESS}_map.png",flush=True)
    N=min(160,len(traj)); idx=np.linspace(0,len(traj)-1,N).astype(int)
    fig,ax=plt.subplots(figsize=(13,9)); setup(ax); ax.set_title(f"{SESS} 2D仿真:箭头(清理轨迹)+激光")
    trail,=ax.plot([],[],"-",color="#1b6cff",lw=2,zorder=5)
    rayc=LineCollection([],colors="#e6c200",linewidths=0.7,alpha=0.85,zorder=4); ax.add_collection(rayc)
    hold={"p":None}
    def upd(j):
        i=idx[j]; px,py=traj[i]; a=th[i]
        trail.set_data(traj[:i+1,0],traj[:i+1,1])
        rayc.set_segments([[(px,py),(ex,ey)] for ex,ey in lidar(px,py,a,Wn)])
        if hold["p"] is not None: hold["p"].remove()
        hold["p"]=ax.add_patch(FancyArrow(px,py,2.0*np.cos(a),2.0*np.sin(a),width=0.6,head_width=1.6,head_length=1.2,color="#111",zorder=6))
        return trail,rayc
    FuncAnimation(fig,upd,frames=N,interval=70,blit=False).save(f"outputs/sim2d_{SESS}.gif",writer=PillowWriter(fps=15))
    print(f"[OK] sim2d_{SESS}.gif",flush=True)


if __name__=="__main__":
    main()
