"""几何候选簇 + 近距facing真图VLM确认 → 自主解析任务目标(不手标)。
①世界ply桌面层(z0.85~1.55)DBSCAN出候选簇; ②每簇找最正对的真实采集帧; ③近距VLM yes/no给类别分。
产出 outputs/lab_targets.json + recon_verify/lab_targets_map.png"""
import os, sys, glob, json, numpy as np, cv2, trimesh, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import open3d as o3d
sys.path.insert(0,"scripts"); sys.path.insert(0,"arch_e2e/pipeline")
import sim_fullchain as sf
ROOT="/data/DongBaorong/BEVLOOK/robot_bev_sim"
S="colmapjoint_110744s2"
W=np.load(f"{ROOT}/outputs/sim_world_{S}.npz"); wp=W["waypoints_xy"]; yaw=W["yaw"]
frames=sorted(glob.glob(f"{ROOT}/outputs/vggto_{S}/frames_zed/*.jpg"))

# --- ① 桌面层聚类 ---
P=np.asarray(trimesh.load(f"{ROOT}/outputs/gs_vggto_{S}_world.ply",process=False).vertices)
xlo,xhi=wp[:,0].min()-2.5,wp[:,0].max()+2.5; ylo,yhi=wp[:,1].min()-2.5,wp[:,1].max()+2.5
m=(P[:,2]>0.85)&(P[:,2]<1.55)&(P[:,0]>xlo)&(P[:,0]<xhi)&(P[:,1]>ylo)&(P[:,1]<yhi)
pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(P[m])
pcd=pcd.voxel_down_sample(0.03); pcd,_=pcd.remove_statistical_outlier(20,2.0)
Qd=np.asarray(pcd.points); lab=np.array(pcd.cluster_dbscan(eps=0.13,min_points=30))
cls=[]
for l in range(lab.max()+1):
    c=Qd[lab==l]; ctr=c.mean(0); size=c.max(0)-c.min(0)
    if size[0]>3 or size[1]>3 or len(c)<80: continue
    cls.append((ctr,size,len(c)))
cls.sort(key=lambda x:-x[2])
print(f"候选簇 {len(cls)}")

# --- ② 每簇最正对真实帧 ---
def best_frame(tx,ty):
    best=None
    for i in range(len(wp)):
        dx,dy=tx-wp[i,0],ty-wp[i,1]; dist=np.hypot(dx,dy)
        if dist<0.3 or dist>3.5: continue
        err=abs((np.arctan2(dy,dx)-yaw[i]+np.pi)%(2*np.pi)-np.pi)
        s=err+0.15*dist
        if best is None or s<best[0]: best=(s,i,dist,err)
    return best

# --- ③ 近距VLM类别确认 ---
QS={"cabinet":"Is there a metal cabinet or locker in the center front of this image? Answer yes or no.",
    "microscope":"Is there a microscope in the center front of this image? Answer yes or no.",
    "yellow":"Is there bright yellow equipment or a yellow workbench in the center front of this image? Answer yes or no."}
def yes(a): return (a or "").strip().lower().startswith("y")
recs=[]
for ci,(ctr,size,n) in enumerate(cls):
    b=best_frame(ctr[0],ctr[1])
    if b is None: continue
    _,i,dist,err=b
    if np.degrees(err)>30: continue
    im=cv2.imread(frames[i])
    votes={k:yes(sf.vlm_api(im,q,max_new_tokens=6,timeout=40)[0]) for k,q in QS.items()}
    recs.append({"ci":ci,"xy":[round(float(ctr[0]),2),round(float(ctr[1]),2)],"z":round(float(ctr[2]),2),
                 "size":[round(float(s),2) for s in size],"n":int(n),"frame":int(i),
                 "robot":[round(float(wp[i,0]),2),round(float(wp[i,1]),2)],"votes":votes})
    print(f"C{ci} xy({ctr[0]:.1f},{ctr[1]:.1f}) f#{i} | "+" ".join(f"{k}:{'Y' if v else 'n'}" for k,v in votes.items()))

json.dump({"clusters":recs},open(f"{ROOT}/outputs/lab_targets.json","w"),indent=1)
# 地图
fig,ax=plt.subplots(figsize=(9,10))
ax.scatter(Qd[:,0],Qd[:,1],c='lightgray',s=1,alpha=0.4); ax.plot(wp[:,0],wp[:,1],'-',c='red',lw=0.7,alpha=0.5)
ax.scatter([0],[0],c='lime',s=160,marker='s',zorder=6,label='origin')
for r in recs:
    v=r["votes"]; tag=[k[:3] for k,val in v.items() if val]
    col='blue' if v['cabinet'] else ('orange' if v['yellow'] else ('red' if v['microscope'] else 'gray'))
    ax.scatter([r["xy"][0]],[r["xy"][1]],c=col,s=130,marker='o',edgecolor='k',zorder=5)
    ax.text(r["xy"][0]+0.08,r["xy"][1],f"C{r['ci']}"+("|"+",".join(tag) if tag else ""),fontsize=7,zorder=6)
ax.set_aspect('equal'); ax.legend(); ax.set_title("110744 autonomous targets: geom cluster + facing VLM confirm")
plt.tight_layout(); plt.savefig(f"{ROOT}/outputs/recon_verify/lab_targets_map.png",dpi=110)
print("saved lab_targets.json + lab_targets_map.png")
