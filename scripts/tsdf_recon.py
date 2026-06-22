"""TSDF 从原始ZED米制深度重建(绕开3DGS floater)。
ZED深度=双目米制(npz的Z通道,mm),poses=COLMAP×真实米制尺度8.276。
Open3D ScalableTSDFVolume多帧带权融合→干净几何(无floater)。
输出: tsdf_111450.ply + tsdf_bev_111450.png
"""
import os, sys, glob, numpy as np, open3d as o3d, pycolmap, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"]=["Noto Sans CJK JP"]; matplotlib.rcParams["axes.unicode_minus"]=False
SESS=sys.argv[1] if len(sys.argv)>1 else "111450"
DATA=f"data/{SESS}"
# sparse 模型目录:优先取 sparse 下注册图最多的子模型
_sp=f"outputs/_colmap_{SESS}/sparse"
_subs=[d for d in sorted(glob.glob(f"{_sp}/*")) if os.path.isdir(d) and os.path.exists(f"{d}/cameras.bin")]
MODEL=max(_subs,key=lambda d:pycolmap.Reconstruction(d).num_reg_images()) if _subs else f"{_sp}/0"
print(f"[INFO] {SESS} 用模型 {MODEL}",flush=True)
SCALE=8.276; STRIDE=3; VOX=0.25; TRUNC=12.0
rec=pycolmap.Reconstruction(MODEL); ims=sorted(rec.images.values(),key=lambda im:im.name)
cam=list(rec.cameras.values())[0]; K=np.array(cam.calibration_matrix())

def _metric_scale(rec):
    """ZED米制深度 vs COLMAP相机深度 → 真实尺度 m/单位(按session自算)。"""
    p3=rec.points3D; rs=[]
    for im in list(rec.images.values())[::20]:
        npz=f"{DATA}/pointcloud/zed/{os.path.splitext(im.name)[0]}.npz"
        if not os.path.exists(npz): continue
        Z=np.load(npz)["xyzrgba"].astype(np.float32)[...,2]/1000.0
        R=np.array(im.cam_from_world().rotation.matrix()); t=np.array(im.cam_from_world().translation)
        for p2 in im.points2D:
            if not p2.has_point3D(): continue
            dc=(R@np.array(p3[p2.point3D_id].xyz)+t)[2]
            c,r=int(p2.xy[0]),int(p2.xy[1])
            if 0<=r<720 and 0<=c<1280 and dc>0.1:
                zd=Z[r,c]
                if 0.3<zd<18 and np.isfinite(zd): rs.append(zd/dc)
    return float(np.median(rs)) if rs else 8.276
SCALE=_metric_scale(rec); print(f"[INFO] {SESS} 米制尺度 {SCALE:.3f} m/单位",flush=True)
intr=o3d.camera.PinholeCameraIntrinsic(1280,720,K[0,0],K[1,1],K[0,2],K[1,2])
# 内点相机
cc=np.array([-(np.array(im.cam_from_world().rotation.matrix()).T@np.array(im.cam_from_world().translation)) for im in ims])
cmed=np.median(cc,0); inl=np.linalg.norm(cc-cmed,axis=1)<5*np.median(np.linalg.norm(cc-cmed,axis=1))
def _linemask(xy,tol=2.5):
    m=np.ones(len(xy),bool)
    for _ in range(3):
        p=xy[m];ctr=p.mean(0);_,_,vt=np.linalg.svd(p-ctr);d=vt[0];nn=np.array([-d[1],d[0]])
        m=np.abs((xy-ctr)@nn)<tol
    return m
keep=inl.copy(); keep[np.where(inl)[0]]=_linemask(cc[inl,:2]*SCALE)  # 米制下tol=2.5m
print("[INFO] 剔除误注册相机:",int(keep.sum()),"/",len(keep),flush=True)
ims=[im for im,k in zip(ims,keep) if k]
vol=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=VOX, sdf_trunc=VOX*4,
     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
n=0
for im in ims[::STRIDE]:
    stem=os.path.splitext(im.name)[0]; npz=f"{DATA}/pointcloud/zed/{stem}.npz"
    jpg=f"{DATA}/images/zed/{im.name}"
    if not (os.path.exists(npz) and os.path.exists(jpg)): continue
    Z=np.load(npz)["xyzrgba"].astype(np.float32)[...,2]/1000.0  # 米 (720,1280)
    rgb=cv2.cvtColor(cv2.imread(jpg),cv2.COLOR_BGR2RGB)
    if rgb.shape[:2]!=(720,1280): rgb=cv2.resize(rgb,(1280,720))
    Z[(Z<0.3)|(Z>TRUNC)|~np.isfinite(Z)]=0
    depth=o3d.geometry.Image((Z*1000).astype(np.uint16))   # mm uint16
    color=o3d.geometry.Image(np.ascontiguousarray(rgb))
    rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(color,depth,depth_scale=1000.0,depth_trunc=TRUNC,convert_rgb_to_intensity=False)
    R=np.array(im.cam_from_world().rotation.matrix()); t=np.array(im.cam_from_world().translation)*SCALE
    ext=np.eye(4); ext[:3,:3]=R; ext[:3,3]=t   # 米制 world->cam
    vol.integrate(rgbd,intr,ext); n+=1
    if n%40==0: print("integrated",n,flush=True)
print("[INFO] 融合帧",n,flush=True)
pcd=vol.extract_point_cloud()
o3d.io.write_point_cloud(f"outputs/tsdf_{SESS}.ply",pcd)
P=np.asarray(pcd.points); C=np.asarray(pcd.colors)
print("[OK] TSDF点云",len(P),f"→ tsdf_{SESS}.ply",flush=True)
# 重力对齐+BEV
import importlib.util
s2=importlib.util.spec_from_file_location("cp","scripts/colmap_postprocess.py"); cp=importlib.util.module_from_spec(s2); s2.loader.exec_module(cp)
Rg=cp.align_gravity(P); P=(Rg@P.T).T; P[:,2]-=np.median(P[:,2])
fig,axs=plt.subplots(1,2,figsize=(22,9))
axs[0].scatter(P[:,0],P[:,1],s=0.3,c=np.clip(C,0,1),alpha=0.6); axs[0].set_aspect("equal"); axs[0].set_title("TSDF俯视(真彩) 米制")
axs[1].scatter(P[:,0],P[:,1],s=0.3,c=P[:,2],cmap="turbo",alpha=0.6); axs[1].set_aspect("equal"); axs[1].set_title("TSDF俯视(色=高度)")
plt.tight_layout(); plt.savefig(f"outputs/tsdf_bev_{SESS}.png",dpi=110); print(f"[OK] tsdf_bev_{SESS}.png",flush=True)
