"""系统 py3.10(有 pycolmap/open3d)预计算每帧 map->camera 位姿,存 npz 给 ros2 馈送节点用。
用法: OPENBLAS_NUM_THREADS=1 python3 scripts/precompute_feeder_poses.py <session>
输出: outputs/feeder_poses_<session>.npz (names, Rot Nx3x3, Trans Nx3)
"""
import os, sys, numpy as np, pycolmap, open3d as o3d, importlib.util
s2=importlib.util.spec_from_file_location("cp","scripts/colmap_postprocess.py"); cp=importlib.util.module_from_spec(s2); s2.loader.exec_module(cp)
SESS=sys.argv[1] if len(sys.argv)>1 else "111450"
DATA=f"data/{SESS}"; MODEL=f"outputs/_colmap_{SESS}/sparse/1"
rec=pycolmap.Reconstruction(MODEL); ims=sorted(rec.images.values(),key=lambda im:im.name)
p3=rec.points3D; rs=[]
for im in ims[::20]:
    npz=f"{DATA}/pointcloud/zed/{os.path.splitext(im.name)[0]}.npz"
    if not os.path.exists(npz): continue
    Z=np.load(npz)["xyzrgba"].astype(np.float32)[...,2]/1000.0
    R=np.array(im.cam_from_world().rotation.matrix()); t=np.array(im.cam_from_world().translation)
    for p2 in im.points2D:
        if not p2.has_point3D(): continue
        dc=(R@np.array(p3[p2.point3D_id].xyz)+t)[2]; c,r=int(p2.xy[0]),int(p2.xy[1])
        if 0<=r<720 and 0<=c<1280 and dc>0.1:
            zd=Z[r,c]
            if 0.3<zd<18 and np.isfinite(zd): rs.append(zd/dc)
SCALE=float(np.median(rs)) if rs else 8.276
cc=np.array([-(np.array(im.cam_from_world().rotation.matrix()).T@np.array(im.cam_from_world().translation)) for im in ims])
cmed=np.median(cc,0); inl=np.linalg.norm(cc-cmed,axis=1)<5*np.median(np.linalg.norm(cc-cmed,axis=1))
km=cp.clean_camera_mask(cc[inl]*SCALE); keep=inl.copy(); keep[np.where(inl)[0]]=km
P=np.asarray(o3d.io.read_point_cloud(f"outputs/tsdf_{SESS}.ply").points)
Rg=cp.align_gravity(P); gz=float(np.median((Rg@P.T).T[:,2]))
names=[]; Rots=[]; Trans=[]
for im,k in zip(ims,keep):
    if not k: continue
    R=np.array(im.cam_from_world().rotation.matrix()); t=np.array(im.cam_from_world().translation)
    Rwc=R.T; twc=-R.T@t*SCALE
    Rot=Rg@Rwc; Tr=Rg@twc; Tr[2]-=gz
    names.append(im.name); Rots.append(Rot); Trans.append(Tr)
np.savez(f"outputs/feeder_poses_{SESS}.npz", names=np.array(names), Rot=np.array(Rots), Trans=np.array(Trans), scale=SCALE, gz=gz)
print(f"[OK] feeder_poses_{SESS}.npz  帧{len(names)} scale={SCALE:.2f} gz={gz:.2f}",flush=True)
