"""为合并 session(colmapjoint_all)建 ZED 深度融合稠密初始点云(归一化帧,与recon.ply同坐标)。
fuse_zed.py 只吃单 raw_scene;此处按 frames_zed 名里的 A/B 前缀路由到 data/114830 与 data/113628。
产出: outputs/vggto_<session>/zed_fused.ply  作 3DGS INIT_PLY。

用法: PYTHONPATH=third_party/gs_playground/.venv/lib/python3.10/site-packages \
      ~/discoverse_venv/bin/python scripts/build_zed_init_merged.py colmapjoint_all
"""
import os, sys, glob, importlib.util
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
SESSION = sys.argv[1] if len(sys.argv) > 1 else "colmapjoint_all"
PER = int(os.environ.get("PER_FRAME", "8000"))
VOXEL_M = float(os.environ.get("VOXEL_M", "0.02"))     # 2cm 体素降采样
A_SCENE, B_SCENE = "114830", "113628"

from fuse_zed import decode_zed_npz                      # 复用真彩解码

spec = importlib.util.spec_from_file_location("bev", os.path.join(ROOT, "scripts", "bev_topdown.py"))
bev = importlib.util.module_from_spec(spec); spec.loader.exec_module(bev)
import trimesh, open3d as o3d

vdir = os.path.join(ROOT, "outputs", f"vggto_{SESSION}")
extr = np.load(os.path.join(vdir, "cameras.npz"))["extrinsic"]     # (N,3,4) world2cam 归一化
N = len(extr)
fz = sorted(glob.glob(os.path.join(vdir, "frames_zed", "*")))
names = [os.path.basename(p) for p in fz]
ts = [n.split("_")[-1].split(".")[0] for n in names]
scene = [A_SCENE if "_A_" in n else B_SCENE for n in names]

# k: 归一化×k=米 (复用 align/bev 的重力+地面+相机高1.2m)
R = extr[:, :3, :3]; t = extr[:, :3, 3]
cam_c = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)
pts_v = np.asarray(trimesh.load(os.path.join(vdir, "recon.ply"), process=False).vertices, dtype=np.float64)
g = bev.estimate_gravity_from_cameras(extr)
n_floor, _, _ = bev.ransac_ground_normal(pts_v, g)
Rz = bev.rotation_aligning(n_floor, np.array([0, 0, 1.0]))
rz_cam = cam_c @ Rz.T
ground_z = float(np.percentile((pts_v @ Rz.T)[:, 2], 2.0))
cam_h_norm = float(np.median(rz_cam[:, 2]) - ground_z)
k = 1.2 / max(cam_h_norm, 1e-6); inv_k = 1.0 / k
print(f"[init] N={N} k(norm->m)={k:.3f} 米->归一化×{inv_k:.4f} PER={PER} voxel={VOXEL_M}m", flush=True)

acc = o3d.geometry.PointCloud()
miss = 0
for i in range(N):
    f = os.path.join(ROOT, "data", scene[i], "pointcloud", "zed", ts[i] + ".npz")
    if not os.path.exists(f):
        miss += 1; continue
    xyz_m, rgb = decode_zed_npz(f)
    if len(xyz_m) == 0:
        continue
    if len(xyz_m) > PER:
        sel = np.random.choice(len(xyz_m), PER, replace=False)
        xyz_m, rgb = xyz_m[sel], rgb[sel]
    Ri = extr[i, :3, :3]; ci = -Ri.T @ extr[i, :3, 3]
    wp = (Ri.T @ (xyz_m * inv_k).T).T + ci
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(wp.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    acc += pc
    if i % 200 == 0:
        print(f"  {i}/{N}  累计点 {len(acc.points)}", flush=True)

print(f"[init] 缺深度帧 {miss}/{N}; 融合前 {len(acc.points)} 点", flush=True)
acc = acc.voxel_down_sample(VOXEL_M * inv_k)             # 体素在归一化单位
acc, _ = acc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
out = os.path.join(vdir, "zed_fused.ply")
o3d.io.write_point_cloud(out, acc)
print(f"[ok] -> {out}  ({len(acc.points)} 点, 归一化帧, 真彩)", flush=True)
