"""用 VGGT 位姿把 ZED 每帧度量点云(pointcloud/zed/*.npz)融合进 VGGT 归一化世界帧。

动机:ZED 是立体深度相机,每帧给 720x1280 度量(mm)有组织点云(96%+有效,近场干净)。
VGGT 给可靠的相机位姿(走廊里 ICP 会滑动歧义,所以位姿仍用 VGGT)。二者结合:
  VGGT 位姿(准) + ZED 深度(度量/稠密) → 干净度量几何, 作 3DGS 初始化 / 物理几何 / 度量尺度。

坐标:ZED 点云在相机光学系(OpenCV: +z前 +x右 +y下),与 VGGT 相机系一致。
尺度:VGGT 是归一化单位。复用 align 的"重力+地面+相机高1.2m"估 k(归一化×k=米),
      则 ZED(米) → 归一化 = /k。再用 VGGT cam→world 摆进世界帧。

用法: PYTHONPATH=third_party/gs_playground/.venv/lib/python3.10/site-packages \
      ~/discoverse_venv/bin/python scripts/fuse_zed.py <session> <raw_scene> [--max_pts 3000000]
  session   = vggto_<session> 目录名(如 114830c / 113628c)
  raw_scene = data/<raw_scene>/pointcloud/zed 所在(如 114830 / 113628)
产出: outputs/vggto_<session>/zed_fused.ply (VGGT 归一化帧, 与 recon.ply 同坐标)
"""
import os, sys, glob, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def decode_zed_npz(path):
    """返回 (xyz_m (K,3), rgb (K,3) uint8 真彩)。
    颜色不从坏掉的float16 rgba通道解,而从同名RGB图按像素对应取(点云有组织720x1280)。"""
    import cv2
    d = np.load(path)["xyzrgba"]                         # (720,1280,4) float16, ch0-2=xyz(mm)
    xyz = d[..., :3].astype(np.float32) / 1000.0         # (H,W,3) m
    H, W = xyz.shape[:2]
    rgb_path = path.replace("/pointcloud/", "/images/").replace(".npz", ".jpg")
    img = cv2.imread(rgb_path)
    if img is None:
        rgb_full = np.full((H, W, 3), 128, np.uint8)     # 退化灰
    else:
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        rgb_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fin = np.isfinite(xyz).all(2) & (xyz[..., 2] > 0.3) & (xyz[..., 2] < 12.0)
    return xyz[fin], rgb_full[fin]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("raw_scene")
    ap.add_argument("--cam_height", type=float, default=1.2)
    ap.add_argument("--voxel", type=float, default=0.0)   # 归一化单位下的体素降采样(0=按max_pts)
    ap.add_argument("--max_pts", type=int, default=3000000)
    args = ap.parse_args()
    import trimesh, open3d as o3d
    import importlib.util
    spec = importlib.util.spec_from_file_location("bev", os.path.join(ROOT, "scripts", "bev_topdown.py"))
    bev = importlib.util.module_from_spec(spec); spec.loader.exec_module(bev)

    vdir = os.path.join(ROOT, "outputs", f"vggto_{args.session}")
    cams = np.load(os.path.join(vdir, "cameras.npz"))
    extr = cams["extrinsic"]                              # (N,3,4) world2cam, 归一化
    N = len(extr)

    # 相机 i -> 时间戳: frames_zed 排序后, 若数==N 则1:1, 否则按 run_vggto 的 linspace 映射
    fz = sorted(glob.glob(os.path.join(vdir, "frames_zed", "*")))
    M = len(fz)
    if M == N:
        idx = np.arange(N)
    else:
        idx = np.linspace(0, M - 1, N).astype(int)
    ts = [os.path.basename(fz[i]).split("_")[-1].split(".")[0] for i in idx]

    # k: 归一化×k=米 (复用 align/bev 的重力+地面+相机高)
    R = extr[:, :3, :3]; t = extr[:, :3, 3]
    cam_c = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)    # 归一化 cam 中心
    pts_v = np.asarray(trimesh.load(os.path.join(vdir, "recon.ply"), process=False).vertices, dtype=np.float64)
    g = bev.estimate_gravity_from_cameras(extr)
    n_floor, _, _ = bev.ransac_ground_normal(pts_v, g)
    Rz = bev.rotation_aligning(n_floor, np.array([0, 0, 1.0]))
    rz_cam = cam_c @ Rz.T
    rz_pz = (pts_v @ Rz.T)[:, 2]
    ground_z = float(np.percentile(rz_pz, 2.0))
    cam_h_norm = float(np.median(rz_cam[:, 2]) - ground_z)
    k = args.cam_height / max(cam_h_norm, 1e-6)           # 归一化×k=米
    inv_k = 1.0 / k                                       # 米 -> 归一化
    print(f"[fuse] N={N} k(norm->m)={k:.3f}  ZED米->归一化×{inv_k:.4f}", flush=True)

    pcz = os.path.join(ROOT, "data", args.raw_scene, "pointcloud", "zed")
    REFINE = os.environ.get("REFINE", "0") == "1"
    rdist = float(os.environ.get("REFINE_DIST", "0.08")) * inv_k     # 米->归一化(ICP搜索半径)
    per = int(os.environ.get("PER_FRAME", "20000"))

    def world_pts(i):
        f = os.path.join(pcz, ts[i] + ".npz")
        if not os.path.exists(f):
            return None, None
        xyz_m, rgb = decode_zed_npz(f)
        if len(xyz_m) == 0:
            return None, None
        if len(xyz_m) > per:
            sel = np.random.choice(len(xyz_m), per, replace=False)
            xyz_m, rgb = xyz_m[sel], rgb[sel]
        Ri = extr[i, :3, :3]; ci = -Ri.T @ extr[i, :3, 3]
        return ((Ri.T @ (xyz_m * inv_k).T).T + ci).astype(np.float32), rgb

    if REFINE:
        # 帧到模型 ICP 精化: VGGT位姿当初值, 小半径只局部贴紧(不会走廊滑动崩溃)
        acc = o3d.geometry.PointCloud(); used = 0; n_ref = 0
        for i in range(N):
            wp_i, rgb = world_pts(i)
            if wp_i is None:
                continue
            cur = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(wp_i.astype(np.float64)))
            cur.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
            cur = cur.voxel_down_sample(rdist * 0.5)
            if len(acc.points) > 200:
                icp = o3d.pipelines.registration.registration_icp(
                    cur, acc, rdist, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25))
                if icp.fitness > 0.3:                      # 只接受可靠配准, 否则保持VGGT位姿
                    cur.transform(icp.transformation); n_ref += 1
            acc += cur
            acc = acc.voxel_down_sample(rdist * 0.45)
            acc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rdist * 3, max_nn=30))
            used += 1
            if i % 10 == 0:
                print(f"  [refine] {i}/{N} acc={len(acc.points)}", flush=True)
        P = np.asarray(acc.points, np.float32)
        C = (np.asarray(acc.colors) * 255).astype(np.float32)
        print(f"[fuse] REFINE 融合{used}帧 ICP精化{n_ref}帧 -> {len(P)}点", flush=True)
    else:
        allp, allc = [], []; used = 0
        for i in range(N):
            wp_i, rgb = world_pts(i)
            if wp_i is None:
                continue
            allp.append(wp_i); allc.append(rgb); used += 1
        P = np.concatenate(allp); C = np.concatenate(allc)
        print(f"[fuse] 融合 {used}/{N} 帧, {len(P)} 点", flush=True)

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(C.astype(np.float64) / 255.0)
    # 体素降采样(默认按目标点数推体素)
    if args.voxel <= 0:
        ext = (P.max(0) - P.min(0)); vol = float(np.prod(ext))
        voxel = (vol / max(args.max_pts, 1)) ** (1 / 3) if vol > 0 else 0.005
        voxel = min(voxel, float(os.environ.get("VOXEL_CAP", "0.006")))   # 别过度降采样
    else:
        voxel = args.voxel
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.5)
    out = os.path.join(vdir, "zed_fused.ply")
    o3d.io.write_point_cloud(out, pc)
    print(f"[fuse] voxel={voxel:.4f} -> {len(pc.points)} 点 -> {out}", flush=True)


if __name__ == "__main__":
    main()
