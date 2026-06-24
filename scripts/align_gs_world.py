"""把 VGGT 训练出的 3DGS(归一化帧)烘焙成"仿真器世界帧"的 ply:
  重力对齐(up→+z) + 米制缩放(用相机离地高度~1.2m 锚定) + 地面落到 z=0 + 起点相机移到 xy 原点。
同时导出机器人轨迹 waypoints(world xy + yaw)供 Part B 仿真器用。

复用:bev_topdown 的重力对齐三件套、gaussian_renderer.transform_gs_model.transform_gaussian。

用法: ~/discoverse_venv/bin/python scripts/align_gs_world.py [session] [--cam_height 1.2]
产出: outputs/gs_vggto_<s>_world.ply, outputs/sim_world_<s>.npz
"""
import os, sys, argparse, importlib.util
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_bev():
    spec = importlib.util.spec_from_file_location("bev", os.path.join(ROOT, "scripts", "bev_topdown.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", default="114830")
    ap.add_argument("--cam_height", type=float, default=1.2, help="相机离地高度(米),用于米制锚定")
    args = ap.parse_args()
    s = args.session
    vdir = os.path.join(ROOT, "outputs", f"vggto_{s}")
    ply_in = os.path.join(ROOT, "outputs", f"gs_vggto_{s}.ply")
    ply_out = os.path.join(ROOT, "outputs", f"gs_vggto_{s}_world.ply")

    bev = _load_bev()
    import trimesh
    from gaussian_renderer.transform_gs_model import transform_gaussian
    from gaussian_renderer.core.util_gau import load_ply, save_ply

    # 1) 重力对齐旋转 Rz(在 VGGT 原始帧算,旋转对方向有效,归一化帧同样适用)
    pts = np.asarray(trimesh.load(os.path.join(vdir, "recon.ply"), process=False).vertices, dtype=np.float64)
    cams = np.load(os.path.join(vdir, "cameras.npz")); extr = cams["extrinsic"]
    g = bev.estimate_gravity_from_cameras(extr)
    n_floor, inl, nps = bev.ransac_ground_normal(pts, g)
    Rz = bev.rotation_aligning(n_floor, np.array([0, 0, 1.0]))
    print(f"[align] up={n_floor.round(3)} inliers={inl}/{nps}")

    # 2) 归一化参数(gs 的坐标 = (orig - center)/s_norm)
    cs = np.load(ply_in + ".norm.npy"); center, s_norm = cs[:3].astype(np.float64), float(cs[3])

    # 相机中心(VGGT帧)→ 归一化帧 → Rz 旋转
    R = extr[:, :3, :3]; t = extr[:, :3, 3]
    cam_c = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), t)   # VGGT world
    cam_n = (cam_c - center) / s_norm                                 # normalized
    rz_cam = cam_n @ Rz.T                                             # gravity-aligned

    # gs means(归一化帧)→ Rz,用于求地面高度
    gp = np.asarray(trimesh.load(ply_in, process=False).vertices, dtype=np.float64)
    rz_gp_z = (gp @ Rz.T)[:, 2]
    ground_z = float(np.percentile(rz_gp_z, 2.0))                      # 地面(低分位)
    cam_h_norm = float(np.median(rz_cam[:, 2]) - ground_z)             # 相机离地(归一化单位)
    k = args.cam_height / max(cam_h_norm, 1e-6)                        # 米制缩放
    print(f"[scale] ground_z={ground_z:.3f} cam_h_norm={cam_h_norm:.3f} -> k={k:.3f} (cam_height={args.cam_height}m)")

    # 3) 平移:地面到 z=0,起点相机到 xy 原点
    tz = -k * ground_z
    txy = -k * rz_cam[0, :2]
    t_world = np.array([txy[0], txy[1], tz], dtype=np.float64)
    T = np.eye(4); T[:3, :3] = Rz; T[:3, 3] = t_world

    # 4) 烘焙进新 ply(transform_gaussian: 先缩放 means+scale,再 R@x + t)
    gd = load_ply(ply_in)
    gd = transform_gaussian(gd, T, scale_factor=float(k), rescale_first=True, silent=False)
    save_ply(gd, ply_out)
    print(f"[ok] world ply -> {ply_out}")

    # 5) 机器人轨迹 waypoints(world)= k*rz_cam + t
    wp = (rz_cam * k) + t_world
    # yaw 用**相机实际朝向**(光轴+z前向, 投到地面), 而非相邻waypoint切向——
    # 近直线/抖动轨迹下切向会乱转, 而相机朝向=3DGS被观测方向, 渲染才清晰。
    fwd_world = np.einsum("nij,j->ni", np.transpose(R, (0, 2, 1)), np.array([0, 0, 1.0]))  # cam +z 在VGGT世界
    rz_fwd = fwd_world @ Rz.T                                  # 重力对齐
    yaw = np.arctan2(rz_fwd[:, 1], rz_fwd[:, 0])
    # 解卷绕 + 轻微平滑(去抖, 不改大趋势)
    yaw = np.unwrap(yaw)
    kk = 2
    yaw = np.convolve(np.pad(yaw, kk, mode="edge"), np.ones(2 * kk + 1) / (2 * kk + 1), "valid")
    np.savez(os.path.join(ROOT, "outputs", f"sim_world_{s}.npz"),
             waypoints_xy=wp[:, :2], waypoints_z=wp[:, 2], yaw=yaw,
             cam_height=args.cam_height, k=k, Rz=Rz, t_world=t_world,
             ground_z=ground_z, center=center, s_norm=s_norm)
    print(f"[ok] trajectory -> outputs/sim_world_{s}.npz  ({len(wp)} waypoints, "
          f"span {np.ptp(wp[:,0]):.1f}x{np.ptp(wp[:,1]):.1f}m)")


if __name__ == "__main__":
    main()
