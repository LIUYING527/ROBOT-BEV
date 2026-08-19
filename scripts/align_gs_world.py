"""把 VGGT 训练出的 3DGS(归一化帧)烘焙成"仿真器世界帧"的 ply:
  重力对齐(up→+z) + 米制缩放(用相机离地高度~1.2m 锚定) + 地面落到 z=0 + 起点相机移到 xy 原点。
同时导出机器人轨迹 waypoints(world xy + yaw)供 Part B 仿真器用。

复用:bev_topdown 的重力对齐三件套、gaussian_renderer.transform_gs_model.transform_gaussian。

用法: ~/discoverse_venv/bin/python scripts/align_gs_world.py [session] [--cam_height 1.2]
产出: outputs/gs_vggto_<s>_world.ply, outputs/sim_world_<s>.npz

⭐室外长距离单程场景(如111450)三件套(默认关, 向后兼容):
  --crop      : 按轨迹bbox裁掉离群floater高斯(室外天空/远景炸点, 治"乱")
  --untwist   : 治SfM roll漂移(地面法线沿路径渐变, 末端最严重)→逐段测地面法线局部旋正(平滑过渡), 整条水平
  --relevel   : 单次全局纯旋转再找平(untwist的全局版, 一般被untwist取代)
  ⚠改坐标 → 必重建 device_clusters_<s>/nav_ds_<s>(都从sim_world_<s>.npz读)
  ⚠别用 --robust_ground(近轨迹点含墙/杆反而拟合更差+破坏米制尺度)
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
    # ⭐室外场景抗floater/找平(默认关, 向后兼容其它session):
    ap.add_argument("--robust_ground", action="store_true",
                    help="只用贴近相机轨迹的点拟合地面法线/ground_z(剔除远景floater污染→治残余倾斜)")
    ap.add_argument("--near_keep", type=float, default=0.4,
                    help="robust_ground: 按到最近相机距离保留最近的这个比例的点(0.4=最近40%)")
    ap.add_argument("--crop", action="store_true",
                    help="烘焙进世界帧后按轨迹bbox裁掉离群floater高斯")
    ap.add_argument("--crop_margin", type=float, default=3.0, help="crop: 轨迹x/y向外扩这么多米")
    ap.add_argument("--crop_ywidth", type=float, default=10.0, help="crop: 轨迹y范围两侧各留这么多米")
    ap.add_argument("--crop_zlo", type=float, default=-1.0, help="crop: 保留z下界(米, 地面≈0)")
    ap.add_argument("--crop_zhi", type=float, default=6.0, help="crop: 保留z上界(米)")
    ap.add_argument("--relevel", action="store_true",
                    help="裁剪后在世界帧用干净地面点做一次纯旋转再找平(治残余倾斜, 不改尺度), 同步旋转waypoints")
    ap.add_argument("--untwist", action="store_true",
                    help="⭐治SfM roll漂移(沿路径地面法线渐变): 逐段测地面法线→局部旋正(平滑过渡), 整条水平")
    ap.add_argument("--untwist_win", type=float, default=4.0, help="untwist: 每段沿路径半窗(米)")
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

    # 相机中心(VGGT world), 供 robust_ground 近轨迹过滤 + 后面米制化复用
    R_all = extr[:, :3, :3]; t_all = extr[:, :3, 3]
    cam_c_vggt = -np.einsum("nij,nj->ni", np.transpose(R_all, (0, 2, 1)), t_all)   # (M,3) VGGT world

    pts_gnd = pts
    if args.robust_ground:
        # 只保留【到最近相机距离】最小的 near_keep 比例的点→剔除远景/floater对地面拟合的污染
        from scipy.spatial import cKDTree
        dmin, _ = cKDTree(cam_c_vggt).query(pts)
        keep = dmin <= np.percentile(dmin, 100.0 * args.near_keep)
        pts_gnd = pts[keep]
        print(f"[align] robust_ground: 近轨迹点 {keep.sum()}/{len(pts)} (最近{args.near_keep*100:.0f}%) 用于地面拟合")

    n_floor, inl, nps = bev.ransac_ground_normal(pts_gnd, g)
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
    if args.robust_ground:
        # ground_z 也只用近轨迹高斯(否则场景下方floater压低ground_z→米制/落地偏)
        from scipy.spatial import cKDTree
        dmin, _ = cKDTree(cam_n).query(gp)
        gp = gp[dmin <= np.percentile(dmin, 100.0 * args.near_keep)]
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

    # 机器人轨迹 waypoints(world)= k*rz_cam + t  (提前算, crop 的 bbox 要用)
    wp = (rz_cam * k) + t_world

    # 4) 烘焙进新 ply(transform_gaussian: 先缩放 means+scale,再 R@x + t)
    gd = load_ply(ply_in)
    gd = transform_gaussian(gd, T, scale_factor=float(k), rescale_first=True, silent=False)
    if args.crop:
        xyz = np.asarray(gd.xyz, dtype=np.float64)
        M, YW = args.crop_margin, args.crop_ywidth
        x0, x1 = wp[:, 0].min() - M, wp[:, 0].max() + M
        y0, y1 = wp[:, 1].min() - YW, wp[:, 1].max() + YW
        keep = ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
                & (xyz[:, 2] >= args.crop_zlo) & (xyz[:, 2] <= args.crop_zhi))
        nb = len(keep)
        gd.xyz = gd.xyz[keep]; gd.rot = gd.rot[keep]; gd.scale = gd.scale[keep]
        gd.opacity = gd.opacity[keep]; gd.sh = gd.sh[keep]
        print(f"[crop] 保留 {int(keep.sum())}/{nb} 高斯 "
              f"bbox x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}] z[{args.crop_zlo},{args.crop_zhi}]")
        # crop 后复测地面倾斜(核对找平)
        z = xyz[keep][:, 2]; g_lo = xyz[keep][z < np.percentile(z, 20)]
        if len(g_lo) > 100:
            c = g_lo.mean(0); _, _, vt = np.linalg.svd(g_lo - c, full_matrices=False)
            n = vt[-1] / np.linalg.norm(vt[-1]); n = n if n[2] > 0 else -n
            print(f"[crop] 裁后地面法线 {n.round(3)} 倾斜 {np.degrees(np.arccos(min(1, n[2]))):.2f}°")
    # yaw 的相机前向(VGGT +z→重力对齐), relevel 时也要一起旋
    fwd_world = np.einsum("nij,j->ni", np.transpose(R, (0, 2, 1)), np.array([0, 0, 1.0]))  # cam +z 在VGGT世界
    rz_fwd = fwd_world @ Rz.T                                  # 重力对齐

    # 4.5) ⭐relevel: 裁剪后在世界帧用干净近地面点拟合真实地面法线→纯旋转再找平(不改尺度)
    if args.relevel:
        xyzw = np.asarray(gd.xyz, dtype=np.float64)
        # 近地面点=沿x/y分格取每格最低点(纯地面, 排除墙基/curb对法线的抬升偏置)
        lo = xyzw[xyzw[:, 2] < np.percentile(xyzw[:, 2], 20)]
        xi = np.round(lo[:, 0]).astype(int); yi = np.round(lo[:, 1]).astype(int)
        key = xi * 1000 + yi; order = np.lexsort((lo[:, 2], key))
        ls = lo[order]; ks = key[order]; first = np.concatenate([[True], ks[1:] != ks[:-1]])
        floor = ls[first]                                          # 每格最低点
        # 平面回归 z = a*x + b*y + c → 法线 [-a,-b,1]
        A = np.column_stack([floor[:, 0], floor[:, 1], np.ones(len(floor))])
        coef, *_ = np.linalg.lstsq(A, floor[:, 2], rcond=None)
        n_w = np.array([-coef[0], -coef[1], 1.0]); n_w /= np.linalg.norm(n_w)
        c = floor.mean(0)
        Rc = bev.rotation_aligning(n_w, np.array([0, 0, 1.0]))    # 把残余地面法线转到+z
        anchor = np.array([c[0], c[1], c[2]], dtype=np.float64)   # 绕地面质心转(该点不动)
        Tc = np.eye(4); Tc[:3, :3] = Rc; Tc[:3, 3] = anchor - Rc @ anchor
        gd = transform_gaussian(gd, Tc, scale_factor=1.0, rescale_first=False, silent=True)
        wp = (wp - anchor) @ Rc.T + anchor                        # waypoints 同步旋转
        rz_fwd = rz_fwd @ Rc.T                                    # 前向同步旋转→yaw正确
        # 再把地面落回 z=0(旋转后地面在 anchor.z 附近)
        gxyz = np.asarray(gd.xyz, dtype=np.float64)
        gz0 = float(np.percentile(gxyz[:, 2], 2.0))
        gd.xyz = (gxyz - np.array([0, 0, gz0])).astype(np.float32)
        wp[:, 2] -= gz0
        print(f"[relevel] 残余地面法线 {n_w.round(3)} -> 旋正; 地面落回z=0(降 {gz0:.3f}m)")

    # 4.6) ⭐untwist: 治SfM roll漂移(地面法线沿路径渐变) — 逐waypoint测局部地面法线→局部旋正, 平滑过渡
    if args.untwist:
        from scipy.spatial import cKDTree
        from scipy.spatial.transform import Rotation as Rsc
        pw = wp[:, :2].copy()                                      # waypoint xy(随迭代更新)
        N_ITER = 6                                                 # ⭐迭代: 每轮测残留拧→再修, 收敛到平
        for it in range(N_ITER):
            gxyz = np.asarray(gd.xyz, dtype=np.float64)
            # 路径切向(前向轴)= roll的旋转轴; 只修roll(左右拧), 不动pitch(前后坡, 逐段修会自打架不收敛)
            tang = np.gradient(pw, axis=0); tang = np.column_stack([tang, np.zeros(len(pw))])
            tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
            # ⭐干净路面点=逐0.5m格取最低点(排除curb/墙基/floater污染, 同relevel鲁棒做法)
            lowz = gxyz[gxyz[:, 2] < np.percentile(gxyz[:, 2], 30)]
            gi_ = np.round(lowz[:, 0] / 0.5).astype(np.int64); gj_ = np.round(lowz[:, 1] / 0.5).astype(np.int64)
            key = gi_ * 100000 + gj_; order = np.lexsort((lowz[:, 2], key)); ls = lowz[order]; ks = key[order]
            floor_pts = ls[np.concatenate([[True], ks[1:] != ks[:-1]])]   # 每格最低点=真路面
            ftree = cKDTree(floor_pts[:, :2])
            rotvec = np.full((len(pw), 3), np.nan)
            for i in range(len(pw)):
                # 自适应窗口: 稀疏(远端)自动扩大, 每段都拿到自己的路面(不从近端插值)
                for win in (args.untwist_win, args.untwist_win * 1.8, args.untwist_win * 3.0):
                    idx = ftree.query_ball_point(pw[i], win)
                    if len(idx) >= 12:
                        break
                if len(idx) < 12:
                    continue
                loc = floor_pts[idx]
                A = np.column_stack([loc[:, 0], loc[:, 1], np.ones(len(loc))])   # lstsq 平面→法线
                coef, *_ = np.linalg.lstsq(A, loc[:, 2], rcond=None)
                ni = np.array([-coef[0], -coef[1], 1.0]); ni /= np.linalg.norm(ni)
                # ⭐只取roll: 法线在(右-上)平面里相对竖直的倾角, 绕前向轴转-θ归零(pitch分量不碰)
                ti = tang[i]; right = np.array([ti[1], -ti[0], 0.0])
                theta = np.arctan2(float(ni @ right), float(ni[2]))
                rotvec[i] = -theta * ti
            valid = ~np.isnan(rotvec[:, 0])
            if valid.sum() < 2:
                break
            ii = np.arange(len(pw))
            for c3 in range(3):
                rotvec[:, c3] = np.interp(ii, ii[valid], rotvec[valid, c3])   # 补NaN端点
            kk = 4
            for c3 in range(3):                                   # 平滑防seam
                rotvec[:, c3] = np.convolve(np.pad(rotvec[:, c3], kk, mode="edge"),
                                            np.ones(2 * kk + 1) / (2 * kk + 1), "valid")
            maxroll = np.degrees(np.linalg.norm(rotvec, axis=1)).max()
            wtree = cKDTree(pw); _, gi = wtree.query(gxyz[:, :2])
            Rmats = Rsc.from_rotvec(rotvec).as_matrix(); Rg = Rmats[gi]
            ctr = np.column_stack([pw[gi], np.zeros(len(gi))])     # 绕各自最近waypoint地面点旋
            gd.xyz = (ctr + np.einsum("nij,nj->ni", Rg, gxyz - ctr)).astype(np.float32)
            q_wxyz = np.asarray(gd.rot, dtype=np.float64)          # quats 同步旋
            q_old = Rsc.from_quat(q_wxyz[:, [1, 2, 3, 0]])
            gd.rot = (Rsc.from_matrix(Rg) * q_old).as_quat()[:, [3, 0, 1, 2]].astype(np.float32)
            wc = np.column_stack([pw, np.zeros(len(pw))])         # waypoints 同步旋
            wp3 = np.column_stack([wp[:, 0], wp[:, 1], wp[:, 2] if wp.shape[1] > 2 else np.zeros(len(wp))])
            wp = wc + np.einsum("nij,nj->ni", Rmats, wp3 - wc); pw = wp[:, :2].copy()
            print(f"[untwist] iter{it+1}/{N_ITER}: {int(valid.sum())}/{len(pw)}段; 本轮最大旋正 {maxroll:.2f}°", flush=True)
            if maxroll < 0.4:
                break
        gz0 = float(np.percentile(np.asarray(gd.xyz)[:, 2], 2.0))   # 地面落回z=0
        gd.xyz = (np.asarray(gd.xyz, dtype=np.float64) - np.array([0, 0, gz0])).astype(np.float32)
        wp[:, 2] -= gz0
        print(f"[untwist] 完成; 地面落回z=0(降 {gz0:.3f}m)", flush=True)

    save_ply(gd, ply_out)
    print(f"[ok] world ply -> {ply_out}")

    # 5) yaw(相机朝向): 近直线/抖动轨迹下切向会乱转, 用相机实际朝向=3DGS被观测方向, 渲染才清晰。
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
