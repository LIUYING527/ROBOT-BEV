"""把 ArtiFixer3D 皮肤(nerfstudio归一化系)配准到 sim world 系, 并把高斯搬过去存新 ply。
两片同一走廊稠密点云 -> open3d 去floater+降采样+尺度粗归一+FGR全局+带尺度ICP -> 单个相似变换 M(4x4)。
验证范围贴合后, 分解 M=(s,R,t), 用 gaussian_renderer.transform_gaussian 搬 A3D 高斯 -> world 系 ply。
输出: outputs/gs_a3d_colmapjoint_all_world.ply + outputs/_a3d_align.npz
"""
import os, sys, numpy as np, trimesh, open3d as o3d
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
A_PLY = os.path.join(ROOT, "outputs", "artifixer3d_skin_colmapframe.ply")
W_PLY = os.path.join(ROOT, "outputs", "gs_vggto_colmapjoint_all_world.ply")


def pcd_from_xyz(v, vox):
    lo, hi = np.percentile(v, [1, 99], axis=0)                 # 便宜地裁 floater
    v = v[((v >= lo) & (v <= hi)).all(1)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))
    pc = pc.voxel_down_sample(vox)                             # 大减点
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    return pc


def pca_axes(x):
    xc = x - x.mean(0); ev, evec = np.linalg.eigh(xc.T @ xc / len(xc))
    return evec[:, ::-1]                  # 列=主轴, 降序(第0列=最长=走廊方向)


def main():
    print("[load] open3d 读 xyz(快, 忽略SH) ...", flush=True)
    vA = np.asarray(o3d.io.read_point_cloud(A_PLY).points).astype(np.float64)
    vW = np.asarray(o3d.io.read_point_cloud(W_PLY).points).astype(np.float64)
    print(f"[load] A3D {len(vA)} pts, world {len(vW)} pts", flush=True)
    A = pcd_from_xyz(vA, 0.03); W = pcd_from_xyz(vW, 0.08)
    a = np.asarray(A.points); w = np.asarray(W.points)
    axl = lambda x: float(np.sqrt(np.linalg.eigvalsh((x - x.mean(0)).T @ (x - x.mean(0)) / len(x))[-1]))
    Ac, Wc = a.mean(0), w.mean(0)
    s0 = axl(w) / axl(a)
    print(f"[coarse] s0={s0:.3f}", flush=True)
    Aax, Wax = pca_axes(a), pca_axes(w)                        # 主轴(走廊方向=第0列)

    def compose(Rc):                                          # x_world = Rc@(s0*(x-Ac)) + Wc
        M = np.eye(4); M[:3, :3] = s0 * Rc; M[:3, 3] = Wc - s0 * Rc @ Ac; return M

    best = None                                               # 翻转搜索(主轴符号歧义), 选ICP最优
    for f0 in (1, -1):
        for f1 in (1, -1):
            F = np.diag([f0, f1, f0 * f1]).astype(float)      # det=+1 保右手系
            Rc = Wax @ F @ Aax.T
            Mc = compose(Rc)
            At = o3d.geometry.PointCloud(A); At.transform(Mc)
            icp = o3d.pipelines.registration.registration_icp(
                At, W, 0.6, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            print(f"  flip({f0:+d},{f1:+d}) icp_fit={icp.fitness:.3f} rmse={icp.inlier_rmse:.3f}", flush=True)
            if best is None or icp.fitness > best[0]:
                best = (icp.fitness, icp.inlier_rmse, icp.transformation @ Mc)
    fitness, rmse, M = best
    print(f"[best] fitness={fitness:.3f} rmse={rmse:.3f}", flush=True)
    L = M[:3, :3]; s = float(np.cbrt(np.linalg.det(L))); R = L / s; t = M[:3, 3]
    a0 = np.asarray(pcd_from_xyz(vA, 0.05).points)
    aw = (a0 @ L.T) + t
    print(f"[check] A3D->world 范围 x[{aw[:,0].min():.1f},{aw[:,0].max():.1f}] y[{aw[:,1].min():.1f},{aw[:,1].max():.1f}] z[{aw[:,2].min():.1f},{aw[:,2].max():.1f}]")
    print(f"[check] world 目标   范围 x[{w[:,0].min():.1f},{w[:,0].max():.1f}] y[{w[:,1].min():.1f},{w[:,1].max():.1f}] z[{w[:,2].min():.1f},{w[:,2].max():.1f}]")
    print(f"[fit] s={s:.3f} fitness={fitness:.3f} rmse={rmse:.3f}")
    np.savez(os.path.join(ROOT, "outputs", "_a3d_align.npz"), M=M, s=s, R=R, t=t,
             fitness=fitness, rmse=rmse)

    # 搬高斯 -> world ply(复用 align_gs_world 同款 API)
    from gaussian_renderer.transform_gs_model import transform_gaussian
    from gaussian_renderer.core.util_gau import load_ply, save_ply
    gd = load_ply(A_PLY)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    gd = transform_gaussian(gd, T, scale_factor=s, rescale_first=True, silent=False)
    out = os.path.join(ROOT, "outputs", "gs_a3d_colmapjoint_all_world.ply")
    save_ply(gd, out)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
