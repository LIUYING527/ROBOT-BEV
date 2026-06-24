"""把稠密 ZED 世界帧点云(zed_fused_world.ply, VGGT位姿融合)做成忠实细节网格(Poisson)。
= 满足 SLAM 精度的几何, 直接当碰撞/导航底座, 不再抽象成2墙。
用法: ~/discoverse_venv/bin/python scripts/dense_map.py <session> [--poisson_depth 10]
产出: outputs/dense_mesh_<s>.ply(网格) + _dense_mesh_<s>.png(俯视/侧视验精度)
"""
import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--poisson_depth", type=int, default=10)
    ap.add_argument("--voxel", type=float, default=0.02)   # 米
    args = ap.parse_args()
    s = args.session
    pc = o3d.io.read_point_cloud(os.path.join(ROOT, "outputs", f"vggto_{s}", "zed_fused_world.ply"))
    print(f"[mesh] 输入稠密云 {len(pc.points)} 点")
    pc = pc.voxel_down_sample(args.voxel)
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 4, max_nn=30))
    pc.orient_normals_consistent_tangent_plane(20)
    print(f"[mesh] 清理后 {len(pc.points)} 点, Poisson 重建(depth={args.poisson_depth})...")
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=args.poisson_depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.06))   # 剪低密度(外推噪声)
    mesh.compute_vertex_normals()
    out = os.path.join(ROOT, "outputs", f"dense_mesh_{s}.ply")
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"[mesh] -> {out}  顶点{len(mesh.vertices)} 面{len(mesh.triangles)}")

    # 渲染验精度: 点云俯视(带色) + 网格俯视 + 侧视
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P = np.asarray(pc.points); C = np.asarray(pc.colors)
    V = np.asarray(mesh.vertices)
    fig, axs = plt.subplots(1, 3, figsize=(18, 8))
    m = (P[:, 2] > 0.2) & (P[:, 2] < 2.2)
    axs[0].scatter(P[m, 0], P[m, 1], s=0.3, c=C[m], lw=0); axs[0].set_aspect("equal")
    axs[0].set_title(f"稠密点云俯视 {m.sum()}点")
    vm = (V[:, 2] > 0.2) & (V[:, 2] < 2.2)
    axs[1].scatter(V[vm, 0], V[vm, 1], s=0.3, c="0.3", lw=0); axs[1].set_aspect("equal")
    axs[1].set_title(f"网格顶点俯视 {len(V)}点")
    axs[2].scatter(P[:, 0], P[:, 2], s=0.3, c=C, lw=0); axs[2].set_aspect("equal")
    axs[2].set_title("点云侧视(墙/地是否成锐面)")
    fig.savefig(os.path.join(ROOT, "outputs", f"_dense_mesh_{s}.png"), dpi=90, bbox_inches="tight")
    print(f"[mesh] saved _dense_mesh_{s}.png")


if __name__ == "__main__":
    main()
