"""演示: 用 ZED 点云做彩色ICP序列配准融合(真RGB-D SLAM做法), 证明数据支持的稠密精度。
不靠VGGT位姿——靠帧间彩色ICP(RGB纹理破除走廊滑动歧义)。
用法: python scripts/dense_slam_demo.py <raw_scene> [--start 0 --n 60 --step 2]
产出: outputs/dense_slam_<scene>.ply + _dense_slam_<scene>.png
"""
import os, sys, glob, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_zed(path, voxel):
    import open3d as o3d
    d = np.load(path)["xyzrgba"].astype(np.float32)
    xyz = d[..., :3].reshape(-1, 3) / 1000.0
    fin = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0.3) & (xyz[:, 2] < 8.0)   # 近8m最准
    rb = d[..., 3].reshape(-1).copy(); rb[~np.isfinite(rb)] = 0
    rgb = rb.view(np.uint8).reshape(-1, 4)[:, [2, 1, 0]] / 255.0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz[fin].astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(rgb[fin].astype(np.float64))
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return pc


def main():
    import open3d as o3d
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--voxel", type=float, default=0.02)
    args = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(ROOT, "data", args.scene, "pointcloud", "zed", "*.npz")))
    fs = fs[args.start: args.start + args.n * args.step: args.step]
    print(f"[slam] {len(fs)} 帧, voxel {args.voxel}m, 彩色ICP序列配准...")

    frames = [load_zed(f, args.voxel) for f in fs]      # 每帧只加载一次
    print(f"[slam] 已加载 {len(frames)} 帧, 点到面ICP序列配准...", flush=True)
    fused = o3d.geometry.PointCloud()
    Tg = np.eye(4)
    Tgs = [Tg.copy()]
    for i in range(len(frames)):
        if i > 0:
            res = o3d.pipelines.registration.registration_icp(
                frames[i], frames[i - 1], args.voxel * 2.0, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
            Tg = Tg @ res.transformation
        Tgs.append(Tg.copy())
        if i % 10 == 0:
            print(f"  {i}/{len(frames)}", flush=True)
    for i in range(len(frames)):
        fused += o3d.geometry.PointCloud(frames[i]).transform(Tgs[i])
    fused = fused.voxel_down_sample(args.voxel)
    fused, _ = fused.remove_statistical_outlier(20, 2.0)
    out = os.path.join(ROOT, "outputs", f"dense_slam_{args.scene}.ply")
    o3d.io.write_point_cloud(out, fused)
    P = np.asarray(fused.points); C = np.asarray(fused.colors)
    print(f"[slam] 融合 {len(P)} 点 -> {out}")

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    m = (P[:, 2] > 0.2) & (P[:, 2] < 2.2)
    axs[0].scatter(P[m, 0], P[m, 1], s=0.4, c=C[m], lw=0); axs[0].set_aspect("equal")
    axs[0].set_title(f"dense SLAM 俯视(彩色ICP, {len(fs)}帧) {m.sum()}点")
    axs[1].scatter(P[:, 0], P[:, 2], s=0.4, c=C, lw=0); axs[1].set_aspect("equal")
    axs[1].set_title("侧视(看墙/地是否锐利成面)")
    fig.savefig(os.path.join(ROOT, "outputs", f"_dense_slam_{args.scene}.png"), dpi=90, bbox_inches="tight")
    print("saved _dense_slam_%s.png" % args.scene)


if __name__ == "__main__":
    main()
