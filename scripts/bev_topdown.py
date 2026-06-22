#!/usr/bin/env python3
"""把 VGGTΩ 输出的点云做重力对齐后出干净俯视图(BEV)。

为什么需要:VGGTΩ 世界系以第一帧相机为原点,Y 不是真正的"上",且相机沿轨迹有倾斜。
直接拿 X-Z 当俯视会把地面/天花板抹成斜带。这里:
  1) 用相机姿态(相机+Y≈向下)给重力先验
  2) RANSAC 在近水平点里拟合地面,精修重力法向
  3) 旋转点云使重力竖直 -> 真正的俯视投影
  4) 出 RGB俯视 / 高度着色 / 去地面天花板的占据图

用法: python3 scripts/bev_topdown.py --target_dir outputs/vggto_114830
"""
import os, argparse
import numpy as np


def estimate_gravity_from_cameras(extr):
    """相机 +Y(OpenCV 向下) 在世界系下的平均方向 ~ 重力(向下)。"""
    R = extr[:, :3, :3]
    ey = np.array([0.0, 1.0, 0.0])
    down = np.einsum("nij,j->ni", np.transpose(R, (0, 2, 1)), ey)  # cam axis in world
    down /= np.linalg.norm(down, axis=1, keepdims=True)
    g = down.mean(0)
    return g / np.linalg.norm(g)


def ransac_ground_normal(pts, g_prior, n_iter=2000, ang_tol_deg=25.0, dist_frac=0.01, seed_pts=80000):
    """在与重力先验近平行(法向近 g)的平面里 RANSAC 拟合地面,返回精修法向。"""
    rng = np.random.default_rng(0)
    P = pts if len(pts) <= seed_pts else pts[rng.choice(len(pts), seed_pts, replace=False)]
    extent = np.linalg.norm(P.max(0) - P.min(0))
    thr = dist_frac * extent
    cos_tol = np.cos(np.deg2rad(ang_tol_deg))
    best_n, best_in = g_prior, -1
    for _ in range(n_iter):
        i = rng.choice(len(P), 3, replace=False)
        a, b, c = P[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if abs(n @ g_prior) < cos_tol:   # 平面法向必须接近重力(=水平面)
            continue
        d = np.abs((P - a) @ n)
        inl = int((d < thr).sum())
        if inl > best_in:
            best_in, best_n = inl, n
    # 统一符号使其指向"上"(与 -g_prior 同向)
    if best_n @ (-g_prior) < 0:
        best_n = -best_n
    return best_n, best_in, len(P)


def rotation_aligning(a, b):
    """返回把单位向量 a 旋到 b 的旋转矩阵(Rodrigues)。"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--floor_band", type=float, default=0.12,
                    help="占据图里从地面往上剔除的高度比例(相对总高)")
    ap.add_argument("--ceil_band", type=float, default=0.12,
                    help="占据图里从天花板往下剔除的高度比例")
    args = ap.parse_args()
    td = args.target_dir

    import trimesh
    pc = trimesh.load(os.path.join(td, "recon.ply"))
    pts = np.asarray(pc.vertices, dtype=np.float64)
    rgb = np.asarray(pc.colors, dtype=np.float64)[:, :3] / 255.0
    cams = np.load(os.path.join(td, "cameras.npz"))
    extr = cams["extrinsic"]

    g = estimate_gravity_from_cameras(extr)
    print(f"[i] 相机重力先验(down): {g.round(3)}")
    n_floor, inl, nps = ransac_ground_normal(pts, g)
    print(f"[i] RANSAC 地面法向(up): {n_floor.round(3)}  inliers={inl}/{nps}")

    up = n_floor                                  # 精修后的"上"
    Rz = rotation_aligning(up, np.array([0, 0, 1.0]))  # 把 up 旋到 +Z
    P = pts @ Rz.T
    cam_centers = -np.einsum("nij,nj->ni", np.transpose(extr[:, :3, :3], (0, 2, 1)), extr[:, :3, 3])
    Cc = cam_centers @ Rz.T

    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    z0, z1 = np.quantile(z, 0.01), np.quantile(z, 0.99)
    H = z1 - z0
    print(f"[i] 对齐后高度跨度 ~{H:.2f}(单位=VGGT尺度)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def setup(ax, title):
        ax.set_aspect("equal"); ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    # (1) RGB 俯视
    fig, ax = plt.subplots(figsize=(8, 8))
    order = np.argsort(z)                 # 高的点后画(盖住),近似遮挡
    ax.scatter(x[order], y[order], s=0.2, c=np.clip(rgb[order], 0, 1))
    ax.plot(Cc[:, 0], Cc[:, 1], "-", color="red", lw=1.0, label="camera path")
    setup(ax, "BEV RGB (gravity-aligned top-down)"); ax.legend(loc="upper right")
    f1 = os.path.join(td, "bev_topdown_rgb.png"); fig.savefig(f1, dpi=130); plt.close(fig)
    print(f"[o] {f1}")

    # (2) 高度着色
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(x, y, s=0.2, c=np.clip((z - z0) / max(H, 1e-6), 0, 1), cmap="turbo")
    ax.plot(Cc[:, 0], Cc[:, 1], "-", color="white", lw=1.0)
    setup(ax, "BEV height (turbo, low->high)"); fig.colorbar(sc, ax=ax, shrink=0.7)
    f2 = os.path.join(td, "bev_topdown_height.png"); fig.savefig(f2, dpi=130); plt.close(fig)
    print(f"[o] {f2}")

    # (3) 占据图:剔除地面带 + 天花板带,只留墙/物体,做 2D 直方图
    lo = z0 + args.floor_band * H
    hi = z1 - args.ceil_band * H
    m = (z > lo) & (z < hi)
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hist2d(x[m], y[m], bins=350, cmap="Greys", cmin=1, norm=LogNorm())
    ax.plot(Cc[:, 0], Cc[:, 1], "-", color="red", lw=1.2, label="camera path")
    setup(ax, f"BEV occupancy (walls/objects, floor+ceil removed)  {m.sum()} pts")
    ax.legend(loc="upper right")
    f3 = os.path.join(td, "bev_occupancy.png"); fig.savefig(f3, dpi=130); plt.close(fig)
    print(f"[o] {f3}")

    # 存对齐后的点云,后续给仿真器/拟合墙用
    out_ply = os.path.join(td, "recon_aligned.ply")
    trimesh.PointCloud(vertices=P, colors=(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).export(out_ply)
    print(f"[o] {out_ply}")


if __name__ == "__main__":
    main()
