#!/usr/bin/env python3
"""把重力对齐后的点云从多个角度渲染出来,直观看 VGGTΩ 重建质量。
出一张 2x2 多视角图: 斜45° / 正前 / 侧面 / 俯视,叠相机轨迹。

用法: python3 scripts/render_views.py --target_dir outputs/vggto_114830 [--n 150000]
"""
import os, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--n", type=int, default=150000, help="降采样点数(matplotlib 3D 太多会很慢)")
    args = ap.parse_args()
    td = args.target_dir

    import trimesh
    pc = trimesh.load(os.path.join(td, "recon_aligned.ply"))
    P = np.asarray(pc.vertices, dtype=np.float64)
    rgb = np.asarray(pc.colors, dtype=np.float64)[:, :3] / 255.0

    # 相机轨迹也转到对齐坐标系: 复用 bev 的对齐结果不方便,这里直接从 cameras.npz 重算一致的对齐
    cams = np.load(os.path.join(td, "cameras.npz"))
    extr = cams["extrinsic"]
    R = extr[:, :3, :3]
    ey = np.array([0.0, 1.0, 0.0])
    down = np.einsum("nij,j->ni", np.transpose(R, (0, 2, 1)), ey)
    down /= np.linalg.norm(down, axis=1, keepdims=True)
    g = down.mean(0); g /= np.linalg.norm(g)
    up = -g
    # 同 bev_topdown 的 rotation_aligning(up -> +Z)
    a = up / np.linalg.norm(up); b = np.array([0, 0, 1.0])
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-9:
        Rz = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rz = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    cam_c = -np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), extr[:, :3, 3])
    Cc = cam_c @ Rz.T

    rng = np.random.default_rng(0)
    if len(P) > args.n:
        idx = rng.choice(len(P), args.n, replace=False)
        P = P[idx]; rgb = rgb[idx]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = [("oblique 45°", 22, -60), ("front", 5, -90), ("side", 5, 0), ("top-down", 89, -90)]
    fig = plt.figure(figsize=(14, 12))
    for i, (name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.3, c=np.clip(rgb, 0, 1), depthshade=False)
        ax.plot(Cc[:, 0], Cc[:, 1], Cc[:, 2], "-", color="red", lw=1.5)
        ax.scatter(Cc[0, 0], Cc[0, 1], Cc[0, 2], c="lime", s=40, label="start")
        ax.scatter(Cc[-1, 0], Cc[-1, 1], Cc[-1, 2], c="magenta", s=40, label="end")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name} (elev={elev},azim={azim})")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z(up)")
        try:
            ax.set_box_aspect((np.ptp(P[:, 0]), np.ptp(P[:, 1]), np.ptp(P[:, 2])))
        except Exception:
            pass
        if i == 1:
            ax.legend(loc="upper right")
    fig.suptitle("VGGT-Omega 114830 reconstruction (gravity-aligned, red=camera path)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(td, "views_3d.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[o] {out}  ({len(P)} pts shown)")


if __name__ == "__main__":
    main()
