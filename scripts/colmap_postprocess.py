"""读 COLMAP 已存模型 → 重力对齐 + 尺度标定 → 轨迹/点云/俯视BEV(不重跑SfM)。

用法: python scripts/colmap_postprocess.py 111450 [model_idx]
"""
import os
import sys
import glob
import struct
import sqlite3

import numpy as np
import pycolmap
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False


def real_path_length(session):
    dbs = sorted(glob.glob(f"data/{session}/rosbag*/*.db3"))
    T, V = [], []
    for db in dbs:
        con = sqlite3.connect(db); cur = con.cursor()
        cur.execute("SELECT m.timestamp,m.data FROM messages m JOIN topics t "
                    "ON m.topic_id=t.id WHERE t.name='/svtrobot_cmd'")
        for ts, data in cur.fetchall():
            T.append(ts * 1e-9); V.append(struct.unpack_from("<6d", data, 4)[0])
        con.close()
    T = np.array(T); V = np.array(V); o = np.argsort(T); T, V = T[o], V[o]
    dt = np.clip(np.diff(T, prepend=T[0]), 0, 0.5)
    return float(np.sum(np.abs(V) * dt))


def align_gravity(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if len(pts) > 80000:
        pcd = pcd.voxel_down_sample(0.3)
    plane, inl = pcd.segment_plane(0.2, 3, 500)
    n = np.array(plane[:3]); n = n / np.linalg.norm(n)
    b = np.array([0, 0, 1.0])
    v = np.cross(n, b); s = np.linalg.norm(v); c = float(np.dot(n, b))
    if s < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "111450"
    midx = sys.argv[2] if len(sys.argv) > 2 else "1"
    rec = pycolmap.Reconstruction(f"outputs/_colmap_{session}/sparse/{midx}")
    print(f"[INFO] 模型{midx}: 注册图 {rec.num_reg_images()}  三维点 {rec.num_points3D()}")

    name2c = {}
    for img in rec.images.values():
        name2c[img.name] = np.array(img.projection_center())
    names = sorted(name2c.keys())
    C = np.array([name2c[n] for n in names])
    P = np.array([p.xyz for p in rec.points3D.values()])

    # --- 稳健去离群(少数相机/点被错误三角化到极远处)---
    med = np.median(P, axis=0)
    dP = np.linalg.norm(P - med, axis=1)
    rad = 5 * np.median(dP)                        # 真实场景尺度的稳健半径
    P = P[dP < rad]
    dC = np.linalg.norm(C - med, axis=1)
    cmask = dC < rad
    C = C[cmask]
    print(f"[INFO] 去离群后: 相机 {cmask.sum()}/{len(cmask)}  点 {len(P)}")

    # --- 尺度:用相邻步长中位数(机器人近匀速)---
    steps = np.linalg.norm(np.diff(C, axis=0), axis=1)
    med_step = np.median(steps[steps > 1e-6])
    real_len = real_path_length(session)
    real_step = real_len / (len(C) - 1)
    scale = real_step / med_step if med_step > 1e-9 else 1.0
    P *= scale; C *= scale
    print(f"[INFO] 尺度: 中位步长 {med_step:.3f} → {real_step:.3f}m (×{scale:.2f}); "
          f"净位移 {np.linalg.norm(C[-1]-C[0]):.1f}m")

    Rg = align_gravity(P)
    P = (Rg @ P.T).T; C = (Rg @ C.T).T

    z = P[:, 2]; zlo, zhi = np.percentile(z, [2, 98])
    keep = (z > zlo) & (z < zhi)
    Pf = P[keep]
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(Pf)
    os.makedirs("outputs", exist_ok=True)
    o3d.io.write_point_cloud(f"outputs/colmap_{session}.ply", pc)
    np.savez(f"outputs/traj_{session}_colmap.npz", x=C[:, 0], y=C[:, 1], z=C[:, 2])
    print(f"[OK] outputs/colmap_{session}.ply ({len(Pf)} 点) + 轨迹")

    fig, ax = plt.subplots(figsize=(12, 9))
    sc = ax.scatter(Pf[:, 0], Pf[:, 1], s=0.5, c=Pf[:, 2], cmap="turbo", alpha=0.55)
    ax.plot(C[:, 0], C[:, 1], "-", color="red", lw=2, label="相机轨迹")
    ax.plot(C[0, 0], C[0, 1], "o", color="lime", ms=13, mec="k", label="起点")
    ax.plot(C[-1, 0], C[-1, 1], "s", color="k", ms=10, label="终点")
    ax.set_aspect("equal"); ax.legend(); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{session} COLMAP 全局SfM重建 俯视BEV "
                 f"(注册{rec.num_reg_images()}张/{rec.num_points3D()}点, 色=高度)")
    plt.colorbar(sc, ax=ax, label="高度(m)")
    fig.tight_layout(); fig.savefig(f"outputs/colmap_{session}_bev.png", dpi=130)
    print(f"[OK] outputs/colmap_{session}_bev.png")


if __name__ == "__main__":
    main()


def clean_camera_mask(C, tol=2.5, iters=3):
    """剔除偏离主轨迹线的离群相机(COLMAP误注册)。C:Nx3相机中心(按时间序)。
    轨迹近直线→PCA主轴拟合线,保留垂直残差<tol(米)的。返回bool掩码。"""
    import numpy as np
    xy = C[:, :2].astype(float)
    m = np.ones(len(xy), bool)
    for _ in range(iters):
        p = xy[m]
        ctr = p.mean(0)
        _, _, vt = np.linalg.svd(p - ctr)
        d = vt[0]; n = np.array([-d[1], d[0]])
        resid = np.abs((xy - ctr) @ n)
        m = resid < tol
    return m
