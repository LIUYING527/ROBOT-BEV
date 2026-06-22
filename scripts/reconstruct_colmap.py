"""COLMAP (pycolmap) 全局 SfM 重建 —— 现成工具,全局BA+特征回环,抗漂移。

手搓里程计/ICP 过不了漂移关。COLMAP 不做序列累加:把所有图像两两特征匹配,
放进全局光束法平差一起解相机位姿+三维点,天然抗漂移、靠特征自动闭环。
A100 GPU 跑 SIFT 很快。

流程:strided ZED RGB → extract_features(GPU) → match_exhaustive → incremental_mapping
      → 取最大模型的相机轨迹+三维点 → RANSAC地面对齐重力 → 用cmd路程标定尺度 → 导出。

用法: python scripts/reconstruct_colmap.py 111450 [img_stride]
输出: outputs/colmap_<session>.ply / _bev.png, traj_<session>_colmap.npz
"""
import os
import sys
import glob
import shutil
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
    """从 /svtrobot_cmd 积分真实路程(米),用于标定 COLMAP 的尺度。"""
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
    """RANSAC 拟合地面 → 旋转矩阵使地面法向竖直(+Z)。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(0.2) if len(pts) > 50000 else pcd
    plane, inl = pcd.segment_plane(0.15, 3, 400)
    n = np.array(plane[:3]); n = n / np.linalg.norm(n)
    b = np.array([0, 0, 1.0])
    v = np.cross(n, b); s = np.linalg.norm(v); c = float(np.dot(n, b))
    if s < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "111450"
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    work = f"outputs/_colmap_{session}"
    img_dir = f"{work}/images"
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(img_dir)

    allimg = sorted(glob.glob(f"data/{session}/images/zed/*.jpg"))
    sel = allimg[::stride]
    for p in sel:
        os.symlink(os.path.abspath(p), os.path.join(img_dir, os.path.basename(p)))
    print(f"[INFO] {session}: {len(allimg)} 图 → SfM 用 {len(sel)} 张 (stride={stride})")

    db = f"{work}/database.db"
    print("[INFO] 提特征(GPU SIFT)…")
    pycolmap.extract_features(db, img_dir, device=pycolmap.Device.auto)
    print("[INFO] exhaustive 匹配(GPU)…")
    pycolmap.match_exhaustive(db)
    print("[INFO] 增量建图(全局BA)…")
    recs = pycolmap.incremental_mapping(db, img_dir, f"{work}/sparse")
    if not recs:
        print("[ERR] COLMAP 没建出模型"); return
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"[INFO] 模型: 注册 {rec.num_reg_images()}/{len(sel)} 张, 三维点 {rec.num_points3D()}")

    # 相机轨迹(按时间戳排序)
    name2c = {}
    for img in rec.images.values():
        name2c[img.name] = np.array(img.cam_from_world.inverse().translation)
    names = sorted(name2c.keys())               # 文件名=时间戳,排序=时间序
    C = np.array([name2c[n] for n in names])
    P = np.array([p.xyz for p in rec.points3D.values()])

    # 重力对齐(地面→水平),BEV 取对齐后的 (x,y),高度=z
    Rg = align_gravity(P)
    P = (Rg @ P.T).T; C = (Rg @ C.T).T

    # 尺度标定:让轨迹总长 = cmd 真实路程
    colmap_len = float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1)))
    real_len = real_path_length(session)
    scale = real_len / colmap_len if colmap_len > 1e-6 else 1.0
    P *= scale; C *= scale
    print(f"[INFO] 尺度标定: COLMAP轨迹长 {colmap_len:.2f} → 真实 {real_len:.1f}m (×{scale:.2f})")
    print(f"[INFO] 轨迹净位移 {np.linalg.norm(C[-1]-C[0]):.1f}m (来回应较小)")

    # 高度裁剪去离群,导出
    z = P[:, 2]; zlo, zhi = np.percentile(z, [1, 99])
    P = P[(z > zlo) & (z < zhi)]
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P)
    os.makedirs("outputs", exist_ok=True)
    o3d.io.write_point_cloud(f"outputs/colmap_{session}.ply", pc)
    np.savez(f"outputs/traj_{session}_colmap.npz", x=C[:, 0], y=C[:, 1], z=C[:, 2])
    print(f"[OK] 三维点 {len(P)} → outputs/colmap_{session}.ply")

    # 俯视 BEV(色=高度)+ 轨迹
    fig, ax = plt.subplots(figsize=(12, 9))
    zz = P[:, 2]
    sc = ax.scatter(P[:, 0], P[:, 1], s=0.6, c=zz, cmap="turbo", alpha=0.6)
    ax.plot(C[:, 0], C[:, 1], "-", color="red", lw=2, label="相机轨迹")
    ax.plot(C[0, 0], C[0, 1], "o", color="lime", ms=12, mec="k", label="起点")
    ax.plot(C[-1, 0], C[-1, 1], "s", color="k", ms=10, label="终点")
    ax.set_aspect("equal"); ax.legend(); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{session} COLMAP 全局SfM重建 俯视BEV (注册{rec.num_reg_images()}张, 色=高度)")
    plt.colorbar(sc, ax=ax, label="高度(m)")
    fig.tight_layout(); fig.savefig(f"outputs/colmap_{session}_bev.png", dpi=130)
    print(f"[OK] 已保存 outputs/colmap_{session}_bev.png")


if __name__ == "__main__":
    main()
