"""从世界帧 3DGS 点云拟合「干净几何世界」: 两侧走廊墙 + 障碍盒。

输入 (align_gs_world.py 产出, 世界帧 z-up 米制 地面z=0):
  outputs/gs_vggto_<s>_world.ply   3DGS 高斯 (xyz/opacity/scale)
  outputs/sim_world_<s>.npz        waypoints_xy (机器人轨迹折线, 走廊中心)
输出:
  outputs/world_geometry_<s>.json   walls(分段折线+高度+厚度) + boxes(center+halfsize)
  outputs/world_geometry_<s>_check.png  俯视拟合叠点云, 人眼把关

墙拟合用「相对轨迹的横向偏移」而非全局 x 直方图: 对每个中层点求到轨迹折线的
带符号垂距(左负右正)与沿轨迹弧长 s, 横向偏移直方图的两个峰=左右墙距离, 按 s 分 bin
取中位偏移得墙中心折线 -> 对走廊弯曲/朝向都鲁棒。

用法: PYTHONPATH=third_party/gs_playground/.venv/lib/python3.10/site-packages \
      ~/discoverse_venv/bin/python scripts/fit_world_geometry.py [session]
"""
import os
import sys
import json

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION = sys.argv[1] if len(sys.argv) > 1 else "114830c"

# ---- 阈值 (世界帧米制) ----
OP_MIN = 0.15          # 不透明度下限(剔飘片)
SCALE_MAX = 0.3        # 单高斯最大尺度(剔超大浮点), 米
Z_FLOOR = 0.18         # 地面层上限
Z_WALL_LO, Z_WALL_HI = 0.3, 2.2   # 取中层点拟合墙(避开地面与天花板浮点)
WALL_THICK = 0.12
WALL_H_CAP = 2.6
SEG_LEN = 2.0          # 墙分段沿轨迹弧长步长(米)
COVER_MIN = 0.5        # 段内点覆盖率下限(防假墙)
CROP_PAD = 3.0         # 轨迹包围盒外扩(米), 裁离群浮点
WALL_BAND = 0.45       # 距墙中心 ±此值内算"属于墙"的点(从障碍里剔除)
PATH_CLEAR = 0.6       # 障碍盒中心距轨迹 <此值 视为挡路噪声剔除(真实障碍贴墙在路侧)


def load_points():
    # GEOM_PLY: 用普通点云(如 ZED 深度融合 zed_fused_world.ply)提几何, 比从高斯硬拟合干净
    geom_ply = os.environ.get("GEOM_PLY")
    if geom_ply:
        import open3d as o3d
        xyz = np.asarray(o3d.io.read_point_cloud(geom_ply).points, dtype=np.float64)
        print(f"[denoise] 普通点云 {geom_ply}: {len(xyz)} 点(ZED深度,无需opacity/scale过滤)")
        return xyz
    from gaussian_renderer.core.util_gau import load_ply
    gd = load_ply(os.path.join(ROOT, "outputs", f"gs_vggto_{SESSION}_world.ply"))
    xyz = np.asarray(gd.xyz, dtype=np.float64)
    op = np.asarray(gd.opacity, dtype=np.float64).reshape(-1)
    sc = np.asarray(gd.scale, dtype=np.float64).max(axis=1)
    keep = (op > OP_MIN) & (sc < SCALE_MAX)
    xyz = xyz[keep]
    print(f"[denoise] opacity/scale 过滤后 {len(xyz)}/{len(keep)}")
    return xyz


def crop_to_traj(xyz, wp):
    lo = wp.min(0) - CROP_PAD
    hi = wp.max(0) + CROP_PAD
    m = (xyz[:, 0] > lo[0]) & (xyz[:, 0] < hi[0]) & (xyz[:, 1] > lo[1]) & (xyz[:, 1] < hi[1])
    print(f"[crop] 轨迹包围盒裁剪后 {m.sum()}/{len(xyz)}")
    return xyz[m]


def smooth_centerline(wp, step=0.5, win=2.0):
    """轨迹重度平滑 + 按弧长等间隔重采样, 返回 (节点 xy, 单位切向, 单位法向, 弧长)。"""
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    s_in = np.concatenate([[0], np.cumsum(seg)])
    total = s_in[-1]
    s_u = np.arange(0, total, step)
    cx = np.interp(s_u, s_in, wp[:, 0])
    cy = np.interp(s_u, s_in, wp[:, 1])
    # 移动平均平滑(窗口 win 米)
    k = max(1, int(win / step))
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    cx = np.convolve(np.pad(cx, k, mode="edge"), ker, "valid")
    cy = np.convolve(np.pad(cy, k, mode="edge"), ker, "valid")
    C = np.stack([cx, cy], axis=1)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    Nrm = np.stack([-T[:, 1], T[:, 0]], axis=1)            # 左法向
    return C, T, Nrm, s_u


def traj_frame(pts_xy, C, Nrm, s_u):
    """对每个点求最近中心线节点的带符号横向偏移(沿法向)与节点弧长。"""
    from scipy.spatial import cKDTree
    tree = cKDTree(C)
    _, k = tree.query(pts_xy)
    rel = pts_xy - C[k]
    lateral = np.einsum("nj,nj->n", rel, Nrm[k])           # 沿左法向的带符号偏移
    return lateral, k


def fit_wall_side(C, Nrm, s_u, lateral, node_idx, side_sign, peak_off):
    """某侧墙: 每个中心线节点取带内点的中位偏移(平滑) -> 墙 = 中心线 + 法向*偏移。"""
    band = (np.sign(peak_off) == side_sign) & (np.abs(lateral - peak_off) < 0.6)
    if band.sum() < 50:
        return []
    nn = len(C)
    off = np.full(nn, np.nan)
    bi = node_idx[band]; bl = lateral[band]
    for n in range(nn):
        m = bi == n
        if m.sum() >= 8:
            off[n] = np.median(bl[m])
    cover = np.isfinite(off).mean()
    print(f"[wall {('R' if side_sign>0 else 'L')}] peak={peak_off:.2f} 带内点={band.sum()} "
          f"覆盖率={cover:.2f}")
    if cover < COVER_MIN:
        print(f"  -> 覆盖率<{COVER_MIN}, 丢弃(疑似假墙)")
        return []
    # 缺口用常数峰值填 + 重度平滑偏移(去抖)
    off[~np.isfinite(off)] = peak_off
    k = 6
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    off = np.convolve(np.pad(off, k, mode="edge"), ker, "valid")
    wallpts = C + Nrm * off[:, None]
    # 抽稀成线段(每 SEG_LEN 一节点)
    stepn = max(1, int(SEG_LEN / (s_u[1] - s_u[0])))
    nodes = wallpts[::stepn]
    if not np.allclose(nodes[-1], wallpts[-1]):
        nodes = np.vstack([nodes, wallpts[-1]])
    return [(nodes[i].tolist(), nodes[i + 1].tolist()) for i in range(len(nodes) - 1)]


def main():
    import open3d as o3d
    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{SESSION}.npz"))
    wp = np.asarray(W["waypoints_xy"], dtype=np.float64)

    xyz = load_points()
    xyz = crop_to_traj(xyz, wp)

    # 统计离群剔除
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    xyz = np.asarray(pc.points)
    print(f"[denoise] 统计离群后 {len(xyz)}")

    C, T, Nrm, s_u = smooth_centerline(wp)
    z = xyz[:, 2]
    mid = (z > Z_WALL_LO) & (z < Z_WALL_HI)
    midp = xyz[mid]
    lateral, node_idx = traj_frame(midp[:, :2], C, Nrm, s_u)

    # 横向偏移直方图找两墙(左峰<0, 右峰>0)
    hist, edges = np.histogram(lateral, bins=60, range=(-6, 6))
    ctr = 0.5 * (edges[:-1] + edges[1:])
    left_mask = ctr < -0.5; right_mask = ctr > 0.5
    peakL = ctr[left_mask][hist[left_mask].argmax()]
    peakR = ctr[right_mask][hist[right_mask].argmax()]
    print(f"[walls] 横向偏移峰 L={peakL:.2f}m R={peakR:.2f}m (走廊宽≈{peakR-peakL:.2f}m)")

    h95 = min(WALL_H_CAP, float(np.percentile(z[mid], 95)))
    walls = []
    walls.append({"segs": fit_wall_side(C, Nrm, s_u, lateral, node_idx, -1, peakL),
                  "height": h95, "thick": WALL_THICK})
    walls.append({"segs": fit_wall_side(C, Nrm, s_u, lateral, node_idx, +1, peakR),
                  "height": h95, "thick": WALL_THICK})
    walls = [w for w in walls if w["segs"]]

    # 障碍盒: 中层点里, 不属于两墙带、且在走廊内的点 -> DBSCAN 聚类
    near_wall = (np.abs(lateral - peakL) < WALL_BAND) | (np.abs(lateral - peakR) < WALL_BAND)
    inside = (lateral > peakL + WALL_BAND) & (lateral < peakR - WALL_BAND)
    obs_mask = (~near_wall) & inside
    boxes = []
    obs = midp[obs_mask]
    # 轨迹加密(用于"盒是否挡路"判定): 真实障碍贴墙在路侧, 压在走过的路上的=拟合噪声须剔除
    from scipy.spatial import cKDTree
    dense = []
    for a, b in zip(wp[:-1], wp[1:]):
        n = max(2, int(np.linalg.norm(b - a) / 0.2))
        dense.append(np.linspace(a, b, n))
    traj_tree = cKDTree(np.vstack(dense))
    n_block = 0
    if len(obs) > 50:
        opc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obs))
        labels = np.array(opc.cluster_dbscan(eps=0.4, min_points=80))
        for lab in range(labels.max() + 1):
            cl = obs[labels == lab]
            if len(cl) < 80:
                continue
            mn, mx = cl.min(0), cl.max(0)
            half = (mx - mn) / 2.0
            center = (mx + mn) / 2.0
            vol = np.prod(2 * half)
            if vol < 0.05 or (2 * half).min() < 0.15 or (2 * half).max() > 5.0:
                continue
            if traj_tree.query(center[:2])[0] < PATH_CLEAR:    # 中心压在走过的路上 -> 拟合噪声, 剔
                n_block += 1; continue
            center[2] = max(center[2], half[2])           # 盒底不穿地
            boxes.append({"center": center.tolist(), "half": half.tolist()})
    print(f"[boxes] 障碍盒 {len(boxes)} 个 (剔除挡路噪声盒 {n_block} 个)")

    geom = {"session": SESSION, "floor_z": 0.0, "walls": walls, "boxes": boxes,
            "wall_offsets": [float(peakL), float(peakR)]}
    outp = os.path.join(ROOT, "outputs", f"world_geometry_{SESSION}.json")
    with open(outp, "w") as f:
        json.dump(geom, f, indent=1)
    print(f"[ok] -> {outp}")

    save_check(xyz, midp, wp, walls, boxes)


def save_check(xyz, midp, wp, walls, boxes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7, 14))
    sub = midp[np.random.choice(len(midp), min(60000, len(midp)), replace=False)]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.4, c="0.7", lw=0)
    ax.plot(wp[:, 0], wp[:, 1], "-", c="tab:green", lw=1.5, label="traj")
    for w in walls:
        for a, b in w["segs"]:
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", c="tab:blue", lw=2.5)
    for bx in boxes:
        c, h = bx["center"], bx["half"]
        ax.add_patch(Rectangle((c[0] - h[0], c[1] - h[1]), 2 * h[0], 2 * h[1],
                               fill=False, ec="tab:red", lw=1.5))
    ax.set_aspect("equal"); ax.legend(loc="upper right")
    ax.set_title(f"world_geometry {SESSION}: walls(blue) boxes(red)")
    outp = os.path.join(ROOT, "outputs", f"world_geometry_{SESSION}_check.png")
    fig.savefig(outp, dpi=90, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] check -> {outp}")


if __name__ == "__main__":
    main()
