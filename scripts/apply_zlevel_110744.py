"""110744s2 z场扶平 (0804, 用户判"地平面还是有问题")。

实测(apply前, world_lvl ply): 地面残斜 全局2.58°, 白桌2.52°/原位3.24°/深处2.47°/柜1.38°/微1.03°。
0731 untwist=逐waypoint局部【旋转】+kNN平滑, 对非刚性弯曲(地面扭成波浪)只能扶个大概。
本脚本换【z场竖向反弯】: 地面高度场h(x,y)(0.4m格中位数+就近填充+高斯平滑), z'=z-h(x,y)
→地面逐格压到z≈0, 弯曲多强都扶得平。纯竖向位移场, quats/yaw/front角全不变(梯度≤3°的
splat朝向误差可忽略)。

同时变换: world ply(占据/BEV) + render ply(pre-align, 走world帧施场再逆映射, 与
apply_untwist_render_110744同舞) + npz waypoints_z + node_clusters。targets是2D(xy)不用动。
产出: gs_vggto_colmapjoint_110744s2_world_zlvl.ply / _zlvl.ply + 软链改指(旧lvl版留盘不删)
用法: ~/discoverse_venv/bin/python scripts/apply_zlevel_110744.py
"""
import os, shutil
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SES = "colmapjoint_110744s2"
WORLD_LINK = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}_world.ply")     # 软链(消费端读这个)
WORLD_SRC = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}_world_lvl.ply")  # ⭐输入固定读lvl版(首跑翻车把link改指过zlvl, 别跟链)
WORLD_OUT = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}_world_zlvl.ply")
RENDER_LINK = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}.ply")          # 软链(capfpv读这个)
RENDER_SRC = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}_lvl.ply")       # ⭐同上, 输入固定读lvl版
RENDER_OUT = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}_zlvl.ply")
NPZ = os.path.join(ROOT, "outputs", f"sim_world_{SES}.npz")
CLUSTERS = os.path.join(ROOT, "outputs", "cg_lab110744s2", "node_clusters_mission3.npz")

CELL = 0.4           # 高度场格子(米)
SMOOTH = 1.5         # 高斯平滑σ(格)=0.6m
BAND = 0.20          # 地面带 |z|<0.2(扶平后地面≈0, 残弯±10cm内)


def build_hfield(xyz, wp):
    """地面高度场h(x,y): ⭐逐waypoint局部平面拟合(z∈[-0.12,0.12]+3×70%残差修剪, 与
    apply_untwist_110744同款=家具腿/柜底被修剪拒绝)取平面在waypoint处的高度h_i,
    再kNN高斯平滑+反距离插值到场内任意xy(场外取最近waypoint=常数平移不弯)。
    ⚠首版网格中位数法翻车: |z|<0.2带混进家具底部→场脏→越扶越歪(已回滚)。"""
    fl = xyz[(xyz[:, 2] > -0.12) & (xyz[:, 2] < 0.12)]
    tree = cKDTree(fl[:, :2])
    h = np.zeros(len(wp)); ok = np.zeros(len(wp), bool)
    for i, (sx, sy) in enumerate(wp):
        idx = tree.query_ball_point([sx, sy], 1.5)
        if len(idx) < 300:
            continue
        Q = fl[idx].astype(np.float64)
        for _ in range(3):
            c = Q.mean(0); Qc = Q - c
            _, _, Vt = np.linalg.svd(Qc, full_matrices=False)
            n = Vt[2]
            if n[2] < 0: n = -n
            r = np.abs(Qc @ n)
            Q = Q[r < np.percentile(r, 70)]
        if abs(n[2]) > 0.5:                            # 拟合必须像地面(近水平)
            h[i] = c[2] - (n[0] * (sx - c[0]) + n[1] * (sy - c[1])) / n[2]
            ok[i] = True
    tree_wp = cKDTree(wp)
    for i in np.where(~ok)[0]:                         # 无效点近邻填
        _, ii = tree_wp.query(wp[i], k=15)
        h[i] = h[ii][ok[ii]].mean() if ok[ii].any() else 0.0
    _, knn = tree_wp.query(wp, k=15)                   # kNN高斯平滑(σ=1m, 与untwist同)
    d_knn = np.linalg.norm(wp[knn] - wp[:, None, :], axis=2)
    w = np.exp(-(d_knn / 1.0) ** 2); w /= w.sum(1, keepdims=True)
    hs = (h[knn] * w).sum(1)
    print(f"[zlvl] 高度场: 有效{ok.sum()}/{len(wp)}, h范围[{hs.min():+.3f},{hs.max():+.3f}]m", flush=True)

    def h0_of(pxy):                                    # 粗场: 反距离插值(8近邻), pxy=(N,2)
        _, ka = tree_wp.query(pxy, k=8)
        da = np.linalg.norm(pxy[:, None, :] - wp[ka], axis=2)
        wa = 1.0 / (da ** 2 + 0.05); wa /= wa.sum(1, keepdims=True)
        return (hs[ka] * wa).sum(1)

    # ⭐精修(粗场只治长波, 残2°+): 粗场±5cm残差带=干净真地面(家具底被排) →
    # 0.25m格中位数 → 就近填充 → σ1格平滑。实测median倾斜 2.24→1.37°(σ1.0纯粗场)。
    res0 = xyz[:, 2] - h0_of(xyz[:, :2])
    tf = xyz[np.abs(res0) < 0.05]
    print(f"[zlvl] 精修真地面点 {len(tf)}", flush=True)
    CELL = 0.25
    lo = wp.min(0) - 1.0; hi = wp.max(0) + 1.0
    nx = int(np.ceil((hi[0] - lo[0]) / CELL)); ny = int(np.ceil((hi[1] - lo[1]) / CELL))
    ix = np.clip(((tf[:, 0] - lo[0]) / CELL).astype(int), 0, nx - 1)
    iy = np.clip(((tf[:, 1] - lo[1]) / CELL).astype(int), 0, ny - 1)
    key = iy * nx + ix
    order = np.argsort(key, kind="stable"); key_s, z_s = key[order], tf[order, 2]
    uk, start = np.unique(key_s, return_index=True)
    grid = np.full((ny, nx), np.nan)
    for k0, grp in zip(uk, np.split(z_s, start[1:])):
        if len(grp) >= 15:
            grid[k0 // nx, k0 % nx] = np.median(grp)
    valid = ~np.isnan(grid)
    yy, xx = np.mgrid[0:ny, 0:nx]
    t2 = cKDTree(np.stack([yy[valid], xx[valid]], 1))
    _, nn = t2.query(np.stack([yy.ravel(), xx.ravel()], 1))
    H = gaussian_filter(grid[valid].ravel()[nn].reshape(ny, nx), 1.0)
    print(f"[zlvl] 精修场 {ny}x{nx} 有效格{valid.sum()}, H范围[{H.min():+.3f},{H.max():+.3f}]m", flush=True)

    def h_of(pxy):                                     # 精修场双线性插值
        fx = np.clip((pxy[:, 0] - lo[0]) / CELL - 0.5, 0, nx - 1.001)
        fy = np.clip((pxy[:, 1] - lo[1]) / CELL - 0.5, 0, ny - 1.001)
        x0 = fx.astype(int); y0 = fy.astype(int)
        dx = fx - x0; dy = fy - y0
        return (H[y0, x0] * (1 - dx) * (1 - dy) + H[y0, x0 + 1] * dx * (1 - dy) +
                H[y0 + 1, x0] * (1 - dx) * dy + H[y0 + 1, x0 + 1] * dx * dy)
    return h_of


def measure_tilt(xyz, label):
    """⭐与0804诊断同法: 只取最低切片(z<2%分位+0.15)=纯地面, 别用±band(混家具=假倾斜)。"""
    z0 = np.percentile(xyz[:, 2], 2)
    fl = xyz[xyz[:, 2] < z0 + 0.15]
    stations = {"iron_cabinet": (-0.57, -3.55), "microscope": (-0.64, -2.96),
                "white_table": (1.26, -0.47), "origin": (0, 0), "far_corner": (0.3, -5.0)}
    out = []
    for nm, (x, y) in stations.items():
        sel = fl[np.hypot(fl[:, 0] - x, fl[:, 1] - y) < 0.8]
        if len(sel) < 50:
            continue
        Q = sel - sel.mean(0)
        _, _, vt = np.linalg.svd(Q, full_matrices=False)
        n = vt[2]; n = n / np.linalg.norm(n)
        if n[2] < 0: n = -n
        t = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
        out.append(f"{nm}={t:.2f}°")
    print(f"[zlvl] {label} 地面倾斜: {' '.join(out)}", flush=True)

def load_gauss_raw(path, need_norm=True):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"; f.readline()
        n = int(f.readline().split()[-1]); props = []
        ln = f.readline()
        while ln.strip() != b"end_header":
            if ln.startswith(b"property"): props.append(ln.split()[-1].decode())
            ln = f.readline()
        d = np.frombuffer(f.read(n * len(props) * 4), np.float32).reshape(n, len(props))
    g = {p: d[:, i].copy() for i, p in enumerate(props)}
    if need_norm:
        cs = np.load(path + ".norm.npy"); center, s = cs[:3].astype(np.float64), float(cs[3])
    else:
        center, s = None, None
    return g, props, center, s


def save_gauss_raw(g, props, path):
    d = np.stack([g[p] for p in props], 1).astype(np.float32)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(d)}\n" +
              "".join(f"property float {p}\n" for p in props) + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode()); f.write(d.tobytes())


def main():
    # 1) 在world_lvl ply上建高度场 + 验证扶平效果(保留全属性, 只改xyz)
    gw, props_w, _, _ = load_gauss_raw(WORLD_SRC, need_norm=False)
    xyz = np.stack([gw["x"], gw["y"], gw["z"]], 1).astype(np.float64)
    print(f"[zlvl] world ply {xyz.shape[0]} gaussians, props={len(props_w)}", flush=True)
    measure_tilt(xyz, "扶平前")
    cap = dict(np.load(NPZ))
    wp = cap["waypoints_xy"].astype(np.float64)
    h_of = build_hfield(xyz, wp)
    xyz2 = xyz.copy(); xyz2[:, 2] -= h_of(xyz[:, :2])
    measure_tilt(xyz2, "扶平后")
    gw["x"], gw["y"], gw["z"] = xyz2[:, 0], xyz2[:, 1], xyz2[:, 2]
    save_gauss_raw(gw, props_w, WORLD_OUT)
    print(f"[zlvl] world ply -> {WORLD_OUT}", flush=True)

    # 2) render ply: pre-align归一化帧→colmap帧→world帧→施z场→逆回
    Rz = cap["Rz"].astype(np.float64); k = float(cap["k"])
    t_world = cap["t_world"].astype(np.float64); center_w = cap["center"].astype(np.float64)
    s_norm = float(cap["s_norm"])
    g, props, center, s = load_gauss_raw(RENDER_SRC)
    means_norm = np.stack([g["x"], g["y"], g["z"]], 1).astype(np.float64)
    pc = means_norm * s + center
    pw = t_world + (k / s_norm) * (Rz @ (pc - center_w).T).T        # world帧
    pw[:, 2] -= h_of(pw[:, :2])                                     # 施z场(quats不动=纯竖向)
    pc2 = center_w + (s_norm / k) * (Rz.T @ (pw - t_world).T).T
    means2 = (pc2 - center) / s
    g["x"], g["y"], g["z"] = means2[:, 0], means2[:, 1], means2[:, 2]
    save_gauss_raw(g, props, RENDER_OUT)
    shutil.copy(RENDER_SRC + ".norm.npy", RENDER_OUT + ".norm.npy")
    print(f"[zlvl] render ply -> {RENDER_OUT}", flush=True)

    # 3) npz: waypoints_z 同步(采集相机高度跟着地面走)
    wxy = cap["waypoints_xy"].astype(np.float64)
    cap["waypoints_z"] = cap["waypoints_z"].astype(np.float64) - h_of(wxy)
    shutil.copy(NPZ, NPZ.replace(".npz", "_PREZLVL0804.npz"))
    np.savez(NPZ, **cap)
    print(f"[zlvl] npz -> {NPZ} (备份_PREZLVL0804)", flush=True)

    # 4) clusters 同步
    if os.path.exists(CLUSTERS):
        cl = dict(np.load(CLUSTERS))
        for kk in cl:
            P = cl[kk].astype(np.float64)
            P[:, 2] -= h_of(P[:, :2])
            cl[kk] = P.astype(np.float32)
        shutil.copy(CLUSTERS, CLUSTERS.replace(".npz", "_PREZLVL0804.npz"))
        np.savez(CLUSTERS, **cl)
        print(f"[zlvl] clusters -> {CLUSTERS} (备份_PREZLVL0804)", flush=True)

    # 5) 软链改指(world.ply现是指向world_lvl.ply的软链, 直接改指zlvl; render同理)
    for link, out in ((WORLD_LINK, WORLD_OUT), (RENDER_LINK, RENDER_OUT)):
        if os.path.islink(link):
            os.remove(link)
        else:
            shutil.move(link, link.replace(".ply", "_PREZLVL0804.ply"))
        os.symlink(os.path.basename(out), link)
        print(f"[zlvl] 软链 {os.path.basename(link)} -> {os.path.basename(out)}", flush=True)


if __name__ == "__main__":
    main()
