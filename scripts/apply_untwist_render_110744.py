"""把0731世界扶平warp同步到capfpv渲染用的pre-align ply (0731续, 用户"虚线对地面还是歪的")。

真因: CaptureFPV渲染的是 gs_vggto_<session>.ply(pre-align归一化模型)+npz刚性映射,
之前apply_untwist_110744.py只扶平了world ply(只占据栅格用)→FPV画面一点没变。

做法: pre-align ply反归一化→colmap帧→刚性映射到world帧→施加与apply_untwist_110744
完全相同的warp(逐waypoint局部地面法线+空间kNN平滑+近邻混合旋绕锚点)→逆变换回
归一化帧保存。quats用共轭旋转 R_c = Rz⁻¹·R_w·Rz 左乘。

产出: outputs/gs_vggto_colmapjoint_110744s2_lvl.ply + 软链 gs_vggto_colmapjoint_110744s2.ply改指(旧bak保留)
用法: ~/discoverse_venv/bin/python scripts/apply_untwist_render_110744.py
"""
import os, shutil
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rsc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SES = "colmapjoint_110744s2"
PLY_LINK = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}.ply")          # capfpv读这个
PLY_OUT = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}_lvl.ply")
WORLD_PLY = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}_world.ply")   # 已扶平(测法线用)
NPZ = os.path.join(ROOT, "outputs", f"sim_world_{SES}.npz")
KNN_SMOOTH, SIGMA, KNN_APPLY = 15, 1.0, 8


def load_gauss_raw(path):
    """与capture_fpv._load_gauss同格式, 但保留原始numpy(归一化帧)。"""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"; f.readline()
        n = int(f.readline().split()[-1]); props = []
        ln = f.readline()
        while ln.strip() != b"end_header":
            if ln.startswith(b"property"): props.append(ln.split()[-1].decode())
            ln = f.readline()
        d = np.frombuffer(f.read(n * len(props) * 4), np.float32).reshape(n, len(props))
    g = {p: d[:, i].copy() for i, p in enumerate(props)}
    cs = np.load(path + ".norm.npy"); center, s = cs[:3].astype(np.float64), float(cs[3])
    return g, props, center, s


def save_gauss_raw(g, props, path):
    order = props
    d = np.stack([g[p] for p in order], 1).astype(np.float32)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(d)}\n" +
              "".join(f"property float {p}\n" for p in order) +
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode()); f.write(d.tobytes())


def main():
    cap = dict(np.load(NPZ))
    wp = cap["waypoints_xy"].astype(np.float64)
    Rz = cap["Rz"].astype(np.float64); k = float(cap["k"])
    t_world = cap["t_world"].astype(np.float64); center_w = cap["center"].astype(np.float64)
    s_norm = float(cap["s_norm"])

    # 1) warp场: 在已扶平world ply上重算逐waypoint地面法线纠正(与apply_untwist_110744相同)
    import trimesh
    xyz_w = np.asarray(trimesh.load(WORLD_PLY, process=False).vertices, dtype=np.float64)
    fl = xyz_w[(xyz_w[:, 2] > -0.12) & (xyz_w[:, 2] < 0.12)]
    tree = cKDTree(fl[:, :2])
    rv = np.zeros((len(wp), 3))
    for i, (sx, sy) in enumerate(wp):
        idx = tree.query_ball_point([sx, sy], 1.5)
        if len(idx) < 300: continue
        Q = fl[idx]
        for _ in range(3):
            c = Q.mean(0); Qc = Q - c
            _, _, Vt = np.linalg.svd(Qc, full_matrices=False)
            n = Vt[2]
            if n[2] < 0: n = -n
            r = np.abs(Qc @ n)
            Q = Q[r < np.percentile(r, 70)]
        axis = np.cross(n, [0, 0, 1]); na = np.linalg.norm(axis)
        if na > 1e-8:
            rv[i] = axis / na * np.arccos(np.clip(n[2], -1, 1))
    tree_wp = cKDTree(wp)
    _, knn = tree_wp.query(wp, k=KNN_SMOOTH)
    d_knn = np.linalg.norm(wp[knn] - wp[:, None, :], axis=2)
    w = np.exp(-(d_knn / SIGMA) ** 2); w /= w.sum(1, keepdims=True)
    rv_s = (rv[knn] * w[:, :, None]).sum(1)
    print(f"[lvl-render] warp场: mean {np.degrees(np.linalg.norm(rv_s,axis=1)).mean():.2f}°", flush=True)

    # ⚠注意: 这里测的是【已扶平world ply】的残余, 我们要的warp是把【未扶平几何】扶平的场。
    # 正确做法: 在PREUNTWIST世界帧(=渲染几何当前状态)上测。world ply已扶平→残余场≈0,
    # 所以改为直接复用apply_untwist_110744测得的场: 在bak world ply上重测。
    bak = os.path.join(ROOT, "outputs", "gs_vggto_colmapjoint_110744_world_stride2_bak.ply")
    xyz_b = np.asarray(trimesh.load(bak, process=False).vertices, dtype=np.float64)
    fl = xyz_b[(xyz_b[:, 2] > -0.12) & (xyz_b[:, 2] < 0.12)]
    tree = cKDTree(fl[:, :2])
    rv = np.zeros((len(wp), 3))
    for i, (sx, sy) in enumerate(wp):
        idx = tree.query_ball_point([sx, sy], 1.5)
        if len(idx) < 300: continue
        Q = fl[idx]
        for _ in range(3):
            c = Q.mean(0); Qc = Q - c
            _, _, Vt = np.linalg.svd(Qc, full_matrices=False)
            n = Vt[2]
            if n[2] < 0: n = -n
            r = np.abs(Qc @ n)
            Q = Q[r < np.percentile(r, 70)]
        axis = np.cross(n, [0, 0, 1]); na = np.linalg.norm(axis)
        if na > 1e-8:
            rv[i] = axis / na * np.arccos(np.clip(n[2], -1, 1))
    rv_s = (rv[knn] * w[:, :, None]).sum(1)
    print(f"[lvl-render] bak上warp场: mean {np.degrees(np.linalg.norm(rv_s,axis=1)).mean():.2f}° "
          f"max {np.degrees(np.linalg.norm(rv_s,axis=1)).max():.2f}°", flush=True)

    # 2) pre-align ply → colmap帧 → world帧
    g, props, center, s = load_gauss_raw(PLY_LINK)  # norm.npy在链接名旁边
    N = len(g["x"])
    means_norm = np.stack([g["x"], g["y"], g["z"]], 1).astype(np.float64)
    pc = means_norm * s + center                                   # colmap帧(米制前)
    pw = t_world + (k / s_norm) * (Rz @ (pc - center_w).T).T       # world帧(与bak world ply同帧)
    print(f"[lvl-render] {N} gaussians mapped to world", flush=True)

    # 3) world帧施加warp
    _, ka = tree_wp.query(pw[:, :2], k=KNN_APPLY)
    da = np.linalg.norm(pw[:, None, :2] - wp[ka], axis=2)
    wa = 1.0 / (da ** 2 + 0.05); wa /= wa.sum(1, keepdims=True)
    RV = (rv_s[ka] * wa[:, :, None]).sum(1)
    anch3 = np.concatenate([wp, np.zeros((len(wp), 1))], 1)
    A = (anch3[ka] * wa[:, :, None]).sum(1)
    Rs = Rsc.from_rotvec(RV)
    pw2 = Rs.apply(pw - A) + A
    # 回colmap帧→回归一化
    pc2 = center_w + (s_norm / k) * (Rz.T @ (pw2 - t_world).T).T
    means2 = (pc2 - center) / s
    g["x"], g["y"], g["z"] = means2[:, 0], means2[:, 1], means2[:, 2]

    # 4) quats: world帧旋转R_w → colmap帧 R_c = Rz⁻¹ R_w Rz (作用在旋转部分, 平移无关)
    Rz_r = Rsc.from_matrix(Rz)
    Rc = Rz_r.inv() * Rs * Rz_r                                   # 共轭
    q = np.stack([g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]], 1).astype(np.float64)  # wxyz
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    r_old = Rsc.from_quat(q[:, [1, 2, 3, 0]])
    r_new = (Rc * r_old).as_quat()
    qn = r_new[:, [3, 0, 1, 2]]
    for j in range(4):
        g[f"rot_{j}"] = qn[:, j]

    save_gauss_raw(g, props, PLY_OUT)
    shutil.copy(PLY_LINK + ".norm.npy", PLY_OUT + ".norm.npy")
    print(f"[lvl-render] ply -> {PLY_OUT}", flush=True)
    if os.path.islink(PLY_LINK):
        os.remove(PLY_LINK)
    else:
        shutil.move(PLY_LINK, PLY_LINK.replace(".ply", "_PRELVL0731.ply"))
    os.symlink(os.path.basename(PLY_OUT), PLY_LINK)
    print(f"[lvl-render] 软链 {os.path.basename(PLY_LINK)} -> {os.path.basename(PLY_OUT)}", flush=True)


if __name__ == "__main__":
    main()
