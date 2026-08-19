"""110744s2 世界局部扶平 (0731, 用户"还是斜的"真因=世界本身roll/pitch扭曲)。

实测(apply前): 主区(白桌/origin/走廊)局部地面倾斜3.6~4.2°@方位19°, 南端铁柜区0.25~1.9°
=空间渐变扭曲, 全局旋转治不了 → 逐waypoint测局部地面法线→最小旋正→空间kNN平滑→
每个高斯按其近邻waypoint的混合旋绕局部锚点转(111450 untwist同款思路, 但直接作用在当前
world ply上, 不重跑align避免坐标漂移)。

同时变换: sim_world npz(waypoints不动=锚点本身, yaw按各自旋正更新) / targets json /
node_clusters_mission3.npz。⚠CG traj.txt未变(重跑三角化时才需要, 遗留)。

用法: ~/discoverse_venv/bin/python scripts/apply_untwist_110744.py
产出: outputs/gs_vggto_colmapjoint_110744s2_world_lvl.ply (+world.ply软链改指它, 旧bak保留)
      outputs/sim_world_colmapjoint_110744s2.npz (旧版备份_PRELVL0731)
      outputs/lab_mission_targets_zed_front_perp_lvl.json (坐标变换后; front重算见lab_mission前处理)
"""
import os, json, shutil
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rsc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SES = "colmapjoint_110744s2"
PLY_IN = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}_world.ply")
PLY_OUT = os.path.join(ROOT, "outputs", f"gs_vggto_{SES}_world_lvl.ply")
NPZ = os.path.join(ROOT, "outputs", f"sim_world_{SES}.npz")
TARGETS_IN = os.path.join(ROOT, "outputs", "lab_mission_targets_zed_front_perp.json")
TARGETS_OUT = os.path.join(ROOT, "outputs", "lab_mission_targets_zed_front_perp_lvl.json")
CLUSTERS = os.path.join(ROOT, "outputs", "cg_lab110744s2", "node_clusters_mission3.npz")

KNN_SMOOTH = 15      # 法线空间平滑: 每waypoint取15个近邻
SIGMA = 1.0          # 平滑高斯权重σ(米)
KNN_APPLY = 8        # 高斯旋正: 8近邻waypoint混合


def local_floor_normals(xyz, wp):
    """逐waypoint测局部地面法线(z∈[-0.12,0.12], r<1.5m, 3×70%修剪)。"""
    tree_pts = np.asarray(xyz)
    fl = tree_pts[(tree_pts[:, 2] > -0.12) & (tree_pts[:, 2] < 0.12)]
    tree = cKDTree(fl[:, :2])
    ns = np.zeros((len(wp), 3)); ok = np.zeros(len(wp), bool)
    for i, (sx, sy) in enumerate(wp):
        idx = tree.query_ball_point([sx, sy], 1.5)
        if len(idx) < 300:
            continue
        Q = fl[idx]
        for _ in range(3):
            c = Q.mean(0); Qc = Q - c
            _, _, Vt = np.linalg.svd(Qc, full_matrices=False)
            n = Vt[2]
            if n[2] < 0:
                n = -n
            r = np.abs(Qc @ n)
            Q = Q[r < np.percentile(r, 70)]
        ns[i] = n; ok[i] = True
    return ns, ok


def main():
    from gaussian_renderer.core.util_gau import load_ply, save_ply
    gd = load_ply(PLY_IN)
    xyz = np.asarray(gd.xyz, dtype=np.float64)
    print(f"[lvl] {xyz.shape[0]} gaussians", flush=True)

    cap = dict(np.load(NPZ))
    wp = cap["waypoints_xy"].astype(np.float64)

    # 1) 逐waypoint局部地面法线 → 最小旋正rotvec
    ns, ok = local_floor_normals(xyz, wp)
    print(f"[lvl] 局部法线有效 {ok.sum()}/{len(wp)}", flush=True)
    z = np.array([0.0, 0.0, 1.0])
    rv = np.zeros((len(wp), 3))
    for i in np.where(ok)[0]:
        n = ns[i]
        axis = np.cross(n, z)
        na = np.linalg.norm(axis)
        if na < 1e-8:
            continue
        ang = np.arccos(np.clip(n[2], -1, 1))
        rv[i] = axis / na * ang                      # n→z 的旋转 = 把世界反向扶平
    # 无效点用近邻填
    tree_wp = cKDTree(wp)
    for i in np.where(~ok)[0]:
        _, ii = tree_wp.query(wp[i], k=KNN_SMOOTH)
        rv[i] = rv[ii][ok[ii]].mean(0) if ok[ii].any() else 0.0
    # 2) 空间kNN高斯权重平滑(沿空间而非序号, 采集有回环)
    _, knn = tree_wp.query(wp, k=KNN_SMOOTH)
    d_knn = np.linalg.norm(wp[knn] - wp[:, None, :], axis=2)
    w = np.exp(-(d_knn / SIGMA) ** 2); w /= w.sum(1, keepdims=True)
    rv_s = (rv[knn] * w[:, :, None]).sum(1)
    deg = np.degrees(np.linalg.norm(rv_s, axis=1))
    print(f"[lvl] 平滑后旋正角: mean {deg.mean():.2f}° max {deg.max():.2f}°", flush=True)

    # 3) 每个高斯: 8近邻waypoint混合rotvec + 混合锚点(waypoint xy, z=0)
    _, ka = tree_wp.query(xyz[:, :2], k=KNN_APPLY)
    da = np.linalg.norm(xyz[:, None, :2] - wp[ka], axis=2)
    wa = 1.0 / (da ** 2 + 0.05); wa /= wa.sum(1, keepdims=True)
    RV = (rv_s[ka] * wa[:, :, None]).sum(1)                    # (N,3) 混合rotvec
    anch3 = np.concatenate([wp, np.zeros((len(wp), 1))], 1)    # (M,3) 锚点表
    A = (anch3[ka] * wa[:, :, None]).sum(1)                    # (N,3) 混合锚点
    print("[lvl] 旋转向量混合完成, 开始应用(N大, 稍候)...", flush=True)
    Rs = Rsc.from_rotvec(RV)
    gd.xyz = (Rs.apply(xyz - A) + A).astype(np.float32)
    # quat: wxyz → scipy xyzw; rot' = R ⊗ rot
    rq = np.asarray(gd.rot, dtype=np.float64)
    r_old = Rsc.from_quat(rq[:, [1, 2, 3, 0]])
    r_new = (Rs * r_old).as_quat()
    gd.rot = r_new[:, [3, 0, 1, 2]].astype(np.float32)
    save_ply(gd, PLY_OUT)
    print(f"[lvl] ply -> {PLY_OUT}", flush=True)

    # 4) npz: waypoints不动(锚点), yaw按自身旋正更新
    yaw = cap["yaw"].astype(np.float64).copy()
    Rs_wp = Rsc.from_rotvec(rv_s)
    fwd = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], 1)
    fwd2 = Rs_wp.apply(fwd)
    cap["yaw"] = np.arctan2(fwd2[:, 1], fwd2[:, 0])
    cap["ground_z"] = np.array(0.0)
    shutil.copy(NPZ, NPZ.replace(".npz", "_PRELVL0731.npz"))
    np.savez(NPZ, **cap)
    print(f"[lvl] npz -> {NPZ} (旧版备份_PRELVL0731)", flush=True)

    # 5) targets: 每点按最近waypoint混合旋绕锚点转
    T = json.load(open(TARGETS_IN))
    for nm in ("iron_cabinet", "microscope", "pipette_table", "origin"):
        p = np.array(T[nm], dtype=np.float64)
        _, kk = tree_wp.query(p[:2], k=KNN_APPLY)
        dd = np.linalg.norm(wp[kk] - p[:2], axis=1)
        ww = 1.0 / (dd ** 2 + 0.05); ww /= ww.sum()
        Rp = Rsc.from_rotvec((rv_s[kk] * ww[:, None]).sum(0))
        ap = np.concatenate([(wp[kk] * ww[:, None]).sum(0), [0.0]])
        p2 = Rp.apply(p - ap) + ap
        T[nm] = [round(float(v), 3) for v in p2]
        print(f"[lvl] target {nm}: {np.round(p,3)} -> {np.round(p2,3)}", flush=True)
    json.dump(T, open(TARGETS_OUT, "w"), indent=2)
    print(f"[lvl] targets -> {TARGETS_OUT}", flush=True)

    # 6) node clusters 同法
    if os.path.exists(CLUSTERS):
        cl = dict(np.load(CLUSTERS))
        for k in cl:
            P = cl[k].astype(np.float64)
            _, kk = tree_wp.query(P[:, :2], k=KNN_APPLY)
            dd = np.linalg.norm(P[:, None, :2] - wp[kk], axis=2)
            ww = 1.0 / (dd ** 2 + 0.05); ww /= ww.sum(1, keepdims=True)
            Rc = Rsc.from_rotvec((rv_s[kk] * ww[:, :, None]).sum(1))
            ac = np.concatenate([wp, np.zeros((len(wp), 1))], 1)
            Ac = (ac[kk] * ww[:, :, None]).sum(1)
            cl[k] = (Rc.apply(P - Ac) + Ac).astype(np.float32)
        shutil.copy(CLUSTERS, CLUSTERS.replace(".npz", "_PRELVL0731.npz"))
        np.savez(CLUSTERS, **cl)
        print(f"[lvl] clusters -> {CLUSTERS} (备份_PRELVL0731)", flush=True)

    # 7) 软链改指lvl版(旧bak文件保留)
    link = PLY_IN
    if os.path.islink(link):
        os.remove(link)
    elif os.path.exists(link):
        shutil.move(link, link.replace(".ply", "_PRELVL0731.ply"))
    os.symlink(os.path.basename(PLY_OUT), link)
    print(f"[lvl] 软链 {os.path.basename(link)} -> {os.path.basename(PLY_OUT)}", flush=True)


if __name__ == "__main__":
    main()
