"""场景图节点位置精化 = 多视三角化 (0730, 用户"crops这么好为啥不用")。

原理: 每个节点存了建图时的全部检测(image_idx/xyxy/conf, 即外观卡片crops的来源)——
每个检测的bbox中心射线(该帧c2w + 内参)从世界不同位置射向目标, 最小二乘求交+外点修剪
= 节点精确3D位置。比过合并点簇的bbox中心/质心准(实测vs zed GT: 显微镜50→28cm, 白桌88→38cm;
铁柜57cm是"物体中心vs GT表面点"的参照差异非误差)。射线中位残差同时是节点质量诊断
(过合并杂簇的射线不交于一点)。

用法: ~/micromamba/envs/cg/bin/python scripts/cg_triangulate_nodes.py
产物: outputs/lab_mission_targets_rag_tri.json (可直接喂 lab_mission.py --targets)
"""
import gzip, pickle, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG = os.path.join(ROOT, "outputs", "cg_lab110744s2", "lab1")
NODES = {"iron_cabinet": 130, "microscope": 148, "pipette_table": 20}
FX, FY, CX, CY = 520.0802, 520.0802, 640.0, 360.0   # dataconfig lab110744s2.yaml


def triangulate(o, traj, conf_min=0.25, trim=0.15):
    rays = []
    for fi, bb, cf in zip(o["image_idx"], o["xyxy"], o["conf"]):
        if cf < conf_min:
            continue
        c2w = traj[fi * 2]                        # mapping stride=2 → traj行=idx*2
        u, v = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        d = np.array([(u - CX) / FX, (v - CY) / FY, 1.0]); d /= np.linalg.norm(d)
        rays.append((c2w[:3, 3], c2w[:3, :3] @ d))
    if len(rays) < 3:
        return None, 0, None
    def ls_int(rr):
        A = np.zeros((3, 3)); b = np.zeros(3)
        for o_, d_ in rr:
            P = np.eye(3) - np.outer(d_, d_); A += P; b += P @ o_
        return np.linalg.solve(A, b)
    p = ls_int(rays)
    keep = [r for r in rays if np.linalg.norm(np.cross(p - r[0], r[1])) < trim]
    if len(keep) >= 3:
        p = ls_int(keep)
    else:
        keep = rays
    resid = float(np.median([np.linalg.norm(np.cross(p - r[0], r[1])) for r in keep]))
    return p, len(keep), resid


def front_yaw(o, traj, topk=5, conf_min=0.25):
    """⭐目标正面方向 = 最大检测(pixel_area topk)的bbox中心射线方位加权圆均值。
    检测器/SAM在正脸视角给最大最稳的框 → 该方向即"正对"视角(0730实测: 铁柜-134°门冲东北,
    和墙法线180°差46°——只打目标点不保证正脸, 必须显式取正面)。返回(front_deg, 明细list)。"""
    cands = [(j, o["pixel_area"][j]) for j in range(len(o["image_idx"])) if o["conf"][j] >= conf_min]
    cands = sorted(cands, key=lambda t: -t[1])[:topk]
    ys, ws = [], []
    for j, a in cands:
        c2w = traj[o["image_idx"][j] * 2]
        bb = o["xyxy"][j]
        u = (bb[0] + bb[2]) / 2
        d = np.array([(u - CX) / FX, 0.0, 1.0]); d /= np.linalg.norm(d)
        w = c2w[:3, :3] @ d
        yaw = np.arctan2(w[1], w[0])
        ys.append(yaw); ws.append(float(a) * float(o["conf"][j]))
    ys = np.array(ys); ws = np.array(ws)
    ang = np.arctan2((np.sin(ys) * ws).sum(), (np.cos(ys) * ws).sum())
    return float(np.degrees(ang)), [(round(float(np.degrees(y)), 1), round(a, 0)) for y, a in zip(ys, ws)]


def main():
    d = pickle.load(gzip.open(os.path.join(CG, "pcd_saves", "full_pcd_ram_cfslam_post.pkl.gz"), "rb"))
    objs = d["objects"]
    traj = np.loadtxt(os.path.join(CG, "traj.txt")).reshape(-1, 4, 4)
    out = {"origin": [0.0, 0.0, 0.0]}
    fronts = {}
    for name, idx in NODES.items():
        p, nray, resid = triangulate(objs[idx], traj)
        assert p is not None, f"{name}: 射线不足"
        out[name] = [round(float(v), 3) for v in p]
        fdeg, detail = front_yaw(objs[idx], traj)
        fronts[name] = round(fdeg, 1)
        print(f"{name} node{idx}: 射线{nray}条 中位残差{resid*100:.1f}cm -> {out[name]} 正面={fdeg:.0f}° (top5射线{detail})", flush=True)
    out["front_deg"] = fronts
    dst = os.path.join(ROOT, "outputs", "lab_mission_targets_rag_tri.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"-> {dst}", flush=True)


if __name__ == "__main__":
    main()
