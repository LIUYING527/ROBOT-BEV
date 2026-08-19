"""GS重定位深度偏差三假设鉴别诊断 (0805断点续)。
在GT位姿(零扰动)下, 对N帧计算 r = render_ED - zed_obs:
  假设1(参考系常数平移): r≈常数a, 与深度无关 → b≈0
  假设2(全局尺度误差):   r∝深度 → b≠0且跨帧一致; a≈0
  假设3(ED期望偏差):     局部结构相关, a,b都不稳
独立判据: zed点云直接测地板→相机高度 h_zed, 与align假设的cam_height=1.2对比。
  尺度比 s = h_zed/1.2; 若s≠1则渲染深度(世界米)=真深度/s → 预测 b ≈ (1/s - 1)。
用法: CUDA_VISIBLE_DEVICES=4 ~/discoverse_venv/bin/python scripts/gs_reloc_diag.py
"""
import os, sys, glob
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gs_relocate import GSReloc, DEV


def fit_ab(r, d):
    """稳健线性拟合 r = a + b*d (两次sigma裁剪)。"""
    m = np.isfinite(r) & np.isfinite(d)
    r, d = r[m], d[m]
    for _ in range(2):
        b, a = np.polyfit(d, r, 1)
        res = r - (a + b * d)
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        keep = np.abs(res - np.median(res)) < 3 * s
        r, d = r[keep], d[keep]
    b, a = np.polyfit(d, r, 1)
    return a, b


def floor_cam_height(npz_path):
    """zed点云(IMAGE系: x右 y下 z前)拟合地板平面 → 相机离地高度(米)。"""
    d = np.load(npz_path)["xyzrgba"][..., :3].astype(np.float32) / 1000.0  # (720,1280,3)
    H, W = d.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    # 下半幅 + 深度合理 的候选地板点
    m = (yy > 0.55 * H) & (d[..., 2] > 0.5) & (d[..., 2] < 5.0) & np.isfinite(d[..., 0])
    P = d[m]
    if len(P) < 2000:
        return None, 0
    sub = P[np.random.RandomState(0).choice(len(P), min(20000, len(P)), replace=False)]
    # 迭代平面拟合 + inlier裁剪
    inl = np.ones(len(sub), bool)
    for _ in range(3):
        Q = sub[inl]
        c = Q.mean(0)
        U, S, Vt = np.linalg.svd(Q - c, full_matrices=False)
        n = Vt[2]
        if n[1] < 0:
            n = -n                      # 法线朝下(y正)
        res = np.abs((sub - c) @ n)
        inl = res < 0.03
    Q = sub[inl]; c = Q.mean(0)
    U, S, Vt = np.linalg.svd(Q - c, full_matrices=False)
    n = Vt[2]
    if n[1] < 0:
        n = -n
    h = float(abs(c @ n))               # 原点到平面距离 = 相机高度
    tilt = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))  # 法线 vs 正y(含相机俯仰)
    return h, tilt


def main():
    g = GSReloc("colmapjoint_110744s2", "data/lab/110744")
    Ho, Wo = 360, 640
    frames = [20, 60, 100, 150, 200, 250, 300, 350, 400, 450]
    print(f"{'帧':>5} {'n':>7} {'med_r':>8} {'a':>8} {'b':>8} {'h_zed':>7} {'tilt°':>6}")
    bs, hs = [], []
    for j in frames:
        if j >= len(g.names):
            continue
        d_obs, m_obs = g.obs_depth_m(g.names[j], Ho, Wo)
        if d_obs is None or m_obs.sum() < 3000:
            continue
        C, R = g.gt_world_pose(j)
        V = g.viewmat_from_world(torch.tensor(C, device=DEV, dtype=torch.float32),
                                 torch.tensor(R, device=DEV, dtype=torch.float32))
        with torch.no_grad():
            ed, al = g.render_ed(V, Ho, Wo)
        m = m_obs & (al > 0.6) & (ed > 0.05) & (ed < 15.0)
        r = (ed - d_obs)[m].cpu().numpy()
        dd = d_obs[m].cpu().numpy()
        # 尾裁剪(去floater/混深)
        lo, hi = np.percentile(r, [5, 95])
        keep = (r > lo) & (r < hi)
        a, b = fit_ab(r[keep], dd[keep])
        npz = os.path.join(g.raw_pc, g.names[j].replace(".jpg", ".npz"))
        h, tilt = floor_cam_height(npz)
        bs.append(b)
        if h:
            hs.append(h)
        print(f"{j:>5} {int(m.sum()):>7} {np.median(r):>8.3f} {a:>8.3f} {b:>8.3f} "
              f"{(h if h else float('nan')):>7.3f} {(tilt if h else float('nan')):>6.1f}", flush=True)
    bs = np.array(bs); hs = np.array(hs)
    print("\n===== 鉴别 =====")
    print(f"b(残差~深度斜率): 中位 {np.median(bs):.4f}  均值 {bs.mean():.4f}  散布 {bs.std():.4f}")
    if len(hs):
        h_med = float(np.median(hs))
        s = h_med / 1.2
        print(f"zed地板测相机高: 中位 {h_med:.3f}m (vs align假设1.2m) → 尺度比 s={s:.4f}")
        print(f"若纯尺度误差, 预测 b ≈ (1/s - 1) = {1/s - 1:.4f}")
    print("判读: |b|>>0且跨帧一致+与预测吻合→假设2(尺度); b≈0而a跨帧变→假设1(参考系); 都乱→假设3(ED)")


if __name__ == "__main__":
    main()
