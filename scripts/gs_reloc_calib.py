"""两参数深度标定: ed = s*d_zed + o 全局联合拟合(汇集多帧GT位姿残差)。
输出最优(s,o)供 gs_relocate OBS_SCALE/OBS_OFFSET 使用。
用法: CUDA_VISIBLE_DEVICES=5 ~/discoverse_venv/bin/python scripts/gs_reloc_calib.py
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import gs_relocate as gr
from gs_relocate import GSReloc, DEV

gr.GSReloc.__init__.__defaults__  # noop


def main():
    g = GSReloc("colmapjoint_110744s2", "data/lab/110744")
    g.obs_scale = 1.0                     # 原始zed米, 拟合(s,o)
    Ho, Wo = 360, 640
    frames = list(range(10, 500, 25))
    D, E = [], []
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
        if m.sum() < 3000:
            continue
        d = d_obs[m].cpu().numpy(); e = ed[m].cpu().numpy()
        keep = np.random.RandomState(j).choice(len(d), min(8000, len(d)), replace=False)
        D.append(d[keep]); E.append(e[keep])
    D = np.concatenate(D); E = np.concatenate(E)
    print(f"pooled {len(D)} px from {len(D)//8000+1} frames")
    # 稳健迭代: 最小化 |e - (s*d+o)|, 两轮huber-ish裁剪
    s, o = 1.0, 0.0
    for it in range(4):
        r = E - (s * D + o)
        lo, hi = np.percentile(r, [10, 90])
        m = (r > lo) & (r < hi)
        A = np.stack([D[m], np.ones(m.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, E[m], rcond=None)
        s, o = float(sol[0]), float(sol[1])
        print(f"  it{it}: s={s:.4f} o={o:.4f}  inlier{r[m].mean():+.4f}±{r[m].std():.4f}m")
    print(f"\n>>> OBS_SCALE={s:.3f} OBS_OFFSET={o:.3f}")
    print("判读: o显著>0=ED期望深度在首面之后(GS半透明混合固有); s即尺度校正")


if __name__ == "__main__":
    main()
