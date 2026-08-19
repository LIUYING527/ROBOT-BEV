"""burst-RELOC 验证(真zed数据): 同一扰动Δ下, 相邻3帧各自reloc→估计Δ取平均。
内容相关偏差跨视角部分去相关 → 平均应压过单帧噪声底(5.6cm→目标2~3cm)。
真实对应: 机器人到站小角度转动采多帧(里程计短时准确=Δ共享)。
用法: CUDA_VISIBLE_DEVICES=4 ~/discoverse_venv/bin/python scripts/gs_reloc_burst.py
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gs_relocate import GSReloc


def main():
    g = GSReloc("colmapjoint_110744s2", "data/lab/110744")
    rng = np.random.RandomState(1)
    centers = list(range(30, 480, 40))          # 12个位置
    rows = []
    for j in centers:
        triple = [j - 1, j, j + 1]
        # 共享扰动Δ(模拟: 短时里程计准, 绝对位姿带误差)
        a = rng.uniform(0, 2 * np.pi); rr = 0.3 * np.sqrt(rng.uniform(0, 1))
        dx, dy = rr * np.cos(a), rr * np.sin(a)
        dyaw = np.radians(rng.uniform(-15, 15))
        c, s = np.cos(dyaw), np.sin(dyaw)
        Rp = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
        errs, yerrs = [], []
        ok = True
        for k in triple:
            C_gt, R_gt = g.gt_world_pose(k)
            C0 = C_gt + np.array([dx, dy, 0.0]); R0 = Rp @ R_gt
            C1, R1, tr = g.optimize(k, C0, R0)
            if C1 is None:
                ok = False; break
            errs.append(C1 - C_gt)
            yerrs.append(abs(np.degrees(np.arctan2(R1[1, 2], R1[0, 2]) - np.arctan2(R_gt[1, 2], R_gt[0, 2]))))
        if not ok:
            continue
        E = np.stack(errs)
        single = np.median(np.linalg.norm(E[:, :2], axis=1))          # 单帧水平误差(中位)
        avg = np.linalg.norm(E.mean(0)[:2])                            # 平均后水平误差
        yaw_single = float(np.median(yerrs)); yaw_avg = abs(np.mean(yerrs) * 0 + np.mean(yerrs))  # yaw无符号信息, 只报单帧
        rows.append((j, single, avg, yaw_single))
        print(f"pos#{j:3d}: 单帧水平中位 {single*100:5.1f}cm → burst3平均 {avg*100:5.1f}cm   (yaw单帧{yaw_single:.2f}°)", flush=True)
    ss = np.array([r[1] for r in rows]); aa = np.array([r[2] for r in rows])
    print(f"\n===== 汇总(n={len(rows)}) =====")
    print(f"单帧: 水平中位 {np.median(ss)*100:.1f}cm  p90 {np.percentile(ss,90)*100:.1f}")
    print(f"burst3平均: 水平中位 {np.median(aa)*100:.1f}cm  p90 {np.percentile(aa,90)*100:.1f}")


if __name__ == "__main__":
    main()
