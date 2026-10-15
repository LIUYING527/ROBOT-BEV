#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dual_locate 诊断分析 —— 把慢脑的失败拆成「编造」和「指错方向」两类。

用法: python analyze_dual.py <diag_dual.jsonl>
"""
import json
import math
import sys

import numpy as np

W, H, HFOV = 640, 480, 79.0
FX = (W / 2.0) / math.tan(math.radians(HFOV / 2.0))


def main(path):
    rows = []
    for ln in open(path, encoding='utf-8'):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if 'o_vis' in r:
            rows.append(r)
    if not rows:
        print(f'{path}: 还没有带 o_vis 的行(检查 --dual_locate 是否生效)')
        return

    print(f'===== {path} =====')
    print(f'总步数 {len(rows)}  |  ep {len(set(r["ep"] for r in rows))} 个')
    print()

    sb = [r for r in rows if r.get('vis')]
    tp = [r for r in rows if r.get('vis') and r['o_vis'] == 1]
    fp = [r for r in rows if r.get('vis') and r['o_vis'] == 0]
    fn = [r for r in rows if not r.get('vis') and r['o_vis'] == 1]
    tn = [r for r in rows if not r.get('vis') and r['o_vis'] == 0]

    print('【判据1】可见性混淆矩阵 (oracle = 参考路径终点几何投影 + 深度遮挡检查)')
    print(f'  oracle 真实可见 {len(tp) + len(fn):5d} 步  |  真实不可见 {len(fp) + len(tn):5d} 步')
    print(f'  慢脑 报可见     {len(sb):5d} 步  ({100 * len(sb) / len(rows):.1f}%)')
    print()
    print(f'  ⭐ 编造 (慢脑说可见 / 实际不可见): {len(fp):5d} 步'
          f'  = 慢脑报可见的 {100 * len(fp) / max(len(sb), 1):.1f}%')
    print(f'     漏检 (实际可见 / 慢脑说不可见): {len(fn):5d} 步'
          f'  = 真实可见的 {100 * len(fn) / max(len(tp) + len(fn), 1):.1f}%')
    if len(fp) + len(tn):
        print(f'     oracle不可见时的误报率: {100 * len(fp) / (len(fp) + len(tn)):.1f}%'
              f'   (离线同口径 = 33.7%)')
    print()

    print('【判据2】慢脑在「真实可见」时的定位精度 —— 这是慢脑的真实能力')
    if tp:
        enx = np.array([abs(r['nx'] - r['o_nx']) for r in tp])
        eny = np.array([abs(r['ny'] - r['o_ny']) for r in tp])
        lat = enx * W * np.array([r['o_dist'] for r in tp]) / FX
        ang = np.degrees(np.arctan(enx * W / FX))
        print(f'  n={len(tp)}  nx误差中位={np.median(enx):.4f}  ny误差中位={np.median(eny):.4f}')
        print(f'  横向误差中位={np.median(lat):.2f}m  方位角误差中位={np.median(ang):.1f}°')
        hit = np.hypot(enx, eny) < 0.05
        print(f'  像素命中率(0.05) = {100 * hit.mean():.1f}%')
        print(f'  (离线同口径: nx误差0.054 / ny误差0.036 / 横向0.48m / 方位角4~8° / 命中28.7%)')
    else:
        print('  无样本')
    print()

    print('【判据3】慢脑「编造」时给的是什么')
    if fp:
        d = [r['d'] for r in fp if r.get('d') is not None]
        ny = [r['ny'] for r in fp]
        od = [r['o_dist'] for r in fp]
        if d:
            print(f'  n={len(fp)}  报的位置实测深度 d: 中位={np.median(d):.2f}m')
        print(f'  报的 ny: 中位={np.median(ny):.3f}   (画面中下部 = 近处地面/家具)')
        print(f'  真实目标距离 o_dist: 中位={np.median(od):.2f}m')
    print()

    print('【判据4】按真实距离分层: 编造率 vs 距离')
    od = np.array([r['o_dist'] for r in rows])
    is_fp = np.array([bool(r.get('vis') and r['o_vis'] == 0) for r in rows])
    is_sb = np.array([bool(r.get('vis')) for r in rows])
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 99)]:
        m = (od >= lo) & (od < hi)
        if m.sum() < 10:
            continue
        sub_sb = m & is_sb
        rate = 100 * (is_fp & m).sum() / max(sub_sb.sum(), 1)
        print(f'  真实距离[{lo:2d},{hi:2d})m  n={m.sum():5d} | 慢脑报可见 {sub_sb.sum():5d}'
              f' | 其中编造 {rate:5.1f}%')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        main(p)
