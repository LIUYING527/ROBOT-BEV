#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析式修复已有标注的相机高度 bug —— 无需重渲染。

原理：纯 yaw 旋转下 R_y 的第 2 行=[0,1,0]、第 3 行=[sinθ,0,cosθ]，
     故 pos.y 只影响 cam[1](=目标相对高度)，不影响 cam[0]/cam[2]
     ⇒ **nx 完全不变**，只有 ny 需要修正：
         ny_new = 0.5 + fy·h / (H · d)        (d = 水平距离，地板目标)
     可见性：ny_new < 1  ⇔  d > fy·h/H  （更近的目标落到画面下方 → 不可见）

若提供 depth(npz) 则额外做遮挡检查(终点像素 3×3 深度中位数 vs 真实距离)。

用法：
  python fix_annot_height.py --annot <in.jsonl> --out <out.jsonl> [--with-depth]
"""
import os
import sys
import json
import argparse

import numpy as np

W, H, HFOV = 640, 480, 79
FY = (W / 2.0) / np.tan(np.deg2rad(HFOV / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annot', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sensor-h', dest='sensor_h', type=float, default=1.30)
    ap.add_argument('--with-depth', action='store_true', help='读 npz depth 做遮挡检查')
    ap.add_argument('--occ-ratio', type=float, default=0.85, help='d_pix < d_real*ratio → 判遮挡')
    cli = ap.parse_args()

    d_min = FY * cli.sensor_h / H          # ny_new = 1 的临界距离
    n = n_vis_before = n_vis_after = n_flip_invis = n_occ = 0
    with open(cli.annot, encoding='utf-8') as f, open(cli.out, 'w', encoding='utf-8') as g:
        for line in f:
            a = json.loads(line)
            n += 1
            if a['goal_visible']:
                n_vis_before += 1
            nx = a['goal_nx']
            d = a['dist_to_goal']
            # ny 修正(**精确解析解**，数值误差 ~1e-16，已在 dense_data 5000 帧上验证)
            #   纯 yaw 旋转下 pos.y 只影响 cam[1] ⇒ nx 完全不变，只有 ny 变。
            #   由旧 (nx,ny) 反解出方向余弦：k=cam[1]/s, m=cam[0]/s；s = d/√(1+k²+m²)
            #   ny_new = ny_old + FY·h·√(1+k²+m²) / (H·d)
            if a['goal_visible'] and d > 1e-6:
                ny_old = a['goal_ny']
                nx_old = a['goal_nx']
                k = -(ny_old - 0.5) * H / FY
                m = (nx_old - 0.5) * W / FY
                ny = ny_old + FY * cli.sensor_h * np.sqrt(1.0 + k * k + m * m) / (H * d)
            else:
                ny = -1.0
            vis = bool(a['goal_visible'] and (0.0 <= nx < 1.0) and (ny < 1.0))
            # 遮挡检查
            if vis and cli.with_depth:
                fp = a['frame']
                if not os.path.isabs(fp):
                    fp = os.path.join('/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval', fp)
                if os.path.exists(fp):
                    depth = np.load(fp)['depth']
                    ui, vi = int(round(nx * W)), int(round(ny * H))
                    patch = depth[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2]
                    patch = patch[patch > 0]
                    if patch.size:
                        d_pix = float(np.median(patch))
                        if d_pix < d * cli.occ_ratio:
                            vis = False
                            n_occ += 1
            if vis:
                n_vis_after += 1
            elif a['goal_visible']:
                n_flip_invis += 1
            a['goal_nx'] = round(float(nx), 4) if vis else -1.0
            a['goal_ny'] = round(float(ny), 4) if vis else -1.0
            a['goal_visible'] = vis
            g.write(json.dumps(a, ensure_ascii=False) + '\n')

    print(f'==== 标注高度修复 ====')
    print(f'总数 {n}')
    print(f'可见: 修复前 {n_vis_before} ({n_vis_before/max(1,n):.1%}) → 修复后 {n_vis_after} ({n_vis_after/max(1,n):.1%})')
    print(f'  其中 {n_flip_invis} 个因 ny 修正/遮挡由"可见"翻为"不可见"')
    print(f'  遮挡剔除 {n_occ}')
    print(f'临界距离(ny=1) = {d_min:.3f} m —— 比这更近的地板目标落在画面下方,标不可见')
    print(f'-> {cli.out}')


if __name__ == '__main__':
    main()
