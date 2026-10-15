#!/usr/bin/env python3
"""相机高度精标定(高效版) —— 每帧只读一次深度，扫 h 使「投影像素处深度 ≈ 目标真实距离」。

判据：goal 在地板上，相机水平 → 目标像素的平面深度 = 目标水平距离 = dist_to_goal。
"""
import os
import sys
import json
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from slowbrain_proj import quat_to_rot  # noqa: E402

W, H = 640, 480


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', default='s0')
    ap.add_argument('--max_frames', type=int, default=2500)
    ap.add_argument('--near', type=float, default=5.0)
    cli = ap.parse_args()

    hs = np.arange(1.18, 1.43, 0.02)
    errs = {float(h): [] for h in hs}
    hfov = 79.0
    fx = (W / 2.0) / np.tan(np.deg2rad(hfov / 2.0))

    nf = 0
    for line in open(os.path.join(_THIS, 'dense_data', cli.shard, 'meta.jsonl')):
        ep = json.loads(line)
        goal = np.array(ep['goal'], np.float64)
        for fr in ep['frames']:
            d = fr['dist_to_goal']
            if not (1.5 < d < cli.near):
                continue
            p = os.path.join(_THIS, 'dense_data', cli.shard, f"ep{ep['episode_id']}", f"f{fr['vi']:03d}.npz")
            if not os.path.exists(p):
                continue
            depth = np.load(p)['depth']
            pos = np.array(fr['pos'], np.float64)
            rot = np.array(fr['rot'], np.float64)
            R = quat_to_rot(rot)
            for h in hs:
                pc = pos.copy(); pc[1] += h
                cam = R.T @ (goal - pc)
                if cam[2] >= 0:
                    continue
                u = W / 2.0 + fx * cam[0] / (-cam[2])
                v = H / 2.0 - fx * cam[1] / (-cam[2])
                if not (0 <= u < W and 0 <= v < H):
                    continue
                ui, vi = int(round(u)), int(round(v))
                patch = depth[max(0, vi - 3):vi + 4, max(0, ui - 3):ui + 4]
                patch = patch[patch > 0]
                if patch.size < 5:
                    continue
                errs[float(h)].append(abs(float(np.median(patch)) - d))
            nf += 1
            if nf >= cli.max_frames:
                break
        if nf >= cli.max_frames:
            break

    print(f'{"h":>6} {"中位误差":>10} {"均值":>8} {"P90":>8} {"n":>7}')
    best = None
    for h in hs:
        e = np.array(errs[float(h)])
        if e.size == 0:
            continue
        med = float(np.median(e))
        print(f'{h:6.2f} {med:10.3f} {e.mean():8.3f} {np.percentile(e,90):8.3f} {e.size:7d}')
        if best is None or med < best[1]:
            best = (h, med)
    print(f'\n>>> 最优 h = {best[0]:.2f} m (中位误差 {best[1]:.3f} m)')


if __name__ == '__main__':
    main()
