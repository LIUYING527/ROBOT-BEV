#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""投影参数联合标定 —— 用「投影像素处深度 ≈ 目标真实距离」作判据，精调 (相机高度 h, 地平线偏移 v0_off, hfov)。

动机：h=1.30 时中位误差仍有 0.41m，怀疑还有 hfov / pitch 系统误差未消。
做法：网格搜索，在近距帧(目标必可见)上最小化 |depth_at_projected_pixel - dist_to_goal|。
"""
import os
import sys
import json
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from slowbrain_proj import quat_to_rot  # noqa: E402


def project(goal, pos, quat, W, H, hfov, v0_off=0.0):
    R = quat_to_rot(quat)
    cam = R.T @ (goal - pos)
    if cam[2] >= 0:
        return None
    fx = (W / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    u = W / 2.0 + fx * cam[0] / (-cam[2])
    v = H / 2.0 + v0_off - fx * cam[1] / (-cam[2])
    if not (0 <= u < W and 0 <= v < H):
        return None
    return u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', default='s0')
    ap.add_argument('--max_eps', type=int, default=120)
    ap.add_argument('--near', type=float, default=4.0, help='只用 dist_to_goal 小于此值的帧')
    cli = ap.parse_args()
    W, H = 640, 480

    # 预载近距帧(避免重复读盘)
    frames = []
    n = 0
    for line in open(os.path.join(_THIS, 'dense_data', cli.shard, 'meta.jsonl')):
        ep = json.loads(line)
        goal = np.array(ep['goal'], np.float64)
        for fr in ep['frames']:
            d = fr['dist_to_goal']
            if d > cli.near or d < 1.2:
                continue
            p = os.path.join(_THIS, 'dense_data', cli.shard, f"ep{ep['episode_id']}", f"f{fr['vi']:03d}.npz")
            if not os.path.exists(p):
                continue
            frames.append((goal, np.array(fr['pos'], np.float64), np.array(fr['rot'], np.float64), d, p))
        n += 1
        if n >= cli.max_eps:
            break
    print(f'近距帧 {len(frames)} (来自 {n} eps)', flush=True)

    cache = {}
    def depth_of(path):
        if path not in cache:
            cache[path] = np.load(path)['depth']
            if len(cache) > 400:
                cache.pop(next(iter(cache)))
        return cache[path]

    results = []
    for h in [1.15, 1.20, 1.25, 1.30, 1.35]:
        for v0off in [-30, -15, 0, 15, 30]:
            for hfov in [72, 76, 79, 82, 86]:
                errs = []
                for goal, pos, rot, d_true, path in frames:
                    p = pos.copy(); p[1] += h
                    r = project(goal, p, rot, W, H, hfov, v0off)
                    if r is None:
                        continue
                    u, v = r
                    dep = depth_of(path)
                    ui, vi = int(round(u)), int(round(v))
                    patch = dep[max(0, vi - 3):vi + 4, max(0, ui - 3):ui + 4]
                    patch = patch[patch > 0]
                    if patch.size < 5:
                        continue
                    errs.append(abs(float(np.median(patch)) - d_true))
                if errs:
                    results.append((float(np.median(errs)), h, v0off, hfov, len(errs)))

    results.sort()
    print('\n==== 联合标定 top-12 (中位误差从小到大) ====')
    print(f"{'误差(m)':>9} {'h':>6} {'v0_off':>7} {'hfov':>6} {'n':>6}")
    for e, h, v, f, n in results[:12]:
        print(f'{e:9.3f} {h:6.2f} {v:7.0f} {f:6.0f} {n:6d}')
    print(f'\n>>> 最优: h={results[0][1]}, v0_off={results[0][2]}, hfov={results[0][3]}, 误差={results[0][0]:.3f}m')


if __name__ == '__main__':
    main()
