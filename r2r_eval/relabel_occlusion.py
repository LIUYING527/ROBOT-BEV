#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度遮挡重标 —— 修正 slowbrain_proj 的「几何可见」标签缺陷。

问题：slowbrain_proj.py 判断可见只看投影是否落在 FOV 内，不检查遮挡 →
     「终点在 FOV 内但被墙挡」帧被错标成可见 → 模型学出「不确定也吐坐标」→ R2R 44% 误报。
方案（对齐 DualVLN ICLR2026 的 depth-based occlusion filtering）：
     对几何可见的帧，采样终点像素 3×3 邻域深度中位数 d_pix，
     与终点相机系轴向深度 d_real 比较：d_pix < d_real × ratio → 判为遮挡 → 改标不可见。

用法：
  # 1) 统计模式：看 d_pix/d_real 分布，扫阈值选最优
  python relabel_occlusion.py --mode stats --shards s0 --max_eps 200
  # 2) 写出模式：生成重标后的 annot jsonl
  python relabel_occlusion.py --mode write --out slowbrain_annot_occlusion.jsonl
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from slowbrain_proj import quat_to_rot, world_to_cam  # noqa: E402

W, H, HFOV = 640, 480, 79
DENSE_DIR = os.path.join(_THIS, 'dense_data')


def project_with_depth(goal, pos, rot, depth, margin=0.0):
    """返回 (nx, ny, geom_visible, d_pix, d_real)。

    d_real = 终点在相机系的轴向深度(与 habitat depth 同口径)。
    d_pix  = 终点像素 3×3 邻域深度中位数(该像素实际看到的表面距离)。
    """
    cam = world_to_cam(goal, pos, rot)
    zc = cam[2]
    if zc >= 0.0:                                   # 相机后方
        return -1.0, -1.0, False, None, None
    fx = (W / 2.0) / np.tan(np.deg2rad(HFOV / 2.0))
    u = W / 2.0 + fx * cam[0] / (-zc)
    v = H / 2.0 - fx * cam[1] / (-zc)
    if not (margin <= u < W - margin and margin <= v < H - margin):
        return -1.0, -1.0, False, None, None        # FOV 外
    d_real = -zc
    ui, vi = int(round(u)), int(round(v))
    patch = depth[max(0, vi - 1):vi + 2, max(0, ui - 1):ui + 2]
    patch = patch[patch > 0]
    d_pix = float(np.median(patch)) if patch.size else None
    return u / W, v / H, True, d_pix, d_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='stats', choices=['stats', 'write'])
    ap.add_argument('--shards', default='s0', help='逗号分隔的 shard（stats 用），write 模式默认全部')
    ap.add_argument('--ratios', default='0.75,0.85,0.9,0.95', help='遮挡判定阈值扫描')
    ap.add_argument('--ratio', type=float, default=0.85, help='write 模式采用的阈值')
    ap.add_argument('--max_eps', type=int, default=0, help='每 shard 最多 episode 数(0=全部)')
    ap.add_argument('--out', default='slowbrain_annot_occlusion.jsonl')
    ap.add_argument('--scene-filter', default='train', choices=['train', 'dev', 'all'])
    ap.add_argument('--sensor-h', dest='sensor_h', type=float, default=1.30,
                    help='相机在 agent 原点上方的高度(m)，必须与 gen_slowbrain_from_dense.py 一致')
    cli = ap.parse_args()

    ratios = [float(r) for r in cli.ratios.split(',')]

    # 场景 split（避免泄漏）
    split_map = '/data/DongBaorong/BEVLOOK/benchmarks/mp3d_r2r/scene_split.json'
    allow = None
    if os.path.exists(split_map):
        sm = json.load(open(split_map))
        if cli.scene_filter == 'train':
            allow = set(sm['train_scenes'])
        elif cli.scene_filter == 'dev':
            allow = set(sm['dev_scenes'])

    shards = ([s.strip() for s in cli.shards.split(',')] if cli.mode == 'stats'
              else sorted(os.path.basename(p) for p in glob.glob(os.path.join(DENSE_DIR, 's*'))))

    fout = open(cli.out, 'w', encoding='utf-8') if cli.mode == 'write' else None

    # 统计量
    n_total = n_geom_invis = n_geom_vis = 0
    ratio_hist = []                      # 几何可见帧的 d_pix/d_real
    occ_counts = {r: 0 for r in ratios}

    for shard in shards:
        meta_path = os.path.join(DENSE_DIR, shard, 'meta.jsonl')
        if not os.path.exists(meta_path):
            continue
        n_eps = 0
        for line in open(meta_path, encoding='utf-8'):
            ep = json.loads(line)
            if allow is not None and ep['scene'] not in allow:
                continue
            goal = np.array(ep['goal'], np.float64)
            for fr in ep['frames']:
                npz = os.path.join(DENSE_DIR, shard, f"ep{ep['episode_id']}", f"f{fr['vi']:03d}.npz")
                if not os.path.exists(npz):
                    continue
                pos = np.array(fr['pos'], np.float64)
                pos[1] += cli.sensor_h          # ⭐相机在 agent 原点上方 sensor_h 米
                rot = np.array(fr['rot'], np.float64)
                depth = np.load(npz)['depth']
                nx, ny, geom_vis, d_pix, d_real = project_with_depth(goal, pos, rot, depth)
                n_total += 1
                if not geom_vis:
                    n_geom_invis += 1
                    continue
                n_geom_vis += 1
                if d_pix is not None and d_real and d_real > 1e-6:
                    r = d_pix / d_real
                    ratio_hist.append(r)
                    for rr in ratios:
                        if r < rr:
                            occ_counts[rr] += 1
                if cli.mode == 'write':
                    occluded = (d_pix is not None and d_real and d_pix < d_real * cli.ratio)
                    vis = not occluded
                    fout.write(json.dumps({
                        'episode_id': ep['episode_id'], 'scene': ep['scene'],
                        'instruction': ep['instruction'],
                        'frame': os.path.join(shard, f"ep{ep['episode_id']}", f"f{fr['vi']:03d}.npz"),
                        'goal_nx': round(float(nx), 4) if vis else -1.0,
                        'goal_ny': round(float(ny), 4) if vis else -1.0,
                        'goal_visible': bool(vis),
                        'dist_to_goal': fr['dist_to_goal'],
                    }, ensure_ascii=False) + '\n')
            n_eps += 1
            if cli.max_eps and n_eps >= cli.max_eps:
                break
            if n_total % 2000 == 0:
                print(f'  [{shard}] {n_total} frames...', flush=True)

    print(f'\n==== 深度遮挡重标统计 ====')
    print(f'总帧 {n_total} | 几何不可见 {n_geom_invis} ({n_geom_invis/max(1,n_total)*100:.1f}%) '
          f'| 几何可见 {n_geom_vis} ({n_geom_vis/max(1,n_total)*100:.1f}%)')
    if ratio_hist:
        rh = np.array(ratio_hist)
        print(f'\n几何可见帧的 d_pix/d_real 分布(n={len(rh)}):')
        for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            print(f'  P{q:02d} = {np.percentile(rh, q):.3f}')
        print(f'\n按阈值判定为「实际被遮挡」的几何可见帧:')
        for rr in ratios:
            c = occ_counts[rr]
            print(f'  ratio<{rr}: {c}/{n_geom_vis} = {c/max(1,n_geom_vis)*100:.1f}%  '
                  f'(全量占比 {c/max(1,n_total)*100:.1f}%)')
        print(f'\n→ 重标后预计不可见占比: '
              f'{(n_geom_invis + occ_counts[cli.ratio])/max(1,n_total)*100:.1f}% '
              f'(ratio={cli.ratio})')
    if fout:
        fout.close()
        print(f'\n写出: {cli.out}')


if __name__ == '__main__':
    main()
