#!/usr/bin/env python3
"""路径投影叠加验证 —— 把 ref_path/dense_path 投影到当前帧。

正确投影下，路径点应沿【地板】从近到远延伸(近处低、远处趋于地平线)。
旧口径(h=0)会把整条路径压在 ny=0.5 的水平线上。
"""
import os
import sys
import json

import numpy as np
from PIL import Image, ImageDraw

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from slowbrain_proj import project_goal_normalized  # noqa: E402

W, H, HFOV = 640, 480, 79


def main():
    shard = sys.argv[1] if len(sys.argv) > 1 else 's0'
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    out = sys.argv[3] if len(sys.argv) > 3 else '/tmp/path_overlay.png'

    tiles = []
    n = 0
    for line in open(os.path.join(_THIS, 'dense_data', shard, 'meta.jsonl')):
        ep = json.loads(line)
        for fi, fr in enumerate(ep['frames']):
            d = fr['dist_to_goal']
            if not (5.0 < d < 7.0):
                continue
            p = os.path.join(_THIS, 'dense_data', shard, f"ep{ep['episode_id']}", f"f{fr['vi']:03d}.npz")
            if not os.path.exists(p):
                continue
            pos = np.array(fr['pos'], np.float64)
            rot = np.array(fr['rot'], np.float64)
            path = np.array(ep['dense_path'] if 'dense_path' in ep else ep['ref_path'], np.float64)

            rgb = np.load(p)['rgb']
            for tag, h in (('OLD', 0.0), ('NEW', 1.30)):
                im = Image.fromarray(rgb.copy())
                dr = ImageDraw.Draw(im)
                pc = pos.copy(); pc[1] += h
                pts = []
                for q in path:
                    nx, ny, vis = project_goal_normalized(q, pc, rot, W, H, HFOV)
                    pts.append((nx * W, ny * H) if vis else None)
                # 画折线(仅可见段)
                prev = None
                for pt in pts:
                    if pt is None:
                        prev = None
                        continue
                    if prev is not None:
                        dr.line([prev, pt], fill=(255, 60, 60) if tag == 'NEW' else (0, 170, 255), width=3)
                    prev = pt
                # 终点(goal)
                gx, gy, gv = project_goal_normalized(np.array(ep['goal'], np.float64), pc, rot, W, H, HFOV)
                if gv:
                    dr.ellipse([gx * W - 8, gy * H - 8, gx * W + 8, gy * H + 8],
                               outline=(255, 0, 255), width=3)
                dr.text((6, 6), f'{tag}  d={d:.1f}m  {"RED=NEW(h=1.3) ORANGE=OLD(h=0)" if tag=="NEW" else ""}',
                        fill=(255, 255, 0))
                tiles.append(np.array(im))
            n += 1
            break
        if n >= n_eps or len(tiles) >= 4:
            break

    if tiles:
        # 取前 2 组(OLD/NEW 成对)上下拼
        tiles = tiles[:4]
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]) if len(tiles) >= 4 else np.hstack(tiles)
        Image.fromarray(grid).save(out)
        print(f'-> {out}')


if __name__ == '__main__':
    main()
