#!/usr/bin/env python3
"""投影可视化验证(修复版) —— 把 goal 投影点画在真实帧上，对比 h=0(旧) vs h=1.30(新)。

红叉=投影目标位置；绿点=画面中心。若投影正确，红叉应落在目标(地板)位置上。
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


def draw(img, x, y, color, size=14, width=3):
    d = ImageDraw.Draw(img)
    d.line([(x - size, y), (x + size, y)], fill=color, width=width)
    d.line([(x, y - size), (x, y + size)], fill=color, width=width)


def main():
    shard = sys.argv[1] if len(sys.argv) > 1 else 's0'
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out = sys.argv[3] if len(sys.argv) > 3 else '/tmp/proj_check_fix.png'

    rows = []
    n = 0
    for line in open(os.path.join(_THIS, 'dense_data', shard, 'meta.jsonl')):
        ep = json.loads(line)
        goal = np.array(ep['goal'], np.float64)
        for fr in ep['frames']:
            d = fr['dist_to_goal']
            if not (2.0 < d < 6.0):            # 中距: 目标应清晰可见
                continue
            p = os.path.join(_THIS, 'dense_data', shard, f"ep{ep['episode_id']}", f"f{fr['vi']:03d}.npz")
            if not os.path.exists(p):
                continue
            rows.append((p, goal, np.array(fr['pos'], np.float64), np.array(fr['rot'], np.float64), d))
            break
        n += 1
        if n >= n_eps or len(rows) >= 4:
            break

    tiles = []
    for path, goal, pos, rot, d in rows:
        rgb = np.load(path)['rgb']
        im = Image.fromarray(rgb.copy())
        # 旧口径 h=0
        nx0, ny0, v0 = project_goal_normalized(goal, pos, rot, W, H, HFOV)
        if v0:
            draw(im, nx0 * W, ny0 * H, (0, 160, 255), 16, 3)      # 橙色 = 旧
        # 新口径 h=1.30
        pc = pos.copy(); pc[1] += 1.30
        nx1, ny1, v1 = project_goal_normalized(goal, pc, rot, W, H, HFOV)
        if v1:
            draw(im, nx1 * W, ny1 * H, (255, 0, 0), 10, 3)        # 红 = 新
        draw(im, W // 2, H // 2, (0, 255, 0), 6, 1)               # 绿 = 中心
        d2 = ImageDraw.Draw(im)
        d2.text((6, 6), f'd={d:.1f}m  OLD(orange) ny={ny0:.3f}  NEW(red) ny={ny1:.3f}',
                fill=(255, 255, 0))
        tiles.append(np.array(im))
        print(f'{os.path.basename(path)}: d={d:.2f} OLD=({nx0:.3f},{ny0:.3f}) NEW=({nx1:.3f},{ny1:.3f})', flush=True)

    if tiles:
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]) if len(tiles) >= 4 \
            else np.hstack(tiles)
        Image.fromarray(grid).save(out)
        print(f'-> {out}')


if __name__ == '__main__':
    main()
