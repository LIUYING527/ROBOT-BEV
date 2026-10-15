#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""投影正确性可视化验证：把 goal 投影像素用红叉标在真实帧上，人工检查是否指向目标。

取 visible + invisible 样本各几个，拼接成一张大图。
"""
import json
import sys
import os
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from slowbrain_proj import project_goal_normalized

W, H, HFOV = 640, 480, 79


def draw_cross(img, x, y, color, size=12):
    h, w = img.shape[:2]
    x, y = int(round(x)), int(round(y))
    for dx in range(-size, size + 1):
        if 0 <= x + dx < w and 0 <= y < h:
            img[y, x + dx] = color
        if 0 <= y + dx < h and 0 <= x < w:
            img[y + dx, x] = color


def main():
    annot = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/DongBaorong/BEVLOOK/benchmarks/slowbrain_data/slowbrain_smoke_annot.jsonl"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "/tmp/proj_check.png"

    vis_samples = []
    invis_samples = []
    with open(annot, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if not os.path.exists(a["frame"]):
                continue
            if a["goal_visible"] and len(vis_samples) < 6:
                vis_samples.append(a)
            elif not a["goal_visible"] and len(invis_samples) < 2:
                invis_samples.append(a)
            if len(vis_samples) >= 6 and len(invis_samples) >= 2:
                break

    tiles = []
    for a in vis_samples:
        rgb = np.load(a["frame"])["rgb"].copy()
        gx, gy = a["goal_nx"] * W, a["goal_ny"] * H
        draw_cross(rgb, gx, gy, (255, 0, 0), size=14)   # 红叉 = 投影目标
        draw_cross(rgb, W // 2, H // 2, (0, 255, 0), size=6)  # 绿点 = 图像中心
        tiles.append(rgb)
    for a in invis_samples:
        rgb = np.load(a["frame"])["rgb"].copy()
        draw_cross(rgb, W // 2, H // 2, (0, 255, 0), size=6)
        tiles.append(rgb)

    # 拼成 2x4
    rows = []
    for r in range(0, len(tiles), 4):
        row = tiles[r:r + 4]
        if row:
            rows.append(np.concatenate(row, axis=1))
    big = np.concatenate(rows, axis=0)

    from PIL import Image
    Image.fromarray(big).save(out_png)
    print(f"已保存 {out_png}（前6张=visible 红叉标投影目标；后2张=invisible 仅中心绿点）")
    print(f"visible 样本 goal 像素:")
    for a in vis_samples:
        print(f"  ep{a['episode_id']} goal=({a['goal_nx']:.2f},{a['goal_ny']:.2f}) "
              f"pixel=({a['goal_nx']*W:.0f},{a['goal_ny']*H:.0f}) d2g={a['dist_to_goal']:.1f}m")


if __name__ == "__main__":
    main()
