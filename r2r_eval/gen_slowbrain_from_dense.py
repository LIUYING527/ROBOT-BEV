#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 dense_data（已渲染快脑数据）后处理重投影 → 慢脑标注。

不需要 habitat / scenes：dense_data 的 meta.jsonl 已含 goal 世界坐标 + 每帧 pos/rot，
直接用 slowbrain_proj 投影 goal → 归一化像素，严格同步到帧。

输出 jsonl：{episode_id, scene, instruction, frame(npz路径), goal_nx, goal_ny, goal_visible, dist_to_goal}

用法:
  python3 gen_slowbrain_from_dense.py --dense dense_data/s0 --out slowbrain_annot.jsonl
"""
import os
import sys
import json
import argparse
import glob

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from slowbrain_proj import project_goal_normalized  # noqa: E402

W, H, HFOV = 640, 480, 79
SPLIT_MAP = "/data/DongBaorong/BEVLOOK/benchmarks/mp3d_r2r/scene_split.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="dense_data", help="dense_data 根目录(含 s0..s5)")
    ap.add_argument("--out", default="slowbrain_annot.jsonl")
    ap.add_argument("--scene-filter", default="train", choices=["train", "dev", "all"])
    ap.add_argument("--max_eps", type=int, default=0, help="最多处理多少 episode(0=全部)")
    ap.add_argument("--sensor-h", dest="sensor_h", type=float, default=1.30,
                    help="相机相对 agent 原点(地面)的高度,m。实测标定=1.30(见 docs/camera-height-bug.md)。"
                         "⚠️ 0=旧的错误口径(标签 ny 恒为 0.5)")
    cli = ap.parse_args()

    with open(SPLIT_MAP) as f:
        sm = json.load(f)
    if cli.scene_filter == "train":
        allow = set(sm["train_scenes"])
    elif cli.scene_filter == "dev":
        allow = set(sm["dev_scenes"])
    else:
        allow = None

    out = open(cli.out, "w", encoding="utf-8")
    n_eps = 0
    n_samples = 0
    n_vis = 0
    n_invis = 0
    if os.path.exists(os.path.join(cli.dense, "meta.jsonl")):
        shards = [cli.dense]                      # --dense 直接指向单个 shard 目录
    else:
        shards = sorted(glob.glob(os.path.join(cli.dense, "s*")))
    for shard in shards:
        meta_path = os.path.join(shard, "meta.jsonl")
        if not os.path.exists(meta_path):
            continue
        for line in open(meta_path, encoding="utf-8"):
            ep = json.loads(line)
            if allow is not None and ep["scene"] not in allow:
                continue
            goal = np.array(ep["goal"], np.float64)
            instruction = ep["instruction"]
            for fr in ep["frames"]:
                pos = np.array(fr["pos"], np.float64)
                pos[1] += cli.sensor_h          # ⭐相机在 agent 原点上方 sensor_h 米
                rot = np.array(fr["rot"], np.float64)
                nx, ny, vis = project_goal_normalized(goal, pos, rot, W, H, HFOV)
                out.write(json.dumps({
                    "episode_id": ep["episode_id"],
                    "scene": ep["scene"],
                    "instruction": instruction,
                    "frame": os.path.join(shard, f"ep{ep['episode_id']}", f"f{fr['vi']:03d}.npz"),
                    "goal_nx": round(float(nx), 4),
                    "goal_ny": round(float(ny), 4),
                    "goal_visible": bool(vis),
                    "dist_to_goal": fr["dist_to_goal"],
                }, ensure_ascii=False) + "\n")
                n_samples += 1
                if vis:
                    n_vis += 1
                else:
                    n_invis += 1
            n_eps += 1
            if cli.max_eps and n_eps >= cli.max_eps:
                break
        if cli.max_eps and n_eps >= cli.max_eps:
            break

    out.close()
    tot = max(1, n_vis + n_invis)
    print(f"==== done: {n_eps} eps, {n_samples} samples | "
          f"visible={n_vis} ({n_vis / tot:.2f}) invisible={n_invis} ({n_invis / tot:.2f}) "
          f"-> {cli.out} ====")


if __name__ == "__main__":
    main()
