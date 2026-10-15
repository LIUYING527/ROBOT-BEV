#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""慢脑数据生成：R2R GT 轨迹重投影 → (指令, 帧, 目标像素坐标)。

与 gen_adapt_data.py / gen_dense_data.py 的差异（慢脑专用）：
  - 新增透视投影：把 goal 世界坐标投影到「当前帧 + 当前姿态」的像素坐标（严格同步到帧）。
  - 保留目标不可见帧（visible=False, goal=(-1,-1)），慢脑要学「看不见就重新扫」。
  - 数据划分：只取 scene_split.json 里的 train_scenes（dev_scenes 单独 --scene-filter dev）。

用法(lhvln env, 需 Matterport3D scenes):
  CUDA_VISIBLE_DEVICES=0 HABITAT_GPU_ID=0 python gen_slowbrain_data.py \
      --split train --max_episodes 20 --out slowbrain_data/smoke
"""
import os
import sys
import json
import gzip
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from run_r2r import make_env, DATA_ROOT          # noqa: E402
from gen_adapt_data import yaw_towards           # noqa: E402
from gen_dense_data import densify               # noqa: E402
from slowbrain_proj import project_goal_normalized  # noqa: E402

DS = 0.25          # 插值步长(m)
SPLIT_MAP = "/data/DongBaorong/BEVLOOK/benchmarks/mp3d_r2r/scene_split.json"


def load_scene_split(mode="train"):
    with open(SPLIT_MAP) as f:
        d = json.load(f)
    if mode == "dev":
        return set(d["dev_scenes"])
    return set(d["train_scenes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--scene-filter", default="train", choices=["train", "dev", "all"],
                    help="train=仅 train_scenes, dev=仅 dev_scenes, all=不过滤")
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=79)
    ap.add_argument("--out", default="slowbrain_data/s0")
    cli = ap.parse_args()

    os.makedirs(cli.out, exist_ok=True)
    allow = load_scene_split(cli.scene_filter) if cli.scene_filter != "all" else None

    env = make_env(cli.split, cli.width, cli.height, cli.hfov)
    eps = list(env.episodes)
    print(f"==== R2R {cli.split}: {len(eps)} episodes, scene_filter={cli.scene_filter} ====", flush=True)

    meta = open(os.path.join(cli.out, "meta.jsonl"), "a", encoding="utf-8")
    done_ids = set()
    mf = os.path.join(cli.out, "meta.jsonl")
    if os.path.exists(mf):
        for ln in open(mf):
            try:
                done_ids.add(json.loads(ln)["episode_id"])
            except Exception:
                pass
        print(f"[resume] {len(done_ids)} eps already done", flush=True)

    n_done = 0
    n_vis = 0
    n_invis = 0
    for ei, ep in enumerate(eps):
        if ei < cli.start:
            continue
        if cli.max_episodes and n_done >= cli.max_episodes:
            break
        scene = ep.scene_id.split("/")[1]
        if allow is not None and scene not in allow:
            continue
        if str(ep.episode_id) in done_ids:
            continue
        env.episodes = [ep]
        env.reset()
        ref = ep.reference_path
        if not ref or len(ref) < 2:
            continue
        goal = np.array(getattr(ep.goals[0], "position", None) or ep.goals[0]["position"], np.float64)
        instruction = ep.instruction.instruction_text if hasattr(ep.instruction, "instruction_text") \
            else str(ep.instruction)

        pts = densify(ref)
        frames = []
        for vi, p in enumerate(pts):
            look = pts[min(vi + 1, len(pts) - 1)] if vi < len(pts) - 1 else goal
            q = yaw_towards(p, look)
            pp = np.array(p, np.float64)
            env.sim.set_agent_state(pp, q)
            o = env.sim.get_sensor_observations()
            rgb = np.ascontiguousarray(o["rgb"][:, :, :3])
            depth = np.squeeze(np.ascontiguousarray(o["depth"])).astype(np.float32)
            d_goal = float(np.linalg.norm(pp - goal))
            # 透视投影：goal 世界坐标 -> 当前帧像素（严格同步到该帧 pose）
            nu, nv, vis = project_goal_normalized(
                goal, pp, q, cli.width, cli.height, cli.hfov)
            frames.append({
                "rgb": rgb, "depth": depth,
                "pos": pp.tolist(), "rot": q.tolist(),
                "vi": vi, "dist_to_goal": d_goal,
                "goal_nx": round(float(nu), 4), "goal_ny": round(float(nv), 4),
                "goal_visible": bool(vis),
            })
            if vis:
                n_vis += 1
            else:
                n_invis += 1

        ep_dir = os.path.join(cli.out, f"ep{ep.episode_id}")
        os.makedirs(ep_dir, exist_ok=True)
        metas = []
        for fi, fr in enumerate(frames):
            np.savez_compressed(os.path.join(ep_dir, f"f{fi:03d}.npz"),
                                rgb=fr["rgb"], depth=fr["depth"])
            metas.append({k: v for k, v in fr.items() if k not in ("rgb", "depth")})
        meta.write(json.dumps({
            "episode_id": str(ep.episode_id), "scene": scene,
            "instruction": instruction, "goal": goal.tolist(),
            "ref_path": [list(map(float, p)) for p in ref],
            "frames": metas,
        }, ensure_ascii=False) + "\n")
        meta.flush()
        n_done += 1
        if n_done % 20 == 0:
            print(f"  [{n_done}] ep{ep.episode_id} frames={len(frames)} "
                  f"vis={n_vis} invis={n_invis}", flush=True)

    print(f"==== done: {n_done} eps, visible={n_vis} invisible={n_invis} "
          f"(ratio vis={n_vis / max(1, n_vis + n_invis):.2f}) -> {cli.out} ====", flush=True)


if __name__ == "__main__":
    main()
