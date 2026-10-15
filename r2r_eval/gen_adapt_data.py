#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R2R 适配数据生成(0826, 路B): teleport 回放 train split reference_path,
渲染 RGB+depth, 落两类样本:
  ① stop 头: 每视点一帧, label=1(距goal<3m)/0(其余), 另加 goal 周边 3-5m jitter 硬负例
  ② 快脑微调: (rgb, depth, pose, goal_ego=前瞻路点, gt_traj=ref path 未来段) —— 本脚本先存原始帧,
     快脑特征(camera/lidar/status)在训练脚本里在线算(与 train_goalnav 管线一致)。

用法(lhvln env):
  CUDA_VISIBLE_DEVICES=7 HABITAT_GPU_ID=0 python gen_adapt_data.py \
      --split train --max_episodes 200 --out adapt_data/shard0
"""
import os
import sys
import json
import gzip
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from run_r2r import make_env, DATA_ROOT  # noqa: E402


def yaw_towards(p_from, p_to):
    """habitat 系: forward = R @ (0,0,-1)。返回看向 p_to 的 yaw 四元数 [x,y,z,w]。"""
    dx, dz = p_to[0] - p_from[0], p_to[2] - p_from[2]
    yaw = np.arctan2(-dx, -dz)          # forward=(sin? ) 见下验证: yaw=0 -> fwd=(0,0,-1)
    # 绕 y 轴旋转 yaw: fwd = ( -sin(yaw)?... ) 直接构造: 我们要 fwd_xz = (dx,dz)/|.|
    # habitat yaw 约定: R_y(θ) 作用于 (0,0,-1) -> (-sinθ? , 0, -cosθ)。取 θ = atan2(-dx, -dz)。
    q = np.array([0.0, np.sin(yaw / 2), 0.0, np.cos(yaw / 2)], np.float64)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--hfov', type=float, default=79)
    ap.add_argument('--jitter_n', type=int, default=4, help='每goal的硬负例jitter数(3-5m环)')
    ap.add_argument('--out', default='adapt_data/shard0')
    cli = ap.parse_args()

    os.makedirs(cli.out, exist_ok=True)
    env = make_env(cli.split, cli.width, cli.height, cli.hfov)
    eps = list(env.episodes)
    print(f'==== {cli.split}: {len(eps)} episodes total ====', flush=True)

    n_done = 0
    n_frames = 0
    meta = open(os.path.join(cli.out, 'meta.jsonl'), 'a', encoding='utf-8')
    for ei, ep in enumerate(eps):
        if ei < cli.start:
            continue
        if cli.max_episodes and n_done >= cli.max_episodes:
            break
        env.episodes = [ep]
        obs = env.reset()
        ref = ep.reference_path
        if not ref or len(ref) < 2:
            continue
        goal = np.array(getattr(ep.goals[0], 'position', None) or ep.goals[0]['position'], np.float64)
        instruction = ep.instruction.instruction_text if hasattr(ep.instruction, 'instruction_text') \
            else str(ep.instruction)

        # ① 沿 ref_path 逐视点 teleport 渲染(朝向=下一视点方向, 末点朝 goal)
        frames = []
        for vi in range(len(ref)):
            p = np.array(ref[vi], np.float64)
            look = np.array(ref[min(vi + 1, len(ref) - 1)], np.float64)
            q = yaw_towards(p, look)
            env.sim.set_agent_state(p, q)
            o = env.sim.get_sensor_observations()
            rgb = np.ascontiguousarray(o['rgb'][:, :, :3])
            depth = np.squeeze(np.ascontiguousarray(o['depth'])).astype(np.float32)
            d_goal = float(np.linalg.norm(p - goal))
            frames.append({'rgb': rgb, 'depth': depth, 'pos': p.tolist(), 'rot': q.tolist(),
                           'vi': vi, 'dist_to_goal': d_goal,
                           'is_stop': bool(d_goal < 3.0)})

        # ② goal 周边 3-5m 环 jitter 硬负例(stop 边界学习)
        rng = np.random.RandomState(ep.episode_id if isinstance(ep.episode_id, int) else 0)
        for ji in range(cli.jitter_n):
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(3.2, 5.0)
            p = goal + np.array([r * np.cos(ang), 0, r * np.sin(ang)])
            p[1] = goal[1]
            if not env.sim.is_navigable(p):
                p2 = np.array(env.sim.pathfinder.snap_point(p), np.float64)   # 吸到最近可行走点
                if np.any(np.isnan(p2)) or np.linalg.norm(p2 - goal) < 3.0:
                    continue
                p = p2
            q = yaw_towards(p, goal)      # 对着 goal 看=最像正例的负例
            env.sim.set_agent_state(p, q)
            o = env.sim.get_sensor_observations()
            rgb = np.ascontiguousarray(o['rgb'][:, :, :3])
            depth = np.squeeze(np.ascontiguousarray(o['depth'])).astype(np.float32)
            frames.append({'rgb': rgb, 'depth': depth, 'pos': p.tolist(), 'rot': q.tolist(),
                           'vi': -1, 'dist_to_goal': float(r),
                           'is_stop': False, 'hard_neg': True})

        ep_dir = os.path.join(cli.out, f'ep{ep.episode_id}')
        os.makedirs(ep_dir, exist_ok=True)
        metas = []
        for fi, fr in enumerate(frames):
            np.savez_compressed(os.path.join(ep_dir, f'f{fi:03d}.npz'),
                                rgb=fr['rgb'], depth=fr['depth'])
            metas.append({k: v for k, v in fr.items() if k not in ('rgb', 'depth')})
        meta.write(json.dumps({
            'episode_id': str(ep.episode_id), 'scene': ep.scene_id.split('/')[-2],
            'instruction': instruction, 'goal': goal.tolist(),
            'ref_path': [list(map(float, p)) for p in ref], 'frames': metas,
        }, ensure_ascii=False) + '\n')
        meta.flush()
        n_frames += len(frames)
        n_done += 1
        if n_done % 20 == 0:
            print(f'  [{n_done}] ep{ep.episode_id} frames={len(frames)} total={n_frames}', flush=True)

    print(f'==== done: {n_done} episodes, {n_frames} frames -> {cli.out} ====', flush=True)


if __name__ == '__main__':
    main()
