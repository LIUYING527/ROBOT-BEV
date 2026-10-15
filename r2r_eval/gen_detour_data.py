#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R2R 绕行数据生成(0910, 治组3撞墙47%): 造"偏离 ref_path + 前方障碍 + goal 在障碍后 + 绕行 GT"样本。

根因(0909 组3诊断坐实): dense 数据沿 ref_path 采样, 模型训练时 BEV 里从无"前方障碍需绕行",
只会"goal 正前方→直走", 一偏离路径撞墙就不会绕。绕行数据逼模型学会"看到障碍 + goal 在障碍后 → 输出绕行轨迹"。

做法(复用 habitat pathfinder=navmesh shortest_path, 不手搓 A*):
  1. densify ref_path → 密集路径, 在中段选偏离点 p0。
  2. 横向偏离: 沿垂直前进方向左/右挪 d∈{0.5,1.0,1.5}m → snap 到可行走点。
  3. 朝向直冲 goal(yaw_towards), 制造"goal 正前方但被挡"处境。
  4. find_path(偏离点→goal) 得绕行最短路径, 绕行比=geodesic/direct > 1.25 才保留。
  5. densify 绕行路径 → dense_path(GT), 渲染偏离帧 rgb+depth, 落盘 dense_data 同格式。

用法(lhvln env, 22):
  CUDA_VISIBLE_DEVICES=6 HABITAT_GPU_ID=0 python gen_detour_data.py \
      --split train --max_episodes 2700 --out detour_data
"""
import os
import sys
import json
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from run_r2r import make_env  # noqa: E402
from gen_dense_data import densify  # noqa: E402
from gen_adapt_data import yaw_towards  # noqa: E402

DETOUR_RATIO = 1.2    # 绕行比阈值: geodesic/direct > 1.2 才判定"确实需绕行"
MIN_OFFSET = 0.3      # 偏离点离原路径最小距离(m), 确保是"偏离"而非"在路径上"
MAX_SNAP = 0.5        # snap 后允许的最大位移(m), 太大说明偏离点附近无可行走区
DEVIATIONS = (1.0, 2.0, 3.0)   # 横向偏离距离档(加大, 逼进"前方有墙"区)
BLOCK_CHECK = 2.5     # 前方障碍检查: 朝 goal 方向该距离内必须不可行走(否则不是撞墙处境)


def perpendicular(fwd):
    """xz 平面的垂直左向量(单位)。fwd=(dx,0,dz) → left=(-dz,0,dx)/|.|"""
    n = np.linalg.norm([fwd[0], fwd[2]])
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return np.array([-fwd[2], 0.0, fwd[0]]) / n


def try_detour(env, p0, fwd, side, d, goal):
    """在 p0 横向偏离 d(side 方向)处, 尝试构造一个绕行样本。返回 dict 或 None。"""
    off = perpendicular(fwd) * (side * d)
    p_dev = np.array(p0, np.float64) + off
    p_dev[1] = p0[1]
    if not env.sim.is_navigable(p_dev):
        p2 = np.array(env.sim.pathfinder.snap_point(p_dev), np.float64)
        if np.any(np.isnan(p2)):
            return None
        if np.linalg.norm(p2 - p_dev) > MAX_SNAP:
            return None
        p_dev = p2
        p_dev[1] = p0[1]
    # 偏离点必须真正偏离原路径
    if np.linalg.norm([p_dev[0] - p0[0], p_dev[2] - p0[2]]) < MIN_OFFSET:
        return None
    direct = float(np.linalg.norm([p_dev[0] - goal[0], p_dev[2] - goal[2]]))
    if direct < 1.5:  # 太靠近 goal, 无绕行意义
        return None

    # 前方障碍检查: 朝 goal 方向 BLOCK_CHECK 内必须被墙挡(否则不是"撞墙"处境)
    dg = np.array([goal[0] - p_dev[0], goal[2] - p_dev[2]], np.float64)
    dn = np.linalg.norm(dg)
    if dn < 1e-6:
        return None
    dg = dg / dn
    blocked = False
    for t in np.arange(0.4, BLOCK_CHECK + 0.2, 0.4):
        pt = np.array([p_dev[0] + dg[0] * t, p_dev[1], p_dev[2] + dg[1] * t], np.float64)
        if not env.sim.is_navigable(pt):
            blocked = True
            break
    if not blocked:
        return None

    # navmesh shortest_path(偏离点 → goal)
    import habitat_sim
    sp = habitat_sim.ShortestPath()
    sp.requested_start = p_dev
    sp.requested_end = goal
    if not env.sim.pathfinder.find_path(sp):
        return None
    if sp.geodesic_distance / direct < DETOUR_RATIO:
        return None
    pts = np.asarray(sp.points)
    if len(pts) < 2:
        return None

    dense = densify([list(map(float, p)) for p in pts])
    q = yaw_towards(p_dev, goal)
    return {'pos': p_dev, 'rot': q, 'dense': dense,
            'ratio': float(sp.geodesic_distance / direct)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--hfov', type=float, default=79)
    ap.add_argument('--out', default='detour_data')
    ap.add_argument('--max_detours_per_ep', type=int, default=8)
    cli = ap.parse_args()

    os.makedirs(cli.out, exist_ok=True)
    env = make_env(cli.split, cli.width, cli.height, cli.hfov)
    eps = list(env.episodes)
    print(f'==== {cli.split}: {len(eps)} episodes total ====', flush=True)

    meta = open(os.path.join(cli.out, 'meta.jsonl'), 'a', encoding='utf-8')
    done_ids = set()
    mf = os.path.join(cli.out, 'meta.jsonl')
    if os.path.exists(mf):
        for ln in open(mf):
            try:
                done_ids.add(json.loads(ln)['src_ep'])
            except Exception:
                pass
        if done_ids:
            print(f'[resume] {len(done_ids)} eps already done', flush=True)

    n_ep = 0
    n_detour = 0
    for ei, ep in enumerate(eps):
        if ei < cli.start:
            continue
        if cli.max_episodes and n_ep >= cli.max_episodes:
            break
        if str(ep.episode_id) in done_ids:
            continue
        env.episodes = [ep]
        env.reset()
        ref = ep.reference_path
        if not ref or len(ref) < 3:
            continue
        goal = np.array(getattr(ep.goals[0], 'position', None) or ep.goals[0]['position'], np.float64)
        instruction = ep.instruction.instruction_text if hasattr(ep.instruction, 'instruction_text') \
            else str(ep.instruction)

        dense_ref = densify(ref)
        detours = []
        # 中段密集点选偏离起点(步长 3 点, 覆盖整条路径)
        for vi in range(2, len(dense_ref) - 2, 3):
            p0 = dense_ref[vi]
            fwd = np.array(dense_ref[min(vi + 1, len(dense_ref) - 1)], np.float64) - p0
            for side in (-1, 1):
                for d in DEVIATIONS:
                    if len(detours) >= cli.max_detours_per_ep:
                        break
                    r = try_detour(env, p0, fwd, side, d, goal)
                    if r is not None:
                        detours.append(r)
                if len(detours) >= cli.max_detours_per_ep:
                    break
            if len(detours) >= cli.max_detours_per_ep:
                break

        if not detours:
            # 记一个占位行(resume 跳过用), 无绕行样本
            meta.write(json.dumps({'episode_id': str(ep.episode_id) + '_none',
                                   'src_ep': str(ep.episode_id), 'scene': ep.scene_id.split('/')[-2],
                                   'instruction': instruction, 'goal': goal.tolist(),
                                   'ref_path': [list(map(float, p)) for p in ref],
                                   'dense_path': [], 'frames': []}, ensure_ascii=False) + '\n')
            meta.flush()
            n_ep += 1
            continue

        # 每个绕行样本独立成一行(合成 episode_id), 与 dense_data 逐行对齐
        for di, r in enumerate(detours):
            syn_id = f'{ep.episode_id}_d{di}'
            epd = os.path.join(cli.out, f'ep{syn_id}')
            os.makedirs(epd, exist_ok=True)
            env.sim.set_agent_state(r['pos'], r['rot'])
            o = env.sim.get_sensor_observations()
            rgb = np.ascontiguousarray(o['rgb'][:, :, :3])
            depth = np.squeeze(np.ascontiguousarray(o['depth'])).astype(np.float32)
            np.savez_compressed(os.path.join(epd, 'f000.npz'), rgb=rgb, depth=depth)
            d_goal = float(np.linalg.norm(r['pos'] - goal))
            meta.write(json.dumps({
                'episode_id': syn_id, 'src_ep': str(ep.episode_id),
                'scene': ep.scene_id.split('/')[-2],
                'instruction': instruction, 'goal': goal.tolist(),
                'ref_path': [list(map(float, p)) for p in ref],
                'dense_path': [list(map(float, p)) for p in r['dense']],
                'detour_ratio': round(r['ratio'], 3),
                'frames': [{'fi': 0, 'vi': 0, 'pos': r['pos'].tolist(), 'rot': r['rot'].tolist(),
                            'dist_to_goal': d_goal, 'is_stop': bool(d_goal < 3.0)}],
            }, ensure_ascii=False) + '\n')
            meta.flush()
            n_detour += 1

        n_ep += 1
        if n_ep % 20 == 0:
            print(f'  [{n_ep}] ep{ep.episode_id} detours={len(detours)} total_detour={n_detour}',
                  flush=True)

    print(f'==== done: {n_ep} episodes, {n_detour} detour samples -> {cli.out} ====', flush=True)


if __name__ == '__main__':
    main()
