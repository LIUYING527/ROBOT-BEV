#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R2R 稠密适配数据生成(0831, 追平DualVLN Phase1): ref_path 按 0.25m 弧长插值
teleport 回放 train split, 每帧渲染 RGB+depth 落盘, 供快脑扩训(50-100x数据量)。

与 gen_adapt_data.py 的差异:
  - 稠密插值: ref_path 稀疏路点(均5点) → 每0.25m一个视点(均~40帧/ep)
  - 朝向抖动增广: 每帧叠加 yaw ±30° 随机偏(策略鲁棒性, DAgger-lite)
  - 位置横向 jitter: 每帧横向 ±0.2m(可行走检查), 覆盖偏航状态分布
  - is_stop 标签保留(距goal<3m), stop头/STOP监督复用同一份数据

用法(lhvln env):
  CUDA_VISIBLE_DEVICES=2 HABITAT_GPU_ID=0 python gen_dense_data.py \
      --split train --start 0 --max_episodes 2700 --out dense_data/s0
"""
import os
import sys
import json
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from run_r2r import make_env, DATA_ROOT  # noqa: E402
from gen_adapt_data import yaw_towards  # noqa: E402

DS = 0.25          # 插值步长(m), 与 VLN-CE forward_step 一致
YAW_JITTER = 25.0  # 小朝向抖动幅度(度), 70%帧
YAW_RECOVER = 120.0  # 大角度恢复帧幅度(度), 30%帧 —— 教模型"偏90°也要持续转弯"
LAT_JITTER = 0.2   # 横向位置抖动(m)
# 0831根因: 旧adapt_data每帧朝向恒沿路径(yaw_towards)→ego系未来路径恒为直线→
# 模型学成"永远直走", goal_ego条件被忽略(oracle目标探针8/8全败, 40步0转弯实锤)。
# 必须用大角度jitter制造"航向偏离路径→GT弯曲"样本, 逼模型使用goal条件。


def densify(ref):
    """ref_path[(n,3)] → 每0.25m一个插值点 [(m,3)], 保留终点。"""
    pts = [np.array(ref[0], np.float64)]
    acc = 0.0
    for i in range(1, len(ref)):
        a = np.array(ref[i - 1], np.float64)
        b = np.array(ref[i], np.float64)
        seg = np.linalg.norm(b - a)
        if seg < 1e-6:
            continue
        d = DS - acc
        while d < seg:
            pts.append(a + (b - a) * (d / seg))
            d += DS
        acc = seg - (d - DS)
    pts.append(np.array(ref[-1], np.float64))
    return pts


def yaw_quat_mul(q, dyaw):
    """q(绕y) 再转 dyaw(弧度)。"""
    q2 = np.array([0.0, np.sin(dyaw / 2), 0.0, np.cos(dyaw / 2)])
    # habitat 四元数 [x,y,z,w], 纯y旋转相乘
    return np.array([
        q[3] * q2[0] + q[0] * q2[3] + q[1] * q2[2] - q[2] * q2[1],
        q[3] * q2[1] - q[0] * q2[2] + q[1] * q2[3] + q[2] * q2[0],
        q[3] * q2[2] + q[0] * q2[1] - q[1] * q2[0] + q[2] * q2[3],
        q[3] * q2[3] - q[0] * q2[0] - q[1] * q2[1] - q[2] * q2[2],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--hfov', type=float, default=79)
    ap.add_argument('--jitter', type=int, default=1, help='0=纯oracle 1=加抖动')
    ap.add_argument('--out', default='dense_data/s0')
    cli = ap.parse_args()

    os.makedirs(cli.out, exist_ok=True)
    env = make_env(cli.split, cli.width, cli.height, cli.hfov)
    eps = list(env.episodes)
    print(f'==== {cli.split}: {len(eps)} episodes total ====', flush=True)

    n_done = 0
    n_frames = 0
    meta = open(os.path.join(cli.out, 'meta.jsonl'), 'a', encoding='utf-8')
    done_ids = set()
    mf = os.path.join(cli.out, 'meta.jsonl')
    if os.path.exists(mf):  # resume: 跳过已完成的ep
        for ln in open(mf):
            done_ids.add(json.loads(ln)['episode_id'])
        print(f'[resume] {len(done_ids)} eps already done', flush=True)

    for ei, ep in enumerate(eps):
        if ei < cli.start:
            continue
        if cli.max_episodes and n_done >= cli.max_episodes:
            break
        if str(ep.episode_id) in done_ids:
            continue
        env.episodes = [ep]
        obs = env.reset()
        ref = ep.reference_path
        if not ref or len(ref) < 2:
            continue
        goal = np.array(getattr(ep.goals[0], 'position', None) or ep.goals[0]['position'], np.float64)
        instruction = ep.instruction.instruction_text if hasattr(ep.instruction, 'instruction_text') \
            else str(ep.instruction)

        pts = densify(ref)
        rng = np.random.RandomState(int(ep.episode_id) if str(ep.episode_id).isdigit() else 0)

        frames = []
        for vi, p in enumerate(pts):
            # 朝向: 看向下一插值点(末点朝goal)
            look = pts[min(vi + 1, len(pts) - 1)] if vi < len(pts) - 1 else goal
            q = yaw_towards(p, look)
            pp = np.array(p, np.float64)
            if cli.jitter and 0 < vi < len(pts) - 1:
                # 朝向抖动: 70%小抖动±25°(鲁棒), 30%大角度±120°(转弯恢复能力)
                if rng.rand() < 0.7:
                    q = yaw_quat_mul(q, np.deg2rad(rng.uniform(-YAW_JITTER, YAW_JITTER)))
                else:
                    q = yaw_quat_mul(q, np.deg2rad(rng.uniform(-YAW_RECOVER, YAW_RECOVER)))
                # 横向 jitter(相对前进方向的左/右)
                fwd = np.array([look[0] - p[0], 0, look[2] - p[2]])
                n = np.linalg.norm(fwd)
                if n > 1e-6:
                    left = np.array([-fwd[2], 0, fwd[0]]) / n
                    cand = pp + left * rng.uniform(-LAT_JITTER, LAT_JITTER)
                    if env.sim.is_navigable(cand):
                        pp = cand
            env.sim.set_agent_state(pp, q)
            o = env.sim.get_sensor_observations()
            rgb = np.ascontiguousarray(o['rgb'][:, :, :3])
            depth = np.squeeze(np.ascontiguousarray(o['depth'])).astype(np.float32)
            d_goal = float(np.linalg.norm(pp - goal))
            frames.append({'rgb': rgb, 'depth': depth, 'pos': pp.tolist(), 'rot': q.tolist(),
                           'vi': vi, 'dist_to_goal': d_goal,
                           'is_stop': bool(d_goal < 3.0)})

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
            'ref_path': [list(map(float, p)) for p in ref],
            'dense_path': [list(map(float, p)) for p in pts],
            'frames': metas,
        }, ensure_ascii=False) + '\n')
        meta.flush()
        n_frames += len(frames)
        n_done += 1
        if n_done % 20 == 0:
            print(f'  [{n_done}] ep{ep.episode_id} frames={len(frames)} total={n_frames}', flush=True)

    print(f'==== done: {n_done} episodes, {n_frames} frames -> {cli.out} ====', flush=True)


if __name__ == '__main__':
    main()
