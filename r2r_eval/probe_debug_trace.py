#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0903 oracle探针失败诊断: 逐步记录(x,y,th,dtg,bend,act,goal_ego,d_end),
分类失败模式 = 终段orbit / bang-bang振荡 / 撞墙卡死 / 真丢失。
用法(lhvln env):
  CUDA_VISIBLE_DEVICES=4 HABITAT_GPU_ID=0 python probe_debug_trace.py \
      --ckpt goalnav_r2r_dense.pt --hfov 79 --ep_ids 1772 1469 201 728 1600 991 1386 1734 \
      --out probe_debug_trace
"""
import os
import sys
import json
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, '/data/DongBaorong/BEVLOOK/benchmarks/LH-VLN/our_eval')
_ARCH = '/data/DongBaorong/BEVLOOK/robot_bev_sim/arch_e2e/pipeline'
_SCRIPTS = '/data/DongBaorong/BEVLOOK/robot_bev_sim/scripts'
_GOALNAV = '/data/DongBaorong/BEVLOOK/robot_bev_sim/goalnav/pipeline'
for p in (_ARCH, _SCRIPTS, _GOALNAV):
    sys.path.insert(0, p)

from run_r2r import make_env  # noqa: E402
from our_agent import OurAgent  # noqa: E402
import stage_b_eval as sb  # noqa: E402
from train_goalnav_turn import status_gh  # noqa: E402
import cv2  # noqa: E402

GOAL_AHEAD = 4.0
ACT2ID = {'stop': 0, 'move_forward': 1, 'turn_left': 2, 'turn_right': 3}


def oracle_goal_ego(ref_ours, x, y, th):
    d = ref_ours - np.array([x, y])
    dist = np.linalg.norm(d, axis=1)
    i0 = int(np.argmin(dist))
    acc = 0.0
    gp = ref_ours[-1]
    for i in range(i0, len(ref_ours) - 1):
        seg = np.linalg.norm(ref_ours[i + 1] - ref_ours[i])
        if acc + seg >= GOAL_AHEAD:
            r = (GOAL_AHEAD - acc) / max(seg, 1e-9)
            gp = ref_ours[i] * (1 - r) + ref_ours[i + 1] * r
            break
        acc += seg
    d_end = float(np.linalg.norm(ref_ours[-1] - np.array([x, y])))
    c, s = np.cos(th), np.sin(th)
    dd = gp - np.array([x, y])
    ego = np.array([c * dd[0] + s * dd[1], -s * dd[0] + c * dd[1]], np.float32)
    return ego, d_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='goalnav_r2r_dense.pt')
    ap.add_argument('--ep_ids', type=str, nargs='+', required=True)
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--stop_dist', type=float, default=1.5)
    ap.add_argument('--hfov', type=float, default=79.0)
    ap.add_argument('--out', default='probe_debug_trace')
    cli = ap.parse_args()

    env = make_env('val_unseen', 640, 480, 79)
    all_eps = {str(e.episode_id): e for e in env.episodes}
    eps = [all_eps[i] for i in cli.ep_ids]
    print(f'==== debug-trace: {len(eps)} eps, ckpt={cli.ckpt} ====', flush=True)

    agent = OurAgent(ckpt=cli.ckpt, hfov=cli.hfov)
    out = open(cli.out + '.jsonl', 'w', encoding='utf-8')

    for ep in eps:
        env.episodes = [ep]
        obs = env.reset()
        ref = np.array(ep.reference_path, np.float64)
        ref_ours = np.stack([ref[:, 0], -ref[:, 2]], 1)
        agent.reset_episode()

        trace = []
        steps = 0
        stopped = False
        while not env.episode_over and steps < cli.max_steps:
            state = env.sim.get_agent_state()
            rgb = np.ascontiguousarray(obs['rgb'][:, :, :3])
            depth = np.squeeze(np.ascontiguousarray(obs['depth'])).astype(np.float32)
            position = np.asarray(state.position, np.float64)
            x, y, th, floor = agent.hab_pose_to_ours(position, state.rotation)
            pts = agent.depth_to_world(depth, position, state.rotation)
            agent.world_pts = np.concatenate([agent.world_pts, pts])[-600000:]

            m = env.get_metrics()
            dtg = float(m.get('distance_to_goal', -1))
            if dtg < cli.stop_dist:
                stopped = True
                break

            goal_ego, d_end = oracle_goal_ego(ref_ours, x, y, th)
            cam = sb.camera_feature(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), agent.device)
            pose = np.array([x, y, th], np.float32)
            bev = sb.lidar_feature_fov(agent.world_pts, pose, floor, agent.device)
            k = max(abs(goal_ego[0]) / 5.0, abs(goal_ego[1]) / 2.0, 1.0)
            st = status_gh(goal_ego / k, np.array([1.0, 0.0], np.float32), bs_vx=agent.vx, device=agent.device)
            st[0, 5] = agent.hist_w / 8.0
            st[0, 6] = agent.hist_gamma
            st[0, 7] = 0.0
            agent.torch.manual_seed(0)
            with agent.torch.no_grad():
                traj = agent.model({'camera_feature': cam, 'lidar_feature': bev,
                                    'status_feature': st})['trajectory'][0].cpu().numpy()
            bend = float(np.degrees(np.arctan2(traj[-1, 1], max(traj[-1, 0], 1e-6))))
            if bend > 15.0:
                act = 'turn_left'
            elif bend < -15.0:
                act = 'turn_right'
            else:
                act = 'move_forward'
            trace.append({'s': steps, 'x': round(float(x), 3), 'y': round(float(y), 3),
                          'th': round(float(np.degrees(th)), 1), 'dtg': round(dtg, 3),
                          'd_end': round(d_end, 3), 'bend': round(bend, 1), 'act': act,
                          'gx': round(float(goal_ego[0]), 2), 'gy': round(float(goal_ego[1]), 2)})
            obs = env.step(ACT2ID[act])
            steps += 1

        m = env.get_metrics()
        ne = float(m.get('distance_to_goal', float('nan')))
        rec = {'episode_id': str(ep.episode_id), 'steps': steps, 'stopped': stopped,
               'ne': round(ne, 3), 'trace': trace}
        out.write(json.dumps(rec) + '\n')
        out.flush()
        print(f'ep{ep.episode_id} steps={steps} ne={ne:.2f} stopped={stopped}', flush=True)

    out.close()


if __name__ == '__main__':
    main()
