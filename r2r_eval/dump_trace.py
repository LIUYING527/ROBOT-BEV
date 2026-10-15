#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""0912 失败案例 trace dump: 跑指定 ep, 记录每步 (位置/动作/脱困/dt g), 出轨迹图。
复用 probe_oracle_goal 的全部口径(oracle goal / 几何停 / 脱困), 只加记录与画图。

用法(lhvln):
  CUDA_VISIBLE_DEVICES=4 python dump_trace.py --ckpt goalnav_r2r_dense_goalmlp_ep5.pt \
      --ep_ids 1833 1469 997 --escape 2 --out trace_oLBM
"""
import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)

from probe_oracle_goal import (  # noqa: E402
    make_env, OurAgent, sb, status_gh, cv2,
    densify_ours, oracle_goal_ego, Escape, EscapeV2, free_dir_ego, ACT2ID, GOAL_AHEAD,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='goalnav_r2r_dense_goalmlp_ep5.pt')
    ap.add_argument('--ep_ids', nargs='+', required=True)
    ap.add_argument('--escape', type=int, default=2)
    ap.add_argument('--hfov', type=float, default=79.0)
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--stop_dist', type=float, default=1.5)
    ap.add_argument('--dense_goal', action='store_true', default=True)
    ap.add_argument('--out', default='trace')
    cli = ap.parse_args()

    env = make_env('val_unseen', 640, 480, 79)
    eps = list(env.episodes)
    want = [str(i) for i in cli.ep_ids]
    by_id = {str(e.episode_id): e for e in eps}
    eps = [by_id[i] for i in want if i in by_id]
    print('选中 %d 条 ep' % len(eps), flush=True)

    agent = OurAgent(ckpt=cli.ckpt, hfov=cli.hfov)
    traces = []

    for ei, ep in enumerate(eps):
        env.episodes = [ep]
        obs = env.reset()
        ref = np.array(ep.reference_path, np.float64)
        ref_ours = np.stack([ref[:, 0], -ref[:, 2]], 1)
        if cli.dense_goal:
            ref_ours = densify_ours(ref_ours)
        agent.reset_episode()

        steps, min_dtg, stopped, n_escape = 0, float('inf'), False, 0
        prev_xy, pending_act = None, None
        esc = EscapeV2() if cli.escape == 2 else Escape()
        rec = []
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
            min_dtg = min(min_dtg, dtg)
            if dtg < cli.stop_dist:
                stopped = True
                rec.append({'step': steps, 'x': x, 'y': y, 'th': th,
                            'z': float(position[1]),
                            'dtg': dtg, 'act': 'STOP', 'esc': 0, 'moved': 0.0,
                            'free_a': None, 'free_clear': None})
                break

            goal_ego, d_end = oracle_goal_ego(ref_ours, x, y, th)
            moved = 0.0 if prev_xy is None else float(np.hypot(x - prev_xy[0], y - prev_xy[1]))
            prev_xy = (x, y)

            # 每 5 步算一次"当前局部最空方向"(诊断脱困选向是否合理); 0=正前, 逆时针正
            free_a, free_clear = None, None
            if steps % 5 == 0 and len(agent.world_pts) > 0:
                _ws = agent.world_pts[::20]
                _dd = _ws[:, :2] - np.array([x, y], np.float64)
                _cc, _ss = np.cos(th), np.sin(th)
                _ep = np.stack([_cc*_dd[:, 0] + _ss*_dd[:, 1],
                                -_ss*_dd[:, 0] + _cc*_dd[:, 1]], 1).astype(np.float32)
                free_a, free_clear = free_dir_ego(_ep)

            acted, is_esc = None, 0
            if cli.escape and pending_act is not None:
                if cli.escape == 2:
                    wp = agent.world_pts
                    if len(wp) > 0:
                        ws = wp[::6]
                        dd = ws[:, :2] - np.array([x, y], np.float64)
                        cc, ss = np.cos(th), np.sin(th)
                        ego_pts = np.stack([cc*dd[:, 0] + ss*dd[:, 1],
                                            -ss*dd[:, 0] + cc*dd[:, 1]], 1).astype(np.float32)
                    else:
                        ego_pts = None
                    e = esc.update(pending_act, moved, goal_ego, steps, ego_pts=ego_pts, dtg=dtg)
                else:
                    e = esc.update(pending_act, moved, goal_ego, steps, dtg=dtg)
                if e is not None:
                    n_escape += 1
                    acted, is_esc = e, 1
            if acted is None:
                cam = sb.camera_feature(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), agent.device)
                pose = np.array([x, y, th], np.float32)
                bev = sb.lidar_feature_fov(agent.world_pts, pose, floor, agent.device)
                k = max(abs(goal_ego[0]) / 5.0, abs(goal_ego[1]) / 2.0, 1.0)
                st = status_gh(goal_ego / k, np.array([1.0, 0.0], np.float32),
                               bs_vx=agent.vx, device=agent.device)
                st[0, 5] = agent.hist_w / 8.0
                st[0, 6] = agent.hist_gamma
                st[0, 7] = 0.0
                agent.torch.manual_seed(0)
                goal_dir = agent.torch.tensor(
                    goal_ego / np.array([5.0, 2.0], np.float32), device=agent.device).unsqueeze(0)
                with agent.torch.no_grad():
                    traj = agent.model({'camera_feature': cam, 'lidar_feature': bev,
                                        'status_feature': st,
                                        'goal_dir': goal_dir})['trajectory'][0].cpu().numpy()
                bend = float(np.degrees(np.arctan2(traj[-1, 1], max(traj[-1, 0], 1e-6))))
                acted = 'turn_left' if bend > 15.0 else ('turn_right' if bend < -15.0 else 'move_forward')

            rec.append({'step': steps, 'x': float(x), 'y': float(y), 'th': float(th),
                        'z': float(position[1]),          # ⭐habitat 高度: 多楼层必须看这个
                        'dtg': float(dtg), 'act': acted, 'esc': is_esc, 'moved': float(moved),
                        'bend': float(bend) if acted and not is_esc else None,
                        'free_a': None if free_a is None else float(free_a),
                        'free_clear': None if free_clear is None else float(free_clear)})
            obs = env.step(ACT2ID[acted])
            pending_act = acted
            steps += 1

        ne = float(env.get_metrics().get('distance_to_goal', float('nan')))
        suc = float(stopped and ne < 3.0)
        traces.append({'episode_id': str(ep.episode_id), 'steps': steps, 'n_escape': n_escape,
                       'success': suc, 'ne': ne, 'min_dtg': min_dtg,
                       'ref_ours': ref_ours.tolist(), 'rec': rec,
                       'start': [float(by_id[str(ep.episode_id)].start_position[0]),
                                 float(by_id[str(ep.episode_id)].start_position[2])]})
        print('  ep%s steps=%d esc=%d min_dtg=%.2f ne=%.2f succ=%.0f' % (
            ep.episode_id, steps, n_escape, min_dtg, ne, suc), flush=True)

    with open(cli.out + '.json', 'w') as f:
        json.dump(traces, f)

    # ---- 画图: 每条约一张 + 合并大图 ----
    n = len(traces)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), squeeze=False)
    for i, t in enumerate(traces):
        ax = axes[0][i]
        R = np.array(t['ref_ours'])
        ax.plot(R[:, 0], R[:, 1], 'g--', lw=2, label='GT ref path')
        ax.plot(R[0, 0], R[0, 1], 'go', ms=8, label='GT start')
        ax.plot(R[-1, 0], R[-1, 1], 'g*', ms=14, label='goal')
        P = np.array([[r['x'], r['y']] for r in t['rec']])
        E = np.array([r['esc'] for r in t['rec']])
        if len(P):
            ax.plot(P[:, 0], P[:, 1], 'b-', lw=1.2, alpha=0.8, label='ours')
            if E.any():
                ax.plot(P[E == 1, 0], P[E == 1, 1], 'r.', ms=3, alpha=0.6, label='escape-active')
            ax.plot(P[0, 0], P[0, 1], 'bo', ms=6)
        ax.set_title('ep%s  steps=%d esc=%d(%.0f%%) min_dtg=%.2f\nsucc=%.0f' % (
            t['episode_id'], t['steps'], t['n_escape'],
            100*t['n_escape']/max(1, t['steps']), t['min_dtg'], t['success']), fontsize=10)
        ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    plt.tight_layout()
    plt.savefig(cli.out + '.png', dpi=110, bbox_inches='tight')
    print('图已存: %s.png' % cli.out, flush=True)


if __name__ == '__main__':
    main()
