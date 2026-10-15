#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R2R 端到端评测：慢脑=InfiniteVL（HTTP 定位终点）+ 快脑=DiffusionDrive。

每步：habitat RGB+depth → 慢脑 HTTP 定位终点像素 → 单点深度反投影 → ego goal
     → 快脑 DD 出轨迹 → bend 量化动作（turn/forward）；终点不可见 → 扫描探索；
     接近终点 < stop_dist → stop。

快脑调用口径与 probe_oracle_goal.py 完全一致（status_gh + goal_dir）。
用法（lhvln env）：
  CUDA_VISIBLE_DEVICES=2 HABITAT_GPU_ID=0 python eval_r2r_infinitevl.py \
      --ckpt goalnav_r2r_dense_k24_warm50_ep7.pt --server http://localhost:8123 --n 10
"""
import os
import sys
import json
import base64
import random
import argparse
import faulthandler
from collections import deque
faulthandler.dump_traceback_later(300, exit=True)  # 5 分钟卡住则 dump 栈定位

import numpy as np
import cv2
import requests

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, '/data/DongBaorong/BEVLOOK/benchmarks/LH-VLN/our_eval')
_ARCH = '/data/DongBaorong/BEVLOOK/robot_bev_sim/arch_e2e/pipeline'
_SCRIPTS = '/data/DongBaorong/BEVLOOK/robot_bev_sim/scripts'
_GOALNAV = '/data/DongBaorong/BEVLOOK/robot_bev_sim/goalnav/pipeline'
for p in (_ARCH, _SCRIPTS, _GOALNAV):
    sys.path.insert(0, p)

from run_r2r import make_env          # noqa: E402
from our_agent import OurAgent        # noqa: E402
import stage_b_eval as sb             # noqa: E402
from train_goalnav_turn import status_gh  # noqa: E402

HFOV = 79.0
ACT2ID = {'stop': 0, 'move_forward': 1, 'turn_left': 2, 'turn_right': 3}
SCAN_MAX = 8  # 连续不可见 8 次（≈120°）后前进换视角


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='goalnav_r2r_dense_k24_warm50_ep7.pt')
    ap.add_argument('--server', default='http://localhost:8123')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--sample', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--stop_dist', type=float, default=2.0)
    ap.add_argument('--stop_confirm', type=int, default=1,
                    help='连续多少步反投影<stop_dist 才真停(1=原行为)。'
                         '0912 实测: 单次慢脑误报就能终结整个 episode(出现 1-10 步就停、真实NE却有8-12m 的样本)')
    ap.add_argument('--nframes', type=int, default=1,
                    help='慢脑输入帧数(1=单帧; 4=多帧, 最后一帧=当前帧)。多帧须配多帧 adapter'
                         '(slowbrain_lora_multiframe)。0912: 多帧漏检 9.5% 最低, 而闭环 91%% 的步数都卡在慢脑判不可见')
    ap.add_argument('--diag', default=None,
                    help='逐步诊断 jsonl 路径(每步记录 dtg/慢脑输出/深度/反投影), 用于定位「5m 墙」成因')
    ap.add_argument('--stop_median_n', type=int, default=0,
                    help='stop 前先对最近 N 步的反投影距离取中位(时序滤波)。0912 实测: depth 修复后慢脑'
                         '定位总体误差仅 8%%, 但 27.4%% 的步会间歇性指错(反投影<2m 而真实 dtg 中位 11.6m), '
                         'stop_confirm 挡不住"连续多次都错"。用中位滤波可滤掉偶发错误。0=不启用')
    ap.add_argument('--geo_check', type=float, default=0.0,
                    help='几何一致性校验阈值。慢脑给 (nx,ny) 后, 若该像素确为地面点, 其距离应为 '
                         'h/tan((ny-0.5)*vfov); 若实测深度 < 该几何预期*阈值 ⇒ 该像素是近处障碍物(非地面) '
                         '⇒ 慢脑指错, 剔除该判决。0912 实测: ny=0.85 几何预期 2.9m, 实测深度仅 0.45m, '
                         '可被干净识别。0=不启用')
    ap.add_argument('--min_stop_depth', type=float, default=0.0,
                    help='stop 判决的最小深度门槛(米)。0912 诊断: 慢脑报可见时坐标处深度中位仅 0.57m'
                         '(指向近处物体), 而真实距离中位 7.49m ⇒ 100%% 的停止判决都是误判。'
                         '几何上地面目标在画面内的最小距离 = 相机高/tan(半垂直FOV) ≈ 1.96m, '
                         '故设 1.9 可滤掉「指向近处障碍物」这类假到达。0=不启用(原行为)')
    ap.add_argument('--ep_ids', default=None,
                    help='逗号分隔的 episode_id 列表: 固定样本集, 用于跨配置严格配对。'
                         '⚠️0913 实测: habitat 的 env.episodes 顺序每次加载都不同, '
                         'random.Random(seed) 无法复现同一批 episode, 跨次对比只能靠统计')
    ap.add_argument('--out', default='r2r_infinitevl')
    cli = ap.parse_args()

    print('[progress] make_env 开始...', flush=True)
    env = make_env('val_unseen', 640, 480, 79)
    print('[progress] make_env 完成', flush=True)
    eps = list(env.episodes)
    if cli.ep_ids:
        # 固定样本集: habitat 的 env.episodes 顺序每次加载都不同, 靠 seed 复现不了(0913 实测),
        # 要做严格配对 A/B 只能显式指定 episode_id。
        want = [s.strip() for s in cli.ep_ids.split(',') if s.strip()]
        idx = {str(e.episode_id): e for e in eps}
        miss = [w for w in want if w not in idx]
        if miss:
            print(f'[warn] ep_ids 有 {len(miss)} 个不在数据集中: {miss[:5]}', flush=True)
        eps = [idx[w] for w in want if w in idx]
    else:
        random.Random(cli.seed).shuffle(eps)
        eps = eps[:cli.sample][:cli.n]
    env.episodes = eps
    print(f'==== R2R+InfiniteVL: {len(eps)} eps, ckpt={cli.ckpt}, server={cli.server} ====', flush=True)

    print('[progress] OurAgent(快脑DD) 加载中...', flush=True)
    agent = OurAgent(ckpt=cli.ckpt, hfov=HFOV)
    print('[progress] OurAgent 加载完成', flush=True)
    results = open(cli.out + '.jsonl', 'w', encoding='utf-8')
    fdiag = open(cli.diag, 'w', encoding='utf-8') if cli.diag else None

    for ei in range(len(eps)):
        env.episodes = [eps[ei]]
        obs = env.reset()
        ep = eps[ei]
        instruction = (obs['instruction']['text'] if isinstance(obs.get('instruction'), dict)
                       else str(ep.instruction.instruction_text))
        agent.reset_episode()
        hist_rgb = deque(maxlen=cli.nframes)   # 多帧缓冲(最后一帧=当前帧)
        steps = 0
        min_dtg = float('inf')
        stopped = False
        scan = 0
        stop_votes = 0
        dist_hist = deque(maxlen=max(1, cli.stop_median_n))   # 时序滤波缓冲

        while not env.episode_over and steps < cli.max_steps:
            # 重置「卡住」计时：dump_traceback_later 是绝对时限、不会自动重置，
            # 不重置的话跑超过 5 分钟就被无条件 exit（30 eps 必挂）。
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(300, exit=True)
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

            # 慢脑定位终点（HTTP）
            hist_rgb.append(rgb)
            fr_send = list(hist_rgb)
            if len(fr_send) < cli.nframes:          # 开局帧数不足: 用最早帧补齐
                fr_send = [fr_send[0]] * (cli.nframes - len(fr_send)) + fr_send
            try:
                r = requests.post(cli.server + '/locate', json={
                    'frames': [{'rgb': base64.b64encode(f.tobytes()).decode(),
                                'shape': list(f.shape)} for f in fr_send],
                    'instruction': instruction}, timeout=90).json()
            except Exception as e:
                print(f'[warn] slowbrain server err: {e}', flush=True)
                r = {'visible': False, 'nx': None, 'ny': None}

            if fdiag is not None:      # 逐步诊断(定位「5m 墙」: 是看不见, 还是看见了但目标不对)
                _dv = depth[(depth > 0.1) & (depth < 30.0)]
                fdiag.write(json.dumps({
                    'ep': str(ep.episode_id), 'step': steps, 'dtg': round(float(dtg), 2),
                    'vis': bool(r['visible']), 'nx': r.get('nx'), 'ny': r.get('ny'),
                    # 全局深度统计: 用于区分「慢脑指错」vs「agent 确实贴在障碍物前」
                    'depth_med': round(float(np.median(_dv)), 2) if _dv.size else None,
                    'depth_p10': round(float(np.percentile(_dv, 10)), 2) if _dv.size else None,
                    'depth_p90': round(float(np.percentile(_dv, 90)), 2) if _dv.size else None,
                }, ensure_ascii=False) + '\n')
                fdiag.flush()          # 0913: 不 flush 则尾部数据靠退出时刷盘, 实测会丢(ep1704 丢了 44 步)

            if not r['visible']:
                scan += 1
                act = 'move_forward' if scan % SCAN_MAX == 0 else 'turn_left'
            else:
                scan = 0
                nx, ny = r['nx'], r['ny']
                H, W = depth.shape
                u = int(min(max(nx * W, 0), W - 1))
                v = int(min(max(ny * H, 0), H - 1))
                patch = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
                patch = patch[(patch > 0.3) & (patch < 12.0)]
                if patch.size == 0:
                    act = 'turn_left'
                else:
                    d = float(np.median(patch))
                    fx = (W / 2.0) / np.tan(np.deg2rad(HFOV / 2.0))
                    goal_ego = np.array([d, -(u - (W - 1) / 2.0) * d / fx], np.float32)
                    dist = float(np.hypot(*goal_ego))
                    # ⭐ 0912: 快脑 DD 的训练口径是「沿路径前瞻 GOAL_AHEAD=4m 的点」
                    # (见 probe_oracle_goal.oracle_goal_ego —— 探针能跑 65.6% 就靠这个口径)。
                    # 而慢脑给的是**终点**, 可能远在 8~20m 外 ⇒ 喂给快脑超出训练分布 ⇒ 走不动。
                    # 故: 保持方向不变, 把长度截断到 4m 后喂快脑; 用真实终点距离判 stop。
                    _GA = 4.0
                    goal_feed = (goal_ego * (_GA / dist)) if dist > _GA else goal_ego
                    goal_ego = goal_feed.astype(np.float32)
                    # 几何一致性: 该像素若真是地面目标, 距离应为 h/tan(俯角); 实测远小于此 ⇒ 指到近处障碍物
                    _vfov = 2.0 * np.degrees(np.arctan(np.tan(np.deg2rad(HFOV / 2.0)) / (float(W) / H)))
                    _pitch = (float(ny) - 0.5) * _vfov
                    _d_geo = (1.30 / np.tan(np.deg2rad(_pitch))) if _pitch > 1.0 else 1e9
                    geo_ok = (cli.geo_check <= 0.0) or (d >= _d_geo * cli.geo_check)
                    if fdiag is not None:
                        fdiag.write(json.dumps({
                            'ep': str(ep.episode_id), 'step': steps, 'dtg': round(float(dtg), 2),
                            'vis': True, 'nx': float(nx), 'ny': float(ny), 'd': round(d, 2),
                            'goal_fwd': round(float(goal_ego[0]), 2),
                            'goal_lat': round(float(goal_ego[1]), 2),
                            'dist': round(float(dist), 2),
                        }, ensure_ascii=False) + '\n')
                        fdiag.flush()  # 0913: 同上, 保证停止判决行实时落盘
                    dist_hist.append(dist)
                    dist_f = float(np.median(dist_hist)) if cli.stop_median_n > 0 else dist
                    if dist_f < cli.stop_dist and d >= cli.min_stop_depth and geo_ok:
                        # 连续确认: 单次慢脑误报不应终结整个 episode(未确认时继续走, 让快脑参与)
                        # 且深度需 >= min_stop_depth: 否则该像素是近处障碍物而非地面目标(0912 诊断)
                        stop_votes += 1
                        if stop_votes >= cli.stop_confirm:
                            stopped = True
                            break
                    else:
                        stop_votes = 0
                    # 快脑 DD（与 probe_oracle_goal 同口径：status_gh + goal_dir）
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
                                            'status_feature': st, 'goal_dir': goal_dir})['trajectory'][0].cpu().numpy()
                    act, bend = agent._traj_to_act(traj)

            obs = env.step(ACT2ID[act])
            steps += 1

        m = env.get_metrics()
        ne = float(m.get('distance_to_goal', float('nan')))
        rec = {'episode_id': str(ep.episode_id), 'steps': steps,
               'success': float(stopped and ne < 3.0),
               'os': float(min_dtg < 3.0), 'ne': ne, 'min_dtg': min_dtg, 'spl': 0.0}
        gt_len = float(m.get('gt_path_length', 0) or 0)
        if rec['success'] > 0 and gt_len > 0:
            rec['spl'] = min(1.0, gt_len / max(steps * 0.25, gt_len))
        results.write(json.dumps(rec) + '\n')
        results.flush()
        print(f'[{ei + 1}/{len(eps)}] ep{ep.episode_id} steps={steps} min_dtg={min_dtg:.2f} '
              f'ne={ne:.2f} succ={rec["success"]:.0f}', flush=True)

    results.close()
    rows = [json.loads(l) for l in open(cli.out + '.jsonl')]
    n = len(rows)
    print(f'==== R2R+INFINITEVL n={n} SR={sum(r["success"] for r in rows) / n * 100:.1f} '
          f'OSR={sum(r["os"] for r in rows) / n * 100:.1f} '
          f'NE={np.mean([r["ne"] for r in rows]):.2f} ====', flush=True)


if __name__ == '__main__':
    main()
