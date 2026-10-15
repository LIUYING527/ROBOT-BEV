#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DualVLN慢脑(HTTP服务) × 我们dense快脑 混合探针(0902, "薅慢脑"实验):
慢脑=dualvlm_slow_server.py(internvla env, 4bit S2) 出像素目标/箭头/STOP;
快脑=goalnav_r2r_dense.pt(goal_ego条件 + bend±15°量化, 0902修复版)。
回答: 换一个训练过的强慢脑, 我们快脑能兑现多少(对照: oracle探针62 / v5慢脑smoke10)。

设计:
  - 每K步(或到达缓存目标/无目标)POST /s2 infer=1; 像素目标→depth反投影→世界目标缓存,
    中间步把世界目标重表达到ego系喂快脑 + POST /s2 infer=0补慢脑历史帧。
  - S2输出箭头→直接执行; STOP→stop(habitat判success)。

用法(lhvln env; 先起dualvlm_slow_server):
  CUDA_VISIBLE_DEVICES=7 python run_r2r_hybrid.py --n 10 --out results_hybrid_smoke10
"""
import os
import sys
import io
import json
import random
import argparse

import numpy as np
import requests
from PIL import Image

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

ACT2ID = {'stop': 0, 'move_forward': 1, 'turn_left': 2, 'turn_right': 3}
ARROW2ACT = {1: 'move_forward', 2: 'turn_left', 3: 'turn_right'}   # 0=stop单独处理, 5=↓不支持
SLOW = 'http://127.0.0.1:8155'


class Escape:
    """0903卡死/转圈检测+固定逃脱序列(oracle探针配对实锤SR62→70, 见probe_oracle_escape.py)。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = []
        self.plan = []
        self.level = 0
        self.last_escape_step = -100
        self.n_escape = 0

    def update(self, act, moved, goal_ego, step):
        if self.plan:
            self.n_escape += 1
            return self.plan.pop(0)
        self.hist.append((act, moved))
        self.hist = self.hist[-24:]
        last4 = self.hist[-4:]
        if len(last4) == 4 and all(a == 'move_forward' for a, _ in last4) \
                and sum(m for _, m in last4) < 0.15 and step - self.last_escape_step > 8:
            side = 'turn_left' if goal_ego[1] > 0 else 'turn_right'
            plans = [
                [side] * 3 + ['move_forward'] * 2,
                [('turn_right' if side == 'turn_left' else 'turn_left')] * 3 + ['move_forward'] * 2,
                ['turn_left'] * 6 + ['move_forward'] * 3,
            ]
            self.plan = plans[min(self.level, 2)][:]
            self.level += 1
            self.last_escape_step = step
            self.hist = []
            self.n_escape += 1
            return self.plan.pop(0)
        last20 = self.hist[-20:]
        if len(last20) == 20:
            net = sum(30 if a == 'turn_left' else (-30 if a == 'turn_right' else 0) for a, _ in last20)
            moved = sum(m for _, m in last20)
            if abs(net) > 300 and moved < 0.5 and step - self.last_escape_step > 8:
                self.plan = ['move_forward'] * 3
                self.last_escape_step = step
                self.hist = []
                self.n_escape += 1
                return self.plan.pop(0)
        return None


def s2_call(rgb, instruction, infer):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=92)
    buf.seek(0)
    r = requests.post(SLOW + '/s2', files={'file': ('f.jpg', buf, 'image/jpeg')},
                      data={'instruction': instruction, 'infer': '1' if infer else '0'},
                      timeout=120)
    r.raise_for_status()
    return r.json()


def pixel_to_world_goal(px, py, depth, x, y, th, hfov=79.0):
    """S2像素目标(384x384坐标) → 原图depth反投影 → 世界点。None若深度无效。"""
    H, W = depth.shape
    u = int(np.clip(px * W / 384.0, 0, W - 1))
    v = int(np.clip(py * H / 384.0, 0, H - 1))
    patch = depth[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
    patch = patch[(patch > 0.3) & (patch < 12.0)]
    if patch.size == 0:
        return None
    d = float(np.median(patch))
    fx = (W / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    ex = d
    ey = -(u - (W - 1) / 2.0) * d / fx
    c, s = np.cos(th), np.sin(th)
    return np.array([x + c * ex - s * ey, y + s * ex + c * ey])


def dock_servo(env, agent, obs, goal_world, x, y, th, steps, min_dtg, cli):
    """终段伺服(0903): S2喊STOP后朝缓存goal_world几何贴近, 收A类停得稍远。
    30°转向对正 + 前进到~dock_target; cap行程防走飞, |angle|>90判定走过头直接停。
    返回(obs, steps, min_dtg, dock_steps)。"""
    dock_steps = 0
    while dock_steps < cli.dock_max_steps:
        dd = goal_world - np.array([x, y])
        dist = float(np.linalg.norm(dd))
        if dist < cli.dock_target:
            break
        if dist > cli.dock_cap:
            break  # 目标太远/走过头, 防走飞
        c, s = np.cos(th), np.sin(th)
        ge_x = c * dd[0] + s * dd[1]
        ge_y = -s * dd[0] + c * dd[1]
        ang = np.degrees(np.arctan2(ge_y, ge_x))
        if abs(ang) > 90:
            break  # 目标在身后=走过头, 不掺和
        if abs(ang) > 15:
            act = 'turn_left' if ang > 0 else 'turn_right'
        else:
            act = 'move_forward'
        obs = env.step(ACT2ID[act])
        steps += 1
        dock_steps += 1
        state = env.sim.get_agent_state()
        x, y, th, _ = agent.hab_pose_to_ours(state.position, state.rotation)
        m = env.get_metrics()
        min_dtg = min(min_dtg, float(m.get('distance_to_goal', -1)))
    obs = env.step(ACT2ID['stop'])
    return obs, steps, min_dtg, dock_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--sample', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ep_ids', nargs='+', default=None,
                    help='显式指定episode id列表(跨进程配对评测; 0902发现env.episodes顺序进程间不确定)')
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--k_replan', type=int, default=4)
    ap.add_argument('--ckpt', default='goalnav_r2r_dense.pt')
    ap.add_argument('--goal_dir', type=int, default=1,
                    help='0916: 1=传goal_dir激活goalmlp(ep5系ckpt必需) / 0=回退0902 dense行为')
    ap.add_argument('--escape', type=int, default=1, help='卡死/转圈脱困原语(0903)')
    ap.add_argument('--dock', type=int, default=0, help='终段伺服(0903): S2喊STOP后朝缓存goal几何贴近收A类')
    ap.add_argument('--dock_target', type=float, default=1.0, help='伺服到该距离(米)内即停')
    ap.add_argument('--dock_cap', type=float, default=4.0, help='目标超过该距离(米)判定走过头, 不掺和')
    ap.add_argument('--dock_max_steps', type=int, default=12, help='伺服最大步数上限')
    ap.add_argument('--slow_url', default=None, help='覆盖SLOW地址(并行多实例)')
    ap.add_argument('--out', default='results_hybrid')
    cli = ap.parse_args()

    global SLOW
    if cli.slow_url:
        SLOW = cli.slow_url

    device = 'cuda:0'
    print('==== loading our fast brain ... ====', flush=True)
    agent = OurAgent(ckpt=cli.ckpt, hfov=79, device=device)
    requests.get(SLOW + '/health', timeout=10).raise_for_status()
    print('==== slow brain server OK ====', flush=True)

    env = make_env('val_unseen', 640, 480, 79)
    eps = list(env.episodes)
    if cli.ep_ids:
        want = [str(i) for i in cli.ep_ids]
        by_id = {str(e.episode_id): e for e in eps}
        eps = [by_id[i] for i in want if i in by_id]
        missing = [i for i in want if i not in by_id]
        if missing:
            print(f'[warn] ep_ids not found: {missing}', flush=True)
    else:
        random.Random(cli.seed).shuffle(eps)
        eps = eps[:cli.sample][:cli.n]
    env.episodes = eps
    print(f'==== HYBRID probe: {len(eps)} eps, slow=DualVLN-S2(http), fast={cli.ckpt} ====', flush=True)

    results = open(cli.out + '.jsonl', 'a', encoding='utf-8')   # append: 断点续跑
    done_ids = set()
    if os.path.exists(cli.out + '.jsonl'):
        for l in open(cli.out + '.jsonl'):
            try:
                done_ids.add(str(json.loads(l)['episode_id']))
            except Exception:
                pass
        if done_ids:
            print(f'[resume] skip {len(done_ids)} done eps', flush=True)
    for ei in range(len(eps)):
        if str(eps[ei].episode_id) in done_ids:
            continue
        env.episodes = [eps[ei]]
        obs = env.reset()
        ep = eps[ei]
        instruction = obs['instruction']['text'] if isinstance(obs.get('instruction'), dict) \
            else str(ep.instruction.instruction_text)
        requests.post(SLOW + '/reset', timeout=10)
        agent.reset_episode()
        goal_world = None
        act_queue = []
        steps, min_dtg = 0, float('inf')
        stopped = False
        s2_calls = 0
        esc = Escape()
        dock_steps = 0
        last_goal = None
        prev_xy = None
        pending_act = None
        print(f'--- ep {ep.episode_id}: {instruction[:80]}', flush=True)

        while not env.episode_over and steps < cli.max_steps:
            state = env.sim.get_agent_state()
            rgb = np.ascontiguousarray(obs['rgb'][:, :, :3])
            depth = np.squeeze(np.ascontiguousarray(obs['depth'])).astype(np.float32)
            x, y, th, floor = agent.hab_pose_to_ours(state.position, state.rotation)
            pts = agent.depth_to_world(depth, np.asarray(state.position, np.float64), state.rotation)
            agent.world_pts = np.concatenate([agent.world_pts, pts])[-600000:]

            m = env.get_metrics()
            min_dtg = min(min_dtg, float(m.get('distance_to_goal', -1)))

            moved = 0.0 if prev_xy is None else float(np.hypot(x - prev_xy[0], y - prev_xy[1]))
            prev_xy = (x, y)
            # 0903脱困优先: 吃上一个动作的实际位移
            if cli.escape and pending_act is not None:
                if goal_world is not None:
                    dd_e = goal_world - np.array([x, y])
                    c_e, s_e = np.cos(th), np.sin(th)
                    ge = np.array([c_e * dd_e[0] + s_e * dd_e[1], -s_e * dd_e[0] + c_e * dd_e[1]])
                else:
                    ge = np.array([1.0, 0.0])
                e = esc.update(pending_act, moved, ge, steps)
                if e is not None:
                    obs = env.step(ACT2ID[e])
                    pending_act = e
                    steps += 1
                    continue

            act = None
            if act_queue:
                act = act_queue.pop(0)
            else:
                need_s2 = goal_world is None or steps % cli.k_replan == 0
                if goal_world is not None and not need_s2:
                    if np.linalg.norm(goal_world - np.array([x, y])) < 0.5:
                        need_s2 = True
                if need_s2:
                    r = s2_call(rgb, instruction, True)
                    s2_calls += 1
                    if r['type'] == 'pixel':
                        goal_world = pixel_to_world_goal(r['pixel'][1], r['pixel'][0],
                                                         depth, x, y, th)
                        if goal_world is None:
                            act = 'move_forward'
                        else:
                            last_goal = goal_world
                    elif r['type'] == 'stop':
                        dock_goal = goal_world if goal_world is not None else last_goal
                        if cli.dock and dock_goal is not None:
                            if goal_world is None:
                                print(f'  [dock] STOP时goal_world=None, 用last_goal '
                                      f'dist={np.linalg.norm(dock_goal - np.array([x, y])):.2f}', flush=True)
                            obs, steps, min_dtg, dock_steps = dock_servo(
                                env, agent, obs, dock_goal, x, y, th, steps, min_dtg, cli)
                        else:
                            obs = env.step(ACT2ID['stop'])
                        stopped = True
                        break
                    elif r['type'] == 'actions':
                        act_queue = [ARROW2ACT[a] for a in r['actions'] if a in ARROW2ACT]
                        act = act_queue.pop(0) if act_queue else 'move_forward'
                    else:
                        act = 'move_forward'
                else:
                    s2_call(rgb, instruction, False)

            if act is None and goal_world is not None:
                dd = goal_world - np.array([x, y])
                c, s = np.cos(th), np.sin(th)
                goal_ego = np.array([c * dd[0] + s * dd[1], -s * dd[0] + c * dd[1]], np.float32)
                if np.linalg.norm(dd) < 0.5:
                    goal_world = None
                    continue
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
                # 0916: goalmlp 系 ckpt(ep5)必须传 goal_dir 才能激活 goal_mlp 那一路,
                # 否则 OurAgent._cond_offset 的 `if goal_dir is not None` 会跳过它、退化成 dense。
                # 口径与 eval_r2r_escape.py:492-495 逐字一致(同样的 [5.0, 2.0] 归一化常数)。
                _inp = {'camera_feature': cam, 'lidar_feature': bev, 'status_feature': st}
                if cli.goal_dir:
                    _inp['goal_dir'] = agent.torch.tensor(
                        goal_ego / np.array([5.0, 2.0], np.float32),
                        device=agent.device).unsqueeze(0)
                with agent.torch.no_grad():
                    traj = agent.model(_inp)['trajectory'][0].cpu().numpy()
                act, bend = agent._traj_to_act(traj)
            if act is None:
                act = 'move_forward'

            obs = env.step(ACT2ID[act])
            pending_act = act
            steps += 1

        m = env.get_metrics()
        ne = float(m.get('distance_to_goal', float('nan')))
        succ = float(m.get('success', 0.0))
        rec = {'episode_id': str(ep.episode_id), 'steps': steps, 's2_calls': s2_calls,
               'n_escape': esc.n_escape, 'dock': dock_steps,
               'success': succ, 'os': float(min_dtg < 3.0), 'ne': ne,
               'min_dtg': min_dtg, 'spl': float(m.get('spl', 0.0)), 'stopped': stopped}
        results.write(json.dumps(rec) + '\n')
        results.flush()
        print(f'[{ei+1}/{len(eps)}] ep{ep.episode_id} steps={steps} s2={s2_calls} esc={esc.n_escape} '
              f'dock={dock_steps} min_dtg={min_dtg:.2f} ne={ne:.2f} succ={succ:.0f}', flush=True)

    results.close()
    rows = [json.loads(l) for l in open(cli.out + '.jsonl')]
    n = len(rows)
    print(f'==== HYBRID n={n} SR={sum(r["success"] for r in rows)/n*100:.1f} '
          f'OSR={sum(r["os"] for r in rows)/n*100:.1f} '
          f'NE={np.mean([r["ne"] for r in rows]):.2f} '
          f'SPL={np.mean([r["spl"] for r in rows]):.3f} ====', flush=True)


if __name__ == '__main__':
    main()
