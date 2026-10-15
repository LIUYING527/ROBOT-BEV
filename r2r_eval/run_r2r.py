#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLN-CE R2R 官方评测客户端(habitat-lab 官方 sim + 官方 measurements)。
agent 无关: 通过 InternNav AgentServer HTTP 协议驱动任意 agent 服务器
(DualVLN start_server.py 或我们的 our_eval/our_server.py)。

口径 = DualVLN 官方 VLN-CE R2R val_unseen(README 数字 NE4.05/OS70.7/SR64.3/SPL58.5):
  640x480 hfov79 RGB-D / forward 0.25m / turn ±15° / 成功阈 3.0m / max 500 步。
  动作: 0stop 1forward 2left 3right; agent 返回 -1(look_down帧) 则不推进仿真,
  连续 max_noop 次强制 turn_left 防死锁(同 LH-VLN run_eval.py 约定)。

用法(lhvln env, 任意 cwd):
  CUDA_VISIBLE_DEVICES=3 HABITAT_GPU_ID=0 python run_r2r.py \
      --port 8023 --split val_unseen --max_episodes 10 --out results_dualvln_smoke.jsonl
"""
import os
import sys
import json
import gzip
import base64
import pickle
import argparse
import time

import numpy as np
import requests

DATA_ROOT = '/data/DongBaorong/BEVLOOK/benchmarks/mp3d_r2r'

# internvla_n1(DualVLN) 默认 model_settings —— 与 r2r_dualvln_server_cfg.py 一致
DUALVLN_SETTINGS = {
    'env_num': 1, 'sim_num': 1,
    'policy_name': 'InternVLAN1_Policy',
    'state_encoder': None,
    'model_path': '/data/DongBaorong/BEVLOOK/benchmarks/checkpoints/InternVLA-N1-DualVLN',
    'camera_intrinsic': None,
    'width': 640, 'height': 480, 'hfov': 79,
    'resize_w': 384, 'resize_h': 384,
    'max_new_tokens': 1024,
    'num_frames': 32, 'num_history': 8, 'num_future_steps': 4,
    'device': 'cuda:0',
    'predict_step_nums': 32,
    'continuous_traj': True,
    'infer_mode': 'partial_async',
    'vis_debug': False,
    'vis_debug_path': './logs/r2r_dualvln/vis_debug',
}

ID2ACTION = {0: 0, 1: 1, 2: 2, 3: 3}  # agent id -> habitat action index(顺序同 task config: stop/move_forward/turn_left/turn_right)


class AgentHTTPClient:
    """InternNav AgentServer 协议(/agent/init, /agent/{name}/step, /agent/{name}/reset)。"""

    def __init__(self, host, port, model_settings):
        self.base_url = f'http://{host}:{port}'
        cfg = {
            'server_host': host, 'server_port': port,
            'model_name': model_settings.pop('model_name', 'internvla_n1'),
            'ckpt_path': '',
            'model_settings': model_settings,
        }
        r = requests.post(f'{self.base_url}/agent/init', json={'agent_config': cfg}, timeout=900)
        r.raise_for_status()
        self.agent_name = r.json()['agent_name']

    def step(self, obs_list):
        payload = base64.b64encode(pickle.dumps(obs_list)).decode('utf-8')
        r = requests.post(f'{self.base_url}/agent/{self.agent_name}/step',
                          json={'observation': payload}, timeout=600)
        r.raise_for_status()
        return r.json()['action']

    def reset(self, reset_index=None):
        r = requests.post(f'{self.base_url}/agent/{self.agent_name}/reset',
                          json={'reset_index': reset_index}, timeout=60)
        r.raise_for_status()


def make_env(split, width, height, hfov, gpu_id=0):
    import habitat
    config = habitat.get_config('benchmark/nav/vln_r2r.yaml')
    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = os.path.join(
            DATA_ROOT, 'r2r', '{split}', '{split}.json.gz')
        config.habitat.dataset.scenes_dir = os.path.join(DATA_ROOT, 'scenes')
        agent_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        agent_cfg.rgb_sensor.width = width
        agent_cfg.rgb_sensor.height = height
        agent_cfg.rgb_sensor.hfov = hfov
        agent_cfg.depth_sensor.width = width
        agent_cfg.depth_sensor.height = height
        agent_cfg.depth_sensor.hfov = hfov
        # ⭐⭐⭐ 0918 重新修复: habitat 默认 normalize_depth=True, depth 返回 [0,1] 而不是米。
        #   下游全部按米用 ⇒ 反投影距离被压缩 10 倍, `dist<stop_dist` 恒成立, 判停/前进全废。
        #   09-12 修过一次, 但本文件后来被还原(实测 0918 reset 后 depth.max=0.8318 而
        #   distance_to_goal=5.035 —— 与 09-12 memory 里的数字逐位相同)。**不要再丢**。
        agent_cfg.depth_sensor.normalize_depth = False
        # 防回归护栏: 这行被"还原"过两次(0912 修好 → 0918 发现又坏了), 每次都要烧一整周实验。
        # 断言的是**配置对象本身**, 任何 yaml/默认值覆盖都会在这里炸, 而不是静默跑出一堆废结果。
        assert agent_cfg.depth_sensor.normalize_depth is False, \
            'depth normalize_depth 必须为 False(米)! 归一化会让全链路距离差 10 倍'
        config.habitat.simulator.turn_angle = 15
        config.habitat.simulator.forward_step_size = 0.25
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id
        config.habitat.environment.max_episode_steps = 500
        config.habitat.task.measurements.success.success_distance = 3.0  # VLN-CE 官方阈值
    return habitat.Env(config)


def run_episode(env, client, max_noop=20, verbose=True):
    obs = env.reset()
    client.reset(reset_index=[0])
    ep = env.current_episode
    instruction = obs['instruction']['text'] if isinstance(obs.get('instruction'), dict) else str(ep.instruction.instruction_text)
    if verbose:
        print(f'--- episode {ep.episode_id} scene={ep.scene_id.split("/")[-2]}: {instruction[:90]}', flush=True)

    noop = 0
    min_dtg = float('inf')
    steps = 0
    while not env.episode_over:
        state = env.sim.get_agent_state()
        rgb = np.ascontiguousarray(obs['rgb'][:, :, :3])
        depth = np.ascontiguousarray(obs['depth']).astype(np.float32)
        ret = client.step([{
            'rgb': rgb,
            'depth': depth,
            'instruction': instruction,
            'position': [float(v) for v in state.position],
            'rotation': [float(state.rotation.x), float(state.rotation.y),
                         float(state.rotation.z), float(state.rotation.w)],
        }])
        act_id = int(ret[0]['action'][0] if isinstance(ret[0], dict) else ret[0])

        if act_id in ID2ACTION:
            noop = 0
            hab_act = ID2ACTION[act_id]
        else:  # -1 look_down 帧 / 未知: 不推进仿真
            noop += 1
            if noop > max_noop:
                hab_act = 2  # turn_left 防死锁
                noop = 0
            else:
                continue

        obs = env.step(hab_act)
        steps += 1
        m = env.get_metrics()
        if 'distance_to_goal' in m:
            min_dtg = min(min_dtg, float(m['distance_to_goal']))
        if verbose and steps % 25 == 0:
            print(f'  step={steps} dtg={m.get("distance_to_goal", -1):.2f}', flush=True)

    m = env.get_metrics()
    return {
        'episode_id': str(ep.episode_id),
        'scene': ep.scene_id.split('/')[-2],
        'instruction': instruction,
        'steps': steps,
        'success': float(m.get('success', 0.0)),
        'spl': float(m.get('spl', 0.0)),
        'ne': float(m.get('distance_to_goal', float('nan'))),
        'os': float(min_dtg < 3.0),  # oracle success: 轨迹曾入 3m 阈
        'min_dtg': min_dtg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=8023)
    ap.add_argument('--split', default='val_unseen')
    ap.add_argument('--max_episodes', type=int, default=0)
    ap.add_argument('--sample', type=int, default=0, help='从整个split随机抽N条(seed固定, 跨场景代表性)')
    ap.add_argument('--ep_ids', nargs='+', default=None,
                    help='显式指定episode id列表(跨进程配对评测; 0902发现env.episodes顺序进程间不确定)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--hfov', type=int, default=79)
    ap.add_argument('--model_name', default='internvla_n1')
    ap.add_argument('--model_settings', default='{}', help='JSON, 合并进 init 请求')
    ap.add_argument('--out', default='r2r_results.jsonl')
    cli = ap.parse_args()

    ms = json.loads(cli.model_settings)
    if cli.model_name == 'internvla_n1':
        ms = {**DUALVLN_SETTINGS, **ms}
    ms['model_name'] = cli.model_name
    client = AgentHTTPClient(cli.host, cli.port, ms)

    env = make_env(cli.split, cli.width, cli.height, cli.hfov)
    n_total = len(env.episodes)
    if cli.ep_ids:
        want = [str(i) for i in cli.ep_ids]
        by_id = {str(e.episode_id): e for e in list(env.episodes)}
        env.episodes = [by_id[i] for i in want if i in by_id]
        print(f'==== R2R {cli.split}: pinned {len(env.episodes)} eps by id ====', flush=True)
    elif cli.sample and cli.sample < n_total:
        import random
        eps = list(env.episodes)
        random.Random(cli.seed).shuffle(eps)
        env.episodes = eps[:cli.sample]
        print(f'==== R2R {cli.split}: sampled {cli.sample}/{n_total} (seed={cli.seed}, 跨场景代表性子集) ====', flush=True)
    n_total = len(env.episodes)
    n_run = min(cli.max_episodes, n_total) if cli.max_episodes else n_total
    print(f'==== running {n_run} episodes ====', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    fout = open(cli.out, 'a', encoding='utf-8')
    results = []
    # 断点续跑(0828: robo2vlm 宕机致 61 条 error 的补跑): 跳过 --out 里已有非 error 结果的 episode
    done_ids = set()
    if os.path.exists(cli.out):
        for ln in open(cli.out):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if 'episode_id' in r and 'error' not in r:
                done_ids.add(str(r['episode_id']))
        if done_ids:
            print(f'==== resume: skip {len(done_ids)} done episodes ====', flush=True)
    # 显式逐episode指派(gen_adapt_data 验证过的可靠模式):
    # habitat reset() 的内部迭代器与 env.episodes 列表序不可靠对齐(0829 topup 实锤:
    # reset 跳读 239 条只覆盖 19/61, 42 条重复) → 每条 env.episodes=[ep] 再 reset
    all_eps = list(env.episodes)
    for i in range(n_run):
        t0 = time.time()
        ep = all_eps[i]
        if str(ep.episode_id) in done_ids:
            continue
        env.episodes = [ep]
        try:
            rec = run_episode(env, client)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec = {'episode_id': f'idx{i}', 'error': str(e)}
        rec['time_s'] = round(time.time() - t0, 1)
        results.append(rec)
        fout.write(json.dumps(rec) + '\n')
        fout.flush()
        if 'error' not in rec:
            print(f'[EP {i+1}/{n_run}] succ={rec["success"]:.0f} spl={rec["spl"]:.2f} '
                  f'ne={rec["ne"]:.2f} os={rec["os"]:.0f} ({rec["time_s"]:.0f}s)', flush=True)

    ok = [r for r in results if 'error' not in r]
    if ok:
        agg = {k: float(np.mean([r[k] for r in ok])) for k in ('success', 'spl', 'ne', 'os')}
        print('\n==== AGG (n=%d) ====' % len(ok))
        for k, v in agg.items():
            print(f'  {k}: {v:.4f}')
        fout.write(json.dumps({'final_metrics': agg, 'n': len(ok)}) + '\n')
    fout.close()
    env.close()


if __name__ == '__main__':
    main()
