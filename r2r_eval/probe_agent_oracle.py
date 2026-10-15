#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端验证(0914): OurAgent.act_r2r 完整链路 + oracle 定位。

为什么需要它:
  79.6% 出自 probe_oracle_goal.py, 但它【绕过 act_r2r】直接调模型 —— 所以证明不了
  "EscapeV3 移植进 act_r2r 之后在真实闭环里也成立"。
  本探针把 act_r2r 的慢脑定位(vlm_locate)换成 oracle 目标点, 其余全走 act_r2r 真实代码:
  地标链/到达判定/confirm判停/快脑出轨迹/脱困接线。不依赖 VLM 服务。

口径说明(与 probe_oracle_goal.py 的差异, 不可直接比数):
  停止机制不同: 本探针是 agent 的 confirm 判停(stop头), 探针是几何停(dtg<stop_dist)。
  过程指标 min_dtg / os(=min_dtg<3m) 不依赖停止机制, 可与探针横向参考。

用法(lhvln env):
  CUDA_VISIBLE_DEVICES=5 HABITAT_GPU_ID=0 python probe_agent_oracle.py \
      --ckpt goalnav_r2r_dense_goalmlp_ep5.pt --escape 3 --v3_noprog 2 --n 50 --out agent_oracle_esc3
"""
import os
import sys
import json
import random
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, '/data/DongBaorong/BEVLOOK/benchmarks/LH-VLN/our_eval')

from run_r2r import make_env  # noqa: E402
from our_agent import OurAgent  # noqa: E402

ACT2ID = {'stop': 0, 'move_forward': 1, 'turn_left': 2, 'turn_right': 3}
GOAL_AHEAD = 4.0


def densify_ours(ref_ours, ds=0.25):
    """稀疏路点 → 0.25m 稠密插值(与 probe_oracle_goal.py 同口径, 直接影响 oracle 目标质量)。"""
    pts = [np.array(ref_ours[0], np.float64)]
    acc = 0.0
    for i in range(1, len(ref_ours)):
        a, b = np.array(ref_ours[i - 1], np.float64), np.array(ref_ours[i], np.float64)
        seg = np.linalg.norm(b - a)
        if seg < 1e-6:
            continue
        d = ds - acc
        while d < seg:
            pts.append(a + (b - a) * (d / seg))
            d += ds
        acc = (seg - (d - ds)) % ds
    pts.append(np.array(ref_ours[-1], np.float64))
    return np.stack(pts)


def oracle_goal_ego(ref_ours, x, y, th):
    """ref_path 上最近点沿弧长前瞻 GOAL_AHEAD → ego系(x前,y左)。与探针逐行一致。"""
    d = ref_ours - np.array([x, y])
    i0 = int(np.argmin(np.linalg.norm(d, axis=1)))
    acc, gp = 0.0, ref_ours[-1]
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
    return np.array([c * dd[0] + s * dd[1], -s * dd[0] + c * dd[1]], np.float32), d_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='goalnav_r2r_dense_goalmlp_ep5.pt')
    ap.add_argument('--hfov', type=int, default=79)
    ap.add_argument('--split', default='val_unseen')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--sample', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ep_ids', nargs='+', default=None)
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--stop_dist', type=float, default=1.5)
    ap.add_argument('--escape', type=int, default=3, help='0=关 / 1/2/3')
    ap.add_argument('--v3_noprog', type=int, default=2)
    ap.add_argument('--turn_deg', type=float, default=30.0)
    ap.add_argument('--stop_head', default='/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/runs/stop_head_r2r.pt')
    ap.add_argument('--no_dense_goal', action='store_true')
    ap.add_argument('--vlm_every', type=int, default=0,
                    help='真VLM模式下调稀定位频率(0=用agent默认3)。VLM单次推理秒级, 跑长链路必须调稀')
    ap.add_argument('--real_vlm', action='store_true',
                    help='★用真 VLM 定位(不替换 vlm_locate), 并记录每次定位点到参考路径的最近距离 '
                         '→ 这才是"真链路"。需要 VLM 服务在线。')
    ap.add_argument('--loc_noise', type=float, default=0.0,
                    help='给 oracle 定位注入高斯噪声(米), 模拟 VLM 定位误差。'
                         '横向 sigma=loc_noise(bbox角度误差为主), 前向=0.5*loc_noise(深度误差较小)。'
                         '同一 ep 内噪声序列固定 => 开关两组配对公平。')
    ap.add_argument('--geo_aux', action='store_true',
                    help='启用 our_agent 的判停兜底(链走完+离最后到达地标<geo_aux_d → 停)')
    ap.add_argument('--geo_aux_d', type=float, default=1.0)
    ap.add_argument('--geo_stop', action='store_true',
                    help='用几何停(到终点<stop_dist)替掉 stop_head 判停 —— 隔离"判停不可靠"这个变量, '
                         '使本探针与 probe_oracle_goal.py 同停止口径, 可直接比 escape 的贡献')
    ap.add_argument('--log_stop', action='store_true',
                    help='记录每步 stop_head 概率 + 真实到终点距离 → 离线看 "真到了的时候 stop头到底给多少分"')
    ap.add_argument('--trace', type=int, default=0, help='每N步打印位姿/动作/调试串(0=关)')
    ap.add_argument('--out', default='probe_agent_oracle')
    cli = ap.parse_args()

    env = make_env(cli.split, 640, 480, cli.hfov)   # 与 probe_oracle_goal.py 同口径
    eps = list(env.episodes)
    if cli.ep_ids:
        by_id = {str(e.episode_id): e for e in eps}
        eps = [by_id[i] for i in cli.ep_ids if i in by_id]
    else:
        random.Random(cli.seed).shuffle(eps)
        eps = eps[:cli.sample][:cli.n]
    env.episodes = eps

    sh = cli.stop_head if (cli.stop_head and os.path.exists(cli.stop_head)) else None
    print(f'==== agent-oracle probe: {len(eps)} eps, ckpt={cli.ckpt} escape={cli.escape} '
          f'v3_noprog={cli.v3_noprog} stop_head={"yes" if sh else "None"} ====', flush=True)
    agent = OurAgent(ckpt=cli.ckpt, hfov=cli.hfov, escape=cli.escape,
                     v3_noprog=cli.v3_noprog, turn_deg=cli.turn_deg, stop_head=sh,
                     geo_aux=cli.geo_aux, geo_aux_d=cli.geo_aux_d,
                     vlm_every=(cli.vlm_every if cli.vlm_every > 0 else 3))
    _ORIG_SHP = agent.stop_head_prob      # 保存原始方法, 循环内 patch 时必须用它, 否则会嵌套套娃

    results = open(cli.out + '.jsonl', 'w', encoding='utf-8')
    for ei, ep in enumerate(eps):
        env.episodes = [ep]
        obs = env.reset()
        ref = np.array(ep.reference_path, np.float64)
        ref_ours = np.stack([ref[:, 0], -ref[:, 2]], 1)
        if not cli.no_dense_goal:
            ref_ours = densify_ours(ref_ours)

        agent.reset_episode()
        # oracle 模式: 注入假地标链, 跳过 VLM 解析(目标由 oracle 提供)。
        # ★real_vlm 模式【不能】注入: 那会让 VLM 去 Det​ect "oracle_landmark" 这种不存在的词 →
        # 164/164 次全部返回 None(0914 实测踩过)。真链路必须让 agent 用真实指令自己解析地标链。
        if not cli.real_vlm:
            agent.r2r_chain = ['oracle_landmark']
            agent.r2r_li = 0
        cur = {'x': 0.0, 'y': 0.0, 'th': 0.0}

        # 定位噪声: 每个 ep 一个固定 seed 的序列 —— 这样"加噪声"这个条件在开关两组间完全一致,
        # 差异只能来自 agent 的策略本身(配对公平), 不会混入随机噪声的运气。
        loc_rng = np.random.RandomState(100000 + ei)

        def oracle_locate(rgb, depth, target, room=None, _r=ref_ours, _c=cur,
                          _rng=loc_rng, _sig=cli.loc_noise):
            ego, d_end = oracle_goal_ego(_r, _c['x'], _c['y'], _c['th'])
            _c['d_end'] = d_end          # 到【终点】距离(上帝视角) —— geo_stop 与 dtg 口径都要用
            if _sig > 0:
                # VLM 定位误差结构: bbox 横向角误差 → 主要打在 ego-y(横向);
                # 深度中值误差较小 → 前向 x 用半个 sigma。
                ego = (ego + np.array([_rng.normal(0, _sig * 0.5),
                                       _rng.normal(0, _sig)], np.float32)).astype(np.float32)
            return ego
        if cli.real_vlm:
            # ★真链路: 用真 VLM 的结果驱动, 同时记录"定位点世界坐标 → 参考路径最近距离"。
            # 为什么用"到参考路径的距离"当偏差度量: 指令描述的地标本就该在参考路径沿线,
            # VLM 定位若准, 结果应落在路径附近。这是可解释的、不依赖"地标=前瞻点"的错误假设。
            _ORIG_LOC = agent.vlm_locate

            def real_locate(rgb, depth, target, room=None, _o=_ORIG_LOC, _c=cur, _r=ref_ours):
                g = None
                try:
                    g = _o(rgb, depth, target, room)
                except Exception as e:
                    loc_log.append([int(_c.get('step', -1)), -1.0, -1.0, -1.0, 0.0, str(type(e).__name__)])
                    return None
                if g is not None and _r is not None:
                    c, s = np.cos(_c['th']), np.sin(_c['th'])
                    gw = np.array([_c['x'] + c * g[0] - s * g[1],
                                   _c['y'] + s * g[0] + c * g[1]])
                    d_path = float(np.min(np.linalg.norm(_r - gw, axis=1)))
                    loc_log.append([int(_c.get('step', -1)), float(g[0]), float(g[1]),
                                    float(d_path), 1.0, ''])
                else:
                    loc_log.append([int(_c.get('step', -1)), -1.0, -1.0, -1.0, 0.0, 'none'])
                return g

            agent.vlm_locate = real_locate
        else:
            agent.vlm_locate = oracle_locate

        if cli.geo_stop:
            def geo_confirm(rgb, instruction, near_geo=False, _c=cur, _d=cli.stop_dist):
                return _c.get('d_end', 1e9) < _d
            agent._confirm_stop = geo_confirm

        ins = ''
        try:
            ins = ep.instruction.instruction_text
        except Exception:
            pass

        st_log = []
        loc_log = []
        if cli.log_stop and agent.stop_head is not None:
            def _logged(rgb, _o=_ORIG_SHP, _l=st_log, _c=cur):
                p = _o(rgb)
                _l.append([float(_c['x']), float(_c['y']), float(_c.get('d_end', -1.0)), float(p)])
                return p
            agent.stop_head_prob = _logged

        steps, stopped, min_dtg = 0, False, float('inf')
        while not env.episode_over and steps < cli.max_steps:
            st = env.sim.get_agent_state()
            position = np.asarray(st.position, np.float64)
            rotation = st.rotation
            x, y, th, _fl = agent.hab_pose_to_ours(position, rotation)
            cur.update(x=x, y=y, th=th, step=steps)
            m = env.get_metrics()
            dtg = float(m.get('distance_to_goal', -1))
            min_dtg = min(min_dtg, dtg)
            obs_dict = {'color_sensor_f': np.ascontiguousarray(obs['rgb'][:, :, :3]),
                        'depth_sensor_f': np.squeeze(np.ascontiguousarray(obs['depth'])).astype(np.float32)}
            a, dbg = agent.act_r2r(obs_dict, position, rotation, ins)
            if cli.trace and steps % cli.trace == 0:
                print(f'  [{steps:3d}] pos=({x:7.3f},{y:7.3f}) th={th:6.3f} a={a:12s} | {dbg}', flush=True)
            if a == 'stop':
                stopped = True
                break
            obs = env.step(ACT2ID[a])
            steps += 1

        m = env.get_metrics()
        ne = float(m.get('distance_to_goal', float('nan')))
        rec = {'episode_id': str(ep.episode_id), 'steps': steps, 'n_escape': agent.n_escape,
               'success': float(stopped and ne < 3.0), 'os': float(min_dtg < 3.0),
               'ne': ne, 'min_dtg': min_dtg, 'stopped': float(stopped)}
        if cli.log_stop:
            rec['stop_trace'] = st_log
        if cli.real_vlm:
            rec['loc_trace'] = loc_log
        results.write(json.dumps(rec) + '\n')
        results.flush()
        print(f'[{ei+1}/{len(eps)}] ep{ep.episode_id} steps={steps} esc={agent.n_escape} '
              f'min_dtg={min_dtg:.2f} ne={ne:.2f} succ={rec["success"]:.0f}', flush=True)

    results.close()
    rows = [json.loads(l) for l in open(cli.out + '.jsonl')]
    n = len(rows)
    print(f'==== AGENT-ORACLE n={n} SR={sum(r["success"] for r in rows)/n*100:.1f} '
          f'OSR={sum(r["os"] for r in rows)/n*100:.1f} '
          f'NE={np.mean([r["ne"] for r in rows]):.2f} '
          f'esc_eps={sum(1 for r in rows if r["n_escape"]>0)}/{n} ====', flush=True)


if __name__ == '__main__':
    main()
