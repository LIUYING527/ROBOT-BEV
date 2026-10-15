#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接线验证(0914): OurAgent.act_r2r 里的 EscapeV3 是否真的被触发、动作序列是否与直接调用一致。

不依赖 habitat / VLM:
  - 直接实例化 OurAgent(需 torch + ckpt), 注入 target_world 绕过慢脑定位
  - 位置固定在原地不动 → moved≈0 → 连续 move_forward 后必须触发卡死脱困
  - 同时用【同样的输入】直接喂一个独立的 EscapeV3, 对比两边的输出动作序列

这就是"移植是否接对线"的判据: 若 act_r2r 内部触发点/动作序列与独立 EscapeV3 不一致,
说明接线错位(口径、顺序、动作名、step 语义任一处)。
"""
import os
import sys

import numpy as np

_OUR_EVAL = '/data/DongBaorong/BEVLOOK/benchmarks/LH-VLN/our_eval'
sys.path.insert(0, _OUR_EVAL)

from escape_v3 import EscapeV3  # noqa: E402


def main():
    import quaternion as npq
    from our_agent import OurAgent

    agent = OurAgent(ckpt='goalnav_r2r_dense_goalmlp_ep5.pt', device='cuda', hfov=79)
    agent.reset_episode()
    print(f'[wiring] escape_mode={agent.escape_mode} cls={type(agent.escape).__name__} '
          f'NO_PROG_N={EscapeV3.NO_PROG_N} TURN_DEG={type(agent.escape).__mro__[1].TURN_DEG}', flush=True)

    # --- 注入目标: 正前方 3m(>1m 到达阈, <45 步超时) ---
    agent.r2r_chain = ['wiring_test_landmark']
    agent.r2r_li = 0
    agent.target_world = np.array([3.0, 0.0], np.float32)
    agent._tw_is_landmark = True

    H, W = 480, 640
    obs = {'color_sensor_f': np.zeros((H, W, 3), np.uint8),
           'depth_sensor_f': np.full((H, W), 3.0, np.float32)}
    pos = np.array([0.0, 1.0, 0.0], np.float64)          # 原地不动
    rot = npq.quaternion(1, 0, 0, 0)

    # 镜像: 独立 EscapeV3, 用 act_r2r 内部实际喂进去的同样参数逐步复算
    ref = EscapeV3()
    ref_out = []
    ref_trig = []
    mismatch = []
    wired_out = []

    N = 40
    for i in range(N):
        # ⚠️ 时序: act_r2r 内部的 escape 判定发生在【慢脑之前】, 所以它吃到的 goal_ego/dtg
        #    是上一帧慢脑留下的值。参考实现必须用同一时刻的值 → 在调用前读。
        pre_goal, pre_dtg = agent._esc_goal_ego, agent._esc_dtg
        act, dbg = agent.act_r2r(obs, pos, rot, 'go forward')
        wired_out.append(act)
        prev_act = wired_out[i - 1] if i > 0 else None
        if prev_act is not None and pre_goal is not None:
            ref_prev = ref_out[-1] if ref_out else prev_act
            x_, y_, th_, _ = agent.hab_pose_to_ours(pos, rot)
            ego_pts = agent._world_pts_ego(x_, y_, th_)
            e = ref.update(ref_prev, 0.0, pre_goal, agent.step_i, ego_pts=ego_pts, dtg=pre_dtg)
            ref_trig.append(e is not None)
            ref_out.append(e if e is not None else prev_act)
            wired_esc = dbg.startswith('escape#')
            ref_esc = e is not None
            if ref_esc != wired_esc:
                mismatch.append((i, f'触发不一致 wired={wired_esc} ref={ref_esc}', act, dbg))
            elif ref_esc and act != e:
                mismatch.append((i, f'动作不一致 wired={act} ref={e}', act, dbg))
        if i < 12 or dbg.startswith('escape#'):
            print(f'  [{i:02d}] act={act:12s} | {dbg}', flush=True)

    print()
    print(f'[wiring] act_r2r(escape=3) : 共 {N} 步, escape 接管 {agent.n_escape} 步')
    print(f'[wiring] 独立 EscapeV3 同输入: 触发 {sum(ref_trig)} 步')

    # --- 对照: escape=0(无脱困) 跑同样输入, 看重放同一个模型行为下结局差多少 ---
    off = OurAgent(ckpt='goalnav_r2r_dense_goalmlp_ep5.pt', device='cuda', hfov=79, escape=0)
    off.reset_episode()
    off.r2r_chain = ['wiring_test_landmark']
    off.r2r_li = 0
    off.target_world = np.array([3.0, 0.0], np.float32)
    off._tw_is_landmark = True
    off_out = [off.act_r2r(obs, pos, rot, 'go forward')[0] for _ in range(N)]
    # 终点距离: 位置不动 → 只看"有没有朝目标走"的意图
    def net_turn(seq):
        return sum(30.0 * ({'turn_left': 1, 'turn_right': -1}.get(a, 0)) for a in seq)

    def dist(seq):
        return {k: int(v) for k, v in zip(*np.unique(seq, return_counts=True))}

    print(f'[wiring] act_r2r(escape=0): 动作分布 {dist(off_out)}, 净转角 {net_turn(off_out):.0f}°')
    print(f'[wiring] act_r2r(escape=3): 动作分布 {dist(wired_out)}, 净转角 {net_turn(wired_out):.0f}°')

    if mismatch:
        print(f'❌ 接线不一致 {len(mismatch)} 处:')
        for m in mismatch[:10]:
            print('   ', m)
        return 1
    print('✅ 接线一致: act_r2r 内部的 escape 触发点与动作序列, 与独立 EscapeV3 逐位相同')
    return 0


if __name__ == '__main__':
    sys.exit(main())
