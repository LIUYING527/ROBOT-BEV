#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 escape_v3.py(抽出的模块) 与 probe_oracle_goal.py(内置类) 行为逐位一致。

为什么必须做: 79.6% 的配方 = EscapeV3 的具体决策序列。抽出成模块时若有一处不等价,
标准协议链路跑出来的就不是那个配方 —— 而这种差异不会报错, 只会静静地给出不同动作。

方法: 把 probe 里的 Escape/EscapeV2/EscapeV3/free_dir_ego 源码抠出来 exec 到独立命名空间,
两套实现喂【完全相同】的输入序列(含随机序列, seed 固定), 逐步比较输出动作。
"""
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from escape_v3 import Escape as ModEscape, EscapeV2 as ModEscapeV2, EscapeV3 as ModEscapeV3  # noqa

# 权威探针(产生 79.6% 的那份)。按优先级找第一个存在的。
_PROBE_CANDS = [
    os.path.join(_HERE, 'probe_oracle_goal.py'),
    '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/probe_oracle_goal.py',
    '/data/DongBaorong/BEVLOOK/best_fastbrain_0913/probe_oracle_goal.py',
]
PROBE = next((p for p in _PROBE_CANDS if os.path.exists(p)), _PROBE_CANDS[0])


def load_probe_classes():
    """从探针源码抠出脱困相关定义(避开 habitat/torch 等重依赖)。"""
    src = open(PROBE).read()
    start = src.index('class Escape:')
    end = src.index('# 注: 0911 曾写过 EscapeV3')
    snippet = src[start:end]
    ns = {'np': np}
    exec(compile(snippet, '<probe-snippet>', 'exec'), ns)
    return ns


def make_ego_pts(rng, n=400, spread=6.0):
    """合成 ego 系点云: 环形分布 + 若干近距障碍(让"最空方向"有区分度)。"""
    ang = rng.uniform(-np.pi, np.pi, n)
    r = rng.uniform(0.5, spread, n)
    pts = np.stack([r * np.cos(ang), r * np.sin(ang)], 1).astype(np.float32)
    for k in range(6):                      # 塞几个近距障碍, 制造净空差异
        a = rng.uniform(-np.pi, np.pi)
        d = rng.uniform(0.6, 2.5)
        pts[k] = [d * np.cos(a), d * np.sin(a)]
    return pts


def run_sequence(cls, seq, seed):
    """喂一串 (act, moved, goal_ego, ego_pts, dtg) 给某个实现, 返回输出动作序列。"""
    rng = np.random.RandomState(seed)
    esc = cls()
    out = []
    for i, (act, moved, goal_ego, dtg) in enumerate(seq):
        ego_pts = make_ego_pts(rng) if cls.__name__.endswith(('V2', 'V3')) else None
        if ego_pts is None:
            e = esc.update(act, moved, np.asarray(goal_ego, np.float32), i, dtg=dtg)
        else:
            e = esc.update(act, moved, np.asarray(goal_ego, np.float32), i,
                           ego_pts=ego_pts, dtg=dtg)
        out.append(e)
        # 命中后 escape 内部会把剩余 plan 逐步 pop —— 用被接管的动作回灌, 还原真实循环
        while esc.plan:
            nxt = esc.plan.pop(0)
            esc.n_steps += 1
            out.append(nxt)
    return out


def gen_seq(kind, seed):
    """构造输入序列。"""
    rng = np.random.RandomState(seed)
    seq = []
    if kind == 'stuck':            # 连续前进但几乎不动 → 必须触发卡死脱困
        for _ in range(60):
            seq.append(('move_forward', 0.02, [3.0 + rng.randn() * 0.5, rng.randn()], 4.0))
    elif kind == 'spin':           # 交替转向原地打转 → 必须触发打转脱困
        for i in range(60):
            seq.append(('turn_left' if i % 2 == 0 else 'turn_right', 0.0,
                        [3.0, 1.0], 4.0))
    elif kind == 'mixed':          # 混合: 走-停-转-卡, 覆盖 level 递增与 no_prog
        acts = ['move_forward', 'turn_left', 'move_forward', 'move_forward',
                'turn_right', 'move_forward', 'move_forward', 'move_forward']
        for i in range(120):
            a = acts[i % len(acts)]
            mv = 0.02 if a == 'move_forward' and (i // 8) % 3 == 0 else rng.uniform(0.2, 0.4)
            seq.append((a, mv, [rng.uniform(2, 6), rng.uniform(-3, 3)], rng.uniform(1.5, 6.0)))
    elif kind == 'recover':        # 卡死-恢复-再卡死: 覆盖 no_prog 清零与 level 循环
        for blk in range(6):
            for _ in range(5):
                seq.append(('move_forward', 0.02, [2.0, 0.5], 3.0))
            for _ in range(6):
                seq.append(('move_forward', 0.30, [2.0, 0.5], 3.0 - blk * 0.4))
    return seq


def main():
    ns = load_probe_classes()
    pairs = [
        ('Escape', ns['Escape'], ModEscape),
        ('EscapeV2', ns['EscapeV2'], ModEscapeV2),
        ('EscapeV3', ns['EscapeV3'], ModEscapeV3),
    ]
    total_fail = 0
    for kind in ('stuck', 'spin', 'mixed', 'recover'):
        for seed in (0, 1, 2, 7, 20260913):
            seq = gen_seq(kind, seed)
            for name, ref_cls, mod_cls in pairs:
                ref = run_sequence(ref_cls, seq, seed)
                mod = run_sequence(mod_cls, seq, seed)
                if ref != mod:
                    total_fail += 1
                    bad = next((i for i, (a, b) in enumerate(zip(ref, mod)) if a != b),
                               min(len(ref), len(mod)))
                    print(f'  ❌ {kind}/seed{seed}/{name}: 第 {bad} 步起分歧 '
                          f'(ref={ref[bad] if bad < len(ref) else "<end>"} '
                          f'mod={mod[bad] if bad < len(mod) else "<end>"})')
                else:
                    n_esc = sum(1 for x in ref if x is not None)
                    if kind == 'stuck' and seed == 0:
                        print(f'  ✅ {kind}/seed{seed}/{name}: {len(ref)} 步全等 (脱困接管 {n_esc} 步)')
    # 汇总
    print()
    n_cases = 4 * 5 * len(pairs)
    if total_fail == 0:
        print(f'✅ 等价性通过: {n_cases} 个用例(4种序列 × 5个seed × 3个实现)输出逐位一致')
    else:
        print(f'❌ {total_fail}/{n_cases} 个用例不一致')
    return 1 if total_fail else 0


if __name__ == '__main__':
    sys.exit(main())
