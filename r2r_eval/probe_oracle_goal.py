#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase0 诊断(0831): 快脑单独上限探针 —— oracle目标点喂快脑, 去掉慢脑/stop两个变量。
goal_ego = ref_path上前瞻4m点(与train_goalnav_r2r训练口径完全一致), 到终点1.5m几何停。
回答: "慢脑欠多少 vs 快脑欠多少" —— 若oracle下SR仍低=快脑/数据问题; 若SR高=慢脑全责。

用法(lhvln env):
  CUDA_VISIBLE_DEVICES=2 HABITAT_GPU_ID=0 python probe_oracle_goal.py \
      --ckpt goalnav_r2rft.pt --n 50 --out probe_oracle_r2rft
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
_ARCH = '/data/DongBaorong/BEVLOOK/robot_bev_sim/arch_e2e/pipeline'
_SCRIPTS = '/data/DongBaorong/BEVLOOK/robot_bev_sim/scripts'
_GOALNAV = '/data/DongBaorong/BEVLOOK/robot_bev_sim/goalnav/pipeline'
for p in (_ARCH, _SCRIPTS, _GOALNAV):
    sys.path.insert(0, p)

from run_r2r import make_env  # noqa: E402
from our_agent import OurAgent  # noqa: E402
import stage_b_eval as sb  # noqa: E402
import sim_fullchain as sf  # noqa: E402
from train_goalnav_turn import status_gh  # noqa: E402
import cv2  # noqa: E402

GOAL_AHEAD = 4.0
WZ_TURN_TH = 0.35


def densify_ours(ref_ours, ds=0.25):
    """ref_ours[(n,2)] 稀疏路点 → 0.25m 稠密插值(与 gen_dense_data.densify 同口径)。"""
    pts = [np.array(ref_ours[0], np.float64)]
    acc = 0.0
    for i in range(1, len(ref_ours)):
        a = np.array(ref_ours[i - 1], np.float64)
        b = np.array(ref_ours[i], np.float64)
        seg = np.linalg.norm(b - a)
        if seg < 1e-6:
            continue
        d = ds - acc
        while d < seg:
            pts.append(a + (b - a) * (d / seg))
            d += ds
        acc = seg - (d - ds)
    pts.append(np.array(ref_ours[-1], np.float64))
    return np.stack(pts)
ACT2ID = {'stop': 0, 'move_forward': 1, 'turn_left': 2, 'turn_right': 3}


def oracle_goal_ego(ref_ours, x, y, th):
    """ref_path(我们系) 上找最近点, 沿弧长前瞻GOAL_AHEAD米 → ego系。返回(goal_ego, 到终点距离)"""
    d = ref_ours - np.array([x, y])
    dist = np.linalg.norm(d, axis=1)
    i0 = int(np.argmin(dist))
    # 弧长前瞻
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


class Escape:
    """0911移植自 probe_oracle_escape.py(0903): 卡死/转圈检测 + 固定逃脱序列。
    0903配对实锤 SR 62→70(+8pt, n=50 dense): 6/8失败=模型顶墙喊前进无自救(训练数据无卡墙状态)。
    纯推理侧, 不动权重 → 零 co-adaptation 风险。escape_plan=待执行动作队列。"""

    # 0911 参数化: 默认值 = v1 原版行为, 保证既有数字严格可比
    STUCK_N = 4          # 卡死判据: 连续 N 步
    STUCK_D = 0.15       # 卡死判据: 这 N 步总位移 < D
    GUARD_DTG = 0.0      # dtg < 此值时不触发(0=关)
    MAX_ESCAPE = 0       # 逃脱接管步数上限(0=无限)
    # 0913: 每步转向角。habitat vln_r2r.yaml 实际是 15°/步, 但历史代码全部按 30 写。
    # 默认 30.0 = 保持既有行为(所有已出数字可比); --turn_deg 15 才是真实角度。
    TURN_DEG = 30.0
    # 打转判据: 最近20步净转角 > SPIN_N_TURNS 次同向转向。原为硬编码 300°(=10*30),
    # 改写成随 TURN_DEG 缩放 —— 否则 --turn_deg 15 时 20 步上限恰好 300, 严格大于永不成立。
    SPIN_N_TURNS = 10

    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = []          # (act, moved_dist)
        self.plan = []
        self.level = 0          # 同一点第几次卡死(0=goal侧90, 1=反侧90, 2=180)
        self.last_escape_step = -100
        self.n_steps = 0        # 被逃脱序列接管的累计步数(cap 判据)

    def _blocked(self, dtg):
        """接近目标保护 / 逃脱次数上限 → 本次不触发。"""
        if self.MAX_ESCAPE and self.n_steps >= self.MAX_ESCAPE:
            return True
        if self.GUARD_DTG and dtg is not None and 0.0 <= dtg < self.GUARD_DTG:
            return True
        return False

    def update(self, act, moved, goal_ego, step, dtg=None):
        if self.plan:
            self.n_steps += 1
            return self.plan.pop(0)
        if self._blocked(dtg):
            return None
        self.hist.append((act, moved))
        self.hist = self.hist[-24:]
        # 卡死: 最近 N 步全 forward 且总位移 < D
        lastN = self.hist[-self.STUCK_N:]
        if len(lastN) == self.STUCK_N and all(a == 'move_forward' for a, _ in lastN) \
                and sum(m for _, m in lastN) < self.STUCK_D and step - self.last_escape_step > 8:
            side = 'turn_left' if goal_ego[1] > 0 else 'turn_right'
            n90 = int(round(90.0 / self.TURN_DEG))     # 0913: 原硬编码 3
            n180 = int(round(180.0 / self.TURN_DEG))   # 0913: 原硬编码 6
            plans = [
                [side] * n90 + ['move_forward'] * 2,
                [('turn_right' if side == 'turn_left' else 'turn_left')] * n90 + ['move_forward'] * 2,
                ['turn_left'] * n180 + ['move_forward'] * 3,
            ]
            self.plan = plans[min(self.level, 2)][:]
            self.level += 1
            self.last_escape_step = step
            self.hist = []
            self.n_steps += 1
            return self.plan.pop(0)
        # 转圈: 最近20步转向净累积>300°且位移<0.5
        last20 = self.hist[-20:]
        if len(last20) == 20:
            net = sum(self.TURN_DEG if a == 'turn_left' else (-self.TURN_DEG if a == 'turn_right' else 0) for a, _ in last20)
            moved20 = sum(m for _, m in last20)
            if abs(net) > self.SPIN_N_TURNS * self.TURN_DEG and moved20 < 0.5 and step - self.last_escape_step > 8:
                self.plan = ['move_forward'] * 3
                self.last_escape_step = step
                self.hist = []
                self.n_steps += 1
                return self.plan.pop(0)
        return None


def free_dir_ego(exyz, max_r=8.0, half_win=20.0):
    """ego系局部点云(N,2或3) → (最空方向的角度, 该方向净空)。0=正前, 逆时针为正(=turn_left)。"""
    if exyz is None or len(exyz) == 0:
        return 0.0, max_r
    r = np.hypot(exyz[:, 0], exyz[:, 1])
    ang = np.degrees(np.arctan2(exyz[:, 1], exyz[:, 0]))
    m0 = (r > 0.35) & (r < max_r)
    if not m0.any():
        return 0.0, max_r
    r, ang = r[m0], ang[m0]
    best_a, best_d = 0.0, -1.0
    for a in range(-180, 180, 20):
        dd = np.abs(((ang - a + 180.0) % 360.0) - 180.0)
        sel = dd < half_win
        nd = float(r[sel].min()) if sel.any() else max_r
        if nd > best_d:
            best_d, best_a = nd, float(a)
    return best_a, best_d


class EscapeV2(Escape):
    """0911 v2: 卡死判据与 v1 相同, 但逃脱方向由【局部占用选最空方向】决定, 不再固定朝 goal 侧。
    动机: 87% 的失败是"走不到"而非"停不准", 按 goal 侧硬转可能把人往墙上带 —— 应该往有空的地方走。
    级别递增: L0=最空方向; L1=最空±60°(换一边); L2=最空±120°(大范围重试)。"""

    def update(self, act, moved, goal_ego, step, ego_pts=None, dtg=None):
        if self.plan:
            self.n_steps += 1
            return self.plan.pop(0)
        if self._blocked(dtg):
            return None
        self.hist.append((act, moved))
        self.hist = self.hist[-24:]
        lastN = self.hist[-self.STUCK_N:]
        stuck = (len(lastN) == self.STUCK_N and all(a == 'move_forward' for a, _ in lastN)
                 and sum(m for _, m in lastN) < self.STUCK_D and step - self.last_escape_step > 8)
        spin = False
        last20 = self.hist[-20:]
        if not stuck and len(last20) == 20:
            net = sum(self.TURN_DEG if a == 'turn_left' else (-self.TURN_DEG if a == 'turn_right' else 0) for a, _ in last20)
            if abs(net) > self.SPIN_N_TURNS * self.TURN_DEG and sum(m for _, m in last20) < 0.5 and step - self.last_escape_step > 8:
                spin = True
        if not (stuck or spin):
            return None
        if spin:
            # 打转: v1 策略(直接前进)实测有效, 保留
            self.plan = ['move_forward'] * 4
            self.last_escape_step = step
            self.hist = []
            self.n_steps += 1
            return self.plan.pop(0)

        tgt, clear = free_dir_ego(ego_pts)
        side = 1.0 if goal_ego[1] >= 0 else -1.0
        if self.level == 1:
            tgt += 60.0 * side
        elif self.level >= 2:
            tgt += 120.0 * (-side)          # 反向大范围重试
        tgt = ((tgt + 180.0) % 360.0) - 180.0
        nt = int(round(abs(tgt) / self.TURN_DEG))   # 0913: 原 /30
        turn_act = 'turn_left' if tgt > 0 else 'turn_right'
        fwd_n = 3 + min(self.level, 2)       # 3,4,5 步
        self.plan = ([turn_act] * nt if nt > 0 else []) + ['move_forward'] * fwd_n
        self.level += 1
        self.last_escape_step = step
        self.hist = []
        self.n_steps += 1
        return self.plan.pop(0)


class EscapeV3(EscapeV2):
    """0912 v3: 规划式脱困 —— 针对 v2 的四处缺陷
    ① v2 只选"最空方向"、不看向: world_pts 是累积点云(含历史轨迹), 身后永远最空
       → v2 会系统性把人往背离目标的方向带。
       v3 打分 = min(净空, CAP) + W_ALIGN*cos(候选方向 - goal方向), 兼顾"能走"与"朝目标"。
    ② v2 的 level 单调递增且 min(level,2) 饱和 → 同一点第 3 次起永远同一套动作。
       v3 level 取模循环, 档位更细 (0/±60/±120/180)。
    ③ v2 无"脱困无效"检测 → 500 步内可触发上百次同一动作而不自知。
       v3 跟踪 best_dtg, 连续 NO_PROG_N 次无改善 → 大范围重定位(掉头走远)。
    ④ v2 fwd_n 只有 3~5 步(0.75~1.25m), 走一点又被判卡死。
       v3 按该方向净空决定走多远(最多 8 步 = 2m)。
    """
    UNKNOWN_CLEAR = 2.5  # ⭐ "无观测"方向的等效净空 —— 不能当最空(0912 诊断)
    MIN_CLEAR = 1.5      # 判定"这个方向走得通"的最小净空(6 步 = 1.5m)
    NO_PROG_N = 4        # 连续 N 次脱困无改善 → 大范围重定位
    LEVELS = (0.0, 60.0, -60.0, 120.0, -120.0, 180.0)
    LEVEL_SAT = False    # True = 消融②: level 退回 v2 的 min(level,2) 饱和
    FWD_V2 = False       # True = 消融④: fwd_n 退回 v2 的 3+min(level,2) 步

    def reset(self):
        super().reset()
        self.best_dtg = float('inf')
        self.no_prog = 0

    def _clear_at(self, pts, a, half_win=25.0, max_r=8.0):
        """ego 系点云中方向 a(度, 0=正前, 逆时针正) ±half_win 扇区内最近点距离。

        ⚠️ 关键(0912 诊断): 累积点云只覆盖"走过"的区域。没点的地方是【未知】,
        不是【空旷】。v2 把两者都当 max_r → 身后(走过、空)与侧前方(没走过、无点)
        并列最空 → 实测失败 case 有 94~95% 的方向倒向正后方, 越救越远。
        这里对"无观测"返回 UNKNOWN_CLEAR(中等值), 不再伪装成最空。
        """
        if pts is None or len(pts) == 0:
            return self.UNKNOWN_CLEAR
        r = np.hypot(pts[:, 0], pts[:, 1])
        ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        m = (r > 0.35) & (r < max_r)
        if not m.any():
            return self.UNKNOWN_CLEAR
        r, ang = r[m], ang[m]
        dd = np.abs(((ang - a + 180.0) % 360.0) - 180.0)
        sel = dd < half_win
        if not sel.any():
            return self.UNKNOWN_CLEAR
        return float(r[sel].min())

    def update(self, act, moved, goal_ego, step, ego_pts=None, dtg=None):
        if self.plan:
            self.n_steps += 1
            return self.plan.pop(0)
        if self._blocked(dtg):
            return None
        self.hist.append((act, moved))
        self.hist = self.hist[-24:]
        lastN = self.hist[-self.STUCK_N:]
        stuck = (len(lastN) == self.STUCK_N and all(a == 'move_forward' for a, _ in lastN)
                 and sum(m for _, m in lastN) < self.STUCK_D
                 and step - self.last_escape_step > 8)
        spin = False
        last20 = self.hist[-20:]
        if not stuck and len(last20) == 20:
            net = sum(self.TURN_DEG if a == 'turn_left' else (-self.TURN_DEG if a == 'turn_right' else 0)
                      for a, _ in last20)
            if abs(net) > self.SPIN_N_TURNS * self.TURN_DEG and sum(m for _, m in last20) < 0.5 \
                    and step - self.last_escape_step > 8:
                spin = True
        if not (stuck or spin):
            return None

        # 无改善计数(v3 新增): 只有 dtg 真的推进了才清零
        if dtg is not None:
            if dtg < self.best_dtg - 0.3:
                self.best_dtg, self.no_prog = dtg, 0
            else:
                self.no_prog += 1

        if spin:
            # 打转: v1/v2 策略(直接前进)实测有效, 保留
            self.plan = ['move_forward'] * 4
            self.last_escape_step = step
            self.hist = []
            self.n_steps += 1
            return self.plan.pop(0)

        # ③ 大范围重定位: 反复救不出来 → 掉头走远, 换一片区域
        if self.no_prog >= self.NO_PROG_N:
            self.no_prog = 0
            _n180 = int(round(180.0 / self.TURN_DEG))   # 0913: 原硬编码 6(=意图180°)
            turn = (['turn_left'] * _n180 if goal_ego[1] >= 0
                    else ['turn_right'] * _n180)
            self.plan = turn + ['move_forward'] * 8
            self.last_escape_step = step
            self.hist = []
            self.n_steps += 1
            return self.plan.pop(0)

        # ① 方向选择: 先在"走得通"(净空≥MIN_CLEAR)的候选里挑最朝目标的;
        #    若全都不通, 退化为"挑最空" —— 此时朝目标已无意义, 先挪出死角再说。
        a_goal = float(np.degrees(np.arctan2(goal_ego[1], goal_ego[0])))
        # ② 档位循环而非饱和 (--v3_levelsat 时退回 v2 饱和, 仅供消融)
        base = (self.LEVELS[min(self.level, len(self.LEVELS) - 1)] if self.LEVEL_SAT
                else self.LEVELS[self.level % len(self.LEVELS)])
        cands = []
        for off in (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0):
            a = ((base + off + 180.0) % 360.0) - 180.0
            cands.append((a, self._clear_at(ego_pts, a)))
        ok = [(a, c) for a, c in cands if c >= self.MIN_CLEAR]
        if ok:
            tgt, clear_sel = max(ok, key=lambda ac: float(np.cos(np.radians(ac[0] - a_goal))))
        else:
            tgt, clear_sel = max(cands, key=lambda ac: ac[1])
        nt = int(round(abs(tgt) / self.TURN_DEG))   # 0913: 原 /30
        turn_act = 'turn_left' if tgt > 0 else 'turn_right'
        # ④ 走多远按该方向净空定(保守 60%), 3~8 步
        # ④ 走多远按该方向净空定(保守 60%), 3~8 步 (--v3_fwd_v2 时退回 v2 口径, 仅供消融)
        fwd_n = (3 + min(self.level, 2) if self.FWD_V2
                 else int(max(3, min(8, clear_sel / 0.25 * 0.6))))
        self.plan = ([turn_act] * nt if nt > 0 else []) + ['move_forward'] * fwd_n
        self.level += 1
        self.last_escape_step = step
        self.hist = []
        self.n_steps += 1
        return self.plan.pop(0)


# 注: 0911 曾写过 EscapeV3(dtg<3 保护 + 次数上限), 实测只救回 3/15 —— 诊断证明
# 误触发发生在 dtg 4~5m 处而非目标附近, 该保护拦不到。已删除, 其能力改由
# Escape.GUARD_DTG / Escape.MAX_ESCAPE 参数表达(默认关, 与 v1 行为一致)。
# (0912 的 EscapeV3 是另一套思路: 方向打分 + 无进展升级, 见上)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='goalnav_r2rft.pt')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--sample', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ep_ids', nargs='+', default=None,
                    help='显式指定episode id列表(跑失败子集用)')
    ap.add_argument('--shard_idx', type=int, default=-1, help='全量分片:第idx片(0基)')
    ap.add_argument('--shard_num', type=int, default=1, help='全量分片:总片数')
    ap.add_argument('--max_steps', type=int, default=500)
    ap.add_argument('--stop_dist', type=float, default=1.5)
    ap.add_argument('--hfov', type=float, default=90.0,
                    help='dense ckpt=79(与训练一致), 旧ckpt=90')
    ap.add_argument('--anchor', default=None, help='自定义anchor npy(测新anchor先验即时贡献)')
    ap.add_argument('--step_num', type=int, default=0,
                    help='推理去噪步数(0=默认2; 5~10 测大转弯去噪是否更稳)')
    ap.add_argument('--dense_goal', action='store_true',
                    help='goal用0.25m稠密插值路径(与训练resample_ego口径一致)')
    ap.add_argument('--escape', type=int, default=0,
                    help='0=裸; 1=0903原版朝goal侧转(全量+15.1pt); 2=局部占用选最空方向(v2); '
                         '3=规划式(v3: 净空+朝目标打分 / 无进展升级 / 走更远)')
    ap.add_argument('--esc_n', type=int, default=4, help='卡死判据: 连续N步(默认4=v1原版)')
    ap.add_argument('--esc_d', type=float, default=0.15, help='卡死判据: N步总位移阈值(默认0.15)')
    ap.add_argument('--esc_guard', type=float, default=0.0, help='接近目标保护: dtg<X不触发(0=关)')
    ap.add_argument('--esc_cap', type=int, default=0, help='逃脱接管步数上限(0=无限)')
    ap.add_argument('--v3_minclear', type=float, default=1.5,
                    help='v3: 判定方向"走得通"的最小净空(米)')
    ap.add_argument('--v3_noprog', type=int, default=4,
                    help='v3: 连续N次脱困无改善→大范围重定位')
    ap.add_argument('--v3_unknown', type=float, default=2.5,
                    help='v3: "无观测"方向的等效净空(不能当最空)')
    ap.add_argument('--split', default='val_unseen',
                    help='habitat split: val_unseen / val_seen (held-out 验证用)')
    ap.add_argument('--v3_levelsat', action='store_true',
                    help='消融②: 关闭 level 取模循环, 退回 v2 的 min(level,2) 饱和')
    ap.add_argument('--v3_fwd_v2', action='store_true',
                    help='消融④: fwd_n 退回 v2 的 3+min(level,2) 步')
    ap.add_argument('--turn_deg', type=float, default=30.0,
                    help='每步转向角。30.0=历史行为(既有数字可比); 15.0=habitat 真实值(0913 修正)')
    ap.add_argument('--out', default='probe_oracle')
    cli = ap.parse_args()

    # 0911 参数注入(默认值 = v1 原版行为, 既有数字全部可比)
    Escape.STUCK_N = cli.esc_n
    Escape.STUCK_D = cli.esc_d
    Escape.GUARD_DTG = cli.esc_guard
    Escape.MAX_ESCAPE = cli.esc_cap
    # v3 参数注入(默认值 = 0912 定版行为)
    EscapeV3.MIN_CLEAR = cli.v3_minclear
    EscapeV3.NO_PROG_N = cli.v3_noprog
    EscapeV3.UNKNOWN_CLEAR = cli.v3_unknown
    EscapeV3.LEVEL_SAT = bool(cli.v3_levelsat)
    EscapeV3.FWD_V2 = bool(cli.v3_fwd_v2)
    Escape.TURN_DEG = float(cli.turn_deg)
    if cli.escape:
        print(f'[escape] mode={cli.escape} N={cli.esc_n} D={cli.esc_d} '
              f'guard={cli.esc_guard} cap={cli.esc_cap} '
              f'v3(minclear={cli.v3_minclear},noprog={cli.v3_noprog},'
              f'unknown={cli.v3_unknown},levelsat={cli.v3_levelsat},'
              f'fwd_v2={cli.v3_fwd_v2},turn_deg={cli.turn_deg})', flush=True)

    env = make_env(cli.split, 640, 480, 79)
    print(f'[split] {cli.split}: {len(list(env.episodes))} episodes', flush=True)
    eps = list(env.episodes)
    if cli.ep_ids:
        want = [str(i) for i in cli.ep_ids]
        by_id = {str(e.episode_id): e for e in eps}
        eps = [by_id[i] for i in want if i in by_id]
        missing = [i for i in want if i not in by_id]
        if missing:
            print(f'[warn] ep_ids not found: {missing}', flush=True)
    elif cli.shard_idx >= 0:
        eps = eps[cli.shard_idx::cli.shard_num]
    else:
        random.Random(cli.seed).shuffle(eps)
        eps = eps[:cli.sample][:cli.n]      # 同300子集的前N条
    env.episodes = eps
    print(f'==== oracle-goal probe: {len(eps)} eps, ckpt={cli.ckpt} ====', flush=True)

    agent = OurAgent(ckpt=cli.ckpt, hfov=cli.hfov)
    if cli.step_num > 0:
        agent.model._trajectory_head._step_num = cli.step_num
        print(f'[step_num] 推理去噪步数 -> {cli.step_num}', flush=True)
    if cli.anchor:
        import torch
        agent.model._trajectory_head.plan_anchor.data = torch.tensor(
            np.load(cli.anchor).astype(np.float32), device=agent.device)
        print(f'[anchor] 覆盖 plan_anchor -> {cli.anchor}', flush=True)
    results = open(cli.out + '.jsonl', 'w', encoding='utf-8')

    for ei in range(len(eps)):
        env.episodes = [eps[ei]]
        obs = env.reset()
        ep = eps[ei]
        ref = np.array(ep.reference_path, np.float64)
        ref_ours = np.stack([ref[:, 0], -ref[:, 2]], 1)   # hab→我们系
        if cli.dense_goal:
            ref_ours = densify_ours(ref_ours)
        agent.reset_episode()

        steps = 0
        min_dtg = float('inf')
        stopped = False
        prev_xy = None
        pending_act = None
        n_escape = 0
        # 0913修 SPL: GT 路径长(逐点欧氏, 精确)
        gt_len = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
        path_len = 0.0
        prev_pos3 = None
        if cli.escape == 3:
            esc = EscapeV3()
        elif cli.escape == 2:
            esc = EscapeV2()
        else:
            esc = Escape()
        while not env.episode_over and steps < cli.max_steps:
            state = env.sim.get_agent_state()
            rgb = np.ascontiguousarray(obs['rgb'][:, :, :3])
            depth = np.squeeze(np.ascontiguousarray(obs['depth'])).astype(np.float32)
            position = np.asarray(state.position, np.float64)
            if prev_pos3 is not None:
                path_len += float(np.linalg.norm(position - prev_pos3))
            prev_pos3 = position
            x, y, th, floor = agent.hab_pose_to_ours(position, state.rotation)
            pts = agent.depth_to_world(depth, position, state.rotation)
            agent.world_pts = np.concatenate([agent.world_pts, pts])[-600000:]

            m = env.get_metrics()
            dtg = float(m.get('distance_to_goal', -1))
            min_dtg = min(min_dtg, dtg)
            if dtg < cli.stop_dist:
                stopped = True
                break

            goal_ego, d_end = oracle_goal_ego(ref_ours, x, y, th)

            # 脱困优先(0903实锤+8pt): 吃上一个动作的实际位移, 命中则跳过模型直接执行逃脱序列
            moved = 0.0 if prev_xy is None else float(np.hypot(x - prev_xy[0], y - prev_xy[1]))
            prev_xy = (x, y)
            if cli.escape and pending_act is not None:
                if cli.escape in (2, 3):
                    # v2/v3: 取 ego 系局部点云(下采样)给逃脱序列选方向
                    wp = agent.world_pts
                    if len(wp) > 0:
                        ws = wp[::6]
                        dd = ws[:, :2] - np.array([x, y], np.float64)
                        cc, ss = np.cos(th), np.sin(th)
                        ego_pts = np.stack([cc * dd[:, 0] + ss * dd[:, 1],
                                            -ss * dd[:, 0] + cc * dd[:, 1]], 1).astype(np.float32)
                    else:
                        ego_pts = None
                    e = esc.update(pending_act, moved, goal_ego, steps, ego_pts=ego_pts, dtg=dtg)
                else:
                    e = esc.update(pending_act, moved, goal_ego, steps, dtg=dtg)
                if e is not None:
                    n_escape += 1
                    obs = env.step(ACT2ID[e])
                    pending_act = e
                    steps += 1
                    continue

            cam = sb.camera_feature(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), agent.device)
            pose = np.array([x, y, th], np.float32)
            bev = sb.lidar_feature_fov(agent.world_pts, pose, floor, agent.device)
            k = max(abs(goal_ego[0]) / 5.0, abs(goal_ego[1]) / 2.0, 1.0)
            st = status_gh(goal_ego / k, np.array([1.0, 0.0], np.float32), bs_vx=agent.vx, device=agent.device)
            st[0, 5] = agent.hist_w / 8.0
            st[0, 6] = agent.hist_gamma
            st[0, 7] = 0.0
            agent.torch.manual_seed(0)
            # ⭐goal_mlp: goal_dir 与训练同口径 (goal_ego / [5,2])
            goal_dir = agent.torch.tensor(
                goal_ego / np.array([5.0, 2.0], np.float32), device=agent.device).unsqueeze(0)
            with agent.torch.no_grad():
                traj = agent.model({'camera_feature': cam, 'lidar_feature': bev,
                                    'status_feature': st, 'goal_dir': goal_dir})['trajectory'][0].cpu().numpy()
            # 0902修: 弃用traj_to_control的yaw通道阈值(实测模型yaw输出≈0,
            # 转向全表达在横向位移, 被R2R无strafe动作空间丢弃 → 0/230假阴性)。
            # 改看轨迹末端弯曲角(与offline_turn_probe同口径), |bend|>15°转(habitat转向=30°)
            bend = float(np.degrees(np.arctan2(traj[-1, 1], max(traj[-1, 0], 1e-6))))
            if bend > 15.0:
                act = 'turn_left'
            elif bend < -15.0:
                act = 'turn_right'
            else:
                act = 'move_forward'
            obs = env.step(ACT2ID[act])
            pending_act = act
            steps += 1

        m = env.get_metrics()
        # habitat success需要stop动作; 这里用oracle停→按dtg<3判定SR口径等价
        ne = float(m.get('distance_to_goal', float('nan')))
        rec = {'episode_id': str(ep.episode_id), 'steps': steps, 'n_escape': n_escape,
               'success': float(stopped and ne < 3.0),
               'os': float(min_dtg < 3.0), 'ne': ne, 'min_dtg': min_dtg,
               'spl': 0.0, 'path_len': path_len, 'gt_len': gt_len}
        # 0913修: 原 gt_len 取自 m.get('gt_path_length') —— habitat 无此 key → 恒 0 → SPL 全 0;
        # 且 path_len=steps*0.25 把转向步也算成位移 → 偏大。现两者都取自真实量, 只影响本字段。
        if rec['success'] > 0 and gt_len > 0:
            rec['spl'] = min(1.0, gt_len / max(path_len, gt_len))
        results.write(json.dumps(rec) + '\n')
        results.flush()
        print(f'[{ei+1}/{len(eps)}] ep{ep.episode_id} steps={steps} esc={n_escape} '
              f'min_dtg={min_dtg:.2f} ne={ne:.2f} succ={rec["success"]:.0f}', flush=True)

    results.close()
    rows = [json.loads(l) for l in open(cli.out + '.jsonl')]
    n = len(rows)
    print(f'==== ORACLE-PROBE n={n} SR={sum(r["success"] for r in rows)/n*100:.1f} '
          f'OSR={sum(r["os"] for r in rows)/n*100:.1f} '
          f'NE={np.mean([r["ne"] for r in rows]):.2f} ====', flush=True)


if __name__ == '__main__':
    main()
