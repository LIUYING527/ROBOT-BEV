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
from slowbrain_proj import project_goal_to_pixel   # noqa: E402  0915 dual_locate
from our_agent import OurAgent        # noqa: E402
import stage_b_eval as sb             # noqa: E402
from train_goalnav_turn import status_gh  # noqa: E402

HFOV = 79.0
ACT2ID = {'stop': 0, 'move_forward': 1, 'turn_left': 2, 'turn_right': 3}


# ── 0925 跨帧历史采样（推理端必须与训练端同口径）──────────────────────────────
# 训练数据由 tools/gen_ego_train_data_span.py 生成：K 帧在 [i-span, i] 上**等距取**，
# span=40 帧 ≈ 10m 位移 ⇒ 8 张图跨越 10 米。此前推理端只有连续帧(deque(nframes))，
# 8 帧只覆盖 ~2m ⇒ 训练/推理**帧采样口径不一致**（[[train-infer-prompt-mismatch]] 同类静默脏化）。
def _span_idx(i, K, span):
    """与 tools/gen_ego_train_data_span.py::window_idx **逐字同口径**。"""
    if K <= 1 or span <= 0:
        return list(range(max(0, i - K + 1), i + 1))
    lo = max(0, i - span)
    if i - lo < K - 1:                      # 历史不足（episode 早期）⇒ 允许重复
        return [lo] * (K - 1) + [i]
    return [int(round(lo + j * (i - lo) / (K - 1))) for j in range(K)]


def _pick_frames(buf, K, span):
    """从帧缓冲里挑 K 帧。span<=0（默认）= 旧的连续帧行为，逐位不变。"""
    buf = list(buf)
    if not buf or K <= 0:
        return buf
    if span and span > 0 and K > 1:
        idx = _span_idx(len(buf) - 1, K, span)
        return [buf[min(max(j, 0), len(buf) - 1)] for j in idx]
    return buf[-K:]
# 0915 诊断: 记录每步最终动作 —— 用来区分「交替转向(左右摆)」vs「一直 forward 但被几何卡住」。
# 环境变量 TRACE_ACT=<路径> 启用; 默认关, 不影响任何既有行为与历史可比性。
_ACT_TRACE_FH = None

# ⭐ 0918 depth 单位运行时护栏 —— 归一化 bug 已发生两次(0912 / 0918), 每次静默烧掉一整周实验。
# 判据: 归一化时 depth = min(真距/10, 1.0) ⇒ **任何一步都不可能 >1.0**;
#       米制下室内场景几十步内必然出现 >1.0。故「累积 50 步无 >1.0 且 dtg>3m」= 归一化。
# ⚠️ 不要用单帧 max 判(密闭小房间单帧 max 合法地 <1m —— 0918 用它误报过一次, 白杀一路)。
_DEPTH_GUARD = {'n': 0, 'max_seen': 0.0, 'done': False}


class Escape:
    """卡死/转圈检测 + 固定逃脱序列。

    0913 从 probe_oracle_goal.py 原样移植(不 import 是为了避开其模块级副作用)。
    纯推理侧, 不动权重 → 零 co-adaptation 风险。

    为什么要移植: 闭环实测 **40% 的 ep 会彻底卡死** —— 位置完全冻结 445/461 步,
    白耗 90% 的步数预算(见 diag_far2a.jsonl 的 ep667/ep888)。而快脑 oracle 探针
    已证明脱困值 +14pt(裸 51.6% → escape2 65.6%), 但闭环里**从未接入**。
    """

    STUCK_N = 4          # 卡死判据: 连续 N 步
    STUCK_D = 0.15       # 卡死判据: 这 N 步总位移 < D
    GUARD_DTG = 0.0      # dtg < 此值时不触发(0=关)
    MAX_ESCAPE = 0       # 逃脱接管步数上限(0=无限)
    # 0914 新增: 滞留(交替振荡)判据 —— stuck 和 spin 同时漏掉的那一类死锁。
    # 闭环 diag 实证: ep294 打满 1000 步、ep493 打满 990 步, 目标全程 100% 可见,
    # dtg 恒定(7.94/7.69), nx 只在两个固定值间周期跳 ⇒ 左右交替转向。
    # 净转角≈0 让 spin 漏, 动作非连续 forward 让 stuck 漏, 于是零位移空烧整轮预算。
    STALL_N = 30         # 滞留判据: 回看最近 N 步
    STALL_D = 0.6        # 滞留判据: N 步净位移 < D(米)
    STALL_MIN_FWD = 5    # 滞留判据: N 步里至少这么多次前进(排除"纯粹原地转向")

    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = []          # (act, moved_dist)
        self.pos_hist = []      # 最近 STALL_N+1 个位置(仅 xy), 供滞留判据算净位移
        self.plan = []
        self.level = 0          # 同一点第几次卡死(0=goal侧90, 1=反侧90, 2=180)
        self.last_escape_step = -100
        self.n_steps = 0        # 被逃脱序列接管的累计步数

    def _blocked(self, dtg):
        """接近目标保护 / 逃脱次数上限 → 本次不触发。"""
        if self.MAX_ESCAPE and self.n_steps >= self.MAX_ESCAPE:
            return True
        if self.GUARD_DTG and dtg is not None and 0.0 <= dtg < self.GUARD_DTG:
            return True
        return False

    def update(self, act, moved, goal_ego, step, dtg=None, pos=None):
        if self.plan:
            self.n_steps += 1
            return self.plan.pop(0)
        if self._blocked(dtg):
            return None
        self.hist.append((act, moved))
        # 保留长度要够滞留判据回看(STALL_N 默认 30 > 原来的 24); spin 仍只取末尾 20 条, 不受影响
        self.hist = self.hist[-max(24, self.STALL_N):]
        if pos is not None:
            self.pos_hist.append(np.asarray(pos, np.float64)[:2])
            self.pos_hist = self.pos_hist[-(self.STALL_N + 1):]
        # 卡死: 最近 N 步全 forward 且总位移 < D
        lastN = self.hist[-self.STUCK_N:]
        if len(lastN) == self.STUCK_N and all(a == 'move_forward' for a, _ in lastN) \
                and sum(m for _, m in lastN) < self.STUCK_D and step - self.last_escape_step > 8:
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
            self.n_steps += 1
            return self.plan.pop(0)
        # 转圈: 最近 20 步转向净累积 > 300° 且位移 < 0.5
        last20 = self.hist[-20:]
        if len(last20) == 20:
            net = sum(30 if a == 'turn_left' else (-30 if a == 'turn_right' else 0)
                      for a, _ in last20)
            moved20 = sum(m for _, m in last20)
            if abs(net) > 300 and moved20 < 0.5 and step - self.last_escape_step > 8:
                self.plan = ['move_forward'] * 3
                self.last_escape_step = step
                self.hist = []
                self.n_steps += 1
                return self.plan.pop(0)
        # 滞留(交替振荡): 最近 N 步里有前进, 但净位移≈0 —— 在同一个点上左右摇摆。
        if self.STALL_N > 0 and len(self.pos_hist) == self.STALL_N + 1:
            disp = float(np.linalg.norm(self.pos_hist[-1] - self.pos_hist[0]))
            n_fwd = sum(1 for a, _ in self.hist if a == 'move_forward')
            if disp < self.STALL_D and n_fwd >= self.STALL_MIN_FWD \
                    and step - self.last_escape_step > 8:
                # 必须用**连续同向转**打破对称。stuck 那套"朝 goal 侧转 90°"不行:
                # 振荡的两个端点通常一个偏左一个偏右, 朝其中一侧转正好落回环里。
                self.plan = ['turn_left'] * 6 + ['move_forward'] * 4
                self.last_escape_step = step
                self.hist = []
                self.pos_hist = []
                self.n_steps += 1
                return self.plan.pop(0)
        return None
SCAN_MAX = 8  # 连续不可见 8 次（≈120°）后前进换视角

GOAL_AHEAD = 4.0   # 与 probe_oracle_goal 同口径: 沿参考路径前瞻 4m 的点
_EGO_SMOOTH = 1.0  # ⭐0923 ego 抖动抑制阈值(米)：位置几乎没动而 ego 点跳变超过它 ⇒ 沿用上一步。
                   #   实测死锁时抖幅 ~7.4m，而正常输出变化多在 1m 内 ⇒ 1.0 能只抓真抖动。


def densify_ours(ref_ours, ds=0.25):
    """稀疏路点 → 0.25m 稠密插值(与 gen_dense_data.densify / probe_oracle_goal 同口径)。"""
    pts = [np.array(ref_ours[0], np.float64)]
    acc = 0.0
    for i in range(1, len(ref_ours)):
        a = np.array(ref_ours[i - 1], np.float64)
        b = np.array(ref_ours[i], np.float64)
        seg = float(np.linalg.norm(b - a))
        if seg < 1e-6:
            continue
        d = ds - acc
        while d < seg:
            pts.append(a + (b - a) * (d / seg))
            d += ds
        acc = seg - (d - ds)
    pts.append(np.array(ref_ours[-1], np.float64))
    return np.stack(pts)


def oracle_goal_ego(ref_ours, x, y, th):
    """⭐0918 天花板实验用: ref_path(我们系)上找最近点, 沿弧长前瞻 GOAL_AHEAD 米 → ego 系。
    与 probe_oracle_goal.oracle_goal_ego 逐字同口径(探针能跑到 65.6%→77.7% 就靠它)。
    目的: 把「慢脑」这一环换成完美目标, 其余控制器一字不改, 量出**控制器的天花板**,
    从而判断该继续修慢脑还是修控制器。"""
    d = ref_ours - np.array([x, y])
    i0 = int(np.argmin(np.linalg.norm(d, axis=1)))
    acc, gp = 0.0, ref_ours[-1]
    for i in range(i0, len(ref_ours) - 1):
        seg = float(np.linalg.norm(ref_ours[i + 1] - ref_ours[i]))
        if acc + seg >= GOAL_AHEAD:
            r = (GOAL_AHEAD - acc) / max(seg, 1e-9)
            gp = ref_ours[i] * (1 - r) + ref_ours[i + 1] * r
            break
        acc += seg
    c, s = np.cos(th), np.sin(th)
    dd = gp - np.array([x, y])
    return np.array([c * dd[0] + s * dd[1], -s * dd[0] + c * dd[1]], np.float32)


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
    ap.add_argument('--span', type=int, default=0,
                    help='多帧的**跨帧跨度**(0=旧的连续帧行为)。训练端 gen_ego_train_data_span.py '
                         '用 --span 40（8 帧跨越 40 步≈10m）；⭐0925 实测推理端一直没传 ⇒ 训练用 10m '
                         '跨度的帧、推理只给 2m 连续帧 = 口径不一致。配 --nframes 8 --span 40 才对齐训练。')
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
    ap.add_argument('--stuck_gate_steps', type=int, default=0,
                    help='⭐0924「位移门控」：若最近 N 步净位移 < --stuck_gate_d 米（= agent 真的卡住），'
                         '则**禁止判停**。0=关闭。'
                         '动机：`--min_steps` 那一刀切帮了"卡死"的 ep 却损害了"能走"的 ep'
                         '（同 55ep 配对：SR +12.7pt 但 OSR −16.4pt）；位移门控只在真卡住时生效，'
                         '期望两者兼顾。判据来自实测：失败 ep 的 dtg 从头到尾不变（agent 原地转圈），'
                         '而它们的判停全部是假到达（‖ego‖<2.5）。')
    ap.add_argument('--stuck_gate_d', type=float, default=0.5,
                    help='位移门控的净位移阈值（米）。配合 --stuck_gate_steps 使用。')
    ap.add_argument('--min_steps', type=int, default=0,
                    help='⭐0924「最小步数门控」：前 N 步**不允许判停**，0=原行为。 '
                         '依据：ego 版闭环 40/40 个失败 ep **全部**是「模型在远处报 ‖ego‖<stop_dist ⇒ 假到达」，'
                         '其中 9 个在**第 1 步**就停（真实 dtg 7.4~13.4m）。起点报 0 时没有任何历史，'
                         '任何时序类判据都挡不住 ⇒ 只能用最小步数挡住。')
    ap.add_argument('--min_stop_depth', type=float, default=0.0,
                    help='stop 判决的最小深度门槛(米)。0912 诊断: 慢脑报可见时坐标处深度中位仅 0.57m'
                         '(指向近处物体), 而真实距离中位 7.49m ⇒ 100%% 的停止判决都是误判。'
                         '几何上地面目标在画面内的最小距离 = 相机高/tan(半垂直FOV) ≈ 1.96m, '
                         '故设 1.9 可滤掉「指向近处障碍物」这类假到达。0=不启用(原行为)')
    ap.add_argument('--goal_len4', type=int, default=0,
                    help='0915 A/B: 把喂给快脑的目标点**一律**归一化到 4m(快脑训练口径)。'
                         '原写法 `if dist > 4.0` 只处理了「太远」, 漏了「太近」—— 而慢脑所指像素深度 d '
                         '中位仅 1.0m(它偏向画面中下部), 90.2%% 的步 dist<4 ⇒ 快脑收到 ~1m 的目标点 '
                         '⇒ 分布外 ⇒ 走不动。1=一律归一化(方向不变, 只补长度下限)。0=原行为')
    ap.add_argument('--ego_lat_clamp', type=float, default=0.0,
                    help='0927 侧方自锁 A/B: 把**喂给快脑**的 ego 横向分量夹到 ±该值(米)。'
                         '依据 docs/0927-OSR崩塌根因-侧方自锁.md —— K=8 闭环里模型输出的原始方向角'
                         '中位 73.9°(横向 -3.3m / 前向 1.0m)，而训练集标签 |角| 中位仅 23.5%%'
                         '⇒ 快脑收到侧方目标只会原地转 ⇒ 自锁走不动(净损的 58%% 是这个)。'
                         '0=原行为(不夹)。')
    ap.add_argument('--ego_norm4', type=int, default=0,
                    help='0927 配合 --ego_lat_clamp: 夹完横向后把模长归一化回 4m(快脑训练口径)。'
                         '单独用=goal_len4 的等价物(方向不变)，故只在配合 clamp 时才有意义。')
    ap.add_argument('--perturb_lat', type=float, default=0.0,
                    help='0928 敏感度扫描: 给喂给快脑的目标**横向平移**这么多米（正=左）。'
                         '用来量「慢脑目标误差 → 闭环 SR」的衰减曲线。')
    ap.add_argument('--perturb_scale', type=float, default=1.0,
                    help='0928 敏感度扫描: 把喂给快脑的目标**距离缩放**这个倍数（1.0=不缩放）。')
    ap.add_argument('--frozen_stop', type=int, default=0,
                    help='0915 stopfix 「停滞豁免」: 位置连续 N 步几乎不动(|Δpos|<2cm)时, '
                         '允许跳过 min_stop_depth 闸门判停。0=关(原行为)。'
                         '依据: ep1429 停在 dtg=0.90m 却没停(被 d>=min_stop_depth 挡住), 白烧 370 步。')
    ap.add_argument('--ep_ids', default=None,
                    help='逗号分隔的 episode_id 列表: 固定样本集, 用于跨配置严格配对。'
                         '⚠️0913 实测: habitat 的 env.episodes 顺序每次加载都不同, '
                         'random.Random(seed) 无法复现同一批 episode, 跨次对比只能靠统计')
    ap.add_argument('--arrive_a', type=float, default=0.0,
                    help='到达判据: 慢脑历史报过的 dist < A 且随后连续 K 步不可见 ⇒ 判到达。'
                         '⚠️0913 根因: 慢脑报的 dist 有 ≈2.1m 硬下界(相机高1.30/tan(半FOV31.7)), '
                         '目标 <2.1m 时不在画面内 ⇒ stop_dist 永远等不到; 而「走过终点」也会判不可见。'
                         '实测 far2a 15eps 重放: A=4.0,K=5 把 SR 从 0 提到 27%%。0=关闭')
    ap.add_argument('--arrive_k', type=int, default=5,
                    help='到达判据的连续不可见步数 K(配合 --arrive_a)')
    ap.add_argument('--arrive_guard', type=int, default=0,
                    help='到达判据保护期(步): escape/stall 触发后的这么多步内不判到达。0=关。'
                         '0914: ep294 里 stall 打破死锁后转向 ⇒ 目标滑出画面 ⇒ arrive 立刻'
                         '误判"到达"(真实还有 8.48m) ⇒ 提前停, 脱困收益全被吃掉。')
    ap.add_argument('--stall_n', type=int, default=30,
                    help='滞留(交替振荡)判据: 回看最近 N 步。0=关闭。'
                         '0914 新增: stuck(要求连续 forward)和 spin(要求净转角>300°)都漏掉'
                         '"左右交替转向"型死锁 —— 净转角≈0 且动作不连续, 两个检测器同时失效。'
                         '该型死锁在闭环 diag 里会打满 1000 步且零位移。')
    ap.add_argument('--stall_d', type=float, default=0.6,
                    help='滞留判据的净位移阈值(米): N 步净位移 < 此值即判滞留')
    ap.add_argument('--stall_min_fwd', type=int, default=5,
                    help='滞留判据里「N 步中至少这么多次 forward」的下限。默认 5（=原行为）。'
                         '⚠️ 0915 trace 实证: 摇摆死锁 ep63/ep360 的 500 步里 forward 只有 0~1 次、'
                         '翻转率 1.00、路径长 0.0~0.2m ⇒ n_fwd≈0 被这条挡在门外。'
                         '改成 0 可让它触发。安全性: 正常 scan 是 7转1进, 30 步内 ~3.75 次 forward、'
                         '位移 ~0.94m > stall_d(0.6) ⇒ 不会误伤 scan。')
    ap.add_argument('--sb_smooth', type=float, default=0.0,
                    help='慢脑双稳态抖动抑制阈值(归一化像素)。0=关。'
                         '位置几乎没变(画面没变)时, 若慢脑输出跳变超过此值则沿用上一帧。'
                         '0914 实证死锁 ep 的近静止 |Δnx| 达 0.42~0.60 且完全确定性; 建议 0.30')
    ap.add_argument('--scan_turn', type=int, default=7,
                    help='不可见时的探索: 周期内先转这么多步(TURN_DEG 各一次), 默认 7(原行为)')
    ap.add_argument('--scan_fwd', type=int, default=1,
                    help='不可见时的探索: 周期内转完后再前进这么多步, 默认 1(原行为)。'
                         '0914: 成败的分界线是「看得见目标的步数占比」(成功 0.717 / 失败 0.461), '
                         '而原策略 8 步才前进 1 步≈原地扫地。要试整圈+走远用 --scan_turn 12 --scan_fwd 4')
    ap.add_argument('--dual_locate', type=int, default=0,
                    help='0915 诊断: 每步额外记录 oracle 投影(参考路径终点, 含遮挡检查)与慢脑输出并排, '
                         '用于区分「慢脑指错方向」vs「目标根本不可见时的编造」。1=启用')
    ap.add_argument('--goal_memory', type=int, default=0,
                    help='⭐0922 目标记忆: 慢脑报「不可见」时, 用「上次可见目标的世界坐标 + 当前位姿」外推, '
                         '经 _oracle 通道喂给快脑 ⇒ **让目标永远可用**。'
                         '依据: 臂B 实测真值口径有 68%% 的步无目标(可用率仅 31.7%%), 触发扫描/脱困 ⇒ 走不动; '
                         '而「最远可见 waypoint」索引实测**几乎不跳变**(Δ中位 0) ⇒ 记忆不会过期。'
                         '⚠️ 与旧的 --dr_stop 不同: 那个用里程计判「是否到达」(已否决), 这个只用来「外推目标方向」。'
                         '默认 0=原行为逐位不变。')
    ap.add_argument('--goal_memory_ttl', type=int, default=200,
                    help='目标记忆的有效期(步); 超过则不再采信(回到原「不可见」行为)。默认 200')
    ap.add_argument('--oracle_pixel', type=int, default=0,
                    help='⭐0922 分离实验(臂 B): 目标换成「真值像素」(最远可见 waypoint 口径, 与训练同口径), '
                         '但**判停一字不动**(仍走像素→depth 反投影 + 全部原有闸门)。'
                         '⚠️ 与 --oracle_goal 的区别: 那个连判停也换成真值 dtg ⇒ 混淆了两个变量; '
                         '这个只换「目标来源」⇒ 单独量出**慢脑像素误差**的代价。'
                         '判读: B≈C(68%%) ⇒ 慢脑像素是主因(继续修慢脑); B≈A(11%%) ⇒ 反投影/判停是主因(该改接口)。'
                         '等价于「慢脑像素 100%% 命中」时的 SR 上界。本开关会自动打开 dual_locate。')
    ap.add_argument('--save_frames', default=None,
                    help='⭐0918: 把闭环每步的 rgb+位姿存到该目录(每个 ep 一个子目录)。'
                         '⚠️ 连 depth 一起存: 再标注要用「遮挡检查」, 它需要 depth(否则标签判不出来)。\n                        用途 = 「on-policy 再标注」: 闭环真实走过的位姿正是慢脑最需要适应的分布, '
                         '而 detour_data 只有"横向偏离+朝向goal"一种。存下来后用 '
                         'tools/gen_slowbrain_farthest.py 的同一套判据重新打标签即可。')
    ap.add_argument('--save_stride', type=int, default=2,
                    help='上条: 每多少步存一帧(默认 2, 省一半磁盘)')
    ap.add_argument('--arrive_win', type=int, default=60,
                    help='arrive 判据只看最近这么多次"报出 dist"的读数(滑动窗口)。'
                         '⚠️ 不用全程 min: 几百步前的偶发近读数会一直污染判据。')
    ap.add_argument('--arrive_stuck_n', type=int, default=0,
                    help='⭐0918 到达判据: 在 stuck_d 米内**持续不可见** >= N 步 ⇒ 判到达并停。'
                         '0=关(默认)。依据: 集A 的失败形态是「走到最近点后慢脑转不可见、原地扫视几百步」'
                         '(ep493 卡在 dtg=1.05m 打满 500 步)。对齐失败形态设计, 不是在调参。')
    ap.add_argument('--arrive_stuck_gate', type=float, default=3.0,
                    help='stuck 判据的距离闸门: 要求**慢脑历史报过**的最小 dist < 此值。'
                         '不加闸门会误停(保守型 adapter 有些 ep 全程不可见, 60 步就停)。')
    ap.add_argument('--arrive_stuck_win', type=int, default=1,
                    help='⭐0918 入口1: 闸门用哪段 dist 取最小。1(默认)=最近 arrive_win 次读数'
                         '(与 arrive 判据同口径); 0=全程最小(0918 旧行为)。'
                         '旧行为实证被早段偶发小值污染: ep1204 早段报过 1.65 却该挡、'
                         'ep805 历史 min 2.66 该保 —— 两者用全程 min 分不开, 必须换成最近窗口。')
    ap.add_argument('--arrive_stuck_d', type=float, default=0.5,
                    help='上条的位移阈值(米): 距锚点超过它就重置锚点(正常扫视每周期前进~1m, 不会触发)')
    ap.add_argument('--oracle_goal', type=int, default=0,
                    help='⭐0918 天花板实验: 1=完全绕过慢脑, 目标取「参考路径上沿弧长前瞻 4m 的点」'
                         '(与 probe_oracle_goal 同口径), 并用真值 dtg<stop_dist 判停。'
                         '其余控制器(escape/scan/快脑口径)一字不改 ⇒ 量出控制器天花板, '
                         '用来判断该继续修慢脑还是修控制器。')
    # ---- 0921 里程计推算到达(dead-reckoning arrival) ----
    # 动机(0921 实测 mix_fix/base 的 diag+trace 配对, 3543 步): 慢脑报的"目标距离"只在
    # 2.5~6m 带内跟真值走(报出/真值 = 0.82~1.20), 出带后基本是常数
    # (真值 0.5m 处报 3.9、20m 处报 4.5, r(dtg,报出)=0.127)。而原判停 `dist<stop_dist`
    # 恰恰只在信号失效的区间里做判断 ⇒ 判停靠运气(见 docs/慢脑进度.md #14)。
    # 解法: 只在**校准带内**取 fix, 记下目标的世界坐标, 之后用自身位姿积分推算剩余距离,
    # 不再依赖"目标出画之后"的慢脑输出。方向由几何给出, 不需要模型可信。
    # ⚠️ 本臂用仿真位姿当完美里程计 ⇒ 量的是"里程计无漂移"时的上界。
    #
    # ❌❌ 0921 结论: **本方案已被离线实测否决, 不要再上卡跑它**(见 docs/慢脑进度.md #14):
    #   用 diag 的真实 dist/d 复算, 推算剩余 vs 真值 r=0.137、中位误差 2.27m —— 根因是"报出的目标点"
    #   跟着 agent 走(近似恒在 ~4m 前方), fix 本身不可信。同一份数据还测出: 慢脑 dist 对
    #   「到了 vs 中途」的 AUC 只有 0.54~0.58(三个 adapter 都一样) ⇒ 信号里根本没有距离信息,
    #   任何基于它的判停都不可能成立。开关保留仅供复核, 默认关。
    #   同类被否决的还有「目标滑出画面下缘」信号: 首次 ny>=0.93 时真值 dtg 中位 7.75m(不是 2m)。
    ap.add_argument('--dr_stop', type=int, default=0,
                    help='1=里程计推算到达判停。❌0921 离线实测已否决(r=0.137), 别上卡跑; 保留仅供复核。默认 0=原行为逐位不变')
    ap.add_argument('--dr_band_lo', type=float, default=3.0,
                    help='校准带下界(米): 只有报出 dist 落在此带内才采信为 fix')
    ap.add_argument('--dr_band_hi', type=float, default=6.0,
                    help='校准带上界(米): 报出 dist 超出说明已饱和(远处)⇒ 不采信')
    ap.add_argument('--dr_target', type=float, default=2.5,
                    help='推算剩余距离 < 此值即判到达(默认与 stop_dist 同口径)')
    ap.add_argument('--dr_confirm', type=int, default=2,
                    help='连续 N 步满足才停(防单步抖动)')
    ap.add_argument('--dr_log', default=None,
                    help='逐 step 记录推算剩余距离, 便于审计(为什么停/为什么没停)')
    ap.add_argument('--sb_stop_confirm', type=int, default=1,
                    help='⭐0921 慢脑自带 STOP 输出: 连续 N 步吐 STOP 才认(默认 1=DualVLN 口径, 见到就停)。'
                         '旧 adapter 从不吐 STOP ⇒ 本开关对既有基线零影响')
    ap.add_argument('--out', default='r2r_infinitevl')
    ap.add_argument('--split', default='val_unseen',
                    help='⭐0919: 数据集 split。默认 val_unseen(评测用, 历史结果不变)。'
                         '传 train 时可跑 R2R 训练集 —— 采集 on-policy 帧**必须**用它: '
                         'val_unseen 的 scene 就是评测场景(11个里9个被评测ep占用), '
                         '在上面采帧训练 = scene 级泄漏。train split 有 10819 个 ep、'
                         '59 个 scene, 与 val_unseen 零重叠(官方分离)。')
    cli = ap.parse_args()
    if cli.oracle_pixel:
        cli.dual_locate = 1      # oracle_pixel 依赖 dual_locate 算出的「真值像素」(ov_nx/ov_ny)

    print('[progress] make_env 开始...', flush=True)
    env = make_env(cli.split, 640, 480, 79)
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
    fdr = open(cli.dr_log, 'w', encoding='utf-8') if cli.dr_log else None
    global _ACT_TRACE_FH
    if os.environ.get('TRACE_ACT'):
        _ACT_TRACE_FH = open(os.environ['TRACE_ACT'], 'w', encoding='utf-8')

    for ei in range(len(eps)):
        env.episodes = [eps[ei]]
        obs = env.reset()
        ep = eps[ei]
        instruction = (obs['instruction']['text'] if isinstance(obs.get('instruction'), dict)
                       else str(ep.instruction.instruction_text))
        agent.reset_episode()
        # 0915 dual_locate: oracle 目标 = 参考路径终点(世界系)
        # ⭐0918 天花板实验: oracle 目标用(参考路径 → 我们系 → 0.25m 稠密化)
        _ref_ours = None
        if cli.oracle_goal or cli.save_frames:   # ⭐0924 save_frames 也要参考路径（算 DAgger 的 ego 真值）
            try:
                _rp = np.asarray(ep.reference_path, np.float64)      # habitat (x, y, z)
                _ref_ours = densify_ours(np.stack([_rp[:, 0], -_rp[:, 2]], 1))
            except Exception as _e:
                if cli.oracle_goal:
                    raise RuntimeError(f'oracle_goal: ep{ep.episode_id} 取 reference_path 失败: {_e}')

        _gt_goal = None
        _ref_path = None
        if cli.dual_locate:
            for _attr in ('reference_path', 'goals'):
                try:
                    _v = getattr(ep, _attr)
                    if _v is None or len(_v) == 0:
                        continue
                    _gt_goal = (np.asarray(_v, np.float64)[-1] if _attr == 'reference_path'
                                else np.asarray(_v[0].position, np.float64))
                    if _attr == 'reference_path':
                        _ref_path = np.asarray(_v, np.float64)
                    break
                except Exception:
                    continue
            if _gt_goal is None:
                print(f'[warn] dual_locate: ep{ep.episode_id} 拿不到 oracle 目标, 该 ep 关闭', flush=True)
        # 多帧缓冲(最后一帧=当前帧)。span>0 时要留住 span+1 帧才能跨帧等距采样。
        _bufmax = cli.nframes if (cli.span or 0) <= 0 else max(cli.nframes, cli.span + 1)
        hist_rgb = deque(maxlen=_bufmax)
        mem_goal_world = None                  # ⭐0922 目标记忆: 上次可见目标的世界坐标(x,y)
        mem_goal_step = -10**9                 # 上次更新的步数(算 TTL 用)
        mem_used = False                       # 本步是否用了记忆(写进 diag 供验证)
        steps = 0
        min_dtg = float('inf')
        stopped = False
        scan = 0
        stop_votes = 0
        dist_hist = deque(maxlen=max(1, cli.stop_median_n))   # 时序滤波缓冲
        # 0913 脱困状态: 每个 episode 独立重置
        esc = Escape()
        esc.STALL_N = cli.stall_n
        esc.STALL_D = cli.stall_d
        esc.STALL_MIN_FWD = cli.stall_min_fwd
        prev_act = None                     # 上一步动作(卡死判据要"连续 N 步 forward")
        prev_pos = None                     # 上一步位置(算位移)
        prev_th = None                      # 上一步朝向(0915 stopfix: 停滞判据要连朝向一起看)
        _pos_gate = deque(maxlen=max(1, cli.stuck_gate_steps or 1))   # ⭐0924 位移门控：最近 N 步位置
        esc_goal = np.array([1.0, 0.0], np.float32)   # 逃脱转向侧的依据, 默认朝正前
        # 0913 到达判据状态: 慢脑历史报过的最小 dist + 连续不可见计数
        approach_min = float('inf')
        invis_streak = 0
        # 0918 「卡住且看不见」判据状态: 不可见期间的位置锚点 + 锚点对应的步号
        _invis_anchor, _invis_anchor_step = None, 0
        # 0918 arrive 判据的"最近窗口": 只看最近 arrive_win 次"报出 dist"的读数
        _dist_win = deque(maxlen=max(1, cli.arrive_win))
        # 0914 慢脑「双稳态抖动」抑制状态
        prev_nx, prev_ny = None, None
        lock_streak = 0
        prev_oge = None                     # ⭐0923 ego 抖动抑制用（见下方 _oge 分支）
        lock_streak_e = 0
        _frz = 0            # 0915 stopfix: 位置停滞连续步数(供「停滞豁免」判停)
        # 0921 里程计推算到达(见 --dr_stop 帮助): 目标世界坐标 + 采信它的 fix 信息
        _dr_goal = None
        _dr_fix_step, _dr_fix_dist = -1, None
        _dr_hits = 0
        # 0921 慢脑自带 STOP: 连续 N 次才认(默认 1 = DualVLN 口径, 见到就停)
        sb_stop_votes = 0

        while not env.episode_over and steps < cli.max_steps:
            # 重置「卡住」计时：dump_traceback_later 是绝对时限、不会自动重置，
            # 不重置的话跑超过 5 分钟就被无条件 exit（30 eps 必挂）。
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(300, exit=True)
            state = env.sim.get_agent_state()
            rgb = np.ascontiguousarray(obs['rgb'][:, :, :3])
            depth = np.squeeze(np.ascontiguousarray(obs['depth'])).astype(np.float32)
            position = np.asarray(state.position, np.float64)
            if cli.save_frames and steps % max(1, cli.save_stride) == 0:
                # ⭐0918 on-policy 存档: 闭环真实位姿(含偏离路径/背对目标的那些), 供离线再标注
                _sd = os.path.join(cli.save_frames, f'ep{ep.episode_id}')
                os.makedirs(_sd, exist_ok=True)
                np.savez_compressed(
                    os.path.join(_sd, f's{steps:04d}.npz'), rgb=rgb, depth=depth,
                    pos=position.astype(np.float32),
                    rot=np.array([state.rotation.x, state.rotation.y,
                                  state.rotation.z, state.rotation.w], np.float32))
                # ⭐0924 DAgger 标注：闭环真实位姿下的 ego 真值（用参考路径现场算，无需事后 habitat）
                if _ref_ours is not None:
                    try:
                        _ax, _ay, _ath, _ = agent.hab_pose_to_ours(position, state.rotation)
                        _oge = oracle_goal_ego(_ref_ours, _ax, _ay, _ath)
                        with open(os.path.join(cli.save_frames, '_annot.jsonl'), 'a', encoding='utf-8') as _fa:
                            _fa.write(json.dumps({
                                'episode_id': str(ep.episode_id), 'step': int(steps),
                                'frame': os.path.join(_sd, f's{steps:04d}.npz'),
                                'instruction': instruction,
                                'scene': str(getattr(ep, 'scene_id', '')),
                                'ego_fwd': round(float(_oge[0]), 3),
                                'ego_lat': round(float(_oge[1]), 3),
                            }, ensure_ascii=False) + '\n')
                    except Exception:
                        pass
            x, y, th, floor = agent.hab_pose_to_ours(position, state.rotation)
            if cli.stuck_gate_steps > 0:      # ⭐0924 位移门控：记录位置
                _pos_gate.append(np.asarray(position, np.float64)[:2])
            # 0915 stopfix: 「真停滞」计数 —— xyz 与朝向**同时**几乎不动才算。
            # ⚠️ 只比 position 会把「原地扫视」误判成停滞(转向不改 position) ⇒ 反而制造误停。
            if prev_pos is not None and prev_th is not None:
                _dp = float(np.linalg.norm(position - prev_pos))
                _dth = abs(float(th) - float(prev_th))
                _dth = min(_dth, abs(2 * np.pi - _dth))
                _frz = (_frz + 1) if (_dp < 0.02 and _dth < 0.02) else 0
            else:
                _frz = 0
            pts = agent.depth_to_world(depth, position, state.rotation)
            agent.world_pts = np.concatenate([agent.world_pts, pts])[-600000:]

            m = env.get_metrics()
            dtg = float(m.get('distance_to_goal', -1))
            min_dtg = min(min_dtg, dtg)

            # ⭐ 0918 depth 单位护栏(见文件头 _DEPTH_GUARD 说明): 宁可炸, 也不产出基于坏 depth 的废结论。
            if not _DEPTH_GUARD['done']:
                _DEPTH_GUARD['n'] += 1
                _DEPTH_GUARD['max_seen'] = max(_DEPTH_GUARD['max_seen'], float(depth.max()))
                if _DEPTH_GUARD['n'] >= 50:
                    _DEPTH_GUARD['done'] = True
                    if _DEPTH_GUARD['max_seen'] <= 1.0:
                        raise RuntimeError(
                            f'50 步内 depth 最大值始终 <=1.0 (={_DEPTH_GUARD["max_seen"]:.3f}), '
                            f'而当前 dtg={dtg:.2f}m ⇒ depth 是归一化值, make_env 的 '
                            f'normalize_depth=False 没生效。拒绝产出废结果。')

            # 0915 dual_locate: oracle 投影(参考路径终点) —— 与慢脑输出并排, 判定"指错方向"还是"不可见时编造"
            _orc = {}
            # ⭐0922: 无条件初始化 —— 供 --oracle_pixel 在「dual_locate 关闭」或「拿不到参考路径」时安全读取
            _ovl = {'ov_vis': 0, 'ov_nx': None, 'ov_ny': None, 'ov_dist': None, 'ov_j': None}
            if cli.dual_locate and _gt_goal is not None:
                _SENSOR_H = 1.25    # habitat benchmark/nav/vln_r2r.yaml: rgb_sensor.position[1]
                _q = np.array([state.rotation.x, state.rotation.y,
                               state.rotation.z, state.rotation.w], np.float64)
                _cp = np.asarray(position, np.float64) + np.array([0.0, _SENSOR_H, 0.0])
                _oh, _ow = int(rgb.shape[0]), int(rgb.shape[1])
                _ou, _ov, _ovis = project_goal_to_pixel(_gt_goal, _cp, _q, _ow, _oh, HFOV)
                _od = float(np.linalg.norm(_gt_goal - _cp))
                # 0918: 把**原始**投影值与遮挡判据的量全部记下来 —— 只记 o_vis 的话,
                # 分不清 "镜头外/在身后" (project 判的) 与 "被挡住" (遮挡判据判的),
                # 也看不到遮挡判据到底差多少。o_vis 曾被坏 depth 变成恒 0 而无人察觉。
                _oin = bool(_ovis)
                _odpix = None
                if _ovis:           # 遮挡检查: 该像素实测深度 << 相机到目标真实距离 ⇒ 被墙/家具挡住
                    _pu, _pv = int(_ou), int(_ov)
                    _pt = depth[max(0, _pv - 2):_pv + 3, max(0, _pu - 2):_pu + 3]
                    _pt = _pt[(_pt > 0.3) & (_pt < 12.0)]
                    _odpix = round(float(np.median(_pt)), 2) if _pt.size else None
                    if _pt.size and float(np.median(_pt)) < _od * 0.85:
                        _ovis = False
                # ⭐ 0918 口径对齐(关键): 慢脑训练目标是「沿路径**最远的可见 waypoint**」(局部目标),
                # **不是** episode 终点。用终点当 oracle 是错的 —— 实测终点只有 6.6% 的步在画面内
                # (8.3m 外、常拐弯), 于是 o_vis 恒 0, 看起来像"慢脑在编造"。
                # 这里按**与训练完全相同**的口径算 oracle: 从最远往近扫参考路径, 取第一个
                # 「投影在框内 且 未被遮挡」的点。occ_ratio 用 gen_slowbrain_farthest 的默认 0.95。
                _ovl = {'ov_vis': 0, 'ov_nx': None, 'ov_ny': None, 'ov_dist': None, 'ov_j': None}
                if _ref_path is not None and len(_ref_path):
                    for _j in range(len(_ref_path) - 1, -1, -1):
                        _u2, _v2, _ok2 = project_goal_to_pixel(_ref_path[_j], _cp, _q, _ow, _oh, HFOV)
                        if not _ok2:
                            continue
                        _d2 = float(np.linalg.norm(_ref_path[_j] - _cp))
                        _qu, _qv = int(_u2), int(_v2)
                        _pt2 = depth[max(0, _qv - 2):_qv + 3, max(0, _qu - 2):_qu + 3]
                        _pt2 = _pt2[(_pt2 > 0.3) & (_pt2 < 12.0)]
                        if _pt2.size and float(np.median(_pt2)) < _d2 * 0.95:
                            continue
                        _ovl = {'ov_vis': 1, 'ov_nx': round(float(_u2) / _ow, 4),
                               'ov_ny': round(float(_v2) / _oh, 4),
                               'ov_dist': round(_d2, 2), 'ov_j': int(_j)}
                        break
                _orc = {'o_vis': int(_ovis),
                        'o_inframe': int(_oin),
                        'o_pixd': _odpix,
                        'o_u': round(float(_ou), 1) if _oin else None,
                        'o_v': round(float(_ov), 1) if _oin else None,
                        'o_nx': round(float(_ou) / _ow, 4) if _ovis else None,
                        'o_ny': round(float(_ov) / _oh, 4) if _ovis else None,
                        'o_dist': round(_od, 2),
                        **_ovl}

            # 慢脑定位终点（HTTP）
            hist_rgb.append(rgb)
            fr_send = _pick_frames(hist_rgb, cli.nframes, cli.span)
            if len(fr_send) < cli.nframes:          # 开局帧数不足: 用最早帧补齐
                fr_send = [fr_send[0]] * (cli.nframes - len(fr_send)) + fr_send
            if cli.oracle_goal:
                # ⭐0918 天花板实验: 完全跳过慢脑 HTTP。目标 = 沿参考路径前瞻 4m 的点(ego 系),
                # 判停用真值 dtg(与 probe_oracle_goal 同口径 —— 探针 65.6%→77.7% 就是这个口径)。
                # 其余控制器(escape/scan/快脑 status_gh+goal_dir)一字不改 ⇒ 差异只来自「目标来源」。
                if dtg < cli.stop_dist:
                    stopped = True
                    break
                r = {'visible': True, 'nx': None, 'ny': None,
                     '_oracle': oracle_goal_ego(_ref_ours, x, y, th)}
            elif cli.oracle_pixel:
                # ⭐0922 分离实验(臂 B): 只把「目标来源」换成真值像素(最远可见 waypoint, 与训练同口径),
                #   判停**一字不动** —— 继续走「像素→depth 反投影」和全部原有闸门。
                #   这样 A(慢脑像素)/B(真值像素)/C(oracle ego 点) 三点就能把误差来源分开:
                #     B≈C ⇒ 反投影链路没问题, 慢脑像素是主因 ⇒ 继续修慢脑
                #     B≈A ⇒ 反投影/判停本身不灵 ⇒ 该改接口(让慢脑直接输出 ego 点, 绕开反投影)
                if _ovl.get('ov_vis'):
                    r = {'visible': True, 'nx': _ovl['ov_nx'], 'ny': _ovl['ov_ny'], 'stop': False}
                else:
                    r = {'visible': False, 'nx': None, 'ny': None, 'stop': False}
            else:
                try:
                    r = requests.post(cli.server + '/locate', json={
                        'frames': [{'rgb': base64.b64encode(f.tobytes()).decode(),
                                    'shape': list(f.shape)} for f in fr_send],
                        'instruction': instruction}, timeout=90).json()
                except Exception as e:
                    print(f'[warn] slowbrain server err: {e}', flush=True)
                    r = {'visible': False, 'nx': None, 'ny': None}
                # ⭐0923 ego 版：slowbrain_server_ego 直接回 ego 相对坐标（米）⇒ 走已有的
                #   `_oracle` 通道喂快脑，**完全跳过「像素→depth 反投影」**
                #   （臂 B/C 实测那条链路吃掉 27.8pt：B(完美像素+反投影)41.8% ≪ C(ego点)69.6%）。
                #   visible 在这里的含义从「目标在不在画面里」变成「目标是否可用（解析成功）」。
                if r.get('ego_fwd') is not None:
                    r['_oracle'] = np.array([r['ego_fwd'], r['ego_lat']], np.float32)
                    r['visible'] = True

            # ---- ⭐0921 慢脑自带的 STOP 输出(DualVLN/NaVid 口径) ----
            # 训练数据里 `dtg<3m` 的帧标签就是 `[TARGET] STOP`（见 tools/gen_stop_dataset_0921.py）。
            # 旧 adapter 从不输出 STOP ⇒ 这里恒为 False, 对既有基线**零影响**。
            # 这是「判停」的正路：不再靠"报像素→反投影距离"（那个量在 dtg<2m 区间 93% 无像素可报）。
            if r.get('stop'):
                sb_stop_votes += 1
                if sb_stop_votes >= cli.sb_stop_confirm:
                    stopped = True
                    break
            else:
                sb_stop_votes = 0

            # ⭐0922 目标记忆: 让目标**永远可用**(依据: 臂B 实测真值口径 68% 的步无目标)
            #   可见 ⇒ 反投影出 ego 点(与下面 _oge 分支同一套公式) ⇒ 转世界坐标存起来
            #   不可见 ⇒ 用当前位姿把记忆中的世界坐标转回 ego 点, 经 _oracle 喂快脑
            #   ⚠️ 必须在 diag 之前执行, 否则 'mem_hit' 会记成上一步的值
            mem_used = False
            if cli.goal_memory:
                if r.get('visible') and r.get('nx') is not None:
                    _mu = int(min(max(r['nx'] * depth.shape[1], 0), depth.shape[1] - 1))
                    _mv = int(min(max(r['ny'] * depth.shape[0], 0), depth.shape[0] - 1))
                    _mp = depth[max(0, _mv - 2):_mv + 3, max(0, _mu - 2):_mu + 3]
                    _mp = _mp[(_mp > 0.3) & (_mp < 12.0)]
                    if _mp.size:
                        _md = float(np.median(_mp))
                        _mfx = (depth.shape[1] / 2.0) / np.tan(np.deg2rad(HFOV / 2.0))
                        _mge = np.array([_md, -(_mu - (depth.shape[1] - 1) / 2.0) * _md / _mfx], np.float32)
                        _mc, _ms = float(np.cos(th)), float(np.sin(th))
                        mem_goal_world = np.array([x + _mc * float(_mge[0]) - _ms * float(_mge[1]),
                                                   y + _ms * float(_mge[0]) + _mc * float(_mge[1])])
                        mem_goal_step = steps
                elif mem_goal_world is not None and (steps - mem_goal_step) <= cli.goal_memory_ttl:
                    _mc, _ms = float(np.cos(th)), float(np.sin(th))
                    _mdx = mem_goal_world[0] - x
                    _mdy = mem_goal_world[1] - y
                    r = {'visible': True, 'nx': None, 'ny': None,
                         '_oracle': np.array([_mc * _mdx + _ms * _mdy,
                                              -_ms * _mdx + _mc * _mdy], np.float32),
                         'stop': bool(r.get('stop'))}
                    mem_used = True

            if fdiag is not None:      # 逐步诊断(定位「5m 墙」: 是看不见, 还是看见了但目标不对)
                _dv = depth[(depth > 0.1) & (depth < 30.0)]
                fdiag.write(json.dumps({
                    'ep': str(ep.episode_id), 'step': steps, 'dtg': round(float(dtg), 2),
                    'vis': bool(r['visible']), 'nx': r.get('nx'), 'ny': r.get('ny'),
                    # 全局深度统计: 用于区分「慢脑指错」vs「agent 确实贴在障碍物前」
                    'depth_med': round(float(np.median(_dv)), 2) if _dv.size else None,
                    'depth_p10': round(float(np.percentile(_dv, 10)), 2) if _dv.size else None,
                    'depth_p90': round(float(np.percentile(_dv, 90)), 2) if _dv.size else None,
                    'mem_hit': int(mem_used),      # ⭐0922: 本步目标是否来自「目标记忆」而非真可见
                    **_orc,
                }, ensure_ascii=False) + '\n')
                fdiag.flush()          # 0913: 不 flush 则尾部数据靠退出时刷盘, 实测会丢(ep1704 丢了 44 步)

            if not r['visible']:
                invis_streak += 1
                # ⭐0918 「卡在原地且看不见」→ 停。这是**对着实测失败形态**设计的判据。
                # 依据(diag_dfx_NA.jsonl, 集A): 4 个「差一点」的失败**形态完全一致** ——
                # agent 走到最近点后慢脑转为不可见, 然后**位置和朝向几乎不动**地扫视几百步:
                #   ep493 : 最后报 dist=1.88, 之后全不可见, dtg 卡在 1.05m, 打满 500 步
                #   ep898 : dtg 卡在 2.37m;  ep1007: 卡在 2.85m;  ep1204: 卡在 4.63m
                # ⚠️ 用 approach_min 的旧 arrive 判据(A=4,K=6)实测**净负**: approach_min<4
                #    在前几步就满足, 于是任何 6 步不可见都触发 ⇒ ep898 第 29 步就误停在 5.25m
                #    (基线能走到 2.37m)。改成「位移锚点」: 只有当 agent 在 stuck_d 米内
                #    持续不可见 >= stuck_n 步才停 —— 正常扫视每周期会前进 ~1m, 不会触发。
                # ⚠️ 必须有距离闸门: 新 adapter 是"保守型", 有些 ep **从第 0 步起就全程不可见**
                # (ep1777 前 61 步一次都没报过可见), 裸的 stuck 判据会在 60 步时误停在 6.27m。
                # 实测闸门取值: ep1204 报过 min dist **1.04**、ep805 报过 **2.66**(该保),
                # 而误停的 ep1777 **一次都没报过**(该挡) ⇒ `approach_min < 3.0` 正好分开。
                # ⭐0918 入口1: 闸门也改用「最近窗口」最小值。全程 min 会被早段偶发小值污染
                # —— 实测 ep1204 早段报过 1.65(该挡)、ep805 历史 min 2.66(该保),
                #    两者用全程 min 分不开; 最近窗口能分开("早段那个小值已经滑出窗口")。
                _rmin = min(_dist_win) if _dist_win else float('inf')
                _gate_ref = approach_min if cli.arrive_stuck_win <= 0 else _rmin
                if cli.arrive_stuck_n > 0 and _gate_ref < cli.arrive_stuck_gate:
                    _p_now = np.asarray(position, np.float64)[:2]
                    if _invis_anchor is None:
                        _invis_anchor, _invis_anchor_step = _p_now.copy(), steps
                    elif float(np.linalg.norm(_p_now - _invis_anchor)) > cli.arrive_stuck_d:
                        _invis_anchor, _invis_anchor_step = _p_now.copy(), steps
                    if steps - _invis_anchor_step >= cli.arrive_stuck_n:
                        stopped = True
                        break
                # ---- 0913 到达判据: 曾接近过 + 持续不可见 ⇒ 目标已滑出画面(到达/走过) ----
                # 根因: 慢脑报的 dist 有 ≈2.1m 硬下界, stop_dist 永远等不到;
                #       而 agent 走过终点后 dense_path[i+1:] 为空, 同样判不可见。
                # 实测 far2a 15eps 重放: A=4.0,K=5 → SR 0% → 27%。
                # ⭐0918 修正: 用**最近 arrive_win 步内**报告过的最小 dist, 而不是全程 approach_min。
                # 实测(diag_dfx_NA/C2): 「最近点之后的最长连续不可见」只有 10~32 步(不是几百步),
                # 所以 A=4/K=6 那种宽 A 会**很早**就满足 approach_min<4 然后被 10 步不可见触发
                # ⇒ ep898 第 29 步误停在 5.25m。而窄 A(2.5) 配"最近窗口"既要求真的近过、
                # 又不会被几百步前的偶发近读数污染。
                # (上面的 `_rmin` 已按同一窗口算好, 两条判据共用一个口径)
                if cli.arrive_a > 0 and _rmin < cli.arrive_a \
                        and invis_streak >= cli.arrive_k \
                        and steps - esc.last_escape_step > cli.arrive_guard:
                    # 0914: arrive_guard 保护期 —— escape/stall 刚触发过时, 目标不可见是
                    # "被我们自己转走的", 不是"走到了". 实证 ep294: stall 在 step~30 打破死锁,
                    # agent 转向后目标滑出画面(vis 1→0), arrive 判据立刻误判到达并提前停,
                    # 而真实距离还有 8.48m —— 脱困的收益被这个误停完全吃掉。
                    stopped = True
                    break
                scan += 1
                # 0914: 探索周期参数化。原为「7 步转(210°) + 1 步前进」——近乎原地扫地,
                # 且 210° 不是整圈会漏掉身后的方向。默认值 (7,1) 经 (scan-1)%period 还原后
                # 与原 `scan % 8 == 0` **逐位一致**, 所以历史结果仍可比。
                # 要试「转满整圈再走远」用 --scan_turn 12 --scan_fwd 4 (12×30°=360°)。
                _p = cli.scan_turn + cli.scan_fwd
                _ph = (scan - 1) % _p if _p > 0 else 0
                act = 'move_forward' if _ph >= cli.scan_turn else 'turn_left'
            else:
                scan = 0
                invis_streak = 0
                _invis_anchor, _invis_anchor_step = None, 0   # 0918: 看见即重置"卡住且看不见"的锚点
                # ⭐0923 ego 模式：slowbrain_server_ego **不返回 nx/ny**（它直接给 ego 点，
                #   由上面的 `_oracle` 通道喂快脑）⇒ 必须用 .get，否则 KeyError: 'nx'。
                #   nx/ny=None 时下游的像素系逻辑（双稳态抖动抑制等）会走 `if nx is None` 的安全分支。
                nx, ny = r.get('nx'), r.get('ny')
                # ---- 0914 慢脑双稳态抖动抑制 (治本; stall 只是治标) ----
                # 实证: 死锁 ep 在**位置几乎没变(画面没变)**时, 相邻步 nx 相差 0.42~0.60
                # (半个画面宽), 且**完全确定性** —— far2h/far2j/sb_OFF_A 三次跑出逐位相同的 0.5131。
                # 快脑据此左右摇摆 ⇒ 原地打转 ⇒ 打满 500 步。这里在源头掐掉跳变。
                # ⚠️ 只在「几乎静止」时启用: agent 真在移动/转身时, 目标像素本来就该大幅移动。
                if nx is None:
                    # ⭐0922 修复: 目标来自 _oracle(oracle_goal / 目标记忆)时 nx 为 None,
                    #   而下面的平滑会对 nx 做减法 ⇒ TypeError: NoneType - float
                    #   (0922 memv 三片实测就崩在这; 而 oracle_goal 只是没设 sb_smooth 才没暴露)
                    #   这类目标本就无可平滑的像素 ⇒ 重置状态并跳过。
                    prev_nx, prev_ny, lock_streak = None, None, 0
                elif cli.sb_smooth > 0:
                    _moved_since = (float(np.linalg.norm(position - prev_pos))
                                    if prev_pos is not None else 99.0)
                    if prev_nx is not None and _moved_since < 0.05 \
                            and abs(nx - prev_nx) > cli.sb_smooth:
                        lock_streak += 1
                        if lock_streak <= 30:      # 安全阀: 抑制太久就放行, 避免永久锁死
                            nx, ny = prev_nx, prev_ny
                    else:
                        lock_streak = 0
                    prev_nx, prev_ny = nx, ny
                H, W = depth.shape
                _oge = r.get('_oracle')
                # ⭐0923 ego 双稳态抖动抑制 —— **必须有，否则 ego 模式必然原地打转**。
                #   上面那段 nx 版平滑在 `nx is None` 时被显式跳过（0922 为修 NoneType 崩溃），
                #   而 ego 模式下 nx 恒为 None ⇒ **专治「原地打转」的抑制整个失效**。
                #   实测 ep1608(冒烟)：agent 一步没动(dtg 恒 6.97m)，而 goal_lat 在
                #   +3.52 / -3.92 之间**严格隔步交替** 60 步 ⇒ 快脑被左右指挥 ⇒ 锁死到打满步数。
                #   ego 点是 2D 向量，与 nx 同理可平滑：位置几乎没动而点大幅跳变 ⇒ 沿用上一步。
                #   阈值 1.0m：实测抖动幅度 ~7.4m，而正常输出变化多在 1m 内。
                if _oge is not None:
                    _moved = (float(np.linalg.norm(position - prev_pos))
                              if prev_pos is not None else 99.0)
                    _jump = (float(np.linalg.norm(_oge - prev_oge))
                             if prev_oge is not None else 0.0)
                    if prev_oge is not None and _moved < 0.05 and _jump > _EGO_SMOOTH:
                        lock_streak_e += 1
                        if lock_streak_e <= 30:      # 安全阀：抑制太久就放行，避免永久锁死
                            _oge = prev_oge
                    else:
                        lock_streak_e = 0
                    prev_oge = np.asarray(_oge, np.float32)
                if _oge is not None:
                    # 天花板臂: patch/d 只是占位(让下面公共尾部原样跑), goal_ego 直接用 oracle
                    u, v = int(W // 2), int(H // 2)
                    patch = np.array([float(np.linalg.norm(_oge))], np.float32)
                else:
                    u = int(min(max(nx * W, 0), W - 1))
                    v = int(min(max(ny * H, 0), H - 1))
                    patch = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
                    patch = patch[(patch > 0.3) & (patch < 12.0)]
                if patch.size == 0:
                    act = 'turn_left'
                else:
                    d = float(np.median(patch))
                    fx = (W / 2.0) / np.tan(np.deg2rad(HFOV / 2.0))
                    if _oge is not None:
                        goal_ego = np.asarray(_oge, np.float32)
                    else:
                        goal_ego = np.array([d, -(u - (W - 1) / 2.0) * d / fx], np.float32)
                    dist = float(np.hypot(*goal_ego))
                    # 0921: 只在**校准带内**采信一次 fix。带外慢脑报的距离已饱和(近/远都报~4m),
                    # 拿它定位会把目标放到错误的地方。带内实测比值 0.82~1.20 ⇒ 可信。
                    if cli.dr_stop > 0 and cli.dr_band_lo <= dist <= cli.dr_band_hi:
                        # ego→世界: (x,y,th) 为数学约定(实证 atan2(dy,dx)≡th, 中位误差 0.1°);
                        # goal_ego[1]>0 表示目标在**左**(eval_r2r_escape.py:114 turn_left)
                        _c, _s = float(np.cos(th)), float(np.sin(th))
                        _dr_goal = np.array([x + _c * float(goal_ego[0]) - _s * float(goal_ego[1]),
                                             y + _s * float(goal_ego[0]) + _c * float(goal_ego[1])])
                        _dr_fix_step, _dr_fix_dist = steps, float(dist)
                    # ⭐ 0912: 快脑 DD 的训练口径是「沿路径前瞻 GOAL_AHEAD=4m 的点」
                    # (见 probe_oracle_goal.oracle_goal_ego —— 探针能跑 65.6% 就靠这个口径)。
                    # 而慢脑给的是**终点**, 可能远在 8~20m 外 ⇒ 喂给快脑超出训练分布 ⇒ 走不动。
                    # 故: 保持方向不变, 把长度截断到 4m 后喂快脑; 用真实终点距离判 stop。
                    _GA = 4.0
                    # 0915 A/B (goal_len4): 原写法只截断「太远」, 漏了「太近」——
                    # 慢脑所指像素深度 d 中位仅 1.0m(它偏画面中下部), 90.2% 的步 dist<4,
                    # ⇒ 快脑收到 ~1m 的目标点, 而它训练口径是「4m 前瞻点」⇒ 分布外 ⇒ 走不动。
                    # goal_len4=1 时一律归一化到 4m(正标量缩放, 方向不变); dist 本身不动, 判停逻辑不变。
                    goal_feed = (goal_ego * (_GA / dist)) if (dist > _GA or cli.goal_len4) else goal_ego
                    goal_ego = goal_feed.astype(np.float32)
                    # ⭐0928 敏感度扫描：受控扰动（只用于量「目标误差→SR」的曲线）
                    if cli.perturb_lat != 0.0:
                        goal_ego = np.array([goal_ego[0], goal_ego[1] + cli.perturb_lat], np.float32)
                    if cli.perturb_scale != 1.0:
                        goal_ego = (goal_ego * cli.perturb_scale).astype(np.float32)
                    # ⭐0927 侧方自锁 A/B（见 docs/0927-OSR崩塌根因-侧方自锁.md）
                    #   K=8 闭环里模型原始输出的方向角中位 73.9°（横向 -3.3 / 前向 1.0），
                    #   而训练集标签 |角| 中位仅 23.5% ⇒ 快脑收到「正侧方」目标 ⇒ 原地转 ⇒ 自锁。
                    #   本分支把横向夹到 ±clamp；ego_norm4=1 时再把模长补回 4m（快脑训练口径）。
                    _fwd_pre, _lat_pre = float(goal_ego[0]), float(goal_ego[1])
                    if cli.ego_lat_clamp > 0:
                        _lat = float(np.clip(_lat_pre, -cli.ego_lat_clamp, cli.ego_lat_clamp))
                        _fwd = _fwd_pre
                        if cli.ego_norm4:
                            _n = float(np.hypot(_fwd, _lat))
                            if _n > 1e-6:
                                _fwd, _lat = _fwd * _GA / _n, _lat * _GA / _n
                        goal_ego = np.array([_fwd, _lat], np.float32)
                    # 几何一致性: 该像素若真是地面目标, 距离应为 h/tan(俯角); 实测远小于此 ⇒ 指到近处障碍物
                    _vfov = 2.0 * np.degrees(np.arctan(np.tan(np.deg2rad(HFOV / 2.0)) / (float(W) / H)))
                    # 0918: oracle 天花板臂 ny=None ⇒ 跳过几何一致性(其默认 geo_check=0 时结果恒 True, 无影响)
                    if ny is None:
                        geo_ok = True
                    else:
                        _pitch = (float(ny) - 0.5) * _vfov
                        _d_geo = (1.30 / np.tan(np.deg2rad(_pitch))) if _pitch > 1.0 else 1e9
                        geo_ok = (cli.geo_check <= 0.0) or (d >= _d_geo * cli.geo_check)
                    if fdiag is not None:
                        fdiag.write(json.dumps({
                            'ep': str(ep.episode_id), 'step': steps, 'dtg': round(float(dtg), 2),
                            'vis': True,
                            'nx': (float(nx) if nx is not None else None),
                            'ny': (float(ny) if ny is not None else None), 'd': round(d, 2),
                            'goal_fwd': round(float(goal_ego[0]), 2),
                            'goal_lat': round(float(goal_ego[1]), 2),
                            'dist': round(float(dist), 2),
                            'lat_pre': round(_lat_pre, 2),   # 0927: 夹之前的横向(用于核对 A/B 真的生效)
                            'fwd_pre': round(_fwd_pre, 2),
                            'lat_clamp': float(cli.ego_lat_clamp),
                            **_orc,
                        }, ensure_ascii=False) + '\n')
                        fdiag.flush()  # 0913: 同上, 保证停止判决行实时落盘
                    dist_hist.append(dist)
                    approach_min = min(approach_min, dist)   # 0913: 供到达判据
                    _dist_win.append(dist)                   # 0918: arrive 判据的"最近窗口" 
                    dist_f = float(np.median(dist_hist)) if cli.stop_median_n > 0 else dist
                    # 0915 stopfix: 「停滞豁免」—— agent 卡在原地且慢脑报"目标在 stop_dist 内"时,
                    #   即使 d < min_stop_depth 也认。依据: 真到达时目标 < 2.1m 几何上不在画面内,
                    #   慢脑给的 d 本就不可信(它指中下部近处), min_stop_depth 会把它挡在门外。
                    #   实证 ep1429 停在 dtg=0.90m(远优于 stop_dist=2.5)却不停, 白烧 370 步。
                    #   frozen_stop=0(默认) => 原行为不变。
                    _frz_ok = (cli.frozen_stop > 0 and _frz >= cli.frozen_stop)
                    # ⭐0921修: oracle 模式必须跳过这条几何判停。
                    #   本意是给「非 oracle」用慢脑报的距离判停; 但 oracle 下 ny=None
                    #   ⇒ geo_ok 被强制 True, 守卫只剩 dist_f<stop_dist。而 dist 是
                    #   「到 4m 前瞻点的距离」非真值 dtg ⇒ agent 走到某些位置 dist<2.5
                    #   会误判到达: 实证 ep1105 最后 2 步 dist=2.34 ⇒ 停在 dtg=5.61m。
                    #   全量: 41~48% 的失败 ep 在 50 步内被掐断, 而 dtg 仍有 5.3~5.6m。
                    # ⭐0923 ego 模式：`_oge is not None` ⇒ 判停**必须跳过 min_stop_depth**。
                    #   那个阈值是给**像素版**标的（runner 传 2.3）：像素版 d 是「画面中下部地标的深度」，
                    #   而 ego 模式下 d == ||ego||。实测「该停」时 ||ego|| 中位仅 **1.25m**
                    #   ⇒ 会被 2.3 整个挡在门外，判停窗口被压成 [2.3, 2.5] 的 **0.2m 窄带**
                    #   ⇒ 几乎永不触发 = 就是 memory 里那个「判停窄窗口机制」（"到了不停"的老机制）。
                    #   跳过后的判据只剩 dist_f < stop_dist，即 **||ego|| < stop_dist**：
                    #   真值口径实测精度 96.4% / 召回 98.9%（对照旧像素版 AUC 仅 0.54~0.58）。
                    # ⭐0924 位移门控：agent 真的卡住（最近 N 步几乎没动）时禁止判停
                    _stuck_now = False
                    if cli.stuck_gate_steps > 0 and len(_pos_gate) == cli.stuck_gate_steps:
                        _stuck_now = float(np.linalg.norm(_pos_gate[-1] - _pos_gate[0])) < cli.stuck_gate_d
                    if (not cli.oracle_goal) and dist_f < cli.stop_dist and (d >= cli.min_stop_depth or _frz_ok or _oge is not None) and geo_ok \
                            and steps >= cli.min_steps and not _stuck_now:
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
                    esc_goal = goal_ego.copy()   # 0913: 供脱困判转向侧(只看 goal_ego[1] 的符号)

            # ---- 0921 里程计推算到达(dead-reckoning arrival) ----
            # 每步都跑(可见与否都一样): 这正是它的意义 —— 目标出画之后不再依赖慢脑输出。
            # 用自身位姿(x,y)对 fix 时记下的目标世界坐标算剩余距离, 小于 dr_target 即判到达。
            if cli.dr_stop > 0:
                _dr_remain = (float(np.hypot(_dr_goal[0] - x, _dr_goal[1] - y))
                              if _dr_goal is not None else None)
                if fdr is not None:
                    fdr.write(json.dumps({
                        'ep': str(ep.episode_id), 'step': steps, 'dtg': round(float(dtg), 2),
                        'vis': bool(r['visible']),
                        'dr_remain': (round(_dr_remain, 3) if _dr_remain is not None else None),
                        'dr_hits': _dr_hits, 'fix_step': _dr_fix_step, 'fix_dist': _dr_fix_dist,
                    }, ensure_ascii=False) + '\n')
                    fdr.flush()
                if _dr_remain is not None:
                    if _dr_remain < cli.dr_target:
                        _dr_hits += 1
                        if _dr_hits >= cli.dr_confirm:
                            stopped = True
                            break
                    else:
                        _dr_hits = 0

            # ---- 卡死/转圈脱困(0913 移植自 probe_oracle_goal.Escape) ----
            # 判据基于「上一步的动作 + 位移」, 触发则本步强制走逃脱序列, 覆盖慢脑/快脑的决策。
            if prev_act is not None:
                _moved = float(np.linalg.norm(position - prev_pos)) if prev_pos is not None else 0.0
                _ea = esc.update(prev_act, _moved, esc_goal, steps, dtg=float(dtg), pos=prev_pos)
                if _ea:
                    act = _ea

            if _ACT_TRACE_FH is not None:     # 0915 诊断: 逐 step 记 动作+位姿+真实距离
                _ACT_TRACE_FH.write(json.dumps({
                    'ep': str(ep.episode_id), 'step': steps, 'act': act,
                    'dtg': round(float(dtg), 3),
                    'x': round(float(x), 3), 'y': round(float(y), 3), 'th': round(float(th), 4),
                }, ensure_ascii=False) + '\n')
                _ACT_TRACE_FH.flush()

            obs = env.step(ACT2ID[act])
            steps += 1
            prev_act, prev_pos, prev_th = act, position, th

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
    if fdr is not None:
        fdr.close()
    rows = [json.loads(l) for l in open(cli.out + '.jsonl')]
    n = len(rows)
    print(f'==== R2R+INFINITEVL n={n} SR={sum(r["success"] for r in rows) / n * 100:.1f} '
          f'OSR={sum(r["os"] for r in rows) / n * 100:.1f} '
          f'NE={np.mean([r["ne"] for r in rows]):.2f} ====', flush=True)


if __name__ == '__main__':
    main()
