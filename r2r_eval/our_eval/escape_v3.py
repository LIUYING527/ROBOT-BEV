"""独立脱困模块 EscapeV3 —— 从 probe_oracle_goal.py 抽出, 零外部依赖(只要 numpy)。

这是 R2R 快脑当前最好成绩的**关键推理侧配方**, 与权重无关、零训练改动:
    goalmlp_ep5 ckpt + EscapeV3 脱困 + --v3_noprog 2  →  val_unseen 全量 SR 79.6%
    (同一 ckpt: 无脱困 51.1% / 旧 v2 脱困 65.6% / v3 默认 noprog4 77.8%)

用法(与探针里完全一致):
    esc = EscapeV3()                      # 或 EscapeV2() / Escape()
    esc.reset()                           # 每个 episode 开始
    act = esc.update(prev_act, moved_m, goal_ego, step, dtg)
    # 返回 None → 本步走模型; 返回动作 → 本步跳过模型直接执行
    # 命中后把内部 plan 逐步 pop 执行, 每个被接管的步 n_steps += 1

⚠️ 参数注入(与 CLI 对应, 默认值 = 0913 定版行为):
    Escape.STUCK_N / STUCK_D / GUARD_DTG / MAX_ESCAPE / TURN_DEG
    EscapeV2.MIN_CLEAR, EscapeV3.NO_PROG_N / UNKNOWN_CLEAR / LEVEL_SAT / FWD_V2

⚠️ TURN_DEG 保持 30.0: habitat vln_r2r.yaml 实际是 15°/步, 但全部历史数字都是在 30 下产生的,
    且 0913 实测 30 优于 15(+6.0pt @100 条单层失败集)。escape 的参数就是在这个有效行为下
    调出来的, 不要当 bug 去"修正"。
"""
import numpy as np  # 0914修: 原文件声明"只要numpy"却没import —— 抽出后从未被真正import过,
                    # 探针里用的是它自己内置的同名类, 所以这个缺失一直没暴露。


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


