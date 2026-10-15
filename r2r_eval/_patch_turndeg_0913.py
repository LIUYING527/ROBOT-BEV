"""0913 修 escape 里的转向角换算 —— 所有处都按 30°/步 写, 但 habitat 实际是 15°/步。

铁证: habitat 的 benchmark/nav/vln_r2r.yaml 默认 turn_angle=15, 动作空间里
      TurnLeftAction/TurnRightAction 的 tilt_angle 都是 15; make_env 也显式设 15。
      而 Escape 的代码注释写着 "R2R turn = 30°/步"。

后果(每一处转向都只有意图的一半):
  - V1 逃脱计划 [side]*3 / ['turn_left']*6      → 意图 90°/180°, 实际 45°/90°
  - V2/V3 nt = round(|tgt|/30)                   → 想转到 tgt, 实际转 tgt/2
  - V3 无进展"掉头走远" ['turn_left']*6          → 意图 180°, 实际 90°(根本没掉头)
  - V3 LEVELS=(0,±60,±120,180) 的设计覆盖范围    → 实际只能到 ±90°
  - 打转检测 net=sum(±30), 阈值 300             → 量的其实是 150° 净转角

修法: 提出类属性 Escape.TURN_DEG, 全部换算改用它。
⚠️ 默认 30.0 = **逐位保持现有行为**(项目惯例, 既有数字全部可比);
   用 --turn_deg 15 才是修正后的真实角度, 需实测 A/B 后再决定是否改默认。
"""
import py_compile
import shutil

F = '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/probe_oracle_goal.py'
shutil.copyfile(F, F + '.bak0913td')
src = open(F, encoding='utf-8').read()


def sub(old, new, tag, n_expect=1):
    n = src.count(old)
    assert n == n_expect, f'[{tag}] 匹配 {n} 次, 期望 {n_expect} 次 —— 放弃'
    return src.replace(old, new)


# --- 1) 基类加 TURN_DEG ---
src = sub(
    """    MAX_ESCAPE = 0       # 逃脱接管步数上限(0=无限)
""",
    """    MAX_ESCAPE = 0       # 逃脱接管步数上限(0=无限)
    # 0913: 每步转向角。habitat vln_r2r.yaml 实际是 15°/步, 但历史代码全部按 30 写。
    # 默认 30.0 = 保持既有行为(所有已出数字可比); --turn_deg 15 才是真实角度。
    TURN_DEG = 30.0
""", '1')

# --- 2) V1 逃脱计划的 90°/180° 换算 ---
src = sub(
    """            plans = [
                [side] * 3 + ['move_forward'] * 2,
                [('turn_right' if side == 'turn_left' else 'turn_left')] * 3 + ['move_forward'] * 2,
                ['turn_left'] * 6 + ['move_forward'] * 3,
            ]
""",
    """            n90 = int(round(90.0 / self.TURN_DEG))     # 0913: 原硬编码 3
            n180 = int(round(180.0 / self.TURN_DEG))   # 0913: 原硬编码 6
            plans = [
                [side] * n90 + ['move_forward'] * 2,
                [('turn_right' if side == 'turn_left' else 'turn_left')] * n90 + ['move_forward'] * 2,
                ['turn_left'] * n180 + ['move_forward'] * 3,
            ]
""", '2')

# --- 3) 打转检测的三处净转角换算(单行1 + 多行2, 前缀一致) ---
src = sub(
    """sum(30 if a == 'turn_left' else (-30 if a == 'turn_right' else 0)""",
    """sum(self.TURN_DEG if a == 'turn_left' else (-self.TURN_DEG if a == 'turn_right' else 0)""",
    '3', n_expect=3)

# --- 4) V2 的 nt ---
src = sub(
    """        nt = int(round(abs(tgt) / 30.0))     # R2R turn = 30°/步
""",
    """        nt = int(round(abs(tgt) / self.TURN_DEG))   # 0913: 原 /30
""", '4')

# --- 5) V3 的 nt ---
src = sub(
    """        nt = int(round(abs(tgt) / 30.0))
""",
    """        nt = int(round(abs(tgt) / self.TURN_DEG))   # 0913: 原 /30
""", '5')

# --- 6) V3 无进展掉头: 180° ---
src = sub(
    """            turn = ['turn_left'] * 6 if goal_ego[1] >= 0 else ['turn_right'] * 6
""",
    """            _n180 = int(round(180.0 / self.TURN_DEG))   # 0913: 原硬编码 6(=意图180°)
            turn = (['turn_left'] * _n180 if goal_ego[1] >= 0
                    else ['turn_right'] * _n180)
""", '6')

# --- 7) CLI ---
src = sub(
    """    ap.add_argument('--out', default='probe_oracle')
""",
    """    ap.add_argument('--turn_deg', type=float, default=30.0,
                    help='每步转向角。30.0=历史行为(既有数字可比); 15.0=habitat 真实值(0913 修正)')
    ap.add_argument('--out', default='probe_oracle')
""", '7')

# --- 8) 注入 ---
src = sub(
    """    EscapeV3.FWD_V2 = bool(cli.v3_fwd_v2)
""",
    """    EscapeV3.FWD_V2 = bool(cli.v3_fwd_v2)
    Escape.TURN_DEG = float(cli.turn_deg)
""", '8')

# --- 9) 日志带 turn_deg ---
src = sub(
    """              f'unknown={cli.v3_unknown},levelsat={cli.v3_levelsat},'
              f'fwd_v2={cli.v3_fwd_v2})', flush=True)
""",
    """              f'unknown={cli.v3_unknown},levelsat={cli.v3_levelsat},'
              f'fwd_v2={cli.v3_fwd_v2},turn_deg={cli.turn_deg})', flush=True)
""", '9')

open(F, 'w', encoding='utf-8').write(src)
py_compile.compile(F, doraise=True)
print('九处替换全部命中 + 语法检查通过')
print('备份: %s.bak0913td' % F)
