"""0913 修打转判据的 300° 硬编码 —— 否则 --turn_deg 15 会让它静默失效。

打转判据: net = sum(±TURN_DEG over 最近20步), 触发条件 abs(net) > 300。
  20 步的数学上限 = 20 * TURN_DEG。
    TURN_DEG=30 → 上限 600, 阈值 300 可达(需 ≥11 步同向) → 能触发
    TURN_DEG=15 → 上限 300, 而条件是**严格大于** → 永久不可达, 探测器死掉
  ⇒ 直接改 TURN_DEG 会把"只改转向角"变成"改转向角 + 关掉打转检测", A/B 结果不可解释。

修法: 阈值改写成 "10 次同向转向" 的角度, threshold = SPIN_N_TURNS * TURN_DEG。
  10 * 30 = 300 → 默认行为**逐位不变**;
  10 * 15 = 150 → TURN_DEG=15 时语义保持(同样是"10 次同向转向"), A/B 才是干净的。
"""
import py_compile
import shutil

F = '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/probe_oracle_goal.py'
shutil.copyfile(F, F + '.bak0913spin')
src = open(F, encoding='utf-8').read()


def sub(old, new, tag, n_expect=1):
    n = src.count(old)
    assert n == n_expect, f'[{tag}] 匹配 {n} 次, 期望 {n_expect} 次 —— 放弃'
    return src.replace(old, new)


# --- 1) 基类加 SPIN_N_TURNS ---
src = sub(
    """    TURN_DEG = 30.0
""",
    """    TURN_DEG = 30.0
    # 打转判据: 最近20步净转角 > SPIN_N_TURNS 次同向转向。原为硬编码 300°(=10*30),
    # 改写成随 TURN_DEG 缩放 —— 否则 --turn_deg 15 时 20 步上限恰好 300, 严格大于永不成立。
    SPIN_N_TURNS = 10
""", '1')

# --- 2) 三处阈值 ---
src = sub(
    """abs(net) > 300""",
    """abs(net) > self.SPIN_N_TURNS * self.TURN_DEG""",
    '2', n_expect=3)

open(F, 'w', encoding='utf-8').write(src)
py_compile.compile(F, doraise=True)
print('两处替换全部命中 + 语法检查通过 (阈值 3 处)')
print('默认 TURN_DEG=30 时: 10*30 = 300 → 与原硬编码逐位一致')
print('备份: %s.bak0913spin' % F)
