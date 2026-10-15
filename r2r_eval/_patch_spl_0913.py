"""0913 修 probe_oracle_goal.py 的 SPL 全 0 bug。

病根(实测): `m.get('gt_path_length', 0)` —— habitat metrics 里没有这个 key,
恒为 0, 于是 `if rec['success']>0 and gt_len>0` 永不成立, **1431 条成功 ep 全部 spl=0.0**。
次病: path_len = steps*0.25 把转向步也算成 0.25m 位移, 系统性偏大。

修法(零外部依赖, 不碰控制流):
  - gt_len 从 ep.reference_path 逐点欧氏距离求和(精确)
  - path_len 从每步 state.position 的 3D 位移累加(精确)
只新增 spl/path_len/gt_len 三个字段值, 不改 success/os/ne —— SR/OSR/NE 逐位不变。
"""
import py_compile
import shutil
import sys

F = '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/probe_oracle_goal.py'
shutil.copyfile(F, F + '.bak0913pre')
src = open(F, encoding='utf-8').read()


def sub(old, new, tag):
    n = src.count(old)
    assert n == 1, f'[{tag}] 匹配 {n} 次, 期望 1 次 —— 放弃'
    return src.replace(old, new)


# --- A) episode 初始化: 加 gt_len / path_len / prev_pos3 ---
A_OLD = """        pending_act = None
        n_escape = 0
"""
A_NEW = """        pending_act = None
        n_escape = 0
        # 0913修 SPL: GT 路径长(逐点欧氏, 精确)
        gt_len = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
        path_len = 0.0
        prev_pos3 = None
"""
src = sub(A_OLD, A_NEW, 'A')

# --- B) 主循环: 逐帧累加实走路径长 ---
B_OLD = """            position = np.asarray(state.position, np.float64)
"""
B_NEW = """            position = np.asarray(state.position, np.float64)
            if prev_pos3 is not None:
                path_len += float(np.linalg.norm(position - prev_pos3))
            prev_pos3 = position
"""
src = sub(B_OLD, B_NEW, 'B')

# --- C) 收尾: 用真值算 SPL ---
C_OLD = """               'spl': 0.0}
        gt_len = float(m.get('gt_path_length', 0) or 0)
        if rec['success'] > 0 and gt_len > 0:
            path_len = steps * 0.25  # 近似(混合动作, 粗)
            rec['spl'] = min(1.0, gt_len / max(path_len, gt_len))
"""
C_NEW = """               'spl': 0.0, 'path_len': path_len, 'gt_len': gt_len}
        # 0913修: 原 gt_len 取自 m.get('gt_path_length') —— habitat 无此 key → 恒 0 → SPL 全 0;
        # 且 path_len=steps*0.25 把转向步也算成位移 → 偏大。现两者都取自真实量, 只影响本字段。
        if rec['success'] > 0 and gt_len > 0:
            rec['spl'] = min(1.0, gt_len / max(path_len, gt_len))
"""
src = sub(C_OLD, C_NEW, 'C')

open(F, 'w', encoding='utf-8').write(src)
py_compile.compile(F, doraise=True)
print('三处替换全部命中 + 语法检查通过')
print('备份: %s.bak0913pre' % F)
sys.exit(0)
