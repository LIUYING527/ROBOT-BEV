"""0913 给 probe_oracle_goal.py 加消融开关 + --split, 为下面两件事铺路:

  (1) EscapeV3 四修复的消融(论文必答). 现有 CLI 已能关掉其中三条, 缺 ②④:
        --v3_unknown 8.0  关①(未知当最空, 退回 v2 行为)
        --v3_minclear 0.0 关①的"先筛可通行"部分
        --v3_noprog 9999  关③(永不升级)
        --v3_levelsat     关②(NEW: level 退回 min(level,2) 饱和)
        --v3_fwd_v2       关④(NEW: fwd_n 退回 v2 的 3+min(level,2) 步)
  (2) --split 跑 val_seen held-out(778 条). 原先 make_env('val_unseen',...) 是硬编码。

⚠️ 两个新开关默认 False = 逐位保持现有行为, 既有全部数字不受影响。
"""
import py_compile
import shutil

F = '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/probe_oracle_goal.py'
shutil.copyfile(F, F + '.bak0913abl')
src = open(F, encoding='utf-8').read()


def sub(old, new, tag):
    n = src.count(old)
    assert n == 1, f'[{tag}] 匹配 {n} 次, 期望 1 次 —— 放弃'
    return src.replace(old, new)


# --- 1) EscapeV3 加 LEVEL_SAT / FWD_V2 类属性 ---
src = sub(
    """    LEVELS = (0.0, 60.0, -60.0, 120.0, -120.0, 180.0)
""",
    """    LEVELS = (0.0, 60.0, -60.0, 120.0, -120.0, 180.0)
    LEVEL_SAT = False    # True = 消融②: level 退回 v2 的 min(level,2) 饱和
    FWD_V2 = False       # True = 消融④: fwd_n 退回 v2 的 3+min(level,2) 步
""", '1')

# --- 2) ② level 取模 vs 饱和 ---
src = sub(
    """        base = self.LEVELS[self.level % len(self.LEVELS)]   # ② 档位循环而非饱和
""",
    """        # ② 档位循环而非饱和 (--v3_levelsat 时退回 v2 饱和, 仅供消融)
        base = (self.LEVELS[min(self.level, len(self.LEVELS) - 1)] if self.LEVEL_SAT
                else self.LEVELS[self.level % len(self.LEVELS)])
""", '2')

# --- 3) ④ fwd_n 按净空 vs v2 固定 3~5 步 ---
src = sub(
    """        fwd_n = int(max(3, min(8, clear_sel / 0.25 * 0.6)))
""",
    """        # ④ 走多远按该方向净空定(保守 60%), 3~8 步 (--v3_fwd_v2 时退回 v2 口径, 仅供消融)
        fwd_n = (3 + min(self.level, 2) if self.FWD_V2
                 else int(max(3, min(8, clear_sel / 0.25 * 0.6))))
""", '3')

# --- 4) 新增 CLI: --split / --v3_levelsat / --v3_fwd_v2 ---
src = sub(
    """    ap.add_argument('--out', default='probe_oracle')
""",
    """    ap.add_argument('--split', default='val_unseen',
                    help='habitat split: val_unseen / val_seen (held-out 验证用)')
    ap.add_argument('--v3_levelsat', action='store_true',
                    help='消融②: 关闭 level 取模循环, 退回 v2 的 min(level,2) 饱和')
    ap.add_argument('--v3_fwd_v2', action='store_true',
                    help='消融④: fwd_n 退回 v2 的 3+min(level,2) 步')
    ap.add_argument('--out', default='probe_oracle')
""", '4')

# --- 5) 参数注入 ---
src = sub(
    """    EscapeV3.UNKNOWN_CLEAR = cli.v3_unknown
""",
    """    EscapeV3.UNKNOWN_CLEAR = cli.v3_unknown
    EscapeV3.LEVEL_SAT = bool(cli.v3_levelsat)
    EscapeV3.FWD_V2 = bool(cli.v3_fwd_v2)
""", '5')

# --- 6) make_env 用 --split ---
src = sub(
    """    env = make_env('val_unseen', 640, 480, 79)
""",
    """    env = make_env(cli.split, 640, 480, 79)
    print(f'[split] {cli.split}: {len(list(env.episodes))} episodes', flush=True)
""", '6')

# --- 7) escape 打印带上消融开关, 便于事后从日志还原配置 ---
src = sub(
    """              f'unknown={cli.v3_unknown})', flush=True)
""",
    """              f'unknown={cli.v3_unknown},levelsat={cli.v3_levelsat},'
              f'fwd_v2={cli.v3_fwd_v2})', flush=True)
""", '7')

open(F, 'w', encoding='utf-8').write(src)
py_compile.compile(F, doraise=True)
print('七处替换全部命中 + 语法检查通过')
print('备份: %s.bak0913abl' % F)
