"""0913 全天结果汇总 —— 扫描 r2r_eval 下所有结果 jsonl, 输出 SUMMARY_0913.md。

可随时手动跑(与 chain4 收尾调用同一份), 幂等:
    python3 _summarize_0913.py
⚠️ 一律从 jsonl 重算, 不信日志里那行 ORACLE-PROBE —— 0913 出过 sr() 把 OSR 当 SR 解析的事故。
"""
import glob
import gzip
import json
import os
import re
import time

R = '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval'
D = '/data/DongBaorong/BEVLOOK/benchmarks/mp3d_r2r/r2r/val_unseen/val_unseen.json.gz'
OUT = os.path.join(R, 'SUMMARY_0913.md')
by = {str(e['episode_id']): e for e in json.load(gzip.open(D))['episodes']}


def yspan(e):
    v = [p[1] for p in e['reference_path']]
    return max(v) - min(v)


def load(pat):
    """返回 {ep_id: rec}; pat 里可含 *, 会合并所有匹配文件。"""
    rs = []
    # GROUPS 里的模式一律写成 '/xxx_s*.jsonl' 形式; 必须 lstrip 掉前导斜杠,
    # 否则 os.path.join 会把它当绝对路径直接吞掉 R —— 0914 之前汇总表一直全空就是这个原因。
    for f in sorted(glob.glob(os.path.join(R, pat.lstrip('/')))):
        try:
            rs += [json.loads(l) for l in open(f)]
        except Exception:
            continue
    for r in rs:
        e = by.get(r['episode_id'])
        if e:
            r['ys'] = yspan(e)
    return {r['episode_id']: r for r in rs}


def stat(dd):
    rs = list(dd.values())
    n = len(rs)
    if not n:
        return None
    suc = sum(1 for r in rs if r['success'] > 0)
    spl = (sum(r.get('spl', 0) for r in rs) / suc) if suc else 0.0
    return dict(sr=100 * suc / n, osr=100 * sum(r['os'] for r in rs) / n,
                ne=sum(r['ne'] for r in rs) / n, spl=spl, n=n)


# (glob, 显示名, 说明) —— 顺序即输出顺序
GROUPS = [
    ('/np2full_s*.jsonl',            'v3 + noprog2 (最佳)',        ''),
    ('/recheck_v3_s*.jsonl',         'v3 默认 noprog4',             ''),
    ('/full_goalmlp_ep5_escape2_s*.jsonl', '旧 v2 脱困 (0911 基线)', ''),
    ('/r2rft_v3_s*.jsonl',           'r2rft ckpt + v3',             ''),
    ('/ep9_v3_s*.jsonl',             'ep9 ckpt + v3',               ''),
    ('/dense_v3_s*.jsonl',           'dense ckpt + v3 (noprog4)',   ''),
    ('/abl_base.jsonl',              '消融基准 (noprog2, 300子集)', ''),
    ('/abl_off1a.jsonl',             '消融 −①未知当最空',           ''),
    ('/abl_off1b.jsonl',             '消融 −①不筛可通行',           ''),
    ('/abl_off2.jsonl',              '消融 −②level取模',            ''),
    ('/abl_off3.jsonl',              '消融 −③无进展重定位',         ''),
    ('/abl_off4.jsonl',              '消融 −④步长按净空',           ''),
    ('/abl_td_alt.jsonl',            '消融 反侧转向角',             ''),
    ('/td30_fail.jsonl',             'A/B turn_deg30 @100失败集',   ''),
    ('/td15_fail.jsonl',             'A/B turn_deg15 @100失败集',   ''),
    ('/vs300_base_v2.jsonl',         'val_seen300 旧v2基线',        ''),
    ('/vs300_main.jsonl',            'val_seen300 主配置',          ''),
    ('/main_spl_s*.jsonl',           '主配置全量(补SLP)',           ''),
    ('/dense_np2_s*.jsonl',          'dense + noprog2 全量',        ''),
    ('/full_abl_off*_s*.jsonl',      '消融落全量',                  ''),
    ('/rep_np2_r*_s*.jsonl',         '主配置复现(多轮)',            ''),
    ('/ep9_np2_r*_s*.jsonl',         'ep9 + noprog2 全量',          ''),
    ('/vsfull_s*.jsonl',             'val_seen 全量 主配置',        ''),
    ('/vs_base_v2_r*_s*.jsonl',      'val_seen 全量 旧v2基线',      ''),
]

lines = ['# R2R 快脑 0913 汇总(自动生成, %s)' % time.strftime('%F %H:%M'),
         '', '> 全部数字直接从结果 jsonl 重算, 未使用日志里的 ORACLE-PROBE 行。', '']
lines.append('| 配置 | n | SR | OSR | NE | SPL |')
lines.append('|---|---|---|---|---|---|')
summary = {}
for pat, name, _ in GROUPS:
    dd = load(pat)
    st = stat(dd)
    if not st:
        lines.append('| %s | — | 待跑/无数据 | | | |' % name)
        continue
    summary[name] = st
    lines.append('| %s | %d | **%.1f%%** | %.1f | %.2f | %.3f |' % (
        name, st['n'], st['sr'], st['osr'], st['ne'], st['spl']))

# 单层/多楼层拆分(最佳配置)
main = load('/np2full_s*.jsonl')
if main:
    lines += ['', '## 最佳配置按楼层拆分', '',
              '| 子集 | n | SR | OSR | NE |', '|---|---|---|---|---|']
    for nm, lo, hi in [('单层 yspan<1m', 0, 1.0), ('多楼层 yspan>=1m', 1.0, 1e9)]:
        sub = {k: v for k, v in main.items() if lo <= v.get('ys', 0) < hi}
        st = stat(sub)
        if st:
            lines.append('| %s | %d | **%.1f%%** | %.1f | %.2f |' % (
                nm, st['n'], st['sr'], st['osr'], st['ne']))

# 消融增量
base = load('/abl_base.jsonl')
if base:
    lines += ['', '## 消融: 每关掉一处修复掉多少(300 子集, 配对)', '',
              '| 关掉的修复 | 掉多少条 | 说明 |', '|---|---|---|']
    for tag, lb in [('abl_off1a', '①未知当最空'), ('abl_off1b', '①不筛可通行'),
                    ('abl_off2', '②level 取模循环'), ('abl_off3', '③无进展重定位'),
                    ('abl_off4', '④步长按净空')]:
        d = load('/%s.jsonl' % tag)
        com = [k for k in base if k in d]
        if not com:
            lines.append('| %s | — | 待跑 |' % lb)
            continue
        drop = sum(base[k]['success'] for k in com) - sum(d[k]['success'] for k in com)
        lines.append('| %s | %+d | n=%d |' % (lb, -drop, len(com)))

if not summary:
    # 0914: GROUPS 的 glob 全带前导 '/' 导致过 os.path.join 吞掉 R, 24 组全空还照写不误,
    # 再被调用点的 `|| true` 吞掉 —— 一整夜的汇总表都是空的而没人发现。宁可炸, 不要静默。
    raise SystemExit('❌ 所有分组都无数据: 先查 GROUPS 的 glob 模式与 R 是否拼错, 不要静默写空表')

open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('已写 %s (%d 行)' % (OUT, len(lines)))
print('\n'.join(lines[:8]))
