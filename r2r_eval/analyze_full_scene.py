#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量 oracle 结果的逐场景归因分析(0910)。
用法: python analyze_full_scene.py <jsonl前缀> [jsonl前缀2 ...]
例:  python analyze_full_scene.py full_goalmlp_ep5 full_dense
会自动拼接 <前缀>_s0..s7.jsonl。
输出: 总体 SR / 逐场景 SR / 失败画像(min_dtg 分档)。
"""
import json
import gzip
import sys
import glob
from collections import Counter, defaultdict

DATA = '/data/DongBaorong/BEVLOOK/benchmarks/mp3d_r2r/r2r/val_unseen/val_unseen.json.gz'
meta = {}
for e in json.load(gzip.open(DATA))['episodes']:
    meta[str(e['episode_id'])] = e


def scene_of(eid):
    m = meta.get(str(eid))
    return m['scene_id'].split('/')[-2] if m else '?'


def gd_of(eid):
    m = meta.get(str(eid))
    return (m.get('info') or {}).get('geodesic_distance') if m else None


def load(prefix):
    rows = []
    files = sorted(glob.glob(prefix + '_s*.jsonl')) or [prefix + '.jsonl']
    for f in files:
        try:
            for l in open(f, errors='ignore'):
                l = l.strip()
                if l:
                    rows.append(json.loads(l))
        except FileNotFoundError:
            pass
    return rows


def report(prefix):
    rows = load(prefix)
    n = len(rows)
    if not n:
        print("== %s: 无数据 ==" % prefix)
        return
    sr = sum(r['success'] for r in rows) / n * 100
    osr = sum(r['os'] for r in rows) / n * 100
    ne = sum(r['ne'] for r in rows) / n
    stuck = sum(1 for r in rows if r['steps'] >= 500)
    print("=" * 66)
    print("== %s: n=%d  SR=%.1f  OSR=%.1f  NE=%.2f  卡死=%d(%.0f%%)" % (
        prefix, n, sr, osr, ne, stuck, stuck / n * 100))
    print("=" * 66)

    # 逐场景
    by = defaultdict(lambda: {'n': 0, 'ok': 0, 'stuck': 0})
    for r in rows:
        sc = scene_of(r['episode_id'])
        by[sc]['n'] += 1
        by[sc]['ok'] += r['success']
        by[sc]['stuck'] += 1 if r['steps'] >= 500 else 0
    print("\n%-16s %5s %7s %8s" % ("场景", "条数", "SR%", "卡死%"))
    print("-" * 42)
    for sc, v in sorted(by.items(), key=lambda x: x[1]['ok'] / max(x[1]['n'], 1)):
        print("%-16s %5d %7.1f %8.0f" % (sc, v['n'], v['ok'] / v['n'] * 100,
                                         v['stuck'] / v['n'] * 100))

    # 失败画像
    fails = [r for r in rows if not r['success']]
    if fails:
        print("\n失败画像 (n=%d):" % len(fails))
        bands = [('<3m 差一点', 0, 3), ('3~6m 中途卡住', 3, 6), ('>=6m 完全走不到', 6, 1e9)]
        for name, lo, hi in bands:
            c = sum(1 for r in fails if lo <= r['min_dtg'] < hi)
            print("   %-18s %4d (%.0f%%)" % (name, c, c / len(fails) * 100))

    # 目标距离 vs 成败
    ok_d = [gd_of(r['episode_id']) for r in rows if r['success']]
    bad_d = [gd_of(r['episode_id']) for r in fails]
    ok_d = [d for d in ok_d if d]
    bad_d = [d for d in bad_d if d]
    if ok_d and bad_d:
        ok_d.sort(); bad_d.sort()
        print("\n目标距离: 成功组中位%.1fm  失败组中位%.1fm" % (
            ok_d[len(ok_d) // 2], bad_d[len(bad_d) // 2]))
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        report(p)
