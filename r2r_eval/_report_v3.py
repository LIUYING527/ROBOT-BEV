"""0912 全量 v3 结果汇总: 与 v2 全量严格同口径配对对比。
用法(22, 系统python3即可): python3 _report_v3.py
"""
import json, glob, gzip, os

R2R = '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval'
D = '/data/DongBaorong/BEVLOOK/benchmarks/mp3d_r2r/r2r/val_unseen/val_unseen.json.gz'
d = json.load(gzip.open(D))
by_id = {str(e['episode_id']): e for e in d['episodes']}


def yspan(e):
    ys = [p[1] for p in e['reference_path']]
    return max(ys) - min(ys)


def load(pat):
    rs = []
    for f in sorted(glob.glob(R2R + '/' + pat)):
        rs += [json.loads(l) for l in open(f)]
    return rs


v3 = load('full_v3_s*.jsonl')
v2 = load('full_goalmlp_ep5_escape2_s*.jsonl')

if not v3:
    print('v3 结果还没出来'); raise SystemExit

for r in v3 + v2:
    e = by_id.get(r['episode_id'])
    if e:
        r['yspan'] = yspan(e)
        r['scene'] = e['scene_id'].split('/')[-1].replace('.glb', '')


def stat(rs, label):
    if not rs:
        return
    n = len(rs); s = sum(r['success'] for r in rs); o = sum(r['os'] for r in rs)
    ne = sum(r['ne'] for r in rs) / n
    print('  %-14s n=%4d  SR=%5.1f%%  OSR=%5.1f%%  NE=%.2f' % (label, n, 100*s/n, 100*o/n, ne))


print('=' * 72)
print('全量 v3 vs v2 (val_unseen 全量 @22, 同 ckpt goalmlp_ep5, 同 hfov79, 非dense)')
print('=' * 72)
stat(v3, 'v3 (planner)')
stat(v2, 'v2 (基线)')

print('\n--- 按 yspan 拆分 (多楼层 = 架构限制) ---')
for lo, hi, name in [(0, 1.0, '单层 yspan<1m'), (1.0, 1e9, '多楼层 yspan>=1m')]:
    stat([r for r in v3 if lo <= r.get('yspan', 0) < hi], 'v3 ' + name)
    stat([r for r in v2 if lo <= r.get('yspan', 0) < hi], 'v2 ' + name)

print('\n--- 严格配对 (同 ep, 只v2失败→v3成功的"救回") ---')
v2d = {r['episode_id']: r for r in v2}
common = [r for r in v3 if r['episode_id'] in v2d]
resc = [r for r in common if r['success'] > 0 and v2d[r['episode_id']]['success'] == 0]
broke = [r for r in common if r['success'] == 0 and v2d[r['episode_id']]['success'] > 0]
n2 = sum(v2d[r['episode_id']]['success'] for r in common)
n3 = sum(r['success'] for r in common)
print('  配对 %d 条: v2 %d 成功 (%.1f%%)  →  v3 %d 成功 (%.1f%%)   净 %+d 条 / %+.1f pt' % (
    len(common), n2, 100*n2/len(common), n3, 100*n3/len(common),
    n3-n2, 100*(n3-n2)/len(common)))
print('  救回 %d 条; 变差 %d 条' % (len(resc), len(broke)))
if broke:
    print('  变差清单: %s' % ' '.join(r['episode_id'] for r in broke[:20]))

print('\n--- 分场景 (v3) ---')
scenes = {}
for r in v3:
    scenes.setdefault(r.get('scene', '?'), []).append(r)
for s, v in sorted(scenes.items(), key=lambda x: -len(x[1])):
    n = len(v); k = sum(r['success'] for r in v)
    v2v = [r for r in v2 if r.get('scene') == s]
    k2 = sum(r['success'] for r in v2v)
    print('  %-16s v3 %3d/%3d = %5.1f%%   (v2 %3d/%3d = %5.1f%%)' % (
        s, k, n, 100*k/n, k2, len(v2v), 100*k2/max(1, len(v2v))))
