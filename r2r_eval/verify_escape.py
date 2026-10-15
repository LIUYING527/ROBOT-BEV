import json, glob
def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                d[str(r['episode_id'])] = r
    return d

a = load('full_goalmlp_ep5_s*.jsonl')          # 裸(8片)
b = load('full_goalmlp_ep5_escape_s*.jsonl')   # 脱困(5片)
print('裸 n=%d  脱困 n=%d' % (len(a), len(b)))
common = set(a) & set(b)
print('共同 ep: %d   仅裸有: %d   仅脱困有: %d' % (len(common), len(set(a)-set(b)), len(set(b)-set(a))))
print()
both = [k for k in common if a[k]['success'] and b[k]['success']]
none = [k for k in common if not a[k]['success'] and not b[k]['success']]
a_only = [k for k in common if a[k]['success'] and not b[k]['success']]
b_only = [k for k in common if not a[k]['success'] and b[k]['success']]
print('  都成功      : %4d' % len(both))
print('  都失败      : %4d' % len(none))
print('  裸胜/脱困败 : %4d   <- 退步' % len(a_only))
print('  脱困胜/裸败 : %4d   <- 救回' % len(b_only))
print('  净变化      : %+d' % (len(b_only) - len(a_only)))
print()
print('共同集上 裸 SR=%.1f%%  脱困 SR=%.1f%%' % (
    100*sum(a[k]['success'] for k in common)/len(common),
    100*sum(b[k]['success'] for k in common)/len(common)))
print()
# 退步的到底怎么了
print('退步样例(最多5条):')
for k in a_only[:5]:
    print('  ep%-6s 裸 steps=%3d min_dtg=%.2f | 脱困 steps=%3d esc=%s min_dtg=%.2f' % (
        k, a[k]['steps'], a[k]['min_dtg'], b[k]['steps'], b[k].get('n_escape','?'), b[k]['min_dtg']))
print()
esc = [b[k].get('n_escape',0) for k in common]
import statistics
print('脱困触发次数: 中位=%d  均值=%.1f  最大=%d' % (
    statistics.median(esc), statistics.mean(esc), max(esc)))
