import json, glob
def load(pat):
    d={}
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            line=line.strip()
            if line:
                r=json.loads(line); d[str(r['episode_id'])]=r
    return d
bare=load('full_goalmlp_ep5_s*.jsonl')
v1=load('full_goalmlp_ep5_escape_s*.jsonl')
v3=load('probe_v3_lost.jsonl')
lost=[k for k in bare if bare[k]['success'] and not v1[k]['success']]
lost.sort(key=lambda k:int(k))
print('%-7s | %-28s | %-30s | %s' % ('ep','裸(成功)','v1脱困(失败)','v3脱困'))
print('-'*100)
ok3=0
for k in lost:
    b,v1r,v3r=bare[k],v1[k],v3.get(k)
    if v3r and v3r['success']: ok3+=1
    print('%-7s | steps=%3d min_dtg=%.2f      | steps=%3d esc=%3s min_dtg=%.2f | %s' % (
        k, b['steps'], b['min_dtg'], v1r['steps'], v1r.get('n_escape','?'), v1r['min_dtg'],
        ('steps=%3d esc=%3s min_dtg=%.2f %s' % (v3r['steps'], v3r.get('n_escape','?'), v3r['min_dtg'], 'OK' if v3r['success'] else '仍失败')) if v3r else 'N/A'))
print()
print('v3 救回: %d/15' % ok3)
