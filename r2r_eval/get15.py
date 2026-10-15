import json, glob
def load(pat):
    d={}
    for f in sorted(glob.glob(pat)):
        for line in open(f):
            line=line.strip()
            if line:
                r=json.loads(line); d[str(r['episode_id'])]=r
    return d
a=load('full_goalmlp_ep5_s*.jsonl'); b=load('full_goalmlp_ep5_escape_s*.jsonl')
common=set(a)&set(b)
lost=[k for k in common if a[k]['success'] and not b[k]['success']]
lost.sort(key=lambda k:int(k))
print('退步共 %d 条:' % len(lost))
print(' '.join(lost))
