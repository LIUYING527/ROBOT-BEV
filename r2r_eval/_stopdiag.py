import json
import numpy as np
rows = [json.loads(l) for l in open('stopdiag_50.jsonl')]
near = [s for r in rows for s in r.get('stop_trace', []) if 0 <= s[2] < 1.5]
mid  = [s for r in rows for s in r.get('stop_trace', []) if 1.5 <= s[2] < 3.0]
far  = [s for r in rows for s in r.get('stop_trace', []) if s[2] >= 3.0]
print(f"总步数(traced)={sum(len(r.get('stop_trace',[])) for r in rows)}")
for tag, g in (("到终点<1.5m", near), ("1.5~3.0m", mid), (">=3.0m", far)):
    if not g:
        print(f"{tag}: 无样本"); continue
    p = np.array([s[3] for s in g])
    print(f"{tag:12s} n={len(p):5d} 中位={np.median(p):.3f} 均={p.mean():.3f} "
          f"p75={np.percentile(p,75):.3f} p90={np.percentile(p,90):.3f} max={p.max():.3f}")
print()
if near:
    p = np.array([s[3] for s in near])
    print("若按阈值(当前 0.40, 需连续2票):")
    for th in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        print(f"   阈值 {th:.2f}: 该组 {(p>=th).mean()*100:5.1f}% 的步达标")
# 逐 ep 看"到过 1.5m 内"的 ep，其近距步里最大连续达标长度
print()
hit_eps = [r for r in rows if any(0 <= s[2] < 1.5 for s in r.get('stop_trace', []))]
print(f"到过终点1.5m内的 ep: {len(hit_eps)}/{len(rows)}, 其中 success={sum(r['success'] for r in hit_eps):.0f}")
