import json
import numpy as np
rows=[json.loads(l) for l in open('realvlm_smoke3b.jsonl')]
loc=[s for r in rows for s in r.get('loc_trace',[])]
print("总定位调用 %d" % len(loc))
ok=[s for s in loc if s[4]>0]
print("成功返回定位 %d (%.1f%%)" % (len(ok), len(ok)/max(len(loc),1)*100))
none_=[s for s in loc if s[4]==0]
from collections import Counter
print("未定位原因:", Counter(s[5] for s in none_))
if ok:
    d=np.array([s[3] for s in ok])
    print("定位点到参考路径距离: 中位=%.2f 均=%.2f p75=%.2f p90=%.2f max=%.2f" %
          (np.median(d), d.mean(), np.percentile(d,75), np.percentile(d,90), d.max()))
    for th in (0.5,1.0,2.0,3.0,5.0):
        print("   < %.1f m: %5.1f%%" % (th, (d<th).mean()*100))
