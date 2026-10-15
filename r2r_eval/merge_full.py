import json
import sys

tag = sys.argv[1]
n = int(sys.argv[2])
rows = []
miss = []
for i in range(n):
    f = "full_%s_s%d.jsonl" % (tag, i)
    try:
        for l in open(f, errors='ignore'):
            rows.append(json.loads(l))
    except FileNotFoundError:
        miss.append(f)

print("分片缺失:", miss if miss else "无")
nn = len(rows)
if not nn:
    print("无数据")
    sys.exit(0)
sr = sum(r["success"] for r in rows) / nn * 100
osr = sum(r["os"] for r in rows) / nn * 100
ne = sum(r["ne"] for r in rows) / nn
stuck = sum(1 for r in rows if r["steps"] >= 500)
ids = set(r["episode_id"] for r in rows)
print("==== FULL ORACLE %s: n=%d (唯一ep=%d) SR=%.1f OSR=%.1f NE=%.2f 卡死=%d(%.0f%%) ===="
      % (tag, nn, len(ids), sr, osr, ne, stuck, stuck / nn * 100))
