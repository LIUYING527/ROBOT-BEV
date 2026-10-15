import json, glob
def load(tag):
    rows=[]
    for f in sorted(glob.glob(f"/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/full_{tag}_s*.jsonl")):
        for line in open(f):
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows

for tag in ["goalmlp_ep5","r2r_dense"]:
    r=load(tag)
    n=len(r)
    succ=[x for x in r if x.get("success")]
    stuck=[x for x in r if not x.get("success") and x["steps"]>=500]
    early=[x for x in r if not x.get("success") and x["steps"]<500]
    osre=[x for x in r if x.get("os")]
    print(f"===== {tag}: n={n} SR={100*len(succ)/n:.1f} OSR={100*len(osre)/n:.1f}")
    print(f"  成功={len(succ)}  卡死(500步)={len(stuck)}  提前停但失败={len(early)}")
    # 失败者里, 到过目标附近(os=1)的有多少
    fail=[x for x in r if not x.get("success")]
    fail_os=[x for x in fail if x.get("os")]
    print(f"  失败={len(fail)}, 其中曾到过目标(os=1)={len(fail_os)} ({100*len(fail_os)/len(fail):.1f}%) → 差在终止判定")
    print(f"             从未到达(os=0)={len(fail)-len(fail_os)} ({100*(len(fail)-len(fail_os))/len(fail):.1f}%) → 真导航失败")
    # 失败者的 min_dtg 分桶(离目标最近到过多远)
    buckets=[(0,1.5),(1.5,3),(3,5),(5,10),(10,999)]
    print("  失败者 min_dtg 分布(最近距离):")
    for lo,hi in buckets:
        c=sum(1 for x in fail if lo<=x["min_dtg"]<hi)
        print(f"    [{lo:>5},{hi:>5})m : {c:>4} ({100*c/len(fail):.1f}%)")
    print(f"  成功者步数中位={sorted(x['steps'] for x in succ)[len(succ)//2]}")
    print()
