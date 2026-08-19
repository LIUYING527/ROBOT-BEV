#!/usr/bin/env python3
"""
时序慢脑离线评测: 走廊 nav_ds_real 上对比三种检测模式
  single      = 单帧基线(0721评测同款, 逐帧精确率34%的痛点)
  window_any  = 滑窗K帧"ANY帧可见" (时序上下文)
  window_last = 滑窗K帧"只看末帧" (上下文+末帧判决)
GT: kinds=='strafe' => 泵在近旁(正样本); 'cruise' => 巡航(负样本, 含远处可见泵的难负样本)
输出: outputs/vlm_temporal_eval/temporal_results.jsonl + temporal_summary.md

用法: python scripts/eval_temporal_slowbrain.py [--every 6] [--modes single,window_any,window_last]
"""
import os, sys, json, argparse, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_temporal_slowbrain import TemporalSlowBrain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "outputs/nav_ds_real/meta.npz")
OUT = os.path.join(ROOT, "outputs/vlm_temporal_eval")


def metrics(rows):
    rows = [r for r in rows if r["ok"]]
    tp = sum(1 for r in rows if r["gt_yes"] and r["pred_yes"])
    fp = sum(1 for r in rows if (not r["gt_yes"]) and r["pred_yes"])
    fn = sum(1 for r in rows if r["gt_yes"] and (not r["pred_yes"]))
    tn = sum(1 for r in rows if (not r["gt_yes"]) and (not r["pred_yes"]))
    n = max(1, len(rows))
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 3),
            "err": sum(1 for r in rows if not r["ok"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--modes", default="single,window_any,window_last")
    ap.add_argument("--win", type=int, default=4)
    args = ap.parse_args()
    modes = args.modes.split(",")

    os.makedirs(OUT, exist_ok=True)
    z = np.load(DS, allow_pickle=True)
    fpv = z["fpv"]; poses = z["poses"]
    kinds = np.array([str(k) for k in z["kinds"]])
    idxs = list(range(0, len(fpv), args.every))
    print(f"[eval] 帧总数{len(fpv)} 评测{len(idxs)}帧 × {len(modes)}模式 = {len(idxs)*len(modes)}次查询", flush=True)

    brains = {m: TemporalSlowBrain(win=args.win, push_every=1) for m in modes}
    jl = open(os.path.join(OUT, "temporal_results.jsonl"), "w")
    t0 = time.time()
    for c, i in enumerate(idxs):
        img = cv2.imread(str(fpv[i]))
        if img is None:
            continue
        row = {"i": int(i), "x": float(poses[i, 0]), "kind": kinds[i],
               "gt_yes": bool(kinds[i] == "strafe")}
        for m in modes:
            sb = brains[m]
            # 滑窗: 推当前帧及之前win-1帧(从数据集按帧序取, 保持时序)
            sb.buf.clear()
            for j in range(max(0, i - (args.win - 1) * args.every), i + 1, args.every):
                im2 = cv2.imread(str(fpv[j]))
                if im2 is not None:
                    sb.push(im2, pose=poses[j], force=True)
            yes, ok, lat = sb.detect(mode=m)
            row[f"{m}_pred"] = bool(yes); row[f"{m}_ok"] = bool(ok); row[f"{m}_lat"] = round(lat, 2)
        jl.write(json.dumps(row, ensure_ascii=False) + "\n"); jl.flush()
        if c % 20 == 0:
            dt = time.time() - t0
            print(f"  {c}/{len(idxs)} i={i} x={poses[i,0]:5.1f} {kinds[i]:6s} "
                  + " ".join(f"{m}={'Y' if row[f'{m}_pred'] else 'n'}" for m in modes)
                  + f" ({dt:.0f}s)", flush=True)
    jl.close()

    # 汇总
    rows = [json.loads(l) for l in open(os.path.join(OUT, "temporal_results.jsonl"))]
    lines = ["# 时序慢脑离线评测 (走廊 nav_ds_real)", ""]
    lines.append(f"- 帧: 每{args.every}帧采样, n={len(rows)}; 滑窗K={args.win}")
    lines.append(f"- GT: strafe=泵近旁(正), cruise=巡航(负)")
    lines.append("")
    lines.append("| 模式 | 精确率 | 召回 | F1 | FP | FN | 无效 | 平均延迟 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in modes:
        sub = [{"gt_yes": r["gt_yes"], "pred_yes": r[f"{m}_pred"], "ok": r[f"{m}_ok"]} for r in rows]
        mt = metrics(sub)
        lat = [r[f"{m}_lat"] for r in rows if r[f"{m}_ok"]]
        avg_lat = sum(lat) / max(1, len(lat))
        lines.append(f"| {m} | {mt['precision']:.3f} | {mt['recall']:.3f} | {mt['f1']:.3f} "
                     f"| {mt['fp']} | {mt['fn']} | {mt['err']} | {avg_lat:.2f}s |")
    open(os.path.join(OUT, "temporal_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
