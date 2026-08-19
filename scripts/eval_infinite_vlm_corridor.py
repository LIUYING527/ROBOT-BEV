#!/usr/bin/env python3
"""
Infinite VLM(自研, 10.0.0.18:8152) 在走廊蓝水泵巡检真实帧上的目标检测评测。

- 数据: outputs/nav_ds_real/meta.npz (1314真实帧, kinds=cruise/strafe, poses)
- GT:   kinds=='strafe' => 泵在旁边(正样本); 'cruise' => 前方走廊无泵(负样本)
        (kinds 来自真实运动分类, 见记忆 where-how-realmotion-drive-0711)
- 模型: 自研 Infinite VLM, 问题 = "Is there a blue water pump right beside you,
        close and large in view? Answer only yes or no."
- 输出: outputs/infinite_vlm_eval/corridor_results.jsonl (逐帧)
        outputs/infinite_vlm_eval/corridor_summary.json  (汇总指标)

用法: python scripts/eval_infinite_vlm_corridor.py [--every N] [--url URL]
"""
import os, sys, json, argparse, time
import numpy as np
import cv2
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "outputs/nav_ds_real/meta.npz")
OUT = os.path.join(ROOT, "outputs/infinite_vlm_eval")
TGT = "a blue water pump"
Q = f"Is there {TGT} right beside you, close and large in view? Answer only yes or no."


def parse_yes(ans):
    a = (ans or "").strip().lower()
    return a.startswith("yes") or a == "y" or "yes" in a[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1, help="每N帧取一帧(1=全跑)")
    ap.add_argument("--url", default=os.environ.get("VLM_URL", "http://10.0.0.18:8152/api/generate"))
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    z = np.load(DS, allow_pickle=True)
    fpv = z["fpv"]; poses = z["poses"]; kinds = np.array([str(k) for k in z["kinds"]])
    idxs = list(range(0, len(fpv), args.every))
    print(f"[eval] 帧总数 {len(fpv)}, 本次评测 {len(idxs)} 帧, VLM={args.url}", flush=True)

    jl = open(os.path.join(OUT, "corridor_results.jsonl"), "w")
    n_ok = 0; t0 = time.time()
    for c, i in enumerate(idxs):
        path = str(fpv[i])
        img = cv2.imread(path)
        if img is None:
            continue
        ok, buf = cv2.imencode(".jpg", img)
        files = [("images", ("f.jpg", buf.tobytes(), "image/jpeg"))]
        data = {"prompt": Q, "max_new_tokens": "8", "do_sample": "false"}
        try:
            r = requests.post(args.url, files=files, data=data, timeout=60)
            r.raise_for_status()
            ans = r.json().get("answer", "")
        except Exception as e:
            ans = f"__ERR__ {e}"
        pred_yes = parse_yes(ans)
        gt_yes = (kinds[i] == "strafe")
        row = {"i": int(i), "frame": os.path.basename(path), "path": path,
               "x": float(poses[i, 0]), "y": float(poses[i, 1]),
               "kind": kinds[i], "gt_yes": bool(gt_yes),
               "answer": ans, "pred_yes": bool(pred_yes)}
        jl.write(json.dumps(row, ensure_ascii=False) + "\n"); jl.flush()
        n_ok += 1
        if c % 50 == 0:
            dt = time.time() - t0
            print(f"  {c}/{len(idxs)}  x={poses[i,0]:5.1f} kind={kinds[i]:6s} pred={pred_yes} ({dt:.0f}s)", flush=True)
    jl.close()

    # 汇总指标
    rows = [json.loads(l) for l in open(os.path.join(OUT, "corridor_results.jsonl"))]
    rows = [r for r in rows if not str(r["answer"]).startswith("__ERR__")]
    tp = sum(1 for r in rows if r["gt_yes"] and r["pred_yes"])
    fp = sum(1 for r in rows if (not r["gt_yes"]) and r["pred_yes"])
    fn = sum(1 for r in rows if r["gt_yes"] and (not r["pred_yes"]))
    tn = sum(1 for r in rows if (not r["gt_yes"]) and (not r["pred_yes"]))
    n = len(rows)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    summary = {"model": "Infinite VLM (self-developed, 10.0.0.18)", "scene": "corridor blue water pump",
               "target": TGT, "question": Q, "n_frames": n,
               "gt_pos(strafe)": tp + fn, "gt_neg(cruise)": tn + fp,
               "TP": tp, "FP": fp, "FN": fn, "TN": tn,
               "accuracy": round(acc, 4), "precision": round(prec, 4),
               "recall": round(rec, 4), "f1": round(f1, 4)}
    json.dump(summary, open(os.path.join(OUT, "corridor_summary.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\n[summary]", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
