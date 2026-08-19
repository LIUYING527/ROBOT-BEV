#!/usr/bin/env python3
"""
同帧同prompt对比两个VLM后端在走廊蓝水泵巡检上的【准确率 + 延迟】:
  - Infinite VLM(自研, 远程 10.0.0.18:8152)
  - Qwen3.5-4B(本地对照, 127.0.0.1:8152)
用部署时真实的目标检测 prompt。GT=运动分类 kinds(strafe=泵在旁/cruise=走廊)。

输出 outputs/infinite_vlm_eval/compare_results.jsonl + compare_summary.json
"""
import os, json, time, argparse
import numpy as np, cv2, requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs/infinite_vlm_eval")
TGT = "a blue water pump"
Q = f"Is there {TGT} right beside you, close and large in view? Answer only yes or no."
BACKENDS = {
    "infinite": "http://10.0.0.18:8152/api/generate",
    "qwen": "http://127.0.0.1:8152/api/generate",
}


def parse_yes(a):
    a = (a or "").strip().lower()
    return a.startswith("yes") or ("yes" in a[:6] and "no" not in a[:6])


def call(url, img):
    ok, buf = cv2.imencode(".jpg", img)
    files = [("images", ("f.jpg", buf.tobytes(), "image/jpeg"))]
    data = {"prompt": Q, "max_new_tokens": "8", "do_sample": "false"}
    t = time.time()
    try:
        r = requests.post(url, files=files, data=data, timeout=90)
        r.raise_for_status(); a = r.json().get("answer", "")
    except Exception as e:
        a = f"__ERR__{e}"
    return a, time.time() - t


def metrics(recs, key):
    rr = [r for r in recs if not str(r[key + "_ans"]).startswith("__ERR__")]
    tp = sum(1 for r in rr if r["gt_yes"] and r[key + "_yes"])
    fp = sum(1 for r in rr if (not r["gt_yes"]) and r[key + "_yes"])
    fn = sum(1 for r in rr if r["gt_yes"] and (not r[key + "_yes"]))
    tn = sum(1 for r in rr if (not r["gt_yes"]) and (not r[key + "_yes"]))
    n = len(rr)
    lat = [r[key + "_lat"] for r in rr]
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    return {"n": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": round((tp + tn) / n, 4) if n else 0,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0,
            "lat_median_s": round(float(np.median(lat)), 3),
            "lat_mean_s": round(float(np.mean(lat)), 3),
            "lat_p90_s": round(float(np.percentile(lat, 90)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(ROOT, "outputs/nav_ds_real/meta.npz"), allow_pickle=True)
    fpv = z["fpv"]; poses = z["poses"]; kinds = np.array([str(k) for k in z["kinds"]])
    idxs = list(range(0, len(fpv), args.every))
    print(f"[compare] {len(idxs)} 帧, prompt={Q!r}", flush=True)
    recs = []
    for c, i in enumerate(idxs):
        img = cv2.imread(str(fpv[i]))
        if img is None:
            continue
        row = {"i": int(i), "x": float(poses[i, 0]), "kind": kinds[i],
               "gt_yes": bool(kinds[i] == "strafe")}
        for key, url in BACKENDS.items():
            a, lat = call(url, img)
            row[key + "_ans"] = a; row[key + "_yes"] = bool(parse_yes(a)); row[key + "_lat"] = lat
        recs.append(row)
        if c % 20 == 0:
            print(f"  {c}/{len(idxs)} x={poses[i,0]:5.1f} inf={row['infinite_ans'].strip()[:6]!r}"
                  f"({row['infinite_lat']:.2f}s) qwen={row['qwen_ans'].strip()[:6]!r}({row['qwen_lat']:.2f}s)", flush=True)
    with open(os.path.join(OUT, "compare_results.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {"prompt": Q, "gt": "kinds: strafe=pump-beside / cruise=corridor",
            "n_frames": len(recs),
            "infinite": metrics(recs, "infinite"), "qwen": metrics(recs, "qwen")}
    json.dump(summ, open(os.path.join(OUT, "compare_summary.json"), "w"), ensure_ascii=False, indent=2)
    print("\n[summary]", json.dumps(summ, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
