"""用自研 VLM(API) 给现有 obs_seq 去程帧逐帧打 cond 标签 -> json。
供 train_conditional.py --cond_labels 用: 把"VLM 输出的 cond_id"当训练条件(取代 torch.arange 瞎枚举)。

轻量: 只需 requests + cv2 + numpy(无需 torch/mujoco), 慢脑走远程 API。
用法:
  ~/discoverse_venv/bin/python arch_e2e/pipeline/label_cond_vlm.py --session colmapjoint_all
产物: arch_e2e/runs/cond_labels_<session>.json  + 打印 cond 分布
"""
import os, sys, json, argparse, collections
import numpy as np
import cv2
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "arch_e2e", "runs")
API_URL = "http://10.0.0.18:8152/api/generate"

CONDITIONS = ["inspect_left", "cruise", "inspect_right", "pause"]

# 与 sim_fullchain.py 完全一致(train/test 用同一 prompt)
VLM_PROMPT = (
    "You are an inspection robot moving through a corridor. Look at this front camera view and pick "
    "the single best action. Choose EXACTLY ONE word: "
    "cruise (corridor is clear ahead, go straight), "
    "inspect_left (a device/valve/cabinet/opening is on the LEFT, veer left to inspect), "
    "inspect_right (a device/valve/cabinet/opening is on the RIGHT, veer right to inspect). "
    "Return ONLY one word: cruise, inspect_left, or inspect_right."
)


def vlm_api(frame_bgr, prompt, max_new_tokens=32, timeout=120):
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    files = [("images", ("fpv.jpg", buf.tobytes(), "image/jpeg"))]
    data = {"prompt": prompt, "max_new_tokens": str(max_new_tokens), "do_sample": "false"}
    r = requests.post(API_URL, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.json().get("answer", "")


def answer_to_cond(ans):
    t = (ans or "").lower()
    if any(k in t for k in ["pause", "stop", "black", "blur", "block", "dark", "暂停", "停"]):
        return 3
    if any(k in t for k in ["left", "左"]):
        return 0
    if any(k in t for k in ["right", "右"]):
        return 2
    return 1  # cruise


def forward_leg_n(poses):
    d = np.linalg.norm(poses[:, :2] - poses[0, :2], axis=1)
    return int(d.argmax()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--prompt", default=None, help="覆盖默认 prompt(可试不同措辞)")
    args = ap.parse_args()
    prompt = args.prompt or VLM_PROMPT

    seqdir = os.path.join(ROOT, "outputs", f"obs_seq_{args.session}")
    poses = np.load(os.path.join(seqdir, "poses.npz"), allow_pickle=True)["poses"].astype(float)
    nfwd = forward_leg_n(poses)
    print(f"[label] 去程 {nfwd} 帧, prompt:\n  {prompt}\n", flush=True)

    labels = {}
    for i in range(nfwd):
        fname = f"{i:03d}.jpg"
        fp = os.path.join(seqdir, fname)
        fpv = cv2.imread(fp)
        if fpv is None:
            continue
        try:
            ans = vlm_api(fpv, prompt)
        except Exception as e:
            ans = f"[ERR]{e}"
        cid = answer_to_cond(ans)
        labels[fname] = {"answer": ans, "cond": CONDITIONS[cid], "cond_id": cid}
        print(f"  {fname}  answer={ans!r:<20} -> {CONDITIONS[cid]}", flush=True)

    dist = collections.Counter(v["cond"] for v in labels.values())
    out = os.path.join(RUNS, f"cond_labels_{args.session}.json")
    json.dump({"session": args.session, "prompt": prompt, "labels": labels},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[done] {len(labels)} 帧已标 -> {out}")
    print(f"       cond 分布: {dict(dist)}")


if __name__ == "__main__":
    main()
