"""RAG 两阶段检索: CLIP粗排 topK -> VLM精排 (0729)。

动机: 开集打标在工业物体上词表漂移(铁柜->microwave/oven, 白桌->tool/computer),
纯文本查询top-3经常不含真值节点。但节点的存储视角crop外观是对的,
而自研VLM的强项正是看图yes/no(走廊已验证) -> 用它当re-ranker。

流程: CLIP(text+img特征+标签加成) topK=10 -> 每候选取最高conf检测crop(画红框)
-> VLM "红框里是{query}吗 yes/no" -> yes优先按CLIP sim重排 -> top3输出+距真值。

用法:
  CG_VLM_URL=http://10.0.0.18:8152/api/generate PYTHONPATH=third_party/concept-graphs \
  micromamba run -n cg python scripts/cg_rag_vlm_rerank.py [--topk 10] [--map ...]
"""
import argparse
import gzip
import json
import os
import pickle
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_JSON = os.path.join(ROOT, "outputs/lab_mission_targets_zed.json")

QUERIES = {
    "microscope": {
        "prompts": ["microscope", "a microscope on a table", "laboratory microscope"],
        "words": ["microscope", "scope"],
        "vlm_q": "a microscope",
    },
    "iron_cabinet": {
        "prompts": ["iron cabinet", "a metal storage cabinet", "metal equipment cabinet",
                    "cabinet", "metal box equipment"],
        "words": ["cabinet", "iron", "metal", "equipment", "rack", "appliance", "oven", "microwave"],
        "vlm_q": "a metal cabinet or metal equipment box",
    },
    "pipette_table": {
        "prompts": ["white table", "a white work table", "white lab table", "white desk"],
        "words": ["table", "desk", "workbench"],
        "vlm_q": "a white table",
    },
}
TAG_BONUS = 0.05


def load_map(path):
    with gzip.open(path, "rb") as f:
        d = pickle.load(f)
    return d["objects"] if isinstance(d, dict) else d


def majority_tag(o):
    tags = [str(t).lower() for t in o.get("class_name", [])]
    return Counter(tags).most_common(1)[0][0] if tags else "?"


def best_crop(o, margin=0.15):
    """取最高conf检测, 从原帧裁bbox(留边), 画红框。返回BGR图或None。"""
    import cv2
    if not o.get("color_path"):
        return None
    j = int(np.argmax(o["conf"]))
    img = cv2.imread(o["color_path"][j])
    if img is None:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1 = o["xyxy"][j]
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
    crop = img[y0:y1, x0:x1].copy()
    bx0, by0 = int(mx), int(my)
    bx1, by1 = crop.shape[1] - int(mx), crop.shape[0] - int(my)
    cv2.rectangle(crop, (bx0, by0), (bx1, by1), (0, 0, 255), 3)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=os.path.join(
        ROOT, "outputs/cg_lab110744s2/lab1/sg_cache/map/scene_map_cfslam_pruned.pkl.gz"))
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.join(ROOT, "third_party/concept-graphs"))
    from conceptgraph.llm.infinite_vlm import _post as vlm_post

    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-H-14")

    scene_map = load_map(args.map)
    gt = json.load(open(GT_JSON))
    print(f"nodes: {len(scene_map)}")

    text_ft = torch.tensor(np.stack([o["text_ft"] for o in scene_map]), dtype=torch.float32, device=device)
    clip_ft = torch.tensor(np.stack([o["clip_ft"] for o in scene_map]), dtype=torch.float32, device=device)
    text_ft = text_ft / text_ft.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    clip_ft = clip_ft / clip_ft.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    report = {}
    for name, q in QUERIES.items():
        with torch.no_grad():
            qf = model.encode_text(tok(q["prompts"]).to(device)).float()
            qf = qf / qf.norm(dim=-1, keepdim=True)
        sim = 0.5 * (qf @ text_ft.T).mean(0) + 0.5 * (qf @ clip_ft.T).mean(0)
        bonus = torch.tensor([
            sum(1 for t in o.get("class_name", []) if any(w in str(t).lower() for w in q["words"]))
            / max(len(o.get("class_name", [])), 1) for o in scene_map
        ], device=device) * TAG_BONUS
        sim = sim + bonus
        topk = torch.topk(sim, args.topk)

        print(f"\n=== {name} (GT={gt.get(name)}) CLIP top{args.topk} -> VLM re-rank ===")
        cands = []
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            o = scene_map[idx]
            crop = best_crop(o)
            verdict = "no_crop"
            if crop is not None:
                try:
                    ans = vlm_post([crop],
                                   f"The object is outlined with a red rectangle. "
                                   f"Is the outlined object {q['vlm_q']}? Answer only yes or no.",
                                   max_new_tokens=8, timeout=60, retries=2)
                    verdict = "yes" if "yes" in ans.lower() else ("no" if "no" in ans.lower() else ans[:20])
                except Exception as e:
                    verdict = f"err:{e}"
            c = np.asarray(o["bbox_np"], dtype=float).mean(0)
            dist = float(np.linalg.norm(c - np.array(gt[name])))
            cands.append({"node": idx, "tag": majority_tag(o), "clip_sim": round(float(score), 4),
                          "vlm": verdict, "center": c.round(3).tolist(), "dist_to_gt": round(dist, 3)})
            print(f"  node{idx:3d} tag={majority_tag(o):15s} clip={score:.3f} vlm={verdict:10s} dist={dist:.2f}m")

        # 重排: yes优先, 组内按clip sim
        reranked = sorted(cands, key=lambda c: (c["vlm"] != "yes", -c["clip_sim"]))
        report[name] = reranked
        best = reranked[0]
        print(f"  -> FINAL top1: node{best['node']} \"{best['tag']}\" vlm={best['vlm']} dist={best['dist_to_gt']}m")

    out = args.out or os.path.join(ROOT, "outputs/cg_lab110744s2/cg_rag_vlm_rerank_0729.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
