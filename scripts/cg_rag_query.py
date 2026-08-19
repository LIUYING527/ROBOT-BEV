"""ConceptGraphs 场景图 RAG 检索 v2 (0729)。

v2 改动:
- --map 可指定任意场景图/合并结果(兼容无caption_dict的post-merge pkl, tag回退RAM多数票)
- 查询同义词扩展(治"铁柜被打标成equipment"的词表错位)
- 标签命中加成: 节点多视角RAM标签列表里出现查询同义词 -> 加分(确定性规则)
- 同时评估多张地图对比(如 185原版 vs tight0.9版)

用法:
  PYTHONPATH=third_party/concept-graphs micromamba run -n cg python scripts/cg_rag_query.py \
    --map outputs/cg_lab110744s2/lab1/pcd_saves/full_pcd_ram_cfslam_tight_post.pkl.gz \
    --out outputs/cg_lab110744s2/cg_rag_smoke_tight_0729.json
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

# 同义词扩展(查询词 -> CLIP prompt集 + 标签匹配词集)
QUERIES = {
    "microscope": {
        "prompts": ["microscope", "a microscope on a table", "laboratory microscope"],
        "words": ["microscope", "scope"],
    },
    "iron_cabinet": {
        "prompts": ["iron cabinet", "a metal storage cabinet", "metal equipment cabinet",
                    "cabinet", "equipment rack"],
        "words": ["cabinet", "iron", "metal", "equipment", "rack", "appliance", "oven"],
    },
    "pipette_table": {
        "prompts": ["white table", "a white work table", "white lab table", "white desk"],
        "words": ["table", "desk", "workbench"],
    },
}
TAG_BONUS = 0.05  # 标签列表命中查询词集的加成


def node_center(o):
    return np.asarray(o["bbox_np"], dtype=np.float64).mean(0)


def majority_tag(o):
    cap = o.get("caption_dict")
    if isinstance(cap, dict):
        t = cap.get("response", {}).get("object_tag") if isinstance(cap.get("response"), dict) else None
        if t:
            return str(t).lower()
    tags = [str(t).lower() for t in o.get("class_name", [])]
    return Counter(tags).most_common(1)[0][0] if tags else "?"


def tag_hit(o, words):
    tags = [str(t).lower() for t in o.get("class_name", [])]
    if not tags:
        return 0.0
    frac = sum(1 for t in tags if any(w in t for w in words)) / len(tags)
    return frac


def load_map(path):
    with gzip.open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict):
        return d["objects"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=os.path.join(
        ROOT, "outputs/cg_lab110744s2/lab1/sg_cache/map/scene_map_cfslam_pruned.pkl.gz"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag_bonus", type=float, default=TAG_BONUS)
    args = ap.parse_args()

    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-H-14")

    scene_map = load_map(args.map)
    gt = json.load(open(GT_JSON))
    print(f"map: {args.map}\nnodes: {len(scene_map)}")

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
        bonus = torch.tensor([tag_hit(o, q["words"]) for o in scene_map], device=device) * args.tag_bonus
        sim = sim + bonus
        topk = torch.topk(sim, 3)
        print(f"\n=== {name} (GT={gt.get(name)}) ===")
        entries = []
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            o = scene_map[idx]
            c = node_center(o)
            dist = float(np.linalg.norm(c - np.array(gt[name]))) if name in gt else None
            print(f"  sim={score:.3f} node{idx} tag={majority_tag(o)} ndet={o['num_detections']}"
                  f" center={c.round(2).tolist()}"
                  + (f" dist={dist:.2f}m" if dist is not None else ""))
            entries.append({"node": idx, "tag": majority_tag(o), "sim": round(score, 4),
                            "center": c.round(3).tolist(), "dist_to_gt": dist})
        report[name] = entries

    out = args.out or os.path.join(ROOT, "outputs/cg_lab110744s2/cg_rag_smoke_v2_0729.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
