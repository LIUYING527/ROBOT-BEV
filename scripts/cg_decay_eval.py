"""巡逻密度衰减评估 (0729): 同一套GT-free检索规则跑多密度场景图, 出三目标误差表。

规则(与0729闭环一致, 全部GT-free):
- microscope: CLIP(text+clip特征+RAM标签加成) top1
- iron_cabinet: CLIP top10 → VLM-yes(金属柜) → 距microscope拾取点[0.8,1.5m]内CLIP最高
- pipette_table: table类节点中白色像素占比top1
另报: 地图覆盖度(GT 0.3m内有无节点) + 最佳观测位姿帧。

用法: PYTHONPATH=third_party/concept-graphs:scripts micromamba run -n cg python scripts/cg_decay_eval.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "concept-graphs"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from cg_rag_vlm_rerank import load_map, majority_tag, best_crop  # noqa: E402
from conceptgraph.llm.infinite_vlm import _post as vlm  # noqa: E402

GT = json.load(open(os.path.join(ROOT, "outputs/lab_mission_targets_zed.json")))
MAPS = {
    "s2(255帧)": "outputs/cg_lab110744s2/lab1/pcd_saves/full_pcd_ram_cfslam_post.pkl.gz",
    "s5(102帧)": "outputs/cg_lab110744s2/lab1/pcd_saves/full_pcd_ram_s5_post.pkl.gz",
    "s10(51帧)": "outputs/cg_lab110744s2/lab1/pcd_saves/full_pcd_ram_s10_post.pkl.gz",
}
PROMPTS = {
    "microscope": ["microscope", "a microscope on a table", "laboratory microscope"],
    "iron_cabinet": ["iron cabinet", "a metal storage cabinet", "metal equipment cabinet",
                     "cabinet", "metal box equipment"],
}
WORDS = {
    "microscope": ["microscope", "scope"],
    "iron_cabinet": ["cabinet", "iron", "metal", "equipment", "rack", "appliance", "oven", "microwave"],
}
METAL_TAGS = ("microwave", "oven", "appliance", "equipment", "cabinet", "metal")


def ctr(o):
    return np.asarray(o["bbox_np"], dtype=float).mean(0)


def tag_frac(o, words):
    tags = [str(t).lower() for t in o.get("class_name", [])]
    if not tags:
        return 0.0
    return sum(1 for t in tags if any(w in t for w in words)) / len(tags)


def main():
    import torch
    import open_clip
    import cv2

    device = "cuda"
    model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-H-14")

    def clip_rank(objs, name, k=10):
        text_ft = torch.tensor(np.stack([o["text_ft"] for o in objs]), dtype=torch.float32, device=device)
        clip_ft = torch.tensor(np.stack([o["clip_ft"] for o in objs]), dtype=torch.float32, device=device)
        text_ft /= text_ft.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        clip_ft /= clip_ft.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        q = model.encode_text(tok(PROMPTS[name]).to(device)).float()
        q /= q.norm(dim=-1, keepdim=True)
        sim = 0.5 * (q @ text_ft.T).mean(0) + 0.5 * (q @ clip_ft.T).mean(0)
        bonus = torch.tensor([tag_frac(o, WORDS[name]) for o in objs], device=device) * 0.05
        return torch.topk(sim + bonus, min(k, len(objs)))

    report = {}
    for label, rel in MAPS.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"--- {label}: 地图未就绪 {rel}")
            continue
        objs = load_map(path)
        res = {"n_nodes": len(objs)}
        # 覆盖度
        for name in ("microscope", "iron_cabinet", "pipette_table"):
            g = np.array(GT[name])
            dmin = min(np.linalg.norm(np.asarray(o["pcd_np"]) - g, axis=1).min() for o in objs if len(o["pcd_np"]))
            res[f"cover_{name}"] = round(float(dmin), 2)
        # microscope top1
        top = clip_rank(objs, "microscope", 3)
        mi = top.indices[0].item()
        mc = ctr(objs[mi])
        res["microscope"] = {"node": mi, "tag": majority_tag(objs[mi]),
                             "err": round(float(np.linalg.norm(mc - np.array(GT["microscope"]))), 2)}
        # iron_cabinet: top10 → VLM yes → 显微镜邻域[0.8,1.5]内CLIP最高
        top = clip_rank(objs, "iron_cabinet", 10)
        cands = []
        for s_, i_ in zip(top.values.tolist(), top.indices.tolist()):
            o = objs[i_]
            crop = best_crop(o)
            vy = False
            if crop is not None:
                try:
                    ans = vlm([crop], "The object is outlined with a red rectangle. Is the outlined object "
                                      "a metal cabinet or metal equipment box? Answer only yes or no.",
                              max_new_tokens=8, timeout=60, retries=2)
                    vy = "yes" in ans.lower()
                except Exception:
                    pass
            cands.append({"i": i_, "sim": s_, "vlm": vy, "c": ctr(o), "tag": majority_tag(o)})
        near = [c for c in cands if c["vlm"] and 0.8 <= np.linalg.norm(c["c"] - mc) <= 1.5
                and any(t in c["tag"] for t in METAL_TAGS)]
        pick = max(near, key=lambda c: c["sim"]) if near else (
            max([c for c in cands if c["vlm"]], key=lambda c: c["sim"], default=cands[0]))
        res["iron_cabinet"] = {"node": pick["i"], "tag": pick["tag"],
                               "err": round(float(np.linalg.norm(pick["c"] - np.array(GT["iron_cabinet"]))), 2)}
        # pipette_table: table类白像素top1
        def white_frac(o):
            c = best_crop(o)
            if c is None:
                return 0.0
            hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
            return float((cv2.inRange(hsv, np.array((0, 0, 180)), np.array((180, 60, 255))) > 0).mean())
        tabs = [(i, o) for i, o in enumerate(objs)
                if any(k in majority_tag(o) for k in ("table", "desk", "workbench"))]
        if tabs:
            ti, to = max(tabs, key=lambda t: white_frac(t[1]))
            res["pipette_table"] = {"node": ti, "tag": majority_tag(to),
                                    "err": round(float(np.linalg.norm(ctr(to) - np.array(GT["pipette_table"]))), 2)}
        report[label] = res
        print(f"--- {label}: {json.dumps(res)}")

    out = os.path.join(ROOT, "outputs/cg_lab110744s2/cg_decay_eval_0729.json")
    json.dump(report, open(out, "w"), indent=1)
    print("\n==== 衰减表 ====")
    print(f"{'密度':12s} {'节点':>5s} {'micro':>6s} {'cabinet':>8s} {'table':>6s}  覆盖(micro/cab/table)")
    for label, r in report.items():
        print(f"{label:12s} {r['n_nodes']:5d} {r.get('microscope',{}).get('err','-'):>6} "
              f"{r.get('iron_cabinet',{}).get('err','-'):>8} {r.get('pipette_table',{}).get('err','-'):>6}  "
              f"{r.get('cover_microscope')}/{r.get('cover_iron_cabinet')}/{r.get('cover_pipette_table')}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
