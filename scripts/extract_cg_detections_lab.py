#!/usr/bin/env python3
"""ConceptGraphs 每帧检测提取(transformers版GDINO+SAM+open_clip, 替代官方Grounded-SAM repo harness)。
输出官方 gobs pkl.gz 格式 → 官方 cfslam_pipeline_batch 建图直接吃。

gobs keys: xyxy(N,4) / mask(N,H,W)bool / class_id(N,) / classes(list) / confidence(N,) /
           image_feats(N,D) / text_feats(N,D)   —— 与 conceptgraph.slam.utils.gobs_to_detection_list 对齐

用法: CUDA_VISIBLE_DEVICES=3 ~/micromamba/envs/cg/bin/python scripts/extract_cg_detections_lab.py [--start 0 --end -1]
"""
import os, sys, glob, gzip, pickle, argparse
import numpy as np
import cv2
import torch
import open_clip
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection, SamModel, SamProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_ROOT = os.path.join(ROOT, "outputs/cg_lab110744")
SCENE = "lab110744"

# 实验室开放词汇(覆盖指定路径任务目标+常见实验室内物; 近义合并交给mapping的merge步骤)
CLASSES = [
    "iron cabinet", "metal cabinet", "tool cabinet", "cabinet",
    "microscope", "pipette", "white table", "workbench", "yellow equipment",
    "shelf", "monitor", "keyboard", "office chair", "chair", "stool",
    "robot arm", "window", "door", "box", "bottle", "tool",
    "power strip", "whiteboard", "poster", "equipment", "machine",
]
TEXT = ". ".join(CLASSES) + "."
BOX_THR, TEXT_THR, NMS_IOU = 0.25, 0.25, 0.9
CLIP_MODEL, CLIP_PRE = "ViT-H-14", "laion2b_s32b_b79k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    gdino_p = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gdino = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
    sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(dev).eval()
    sam_p = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    clip_model, _, clip_pre = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRE)
    clip_model = clip_model.to(dev).eval()
    clip_tok = open_clip.get_tokenizer(CLIP_MODEL)
    print("[extract] 模型全加载", flush=True)

    # 文本特征(全部类一次算好)
    with torch.no_grad():
        tok = clip_tok(CLASSES).to(dev)
        text_all = clip_model.encode_text(tok).float()
        text_all = torch.nn.functional.normalize(text_all, dim=-1).cpu().numpy()

    frames = sorted(glob.glob(os.path.join(SCENE_ROOT, SCENE, "results", "frame*.jpg")))
    end = len(frames) if args.end < 0 else min(args.end, len(frames))
    outdir = os.path.join(SCENE_ROOT, SCENE, "detections_lab")
    os.makedirs(outdir, exist_ok=True)

    for fi in range(args.start, end):
        fp = frames[fi]
        outp = os.path.join(outdir, os.path.basename(fp).replace(".jpg", ".pkl.gz"))
        if os.path.exists(outp):
            continue
        bgr = cv2.imread(fp)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = gdino_p(images=image, text=TEXT, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = gdino(**inputs)
        res = gdino_p.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=BOX_THR, text_threshold=TEXT_THR,
            target_sizes=[image.size[::-1]])[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"]
        # 清洗标签(去掉##碎片/空格), 映射回CLASSES; 映射不上的丢弃
        cid = []
        keep = []
        for k, lb in enumerate(labels):
            lb2 = lb.replace("##", "").strip()
            hit = next((j for j, c in enumerate(CLASSES) if lb2 in c or c in lb2), None)
            if hit is not None:
                cid.append(hit); keep.append(k)
        if not keep:
            gobs = {"xyxy": np.zeros((0, 4), np.float32), "mask": np.zeros((0, *rgb.shape[:2]), bool),
                    "class_id": np.zeros(0, np.int64), "classes": CLASSES,
                    "confidence": np.zeros(0, np.float32),
                    "image_feats": np.zeros((0, text_all.shape[1]), np.float32),
                    "text_feats": np.zeros((0, text_all.shape[1]), np.float32)}
            with gzip.open(outp, "wb") as f:
                pickle.dump(gobs, f)
            continue
        boxes, scores, cid = boxes[keep], scores[keep], np.array(cid, np.int64)
        # 类无关NMS(只杀近重复框 IoU>0.9, 保留相邻相似物如货架格)
        order = np.argsort(-scores)
        keep2 = []
        while len(order):
            i = order[0]; keep2.append(i)
            if len(order) == 1:
                break
            b0 = boxes[i]; rest = boxes[order[1:]]
            inter = (np.minimum(b0[2], rest[:, 2]) - np.maximum(b0[0], rest[:, 0])).clip(0) * \
                    (np.minimum(b0[3], rest[:, 3]) - np.maximum(b0[1], rest[:, 1])).clip(0)
            iou = inter / ((b0[2] - b0[0]) * (b0[3] - b0[1]) + 1e-9)
            order = order[1:][iou < NMS_IOU]
        boxes, scores, cid = boxes[keep2], scores[keep2], cid[keep2]
        # SAM 细化
        sp = sam_p(image, input_boxes=[boxes.tolist()], return_tensors="pt").to(dev)
        with torch.no_grad():
            so = sam(**sp)
        masks = sam_p.image_processor.post_process_masks(
            so.pred_masks.cpu(), sp["original_sizes"].cpu(), sp["reshaped_input_sizes"].cpu())[0]
        masks = (masks[:, 0].numpy() > 0)
        # CLIP 图像特征(crop+padding20)
        crops = []
        for (x0, y0, x1, y1) in boxes:
            x0 = max(0, int(x0) - 20); y0 = max(0, int(y0) - 20)
            x1 = min(rgb.shape[1], int(x1) + 20); y1 = min(rgb.shape[0], int(y1) + 20)
            crops.append(clip_pre(Image.fromarray(rgb[y0:y1, x0:x1])))
        with torch.no_grad():
            cim = torch.stack(crops).to(dev)
            image_feats = clip_model.encode_image(cim).float()
            image_feats = torch.nn.functional.normalize(image_feats, dim=-1).cpu().numpy()
        gobs = {"xyxy": boxes.astype(np.float32), "mask": masks, "class_id": cid,
                "classes": CLASSES, "confidence": scores.astype(np.float32),
                "image_feats": image_feats.astype(np.float32),
                "text_feats": text_all[cid].astype(np.float32)}
        with gzip.open(outp, "wb") as f:
            pickle.dump(gobs, f)
        if fi % 20 == 0:
            print(f"  frame{fi:05d}: {len(boxes)}obj", flush=True)
    print(f"[extract] 完成 → {outdir}", flush=True)


if __name__ == "__main__":
    main()
