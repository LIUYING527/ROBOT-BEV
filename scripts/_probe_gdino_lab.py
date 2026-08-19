#!/usr/bin/env python3
"""P0c: GroundingDINO+SAM 在 110744s2 实验室真帧上的开集检测小验证。
目的: 建场景图前先确认检测器对工业目标(铁柜/显微镜/移液枪/黄设备)的召回, 不行就换检测器。
用法: HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=3 ~/micromamba/envs/artifixer/bin/python scripts/_probe_gdino_lab.py
"""
import os, glob
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection, SamModel, SamProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = sorted(glob.glob(os.path.join(ROOT, "outputs/vggto_colmapjoint_110744s2/frames_zed/*.jpg")))
OUT = os.path.join(ROOT, "outputs/recon_verify/gdino_probe")
os.makedirs(OUT, exist_ok=True)

# 实验室目标(指定路径任务+巡检候选)
TEXT = "iron cabinet. metal cabinet. microscope. pipette. white table. yellow equipment. workbench. shelf. tool cabinet."
BOX_THR = 0.25
TEXT_THR = 0.25


def main():
    dev = "cuda"
    gdino_p = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gdino = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(dev).eval()
    sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(dev).eval()
    sam_p = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    print("[probe] 模型加载完成", flush=True)

    idxs = np.linspace(0, len(FRAMES) - 1, 12).astype(int)
    tiles = []
    for i in idxs:
        bgr = cv2.imread(FRAMES[i])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = gdino_p(images=image, text=TEXT, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = gdino(**inputs)
        res = gdino_p.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=BOX_THR, text_threshold=TEXT_THR,
            target_sizes=[image.size[::-1]])[0]
        boxes = res["boxes"].cpu().numpy(); labels = res["labels"]; scores = res["scores"].cpu().numpy()
        # SAM 细化前3个框
        if len(boxes):
            si = np.argsort(-scores)[:3]
            sp = sam_p(image, input_boxes=[boxes[si].tolist()], return_tensors="pt").to(dev)
            with torch.no_grad():
                so = sam(**sp)
            masks = sam_p.image_processor.post_process_masks(
                so.pred_masks.cpu(), sp["original_sizes"].cpu(), sp["reshaped_input_sizes"].cpu())[0]
            masks = (masks[:, 0].numpy() > 0)
            for t, j in enumerate(si):
                m = masks[t]
                bgr[m] = (0.4 * bgr[m] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)
        for (x0, y0, x1, y1), lb, sc in zip(boxes, labels, scores):
            cv2.rectangle(bgr, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.putText(bgr, f"{lb}:{sc:.2f}", (int(x0), max(20, int(y0) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(bgr, f"frame{i:04d} n={len(boxes)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        tiles.append(cv2.resize(bgr, (640, 360)))
        print(f"  frame{i:04d}: {len(boxes)}框 {[f'{l}:{s:.2f}' for l, s in zip(labels, scores)]}", flush=True)
    sheet = np.vstack([np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(4)])
    cv2.imwrite(os.path.join(OUT, "gdino_lab_12f.png"), sheet)
    print(f"[probe] → {OUT}/gdino_lab_12f.png")


if __name__ == "__main__":
    main()
