#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InfiniteVL 慢脑定位 HTTP 服务器（infinitevl env）。

接收 POST /locate  {rgb_b64, shape, instruction} → {visible, nx, ny}。
加载 InfiniteVL-LongSFT + LoRA（默认 pure ckpt-3262，val_unseen 47.2% 最优单帧）。
prompt 与 eval_slowbrain.py 单帧口径完全一致。

用法（infinitevl env，单卡）：
  CUDA_VISIBLE_DEVICES=1 python slowbrain_server.py \
      --lora /data/DongBaorong/BEVLOOK/sft_runs/slowbrain_lora_ablation_pure/checkpoint-3262 \
      --port 8123
"""
import os
import re
import json
import base64
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer

import time
import numpy as np
import cv2
import torch
from http.server import ThreadingHTTPServer


# ⭐0921 到达/STOP 监督：与 tools/gen_stop_dataset_0921.py 的标签口径一一对应。
# 训练数据里 `dist_to_goal < 3m` 的帧标签就是 `[TARGET] STOP`（照 DualVLN 的互斥分支）。
# 旧 adapter（farthest_r2 等）从不输出 STOP ⇒ 本改动对既有基线**零影响**。
STOP_RE = re.compile(r'\[TARGET\]\s*STOP', re.I)


def is_stop(text):
    return bool(STOP_RE.search(text or ''))


def parse_ego(text):
    """解析 ego 相对坐标 `[TARGET] 前向 横向`（米；横向左为正、右为负）。

    ⚠️ 与像素版 parse_coord 的关键区别：**必须允许负数**。
      像素版有 `nx < 0 or ny < 0 → None` 的约束（归一化像素不可能为负），
      但横向偏移天然可正可负（约 P05=-3.09 / P95=+3.10）⇒ 照搬会把左半边的样本全判成失败。
    """
    t = (text or '').strip()
    m = re.search(r"\[TARGET\]\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", t)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


class Locator:
    def __init__(self, model_path, lora_path, y2d=None, pass1_dist=None):
        from transformers import AutoModelForCausalLM, AutoProcessor
        from qwen_vl_utils import process_vision_info
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="flash_attention_2")
        if lora_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.process_vision_info = process_vision_info
        # 两遍自举(给了 y2d 才启用): dist 距离先验模型训练时每样本都有 GT 距离行,
        # 而真实导航拿不到 GT 距离 ⇒ 需要用模型自己的 y 几何反推一个距离回来。
        self.y2d = np.load(y2d) if y2d else None
        self.pass1_dist = pass1_dist      # 第一步的固定先验距离(米); None=不带距离行
        self.last_dist = None             # 时序自举: 上一步的距离估计
        self.n_two_pass = 0

    def _gen(self, ips, instruction, dist):
        """单次推理。ips 为图片路径列表, 最后一帧=当前帧。
        dist=None 时不插距离行(与 dist 训练格式不一致, 仅在完全没有先验时用)。
        frame_desc 必须与训练口径一致: 单帧「当前画面」/ 多帧「当前（最后一帧）画面」。"""
        dl = f"目标距离你约 {dist:.1f} 米。\n" if dist is not None else ""
        frame_desc = "当前画面" if len(ips) == 1 else "当前（最后一帧）画面"
        q = (f"{'<image>' * len(ips)}你是导航机器人。指令：{instruction}\n"
             f"请输出目标点相对于你当前位置与朝向的相对坐标，格式 [TARGET] 前向 横向："
             f"前向 = 沿你正前方的距离（米），横向 = 左为正、右为负（米）。\n"
             f"若已到达指令描述的终点，输出 [TARGET] STOP。")   # ⭐0923 必须与 tools/gen_ego_train_data.py 的 prompt_of 逐字一致
        content = [{"type": "image", "image": f"file://{p}"} for p in ips]
        content.append({"type": "text", "text": q})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        gids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        gids_trimmed = [o[len(t):] for t, o in zip(inputs.input_ids, gids)]
        out = self.processor.batch_decode(gids_trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        # 0921: 记录本次是否吐了 STOP，供 do_POST 带回给控制器（两遍/单遍路径都覆盖）
        self.last_stop = is_stop(out)
        return out

    @torch.no_grad()
    def locate(self, frames, instruction):
        """frames: RGB 数组列表(最后一帧=当前)。单元素时行为与原来完全一致。"""
        ips = []
        for j, rgb in enumerate(frames):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ip = f"/tmp/_sb_server_{os.getpid()}_{j}.jpg"
            cv2.imwrite(ip, bgr)
            ips.append(ip)
        # ⭐0923 ego 版：不做两遍自举，也不插「目标距离约 X 米」行。
        #   那两样都是给**像素表示**打补丁用的（像素反推不出距离，才需要 dist 先验行）。
        #   ego 点的模长本身就是距离 ⇒ 不需要任何距离先验 ⇒ 单遍即可。
        return parse_ego(self._gen(ips, instruction, None))


LOCATOR = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_POST(self):
        t0 = time.time()
        try:
            n = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(n))
            if 'frames' in data:      # 多帧: [{'rgb':b64,'shape':[...]}, ...], 最后一帧=当前
                frames = [np.frombuffer(base64.b64decode(f['rgb']), dtype=np.uint8).reshape(f['shape'])
                          for f in data['frames']]
            else:                     # 单帧(向后兼容)
                frames = [np.frombuffer(base64.b64decode(data['rgb']), dtype=np.uint8).reshape(data['shape'])]
            c = LOCATOR.locate(frames, data['instruction'])
            # ⭐0923 ego 版：直接回 ego 相对坐标(米)，由 eval 经 `_oracle` 通道喂快脑，
            #   **完全跳过「像素→depth 反投影」**（臂 B/C 实测那条链路吃掉 27.8pt）。
            resp = {'visible': c is not None,
                    'ego_fwd': c[0] if c else None, 'ego_lat': c[1] if c else None,
                    'stop': bool(getattr(LOCATOR, 'last_stop', False))}
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f'[locate] {time.time() - t0:.2f}s nframes={len(frames)} '
                  f'shape={frames[-1].shape} vis={c is not None} '
                  f'ego={None if c is None else (round(c[0], 2), round(c[1], 2))} '
                  f'stop={bool(getattr(LOCATOR, "last_stop", False))}', flush=True)
        except BrokenPipeError:
            print(f'[locate] broken-pipe at {time.time() - t0:.2f}s', flush=True)
        except Exception as e:
            print(f'[locate] error: {e}', flush=True)
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/DongBaorong/BEVLOOK/benchmarks/checkpoints/InfiniteVL-LongSFT")
    ap.add_argument("--lora", default="/data/DongBaorong/BEVLOOK/sft_runs/slowbrain_lora_ablation_pure/checkpoint-3262")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--y2d", default=None,
                    help="y->距离映射表 npz; 给了则启用两遍自举(dist 距离先验模型必需, 它是唯一没 GT 距离时用距离先验的办法)")
    ap.add_argument("--pass1-dist", type=float, default=None,
                    help="两遍模式第一步的固定先验距离(米); 推荐 4.2(= 训练集距离中位)")
    cli = ap.parse_args()
    global LOCATOR
    LOCATOR = Locator(cli.model, cli.lora, y2d=cli.y2d, pass1_dist=cli.pass1_dist)
    print(f"[server] InfiniteVL loaded lora={cli.lora} port={cli.port} "
          f"two_pass={'ON' if cli.y2d else 'OFF'} pass1_dist={cli.pass1_dist}", flush=True)
    ThreadingHTTPServer(('0.0.0.0', cli.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
