#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""慢脑「完全对齐 DualVLN」的推理服务（infinitevl env,1011）。

与旧 server 的区别（这次是**照抄官方 System2 的任务设置**）：
  * prompt = 官方英文模板（逐字 `internnav/.../internvla_n1_policy.py:60-64`）
  * 观测 resize 到 **384×384**（官方 `resize_w/h=384`）
  * 输出三种之一：
      `x y`        → 384 帧内**像素整数**（离散,CE 分类成立）
      `←←→` / `↑` → turn 动作串（不到达但看不见目标时,教模型"该怎么办"）
      `STOP`       → 到达
  * 响应给 eval 端：{"type": "pixel"|"turn"|"stop"|"none", "px","py","arrows","raw"}

训练数据由 `tools/gen_pixelgoal_from_dense.py` 产出，prompt 与解析口径必须与之逐字一致
（改这里就要同步改那边，否则又是 train/infer prompt 不一致那类坑）。

用法（infinitevl env,单卡）：
  CUDA_VISIBLE_DEVICES=0 python slowbrain_server_pixel384.py \
      --model /data/DongBaorong/BEVLOOK/sft_runs/slowbrain_full_pixelgoal_h8 --port 9201
"""
import os
import re
import json
import base64
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import time
import numpy as np
import cv2
import torch
from PIL import Image

TARGET_RES = 384
STOP_RE = re.compile(r'\bSTOP\b', re.I)
ARROW_CHARS = '←→↑↓'
CONJUNCTIONS = [
    'you can see ', 'in front of you is ', 'there is ',
    'you can spot ', 'you are toward the ', 'ahead of you is ', 'in your sight is ',
]
PROMPT_HEAD = (
    "You are an autonomous navigation assistant. Your task is to {instruction}. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully "
    "completed the task."
)


def parse_output(text):
    """把模型输出解析成 (kind, payload)。kind ∈ pixel/turn/stop/none。"""
    t = (text or '').strip()
    if STOP_RE.search(t):
        return 'stop', None
    arrows = ''.join(c for c in t if c in ARROW_CHARS)
    if arrows:
        return 'turn', arrows
    m = re.search(r'(-?\d+)\s+(-?\d+)', t)          # 像素整数,裸数字(官方格式)
    if m:
        px, py = int(m.group(1)), int(m.group(2))
        if 0 <= px < TARGET_RES and 0 <= py < TARGET_RES:
            return 'pixel', (px, py)
    return 'none', None


class Locator:
    def __init__(self, model_path, lora_path=''):
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

    def _gen(self, ips, instruction, want_hidden=False):
        n_hist = len(ips) - 1
        if n_hist <= 0:
            q = f"{PROMPT_HEAD.format(instruction=instruction)} you can see <image>."
        else:
            ph = '<image>' * n_hist
            q = (f"{PROMPT_HEAD.format(instruction=instruction)} "
                 f"These are your historical observations: {ph}. you can see <image>.")
        # ⚠️1011: 必须复刻 LLaMA-Factory `qwen2_vl` 的 mm_plugin —— 文本里的 `<image>`
        #   占位符要替换成 Qwen 的 vision token,并把 PIL 图单独传给 processor。
        #   否则模型看到字面量 `<image>` 字符串、根本没吃到图(探针实测栽在这)。
        #   注意:不能走 apply_chat_template 的 content 混合写法 —— 那会额外插入 N 个
        #   vision token,和文本里的占位符数量对不上。
        qq = q.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        text = f"<|im_start|>user\n{qq}<|im_end|>\n<|im_start|>assistant\n"
        pil = [Image.open(p).convert('RGB') for p in ips]
        inputs = self.processor(text=[text], images=pil,
                                padding=True, return_tensors="pt").to(self.model.device)
        # ⭐1015 P0-A: 可选吐「最后一层/最后一个生成 token」的 hidden(2048 维),
        #   供快脑的 text_proj 通道当 latent 用(见 docs/1015-接口带宽与快慢脑分工.md)。
        #   默认关闭 ⇒ 逐位不变,不影响既有评测。
        if want_hidden:
            out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                      output_hidden_states=True, return_dict_in_generate=True)
            gids = out.sequences
            # out.hidden_states: tuple(每生成步) -> tuple(每层) -> (1, seq, hidden)
            _h = out.hidden_states[-1][-1][0, -1, :].float().cpu().numpy()
        else:
            gids = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
            _h = None
        gids_trimmed = [o[len(t):] for t, o in zip(inputs.input_ids, gids)]
        txt = self.processor.batch_decode(gids_trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        return (txt, _h) if want_hidden else txt

    @torch.no_grad()
    def locate(self, frames, instruction, want_hidden=False):
        ips = []
        for j, rgb in enumerate(frames):
            im = cv2.resize(rgb, (TARGET_RES, TARGET_RES))       # 官方 384×384(直接拉伸)
            bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            ip = f"/tmp/_sb384_{os.getpid()}_{j}.jpg"
            cv2.imwrite(ip, bgr)
            ips.append(ip)
        if want_hidden:
            raw, hid = self._gen(ips, instruction, want_hidden=True)
        else:
            raw, hid = self._gen(ips, instruction), None
        kind, payload = parse_output(raw)
        return kind, payload, raw, hid


LOCATOR = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_POST(self):
        t0 = time.time()
        try:
            n = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(n))
            if 'frames' in data:
                frames = [np.frombuffer(base64.b64decode(f['rgb']), dtype=np.uint8).reshape(f['shape'])
                          for f in data['frames']]
            else:
                frames = [np.frombuffer(base64.b64decode(data['rgb']), dtype=np.uint8).reshape(data['shape'])]
            kind, payload, raw, _hid = LOCATOR.locate(frames, data['instruction'],
                                                      want_hidden=bool(data.get('want_hidden')))
            if _hid is not None:
                import base64 as _b64
                _hid_b64 = _b64.b64encode(_hid.astype('float16').tobytes()).decode()
            # 兼容旧 eval(`eval_r2r_escape_ego.py` 认 {visible,nx,ny,stop}):
            #   pixel ⇒ nx/ny 归一化到 [0,1],eval 自己按 640×480 反投影
            #   turn/stop ⇒ visible=False 让 eval 走它原有的 scan/判停逻辑
            resp = {'type': kind, 'raw': raw, 'visible': False, 'nx': None, 'ny': None, 'stop': False}
            if kind == 'pixel':
                px, py = int(payload[0]), int(payload[1])
                resp.update(px=px, py=py, visible=True,
                            nx=px / float(TARGET_RES), ny=py / float(TARGET_RES))
            elif kind == 'turn':
                # 只回 arrows,不再伪造像素 —— eval 已改为直接执行转向动作(见 eval_r2r_escape_ego.py 的 ⭐1011 块)。
                # 伪造像素行不通:画面上方是天花板(反投影仍 ~2m),中间偏下 ~1.9m,两头都撞 stop_dist 判停。
                resp['arrows'] = payload
            elif kind == 'stop':
                resp['stop'] = True
            if _hid is not None:
                # ⭐1015: latent 随响应回传(float16 + base64)。供快脑 text_proj 通道使用。
                # ⚠️ 原始 hidden 量级 ±50(std≈4.1), 与快脑 text_proj 现有输入(BERT[CLS], ±3)
                #    差一个数量级 ⇒ 零初始化投影对学习率会过敏感。
                #    固定常数缩放 /8.0(≈2×std) ⇒ 压到 ±6~8, 量级对齐; 用常数而非逐帧现算统计,
                #    避免特征分布随帧漂移(与快脑侧 1015 讨论一致)。
                _SCALE = 8.0
                _hid_s = (_hid / _SCALE).astype('float16')
                import base64 as _b64
                resp['hidden'] = _b64.b64encode(_hid_s.tobytes()).decode()
                resp['hidden_dim'] = int(_hid.shape[0])
                resp['hidden_dtype'] = 'float16'
                resp['hidden_scale'] = _SCALE
                resp['hidden_raw_mean'] = float(_hid.mean())
                resp['hidden_raw_std'] = float(_hid.std())
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f'[locate] {time.time()-t0:.2f}s n={len(frames)} -> {kind} {payload} raw={raw[:40]!r}',
                  flush=True)
        except BrokenPipeError:
            print(f'[locate] broken-pipe at {time.time()-t0:.2f}s', flush=True)
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
    ap.add_argument('--model', required=True, help='adapter 目录 或 基座目录')
    ap.add_argument('--lora', default='')
    ap.add_argument('--port', type=int, default=9201)
    cli = ap.parse_args()
    global LOCATOR
    LOCATOR = Locator(cli.model, cli.lora)
    # 就绪串与旧 server 保持一致,这样现有 runner(靠 grep 这个串判就绪)可直接复用
    print(f'[server] InfiniteVL loaded lora={cli.lora} port={cli.port} '
          f'two_pass=OFF pass1_dist=None [pixel384 res={TARGET_RES}]', flush=True)
    ThreadingHTTPServer(('0.0.0.0', cli.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
