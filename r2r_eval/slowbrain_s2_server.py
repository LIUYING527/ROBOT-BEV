#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把「我们自训的像素目标慢脑」包成 hybrid 链路期望的 `/s2` 接口(1012)。

为什么需要:hybrid(`run_r2r_hybrid.py` + `dualvlm_slow_server.py`)这条链路
**已经用官方 S2 验证到 56ep SR 66.1%**;而 `eval_r2r_escape_ego.py` 那条链路
(ego 目标 + 4m 截断、每步重算)用同样的慢脑是 SR 0。
两条链路的语义不同 ⇒ 直接把我们的模型接到**已验证的 hybrid 链路**上,
才能公平回答"我们的慢脑到底能到多少"。

接口与 `dualvlm_slow_server.py` 逐字一致:
  POST /reset          -> 清历史(stateful 可忽略,我们无状态)
  POST /s2  multipart: file=jpg + data{instruction, infer("1"/"0")}
      infer=1 -> {"type":"pixel","pixel":[y,x]} | {"type":"actions","actions":[1/2/3]} | {"type":"stop"}
      infer=0 -> {"type":"ok"}   (只补历史帧;我们无状态,直接 ok)
⚠️ hybrid 的 `pixel_to_world_goal(px, py, ...)` 里 `u = px*W/384` ⇒ px 必须是 **384 帧坐标**。

启动:
  CUDA_VISIBLE_DEVICES=6 /data/DongBaorong/miniconda3/envs/infinitevl/bin/python \
      slowbrain_s2_server.py --model <adapter_dir> --port 8155
"""
import argparse
import io

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
import os
import re

TARGET_RES = 384
STOP_RE = re.compile(r'\bSTOP\b', re.I)
ARROW_CHARS = '←→↑↓'
ARROW2ACT = {'↑': 1, '←': 2, '→': 3}          # 与 hybrid 的 ARROW2ACT 对齐
PROMPT_HEAD = (
    "You are an autonomous navigation assistant. Your task is to {instruction}. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully "
    "completed the task."
)

app = FastAPI()
NET = None


def parse_output(text):
    t = (text or '').strip()
    if STOP_RE.search(t):
        return 'stop', None
    arrows = [c for c in t if c in ARROW_CHARS]
    if arrows:
        return 'actions', [ARROW2ACT[c] for c in arrows if c in ARROW2ACT]
    m = re.search(r'(-?\d+)\s+(-?\d+)', t)
    if m:
        px, py = int(m.group(1)), int(m.group(2))
        if 0 <= px < TARGET_RES and 0 <= py < TARGET_RES:
            return 'pixel', (py, px)          # ⚠️ hybrid 收 [y, x]
    return 'none', None


class Net:
    def __init__(self, model_path, lora='', num_history=8, span=40):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={'': 0},
            trust_remote_code=True, attn_implementation='flash_attention_2')
        if lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora)
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # ⚠️1012 必须**有状态**:训练时 prompt 含 8 帧历史(span=40 等距),
        #   而 hybrid 的 s2_call 每次只发**当前 1 帧** ⇒ 必须自己缓存帧、拼历史,
        #   否则模型收到的输入与训练口径不符(实测单帧输入下闭环 SR 0,结果作废)。
        self.num_history = num_history
        self.span = span
        self.rgb_list = []

    def reset(self):
        self.rgb_list = []

    def _hist_idx(self, i, K, span):
        """与 gen_pixelgoal_from_dense.py:hist_indices 逐字同口径(等距 + 末位当前帧 + 去重)。"""
        lo = max(0, i - span) if span > 0 else max(0, i - K + 1)
        if i - lo < K - 1:
            idx = [lo] * (K - 1) + [i]
        else:
            idx = [int(round(lo + k * (i - lo) / (K - 1))) for k in range(K)]
        out = []
        for k in idx:
            if not out or k != out[-1]:
                out.append(k)
        return out

    @torch.no_grad()
    def s2_step(self, rgb, instruction):
        self.rgb_list.append(rgb)              # 累积历史帧(与官方 policy 的 rgb_list 同语义)
        i = len(self.rgb_list) - 1
        idxs = self._hist_idx(i, self.num_history, self.span)
        ims = [Image.fromarray(self.rgb_list[k]).convert('RGB').resize((TARGET_RES, TARGET_RES))
               for k in idxs]
        # ⚠️ 必须复刻 LLaMA-Factory 的 mm_plugin:文本 `<image>` → Qwen vision token,
        #   并把 PIL 单独传给 processor(否则模型根本没吃到图,输出全是垃圾)。
        ph = '<|vision_start|><|image_pad|><|vision_end|>'
        n_hist = len(ims) - 1
        if n_hist <= 0:
            q = f"{PROMPT_HEAD.format(instruction=instruction)} you can see {ph}."
        else:
            q = (f"{PROMPT_HEAD.format(instruction=instruction)} "
                 f"These are your historical observations: {ph * n_hist}. you can see {ph}.")
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inp = self.proc(text=[text], images=ims, padding=True,
                        return_tensors='pt').to(self.model.device)
        g = self.model.generate(**inp, max_new_tokens=32, do_sample=False)
        g = [o[len(t):] for t, o in zip(inp.input_ids, g)]
        return self.proc.batch_decode(g, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

    def step_no_infer(self, rgb):
        """infer=0:只补历史帧(hybrid 在非重定位步调用),不推理。"""
        self.rgb_list.append(rgb)


@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'our-pixelgoal-s2'}


@app.post('/reset')
def reset():
    NET.reset()
    return {'status': 'success'}


@app.post('/s2')
async def s2(file: UploadFile = File(...), instruction: str = Form(...),
             infer: str = Form('1')):
    rgb = np.array(Image.open(io.BytesIO(await file.read())).convert('RGB'))
    if infer != '1':
        NET.step_no_infer(rgb)                 # 只补历史帧
        return {'type': 'ok'}
    raw = NET.s2_step(rgb, instruction)
    kind, payload = parse_output(raw)
    if kind == 'pixel':
        return {'type': 'pixel', 'pixel': [int(payload[0]), int(payload[1])], 'raw': raw}
    if kind == 'actions':
        return {'type': 'actions', 'actions': payload, 'raw': raw}
    if kind == 'stop':
        return {'type': 'stop', 'raw': raw}
    return {'type': 'none', 'raw': raw}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--lora', default='')
    ap.add_argument('--port', type=int, default=8155)
    cli = ap.parse_args()
    NET = Net(cli.model, cli.lora)
    print(f'[slow] our pixelgoal S2 ready model={cli.model} port={cli.port}', flush=True)
    uvicorn.run(app, host='0.0.0.0', port=cli.port, log_level='warning')
