#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DualVLN S2 慢脑 HTTP 服务(0902, internvla env, 4bit):
  POST /reset -> 清空episode历史
  POST /s2    multipart: file=rgb(jpg) + data{instruction, infer("1"/"0")}
              infer=1: s2_step出决策 -> {"type":"pixel","pixel":[y,x]} | {"type":"actions","actions":[..]} | {"type":"stop"}
              infer=0: step_no_infer只补历史帧 -> {"type":"ok"}
启动:
  CUDA_VISIBLE_DEVICES=7 /home/DongBaorong/micromamba/envs/internvla/bin/python dualvlm_slow_server.py --port 8155
"""
import argparse
import io
import os
import types

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from transformers import BitsAndBytesConfig

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

_orig_fp = InternVLAN1ForCausalLM.from_pretrained


def _fp_4bit(cls, *a, **kw):
    kw['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    kw['attn_implementation'] = 'sdpa'
    return _orig_fp.__func__(cls, *a, **kw)


# 1011: 4bit 加载在本机会在 `folio_wait_bit_common` 死锁 ⇒ 加开关,可用 fp16 绕开
if os.environ.get('S2_NO4BIT', '0') != '1':
    InternVLAN1ForCausalLM.from_pretrained = classmethod(_fp_4bit)

from internnav.model.basemodel.internvla_n1.internvla_n1_policy import InternVLAN1Net  # noqa: E402
from internnav.model import get_config  # noqa: E402

MODEL_PATH = '/data/DongBaorong/BEVLOOK/benchmarks/checkpoints/InternVLA-N1-DualVLN'
app = FastAPI()
NET = None


def make_slow(device='cuda:0'):
    ms = {'env_num': 1, 'sim_num': 1, 'policy_name': 'InternVLAN1_Policy',
          'state_encoder': None, 'model_path': MODEL_PATH, 'camera_intrinsic': None,
          'width': 640, 'height': 480, 'hfov': 79, 'resize_w': 384, 'resize_h': 384,
          'max_new_tokens': 1024, 'num_frames': 32, 'num_history': 8,
          'num_future_steps': 4, 'device': device, 'predict_step_nums': 32,
          'continuous_traj': True, 'infer_mode': 'partial_async',
          'vis_debug': False, 'vis_debug_path': './logs/hybrid_vis'}
    cfg = get_config('InternVLAN1_Policy')(model_cfg={'model': ms})
    net = InternVLAN1Net(cfg)
    net.model.generate_latents = lambda *a, **k: None   # 混合实验不要扩散latents
    return net


@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'InternVLA-N1-DualVLN-S2-4bit'}


@app.post('/reset')
def reset():
    NET.reset()
    return {'status': 'success'}


@app.post('/s2')
async def s2(file: UploadFile = File(...), instruction: str = Form(...),
             infer: str = Form('1')):
    rgb = np.array(Image.open(io.BytesIO(await file.read())).convert('RGB'))
    if infer != '1':
        NET.step_no_infer(rgb, None, None)
        return {'type': 'ok'}
    out = NET.s2_step(rgb, None, None, instruction, None)
    llm = NET.llm_output
    if out.output_pixel is not None:
        return {'type': 'pixel', 'pixel': [int(out.output_pixel[0]), int(out.output_pixel[1])],
                'raw': llm}
    if out.output_action:
        acts = [int(a) for a in out.output_action]
        if acts and acts[0] == 0:
            return {'type': 'stop', 'raw': llm}
        return {'type': 'actions', 'actions': acts, 'raw': llm}
    return {'type': 'none', 'raw': llm}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8155)
    cli = ap.parse_args()
    NET = make_slow('cuda:0')
    print('[slow] DualVLN S2 4bit ready', flush=True)
    uvicorn.run(app, host='0.0.0.0', port=cli.port, log_level='warning')
