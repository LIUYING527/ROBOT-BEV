#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""slowbrain_server_ego_stream.py —— 流式/长记忆版慢脑 server（1004）

## 与 slowbrain_server_ego.py 的唯一区别
- 原版 = **无状态**：client 每步传 8 帧，server 当全新 prompt 编码 ⇒ **丢掉全部隐状态**
- 本版 = **有状态**：只消费 `frames[-1]`（当前帧）进持久 cache，历史由 cache 承载
  ⇒ 记忆覆盖整段 episode、每步只编码 1 帧（吞吐 ~8x）、"span 40 vs 0"的口径问题自动消失

## client 约定（eval_r2r_escape_ego.py 已加 `ep` 字段）
    POST /locate {"frames":[...], "instruction": str, "ep": str}
  - `ep` 变化 ⇒ 重置 cache（新 episode）
  - 只消费 frames[-1]

## 前置（1004 修的）
必须用 **streaming 代码目录**（见 [[streaming-impl-unblocked-1004]]）：
    benchmarks/checkpoints/InfiniteVL-Streaming-FullSpan40   ← streaming 代码 + 微调权重

用法: CUDA_VISIBLE_DEVICES=0 python slowbrain_server_ego_stream.py \
        --model /data/DongBaorong/BEVLOOK/benchmarks/checkpoints/InfiniteVL-Streaming-FullSpan40 --port 8981
"""
import argparse
import base64
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

TARGET_RE = re.compile(r"\[TARGET\]\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)")


def parse_ego(text):
    """`[TARGET] fwd lat` → (fwd, lat)；解析失败返回 None。"""
    m = TARGET_RE.search(text or '')
    return (float(m.group(1)), float(m.group(2))) if m else None


def is_stop(text):
    return 'STOP' in (text or '').upper()


def clone_cache(model, src):
    """深拷贝推理 cache（照 InfiniteVL 官方 demo），使 Q&A 不污染主历史。"""
    dst = model.allocate_inference_cache(batch_size=1)
    for dl, sl in zip(dst.layers, src.layers):
        if getattr(sl, 'is_sliding', False) and hasattr(sl, '_buf_keys'):
            dl.size = sl.size
            dl.cumulative_length = sl.cumulative_length
            if getattr(dl, 'capacity', 0) > 0 and sl.size > 0:
                L = sl.size
                dl._buf_keys[:, :, :L, :].copy_(sl._buf_keys[:, :, :L, :])
                dl._buf_values[:, :, :L, :].copy_(sl._buf_values[:, :, :L, :])
                dl.keys = dl._buf_keys[:, :, :L, :]
                dl.values = dl._buf_values[:, :, :L, :]
            elif getattr(dl, 'capacity', 0) > 0:
                dl.keys = dl._buf_keys[:, :, :0, :]
                dl.values = dl._buf_values[:, :, :0, :]
        elif hasattr(sl, 'recurrent_state') and hasattr(sl, 'conv_state_q'):
            if getattr(sl, 'conv_state_q', None) is not None:
                dl.conv_state_q.copy_(sl.conv_state_q)
                dl.conv_state_k.copy_(sl.conv_state_k)
                dl.conv_state_v.copy_(sl.conv_state_v)
            if getattr(sl, 'recurrent_state', None) is not None:
                dl.recurrent_state.copy_(sl.recurrent_state)
            dl.seq_len = sl.seq_len
            dl.start = sl.start
    return dst


class StreamLocator:
    def __init__(self, model_path, max_new=40):
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f'[stream-server] 加载 {model_path} …', flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map=None).cuda().eval()
        self.proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if not hasattr(self.model.visual, 'set_graph_bucket'):
            raise RuntimeError('加载到 standard 而非 streaming ⇒ 换用 streaming 模型目录')
        self.max_new = max_new
        self.cur_ep = None
        b = self._img_batch(Image.fromarray(np.zeros((480, 640, 3), np.uint8)))
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        self.model.visual.set_graph_bucket(b['image_grid_thw'])
        self.model.visual.precompute_window_buffers()
        self.model.visual.precompute_full_cu_seqlens()
        self.img_tok = int((b['input_ids'][0] == self.model.config.image_token_id).sum())
        print(f'[stream-server] 就绪 img_tok={self.img_tok}', flush=True)
        self.reset()

    def _img_batch(self, im):
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return self.proc(text=[text], images=[im], return_tensors='pt')

    def reset(self):
        self.cache = self.model.allocate_inference_cache(batch_size=1)
        self.cum = 0
        self.pmax = -1

    def _query_ids(self, instruction):
        qtext = (f"你是导航机器人。指令：{instruction}\n"
                 f"请输出目标点相对于你当前位置与朝向的相对坐标，格式 [TARGET] 前向 横向："
                 f"前向 = 沿你正前方的距离（米），横向 = 左为正、右为负（米）。\n"
                 f"若已到达指令描述的终点，输出 [TARGET] STOP。")
        s = self.proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': qtext}]}],
                                          tokenize=False, add_generation_prompt=True)
        b = self.proc(text=[s], images=None, return_tensors='pt')
        vend = self.model.config.vision_end_token_id
        return torch.cat([torch.full((1, 1), vend, dtype=b['input_ids'].dtype),
                          b['input_ids']], dim=1).to('cuda')

    @torch.inference_mode()
    def locate(self, frames, instruction, ep):
        if ep != self.cur_ep:
            self.reset()
            self.cur_ep = ep
        b = self._img_batch(Image.fromarray(frames[-1]))
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        full = b['input_ids']
        vi = (full[0] == self.model.config.vision_start_token_id).nonzero(as_tuple=False)[0].item()
        ids = full[:, vi + 1: vi + 1 + self.img_tok]
        n = ids.shape[1]
        cp = torch.arange(self.cum, self.cum + n, dtype=torch.long, device='cuda').view(1, -1)
        pos = (self.pmax + 1 + torch.arange(n, device='cuda')).view(1, 1, -1).expand(3, 1, -1).float()
        out = self.model(input_ids=ids, pixel_values=b['pixel_values'], image_grid_thw=b['image_grid_thw'],
                         position_ids=pos, past_key_values=self.cache, cache_position=cp,
                         use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        self.cum += n
        self.pmax += n

        qa = clone_cache(self.model, self.cache)
        q = self._query_ids(instruction)
        qn = q.shape[1]
        qcp = torch.arange(self.cum, self.cum + qn, dtype=torch.long, device='cuda').view(1, -1)
        qpos = (self.pmax + 1 + torch.arange(qn, device='cuda')).view(1, 1, -1).expand(3, 1, -1).float()
        qo = self.model(input_ids=q, past_key_values=qa, cache_position=qcp,
                        position_ids=qpos, use_cache=True, return_dict=True)
        nxt = torch.argmax(qo.logits[:, -1, :], dim=-1, keepdim=True)
        toks = [nxt]
        for k in range(self.max_new):
            if nxt.item() == self.proc.tokenizer.eos_token_id:
                break
            scp = torch.tensor([self.cum + qn + k], dtype=torch.long, device='cuda')
            sp = torch.full((3, 1, 1), float(self.pmax + qn + k + 1), device='cuda')
            qo = self.model(input_ids=nxt, past_key_values=qa, cache_position=scp,
                            position_ids=sp, use_cache=True, return_dict=True)
            nxt = torch.argmax(qo.logits[:, -1, :], dim=-1, keepdim=True)
            toks.append(nxt)
        return self.proc.tokenizer.decode(torch.cat(toks, dim=1)[0], skip_special_tokens=True)


LOCATOR = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_POST(self):
        t0 = time.time()
        try:
            n = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(n))
            frames = [np.frombuffer(base64.b64decode(f['rgb']), dtype=np.uint8).reshape(f['shape'])
                      for f in data['frames']]
            ep = str(data.get('ep', 'default'))
            text = LOCATOR.locate(frames, data['instruction'], ep)
            c = parse_ego(text)
            resp = {'visible': c is not None,
                    'ego_fwd': c[0] if c else None, 'ego_lat': c[1] if c else None,
                    'stop': is_stop(text), 'raw': text}
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            print(f'[stream] {time.time() - t0:.2f}s ep={ep} nframes_in={len(frames)} '
                  f'cum={LOCATOR.cum} vis={c is not None} '
                  f'ego={None if c is None else (round(c[0], 2), round(c[1], 2))} '
                  f'stop={is_stop(text)} raw={text!r}', flush=True)
        except BrokenPipeError:
            print(f'[stream] broken-pipe at {time.time() - t0:.2f}s', flush=True)
        except Exception as e:
            print(f'[stream] error: {type(e).__name__}: {e}', flush=True)
            try:
                self.send_response(500)
                self.end_headers()
            except Exception:
                pass

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='streaming 模型目录（见文件头）')
    ap.add_argument('--port', type=int, default=8981)
    cli = ap.parse_args()
    global LOCATOR
    LOCATOR = StreamLocator(cli.model)
    print(f'[stream-server] listening :{cli.port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', cli.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
