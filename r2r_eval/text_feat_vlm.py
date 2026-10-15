#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⭐1006 真·语义编码器：用 InfiniteVL-4B 的【学习到的词嵌入表】做句向量。

动机: 原 text_feat.py 是 crc32 词袋哈希 + 7 个硬编码动作槽 —— 它不承载语义。
      用户坚持"语义注入 DD 一定有效" ⇒ 先排除"编码器太差"这个解释。

零 GPU 设计(遵守"不许搞崩慢脑训练"):
  · 只从 safetensors 里【懒加载 model.embed_tokens.weight】(151936×2048 bf16 ≈ 622MB, 读进内存)
  · 不加载 transformer 层、不碰任何 GPU
  · 逐条算: tokenize → 查表 → 去 padding 后按 token 平均 → L2 归一化
"""
import json, os
import numpy as np

_MODEL_DIR = os.environ.get(
    'PWS_VLM_DIR', '/data/DongBaorong/BEVLOOK/benchmarks/checkpoints/InfiniteVL-4B')
_DIM = 2048
_E = None            # (vocab, dim) float32 惰性加载
_TOK = None          # tokenizer
_PROJ = None         # (2048,768) 固定随机投影


def _proj():
    global _PROJ
    if _PROJ is None:
        rng = np.random.RandomState(20261006)          # 固定种子 ⇒ 可复现
        _PROJ = (rng.randn(_DIM, 768) / np.sqrt(768.0)).astype(np.float32)
    return _PROJ


def _load():
    global _E, _TOK
    if _E is not None:
        return
    from transformers import AutoTokenizer
    from safetensors import safe_open
    # ⚠️ 必须 use_fast=False: 该权重目录的 tokenizer.json 是坏的
    #   ('data did not match any variant of untagged enum ModelWrapper') => 退回 Python tokenizer
    _TOK = AutoTokenizer.from_pretrained(_MODEL_DIR, trust_remote_code=True, use_fast=False)
    idx = json.load(open(os.path.join(_MODEL_DIR, 'model.safetensors.index.json')))['weight_map']
    key = [k for k in idx if k.endswith('embed_tokens.weight')][0]
    with safe_open(os.path.join(_MODEL_DIR, idx[key]), framework='pt', device='cpu') as f:
        _E = f.get_tensor(key).float().numpy()          # (151936,2048) bf16->f32, 一次性
    # ⭐1006 关键优化: 投影【预先作用到整张表】(一次性 ~239GFLOP), 之后每次调用只做
    #   "查表+求平均", 完全没有矩阵乘 ⇒ 避免 BLAS 为每次微小 matmul 开几十个线程
    #   (实测: 逐次 matmul 时 dataloader worker 各烧 2110% CPU, 训练 12 分钟出不了 step)
    import os as _os2
    _old = _os2.environ.get('OMP_NUM_THREADS'); _os2.environ['OMP_NUM_THREADS'] = '1'
    _E = (_E @ _proj()).astype(np.float32)          # (151936,768)
    if _old is None: _os2.environ.pop('OMP_NUM_THREADS', None)
    else: _os2.environ['OMP_NUM_THREADS'] = _old
    print(f'[text_feat_vlm] 已加载 embedding {_E.shape} ({_E.nbytes/1e6:.0f}MB, 纯 CPU)', flush=True)


_CACHE = {}


def instr_feature(text, dim=_DIM):
    """单条指令 -> (dim,) float32, L2 归一化。

    ⭐1006: 加 memo 缓存 —— 训练时 __getitem__ 每个样本都调一次(272k 样本/epoch),
    而指令只有约 1~2 万条唯一值 ⇒ 不缓存的话慢 tokenizer 会成为瓶颈。
    这是纯函数(同输入同输出), 缓存不改变任何语义。
    """
    c = _CACHE.get(text)
    if c is not None:
        return c
    _load()
    ids = _TOK(text, add_special_tokens=False)['input_ids']
    if not ids:
        return np.zeros(dim, np.float32)
    v = _E[np.asarray(ids, np.int64)].mean(0)          # (768,) 投影已在加载时预先作用
    n = float(np.linalg.norm(v))
    out = (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)
    if len(_CACHE) < 200000:
        _CACHE[text] = out
    return out


def instr_features(texts, dim=_DIM):
    """批量 -> (N, dim)"""
    if not texts:
        return np.zeros((0, dim), np.float32)
    return np.stack([instr_feature(t, dim) for t in texts], 0)


if __name__ == '__main__':
    import time
    ts = ["Exit the bedroom and turn left down the hallway.",
          "Turn right and walk past the kitchen into the living room.",
          "Go upstairs and stop near the bathroom door."]
    t0 = time.time(); vs = instr_features(ts); dt = time.time() - t0
    print('shape', vs.shape, f'首次(含加载) {dt:.1f}s')
    t0 = time.time(); vs2 = instr_features(ts); print(f'第二次(纯计算) {time.time()-t0:.3f}s')
    import numpy as _np
    from text_feat import instr_features as oldf
    ov = oldf(ts)
    print('新编码器 相似度矩阵(应明显有结构):')
    S = vs @ vs.T
    print(_np.round(S, 3))
    print('旧(crc32哈希) 相似度矩阵(应几乎无结构/近正交):')
    print(_np.round(ov @ ov.T, 3))
