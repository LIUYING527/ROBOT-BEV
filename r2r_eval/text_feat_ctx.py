#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""⭐1007 真·上下文句向量查表: 由 InfiniteVL-4B 的【完整 36 层】前向产生(带词序/上下文),
   与 text_feat.py(词袋) / text_feat_vlm.py(词向量平均) 的区别就在这。"""
import json, os
import numpy as np
DIM = 2048
_NPY = os.environ.get('PWS_CTX_NPY', '/tmp/instr_ctx_merged.npy')
_IDX = os.environ.get('PWS_CTX_IDX', '/tmp/instr_ctx_merged_index.json')
_E = None; _D = None; _MISS = [0]
def _load():
    global _E, _D
    if _E is None:
        _E = np.load(_NPY).astype(np.float32)
        _D = {k: i for i, k in enumerate(json.load(open(_IDX, encoding='utf-8')))}
        print(f'[text_feat_ctx] {_E.shape} 查表就绪', flush=True)
def instr_feature(text, dim=DIM):
    _load()
    i = _D.get(text)
    if i is None:
        base = text.strip()
        i = _D.get(base)
    if i is not None:
        return _E[i]
    # ⭐1007 兜底(修正版): 必须返回【与 _E 同维(2048)】的向量。
    #   ⚠️ 原先退化为 text_feat_vlm(768 维) ⇒ 与 text_proj(2048) 维度不符 ⇒ 崩。
    #   这里返回 2048 维零向量, 并计数 + 只告警一次(不漏报)。
    global _MISS
    _MISS[0] += 1
    if _MISS[0] <= 3:
        print(f'[text_feat_ctx] ⚠️ 索引未覆盖第 {_MISS[0]} 条指令, 用零向量: {text[:60]!r}', flush=True)
    return np.zeros(_E.shape[1], np.float32)
def instr_features(texts, dim=DIM):
    if not texts: return np.zeros((0, dim), np.float32)
    return np.stack([instr_feature(t, dim) for t in texts], 0)
