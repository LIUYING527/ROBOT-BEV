#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指令 → 固定维度特征向量（喂给快脑 DD 的 text_proj）。

设计原则（都是被现实约束逼出来的）：
  ① **零外部依赖**：22 上没有 BERT/T5/CLIP，HF 缓存为空且无外网
     ⇒ 不用预训练编码器，用「词袋 + 动作词加权」。
  ② **确定性哈希**：用 `zlib.crc32` 而非内置 `hash()` —— 后者受 PYTHONHASHSEED 影响，
     跨进程/跨次运行不可复现（训练与评测必须一致）。
  ③ **维度 = 768**：与 `transfuser_model_v2.py` 里 `self.text_proj = nn.Linear(768, d_model)`
     对齐 ⇒ **模型结构零改动**。该层是零初始化且**从未被训练过**（训练/评测都没传过 text_feature），
     所以从现有 ckpt 续训时无需迁移它的权重。
  ④ 词袋足以区分关键差异：R2R 指令里 "go up the stairs" vs "turn left into the hallway"
     这种地标/动作差异，正是我们要验证的「语义可控性」的对象。

用法：
    from text_feat import instr_feature, instr_features
    v = instr_feature('Turn left at the couch.')      # (768,) float32, L2 归一化
    M = instr_features(['...', '...'])                 # (N, 768)
"""
import re
import zlib
import numpy as np

DIM = 768

# 动作/关系词加权：它们在 R2R 指令里承载"怎么走"的语义，比冠词/介词信息量大得多。
ACTION_WORDS = {
    'turn': 2.0, 'left': 2.0, 'right': 2.0, 'straight': 1.5, 'forward': 1.5,
    'stop': 2.0, 'enter': 2.0, 'exit': 2.0, 'go': 1.2, 'walk': 1.2,
    'up': 1.5, 'down': 1.5, 'stairs': 2.5, 'staircase': 2.5,
    'past': 1.5, 'through': 1.5, 'into': 1.5, 'out': 1.2, 'around': 1.5,
    'next': 1.2, 'until': 1.2, 'then': 1.2, 'near': 1.2, 'by': 1.2,
}
# 动作词之间共享一个"语义槽"（同一组动作词映射到同一维），让不同措辞的同义动作能对上。
ACTION_SLOTS = [
    ('turn_left', {'turn', 'left'}),
    ('turn_right', {'turn', 'right'}),
    ('forward', {'forward', 'straight', 'walk', 'go', 'move', 'head'}),
    ('ascend', {'up', 'stairs', 'staircase', 'stairway', 'steps'}),
    ('descend', {'down', 'below'}),
    ('stop', {'stop', 'near', 'until'}),
    ('through', {'through', 'past', 'by', 'around', 'into', 'enter', 'exit', 'out'}),
]
_WORD = re.compile(r"[a-z]+")


def _slot_index(name):
    """动作槽 → 固定维索引（用 crc32，跨进程稳定）。"""
    return 512 + zlib.crc32(name.encode()) % (DIM - 512)


_SLOTS = {name: _slot_index(name) for name, _ in ACTION_SLOTS}


def instr_feature(text, dim=DIM):
    """单条指令 → (dim,) float32，L2 归一化。

    ⚠️ 前 512 维 = 词袋(crc32 分桶)；后 256 维 = 动作槽。
       分区是为了让"动作槽"不被词袋淹没 —— 词袋维度多，稀疏性会稀释掉少数几个关键词。
    """
    v = np.zeros(dim, np.float32)
    if not text:
        return v
    toks = _WORD.findall(text.lower())
    for t in toks:
        v[zlib.crc32(t.encode()) % 512] += ACTION_WORDS.get(t, 1.0)
    low = set(toks)
    for name, ws in ACTION_SLOTS:
        hit = len(ws & low)
        if hit:
            v[_SLOTS[name]] += float(hit)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 0 else v


def instr_features(texts, dim=DIM):
    """批量版本 → (N, dim) float32。"""
    if not texts:
        return np.zeros((0, dim), np.float32)
    return np.stack([instr_feature(t, dim) for t in texts]).astype(np.float32)


if __name__ == '__main__':
    # 自检：同起终点不同指令必须给出不同向量；同一条必须完全可复现。
    a = 'Walk up the stairs and stop at the top.'
    b = 'Turn left into the hallway and stop by the window.'
    va, vb = instr_feature(a), instr_feature(b)
    print('dim', va.shape, '| L2', round(float(np.linalg.norm(va)), 4))
    print('cos(a,b) =', round(float(va @ vb), 4), '(应显著 < 1)')
    print('可复现:', bool(np.array_equal(va, instr_feature(a))))
    print('非零维数 a/b:', int((va > 0).sum()), '/', int((vb > 0).sum()))
