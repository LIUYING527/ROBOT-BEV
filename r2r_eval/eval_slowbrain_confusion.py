#!/usr/bin/env python3
"""慢脑定位可见性混淆矩阵分析 —— 精确量化「可见性误判」。

关键区分（R2R 提前 stop 的元凶）：
  - 误报(FP)：GT 不可见，慢脑却判可见 → 输出幻觉坐标 → 反投影近 → 提前 stop
  - 漏检(FN)：GT 可见，慢脑判不可见 → 该走却扫描
  - 定位误差：GT 可见且判可见，但坐标错

用法(infinitevl env)：
  python eval_slowbrain_confusion.py --lora <adapter> --annot <val_unseen_annot> --nframes 1
"""
import os
import re
import json
import argparse

import numpy as np
import torch

from eval_slowbrain import load_model, parse_coord

MODEL = '/data/DongBaorong/BEVLOOK/benchmarks/checkpoints/InfiniteVL-LongSFT'


def frame_num(fp):
    m = re.search(r'f(\d+)', os.path.basename(fp))
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora', required=True)
    ap.add_argument('--annot', required=True)
    ap.add_argument('--max_samples', type=int, default=500)
    ap.add_argument('--nframes', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    cli = ap.parse_args()

    model, processor, process_vision_info = load_model(MODEL, cli.lora)
    import cv2
    from collections import defaultdict

    samples = []
    with open(cli.annot, encoding='utf-8') as f:
        for line in f:
            a = json.loads(line)
            if os.path.exists(a['frame']):
                samples.append(a)

    by_ep = defaultdict(list)
    for a in samples:
        by_ep[a['episode_id']].append(a)
    for ep in by_ep:
        by_ep[ep].sort(key=lambda a: frame_num(a['frame']))

    rng = np.random.RandomState(cli.seed)
    candidates = []
    for ep, frs in by_ep.items():
        for i in range(cli.nframes - 1, len(frs)):
            candidates.append(frs[i - cli.nframes + 1: i + 1])
    if len(candidates) > cli.max_samples:
        idx = rng.choice(len(candidates), cli.max_samples, replace=False)
        candidates = [candidates[i] for i in idx]

    # 混淆矩阵统计
    n_gtvis_predvis = 0          # GT可见 & 判可见
    n_gtvis_predvis_hit = 0      # 其中定位对(误差<0.05)
    n_gtvis_predinvis = 0        # GT可见 & 判不可见(漏检)
    n_gtinvis_predinvis = 0      # GT不可见 & 判不可见(正确)
    n_gtinvis_predvis = 0        # GT不可见 & 判可见(误报←元凶)
    errs = []                    # 判可见时的定位误差
    fp_coords = []               # 误报样本的幻觉坐标(采样)

    for i, frames in enumerate(candidates):
        cur = frames[-1]
        instruction = cur['instruction'].strip()
        content = []
        for j, fr in enumerate(frames):
            rgb = np.load(fr['frame'])['rgb']
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ip = f'/tmp/_sb_cf_{os.getpid()}_{j}.jpg'
            cv2.imwrite(ip, bgr)
            content.append({'type': 'image', 'image': f'file://{ip}'})
        q = (f"{'<image>' * len(frames)}你是导航机器人。指令：{instruction}\n"
             f'请输出目标在当前（最后一帧）画面中的归一化像素坐标，格式 [TARGET] x y，'
             f'若目标不在画面内输出 [TARGET] -1.0 -1.0。')
        content.append({'type': 'text', 'text': q})
        messages = [{'role': 'user', 'content': content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            gids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        gids_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gids)]
        out = processor.batch_decode(gids_trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        pnx, pny, p_invis = parse_coord(out)
        g_vis = cur['goal_visible']

        if g_vis:
            if p_invis is False:  # 判可见
                n_gtvis_predvis += 1
                d = np.hypot(pnx - cur['goal_nx'], pny - cur['goal_ny'])
                errs.append(d)
                if d < 0.05:
                    n_gtvis_predvis_hit += 1
            else:
                n_gtvis_predinvis += 1
        else:
            if p_invis is True:
                n_gtinvis_predinvis += 1
            elif p_invis is False:  # 误报
                n_gtinvis_predvis += 1
                if len(fp_coords) < 8:
                    fp_coords.append((out.strip()[:40], round(pnx, 3), round(pny, 3)))

    n_gtvis = n_gtvis_predvis + n_gtvis_predinvis
    n_gtinvis = n_gtinvis_predvis + n_gtinvis_predinvis
    n = n_gtvis + n_gtinvis

    print('\n==== 慢脑定位可见性混淆矩阵 ====')
    print(f'样本 {n} | GT可见 {n_gtvis} | GT不可见 {n_gtinvis}')
    print(f'GT可见 → 判可见 {n_gtvis_predvis}（定位命中 {n_gtvis_predvis_hit}）| 判不可见(漏检) {n_gtvis_predinvis}')
    print(f'GT不可见 → 判不可见(正确) {n_gtinvis_predinvis} | 判可见(误报) {n_gtinvis_predvis}')
    print(f'\n[关键] 误报率(GT不可见却判可见) = {n_gtinvis_predvis}/{n_gtinvis} = {n_gtinvis_predvis / max(1, n_gtinvis):.4f}')
    print(f'[关键] 漏检率(GT可见却判不可见) = {n_gtvis_predinvis}/{n_gtvis} = {n_gtvis_predinvis / max(1, n_gtvis):.4f}')
    if errs:
        print(f'[定位] 判可见时误差: mean={np.mean(errs):.4f} median={np.median(errs):.4f}')
    if fp_coords:
        print(f'\n[误报样例] GT不可见却判可见的幻觉坐标:')
        for o, x, y in fp_coords:
            print(f'  out={o!r} -> ({x},{y})')


if __name__ == '__main__':
    main()
