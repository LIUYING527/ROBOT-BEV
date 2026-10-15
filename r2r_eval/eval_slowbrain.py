#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""慢脑评测 —— 目标定位命中率（不是 SR）。

指标（文档 §0 正确口径）：
  1. 目标定位命中率：预测像素 vs GT 投影像素的归一化欧氏距离 < hit_radius。
  2. 不可见判定：GT 不可见(goal=(-1,-1))时，模型输出「不可见」也算命中。

用法(infinitevl env):
  python eval_slowbrain.py --model <权重路径> [--lora <adapter>] \
      --annot slowbrain_annot.jsonl --max_samples 200 --hit_radius 0.05
"""
import os
import re
import json
import argparse

import numpy as np
import torch

INVIS_IDS = set()   # 0918: 见 load_model


def parse_coord(text):
    """从模型输出解析归一化坐标；返回 (nx, ny, not_visible)。"""
    t = text.strip()
    if "not visible" in t.lower() or "not in" in t.lower() or "不可见" in t or "不在画面" in t:
        return None, None, True
    m = re.search(r"\[TARGET\]\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)", t)
    if not m:
        m = re.search(r"(-?\d+\.\d{2,})\s+(-?\d+\.\d{2,})", t)
    if m:
        nx, ny = float(m.group(1)), float(m.group(2))
        if nx < 0 or ny < 0:
            return None, None, True
        return nx, ny, False
    return None, None, None


def load_model(model_path, lora_path=None):
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="flash_attention_2")
    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # ⭐0918 预计算「以 '-' 开头的 token 集合」—— 第一个生成 token 落在这里 = 模型判"不可见"
    # (训练/推理的输出都是 `[TARGET] x y`, 不可见时 x = -1.0)
    global INVIS_IDS
    _tok = processor.tokenizer
    INVIS_IDS = set()
    # ⚠️ 必须带**前导空格**的变体: "[TARGET] -1.0" 在坐标位上的 token 是 `" -"`(id 481),
    # 而不是 `"-"`(id 12)。只收 '-1.0' 的第一个 token 会收成 12 ⇒ p_invis 恒 0(踩过)。
    for _s in ('-1.0', '-1', '-', '-0.5', '-0', ' -1.0', ' -1', ' -', ' -0.5', ' -0'):
        _ids = _tok.encode(_s, add_special_tokens=False)
        if _ids:
            INVIS_IDS.add(int(_ids[0]))
    print(f'[p_invis] 负号 token id = {sorted(INVIS_IDS)}', flush=True)
    return model, processor, process_vision_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--annot", required=True)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--hit_radius", type=float, default=0.05, help="归一化像素命中半径(0.05≈32px/640)")
    ap.add_argument("--nframes", type=int, default=1, help="多帧输入帧数(1=单帧; >1=当前帧+前K-1历史帧)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default=None, help="逐样本预测转储 jsonl(供 analyze_localization.py 诊断)")
    ap.add_argument("--with-dist", action="store_true",
                    help="prompt 里带「目标距离你约 X 米」先验(须与 slowbrain_train_dist 训练数据逐字一致)")
    ap.add_argument("--two-pass", action="store_true",
                    help="两遍自举: pass1 不带距离 -> 用输出 y 反推距离 -> pass2 带该距离(闭环可行的验证)")
    ap.add_argument("--dist-noise", type=float, default=0.0,
                    help="对 dist_to_goal 施加相对高斯噪声(模拟自估距离误差), 0=不施加")
    ap.add_argument("--dist-fixed", type=float, default=None,
                    help="用固定值代替 GT 距离(模拟闭环里只能猜一个距离、且猜得不准的处境)")
    ap.add_argument("--y2d", default="/data/DongBaorong/BEVLOOK/tools/dist_prior/y2d.npz",
                    help="y->距离 映射表(npz: ctr, med), 由训练数据拟合; --two-pass 时必需")
    ap.add_argument("--pass1-dist", type=float, default=None,
                    help="两遍模式 pass1 用的固定先验距离(米)。缺省=不带距离行(分布外但最保守); "
                         "给值则 prompt 格式与训练一致(闭环第一步可先用一个猜测距离)")
    cli = ap.parse_args()
    fdump = open(cli.dump, "w", encoding="utf-8") if cli.dump else None

    Y2D_CTR = Y2D_MED = None
    if cli.two_pass:
        _z = np.load(cli.y2d)
        Y2D_CTR, Y2D_MED = _z["ctr"], _z["med"]

    model, processor, process_vision_info = load_model(cli.model, cli.lora)
    import cv2

    samples = []
    with open(cli.annot, encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            if os.path.exists(a["frame"]):
                samples.append(a)

    def frame_num(fp):
        m = re.search(r'f(\d+)', os.path.basename(fp))
        return int(m.group(1)) if m else 0

    # 按 episode 分组 + 帧排序
    from collections import defaultdict
    by_ep = defaultdict(list)
    for a in samples:
        by_ep[a["episode_id"]].append(a)
    for ep in by_ep:
        by_ep[ep].sort(key=lambda a: frame_num(a["frame"]))

    # 构造「当前帧 + 前 K-1 历史帧」候选（当前帧需有足够历史帧）
    rng = np.random.RandomState(cli.seed)
    candidates = []
    for ep, frs in by_ep.items():
        for i in range(cli.nframes - 1, len(frs)):
            candidates.append(frs[i - cli.nframes + 1: i + 1])  # K 帧 [历史..当前]
    if len(candidates) > cli.max_samples:
        idx = rng.choice(len(candidates), cli.max_samples, replace=False)
        candidates = [candidates[i] for i in idx]

    def infer(q, img_paths):
        """单次推理: 给定 prompt 文本与图片路径, 返回模型输出文本。"""
        content = [{"type": "image", "image": f"file://{p}"} for p in img_paths]
        content.append({"type": "text", "text": q})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                  return_dict_in_generate=True, output_scores=True)
        gids = _out.sequences
        # ⭐0918: 取「第一个生成 token」的 softmax 概率, 按「是否以 '-' 开头」求和 = P(判不可见)。
        # 动机: 慢脑在闭环里「不可见却报可见」达 58%, 而它是**贪心解码**、没有阈值可调。
        # 但日志概率给出一个连续分数 ⇒ 可以在推理端调阈值把误报压下去。
        p_inv = None
        try:
            # ⚠️ 不能用 scores[0]: 输出的第一个 token 是 "[TARGET]" 的开头, 不是坐标。
            # (实测这样取到的是 p([ )≈1, 给的 p_invis 恒 0, 四组全是 0.000 —— 无效分数。)
            # 先逐步解码定位 "[TARGET]" 结束的位置, 下一个 token 才是坐标的第一个 token。
            _tok = processor.tokenizer
            _gid = _out.sequences[0][inputs.input_ids.shape[1]:]
            _dec, _k = '', None
            for _i, _t in enumerate(_gid):
                _dec += _tok.decode([int(_t)], skip_special_tokens=True)
                if '[TARGET]' in _dec:
                    _k = _i + 1          # 坐标的起始 token
                    break
            if _k is not None and _k < len(_out.scores):
                _pr = torch.softmax(_out.scores[_k][0].float(), dim=-1)
                p_inv = float(sum(_pr[i] for i in INVIS_IDS if i < _pr.shape[0]))
        except Exception as _e:
            print(f'[warn] p_invis 计算失败: {_e}', flush=True)
        if not hasattr(infer, '_p'):
            infer._p = None
        infer._p = p_inv
        gids_trimmed = [o[len(t):] for t, o in zip(inputs.input_ids, gids)]
        return processor.batch_decode(gids_trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]

    n_hit = 0
    n_vis = 0
    n_invis = 0
    n_vis_hit = 0
    n_invis_hit = 0
    n_bad = 0
    n_pass2 = 0
    errs = []

    for i, frames in enumerate(candidates):
        cur = frames[-1]  # 当前帧（最后一帧）
        instruction = cur["instruction"].strip()
        # K 帧转临时 jpg
        img_paths = []
        for j, fr in enumerate(frames):
            rgb = np.load(fr["frame"])["rgb"]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ip = f"/tmp/_sb_eval_{os.getpid()}_{j}.jpg"
            cv2.imwrite(ip, bgr)
            img_paths.append(ip)
        # ⚠️ prompt 必须与训练数据逐字一致(to_llamafactory.py)：
        #    单帧训练用「当前画面」；多帧训练(gen_multiframe_localize.py)用「当前（最后一帧）画面」
        frame_desc = "当前画面" if len(frames) == 1 else "当前（最后一帧）画面"

        def make_q(dist_val):
            """dist_val=None 时不插距离行 —— 即 pass1 的 prompt(与 slowbrain_train_dist 的训练格式不同, 闭环第一步的真实处境)。"""
            dl = f"目标距离你约 {dist_val:.1f} 米。\n" if dist_val is not None else ""
            return (f"{'<image>' * len(frames)}你是导航机器人。指令：{instruction}\n"
                    f"{dl}"
                    f"请输出目标在{frame_desc}中的归一化像素坐标，格式 [TARGET] x y，"
                    f"若目标不在画面内输出 [TARGET] -1.0 -1.0。")

        d_used = None
        if cli.two_pass:
            # pass1 不带距离(闭环第一步本来就没有距离) -> 用输出 y 几何反推距离 -> pass2 带该距离
            out1 = infer(make_q(cli.pass1_dist), img_paths)
            _, ny1, invis1 = parse_coord(out1)
            if invis1 or ny1 is None:
                out = out1        # 判不可见/解析失败 -> 没有 y 可反推, 保持 pass1 结果
            else:
                d_used = float(np.interp(ny1, Y2D_CTR, Y2D_MED))
                out = infer(make_q(d_used), img_paths)
                n_pass2 += 1
        else:
            if cli.with_dist and cur.get("dist_to_goal") is not None:
                dv = cur["dist_to_goal"]
                if cli.dist_fixed is not None:
                    dv = cli.dist_fixed            # 固定值代替 GT: 闭环里"只能猜一个距离"的处境
                elif cli.dist_noise > 0:
                    # 固定种子: 同一帧的扰动可复现
                    _rs = np.random.RandomState(abs(hash(str(cur.get("frame")))) % (2 ** 31))
                    dv = max(0.5, dv * (1.0 + _rs.normal(0.0, cli.dist_noise)))
                d_used = dv
            out = infer(make_q(d_used), img_paths)

        pnx, pny, p_invis = parse_coord(out)
        _p_score = getattr(infer, '_p', None)   # ⭐0918 不可见概率(连续分数)
        g_vis = cur["goal_visible"]
        if fdump is not None:
            fdump.write(json.dumps({
                "episode_id": cur.get("episode_id"), "dist": cur.get("dist_to_goal"),
                "d_used": d_used,
                "gt_vis": bool(g_vis), "gt_nx": cur.get("goal_nx"), "gt_ny": cur.get("goal_ny"),
                "pred_vis": (None if p_invis is None else (not p_invis)),
                "p_invis": _p_score,
                "pred_nx": pnx, "pred_ny": pny, "raw": out.strip()[:80],
            }, ensure_ascii=False) + "\n")
        if p_invis is None:
            n_bad += 1
            continue
        if g_vis:
            n_vis += 1
            if not p_invis:
                d = np.hypot(pnx - cur["goal_nx"], pny - cur["goal_ny"])
                errs.append(d)
                if d < cli.hit_radius:
                    n_hit += 1
                    n_vis_hit += 1
        else:
            n_invis += 1
            if p_invis:
                n_hit += 1
                n_invis_hit += 1

        if i % 25 == 0:
            print(f"[{i}/{len(candidates)}] out={out.strip()[:50]!r} gt=({cur['goal_nx']},{cur['goal_ny']})", flush=True)

    print("\n==== 慢脑定位评测 ====")
    print(f"样本 {len(candidates)} | 可见 {n_vis} | 不可见 {n_invis} | 解析失败 {n_bad}")
    print(f"总命中率: {n_hit}/{len(candidates) - n_bad} = {n_hit / max(1, len(candidates) - n_bad):.4f}")
    if n_vis:
        print(f"可见样本命中率: {n_vis_hit}/{n_vis} = {n_vis_hit / n_vis:.4f}")
        if errs:
            print(f"可见样本像素误差: mean={np.mean(errs):.4f} median={np.median(errs):.4f}")
    if n_invis:
        print(f"不可见样本命中率: {n_invis_hit}/{n_invis} = {n_invis_hit / n_invis:.4f}")
    if cli.two_pass:
        print(f"两遍自举: pass2 实际执行 {n_pass2} 次 (其余在 pass1 判不可见/解析失败, 保持 pass1 结果)")


if __name__ == "__main__":
    main()
