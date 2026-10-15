#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""慢脑标注 jsonl → LLaMA-Factory qwen2_vl sharegpt 训练数据。

流程：读 slowbrain 标注 → 解码每帧 npz 的 rgb 存 jpg → 生成 sharegpt json + dataset_info 注册。

输出格式（单帧版，验证「图像→坐标」接口）：
  human:  "<image>{instruction}"
  gpt:    "[TARGET] {nx} {ny}"    (可见) 或 "[TARGET] -1.0 -1.0" (不可见)

用法(infinitevl env, 有 cv2):
  python3 to_llamafactory.py --annot slowbrain_annot.jsonl --out slowbrain_data --max_samples 2000
"""
import os
import sys
import json
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)


def decode_rgb(npz_path):
    d = np.load(npz_path)
    return d["rgb"]  # HxWx3 uint8


def fmt_coord(nx, ny, visible):
    if visible:
        return f"[TARGET] {nx:.4f} {ny:.4f}"
    return "[TARGET] -1.0 -1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True, help="gen_slowbrain_from_dense.py 输出的 jsonl")
    ap.add_argument("--out", default="slowbrain_data", help="输出目录(images/ + json)")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--out-json", default=None, help="输出的 sharegpt json 名(默认 slowbrain_train.json)")
    ap.add_argument("--skip-existing", action="store_true", help="jpg 已存在则跳过写盘(内容同源,幂等)")
    ap.add_argument("--ds-name", default="slowbrain_train", help="dataset_info.json 里注册的数据集名")
    cli = ap.parse_args()

    import cv2

    img_dir = os.path.join(cli.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    samples = []
    seen = set()
    n_skip = 0
    with open(cli.annot, encoding="utf-8") as f:
        for line in f:
            if cli.max_samples and len(samples) >= cli.max_samples:
                break
            a = json.loads(line)
            npz = a["frame"]
            if not os.path.exists(npz):
                n_skip += 1
                continue
            key = (a["episode_id"], os.path.basename(npz))
            if key in seen:
                continue
            seen.add(key)
            # 解码 rgb 存 jpg
            jpg_name = f"ep{a['episode_id']}_{os.path.basename(npz)[:-4]}.jpg"
            jpg_path = os.path.join(img_dir, jpg_name)
            if not (cli.skip_existing and os.path.exists(jpg_path)):
                rgb = decode_rgb(npz)
                cv2.imwrite(jpg_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, cli.jpeg_quality])
            # 相对路径（LLaMA-Factory 相对 dataset_dir）
            rel = os.path.join("images", jpg_name)
            instruction = a["instruction"].strip()
            samples.append({
                "messages": [
                    {"role": "user",
                     "content": f"<image>你是导航机器人。指令：{instruction}\n"
                                f"请输出目标在当前画面中的归一化像素坐标，格式 [TARGET] x y，"
                                f"若目标不在画面内输出 [TARGET] -1.0 -1.0。"},
                    {"role": "assistant",
                     "content": fmt_coord(a["goal_nx"], a["goal_ny"], a["goal_visible"])},
                ],
                "images": [rel],
            })

    # 写 sharegpt json
    out_json = os.path.join(cli.out, cli.out_json or "slowbrain_train.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # dataset_info.json：**合并**写入(不覆盖已有数据集注册)
    ds_path = os.path.join(cli.out, "dataset_info.json")
    ds_info = {}
    if os.path.exists(ds_path):
        with open(ds_path, encoding="utf-8") as f:
            ds_info = json.load(f)
    ds_info[cli.ds_name] = {
        "file_name": cli.out_json or "slowbrain_train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant"},
    }
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(ds_info, f, ensure_ascii=False, indent=2)

    n_vis = sum(1 for s in samples if "-1.0 -1.0" not in s["messages"][-1]["content"])
    print(f"==== done: {len(samples)} samples (vis={n_vis} inv={len(samples) - n_vis}), "
          f"skip={n_skip} -> {cli.out} ====")


if __name__ == "__main__":
    main()
