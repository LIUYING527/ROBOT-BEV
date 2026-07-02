#!/usr/bin/env python
"""手工产 ArtiFixer 需要的 caption.h5,跳过默认的 30B Qwen3-VL 描述模型。
用固定 caption + 本地 Wan2.1 的 UMT5 text encoder 编码成 bf16(存uint16) embedding,
写成 run_inference 兼容的 h5(load_encoded_prompt 只读第一个 dataset key)。

依据 data_processing/captioning/generate_captions.py 的 generate_text_embedding/存储格式。
"""
import sys, argparse
from pathlib import Path

AF = Path(__file__).resolve().parent / "third_party/artifixer"
sys.path.insert(0, str(AF))
import h5py
import torch
from data_processing.captioning.generate_captions import (
    get_text_encoder_and_tokenizer,
    generate_text_embedding,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wan_dir", default="/data/DongBaorong/Wan2.1-T2V-14B-Diffusers")
    ap.add_argument("--dataset_name", default="00000.png", help="h5 dataset key(无所谓,只读第一个)")
    ap.add_argument("--caption", default=(
        "A photorealistic walkthrough of an indoor industrial inspection corridor, "
        "with metal pipes, valves, pumps and gauges along concrete walls, a smooth "
        "concrete floor, even ambient lighting, sharp and detailed."))
    ap.add_argument("--max_seq", type=int, default=512)
    args = ap.parse_args()

    print("loading UMT5 text encoder from", args.wan_dir)
    text_encoder, tokenizer = get_text_encoder_and_tokenizer(args.wan_dir)
    emb = generate_text_embedding(args.caption, text_encoder, tokenizer, args.max_seq)
    print("embedding shape:", emb.shape, "dtype:", emb.dtype)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as hf:
        ds = hf.create_dataset(args.dataset_name, data=emb)
        ds.attrs["caption"] = args.caption
        ds.attrs["image_indices"] = [0]
    print("wrote", out)


if __name__ == "__main__":
    main()
