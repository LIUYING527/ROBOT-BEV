"""本地 Qwen3.5-4B VLM HTTP 服务(替代 10.0.0.18 自研VLM服务不在线时)。
契约与 sim_fullchain.vlm_api 期望一致:
  POST /api/generate  multipart: files[]=images(jpg) + data{prompt,max_new_tokens,do_sample} -> {"answer":str,"meta":{}}
  GET  /health -> {"status":"ok","model":...}
⭐Qwen3.5默认thinking, 分类/问答任务 enable_thinking=False 直出答案 + 去<think>(与stage_vlm_slowbrain一致)。
用法(artifixer env, 空卡):
  CUDA_VISIBLE_DEVICES=3 /home/DongBaorong/micromamba/envs/artifixer/bin/python \
    scripts/vlm_server_local.py --port 8152
"""
import argparse, io, time, os
import torch, transformers
from transformers import AutoProcessor
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import uvicorn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "arch_e2e", "vlm", "Qwen3.5-4B")

app = FastAPI()
PROC = None
MODEL = None


def load():
    global PROC, MODEL
    t0 = time.time()
    PROC = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    MODEL = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    print(f"[vlm] 加载 {MODEL_DIR} 完成 {time.time()-t0:.1f}s", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": "Qwen3.5-4B"}


@app.post("/api/generate")
async def generate(images: List[UploadFile] = File(default=[]),
                   prompt: str = Form(...),
                   max_new_tokens: str = Form("32"),
                   do_sample: str = Form("false")):
    imgs = []
    for f in images:
        b = await f.read()
        imgs.append(Image.open(io.BytesIO(b)).convert("RGB"))
    content = [{"type": "image", "image": im} for im in imgs]
    content.append({"type": "text", "text": prompt})
    msgs = [{"role": "user", "content": content}]
    text = PROC.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)   # 关thinking直出答案
    inp = PROC(text=[text], images=imgs if imgs else None, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        gen = MODEL.generate(**inp, max_new_tokens=int(max_new_tokens),
                             do_sample=(str(do_sample).lower() == "true"))
    out = PROC.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    if "</think>" in out:                                    # 万一仍带thinking去掉
        out = out.split("</think>", 1)[1].strip()
    return {"answer": out, "meta": {"dt": round(time.time() - t0, 2), "n_img": len(imgs)}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8152)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    load()
    print(f"[vlm] 服务启动 http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
