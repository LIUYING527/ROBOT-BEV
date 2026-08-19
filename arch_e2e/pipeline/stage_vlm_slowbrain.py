"""VLM 慢脑 —— 语义场景/任务理解（我们自己的架构, 非学长 4 运动模式）。

⭐这一步只【验证慢脑】: 不接 DiffusionDrive 快脑(接快脑必须按我们的新任务条件通路重训)。
目标 = 让 VLM 对每帧 FPV 输出「场景描述 + 该执行的巡检任务(自拟) + 关键物体」,
few-shot 不训, 眼看它能不能把不同巡检场景【区分开】(0617 说话人4 担心 3B 幻觉 → 先验证)。

与学长的区别: 学长 = 4 运动基元 one-hot(cruise/turn/inspect/exit, 靠运动学启发式标);
我们 = 真正的语义场景/任务识别, 要的是场景间【区分度】。注入快脑的形式(离散语义集 / 语言
embedding)待验证后再拍板。

用法(artifixer env, 需先把 VLM 下到 arch_e2e/vlm/):
  CUDA_VISIBLE_DEVICES=N /home/DongBaorong/micromamba/envs/artifixer/bin/python \
    arch_e2e/pipeline/stage_vlm_slowbrain.py --session colmapjoint_all
产物: runs/vlm_slowbrain_<session>/{results.json, index.html(图库,浏览器眼看区分度)}
"""
import os, sys, glob, json, argparse
from _paths import ARCH, RUNS

VLM_DIR = os.path.join(ARCH, "vlm")

PROMPT = (
    "你是巡检机器人的场景理解模块(慢脑)。看这张机器人前视相机画面, 只输出一个 JSON:\n"
    '{"scene": "一句话描述当前场景", '
    '"objects": ["画面里与巡检相关的关键物体, 没有则空列表"], '
    '"task": "机器人此刻应执行的巡检任务, 自拟简短短语(如 直行巡航/靠近配电柜巡检/岔路口选向/接近阀门查看/走廊尽头退出)", '
    '"reason": "一句话依据"}\n'
    "只输出 JSON, 不要多余文字。"
)


def find_model_dir():
    """自动找 vlm/ 下第一个含 config.json 的模型目录。"""
    cands = [d for d in glob.glob(os.path.join(VLM_DIR, "*"))
             if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json"))]
    if not cands:
        sys.exit(f"[慢脑] 在 {VLM_DIR} 下没找到模型(需含 config.json)。\n"
                 f"  先跑 arch_e2e/download_vlm.sh 下 Qwen-VL, 或把模型放进 {VLM_DIR}/")
    return sorted(cands)[0]


def load_vlm(model_dir, device):
    import torch, transformers
    from transformers import AutoProcessor
    print(f"[慢脑] 加载 {model_dir} ...")
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model, errs = None, []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):  # 兼容 Qwen3-VL / Qwen3.5 原生多模态
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
            print(f"[慢脑] 用 {cls_name} 加载成功")
            break
        except Exception as e:
            errs.append(f"{cls_name}: {type(e).__name__}: {e}")
    if model is None:
        sys.exit("[慢脑] 模型加载失败:\n  " + "\n  ".join(errs))
    return proc, model.to(device).eval()


def infer(proc, model, img_path, device):
    import torch
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    # Qwen3.5 默认 thinking, 分类任务关掉直接出 JSON(该 kwarg 模板不认时被忽略, 无害)
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    out = proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    if "</think>" in out:                     # 万一仍带 thinking, 去掉
        out = out.split("</think>", 1)[1].strip()
    # 尽量抠出 JSON
    parsed = None
    if "{" in out and "}" in out:
        try:
            parsed = json.loads(out[out.index("{"): out.rindex("}") + 1])
        except Exception:
            parsed = None
    return out, parsed


def write_gallery(outdir, session, rel_frames, results):
    rows = []
    for fr, r in zip(rel_frames, results):
        p = r.get("parsed") or {}
        objs = "、".join(p.get("objects", []) or []) if isinstance(p.get("objects"), list) else ""
        rows.append(f"""
        <div class="card">
          <img src="{fr}"/>
          <div class="txt">
            <div class="i">#{r['i']:03d}</div>
            <div><b>场景</b>：{p.get('scene', '—')}</div>
            <div><b>任务</b>：<span class="task">{p.get('task', '—')}</span></div>
            <div><b>物体</b>：{objs or '—'}</div>
            <div class="reason"><b>依据</b>：{p.get('reason', '—')}</div>
            {'' if p else f'<div class="raw">RAW: {r["raw"][:200]}</div>'}
          </div>
        </div>""")
    html = f"""<!doctype html><meta charset="utf-8">
<title>VLM 慢脑验证 {session}</title>
<style>
 body{{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:16px}}
 h1{{font-size:18px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px}}
 .card{{display:flex;background:#1c1c1c;border-radius:8px;overflow:hidden}}
 .card img{{width:260px;object-fit:cover}}
 .txt{{padding:10px;font-size:13px;line-height:1.5}}
 .i{{color:#888;font-size:11px}}
 .task{{color:#4fd;font-weight:bold}}
 .reason{{color:#aaa}} .raw{{color:#f77;font-size:11px}}
</style>
<h1>VLM 慢脑（语义场景/任务）验证 — {session} — 眼看不同帧的「任务」是否被区分开</h1>
<div class="grid">{''.join(rows)}</div>"""
    open(os.path.join(outdir, "index.html"), "w").write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--seqdir", default=None)
    ap.add_argument("--model", default=None, help="VLM 目录(默认自动找 vlm/ 下第一个)")
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seqdir = args.seqdir or os.path.join(RUNS, f"obs_seq_{args.session}")
    frames = sorted(glob.glob(os.path.join(seqdir, "*.jpg")))
    if not frames:
        sys.exit(f"[慢脑] {seqdir} 下没有 FPV 帧, 先跑 Stage A。")
    print(f"[慢脑] {len(frames)} 帧 <- {seqdir}")

    proc, model = load_vlm(args.model or find_model_dir(), device)

    outdir = os.path.join(RUNS, f"vlm_slowbrain_{args.session}")
    os.makedirs(outdir, exist_ok=True)
    results, rel_frames = [], []
    for i, fp in enumerate(frames):
        raw, parsed = infer(proc, model, fp, device)
        results.append({"i": i, "frame": os.path.basename(fp), "raw": raw, "parsed": parsed})
        rel_frames.append(os.path.relpath(fp, outdir))
        tag = (parsed or {}).get("task", "?") if parsed else "[未解析]"
        print(f"  #{i:03d} task={tag}", flush=True)

    json.dump({"session": args.session, "model": args.model or find_model_dir(),
               "prompt": PROMPT, "results": results},
              open(os.path.join(outdir, "results.json"), "w"), ensure_ascii=False, indent=1)
    write_gallery(outdir, args.session, rel_frames, results)
    tasks = [(r["parsed"] or {}).get("task") for r in results if r["parsed"]]
    print(f"[慢脑] 完成. 解析成功 {len(tasks)}/{len(results)} 帧; 不同任务 {len(set(tasks))} 种")
    print(f"  -> {outdir}/index.html  (浏览器打开眼看区分度)")
    print(f"  -> {outdir}/results.json")


if __name__ == "__main__":
    main()
