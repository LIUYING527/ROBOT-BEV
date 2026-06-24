"""两段采集联合 COLMAP SfM+BA → 厘米级一致位姿(把114830+113628绑死在一个坐标系)。
治"糊"的关键:VGGT前馈位姿松(~0.4m)→多视角对不齐→糊;COLMAP全局BA位姿紧。
exhaustive匹配(含跨段对,才能把两段连起来)。CPU匹配(预编译pycolmap无CUDA匹配),OPENBLAS_NUM_THREADS=1防崩。

用法: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 \
      ~/discoverse_venv/bin/python scripts/colmap_joint.py [--a_stride 3] [--b_stride 9]
产出: outputs/_colmap_joint/{database.db, sparse/} + 报告两段各注册多少(连上=可训统一3DGS)
"""
import os, sys, glob, shutil, argparse
import numpy as np
import pycolmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_stride", type=int, default=3)    # 114830 抽帧步长
    ap.add_argument("--b_stride", type=int, default=9)    # 113628 抽帧步长
    ap.add_argument("--a_end", type=int, default=288)     # 114830 用到第几帧(前向锚)
    ap.add_argument("--b_end", type=int, default=853)     # 113628 用到第几帧(侧扫覆盖)
    ap.add_argument("--tag", default="")                  # 输出目录后缀(空=_colmap_joint,避免覆盖旧模型)
    args = ap.parse_args()
    work = os.path.join(ROOT, "outputs", "_colmap_joint" + (("_" + args.tag) if args.tag else ""))
    img_dir = os.path.join(work, "images")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(img_dir)

    A = sorted(glob.glob(os.path.join(ROOT, "data", "114830", "images", "zed", "*.jpg")))[0:args.a_end][::args.a_stride]
    B = sorted(glob.glob(os.path.join(ROOT, "data", "113628", "images", "zed", "*.jpg")))[0:args.b_end][::args.b_stride]
    for p in A:
        os.symlink(os.path.abspath(p), os.path.join(img_dir, "A_" + os.path.basename(p)))
    for p in B:
        os.symlink(os.path.abspath(p), os.path.join(img_dir, "B_" + os.path.basename(p)))
    print(f"[joint] 114830 {len(A)}张 + 113628 {len(B)}张 = {len(A)+len(B)}张 → COLMAP", flush=True)

    db = os.path.join(work, "database.db")
    print("[joint] 提特征 SIFT...", flush=True)
    pycolmap.extract_features(db, img_dir, device=pycolmap.Device.auto)
    print("[joint] exhaustive 匹配(CPU,慢,含跨段)...", flush=True)
    pycolmap.match_exhaustive(db)
    print("[joint] 增量建图(全局BA)...", flush=True)
    recs = pycolmap.incremental_mapping(db, img_dir, os.path.join(work, "sparse"))
    if not recs:
        print("[joint] ❌ 没建出模型"); return
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    names = [im.name for im in rec.images.values()]
    na = sum(n.startswith("A_") for n in names); nb = sum(n.startswith("B_") for n in names)
    print(f"[joint] ✅ 最大模型: 注册 {rec.num_reg_images()}张 (114830:{na}/{len(A)}, 113628:{nb}/{len(B)}), 3D点{rec.num_points3D()}", flush=True)
    print(f"[joint] {'两段都注册上=连起来了,可训统一3DGS' if na>10 and nb>10 else '⚠️ 某段没连上,需调参/补帧'}", flush=True)
    print(f"[joint] 模型数={len(recs)} (=1最好;>1说明分裂没全连)", flush=True)


if __name__ == "__main__":
    main()
