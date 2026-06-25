"""O(n) 稠密多段重建:用尽可能多的帧(排除有人段),sequential匹配(内段时序链,O(n))
+ 关键帧桥接跨段(A关键帧×B关键帧,小k²),避免exhaustive O(n²)的数小时。

人帧排除(范围选择,因seg模型本机下不到):
  114830 有人段 288-363 → 用 0:288 + 364:625
  113628 开头有人 0-30   → 用 30:1183

用法: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 ~/discoverse_venv/bin/python scripts/colmap_dense_all.py \
        --a_ranges 0:288,364:625 --b_ranges 30:1183 --a_stride 1 --b_stride 1 \
        --overlap 15 --bridge_stride 15 --tag all
产出: outputs/_colmap_joint_all/{database.db, sparse/} + 报告注册数
"""
import os, sys, glob, shutil, argparse
import numpy as np
import pycolmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gather(session, ranges, stride):
    allf = sorted(glob.glob(os.path.join(ROOT, "data", session, "images", "zed", "*.jpg")))
    out = []
    for r in ranges.split(","):
        a, b = r.split(":")
        out += allf[int(a):int(b)]
    return out[::stride]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_ranges", default="0:288,364:625")   # 114830 排除有人段288-363
    ap.add_argument("--b_ranges", default="30:1183")          # 113628 排除开头有人0-30
    ap.add_argument("--a_stride", type=int, default=1)
    ap.add_argument("--b_stride", type=int, default=1)
    ap.add_argument("--overlap", type=int, default=15)        # sequential 时序窗
    ap.add_argument("--bridge_stride", type=int, default=15)  # 跨段桥接关键帧步长
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()

    work = os.path.join(ROOT, "outputs", "_colmap_joint_" + args.tag)
    img_dir = os.path.join(work, "images")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(img_dir)

    A = gather("114830", args.a_ranges, args.a_stride)
    B = gather("113628", args.b_ranges, args.b_stride)
    namesA = ["A_" + os.path.basename(p) for p in A]
    namesB = ["B_" + os.path.basename(p) for p in B]
    for p, nm in zip(A, namesA):
        os.symlink(os.path.abspath(p), os.path.join(img_dir, nm))
    for p, nm in zip(B, namesB):
        os.symlink(os.path.abspath(p), os.path.join(img_dir, nm))
    print(f"[dense_all] 114830 {len(A)}张 + 113628 {len(B)}张 = {len(A)+len(B)}张", flush=True)

    db = os.path.join(work, "database.db")
    print("[dense_all] 提特征 SIFT...", flush=True)
    pycolmap.extract_features(db, img_dir, device=pycolmap.Device.auto)

    print(f"[dense_all] sequential 匹配(overlap={args.overlap},O(n),内段时序链)...", flush=True)
    sopt = pycolmap.SequentialPairingOptions()
    sopt.overlap = args.overlap
    sopt.quadratic_overlap = True
    pycolmap.match_sequential(db, pairing_options=sopt)

    # 跨段桥接: A关键帧×B关键帧(每两两一对,小k²)
    kA = namesA[::args.bridge_stride]
    kB = namesB[::args.bridge_stride]
    pair_file = os.path.join(work, "bridge_pairs.txt")
    with open(pair_file, "w") as f:
        for a in kA:
            for b in kB:
                f.write(f"{a} {b}\n")
    print(f"[dense_all] 桥接 {len(kA)}×{len(kB)}={len(kA)*len(kB)}对 跨段匹配...", flush=True)
    iopt = pycolmap.ImportedPairingOptions()
    iopt.match_list_path = pair_file
    pycolmap.match_image_pairs(db, pairing_options=iopt)

    print("[dense_all] 增量建图(全局BA)...", flush=True)
    recs = pycolmap.incremental_mapping(db, img_dir, os.path.join(work, "sparse"))
    if not recs:
        print("[dense_all] ❌ 没建出模型"); return
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    names = [im.name for im in rec.images.values()]
    na = sum(n.startswith("A_") for n in names); nb = sum(n.startswith("B_") for n in names)
    print(f"[dense_all] ✅ 最大模型: 注册 {rec.num_reg_images()}张 "
          f"(114830:{na}/{len(A)}, 113628:{nb}/{len(B)}), 3D点{rec.num_points3D()}", flush=True)
    print(f"[dense_all] 模型数={len(recs)} (=1最好)", flush=True)


if __name__ == "__main__":
    main()
