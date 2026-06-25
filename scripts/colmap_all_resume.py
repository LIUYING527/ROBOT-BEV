"""接续 colmap_dense_all 崩在桥接的运行:DB已含特征+sequential匹配,
只用**单线程**补做跨段桥接匹配(避开faiss多线程BLAS的SIGSEGV)+ 增量建图。
用法: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/discoverse_venv/bin/python scripts/colmap_all_resume.py --tag all
"""
import os, sys, glob, argparse
import pycolmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all")
    args = ap.parse_args()
    work = os.path.join(ROOT, "outputs", "_colmap_joint_" + args.tag)
    db = os.path.join(work, "database.db")
    img_dir = os.path.join(work, "images")
    pair_file = os.path.join(work, "bridge_pairs.txt")

    mopt = pycolmap.FeatureMatchingOptions()
    mopt.num_threads = 1                                  # 单线程,避开faiss多线程BLAS崩
    print("[resume] 单线程补桥接匹配...", flush=True)
    iopt = pycolmap.ImportedPairingOptions()
    iopt.match_list_path = pair_file
    pycolmap.match_image_pairs(db, matching_options=mopt, pairing_options=iopt)

    sparse = os.path.join(work, "sparse")
    os.makedirs(sparse, exist_ok=True)
    print("[resume] 增量建图(全局BA)...", flush=True)
    recs = pycolmap.incremental_mapping(db, img_dir, sparse)
    if not recs:
        print("[resume] ❌ 没建出模型"); return
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    names = [im.name for im in rec.images.values()]
    na = sum(n.startswith("A_") for n in names); nb = sum(n.startswith("B_") for n in names)
    print(f"[resume] ✅ 最大模型: 注册 {rec.num_reg_images()}张 (114830:{na}, 113628:{nb}), 3D点{rec.num_points3D()}", flush=True)
    print(f"[resume] 模型数={len(recs)}", flush=True)


if __name__ == "__main__":
    main()
