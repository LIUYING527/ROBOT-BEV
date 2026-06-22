"""续跑 COLMAP 匹配+建图(复用已提取的 database.db,跳过 extract_features)。

背景:reconstruct_colmap.py 跑 114830 时 feature extraction 完成,但 match 阶段
fell back 到 CPU faiss + OpenBLAS 多线程 → SIGSEGV(blas_memory_alloc)崩溃。
本脚本复用 outputs/_colmap_<s>/database.db,强制 GPU 匹配,并限制 OpenBLAS 线程兜底。

用法: OPENBLAS_NUM_THREADS=1 python scripts/resume_colmap_match.py <session>
"""
import os
# 兜底:即使 fall back 到 CPU faiss,单线程 OpenBLAS 也不会段错误
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import pycolmap

session = sys.argv[1] if len(sys.argv) > 1 else "114830"
work = f"outputs/_colmap_{session}"
db = f"{work}/database.db"
img_dir = f"{work}/images"
sparse = f"{work}/sparse"
assert os.path.exists(db), f"找不到 {db},需先跑 extract_features"
os.makedirs(sparse, exist_ok=True)

print(f"[INFO] {session}: 复用 {db},CPU 匹配(预编译 pycolmap 无 CUDA 匹配)", flush=True)
# 此 pycolmap wheel 无 GPU 匹配 → 只能 CPU。崩溃根因是 faiss+OpenBLAS 多线程,
# 已用 OPENBLAS_NUM_THREADS=1 兜底;再用 cpu_brute_force_matcher 走更简单的暴力匹配路径。
mo = pycolmap.FeatureMatchingOptions()
mo.use_gpu = False
mo.num_threads = 8           # matcher worker 并行(OMP),提速
mo.sift.cpu_brute_force_matcher = True   # 暴力匹配,避开崩溃的 faiss IVF 路径
pycolmap.match_exhaustive(db, matching_options=mo, device=pycolmap.Device.cpu)
print("[INFO] 匹配完成,增量建图(全局BA)…", flush=True)
recs = pycolmap.incremental_mapping(db, img_dir, sparse)
if not recs:
    print("[ERR] 没建出模型"); sys.exit(1)
rec = max(recs.values(), key=lambda r: r.num_reg_images())
print(f"[OK] 模型: 注册 {rec.num_reg_images()} 张, 三维点 {rec.num_points3D()}", flush=True)
print(f"[OK] sparse 在 {sparse}/", flush=True)
