"""换顺序匹配续跑(复用已提取特征)。exhaustive CPU 太慢→连续视频用 sequential。
用法: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 python3 scripts/resume_colmap_seq.py <session>
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","8")
import sys, pycolmap
session=sys.argv[1] if len(sys.argv)>1 else "114830"
work=f"outputs/_colmap_{session}"; db=f"{work}/database.db"; img=f"{work}/images"; sparse=f"{work}/sparse"
assert os.path.exists(db); os.makedirs(sparse,exist_ok=True)
mo=pycolmap.FeatureMatchingOptions(); mo.use_gpu=False; mo.num_threads=8; mo.sift.cpu_brute_force_matcher=True
po=pycolmap.SequentialPairingOptions(); po.overlap=20; po.quadratic_overlap=True
print(f"[INFO] {session} 顺序匹配 overlap=20 quad=True (CPU)",flush=True)
pycolmap.match_sequential(db, matching_options=mo, pairing_options=po, device=pycolmap.Device.cpu)
print("[INFO] 顺序匹配完成,增量建图…",flush=True)
recs=pycolmap.incremental_mapping(db, img, sparse)
if not recs: print("[ERR] 没建出模型"); sys.exit(1)
rec=max(recs.values(),key=lambda r:r.num_reg_images())
print(f"[OK] 模型: 注册 {rec.num_reg_images()} 张, 三维点 {rec.num_points3D()}",flush=True)
print(f"[OK] sparse 在 {sparse}/",flush=True)
