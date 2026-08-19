"""修正版单session COLMAP: 共享单相机(CameraMode.SINGLE) — 修 reconstruct_colmap 每图独立相机的bug。
用法: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 python scripts/_recon_single_cam.py <session> [stride]
"""
import os, sys, glob, shutil
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "8")
import pycolmap

session = sys.argv[1] if len(sys.argv) > 1 else "demo191715"
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 2
work = f"outputs/_colmap_{session}_sc"
img_dir = f"{work}/images"
if os.path.exists(work): shutil.rmtree(work)
os.makedirs(img_dir)
allimg = sorted(glob.glob(f"data/{session}/images/zed/*.jpg"))
sel = allimg[::stride]
for p in sel:
    os.symlink(os.path.abspath(p), os.path.join(img_dir, os.path.basename(p)))
print(f"[INFO] {session}: {len(allimg)} 图 → {len(sel)} 张 (stride={stride}) 共享单相机", flush=True)

db = f"{work}/database.db"
print("[INFO] 提特征 (共享单相机 SINGLE)…", flush=True)
pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                          device=pycolmap.Device.auto)
print("[INFO] exhaustive 匹配…", flush=True)
pycolmap.match_exhaustive(db)
print("[INFO] 增量建图…", flush=True)
recs = pycolmap.incremental_mapping(db, img_dir, f"{work}/sparse")
if not recs:
    print("[ERR] 没建出模型"); sys.exit(1)
best = max(recs.values(), key=lambda r: r.num_reg_images())
print(f"[RESULT] 模型数{len(recs)} 最大模型注册 {best.num_reg_images()}/{len(sel)} 张, 三维点 {best.num_points3D()}", flush=True)
