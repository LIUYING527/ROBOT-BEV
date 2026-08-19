"""长距离穿行版单session COLMAP: 共享单相机(SINGLE) + 【顺序匹配】(sequential, O(n) 不是穷举 O(n^2))。
   → 长穿行可上密集帧(不必为压穷举成本而抽稀)。overlap+quadratic_overlap 抓相邻及部分长程重叠。
用法: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 python scripts/_recon_single_cam_seq.py <session> [stride] [overlap]
"""
import os, sys, glob, shutil
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "8")
import pycolmap

session = sys.argv[1] if len(sys.argv) > 1 else "105731"
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 2
overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 15
work = f"outputs/_colmap_{session}_sc"
img_dir = f"{work}/images"
if os.path.exists(work): shutil.rmtree(work)
os.makedirs(img_dir)
allimg = sorted(glob.glob(f"data/{session}/images/zed/*.jpg"))
# RECON_START/RECON_END: 裁掉开头/结尾污染帧(如凑近镜头的路人), 再按 stride 抽稀
start = int(os.environ.get("RECON_START", "0"))
end = int(os.environ.get("RECON_END", "0")) or len(allimg)
win = allimg[start:end]
sel = win[::stride]
for p in sel:
    os.symlink(os.path.abspath(p), os.path.join(img_dir, os.path.basename(p)))
print(f"[INFO] {session}: {len(allimg)} 图, 窗口[{start},{end})={len(win)}帧 → {len(sel)} 张 (stride={stride}) 共享单相机, 顺序匹配overlap={overlap}", flush=True)

db = f"{work}/database.db"
print("[INFO] 提特征 (SINGLE)…", flush=True)
pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, device=pycolmap.Device.auto)
print("[INFO] 顺序匹配 (sequential + quadratic_overlap)…", flush=True)
popt = pycolmap.SequentialPairingOptions(overlap=overlap, quadratic_overlap=True, loop_detection=False)
pycolmap.match_sequential(db, pairing_options=popt)
print("[INFO] 增量建图…", flush=True)
recs = pycolmap.incremental_mapping(db, img_dir, f"{work}/sparse")
if not recs:
    print("[ERR] 没建出模型"); sys.exit(1)
best = max(recs.values(), key=lambda r: r.num_reg_images())
print(f"[RESULT] 模型数{len(recs)} 最大模型注册 {best.num_reg_images()}/{len(sel)} 张, 三维点 {best.num_points3D()}", flush=True)
