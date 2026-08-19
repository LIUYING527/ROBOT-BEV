"""同坐标系三列对比: 同一真实采集帧 → [真实图 | full-1020差版渲染 | f-827新版渲染]。
full(110744)与f(110744f)都基于同一 _colmap_110744_sc(1020帧,同坐标系),按相同帧索引j
用各自模型的平滑真实外参 _Vs_smooth[j] 渲染=完全公平(同相机同帧)。真实图取该帧对应文件。
输出 outputs/recon_verify/lab_110744_full_vs_f_rawV.png
"""
import os, sys, numpy as np, cv2, pycolmap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture_fpv import CaptureFPV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(ROOT, "data", "110744", "images", "zed")

fpv_full = CaptureFPV("colmapjoint_110744")    # 全帧1020差版(当前 gs_..110744.ply)
fpv_f = CaptureFPV("colmapjoint_110744f")      # 剔糊帧827新版

# 两者同一 colmap, images 排序一致 → 相同下标 j = 相同真实帧
rec = pycolmap.Reconstruction(os.path.join(ROOT, "outputs", "_colmap_110744_sc", "sparse", "2"))
ims = sorted(rec.images.values(), key=lambda im: im.name)
n = len(ims)
print(f"[info] colmap images n={n}")

def real_img(name):
    ts = name.split("/")[-1]  # colmap image name = 帧文件名
    # colmap 用的是 frames_zed 里的 4位序号_时间戳.jpg; 真实图目录是 时间戳.jpg
    stamp = ts.split("_")[-1] if "_" in ts else ts
    p = os.path.join(REAL_DIR, stamp)
    im = cv2.imread(p)
    return im

def W(img):
    return cv2.resize(img, (640, 400)) if img is not None else np.zeros((400, 640, 3), np.uint8)

rows = []
for j in [int(x) for x in np.linspace(60, n - 60, 6)]:
    name = ims[j].name
    rj = real_img(name)
    rf = W(rj); cv2.putText(rf, f"REAL #{j}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    ff = W(fpv_full._render_V(fpv_full._Vs[j])[:, :, ::-1]); cv2.putText(ff, "full-1020(510k)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    fn = W(fpv_f._render_V(fpv_f._Vs[j])[:, :, ::-1]); cv2.putText(fn, "f-827(531k)", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    rows.append(np.hstack([rf, ff, fn]))
    print(f"  row j={j} name={name}")

out = np.vstack(rows)
os.makedirs(os.path.join(ROOT, "outputs", "recon_verify"), exist_ok=True)
op = os.path.join(ROOT, "outputs", "recon_verify", "lab_110744_full_vs_f_rawV.png")
cv2.imwrite(op, out)
print("saved", op, out.shape)
