"""同位姿对比 minrecipe(dense极简配方铺全1610帧) vs hq2(全旋钮).
两ply共享同一vdir/cameras/norm → 用COLMAP原始外参直接透视渲染,零对齐误差.
on-path真实相机位 + 一个转头看侧墙的off-axis位.
"""
import os, sys, numpy as np, torch, torch.nn.functional as F, pycolmap, cv2
from gsplat import rasterization
DEV = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "outputs/_colmap_joint_all/sparse/0")
PLYS = {
    "hqbest (SH3+appearance去曝光)": os.path.join(ROOT, "outputs/gs_vggto_colmapjoint_all_hqbest.ply"),
    "hq2 (SH3/遮罩225,无appearance)": os.path.join(ROOT, "outputs/gs_vggto_colmapjoint_all_hq2.ply"),
}

def load_gauss(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"; f.readline()
        n = int(f.readline().split()[-1]); props = []
        ln = f.readline()
        while ln.strip() != b"end_header":
            if ln.startswith(b"property"): props.append(ln.split()[-1].decode())
            ln = f.readline()
        d = np.frombuffer(f.read(n * len(props) * 4), np.float32).reshape(n, len(props))
    g = {p: d[:, i] for i, p in enumerate(props)}
    means = torch.tensor(np.stack([g["x"], g["y"], g["z"]], 1), device=DEV)
    quats = torch.tensor(np.stack([g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]], 1), device=DEV)
    scales = torch.tensor(np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], 1)), device=DEV)
    opac = torch.tensor(1 / (1 + np.exp(-g["opacity"])), device=DEV)
    colors = torch.tensor(np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], 1) * 0.2820948 + 0.5, device=DEV).clamp(0, 1)
    cs = np.load(path + ".norm.npy"); center, s = cs[:3].astype(np.float32), float(cs[3])
    means = means * s + torch.tensor(center, device=DEV)   # 反归一化回COLMAP世界
    scales = scales * s
    return means, F.normalize(quats, dim=-1), scales, opac, colors

def render(gauss, R, t, K, W, H):
    means, quats, scales, opac, colors = gauss
    V = np.eye(4, dtype=np.float32); V[:3, :3] = R; V[:3, 3] = t
    vm = torch.tensor(V, device=DEV)[None]; Ks = torch.tensor(K.astype(np.float32), device=DEV)[None]
    with torch.no_grad():
        img, _, _ = rasterization(means, quats, scales, opac, colors, vm, Ks, W, H,
                                  near_plane=0.01, far_plane=1e4)
    return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

rec = pycolmap.Reconstruction(MODEL)
ims = sorted(rec.images.values(), key=lambda im: im.name)
cam = rec.cameras[ims[0].camera_id]
K = np.array(cam.calibration_matrix(), np.float64); W, Hh = int(cam.width), int(cam.height)
print(f"[INFO] {len(ims)}帧 K@{W}x{Hh}", flush=True)

gaussians = {name: load_gauss(p) for name, p in PLYS.items()}
for name, g in gaussians.items():
    print(f"[INFO] {name}: {g[0].shape[0]} 高斯", flush=True)

# 取沿路径3个真实相机位 + 每个再来一个yaw转头看侧墙
idxs = [350, 750, 1150]
YAW = float(os.environ.get("YAW", "0"))  # 0=on-path; 设70看右侧墙
rows = []
for idx in idxs:
    im = ims[idx]
    R = np.array(im.cam_from_world().rotation.matrix(), np.float64)
    t = np.array(im.cam_from_world().translation, np.float64)
    if YAW != 0:  # 绕相机y轴(竖直)转头
        a = np.deg2rad(YAW); Ry = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        R = Ry @ R; t = Ry @ t
    panels = []
    for name, g in gaussians.items():
        rgb = render(g, R, t, K, W, Hh)
        bgr = rgb[:, :, ::-1].copy()
        cv2.putText(bgr, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(bgr, f"frame {idx}" + (f" yaw{int(YAW)}" if YAW else " on-path"),
                    (10, Hh-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        panels.append(bgr)
    rows.append(np.hstack(panels))
out = np.vstack(rows)
tag = "onpath" if YAW == 0 else f"yaw{int(YAW)}"
op = os.path.join(ROOT, f"outputs/_cmp_minrecipe_vs_hq2_{tag}.png")
cv2.imwrite(op, out)
print(f"[OK] saved {op}  ({out.shape[1]}x{out.shape[0]})", flush=True)
