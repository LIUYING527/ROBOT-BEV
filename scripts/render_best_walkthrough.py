"""最终交付:沿采集轨迹(on-path=数据最密最锐处)渲平滑FPV穿行视频.
2×超采样抗锯齿 + hqbest皮肤(appearance曝光一致,最适合视频).
用法: PLY=outputs/gs_vggto_colmapjoint_all_hqbest.ply python scripts/render_best_walkthrough.py
"""
import os, numpy as np, torch, torch.nn.functional as F, pycolmap, cv2, imageio.v2 as imageio
from gsplat import rasterization
DEV = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLY = os.environ.get("PLY", os.path.join(ROOT, "outputs/gs_vggto_colmapjoint_all_hqbest.ply"))
MODEL = os.path.join(ROOT, "outputs/_colmap_joint_all/sparse/0")
NFRAMES = int(os.environ.get("NFRAMES", "240"))
SS = float(os.environ.get("SS", "2.0"))          # 超采样倍率
OUT = os.environ.get("OUT", os.path.join(ROOT, "outputs/walk_hqbest.mp4"))

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
    quats = F.normalize(torch.tensor(np.stack([g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]], 1), device=DEV), dim=-1)
    scales = torch.tensor(np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], 1)), device=DEV)
    opac = torch.tensor(1 / (1 + np.exp(-g["opacity"])), device=DEV)
    # SH3 系数(N,K,3):f_dc + f_rest,渲染时按视角算高光
    fdc = np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], 1)[:, None, :]
    rest = [k for k in props if k.startswith("f_rest_")]
    if rest:
        fr = np.stack([g[k] for k in sorted(rest, key=lambda x: int(x.split("_")[-1]))], 1)
        fr = fr.reshape(n, 3, -1).transpose(0, 2, 1)             # (N, K-1, 3)
        sh = np.concatenate([fdc, fr], axis=1)
        deg = int(round((sh.shape[1]) ** 0.5)) - 1
    else:
        sh = fdc; deg = 0
    colors = torch.tensor(sh, device=DEV)
    cs = np.load(path + ".norm.npy"); center, s = cs[:3].astype(np.float32), float(cs[3])
    means = means * s + torch.tensor(center, device=DEV)
    scales = scales * s
    return means, quats, scales, opac, colors, deg

means, quats, scales, opac, colors, deg = load_gauss(PLY)
print(f"[INFO] {means.shape[0]} 高斯  SH{deg}", flush=True)
rec = pycolmap.Reconstruction(MODEL)
ims = sorted(rec.images.values(), key=lambda im: im.name)
cc = np.array([-(np.array(im.cam_from_world().rotation.matrix()).T @
                 np.array(im.cam_from_world().translation)) for im in ims])
cmed = np.median(cc, 0)
inl = np.linalg.norm(cc - cmed, axis=1) < 5 * np.median(np.linalg.norm(cc - cmed, axis=1))
ims = [im for im, k in zip(ims, inl) if k]
idx = np.linspace(0, len(ims) - 1, NFRAMES).astype(int)
cam = rec.cameras[ims[0].camera_id]
K0 = np.array(cam.calibration_matrix(), np.float32); W0, H0 = int(cam.width), int(cam.height)
Wr, Hr = int(W0 * SS), int(H0 * SS)
K = K0.copy(); K[:2] *= SS

frames = []
for j, i in enumerate(idx):
    im = ims[i]
    R = np.array(im.cam_from_world().rotation.matrix(), np.float32)
    t = np.array(im.cam_from_world().translation, np.float32)
    V = np.eye(4, dtype=np.float32); V[:3, :3] = R; V[:3, 3] = t
    vm = torch.tensor(V, device=DEV)[None]; Ks = torch.tensor(K, device=DEV)[None]
    with torch.no_grad():
        img, _, _ = rasterization(means, quats, scales, opac, colors, vm, Ks, Wr, Hr,
                                  sh_degree=deg, near_plane=0.01, far_plane=1e4)
    rgb = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    rgb = cv2.resize(rgb, (W0, H0), interpolation=cv2.INTER_AREA)   # 超采样降回→抗锯齿
    frames.append(rgb)
    if (j + 1) % 60 == 0: print(f"  {j+1}/{NFRAMES}", flush=True)

imageio.mimsave(OUT, frames, fps=30, codec="libx264", quality=8)
print(f"[OK] {OUT}  ({len(frames)}帧 {W0}x{H0} @30fps, SS{SS})", flush=True)
