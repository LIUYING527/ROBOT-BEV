"""把 3DGS 渲染成高分辨率"正俯视航拍图"(正交相机)——比二值密度栅格可读得多。

3DGS 是照片级的,从正上方正交投影渲一张图:马路/地标线/建筑/草地一目了然。
流程:载入高斯 → 重力对齐(旋转 means+quats 使地面水平) → 正交相机看下方 → 高分渲染
     → 叠相机轨迹。地面在航拍图里本就该有,不是问题。

用法: python scripts/render_topdown_3dgs.py [像素每米]
输出: outputs/topdown_111450.png
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gsplat import rasterization
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "cp", os.path.join(os.path.dirname(__file__), "colmap_postprocess.py"))
cp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cp)

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA = "outputs/_colmap_111450"; MODEL = DATA + "/sparse/1"
PLY = "outputs/gs_111450.ply"; DEV = "cuda"
PPM = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0   # 像素每米(分辨率)


def load_gauss(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"; f.readline()
        n = int(f.readline().split()[-1]); props = []
        ln = f.readline()
        while ln.strip() != b"end_header":
            if ln.startswith(b"property"):
                props.append(ln.split()[-1].decode())
            ln = f.readline()
        d = np.frombuffer(f.read(n * len(props) * 4), np.float32).reshape(n, len(props))
    g = {p: d[:, i] for i, p in enumerate(props)}
    means = torch.tensor(np.stack([g["x"], g["y"], g["z"]], 1), device=DEV)
    quats = torch.tensor(np.stack([g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]], 1), device=DEV)
    scales = torch.tensor(np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], 1)), device=DEV)
    opac = torch.tensor(1 / (1 + np.exp(-g["opacity"])), device=DEV)
    colors = torch.tensor(np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], 1) * 0.2820948 + 0.5, device=DEV).clamp(0, 1)
    return means, quats, scales, opac, colors


def quat_mul(a, b):  # Hamilton, (...,4) w,x,y,z
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def main():
    means, quats, scales, opac, colors = load_gauss(PLY)
    print(f"[INFO] 高斯 {means.shape[0]}", flush=True)
    # 轻滤离群(保留大部分以保照片感)
    M = means.cpu().numpy()
    med = np.median(M, 0); rad = 6 * np.median(np.linalg.norm(M - med, axis=1))
    k = (np.linalg.norm(M - med, axis=1) < rad) & (opac.cpu().numpy() > 0.05)
    k = torch.tensor(k, device=DEV)
    means, quats, scales, opac, colors = means[k], quats[k], scales[k], opac[k], colors[k]

    # 重力对齐:旋转 means + quats 使地面水平(z 向上)
    Rg = cp.align_gravity(means.cpu().numpy()).astype(np.float32)
    Rt = torch.tensor(Rg, device=DEV)
    means = means @ Rt.T
    # Rg → 四元数,左乘旋转每个高斯
    from numpy import trace
    qw = np.sqrt(max(0, 1 + Rg[0, 0] + Rg[1, 1] + Rg[2, 2])) / 2
    qx = (Rg[2, 1] - Rg[1, 2]) / (4 * qw + 1e-9)
    qy = (Rg[0, 2] - Rg[2, 0]) / (4 * qw + 1e-9)
    qz = (Rg[1, 0] - Rg[0, 1]) / (4 * qw + 1e-9)
    qR = torch.tensor([qw, qx, qy, qz], device=DEV, dtype=torch.float32)
    quats = quat_mul(qR.expand_as(quats), F.normalize(quats, dim=-1))

    # 米制标定(用相机轨迹长≈真实路程)+ 求场景范围
    cs = np.load(PLY + ".norm.npy"); center, s = cs[:3].astype(np.float32), float(cs[3])
    rec = pycolmap.Reconstruction(MODEL)
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    cc = np.array([-(np.array(im.cam_from_world().rotation.matrix()).T @
                     np.array(im.cam_from_world().translation)) for im in ims])
    cmed = np.median(cc, 0); inl = np.linalg.norm(cc - cmed, axis=1) < 5 * np.median(np.linalg.norm(cc - cmed, axis=1))
    traj = ((cc[inl] - center) / s) @ Rg.T
    mp = means.cpu().numpy()
    tlen = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    sc = cp.real_path_length("111450") / tlen if tlen > 1e-6 else 1.0
    means = means * sc; traj = traj * sc; scales = scales * sc
    mp = means.cpu().numpy()

    x0, x1 = np.percentile(mp[:, 0], [1, 99]); y0, y1 = np.percentile(mp[:, 1], [1, 99])
    pad = 2.0; x0 -= pad; x1 += pad; y0 -= pad; y1 += pad
    W = int((x1 - x0) * PPM); H = int((y1 - y0) * PPM)
    W = min(W, 2000); H = min(H, 2000)
    zmax = float(mp[:, 2].max())
    print(f"[INFO] 场景 {x1-x0:.0f}×{y1-y0:.0f}m → {W}×{H}px ({PPM:.0f}px/m)", flush=True)

    # 正交相机:位于上方看下(cam_z = -world_z)
    Rwc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    C = np.array([(x0 + x1) / 2, (y0 + y1) / 2, zmax + 5], np.float32)
    V = np.eye(4, dtype=np.float32); V[:3, :3] = Rwc; V[:3, 3] = -Rwc @ C
    fx = fy = PPM
    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], np.float32)
    vm = torch.tensor(V, device=DEV)[None]; Ks = torch.tensor(K, device=DEV)[None]

    with torch.no_grad():
        img, alpha, _ = rasterization(means, F.normalize(quats, dim=-1), scales, opac, colors,
                                      vm, Ks, W, H, camera_model="ortho",
                                      near_plane=0.01, far_plane=1e4)
    rgb = img[0].clamp(0, 1).cpu().numpy()
    print(f"[INFO] 渲染完成 alpha均 {float(alpha.mean()):.2f}", flush=True)

    fig, ax = plt.subplots(figsize=(W / 130, H / 130))
    ax.imshow(rgb, extent=[x0, x1, y0, y1], origin="lower")
    ax.plot(traj[:, 0], traj[:, 1], "-", color="#1b6cff", lw=2.5, label="相机轨迹")
    ax.plot(traj[0, 0], traj[0, 1], "o", color="lime", ms=12, mec="k", label="起点")
    ax.plot(traj[-1, 0], traj[-1, 1], "s", color="red", ms=11, mec="k", label="终点")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("111450 3DGS 正俯视航拍图(正交渲染)+ 真实轨迹")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig("outputs/topdown_111450.png", dpi=130)
    print("[OK] outputs/topdown_111450.png", flush=True)


if __name__ == "__main__":
    main()
