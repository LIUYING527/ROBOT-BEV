"""3DGS 第一视角穿行回放 GIF —— 沿真实相机轨迹渲染(地面前视数据唯一好看的展示)。
3DGS 只在训练视角附近有效;沿原路径前视渲染=照片级,头顶/环绕则是噪点。
用法: python scripts/render_drivethrough_3dgs.py
输出: outputs/drivethrough_111450.gif
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pycolmap
import imageio.v2 as imageio
from gsplat import rasterization
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "rt", os.path.join(os.path.dirname(__file__), "render_topdown_3dgs.py"))
rt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rt)

DATA = "outputs/_colmap_111450"; MODEL = DATA + "/sparse/1"
PLY = "outputs/gs_111450.ply"; DEV = "cuda"; DS = 2
NFRAMES = 90


def main():
    means, quats, scales, opac, colors = rt.load_gauss(PLY)
    quats = F.normalize(quats, dim=-1)
    print(f"[INFO] 高斯 {means.shape[0]}", flush=True)

    cs = np.load(PLY + ".norm.npy"); center, s = cs[:3].astype(np.float32), float(cs[3])
    rec = pycolmap.Reconstruction(MODEL)
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    # 内点相机(同训练)
    cc = np.array([-(np.array(im.cam_from_world().rotation.matrix()).T @
                     np.array(im.cam_from_world().translation)) for im in ims])
    cmed = np.median(cc, 0)
    inl = np.linalg.norm(cc - cmed, axis=1) < 5 * np.median(np.linalg.norm(cc - cmed, axis=1))
    ims = [im for im, k in zip(ims, inl) if k]
    idx = np.linspace(0, len(ims) - 1, NFRAMES).astype(int)

    frames = []
    for j, i in enumerate(idx):
        im = ims[i]; cam = rec.cameras[im.camera_id]
        K = np.array(cam.calibration_matrix(), np.float32)
        R = np.array(im.cam_from_world().rotation.matrix(), np.float32)
        t = np.array(im.cam_from_world().translation, np.float32)
        c = -R.T @ t; cn = (c - center) / s
        V = np.eye(4, dtype=np.float32); V[:3, :3] = R; V[:3, 3] = -R @ cn
        W, Hh = int(cam.width), int(cam.height)
        Ks = torch.tensor(K, device=DEV).clone(); Ks[:2] /= DS; Ks = Ks[None]
        vm = torch.tensor(V, device=DEV)[None]
        with torch.no_grad():
            img, _, _ = rasterization(means, quats, scales, opac, colors, vm, Ks,
                                      W // DS, Hh // DS, near_plane=0.01, far_plane=1e4)
        frames.append((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        if (j + 1) % 30 == 0:
            print(f"  {j+1}/{NFRAMES}", flush=True)
    imageio.mimsave("outputs/drivethrough_111450.gif", frames, fps=15)
    print("[OK] outputs/drivethrough_111450.gif", flush=True)


if __name__ == "__main__":
    main()
