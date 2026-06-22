#!/usr/bin/env python3
"""Photoreal drive-through of OUR 3DGS asset rendered through the DISCOVERSE /
GS-Playground gaussian_renderer (gaussian_renderer.core.batch_render), using the
real COLMAP training camera poses.

Unlike an external orbit (which is noise for ground-captured 3DGS), rendering
from the trajectory the data was actually captured along gives photoreal frames.
This proves our reconstructed scene works as a DISCOVERSE asset.

  ~/discoverse_venv/bin/python scripts/discoverse_render_drivethrough.py
"""
import os, numpy as np, cv2, pycolmap, imageio.v2 as imageio
from gaussian_renderer.core.util_gau import load_ply

PLY   = "outputs/gs_111450.ply"
MODEL = "outputs/_colmap_111450/sparse/1"
OUT   = "outputs/discoverse_gs_111450"
NFRAMES = 40
W, H = 960, 540

def main():
    os.makedirs(OUT, exist_ok=True)
    from gaussian_renderer.core.batch_rasterization import batch_render
    from gaussian_renderer.core.gaussiandata import GaussianData

    gs = load_ply(PLY).to_cuda()
    if gs.sh.dim() == 2:
        gs.sh = gs.sh.reshape(gs.sh.shape[0], -1, 3).contiguous()
    print(f"[load] {gs.xyz.shape[0]} gaussians")

    cs = np.load(PLY + ".norm.npy"); center, s = cs[:3].astype(np.float64), float(cs[3])

    rec = pycolmap.Reconstruction(MODEL)
    ims = [rec.images[i] for i in sorted(rec.images)]
    # camera centers in COLMAP world, drop misregistered outliers
    cc = np.array([-(np.array(im.cam_from_world().rotation.matrix()).T @
                     np.array(im.cam_from_world().translation)) for im in ims])
    cmed = np.median(cc, axis=0)
    d = np.linalg.norm(cc - cmed, axis=1)
    inl = d < 5 * np.median(d)
    ims = [im for im, k in zip(ims, inl) if k]
    print(f"[poses] {len(ims)} inlier cameras")

    # intrinsics (square-pixel approx that batch_render assumes)
    cam0 = rec.cameras[ims[0].camera_id]
    fy = float(cam0.focal_length_y) if hasattr(cam0, "focal_length_y") else float(cam0.focal_length)
    # COLMAP camera is full-res; scale fy to our render height
    fy_scaled = fy * (H / cam0.height)
    fovy_deg = np.degrees(2 * np.arctan(H / (2 * fy_scaled)))
    print(f"[intr] fovy={fovy_deg:.1f} deg")

    idx = np.linspace(0, len(ims) - 1, NFRAMES).astype(int)
    frames = []
    for n, j in enumerate(idx):
        im = ims[j]
        R = np.array(im.cam_from_world().rotation.matrix(), np.float64)  # world2cam (OpenCV)
        t = np.array(im.cam_from_world().translation, np.float64)
        c = -R.T @ t
        cn = (c - center) / s                  # camera center in normalized gaussian frame
        cam_pos = cn.reshape(1, 3).astype(np.float32)
        cam_xmat = R.T.reshape(1, 9).astype(np.float32)   # cam->world rotation
        color, _ = batch_render(gs, cam_pos, cam_xmat, H, W,
                                np.array([fovy_deg]), y_up=False)
        img = (np.clip(color[0].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        frames.append(img)
        cv2.imwrite(os.path.join(OUT, f"drive_{n:03d}.png"), img[:, :, ::-1])
        if n % 8 == 0:
            print(f"  {n+1}/{NFRAMES}  nonblack {(img.sum(2)>10).mean()*100:.0f}%", flush=True)

    imageio.mimsave(os.path.join(OUT, "drivethrough.gif"), frames, fps=12)
    # a 3x3 contact sheet of evenly spaced frames
    pick = np.linspace(0, len(frames) - 1, 9).astype(int)
    grid = [frames[p] for p in pick]
    sheet = np.vstack([np.hstack(grid[r*3:(r+1)*3]) for r in range(3)])
    cv2.imwrite(os.path.join(OUT, "drivethrough_sheet.png"),
                cv2.resize(sheet[:, :, ::-1], (W*3//2, H*3//2)))
    print(f"[done] {OUT}/drivethrough.gif + drivethrough_sheet.png")

if __name__ == "__main__":
    main()
