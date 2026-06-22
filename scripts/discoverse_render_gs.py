#!/usr/bin/env python3
"""Headless render of OUR 3DGS asset through the DISCOVERSE / GS-Playground
gaussian_renderer stack (gaussian_renderer.core.batch_render).

Proves we can drop our own reconstructed scene (gs_<session>.ply) into the
previous-generation DISCOVERSE renderer and get photoreal views — no display,
no MuJoCo, just the same gsplat rasterizer the framework uses.

Usage:
  ~/discoverse_venv/bin/python scripts/discoverse_render_gs.py \
      --ply outputs/gs_111450.ply --out outputs/discoverse_gs_111450
"""
import argparse, os, sys
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# gaussian_renderer is installed in the discoverse venv
from gaussian_renderer.core.util_gau import load_ply
from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.batch_rasterization import batch_render

# camera convention copied verbatim from gaussian_renderer/simple_viewer.py
CAMERA_RMAT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


def cam_from_orbit(lookat, dist, azimuth_deg, elevation_deg):
    R = CAMERA_RMAT @ Rotation.from_euler(
        "xyz", [np.radians(elevation_deg), np.radians(azimuth_deg), 0.0]
    ).as_matrix()
    pos = lookat + dist * R[:, 2]
    return pos.reshape(1, 3), R.flatten().reshape(1, 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--dist_scale", type=float, default=1.6,
                    help="camera distance = dist_scale * scene radius")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[load] {args.ply}")
    gs = load_ply(args.ply).to_cuda()
    if gs.sh.dim() == 2:
        gs.sh = gs.sh.reshape(gs.sh.shape[0], -1, 3).contiguous()
    xyz = gs.xyz.detach().cpu().numpy()
    print(f"[load] {xyz.shape[0]} gaussians, SH dim {gs.sh.shape[1]}")

    # robust framing: ignore outlier gaussians via 5-95 percentile box
    lo, hi = np.percentile(xyz, 5, axis=0), np.percentile(xyz, 95, axis=0)
    lookat = 0.5 * (lo + hi)
    radius = float(np.linalg.norm(hi - lo)) * 0.5
    dist = max(radius * args.dist_scale, 0.5)
    print(f"[frame] lookat={np.round(lookat,2)} radius={radius:.2f} dist={dist:.2f}")

    # orbit: 8 azimuths x 2 elevations
    azimuths = list(range(0, 360, 45))
    elevations = [-15.0, 15.0]
    frames = []
    for el in elevations:
        for az in azimuths:
            cam_pos, cam_xmat = cam_from_orbit(lookat, dist, az, el)
            color, _ = batch_render(gs, cam_pos, cam_xmat,
                                    args.height, args.width, np.array([args.fov]))
            img = np.clip(color[0].detach().cpu().numpy(), 0, 1)
            img = (img * 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
            fn = os.path.join(args.out, f"view_el{int(el):+d}_az{az:03d}.png")
            cv2.imwrite(fn, img)
            frames.append(img)
            print(f"[render] {fn}  (nonblack px {(img.sum(2)>10).mean()*100:.1f}%)")

    # contact sheet (2 rows x 8 cols)
    h, w = frames[0].shape[:2]
    sheet = np.zeros((h * 2, w * 8, 3), np.uint8)
    for i, im in enumerate(frames):
        r, c = divmod(i, 8)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    sheet_fn = os.path.join(args.out, "contact_sheet.png")
    cv2.imwrite(sheet_fn, cv2.resize(sheet, (w * 8 // 2, h * 2 // 2)))
    print(f"[done] contact sheet -> {sheet_fn}")


if __name__ == "__main__":
    main()
