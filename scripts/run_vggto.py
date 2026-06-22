#!/usr/bin/env python3
"""无头跑 VGGT-Omega 重建 -> 导出 点云(PLY/GLB) + 相机位姿(npz) + 俯视预览(PNG)。

复刻 demo_gradio.run_model 的后处理(避免依赖 gradio)。
用法:
  CKPT=/path/vggt_omega_1b_512.pt \
  python run_vggto.py --target_dir outputs/vggto_114830 [--res 512] [--max_frames 64] [--dry_run]
--dry_run: 跳过加载权重(随机初始化),只验证端到端代码/显存/导出是否通。
"""
import os, sys, glob, time, argparse
import numpy as np

REPO = "/data/DongBaorong/BEVLOOK/robot_bev_sim/third_party/vggt-omega"
sys.path.insert(0, REPO)

import torch
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from visual_util import predictions_to_glb


def unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic):
    depth = depth_map[..., 0]
    n, h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = np.broadcast_to(x[None], (n, h, w)); y = np.broadcast_to(y[None], (n, h, w))
    fx = intrinsic[:, 0, 0][:, None, None]; fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]; cy = intrinsic[:, 1, 2][:, None, None]
    cam = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)
    R = extrinsic[:, :3, :3]; T = extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(R, (0, 2, 1)), cam - T[:, None, None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--max_frames", type=int, default=64)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    td = args.target_dir
    image_names = sorted(glob.glob(os.path.join(td, "images", "*")))
    if args.max_frames and len(image_names) > args.max_frames:
        idx = np.linspace(0, len(image_names) - 1, args.max_frames).astype(int)
        image_names = [image_names[i] for i in idx]
    print(f"[i] {len(image_names)} 帧, res={args.res}, dry_run={args.dry_run}")

    assert torch.cuda.is_available(), "需要 CUDA"
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = VGGTOmega().eval()
    if not args.dry_run:
        ckpt = os.environ.get("CKPT")
        assert ckpt and os.path.isfile(ckpt), f"CKPT 无效: {ckpt}"
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[i] 已载入权重: {ckpt}")
    else:
        print("[i] DRY RUN: 随机权重")
    model = model.to(dev)
    print(f"[t] 模型就绪 {time.time()-t0:.1f}s")

    images = load_and_preprocess_images(image_names, image_resolution=args.res).to(dev)
    print(f"[i] 输入张量 {tuple(images.shape)}")

    t1 = time.time()
    with torch.inference_mode():
        pred = model(images)
    torch.cuda.synchronize()
    print(f"[t] 前向 {time.time()-t1:.1f}s")

    extr, intr = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    pred["extrinsic"] = extr; pred["intrinsic"] = intr

    pnp = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().cpu().numpy()
            if v.shape[0] == 1:
                v = v[0]
            pnp[k] = v
    pnp["world_points_from_depth"] = unproject_depth_map_to_point_map(
        pnp["depth"], pnp["extrinsic"], pnp["intrinsic"])

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[m] 峰值显存 {peak:.2f} GB")

    # 导出
    os.makedirs(td, exist_ok=True)
    np.savez(os.path.join(td, "cameras.npz"),
             extrinsic=pnp["extrinsic"], intrinsic=pnp["intrinsic"],
             depth_conf=pnp.get("depth_conf"))
    scene = predictions_to_glb(pnp, max_points=400000)
    glb = os.path.join(td, "recon.glb"); scene.export(glb)
    print(f"[o] GLB -> {glb}")

    # PLY 点云(置信度过滤)
    pts = pnp["world_points_from_depth"].reshape(-1, 3)
    rgb = (pnp["images"].transpose(0, 2, 3, 1) if pnp["images"].ndim == 4 else pnp["images"])
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).reshape(-1, 3)
    conf = pnp["depth_conf"].reshape(-1)
    thr = np.quantile(conf, 0.3)
    m = conf > thr
    import trimesh
    pc = trimesh.PointCloud(vertices=pts[m], colors=rgb[m])
    ply = os.path.join(td, "recon.ply"); pc.export(ply)
    print(f"[o] PLY ({m.sum()} 点, conf>{thr:.3f}) -> {ply}")

    # 俯视 BEV 预览 (X-Z 平面)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = pts[m]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(p[:, 0], p[:, 2], s=0.2, c=rgb[m] / 255.0)
    ax.set_aspect("equal"); ax.set_title("VGGT-Omega 114830 zed top-down (X-Z)")
    bev = os.path.join(td, "bev_preview.png"); fig.savefig(bev, dpi=120); plt.close(fig)
    print(f"[o] BEV 预览 -> {bev}")
    print(f"[t] 总计 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
