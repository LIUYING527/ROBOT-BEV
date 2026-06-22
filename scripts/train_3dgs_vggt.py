"""VGGT-Ω → 3DGS 训练器(改自 train_3dgs.py)。

输入 = VGGT-Ω 的输出(outputs/vggto_<s>/):
  - cameras.npz: extrinsic(N,3,4) world2cam, intrinsic(N,3,3) K(对应 688x384 处理分辨率)
  - recon.ply: 由 depth 反投影的稠密带色点云(VGGT 原始帧,与 cameras 同坐标系;
               注意用 recon.ply 不是 recon_aligned.ply——后者被重力旋转过,与 cameras 不同帧)
  - frames_zed/: 64 帧源图(sorted 后按索引 i 对齐 extrinsic[i])
COLMAP 那套换成这套;场景归一化/3-NN 初始化/gsplat 训练/ply 导出全部沿用。

环境变量: ITERS(默认7000) DOWNSCALE(默认1,K已是小分辨率) INIT_PTS(默认400000)
用法: ~/discoverse_venv/bin/python scripts/train_3dgs_vggt.py [session]
输出: outputs/gs_vggto_<s>.ply + .norm.npy
"""
import os
import sys
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import trimesh
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

SESSION = sys.argv[1] if len(sys.argv) > 1 else "114830"
VDIR = f"outputs/vggto_{SESSION}"
DEV = "cuda"
DS = int(os.environ.get("DOWNSCALE", "1"))
ITERS = int(os.environ.get("ITERS", "7000"))
INIT_PTS = int(os.environ.get("INIT_PTS", "400000"))
OUT = os.environ.get("OUT", f"outputs/gs_vggto_{SESSION}.ply")


def ssim(a, b):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    win = torch.ones(3, 1, 11, 11, device=a.device) / 121.0
    mu_a = F.conv2d(a, win, padding=5, groups=3)
    mu_b = F.conv2d(b, win, padding=5, groups=3)
    va = F.conv2d(a * a, win, padding=5, groups=3) - mu_a ** 2
    vb = F.conv2d(b * b, win, padding=5, groups=3) - mu_b ** 2
    vab = F.conv2d(a * b, win, padding=5, groups=3) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return s.mean()


def main():
    cams = np.load(os.path.join(VDIR, "cameras.npz"))
    extr, intr = cams["extrinsic"], cams["intrinsic"]          # (N,3,4) world2cam, (N,3,3) K
    N = len(extr)
    # K 对应的处理分辨率(cx*2, cy*2),所有帧一致
    W = int(round(float(np.median(intr[:, 0, 2])) * 2))         # ~688
    H = int(round(float(np.median(intr[:, 1, 2])) * 2))         # ~384
    image_names = sorted(glob.glob(os.path.join(VDIR, "frames_zed", "*")))
    assert len(image_names) == N, f"frames {len(image_names)} != cameras {N}"

    views = []
    for i in range(N):
        K = intr[i].astype(np.float32)
        V = np.eye(4, dtype=np.float32)
        V[:3, :3] = extr[i, :3, :3]
        V[:3, 3] = extr[i, :3, 3]
        views.append((image_names[i], K, V, W, H))
    print(f"[INFO] views {N}  W,H {W}x{H}  downscale {DS}  iters {ITERS}", flush=True)

    # 预缓存所有图到 GPU(resize 到 K 对应的 WxH)
    imgs = {}
    for name, K, V, W, H in views:
        bgr = cv2.imread(name)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W // DS, H // DS))
        imgs[name] = torch.tensor(rgb.copy(), dtype=torch.float32, device=DEV) / 255.0
    print(f"[INFO] 缓存图像 {len(imgs)} 张", flush=True)

    # 初始化高斯(VGGT recon.ply 稠密点,降采样 + 去离群)
    pc = trimesh.load(os.path.join(VDIR, "recon.ply"), process=False)
    P = np.asarray(pc.vertices, dtype=np.float32)
    Col = np.asarray(pc.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
    if len(P) > INIT_PTS:
        sel = np.random.choice(len(P), INIT_PTS, replace=False)
        P, Col = P[sel], Col[sel]
    med = np.median(P, 0)
    keep = np.linalg.norm(P - med, axis=1) < 5 * np.median(np.linalg.norm(P - med, axis=1))
    P, Col = P[keep], Col[keep]

    # 稳健去离群相机(VGGT 一般很干净,保留作保险)
    cams_all = np.array([-V[:3, :3].T @ V[:3, 3] for _, _, V, _, _ in views])
    cmed = np.median(cams_all, 0)
    cdist = np.linalg.norm(cams_all - cmed, axis=1)
    inlier = cdist < 5 * np.median(cdist)
    views = [v for v, k in zip(views, inlier) if k]
    print(f"[INFO] 剔除离群相机后 views {len(views)}/{len(inlier)}", flush=True)

    # 场景归一化(用相机中心,与 train_3dgs 一致)
    camc = np.array([-V[:3, :3].T @ V[:3, 3] for _, _, V, _, _ in views])
    center = camc.mean(0).astype(np.float32)
    s = float(np.linalg.norm(camc - center, axis=1).mean())
    P = (P - center) / s
    nv = []
    for name, K, V, W, H in views:
        R = V[:3, :3]; t = V[:3, 3]
        c = -R.T @ t
        cn = (c - center) / s
        Vn = np.eye(4, dtype=np.float32); Vn[:3, :3] = R; Vn[:3, 3] = -R @ cn
        nv.append((name, K, Vn, W, H))
    views = nv
    np.save(OUT + ".norm.npy", np.concatenate([center, [s]]))
    scene_scale = 1.0
    pt = torch.tensor(P, device=DEV)
    sub = pt[torch.randperm(len(pt))[:4000]]
    nn3 = torch.cdist(sub, pt).topk(4, largest=False).values[:, 1:].mean(dim=1)
    init_scale = float(nn3.median().clamp(min=0.005, max=0.08))
    print(f"[INFO] 初始高斯 {len(P)}  init_scale {init_scale:.4f}", flush=True)

    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.tensor(P, device=DEV)),
        "scales": torch.nn.Parameter(torch.log(torch.full((len(P), 3), init_scale, device=DEV))),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]], device=DEV).repeat(len(P), 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((len(P),), 0.1, device=DEV))),
        "colors": torch.nn.Parameter(torch.tensor(Col, device=DEV)),
    }).to(DEV)
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "colors": 2.5e-3}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k]}], eps=1e-15)
                  for k in params}

    strategy = DefaultStrategy(refine_stop_iter=int(ITERS * 0.7), verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    for step in range(ITERS):
        name, K, V, W, H = random.choice(views)
        Ks = torch.tensor(K, device=DEV).clone()
        Ks[:2] /= DS
        Ks = Ks[None]
        vm = torch.tensor(V, device=DEV)[None]
        Wd, Hd = W // DS, H // DS
        render, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            params["colors"], vm, Ks, Wd, Hd, packed=False)
        info["means2d"].retain_grad()
        gt = imgs[name]
        pred = render[0].clamp(0, 1)
        l1 = (pred - gt).abs().mean()
        a = pred.permute(2, 0, 1)[None]; b = gt.permute(2, 0, 1)[None]
        loss = 0.8 * l1 + 0.2 * (1 - ssim(a, b))
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == ITERS - 1:
            print(f"  step {step}  loss {float(loss):.4f}  N {params['means'].shape[0]}", flush=True)

    save_ply(params, OUT)
    print(f"[OK] 已保存 {OUT}  ({params['means'].shape[0]} 高斯)", flush=True)


def save_ply(params, path):
    xyz = params["means"].detach().cpu().numpy()
    n = len(xyz)
    f_dc = (params["colors"].detach().cpu().numpy() - 0.5) / 0.2820948
    opa = params["opacities"].detach().cpu().numpy().reshape(n, 1)
    scl = params["scales"].detach().cpu().numpy()
    rot = F.normalize(params["quats"], dim=-1).detach().cpu().numpy()
    fields = (["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
               "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    data = np.concatenate([xyz, np.zeros((n, 3), np.float32), f_dc, opa, scl, rot], axis=1).astype(np.float32)
    with open(path, "wb") as f:
        hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
        hdr += "".join("property float %s\n" % p for p in fields) + "end_header\n"
        f.write(hdr.encode())
        f.write(data.tobytes())


if __name__ == "__main__":
    main()
