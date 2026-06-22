"""自包含 3DGS 训练器(gsplat + DefaultStrategy + pycolmap)。

GitHub 连不上拿不到 gsplat examples,这里用 gsplat 库 API 自写:
COLMAP(sparse/1)位姿+稀疏点初始化 → 原始RGB监督(L1+SSIM)→ 致密化 → 导出标准3DGS .ply。
训练在 COLMAP 坐标系(任意尺度)直接做,无需米制。

环境变量: ITERS(默认7000) MAX_IMGS(0=全部) DOWNSCALE(默认2)
用法: python scripts/train_3dgs.py
输出: outputs/gs_111450.ply (SuperSplat 可打开转着看)
"""
import os
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pycolmap
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

DATA = "outputs/_colmap_111450"
MODEL = DATA + "/sparse/1"
DEV = "cuda"
DS = int(os.environ.get("DOWNSCALE", "2"))
ITERS = int(os.environ.get("ITERS", "7000"))
MAX_IMGS = int(os.environ.get("MAX_IMGS", "0"))
OUT = os.environ.get("OUT", "outputs/gs_111450.ply")


def ssim(a, b):
    # a,b: (1,3,H,W); 简化 SSIM(高斯窗)
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
    rec = pycolmap.Reconstruction(MODEL)
    views = []
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        K = np.array(cam.calibration_matrix(), dtype=np.float32)
        cfw = im.cam_from_world()
        V = np.eye(4, dtype=np.float32)
        V[:3, :3] = np.array(cfw.rotation.matrix())
        V[:3, 3] = np.array(cfw.translation)
        views.append((im.name, K, V, int(cam.width), int(cam.height)))
    views.sort(key=lambda x: x[0])
    if MAX_IMGS and len(views) > MAX_IMGS:
        views = views[:: max(1, len(views) // MAX_IMGS)][:MAX_IMGS]
    print(f"[INFO] views {len(views)}  downscale {DS}  iters {ITERS}", flush=True)

    # 预缓存所有图到 GPU
    imgs = {}
    for name, K, V, W, H in views:
        bgr = cv2.imread(os.path.join(DATA, "images", name))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if DS > 1:
            rgb = cv2.resize(rgb, (W // DS, H // DS))
        imgs[name] = torch.tensor(rgb.copy(), dtype=torch.float32, device=DEV) / 255.0
    print(f"[INFO] 缓存图像 {len(imgs)} 张", flush=True)

    # 初始化高斯(COLMAP稀疏点,去离群)
    P = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    Col = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0
    med = np.median(P, 0)
    keep = np.linalg.norm(P - med, axis=1) < 5 * np.median(np.linalg.norm(P - med, axis=1))
    P, Col = P[keep], Col[keep]

    # ⭐先剔除离群相机:COLMAP有~50个被误注册到极远(坐标上千)的相机,
    # 它们会污染归一化(center/s)并提供垃圾监督,必须先稳健去除。
    cams_all = np.array([-V[:3, :3].T @ V[:3, 3] for _, _, V, _, _ in views])
    cmed = np.median(cams_all, 0)
    cdist = np.linalg.norm(cams_all - cmed, axis=1)
    inlier = cdist < 5 * np.median(cdist)
    views = [v for v, k in zip(views, inlier) if k]
    print(f"[INFO] 剔除离群相机后 views {len(views)}/{len(inlier)}", flush=True)

    # ⭐场景归一化:COLMAP坐标任意尺度,不归一化means学习率会爆→发散。
    cams = np.array([-V[:3, :3].T @ V[:3, 3] for _, _, V, _, _ in views])
    center = cams.mean(0).astype(np.float32)
    s = float(np.linalg.norm(cams - center, axis=1).mean())
    P = (P - center) / s
    nv = []
    for name, K, V, W, H in views:
        R = V[:3, :3]; t = V[:3, 3]
        c = -R.T @ t
        cn = (c - center) / s
        Vn = np.eye(4, dtype=np.float32); Vn[:3, :3] = R; Vn[:3, 3] = -R @ cn
        nv.append((name, K, Vn, W, H))
    views = nv
    np.save(OUT + ".norm.npy", np.concatenate([center, [s]]))   # 反归一化用
    scene_scale = 1.0
    pt = torch.tensor(P, device=DEV)
    sub = pt[torch.randperm(len(pt))[:4000]]
    nn3 = torch.cdist(sub, pt).topk(4, largest=False).values[:, 1:].mean(dim=1)  # 每点3-NN均距
    init_scale = float(nn3.median().clamp(min=0.005, max=0.08))                  # 中位数+合理下限(近重复点会让均值塌到0)
    print(f"[INFO] 初始高斯 {len(P)}  scene_scale {scene_scale:.1f}  init_scale {init_scale:.4f}", flush=True)

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
    """标准 3DGS .ply(SuperSplat/常见查看器可读):RGB→SH0 f_dc,raw opacity/scale/rot。"""
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
