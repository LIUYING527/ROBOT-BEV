"""VGGT-Ω → 3DGS 训练器(改自 train_3dgs.py)。

输入 = VGGT-Ω 的输出(outputs/vggto_<s>/):
  - cameras.npz: extrinsic(N,3,4) world2cam, intrinsic(N,3,3) K(对应 688x384 处理分辨率)
  - recon.ply: 由 depth 反投影的稠密带色点云(VGGT 原始帧,与 cameras 同坐标系;
               注意用 recon.ply 不是 recon_aligned.ply——后者被重力旋转过,与 cameras 不同帧)
  - frames_zed/: N 帧源图(sorted 后按索引 i 对齐 extrinsic[i]),原图 1280x720

质量旋钮(对应 recon-quality-knobs 记忆里的糊/重影 4 大根因):
  1. 全分辨率训练: UPSAMPLE>1 把 K 与渲染分辨率从 688x384 上采到接近原图
     (源 JPG 是 1280x720,UPSAMPLE≈1.86 即读全分辨率细节)。
  3. SH 视角相关颜色: SH_DEGREE>0 用球谐系数表达高光/反光,压制"影子高斯"重影。
     导出 f_dc + f_rest(通道优先),与 gaussian_renderer/util_gau(max_sh_degree=3)兼容。
  4. 剪枝: 导出前剔除低不透明度 + 超大尺度的浮点高斯(PRUNE=1)。
  (旋钮 2 加密帧在抽帧/run_vggto 阶段控制,这里不涉及。)

环境变量:
  ITERS(默认7000)  DOWNSCALE(默认1)  INIT_PTS(默认400000)
  UPSAMPLE(默认1.0,>1 升训练分辨率)  SH_DEGREE(默认0,设3 启 SH)
  PRUNE(默认1)  PRUNE_OP(默认0.05)  PRUNE_SCALE(默认0.3,归一化场景单位)
  OUT(默认 outputs/gs_vggto_<s>.ply)
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
UPSAMPLE = float(os.environ.get("UPSAMPLE", "1.0"))
SH_DEGREE = int(os.environ.get("SH_DEGREE", "0"))
PRUNE = os.environ.get("PRUNE", "1") == "1"
PRUNE_OP = float(os.environ.get("PRUNE_OP", "0.05"))
PRUNE_SCALE = float(os.environ.get("PRUNE_SCALE", "0.3"))
OUT = os.environ.get("OUT", f"outputs/gs_vggto_{SESSION}.ply")
RES_SCALE = UPSAMPLE / DS                       # 渲染/GT 相对 K 处理分辨率的缩放
C0 = 0.28209479177387814                        # SH DC 系数 (1/(2*sqrt(pi)))


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
    Wr, Hr = max(1, round(W * RES_SCALE)), max(1, round(H * RES_SCALE))   # 实际训练分辨率
    image_names = sorted(glob.glob(os.path.join(VDIR, "frames_zed", "*")))
    # VGGT 按 --max_frames 对 frames_zed 做 linspace 采样;若帧多于相机, 同样映射回对应帧
    if len(image_names) != N:
        idx = np.linspace(0, len(image_names) - 1, N).astype(int)
        image_names = [image_names[i] for i in idx]
    assert len(image_names) == N, f"frames {len(image_names)} != cameras {N}"

    views = []
    for i in range(N):
        K = intr[i].astype(np.float32)
        V = np.eye(4, dtype=np.float32)
        V[:3, :3] = extr[i, :3, :3]
        V[:3, 3] = extr[i, :3, 3]
        views.append((image_names[i], K, V, W, H))
    print(f"[INFO] views {N}  K分辨率 {W}x{H}  训练分辨率 {Wr}x{Hr}  "
          f"UPSAMPLE {UPSAMPLE}  SH {SH_DEGREE}  iters {ITERS}", flush=True)

    # 预缓存所有图到 GPU(resize 到训练分辨率 Wr x Hr;源图 1280x720,上采=读更全细节)
    imgs = {}
    for name, K, V, W, H in views:
        bgr = cv2.imread(name)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (Wr, Hr), interpolation=cv2.INTER_AREA)
        imgs[name] = torch.tensor(rgb.copy(), dtype=torch.float32, device=DEV) / 255.0
    print(f"[INFO] 缓存图像 {len(imgs)} 张 @ {Wr}x{Hr}", flush=True)

    # 初始化高斯(VGGT recon.ply 稠密点,降采样 + 去离群)
    # 初始点云: 默认 VGGT recon.ply;设 INIT_PLY 可用 ZED 深度融合云(fuse_zed.py 产出,更稠/度量干净)
    init_ply = os.environ.get("INIT_PLY", os.path.join(VDIR, "recon.ply"))
    print(f"[INFO] 初始点云 = {init_ply}", flush=True)
    pc = trimesh.load(init_ply, process=False)
    P = np.asarray(pc.vertices, dtype=np.float32)
    if getattr(pc, "colors", None) is not None and len(pc.colors):
        Col = np.asarray(pc.colors, dtype=np.float32)[:, :3]            # open3d ply (0..1)
    else:
        Col = np.asarray(pc.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
    if Col.max() > 1.5:
        Col = Col / 255.0
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

    n0 = len(P)
    pdict = {
        "means": torch.nn.Parameter(torch.tensor(P, device=DEV)),
        "scales": torch.nn.Parameter(torch.log(torch.full((n0, 3), init_scale, device=DEV))),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]], device=DEV).repeat(n0, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n0,), 0.1, device=DEV))),
    }
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2}
    if SH_DEGREE > 0:
        Ksh = (SH_DEGREE + 1) ** 2
        sh0 = ((Col - 0.5) / C0).astype(np.float32)            # DC 系数 = 逆 SH 颜色
        pdict["sh0"] = torch.nn.Parameter(torch.tensor(sh0, device=DEV)[:, None, :])     # (N,1,3)
        pdict["shN"] = torch.nn.Parameter(torch.zeros(n0, Ksh - 1, 3, device=DEV))       # (N,K-1,3)
        lrs["sh0"] = 2.5e-3
        lrs["shN"] = 2.5e-3 / 20.0
    else:
        pdict["colors"] = torch.nn.Parameter(torch.tensor(Col, device=DEV))
        lrs["colors"] = 2.5e-3
    params = torch.nn.ParameterDict(pdict).to(DEV)
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k]}], eps=1e-15)
                  for k in params}

    strategy = DefaultStrategy(refine_stop_iter=int(ITERS * 0.7), verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    # SH 阶数 warmup: 每 ITERS/(deg+1) 步升一阶(标准 3DGS 做法,稳住早期优化)
    sh_step = max(1, ITERS // (SH_DEGREE + 1)) if SH_DEGREE > 0 else ITERS

    for step in range(ITERS):
        name, K, V, W, H = random.choice(views)
        Ks = torch.tensor(K, device=DEV).clone()
        Ks[:2] *= RES_SCALE                                    # K 同步缩放到训练分辨率
        Ks = Ks[None]
        vm = torch.tensor(V, device=DEV)[None]
        if SH_DEGREE > 0:
            cur_deg = min(SH_DEGREE, step // sh_step)
            colors = torch.cat([params["sh0"], params["shN"]], dim=1)   # (N,K,3)
            render, alpha, info = rasterization(
                params["means"], F.normalize(params["quats"], dim=-1),
                torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                colors, vm, Ks, Wr, Hr, sh_degree=cur_deg, packed=False)
        else:
            render, alpha, info = rasterization(
                params["means"], F.normalize(params["quats"], dim=-1),
                torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                params["colors"], vm, Ks, Wr, Hr, packed=False)
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

    if PRUNE:
        prune_inplace(params)
    save_ply(params, OUT, SH_DEGREE)
    print(f"[OK] 已保存 {OUT}  ({params['means'].shape[0]} 高斯, SH{SH_DEGREE})", flush=True)


def prune_inplace(params):
    """剔除低不透明度 + 超大尺度的浮点高斯(就地截断 params,仅用于导出)。"""
    with torch.no_grad():
        op = torch.sigmoid(params["opacities"])
        sc = torch.exp(params["scales"]).max(dim=1).values
        keep = (op > PRUNE_OP) & (sc < PRUNE_SCALE)
        n_before = keep.numel(); n_keep = int(keep.sum())
        idx = torch.where(keep)[0]
        for k in list(params.keys()):
            params[k] = torch.nn.Parameter(params[k].detach()[idx])
    print(f"[INFO] 剪枝: {n_keep}/{n_before} 保留 "
          f"(op>{PRUNE_OP}, scale<{PRUNE_SCALE})", flush=True)


def save_ply(params, path, sh_degree):
    xyz = params["means"].detach().cpu().numpy()
    n = len(xyz)
    if sh_degree > 0:
        # sh0 (N,1,3) -> f_dc (N,3);  shN (N,K-1,3) -> f_rest (N,3*(K-1)) 通道优先
        f_dc = params["sh0"].detach().cpu().numpy().reshape(n, 3)
        shN = params["shN"].detach().cpu().numpy()                    # (N,K-1,3)
        f_rest = shN.transpose(0, 2, 1).reshape(n, -1)                # 通道优先, 对齐 util_gau
    else:
        f_dc = (params["colors"].detach().cpu().numpy() - 0.5) / C0
        f_rest = np.zeros((n, 0), np.float32)
    opa = params["opacities"].detach().cpu().numpy().reshape(n, 1)
    scl = params["scales"].detach().cpu().numpy()
    rot = F.normalize(params["quats"], dim=-1).detach().cpu().numpy()
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    fields += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate(
        [xyz, np.zeros((n, 3), np.float32), f_dc, f_rest, opa, scl, rot], axis=1).astype(np.float32)
    with open(path, "wb") as f:
        hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
        hdr += "".join("property float %s\n" % p for p in fields) + "end_header\n"
        f.write(hdr.encode())
        f.write(data.tobytes())


if __name__ == "__main__":
    main()
