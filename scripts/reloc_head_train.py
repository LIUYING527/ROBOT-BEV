"""精对准回归头 (0812, 0805调研P1/0806周报P1落地): 把GS render-and-compare重定位实时化。
动机: gs_relocate.py的梯度优化~43s/次(服务器GPU), Orin跑不了 → 场景内PoseNet式回归头:
  输入 = 观测深度图(部署=zed z通道×OBS_SCALE; 训练=真位姿渲染ED+zed式噪声) + believed位姿
  输出 = Δ=(dx,dy,dyaw) (believed相机系: x右 y前) —— 单次前向<10ms, Orin可跑。
数据: COLMAP帧位姿(on-path GT) + ±0.2m抖动位姿渲染ED; 每真位姿采样K个believed扰动
  (两档: ±0.5m/±25°粗 = 收敛域边缘; ±0.15m/±8°细 = 精修区), 目标=believed→true的残差。
⚠口径: 全部世界米(0.70×世界系内自洽); 部署时zed真实米×0.70→世界米, 与GSReloc同一约定。
⚠自洽性: 训练/验证GT都是COLMAP位姿+同一GS地图 → 精度数字是"头逼近优化器"的上限,
  独立真实精度仍以真zed评测5.6cm为准(别over-claim)。

用法:
  # 生成数据(GPU, ~10min):
  CUDA_VISIBLE_DEVICES=1 ~/discoverse_venv/bin/python scripts/reloc_head_train.py gen
  # 训练+验证(GPU, ~30min):
  CUDA_VISIBLE_DEVICES=1 ~/discoverse_venv/bin/python scripts/reloc_head_train.py train
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "outputs", "reloc_head_0812")
IMG_H, IMG_W = 90, 160          # CNN输入分辨率(深度图)
PERTURBS = [("coarse", 0.50, 25.0, 6), ("fine", 0.15, 8.0, 6)]   # (名, r_m, yaw_deg, 每真位姿K个)
JITTER = 2                       # 每COLMAP帧额外抖动的真位姿数(±0.2m/±6°)


def gen(args):
    import torch
    from gs_relocate import GSReloc
    g = GSReloc(args.session, "data/lab/110744")
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    frames = np.linspace(0, len(g.names) - 1, args.max_frames).astype(int) if args.max_frames else np.arange(len(g.names))
    recs = []
    t0 = time.time()
    for fi, j in enumerate(frames):
        C_gt, R_gt = g.gt_world_pose(int(j))
        yaw_gt = np.arctan2(R_gt[1, 2], R_gt[0, 2])
        # 真位姿集 = COLMAP帧 + 抖动
        poses = [(C_gt, yaw_gt)]
        for _ in range(JITTER):
            a = rng.uniform(0, 2 * np.pi); rr = 0.2 * np.sqrt(rng.uniform())
            poses.append((C_gt + np.array([rr * np.cos(a), rr * np.sin(a), 0.0]),
                          yaw_gt + np.radians(rng.uniform(-6, 6))))
        for (C_t, yaw_t) in poses:
            # 真位姿观测(世界米): 0813真zed实测教训=合成噪声(σ1cm+1%)域差太大(自洽5.8cm vs 真zed 43cm)
            # → COLMAP帧位姿直接用真zed观测(部署域本域); 抖动位姿没真观测才渲染ED+噪声。
            is_real = (C_t is C_gt)
            if is_real:
                d_t, m_t = g.obs_depth_m(g.names[int(j)], IMG_H, IMG_W)
                if d_t is None or int(m_t.sum()) < 0.35 * IMG_H * IMG_W:
                    break                                    # 该帧无真zed观测, 抖动样本也没意义(同位姿域)
                obs_img = d_t.cpu().numpy(); valid = m_t.cpu().numpy()
            else:
                # 抖动位姿渲染ED(世界米) → 伪zed观测(噪声+丢点+窗) —— 与lab_mission RELOC同配方
                # 0813实测教训: 强制-22°俯角=域差(采集帧真实-20~-29°, 差6°头就崩) → 用帧真实姿态,
                # yaw扰动绕世界z轴叠加在帧真实俯仰/翻滚上(Ryaw(Δ)@R_gt), 与真zed评测协议同分布。
                import torch as _t
                Ryaw = np.array([[np.cos(yaw_t - yaw_gt), -np.sin(yaw_t - yaw_gt), 0.0],
                                 [np.sin(yaw_t - yaw_gt),  np.cos(yaw_t - yaw_gt), 0.0],
                                 [0.0, 0.0, 1.0]])
                R_t = Ryaw @ R_gt
                with _t.no_grad():
                    V = g.viewmat_from_world(_t.tensor(C_t, device="cuda", dtype=_t.float32),
                                             _t.tensor(R_t, device="cuda", dtype=_t.float32))
                    ed, al = g.render_ed(V, IMG_H, IMG_W)
                    ed = ed.cpu().numpy(); al = al.cpu().numpy()
                sig = 0.01 + 0.01 * ed
                d_n = ed + rng.normal(0, sig).astype(np.float32)
                valid = (al > 0.6) & (ed > 0.3) & (ed < 12.0) & (rng.rand(IMG_H, IMG_W) > 0.05)
                if valid.sum() < 0.35 * IMG_H * IMG_W:
                    continue                                # 视野太差(面壁/看穿)的样本不要
                obs_img = np.where(valid, d_n, 0.0).astype(np.float32)
            # K个believed扰动(两档); 真观测样本加倍采样(部署域权重↑)
            for (pn, r_m, psi_d, K) in PERTURBS:
                for _ in range(K * (2 if is_real else 1)):
                    a = rng.uniform(0, 2 * np.pi); rr = r_m * np.sqrt(rng.uniform())
                    dxw, dyw = rr * np.cos(a), rr * np.sin(a)
                    dyaw = np.radians(rng.uniform(-psi_d, psi_d))
                    yaw_b = yaw_t + dyaw
                    # 目标: believed系下的(true-believed)修正量, x=右 y=前。
                    # 本项目约定(同_cam_CR): fwd=(cos,sin), right=(sin,-cos);
                    # body→world: w = M @ body, M的列=[right, fwd] = [[sin,cos],[-cos,sin]]
                    cb_, sb_ = np.cos(yaw_b), np.sin(yaw_b)
                    M = np.array([[sb_, cb_], [-cb_, sb_]])         # body→world (列=[right, fwd])
                    bx_, by_ = np.linalg.solve(M, np.array([-dxw, -dyw]))
                    recs.append(dict(obs=obs_img, mask=valid,
                                     bel=np.array([C_t[0] + dxw, C_t[1] + dyw, yaw_b], np.float32),
                                     delta=np.array([bx_, by_, -dyaw], np.float32),   # believed系(右,前,dyaw修正)
                                     frame=int(j), lev=pn))
        if fi % 40 == 0:
            print(f"[gen] {fi}/{len(frames)} samples={len(recs)} {time.time()-t0:.0f}s", flush=True)
    obs = np.stack([r["obs"] for r in recs]); msk = np.stack([r["mask"] for r in recs])
    bel = np.stack([r["bel"] for r in recs]); dlt = np.stack([r["delta"] for r in recs])
    frm = np.array([r["frame"] for r in recs])
    np.savez_compressed(os.path.join(OUT, "data.npz"), obs=obs, mask=msk, bel=bel, delta=dlt, frame=frm)
    print(f"[gen] ✅ {len(recs)}样本 -> {OUT}/data.npz ({time.time()-t0:.0f}s)", flush=True)


def train(args):
    import torch, torch.nn as nn
    dev = "cuda"
    dz = np.load(os.path.join(OUT, "data.npz"))
    obs, msk, bel, dlt, frm = dz["obs"], dz["mask"], dz["bel"], dz["delta"], dz["frame"]
    # 按帧划分train/val(同帧扰动不跨边)
    uf = np.unique(frm); rng = np.random.RandomState(0); rng.shuffle(uf)
    vf = set(uf[: max(1, int(0.2 * len(uf)))].tolist())
    vi = np.array([f in vf for f in frm]); ti = ~vi
    print(f"[train] {ti.sum()}训/{vi.sum()}验, 帧{len(uf)}", flush=True)

    SCALE = np.array([0.5, 0.5, np.radians(25)], np.float32)          # 目标归一化

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(          # 2ch(深度+mask) 90x160
                nn.Conv2d(2, 32, 5, 2, 2), nn.ReLU(),      # 45x80
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),     # 23x40
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),    # 12x20
                nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(),   # 6x10
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.pose = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
            self.fuse = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, 3))
        def forward(self, img, ps):
            return self.fuse(torch.cat([self.cnn(img), self.pose(ps)], 1))

    net = Head().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.SmoothL1Loss(beta=0.3)

    def batch(idx, bs, shuf=False):
        order = np.random.permutation(len(idx)) if shuf else np.arange(len(idx))
        for s in range(0, len(idx), bs):
            ii = idx[order[s:s + bs]]
            img = np.stack([obs[ii], msk[ii].astype(np.float32)], 1)          # (B,2,H,W)
            img = torch.tensor(img, device=dev)
            ps = bel[ii].copy(); ps[:, 2] = np.sin(ps[:, 2])
            ps = np.concatenate([ps, np.cos(bel[ii][:, 2:3])], 1)             # (x,y,sin,cos)
            yield img, torch.tensor(ps, device=dev), torch.tensor(dlt[ii] / SCALE, device=dev), ii

    ti_idx, vi_idx = np.where(ti)[0], np.where(vi)[0]
    for ep in range(args.epochs):
        net.train(); tl = 0.0; nb = 0
        for img, ps, y, _ in batch(ti_idx, args.bs, shuf=True):
            opt.zero_grad(); loss = lossf(net(img, ps), y); loss.backward(); opt.step()
            tl += float(loss); nb += 1
        sch.step()
        if ep % 5 == 4 or ep == args.epochs - 1:
            net.eval(); errs = []
            with torch.no_grad():
                for img, ps, y, ii in batch(vi_idx, args.bs):
                    pred = net(img, ps).cpu().numpy() * SCALE
                    gt = dlt[ii]
                    errs.append(np.abs(pred - gt))
            e = np.concatenate(errs)
            em = np.median(e, 0); e9 = np.percentile(e, 90, axis=0)
            print(f"[ep{ep:3d}] loss={tl/nb:.4f} | val中位 |dx|{em[0]*100:.1f}cm |dy|{em[1]*100:.1f}cm "
                  f"|dyaw|{np.degrees(em[2]):.2f}° | p90 {e9[0]*100:.1f}/{e9[1]*100:.1f}cm {np.degrees(e9[2]):.1f}°", flush=True)
    torch.save(net.state_dict(), os.path.join(OUT, "reloc_head.pt"))
    print(f"[train] ✅ ckpt -> {OUT}/reloc_head.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "train"])
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0, help=">0则只用这么多帧(调试)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()
    {"gen": gen, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
