"""GS render-and-compare 重定位 (GSplatLoc路线, 免训练, 治'高精度导航'的定位环节)。
观测=zed度量深度(pointcloud npz, 米) + 带误差初始位姿 → 在3DGS地图上梯度优化SE(3)位姿,
使gsplat渲染期望深度(ED)与观测深度一致 → 输出精修位姿。
实验协议: 取COLMAP注册帧(其外参=GT), 加已知扰动(Δx,Δy~U(-r,r), Δyaw~U(-ψ,ψ)),
coarse-to-fine Adam优化, 测收敛后位置/旋转误差与成功率。

用法:
  CUDA_VISIBLE_DEVICES=6 ~/discoverse_venv/bin/python scripts/gs_relocate.py \
      colmapjoint_110744s2 --raw data/lab/110744 --frames 24 --seed 0
  # 单帧调试:
  ... --frames 1 --idx 200 --viz
"""
import os, sys, json, argparse, glob
import numpy as np
import torch
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from capture_fpv import _load_gauss  # 复用高斯加载(ply+norm)

DEV = "cuda"


def exp_so3(w):
    """axis-angle (...,3) → Rot (...,3,3), Rodrigues, torch可微。"""
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    K = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(w.shape[:-1] + (3, 3))
    st, ct = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return I + st * K + (1 - ct) * (K @ K)


class GSReloc:
    def __init__(self, session, raw_dir):
        import pycolmap
        ply = os.path.join(ROOT, "outputs", f"gs_vggto_{session}.ply")
        self.means, self.quats, self.scales, self.opac, self.colors, self.deg = _load_gauss(ply)
        Wz = np.load(os.path.join(ROOT, "outputs", f"sim_world_{session}.npz"))
        self.Rz = torch.tensor(Wz["Rz"], device=DEV, dtype=torch.float32)   # world->norm 旋转(常量)
        self.k = float(Wz["k"]); self.s_norm = float(Wz["s_norm"])
        self.t_world = torch.tensor(Wz["t_world"], device=DEV, dtype=torch.float32)
        self.center = torch.tensor(Wz["center"], device=DEV, dtype=torch.float32)
        # COLMAP GT: 外参/内参/帧名
        short = session[len("colmapjoint_"):] if session.startswith("colmapjoint_") else session
        rec = None
        for sub in ("sparse/2", "sparse/1", "sparse/0"):
            d = os.path.join(ROOT, "outputs", f"_colmap_joint_{short}", sub)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "images.bin")) or \
               os.path.isdir(d) and os.path.exists(os.path.join(d, "images.txt")):
                rec = pycolmap.Reconstruction(d); break
        if rec is None:
            rec = pycolmap.Reconstruction(os.path.join(ROOT, "outputs", f"_colmap_joint_{short}", "sparse", "0"))
        ims = sorted(rec.images.values(), key=lambda im: im.name)
        self.names = [im.name for im in ims]
        self.Vs = []   # world(colmap)->cam
        for im in ims:
            V = np.eye(4)
            V[:3, :3] = np.array(im.cam_from_world().rotation.matrix())
            V[:3, 3] = np.array(im.cam_from_world().translation)
            self.Vs.append(V)
        self.Vs = np.stack(self.Vs)
        cam = max(rec.cameras.values(), key=lambda c: c.width * c.height)
        self.K0 = np.array(cam.calibration_matrix(), np.float64)
        self.W0, self.H0 = int(cam.width), int(cam.height)
        self.raw_pc = os.path.join(ROOT, raw_dir, "pointcloud", "zed")
        # 0805诊断结论: 世界尺度比真实小~0.70×(align cam_height=1.2假设错, 真实~1.71m),
        # zed观测深度(真实米) × obs_scale → 世界米, 与渲染ED同尺度 (gs_reloc_diag.py实测b=-0.30)
        self.obs_scale = float(os.environ.get("OBS_SCALE", "0.70"))
        # 常量(torch)
        self.RzT = self.Rz.T.contiguous()

    # ---------- 坐标映射: 对齐世界(米) <-> colmap ----------
    def viewmat_from_world(self, C_w, R_c2w):
        """C_w (3,) 世界相机中心(米), R_c2w (3,3) cam->world(世界系) → colmap viewmat (4,4). 全torch可微。"""
        C_col = self.s_norm * (self.RzT @ (C_w - self.t_world)) / self.k + self.center
        R_c2col = self.RzT @ R_c2w
        Rv = R_c2col.T
        V = torch.eye(4, device=DEV)
        V[:3, :3] = Rv; V[:3, 3] = -Rv @ C_col
        return V

    def gt_world_pose(self, j):
        """COLMAP帧j → (C_w 米, R_c2w世界系) numpy。"""
        V = self.Vs[j]
        Rc, tc = V[:3, :3], V[:3, 3]
        C_col = -Rc.T @ tc
        R_c2w_col = Rc.T
        Rz = self.Rz.cpu().numpy()
        C_w = self.t_world.cpu().numpy() + (self.k / self.s_norm) * (Rz @ (C_col - self.center.cpu().numpy()))
        R_c2w = Rz @ R_c2w_col
        return C_w, R_c2w

    # ---------- 观测深度 ----------
    def obs_depth_from_z(self, z_real_m, Ho, Wo):
        """zed点云z通道(真实米, 720x1280) → (Ho,Wo) 世界米深度+valid掩码。服务器/仿真共用同一处理链。"""
        d = np.asarray(z_real_m, np.float32) * self.obs_scale          # 真实米 → 世界米
        valid = np.isfinite(d) & (d > 0.3) & (d < 12.0)
        d = np.where(valid, d, 0.0)
        ds = cv2.resize(d * valid, (Wo, Ho), interpolation=cv2.INTER_AREA)
        vs = cv2.resize(valid.astype(np.float32), (Wo, Ho), interpolation=cv2.INTER_AREA)
        m = vs > 0.6
        ds = np.where(m, ds / np.maximum(vs, 1e-6), 0.0).astype(np.float32)
        return torch.tensor(ds, device=DEV), torch.tensor(m, device=DEV)

    def obs_depth_m(self, name, Ho, Wo):
        """zed npz → (Ho,Wo) 米制z深度 + valid掩码。xyz单位mm, z沿光轴前向。"""
        p = os.path.join(self.raw_pc, name.replace(".jpg", ".npz"))
        if not os.path.exists(p):
            return None, None
        d = np.load(p)["xyzrgba"][..., 2].astype(np.float32) / 1000.0   # (720,1280) m
        d = d * self.obs_scale                                          # 真实米 → 世界米(尺度校正)
        valid = np.isfinite(d) & (d > 0.3) & (d < 12.0)
        d = np.where(valid, d, 0.0)
        # 块内有效均值降采样(避边缘混深): 深度与计数分别area缩
        ds = cv2.resize(d * valid, (Wo, Ho), interpolation=cv2.INTER_AREA)
        vs = cv2.resize(valid.astype(np.float32), (Wo, Ho), interpolation=cv2.INTER_AREA)
        m = vs > 0.6
        ds = np.where(m, ds / np.maximum(vs, 1e-6), 0.0).astype(np.float32)
        return torch.tensor(ds, device=DEV), torch.tensor(m, device=DEV)

    # ---------- 渲染 ----------
    def render_ed(self, V, Ho, Wo):
        from gsplat import rasterization
        K = torch.tensor(self.K0, device=DEV, dtype=torch.float32)
        K[0] *= Wo / self.W0; K[1] *= Ho / self.H0
        out, alpha, _ = rasterization(self.means, self.quats, self.scales, self.opac, self.colors,
                                      V[None], K[None], Wo, Ho, sh_degree=self.deg,
                                      near_plane=0.01, far_plane=1e4, render_mode="RGB+ED")
        ed = out[0, :, :, 3] * (self.k / self.s_norm)     # colmap单位 → 米
        return ed, alpha[0, :, :, 0]

    # ---------- 优化 ----------
    def optimize(self, j, init_Cw, init_R, iters=((320, 180, 110), (640, 360, 70)),
                 lr_t=0.02, lr_w=0.008, trim=0.9, log=False, obs=None):
        """从(init_Cw, init_R)出发, 对帧j的zed观测深度做coarse-to-fine位姿优化。返回(Cw, R, loss轨迹)。
        obs: 预计算观测 {(Wo,Ho):(d,m)} 字典(如闭环仿真的伪传感器), 提供时跳过帧加载。"""
        t_w = torch.tensor(init_Cw, device=DEV, dtype=torch.float32).requires_grad_(True)
        w = torch.zeros(3, device=DEV, dtype=torch.float32).requires_grad_(True)
        R0 = torch.tensor(init_R, device=DEV, dtype=torch.float32)
        opt = torch.optim.Adam([{"params": [t_w], "lr": lr_t}, {"params": [w], "lr": lr_w}])
        trace = []
        for (Wo, Ho, it) in iters:
            if obs is not None:
                d_obs, m_obs = obs[(Wo, Ho)]
            else:
                d_obs, m_obs = self.obs_depth_m(self.names[j], Ho, Wo)
            if d_obs is None or m_obs.sum() < 500:
                return None, None, trace
            for i in range(it):
                opt.zero_grad()
                R = exp_so3(w) @ R0
                V = self.viewmat_from_world(t_w, R)
                ed, al = self.render_ed(V, Ho, Wo)
                m = m_obs & (al > 0.6) & (ed > 0.05) & (ed < 15.0)
                if m.sum() < 200:
                    break
                r = (ed - d_obs)[m]
                # IRLS-lite: 每步砍最差(1-trim)残差(抗floater/边缘混深/动态)
                thr = torch.quantile(r.abs().detach(), trim)
                r = r[r.abs() <= thr]
                loss = torch.sqrt(r * r + 1e-4).mean()     # Charbonnier
                loss.backward()
                opt.step()
                trace.append(float(loss.detach()))
                if log and i % 25 == 0:
                    print(f"    it{i:3d} loss={trace[-1]:.4f} m={int(m.sum())}", flush=True)
        with torch.no_grad():
            R = exp_so3(w) @ R0
        return t_w.detach().cpu().numpy().astype(np.float64), R.cpu().numpy().astype(np.float64), trace


def yaw_of(R):
    f = R[:, 2]
    return float(np.degrees(np.arctan2(f[1], f[0])))


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", default="colmapjoint_110744s2")
    ap.add_argument("--raw", default="data/lab/110744")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--idx", type=int, default=-1, help="单帧调试指定COLMAP帧号")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", default="0.15,8;0.3,15;0.5,25", help="r_m,yaw_deg 分号分隔三档")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()
    g = GSReloc(args.session, args.raw)
    print(f"[reloc] {len(g.names)}帧, K0={g.W0}x{g.H0}, ply={args.session}", flush=True)

    # 选帧: 均匀采样且观测深度足够
    if args.idx >= 0:
        idxs = [args.idx]
    else:
        cand = np.linspace(0, len(g.names) - 1, args.frames * 3).astype(int)
        idxs = []
        for j in cand:
            d, m = g.obs_depth_m(g.names[j], 180, 320)
            if m is not None and m.sum() > 3000:
                idxs.append(int(j))
            if len(idxs) >= args.frames:
                break
    print(f"[reloc] 有效评测帧 {len(idxs)}: {idxs[:8]}...", flush=True)

    levels = []
    for tok in args.levels.split(";"):
        r, psi = tok.split(","); levels.append((float(r), float(psi)))
    rng = np.random.RandomState(args.seed)
    outdir = os.path.join(ROOT, "outputs", f"gs_relocate_{args.session.replace('colmapjoint_', '')}")
    os.makedirs(outdir, exist_ok=True)
    results = []
    for (r_m, psi_d) in levels:
        for j in idxs:
            C_gt, R_gt = g.gt_world_pose(j)
            # 圆盘均匀扰动 + yaw扰动
            a = rng.uniform(0, 2 * np.pi); rr = r_m * np.sqrt(rng.uniform(0, 1))
            dx, dy = rr * np.cos(a), rr * np.sin(a)
            dyaw = np.radians(rng.uniform(-psi_d, psi_d))
            c, s = np.cos(dyaw), np.sin(dyaw)
            Rp = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
            C0 = C_gt + np.array([dx, dy, 0.0]); R_init = Rp @ R_gt
            pos0 = float(np.hypot(dx, dy)); yaw0 = abs(float(np.degrees(dyaw)))
            C1, R1, trace = g.optimize(j, C0, R_init, log=args.log)
            if C1 is None:
                print(f"  L({r_m},{psi_d}) f#{j}: 观测不足, 跳过", flush=True); continue
            pos1 = float(np.linalg.norm(C1 - C_gt))
            e_cam = R_gt.T @ (C1 - C_gt)                      # 相机系分解: x横向 y竖直 z沿光轴(深度)
            rot1 = rot_err_deg(R1, R_gt)
            yaw1 = abs(yaw_of(R1) - yaw_of(R_gt)); yaw1 = min(yaw1, 360 - yaw1)
            rec = dict(level_r=r_m, level_psi=psi_d, frame=int(j),
                       init_pos=round(pos0, 4), init_yaw=round(yaw0, 3),
                       pos_err=round(pos1, 4), rot_err=round(rot1, 3), yaw_err=round(yaw1, 3),
                       lat_cm=round(float(abs(e_cam[0])) * 100, 1), fwd_cm=round(float(abs(e_cam[2])) * 100, 1),
                       horiz_cm=round(float(np.hypot(*(C1 - C_gt)[:2])) * 100, 1),
                       vert_cm=round(float(abs((C1 - C_gt)[2])) * 100, 1),
                       loss0=round(trace[0], 4) if trace else None,
                       loss1=round(trace[-1], 4) if trace else None)
            results.append(rec)
            print(f"  L(±{r_m}m,±{psi_d}°) f#{j:3d}: init({pos0:.3f}m,{yaw0:.1f}°) → "
                  f"err({pos1*100:.1f}cm, yaw{yaw1:.2f}°, rot{rot1:.2f}°) loss {rec['loss0']}→{rec['loss1']}", flush=True)
            if args.viz:
                _viz(g, j, C0, R_init, C1, R1, C_gt, R_gt, outdir)
    # 汇总
    print("\n===== 汇总 =====")
    summ = {}
    for (r_m, psi_d) in levels:
        rs = [x for x in results if x["level_r"] == r_m]
        if not rs: continue
        pe = np.array([x["pos_err"] for x in rs]); ye = np.array([x["yaw_err"] for x in rs])
        s5 = float(np.mean((pe < 0.05) & (ye < 3))); s10 = float(np.mean((pe < 0.10) & (ye < 5)))
        row = dict(n=len(rs), pos_med_cm=round(float(np.median(pe)) * 100, 1),
                   pos_p90_cm=round(float(np.percentile(pe, 90)) * 100, 1),
                   yaw_med_deg=round(float(np.median(ye)), 2),
                   succ_5cm3deg=round(s5, 3), succ_10cm5deg=round(s10, 3))
        summ[f"±{r_m}m,±{psi_d}°"] = row
        print(f"±{r_m}m/±{psi_d}°: n={row['n']} pos中位{row['pos_med_cm']}cm p90={row['pos_p90_cm']}cm "
              f"yaw中位{row['yaw_med_deg']}° 成功率(5cm/3°)={s5:.0%} (10cm/5°)={s10:.0%}")
    with open(os.path.join(outdir, "reloc_eval.json"), "w") as f:
        json.dump(dict(session=args.session, seed=args.seed, summary=summ, results=results), f, ensure_ascii=False, indent=1)
    print(f"[reloc] 保存 {outdir}/reloc_eval.json")


def _viz(g, j, C0, R0, C1, R1, Cgt, Rgt, outdir):
    """初始/优化后/GT 三位姿的渲染深度 vs zed观测深度 拼图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ho, Wo = 360, 640
    d_obs, m_obs = g.obs_depth_m(g.names[j], Ho, Wo)
    ims = []
    for (C, R, tag) in ((C0, R0, "init"), (C1, R1, "optimized"), (Cgt, Rgt, "GT")):
        V = g.viewmat_from_world(torch.tensor(C, device=DEV, dtype=torch.float32),
                                 torch.tensor(R, device=DEV, dtype=torch.float32))
        with torch.no_grad():
            ed, al = g.render_ed(V, Ho, Wo)
        ims.append((tag, ed.cpu().numpy()))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    axes[0, 0].imshow(np.where(m_obs.cpu().numpy(), d_obs.cpu().numpy(), np.nan), cmap="viridis", vmin=0.3, vmax=6)
    axes[0, 0].set_title("zed obs depth (m)")
    for ax, (tag, ed) in zip(axes.flat[1:], ims):
        ax.imshow(ed, cmap="viridis", vmin=0.3, vmax=6); ax.set_title(f"render ED @ {tag}")
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout()
    p = os.path.join(outdir, f"viz_f{j}.png"); plt.savefig(p, dpi=110); plt.close()
    print(f"  [viz] {p}", flush=True)


if __name__ == "__main__":
    main()
