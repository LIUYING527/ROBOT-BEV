"""回归头真zed实测 (0813, 接reloc_head_train.py): 真zed npz观测 + 已知扰动 → reloc_head单次/迭代前向,
与GS优化器(gs_relocate.py, 同帧同扰动)对比终误差。
协议复刻gs_relocate.py: 帧选择(均匀采样+180x320掩码>3000) + 圆盘均匀扰动 + RandomState(0),
±0.3m/±15°档与outputs/gs_relocate_110744s2/reloc_eval.json逐帧同扰动(init_pos应完全一致)。
⚠训练/验证按帧划分(RandomState(0) shuffle前20%=val) → val帧才是未见过真位姿的诚实数字, 分开报。
⚠域差: 训练obs=渲染ED@-22°俯角, 实测obs=真zed(手持真实俯仰~1.71m) —— 这正是真zed实测的意义。

用法:
  CUDA_VISIBLE_DEVICES=6 ~/discoverse_venv/bin/python scripts/reloc_head_eval.py            # 纯head, 快
  ... --gs_live "0.5,25" --gs_frames 8    # 该档现场跑GS优化器对比(慢, ~43s/帧)
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "outputs", "reloc_head_0812")
IMG_H, IMG_W = 90, 160
SCALE = np.array([0.5, 0.5, np.radians(25)], np.float32)     # 与训练一致的目标归一化


def build_head(dev, ckpt=None):
    import torch, torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(2, 32, 5, 2, 2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.pose = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
            self.fuse = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, 3))
        def forward(self, img, ps):
            return self.fuse(torch.cat([self.cnn(img), self.pose(ps)], 1))

    net = Head().to(dev)
    net.load_state_dict(__import__("torch").load(ckpt or os.path.join(OUT, "reloc_head.pt"), map_location=dev))
    net.eval()
    return net


def head_correct(net, obs_np, msk_np, bel, n_iter):
    """obs_np (H,W) 世界米(0填充无效), msk_np (H,W)bool, bel=[x,y,yaw] → n_iter次迭代修正后的[x,y,yaw]。
    迭代合法: 观测在真位姿采集, 不依赖believed —— 同一张obs可反复查询头做coarse→fine。"""
    import torch
    img = torch.tensor(np.stack([obs_np, msk_np.astype(np.float32)])[None], device="cuda")
    bel = np.array(bel, np.float64).copy()
    for _ in range(n_iter):
        ps = np.array([[bel[0], bel[1], np.sin(bel[2]), np.cos(bel[2])]], np.float32)
        with torch.no_grad():
            d = net(img, torch.tensor(ps, device="cuda")).cpu().numpy()[0] * SCALE
        # believed系(右,前) → 世界: M列=[right,fwd]=[[sin,cos],[-cos,sin]] (同训练gen)
        cb, sb = np.cos(bel[2]), np.sin(bel[2])
        M = np.array([[sb, cb], [-cb, sb]])
        bel[:2] += M @ d[:2]
        bel[2] += d[2]
    return bel


def select_frames(g, n_frames):
    """复刻gs_relocate.py帧选择。"""
    cand = np.linspace(0, len(g.names) - 1, n_frames * 3).astype(int)
    idxs = []
    for j in cand:
        d, m = g.obs_depth_m(g.names[j], 180, 320)
        if m is not None and m.sum() > 3000:
            idxs.append(int(j))
        if len(idxs) >= n_frames:
            break
    return idxs


def val_frame_set():
    """复刻训练train/val按帧划分(RandomState(0), 前20% unique帧=val)。"""
    dz = np.load(os.path.join(OUT, "data.npz"))
    uf = np.unique(dz["frame"]); rng = np.random.RandomState(0); rng.shuffle(uf)
    return set(uf[: max(1, int(0.2 * len(uf)))].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--raw", default="data/lab/110744")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", default="1,3", help="head迭代次数, 逗号分隔")
    ap.add_argument("--gs_live", default="", help="'r,psi'档现场跑GS优化器对比")
    ap.add_argument("--gs_frames", type=int, default=8)
    args = ap.parse_args()

    from gs_relocate import GSReloc, yaw_of
    g = GSReloc(args.session, args.raw)
    net = build_head("cuda")
    idxs = select_frames(g, args.frames)
    vset = val_frame_set()
    print(f"[eval] {len(idxs)}帧 (val帧={[j for j in idxs if j in vset]})", flush=True)

    # ±0.3/±15档: 与stored GS结果同rng协议 (该run只跑了这一档, rng从seed 0新鲜开始)
    stored = json.load(open(os.path.join(ROOT, "outputs", "gs_relocate_110744s2", "reloc_eval.json")))
    stored_by_f = {r["frame"]: r for r in stored["results"]}
    levels = [(0.15, 8.0, np.random.RandomState(1015)),
              (0.3, 15.0, np.random.RandomState(args.seed)),      # ← 复刻stored协议
              (0.5, 25.0, np.random.RandomState(1025))]
    n_iters = [int(x) for x in args.iters.split(",")]

    gs_live = None
    if args.gs_live:
        r_, p_ = args.gs_live.split(","); gs_live = (float(r_), float(p_))

    results = []
    for (r_m, psi_d, rng) in levels:
        gs_live_done = 0
        for j in idxs:
            C_gt, R_gt = g.gt_world_pose(j)
            yaw_gt = np.arctan2(R_gt[1, 2], R_gt[0, 2])
            a = rng.uniform(0, 2 * np.pi); rr = r_m * np.sqrt(rng.uniform(0, 1))
            dx, dy = rr * np.cos(a), rr * np.sin(a)
            dyaw = np.radians(rng.uniform(-psi_d, psi_d))
            bel0 = np.array([C_gt[0] + dx, C_gt[1] + dy, yaw_gt + dyaw])
            pos0, yaw0 = float(np.hypot(dx, dy)), abs(float(np.degrees(dyaw)))

            d_obs, m_obs = g.obs_depth_m(g.names[j], IMG_H, IMG_W)
            if d_obs is None or m_obs.sum() < 500:
                print(f"  L(±{r_m},±{psi_d}) f#{j}: 观测不足跳过", flush=True); continue
            rec = dict(level_r=r_m, level_psi=psi_d, frame=int(j), in_val=bool(j in vset),
                       init_pos=round(pos0, 4), init_yaw=round(yaw0, 3))
            for ni in n_iters:
                bel = head_correct(net, d_obs.cpu().numpy(), m_obs.cpu().numpy(), bel0, ni)
                horiz = float(np.hypot(*(bel[:2] - C_gt[:2]))) * 100
                yerr = abs(float(np.degrees(bel[2] - yaw_gt))); yerr = min(yerr, 360 - yerr)
                rec[f"head{ni}_horiz_cm"] = round(horiz, 1); rec[f"head{ni}_yaw_deg"] = round(yerr, 2)
            # stored GS对比(仅±0.3/±15档, 验证同扰动)
            if (r_m, psi_d) == (0.3, 15.0) and j in stored_by_f:
                sr = stored_by_f[j]
                match = abs(sr["init_pos"] - pos0) < 1e-3 and abs(sr["init_yaw"] - yaw0) < 1e-2
                rec["gs_horiz_cm"] = sr["horiz_cm"]; rec["gs_yaw_deg"] = sr["yaw_err"]; rec["gs_match"] = bool(match)
            # 现场GS(可选)
            if gs_live == (r_m, psi_d) and gs_live_done < args.gs_frames:
                import torch
                c_, s_ = np.cos(dyaw), np.sin(dyaw)
                Rp = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]], np.float64)
                C1, R1, _ = g.optimize(j, C_gt + np.array([dx, dy, 0.0]), Rp @ R_gt)
                if C1 is not None:
                    rec["gsl_horiz_cm"] = round(float(np.hypot(*(C1 - C_gt)[:2])) * 100, 1)
                    rec["gsl_yaw_deg"] = round(abs(yaw_of(R1) - yaw_of(R_gt)), 2)
                gs_live_done += 1
            results.append(rec)
            print(f"  L(±{r_m}m,±{psi_d}°) f#{j:3d}{'V' if j in vset else 'T'}: init({pos0:.2f}m,{yaw0:.1f}°) → "
                  + " ".join(f"head{ni}=({rec[f'head{ni}_horiz_cm']:.1f}cm,{rec[f'head{ni}_yaw_deg']:.2f}°)" for ni in n_iters)
                  + (f" | GS=({rec['gs_horiz_cm']:.1f}cm,{rec['gs_yaw_deg']:.2f}°){'' if rec.get('gs_match') else '⚠扰动不匹配'}" if "gs_horiz_cm" in rec else "")
                  + (f" | GS-live=({rec['gsl_horiz_cm']:.1f}cm,{rec['gsl_yaw_deg']:.2f}°)" if "gsl_horiz_cm" in rec else ""), flush=True)

    # 汇总 (全帧 / 仅val帧)
    print("\n===== 汇总 =====")
    summ = {}
    for (r_m, psi_d, _) in levels:
        for sub_name, subset in (("all", results), ("val", [x for x in results if x["in_val"]])):
            rs = [x for x in subset if x["level_r"] == r_m]
            if not rs: continue
            row = {}
            for ni in n_iters:
                h = np.array([x[f"head{ni}_horiz_cm"] for x in rs]); y = np.array([x[f"head{ni}_yaw_deg"] for x in rs])
                row[f"head{ni}"] = dict(horiz_med=round(float(np.median(h)), 1), horiz_p90=round(float(np.percentile(h, 90)), 1),
                                        yaw_med=round(float(np.median(y)), 2),
                                        succ_10cm5deg=round(float(np.mean((h < 10) & (y < 5))), 3))
            gs = [x["gs_horiz_cm"] for x in rs if "gs_horiz_cm" in x]
            if gs:
                gy = [x["gs_yaw_deg"] for x in rs if "gs_horiz_cm" in x]
                row["GS优化器"] = dict(horiz_med=round(float(np.median(gs)), 1), yaw_med=round(float(np.median(gy)), 2))
            gsl = [x["gsl_horiz_cm"] for x in rs if "gsl_horiz_cm" in x]
            if gsl:
                gly = [x["gsl_yaw_deg"] for x in rs if "gsl_horiz_cm" in x]
                row["GS_live"] = dict(n=len(gsl), horiz_med=round(float(np.median(gsl)), 1), yaw_med=round(float(np.median(gly)), 2))
            summ[f"±{r_m}m,±{psi_d}°/{sub_name}"] = row
            print(f"±{r_m}m/±{psi_d}° [{sub_name} n={len(rs)}]: " +
                  " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    with open(os.path.join(OUT, "eval_realzed.json"), "w") as f:
        json.dump(dict(session=args.session, summary=summ, results=results), f, ensure_ascii=False, indent=1)
    print(f"[eval] 保存 {OUT}/eval_realzed.json")


if __name__ == "__main__":
    main()
