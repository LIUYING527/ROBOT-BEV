"""快脑训练集v4 (0802): 观察策略参数化 — v3基础上加参数化时序历史标注。

相对v3的升级(基数据集与v3逐样本一致: 基础随机性仍走rng=RandomState(0)同序列,
新增随机性全部走独立rng1=RandomState(1), 保证v3/v4可比):
  ①历史位姿: 每样本记录ref轨迹上身后0.4m×lag(lag=1..8)的世界系位姿(朝向=路径切线)
  ②⭐窗口W参数化(论文观察策略): P(W=0)=0.25(保护单帧分支=run_inspection等旧脚本),
    其余均匀1..8, 上限=可用历史数(起点附近截断)
  ③⭐衰减γ参数化(=遗忘): γ~U[0.5,1.0], lag h保留γ^h比例的历史点(训练时在BEV层融合)
用法: CUDA_VISIBLE_DEVICES=? MUJOCO_GL=egl PYTHONPATH=third_party/discoverse \
      ~/discoverse_venv/bin/python scripts/gen_nav_dataset_lab_v4.py --session colmapjoint_110744s2
产出: outputs/nav_ds_colmapjoint_110744s2_v4/ (不覆盖v3)
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
from gen_nav_dataset import goal_dir_ego, T
from gen_nav_dataset_lab_v3 import resample_decel, path_to_ego, HORIZON

HIST_MAX = 8
HIST_STRIDE = 0.4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--stride_m", type=float, default=0.4)
    ap.add_argument("--n_jitter", type=int, default=2, help="每弧长点的抖动起点数")
    ap.add_argument("--n_goal", type=int, default=6, help="每起点目标数")
    ap.add_argument("--p_w0", type=float, default=0.25, help="P(W=0)单帧样本比例")
    args = ap.parse_args()
    import sim_fullchain as sf
    from test_env import CorridorTestEnv
    env = CorridorTestEnv(skin_session=args.session)

    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{args.session}.npz"))
    ref = W["waypoints_xy"].astype(np.float64)
    merged_ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    outdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}_v4"); os.makedirs(outdir, exist_ok=True)
    free_pts = np.array([tuple(w) for w in ref if not env._occupied(w[0], w[1])])

    seg = np.linalg.norm(np.diff(ref, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
    base_s = np.arange(0, arc[-1] - 0.5, args.stride_m)
    rng = np.random.RandomState(0)                            # ⭐基础采样与v3同序列(别动)
    rng1 = np.random.RandomState(1)                           # ⭐v4新增随机性(W/γ)独立流
    poses, goals, labels, fpv_paths, kinds, vxs = [], [], [], [], [], []
    hposes, hws, hgammas = [], [], []
    n = 0; n_decel = 0
    for si in base_s:
        i = min(int(np.searchsorted(arc, si)), len(ref) - 2)
        th0 = float(np.arctan2(ref[i + 1, 1] - ref[i, 1], ref[i, 0] - ref[i, 0]))
        # 该弧长点的历史位姿(ref路径身后, 朝向=切线)——对同一起点的所有目标共享
        hist = np.zeros((HIST_MAX, 3), np.float32); n_avail = 0
        for lag in range(1, HIST_MAX + 1):
            s_h = si - HIST_STRIDE * lag
            if s_h <= 0:
                break
            jh = min(int(np.searchsorted(arc, s_h)), len(ref) - 2)
            th_h = float(np.arctan2(ref[jh + 1, 1] - ref[jh, 1], ref[jh + 1, 0] - ref[jh, 0]))
            hist[lag - 1] = [ref[jh, 0], ref[jh, 1], th_h]; n_avail = lag
        for _ in range(args.n_jitter):
            x = float(ref[i, 0]) + rng.uniform(-0.25, 0.25)
            y = float(ref[i, 1]) + rng.uniform(-0.25, 0.25)
            yaw = th0 + rng.uniform(-0.4, 0.4)
            if env._occupied(x, y):
                continue
            fpv_saved = None; tried = 0
            for _ in range(args.n_goal * 4):
                if tried >= args.n_goal:
                    break
                g = free_pts[rng.randint(len(free_pts))]
                dvec = g - np.array([x, y]); dist = np.hypot(*dvec)
                if dist < 1.0 or dist > 6.0:
                    continue
                bearing = np.arctan2(dvec[1], dvec[0])
                if abs(((bearing - yaw + np.pi) % (2 * np.pi)) - np.pi) > np.radians(100):
                    continue
                env.start = (x, y, yaw); env.goal = (float(g[0]), float(g[1]))
                path = sf.plan_path(env)
                if path is None or len(path) < 2:
                    continue
                decel = dist < HORIZON                          # 目标在视窗内 → 减速GT
                lab, plen = path_to_ego(path, x, y, yaw, decel)
                if lab is None or lab[-1, 0] < 0.15:
                    continue
                lab[:, 0] = np.maximum.accumulate(lab[:, 0])
                gd = goal_dir_ego(path, x, y, yaw, ahead=4.0)
                vx = float(rng.uniform(0.15, 0.35) if decel else rng.uniform(0.3, 0.8))
                if fpv_saved is None:
                    env.env.set_pose(x, y, yaw); env.env.render()
                    fpv_saved = f"{n:04d}.jpg"
                    cv2.imwrite(os.path.join(outdir, fpv_saved), env.env.frame()[:, :, ::-1].copy())
                # ⭐v4: 观察策略参数化(W窗口/γ衰减), 独立rng1流
                w = 0 if rng1.rand() < args.p_w0 else int(rng1.randint(1, HIST_MAX + 1))
                w = min(w, n_avail)
                gam = float(rng1.uniform(0.5, 1.0)) if w > 0 else 1.0
                poses.append([x, y, yaw]); goals.append(gd); labels.append(lab)
                fpv_paths.append(fpv_saved); kinds.append("decel" if decel else "cruise")
                vxs.append(vx); hposes.append(hist); hws.append(w); hgammas.append(gam)
                n += 1; tried += 1; n_decel += int(decel)
        if n and n % 400 < args.n_goal * 2:
            print(f"  已生成 {n} (弧长{si:.1f}/{arc[-1]:.1f})", flush=True)
    np.savez(os.path.join(outdir, "meta.npz"),
             poses=np.array(poses, np.float32), goal_dirs=np.array(goals, np.float32),
             base_ego=np.array(labels, np.float32), fpv=np.array(fpv_paths),
             kinds=np.array(kinds), vx=np.array(vxs, np.float32),
             hist_poses=np.array(hposes, np.float32), hist_w=np.array(hws, np.int32),
             hist_gamma=np.array(hgammas, np.float32), merged_ply=merged_ply)
    hws_a = np.array(hws)
    print(f"[done] {n} 样本(减速段{n_decel}) -> {outdir}/meta.npz", flush=True)
    print(f"[hist] W分布: 0:{(hws_a==0).sum()} 1-3:{((hws_a>=1)&(hws_a<=3)).sum()} "
          f"4-6:{((hws_a>=4)&(hws_a<=6)).sum()} 7-8:{(hws_a>=7).sum()}", flush=True)


if __name__ == "__main__":
    main()
