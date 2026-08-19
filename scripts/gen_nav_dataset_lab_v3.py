"""快脑训练集v3 (0731): Qwen-RobotNav训练经验 — 条件连续化+每样本随机化+减速段。

相对v2的升级:
  ①起点连续抖动: xy±0.25m/yaw±0.4rad(替代v2固定dyaw三档), 覆盖更全
  ②目标连续采样拓宽: 方位±100°(含更多横向=strafe行为), 距离[1.0,6.0]m
  ③⭐减速段GT(论文§4.1.2): 目标在视窗内(路径<1.6m)时航点间距前疏后密(二次分布),
    教"平滑减速停到目标"而非匀速冲过
  ④⭐速度随机化(论文speed augmentation): 每样本vx~U(0.3,0.8), 近目标样本U(0.15,0.35),
    存meta["vx"]供训练填status[4](不再全数据集固定0.4)
用法: CUDA_VISIBLE_DEVICES=? MUJOCO_GL=egl PYTHONPATH=third_party/discoverse \
      ~/discoverse_venv/bin/python scripts/gen_nav_dataset_lab_v3.py --session colmapjoint_110744s2
产出: outputs/nav_colmapjoint_110744s2_v3/ (不覆盖v2)
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
from gen_nav_dataset import goal_dir_ego, T

HORIZON = 1.6


def resample_decel(ego, T=T):
    """前疏后密(二次)重采样: 教近目标线性减速(论文: 1.5m内步长线性减小)。"""
    seg = np.linalg.norm(np.diff(ego, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
    if arc[-1] < 1e-3:
        return np.zeros((T, 2), np.float32)
    j = np.arange(1, T + 1)
    tgt = arc[-1] * (1.0 - (1.0 - j / T) ** 2)          # d_j: 间距前大后小
    return np.stack([np.interp(tgt, arc, ego[:, 0]), np.interp(tgt, arc, ego[:, 1])], 1).astype(np.float32)


def path_to_ego(path, x, y, yaw, decel):
    """A*路径→ego系(T,2); 目标在窗内(decel=True)用减速间距, 否则匀速前1.6m。"""
    pts = np.array(path, np.float64)
    c, s = np.cos(yaw), np.sin(yaw); dd = pts - np.array([x, y])
    ego = np.stack([c * dd[:, 0] + s * dd[:, 1], -s * dd[:, 0] + c * dd[:, 1]], 1)
    ego = np.vstack([[0.0, 0.0], ego])
    seg = np.linalg.norm(np.diff(ego, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
    if arc[-1] < 1e-3:
        return None, 0.0
    if decel:                                            # 目标在窗内: 全程+减速间距
        return resample_decel(ego, T), float(arc[-1])
    # 匀速前HORIZON米
    tgt = np.linspace(min(HORIZON, arc[-1]) / T, min(HORIZON, arc[-1]), T)
    return np.stack([np.interp(tgt, arc, ego[:, 0]), np.interp(tgt, arc, ego[:, 1])], 1).astype(np.float32), float(arc[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--stride_m", type=float, default=0.4)
    ap.add_argument("--n_jitter", type=int, default=2, help="每弧长点的抖动起点数")
    ap.add_argument("--n_goal", type=int, default=6, help="每起点目标数")
    args = ap.parse_args()
    import sim_fullchain as sf
    from test_env import CorridorTestEnv
    env = CorridorTestEnv(skin_session=args.session)

    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{args.session}.npz"))
    ref = W["waypoints_xy"].astype(np.float64)
    merged_ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    outdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}_v3"); os.makedirs(outdir, exist_ok=True)
    free_pts = np.array([tuple(w) for w in ref if not env._occupied(w[0], w[1])])

    seg = np.linalg.norm(np.diff(ref, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
    base_s = np.arange(0, arc[-1] - 0.5, args.stride_m)
    rng = np.random.RandomState(0)
    poses, goals, labels, fpv_paths, kinds, vxs = [], [], [], [], [], []
    n = 0; n_decel = 0
    for si in base_s:
        i = min(int(np.searchsorted(arc, si)), len(ref) - 2)
        th0 = float(np.arctan2(ref[i + 1, 1] - ref[i, 1], ref[i, 0] - ref[i, 0]))
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
                poses.append([x, y, yaw]); goals.append(gd); labels.append(lab)
                fpv_paths.append(fpv_saved); kinds.append("decel" if decel else "cruise")
                vxs.append(vx); n += 1; tried += 1; n_decel += int(decel)
        if n and n % 400 < args.n_goal * 2:
            print(f"  已生成 {n} (弧长{si:.1f}/{arc[-1]:.1f})", flush=True)
    np.savez(os.path.join(outdir, "meta.npz"),
             poses=np.array(poses, np.float32), goal_dirs=np.array(goals, np.float32),
             base_ego=np.array(labels, np.float32), fpv=np.array(fpv_paths),
             kinds=np.array(kinds), vx=np.array(vxs, np.float32), merged_ply=merged_ply)
    print(f"[done] {n} 样本(减速段{n_decel}) -> {outdir}/meta.npz", flush=True)


if __name__ == "__main__":
    main()
