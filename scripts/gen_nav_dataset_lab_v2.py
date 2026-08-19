"""⭐快脑训练集v2: 教"朝多样目标收敛"(修v1单回环→绕圈的病)。
每个起点配多个不同方向的目标, GT=A*从起点到目标的直达路径前1.6m(ego系), goal_dir=朝目标方向。
→快脑学"给目标方向就朝它走+避障(模仿A*)", 不再跟回环绕圈。
用法: CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
      ~/discoverse_venv/bin/python scripts/gen_nav_dataset_lab_v2.py --session colmapjoint_110744s2
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
from gen_nav_dataset import resample_ref_from, goal_dir_ego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--stride_m", type=float, default=0.4, help="沿轨迹采起点间距")
    ap.add_argument("--dyaw", type=float, nargs="+", default=[0.0, 0.3, -0.3])
    ap.add_argument("--n_goal", type=int, default=5, help="每起点采多少个不同方向的目标")
    args = ap.parse_args()
    import sim_fullchain as sf
    from test_env import CorridorTestEnv
    env = CorridorTestEnv(skin_session=args.session)

    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{args.session}.npz"))
    ref = W["waypoints_xy"].astype(np.float64)
    merged_ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    outdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}"); os.makedirs(outdir, exist_ok=True)
    # 全部可达自由点(采样目标用): 轨迹点 + 邻域
    free_pts = [tuple(w) for w in ref if not env._occupied(w[0], w[1])]
    free_pts = np.array(free_pts)

    seg = np.linalg.norm(np.diff(ref, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
    base_s = np.arange(0, arc[-1] - 0.5, args.stride_m)
    rng = np.random.RandomState(0)
    poses, goals, labels, fpv_paths, kinds = [], [], [], [], []
    n = 0
    for si in base_s:
        i = min(int(np.searchsorted(arc, si)), len(ref) - 2)
        th0 = float(np.arctan2(ref[i + 1, 1] - ref[i, 1], ref[i + 1, 0] - ref[i, 0]))
        for dy in args.dyaw:
            x, y, yaw = float(ref[i, 0]), float(ref[i, 1]), th0 + dy
            if env._occupied(x, y):
                continue
            fpv_saved = None
            # 采多个不同方向的目标(前方±80°内, 距离1.5~4.5m), A*直达路径当GT
            tried = 0
            for _ in range(args.n_goal * 4):
                if tried >= args.n_goal:
                    break
                g = free_pts[rng.randint(len(free_pts))]
                dvec = g - np.array([x, y]); dist = np.hypot(*dvec)
                if dist < 1.2 or dist > 5.0:
                    continue
                bearing = np.arctan2(dvec[1], dvec[0])
                if abs(((bearing - yaw + np.pi) % (2 * np.pi)) - np.pi) > np.radians(80):  # 只要前方±80°(CRUISE)
                    continue
                env.start = (x, y, yaw); env.goal = (float(g[0]), float(g[1]))
                path = sf.plan_path(env)
                if path is None or len(path) < 2:
                    continue
                lab = resample_ref_from(path, x, y, yaw, horizon=1.6)   # A*路径前1.6m→ego GT
                if lab[-1, 0] < 0.2:
                    continue
                lab[:, 0] = np.maximum.accumulate(lab[:, 0])
                gd = goal_dir_ego(path, x, y, yaw, ahead=4.0)
                if fpv_saved is None:                                  # 该位姿只渲一次FPV
                    env.env.set_pose(x, y, yaw); env.env.render()
                    fpv_saved = f"{n:04d}.jpg"
                    cv2.imwrite(os.path.join(outdir, fpv_saved), env.env.frame()[:, :, ::-1].copy())
                poses.append([x, y, yaw]); goals.append(gd); labels.append(lab)
                fpv_paths.append(fpv_saved); kinds.append("cruise"); n += 1; tried += 1
        if n and len(poses) % 200 < args.n_goal * 3:
            print(f"  已生成 {n} (弧长{si:.1f}/{arc[-1]:.1f})", flush=True)
    np.savez(os.path.join(outdir, "meta.npz"),
             poses=np.array(poses, np.float32), goal_dirs=np.array(goals, np.float32),
             base_ego=np.array(labels, np.float32), fpv=np.array(fpv_paths),
             kinds=np.array(kinds), merged_ply=merged_ply)
    print(f"[done] {n} 目标条件样本(多方向A*直达GT) -> {outdir}/meta.npz", flush=True)


if __name__ == "__main__":
    main()
