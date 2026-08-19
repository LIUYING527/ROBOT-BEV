"""为室内实验室(110744s2)造【快脑微调训练集】: 用真实采集轨迹当专家参考, 沿途采样(带横向/朝向扰动),
渲仿真FPV(sim-to-sim与任务推理同域) + 未来专家轨迹GT(ego系, 由真实轨迹续延=学会避实验室墙+从偏离态恢复)。
产物 outputs/nav_ds_<session>/NNN.jpg + meta.npz(poses,goal_dirs,base_ego,fpv,kinds,merged_ply)。
用法: CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
      ~/discoverse_venv/bin/python scripts/gen_nav_dataset_lab.py --session colmapjoint_110744s2
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
from gen_nav_dataset import resample_ref_from, goal_dir_ego   # 复用GT续延/目标方向


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--stride_m", type=float, default=0.25)
    ap.add_argument("--lat", type=float, nargs="+", default=[0.0, 0.35, -0.35, 0.6, -0.6])
    ap.add_argument("--dyaw", type=float, nargs="+", default=[0.0, 0.2, -0.2])
    args = ap.parse_args()

    from test_env import CorridorTestEnv
    env = CorridorTestEnv(skin_session=args.session)
    # ⭐参考=完整采集轨迹(不用env.ref, 它可能被_forward_only截断; 实验室是绕圈需全程)
    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{args.session}.npz"))
    ref = W["waypoints_xy"].astype(np.float64)
    merged_ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    outdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}"); os.makedirs(outdir, exist_ok=True)

    seg = np.linalg.norm(np.diff(ref, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
    base_s = np.arange(0, arc[-1] - 0.5, args.stride_m)
    poses, goals, labels, fpv_paths, kinds = [], [], [], [], []
    n = 0
    for si in base_s:
        i = min(int(np.searchsorted(arc, si)), len(ref) - 2)
        th = float(np.arctan2(ref[i + 1, 1] - ref[i, 1], ref[i + 1, 0] - ref[i, 0]))
        px = th + np.pi / 2
        for off in args.lat:
            for dy in args.dyaw:
                x = ref[i, 0] + off * np.cos(px); y = ref[i, 1] + off * np.sin(px); yaw = th + dy
                if env._occupied(x, y):
                    continue
                lab = resample_ref_from(ref, x, y, yaw, horizon=1.6)
                if lab[-1, 0] < 0.25:                     # 丢倒退/不推进
                    continue
                lab[:, 0] = np.maximum.accumulate(lab[:, 0])
                env.env.set_pose(x, y, yaw); env.env.render()
                fpv = env.env.frame()[:, :, ::-1].copy()
                cv2.imwrite(os.path.join(outdir, f"{n:04d}.jpg"), fpv)
                poses.append([x, y, yaw]); goals.append(goal_dir_ego(ref, x, y, yaw))
                labels.append(lab); fpv_paths.append(f"{n:04d}.jpg"); kinds.append("cruise"); n += 1
        if n and len(poses) % 100 < (len(args.lat) * len(args.dyaw)):
            print(f"  已生成 {n} (弧长{si:.1f}/{arc[-1]:.1f})", flush=True)
    np.savez(os.path.join(outdir, "meta.npz"),
             poses=np.array(poses, np.float32), goal_dirs=np.array(goals, np.float32),
             base_ego=np.array(labels, np.float32), fpv=np.array(fpv_paths),
             kinds=np.array(kinds), merged_ply=merged_ply)
    print(f"[done] {n} 训练样本 -> {outdir}/meta.npz", flush=True)


if __name__ == "__main__":
    main()
