"""造【纠偏增强】训练集: 沿中线每个位姿, 额外在横移±/转角±的扰动位姿上渲 FPV。
配合 train_conditional 的 recovery GT(从偏离位姿指回中线) → 教快脑"跑偏了怎么拽回来",
治闭环复合误差漂移。全用 3DGS 皮肤在任意位姿渲, 不等团队数据。

基准中线 = 现有 outputs/obs_seq_<session>/poses.npz(已是切线平滑朝向的200前向帧)。
用法(卡7):
  CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=7 PYOPENGL_PLATFORM=egl \
    PYTHONPATH=third_party/discoverse ~/discoverse_venv/bin/python \
    arch_e2e/pipeline/gen_augmented_dataset.py --session colmapjoint_all
产物: outputs/obs_seq_<session>_aug/{NNN.jpg, poses.npz(poses+base_idx+ref_poses+merged_ply)}
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, argparse
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sim_walk_common import make_env

# 扰动集(dy横移左正 米, dψ转角 度): 纯横移 + 纯转 + 组合(偏一侧还朝反向=最需纠偏)
PERTURB = [
    (0.0, 0), (0.0, 0),                                   # 名义(渲2次当轻微增广)
    (0.30, 0), (-0.30, 0), (0.55, 0), (-0.55, 0),         # 横移
    (0.0, 15), (0.0, -15),                                # 转角
    (0.35, -12), (-0.35, 12), (0.5, -18), (-0.5, 18),     # 组合(偏左却朝右等)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fovy", type=float, default=75.0)
    args = ap.parse_args()

    ref_npz = os.path.join(ROOT, "outputs", f"obs_seq_{args.session}", "poses.npz")
    R = np.load(ref_npz, allow_pickle=True)
    ref = R["poses"].astype(np.float64)                   # (N,3) 切线朝向中线
    merged_ply = str(R["merged_ply"]) if "merged_ply" in R else \
        os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    N = len(ref)
    print(f"[aug] 中线 {N} 位姿 x {len(PERTURB)} 扰动 = {N*len(PERTURB)} 帧", flush=True)

    env, _ = make_env(args.session, width=args.width, height=args.height, fovy=args.fovy)
    outdir = os.path.join(ROOT, "outputs", f"obs_seq_{args.session}_aug")
    os.makedirs(outdir, exist_ok=True)

    poses, base_idx, m = [], [], 0
    for i in range(N):
        x, y, th = ref[i]
        el = np.array([-np.sin(th), np.cos(th)])          # ego-left 单位向量(世界系)
        for dy, dpsi in PERTURB:
            xp, yp = x + el[0] * dy, y + el[1] * dy
            thp = th + np.radians(dpsi)
            env.set_pose(float(xp), float(yp), float(thp)); env.render()
            fpv = env.frame()
            cv2.imwrite(os.path.join(outdir, f"{m:04d}.jpg"), fpv[:, :, ::-1])
            poses.append([xp, yp, thp]); base_idx.append(i); m += 1
        if i % 20 == 0:
            print(f"  中线{i}/{N} 已渲{m}帧", flush=True)

    np.savez(os.path.join(outdir, "poses.npz"),
             poses=np.array(poses, np.float32), base_idx=np.array(base_idx, np.int32),
             ref_poses=ref.astype(np.float32), merged_ply=merged_ply,
             session=args.session, n=m)
    print(f"[aug] 存 {m} 帧 -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
