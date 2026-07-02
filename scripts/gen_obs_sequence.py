"""Stage A(非反应式评测) — sim 沿参考轨迹渲染 FPV + 存位姿。
在 discoverse_venv 跑(有 3DGS 渲染器);不含模型。产物给 Stage B(模型 env)消费。

用法:
  CUDA_VISIBLE_DEVICES=N MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=N \
    PYTHONPATH=third_party/discoverse ~/discoverse_venv/bin/python \
    scripts/gen_obs_sequence.py --session colmapjoint_all --n 60
产物: outputs/obs_seq_<session>/{NNN.jpg(FPV), poses.npz(pose+session+merged_ply路径)}
"""
import os, sys, argparse
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sim_walk_common import make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--n", type=int, default=60, help="采样帧数(沿参考轨迹均匀)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)   # 16:9, Stage B resize 到 1024x256(同上车)
    ap.add_argument("--fovy", type=float, default=75.0)
    args = ap.parse_args()

    env, W = make_env(args.session, width=args.width, height=args.height, fovy=args.fovy)
    wp = W["waypoints_xy"]                # (K,2) 世界系参考轨迹
    yaw = W["yaw"] if "yaw" in W else None
    K = len(wp)
    idxs = np.linspace(0, K - 1, min(args.n, K)).astype(int)

    outdir = os.path.join(ROOT, "outputs", f"obs_seq_{args.session}")
    os.makedirs(outdir, exist_ok=True)
    poses = []
    for i, k in enumerate(idxs):
        x, y = float(wp[k][0]), float(wp[k][1])
        # 朝向: 优先用存的 yaw, 否则用指向下一个 waypoint
        if yaw is not None:
            th = float(yaw[k])
        else:
            k2 = min(k + 3, K - 1)
            th = float(np.arctan2(wp[k2][1] - y, wp[k2][0] - x))
        env.set_pose(x, y, th)
        env.render()
        fpv = env.frame()                 # (H,W,3) uint8 RGB
        cv2.imwrite(os.path.join(outdir, f"{i:03d}.jpg"), fpv[:, :, ::-1])  # 存 BGR
        poses.append([x, y, th])
        if i % 10 == 0:
            print(f"  frame {i}/{len(idxs)} pose=({x:.1f},{y:.1f},{th:.2f})", flush=True)

    merged_ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    np.savez(os.path.join(outdir, "poses.npz"),
             poses=np.array(poses, np.float32), session=args.session,
             merged_ply=merged_ply, fovy=args.fovy, n=len(idxs))
    print(f"[stageA] 存 {len(idxs)} 帧 -> {outdir}  (fpv jpg + poses.npz)")


if __name__ == "__main__":
    main()
