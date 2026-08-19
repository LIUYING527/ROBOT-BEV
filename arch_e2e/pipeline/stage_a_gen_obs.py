"""Stage A (非反应式评测) — sim 沿参考轨迹渲染 FPV + 存位姿。（arch_e2e 版）

在 discoverse_venv 跑(有 3DGS 渲染器); 不含模型。产物给 Stage B(模型 env)消费。
注: Stage A 只复用【我们自己】的仿真渲染器(repo/scripts/sim_walk_common + third_party/discoverse
+ 3DGS ply), 不碰学长 robot_orig_code —— 故直接把 repo/scripts 挂进路径, 无需复制整套渲染器。

用法:
  CUDA_VISIBLE_DEVICES=N MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=N \
    PYTHONPATH=<repo>/third_party/discoverse ~/discoverse_venv/bin/python \
    arch_e2e/pipeline/stage_a_gen_obs.py --session colmapjoint_all --n 60
产物: arch_e2e/runs/obs_seq_<session>/{NNN.jpg(FPV), poses.npz(pose+session+merged_ply路径)}
"""
import os, sys, argparse
import numpy as np
import cv2

from _paths import REPO, RUNS
sys.path.insert(0, os.path.join(REPO, "scripts"))   # 复用我们自己的渲染器
from sim_walk_common import make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--n", type=int, default=60, help="采样帧数(沿参考轨迹均匀)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)   # 16:9, Stage B resize 到 1024x256(同上车)
    ap.add_argument("--fovy", type=float, default=75.0)
    ap.add_argument("--forward_only", action="store_true",
                    help="只渲前向段(起点->离起点最远点), 避开返程掉头(供真实路径GT用)")
    ap.add_argument("--tangent_yaw", action="store_true",
                    help="朝向用平滑路径切线(前视~win_m米)而非重建yaw(重建yaw噪, 会把GT转歪)")
    ap.add_argument("--yaw_win_m", type=float, default=2.0, help="切线朝向前视距离(米)")
    ap.add_argument("--outdir", default=None, help="输出目录(默认 arch_e2e/runs/obs_seq_<session>)")
    args = ap.parse_args()

    env, W = make_env(args.session, width=args.width, height=args.height, fovy=args.fovy)
    wp = W["waypoints_xy"]                # (K,2) 世界系参考轨迹
    yaw = None if args.tangent_yaw else (W["yaw"] if "yaw" in W else None)
    K = len(wp)
    lo, hi = 0, K - 1
    if args.forward_only:                 # 截到离起点最远点(前向单程)
        hi = int(np.linalg.norm(wp - wp[0], axis=1).argmax())
        print(f"[stageA] forward_only: 前向段 [0,{hi}] / 共{K}")

    # 切线朝向: 沿路径前视 win_m 米的方向(自然平滑, 朝走廊前方), 治重建yaw噪
    tan_yaw = None
    if args.tangent_yaw:
        seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))])
        tan_yaw = np.zeros(K)
        for k in range(K):
            j = k
            while j + 1 <= hi and seg[j] - seg[k] < args.yaw_win_m:
                j += 1
            j = max(j, min(k + 1, hi))
            tan_yaw[k] = np.arctan2(wp[j][1] - wp[k][1], wp[j][0] - wp[k][0])
        print(f"[stageA] tangent_yaw: 前视{args.yaw_win_m}m 切线朝向(忽略重建yaw)")

    idxs = np.linspace(lo, hi, min(args.n, hi - lo + 1)).astype(int)

    outdir = args.outdir or os.path.join(RUNS, f"obs_seq_{args.session}")
    os.makedirs(outdir, exist_ok=True)
    poses = []
    for i, k in enumerate(idxs):
        x, y = float(wp[k][0]), float(wp[k][1])
        if tan_yaw is not None:
            th = float(tan_yaw[k])
        elif yaw is not None:
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

    # merged_ply 指向我们自己重建的世界底图(repo/outputs), 供 Stage B 建占据/裁 BEV
    merged_ply = os.path.join(REPO, "outputs", f"gs_vggto_{args.session}_world.ply")
    np.savez(os.path.join(outdir, "poses.npz"),
             poses=np.array(poses, np.float32), session=args.session,
             merged_ply=merged_ply, fovy=args.fovy, n=len(idxs))
    print(f"[stageA] 存 {len(idxs)} 帧 -> {outdir}  (fpv jpg + poses.npz)")


if __name__ == "__main__":
    main()
