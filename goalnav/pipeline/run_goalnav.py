"""跑 goal-conditioned DiffusionDrive 闭环: 高层(A*/地图)出目标点 -> DD规划到点 -> 全向控制 -> 循环。
⭐目标点走 status/command 槽(官方原生通路), 无运动模式cond。A3D锐利皮肤。
用法(卡7):
  CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python goalnav/pipeline/run_goalnav.py --steps 500 \
    --skin_ply "$(pwd)/outputs/gs_a3d_colmapjoint_all_world_deg0.ply"
"""
import os, sys, argparse, time, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch, imageio, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
from test_env import CorridorTestEnv
import stage_b_eval as sb
import sim_fullchain as sf                                     # 复用 plan_path/traj_to_control/bev_safe_vel/绘图
from train_goalnav import status_with_goal, GOAL_AHEAD, robot_kmeans_anchors

RUNS = os.path.join(ROOT, "goalnav", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--skin_ply", default=None)
    ap.add_argument("--unify_occ", action="store_true", help="占据用皮肤同源几何(A3D), 治FPV/BEV几何不一致")
    ap.add_argument("--occ_thr", type=int, default=12, help="A3D稠密点云占据阈值")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    occ_ply = args.skin_ply if (args.unify_occ and args.skin_ply) else None
    env = CorridorTestEnv(skin_session=args.session, skin_ply=args.skin_ply,
                          occ_ply=occ_ply, occ_thr=args.occ_thr)
    ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    xyz = sb.read_ply_xyz(ply); _, _, _, floor = sb.build_occ(xyz)
    nav_path = sf.plan_path(env)
    print(f"[goalnav] A* 目标点路径: {len(nav_path) if nav_path is not None else '失败'} 点; "
          f"start={tuple(round(x,1) for x in env.start[:2])} goal={tuple(round(x,1) for x in env.goal)}", flush=True)

    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    ck = torch.load(os.path.join(RUNS, "goalnav_ft.pt"))
    model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
    model._status_encoding.load_state_dict(ck["status_encoding"])
    model.eval()
    print("[goalnav] 加载 goalnav_ft.pt (目标点条件, pristine机制)", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    mp4 = os.path.join(RUNS, f"goalnav_{args.session}_{stamp}.mp4")
    canon = os.path.join(RUNS, f"goalnav_{args.session}.mp4")
    jl = os.path.join(RUNS, f"goalnav_{args.session}_{stamp}.jsonl")
    writer = imageio.get_writer(mp4, fps=args.fps, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    flog = open(jl, "w", encoding="utf-8")
    obs = env.reset(); t0 = time.time(); v_ema = np.zeros(3, np.float32)
    for k in range(args.steps):
        pose = obs["pose"]; fpv_bgr = np.ascontiguousarray(obs["fpv"][:, :, ::-1])
        # 高层: A* 当前前方~GOAL_AHEAD 的局部目标点 -> ego
        d = np.hypot(nav_path[:, 0] - pose[0], nav_path[:, 1] - pose[1]); i0 = int(d.argmin())
        acc = 0.0; jj = i0
        while jj + 1 < len(nav_path) and acc < GOAL_AHEAD:
            acc += float(np.linalg.norm(nav_path[jj + 1] - nav_path[jj])); jj += 1
        c_, s_ = np.cos(pose[2]), np.sin(pose[2]); gx, gy = nav_path[jj] - pose[:2]
        goal_ego = np.array([c_ * gx + s_ * gy, -s_ * gx + c_ * gy], np.float32)
        st = status_with_goal(goal_ego, device=device)
        cam = sb.camera_feature(fpv_bgr, device); bev = sb.lidar_feature_fov(xyz, pose, floor, device)
        torch.manual_seed(0)                                    # ⭐固定扩散噪声=帧间不抖(治闪动)
        with torch.no_grad():
            traj = model({"camera_feature": cam, "lidar_feature": bev, "status_feature": st})["trajectory"][0].cpu().numpy()
        vx, vy, wz = sf.traj_to_control(traj)
        raw_vx, raw_vy, raw_wz = vx, vy, wz                # ⭐诊断: DD原生控制(未加转向项/未安全裁剪)
        dd_lat_mean = float(traj[:, 1].mean())             # DD轨迹横向(左=+)均值
        dd_lat_end = float(traj[-1, 1])                    # DD轨迹终点横向
        # ⭐利用全向底座【原地转】朝目标方向: 转头对准再走, 而非硬crab横移(治轨迹乱/别扭)
        goal_ang = float(np.arctan2(goal_ego[1], goal_ego[0]))
        wz = float(np.clip(wz + 1.2 * goal_ang, -0.8, 0.8))
        wz_after_goal = wz                                 # ⭐诊断: 加了goal_ang转向项后的wz
        if abs(goal_ang) > 0.5:                            # 目标偏离大: 先转对齐, 少横移
            vx *= 0.5; vy *= 0.3
        vx, vy = sf.bev_safe_vel(obs["bev"], vx, vy)
        dg = float(np.hypot(pose[0] - env.goal[0], pose[1] - env.goal[1]))
        if dg < 2.0:
            vx *= max(0.15, dg / 2.0); vy *= max(0.15, dg / 2.0)
        v_ema = 0.6 * v_ema + 0.4 * np.array([vx, vy, wz], np.float32)  # ⭐EMA平滑控制=不顿挫
        vx, vy, wz = float(v_ema[0]), float(v_ema[1]), float(v_ema[2])
        # 可视化
        col = (0, 200, 255)
        fv = fpv_bgr.copy(); sf.draw_traj_on_fpv(fv, pose, traj, col)
        cv2.rectangle(fv, (0, 0), (fv.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(fv, f"goal_ego=({goal_ego[0]:.1f},{goal_ego[1]:+.1f}) -> DD -> vx={vx:.2f} vy={vy:+.2f} wz={wz:+.2f}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 2)
        bv = sf.render_bev(obs["bev"], traj, col); H = fv.shape[0]; bv = cv2.resize(bv, (H, H))
        writer.append_data(np.hstack([fv, bv])[:, :, ::-1])
        obs, term, info = env.step(vx, vy, wz)
        rec = {"step": k, "pose": [round(float(p), 3) for p in pose],
               "goal_ego": [round(float(g), 2) for g in goal_ego],
               "goal_ang": round(goal_ang, 3),
               "dd_lat_mean": round(dd_lat_mean, 3), "dd_lat_end": round(dd_lat_end, 3),
               "raw_vy": round(raw_vy, 3), "raw_wz": round(raw_wz, 3),
               "wz_after_goal": round(wz_after_goal, 3),
               "vx": round(vx, 3), "vy": round(vy, 3), "wz": round(wz, 3),
               "collided": bool(info["collided"]), "reached": bool(info["reached"])}
        flog.write(json.dumps(rec) + "\n"); flog.flush()
        if k % 10 == 0 or term:
            print(f"  step{k:3d} pose=({pose[0]:.1f},{pose[1]:.1f}) goal_ego=({goal_ego[0]:.1f},{goal_ego[1]:+.1f}) "
                  f"vx={vx:.2f} vy={vy:+.2f} wz={wz:+.2f}", flush=True)
        if term:
            print(f"[goalnav] 终止 step{k}: {info}", flush=True); break
    writer.close(); flog.close(); shutil.copy(mp4, canon)
    print(f"[done] {time.time()-t0:.0f}s\n  版本 -> {mp4}\n  同名 -> {canon}", flush=True)


if __name__ == "__main__":
    main()
