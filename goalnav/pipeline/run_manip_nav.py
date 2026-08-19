"""191715 移动操作导航 Demo: 双脑泛化到新场景(实验室工作台, 非走廊泵房)。
任务=移动底盘配合机械臂取放: 去A(白桌+双层玻璃柜)等待取物 → 去B(上方有机械臂的桌)等待放置 → 返回A → 完成。
⭐硬约束(真机条件): 不许提前标定A/B坐标。机器人只靠VLM从相机找目标+判到没到位 = 无地图/无预知坐标/纯相机导航
   (把"VLM替代A*"落地)。VLM=慢脑经理(指方向AHEAD/LEFT/RIGHT/SEARCH + 确认ARRIVED + 叙述操作); DD=快脑(cruise+原地转)。
用法:
  CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
    VLM_URL=http://10.0.0.18:8152/api/generate \
    ~/discoverse_venv/bin/python goalnav/pipeline/run_manip_nav.py --steps 500 \
    --session colmapjoint_demo191715 --skin_ply outputs/gs_vggto_colmapjoint_demo191715_world.ply \
    --unify_occ --occ_thr 12 --wait_s 4
"""
import os, sys, argparse, time, json
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch, imageio, shutil
from PIL import Image, ImageDraw, ImageFont
_PANEL_FONT = None
for _fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(_fp):
        _PANEL_FONT = ImageFont.truetype(_fp, 15); break
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
from test_env import CorridorTestEnv
import stage_b_eval as sb
import sim_fullchain as sf
from train_goalnav import GOAL_AHEAD, robot_kmeans_anchors
from train_goalnav_turn import status_gh

RUNS = os.path.join(ROOT, "goalnav", "runs")


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def turn_wz(traj, cap=0.45):
    """gate-DD原地转轨迹→角速度(用末端预测yaw*增益, cap比转正温和以便找目标时慢慢转)。"""
    return float(np.clip(1.5 * float(traj[-1, 2]), -cap, cap))


def dd_traj(model, fpv_bgr, pose, goal_ego, xyz, floor, device, head_ego=(1.0, 0.0)):
    st = status_gh(np.asarray(goal_ego, np.float32), np.asarray(head_ego, np.float32), device=device)
    cam = sb.camera_feature(fpv_bgr, device); bev = sb.lidar_feature_fov(xyz, pose, floor, device)
    torch.manual_seed(0)
    with torch.no_grad():
        return model({"camera_feature": cam, "lidar_feature": bev, "status_feature": st})["trajectory"][0].cpu().numpy()


class RealFrameLookup:
    """位姿→最近真实采集帧(模拟真机相机, 不给坐标)。meta 可指定 191715 的 nav_ds。"""
    def __init__(self, meta_path):
        from scipy.spatial import cKDTree
        M = np.load(meta_path, allow_pickle=True)
        self.poses = M["poses"].astype(np.float32); self.fpv = [str(x) for x in M["fpv"]]
        p = self.poses
        self.feat = np.stack([p[:, 0], p[:, 1], 2.0 * np.cos(p[:, 2]), 2.0 * np.sin(p[:, 2])], 1)
        self.tree = cKDTree(self.feat); self._cache = {}
    def nearest(self, pose):
        q = np.array([pose[0], pose[1], 2.0 * np.cos(pose[2]), 2.0 * np.sin(pose[2])], np.float32)
        _, i = self.tree.query(q)
        if i not in self._cache:
            self._cache[i] = cv2.imread(self.fpv[i])
        return self._cache[i], i


# 目标语义描述(VLM找的东西), 不含任何坐标
# ⚠robo2vlm对【自然yes/no+方位】答得好, 对A/B/C/D/E多选会懵→默认吐E。故用自然问法。
CABINET = "a white cabinet with glass doors on a table"
ARMTABLE = "the table that has a robotic arm mounted above it"
# 方位问题的解析(robo2vlm会说 "It is on the right." / "In the center." 等)
OPT_SIDE = [("CENTER", ["CENTER", "AHEAD", "FRONT", "MIDDLE"]), ("LEFT", ["LEFT"]),
            ("RIGHT", ["RIGHT"]), ("NONE", ["NONE", "NOT VISIBLE", "CANNOT", "NO "])]


def q_arrived(t):
    return f"Look at your camera view. Is {t} right in front of you and very close (within about one meter)? Answer only yes or no."


def q_side(t):
    return (f"Look at your camera view for {t}. Is it on your LEFT, in the CENTER ahead, or on your RIGHT? "
            f"If you cannot see it at all, answer NONE. Answer with one word: LEFT, CENTER, RIGHT, or NONE.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_demo191715")
    ap.add_argument("--skin_ply", default=None)
    ap.add_argument("--nav_ds", default=os.path.join(ROOT, "outputs", "nav_ds_demo191715", "meta.npz"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--vlm_every", type=int, default=6, help="VLM决策间隔(步)")
    ap.add_argument("--unify_occ", action="store_true")
    ap.add_argument("--occ_thr", type=int, default=12)
    ap.add_argument("--wait_s", type=float, default=4.0, help="到站等待秒(机械臂操作占位)")
    ap.add_argument("--cruise_ahead", type=float, default=1.5, help="cruise前视目标距离(小场景调小)")
    ap.add_argument("--seek_turn_deg", type=float, default=90.0,
                    help="LEFT/RIGHT/SEARCH转角命令(度)。须≥54(gate训练|dyaw|∈[54,90]), 45°是OOD会符号乱→冻住")
    ap.add_argument("--arrive_confirm", type=int, default=2, help="连续N次ARRIVED才算到位(防抖)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def vlm_decide(frames, question, options, max_new=12):
        try:
            ans, _ = sf.vlm_api(frames, question, max_new_tokens=max_new, timeout=90)
        except Exception as e:
            return None, f"ERR:{str(e)[:16]}", False
        up = ans.strip().upper()
        for key, aliases in options:
            if any(a in up for a in aliases):
                return key, ans.strip(), True
        return None, ans.strip(), True

    real = RealFrameLookup(args.nav_ds)
    occ_ply = args.skin_ply if (args.unify_occ and args.skin_ply) else None
    env = CorridorTestEnv(skin_session=args.session, skin_ply=args.skin_ply,
                          occ_ply=occ_ply, occ_thr=args.occ_thr)
    ply = args.skin_ply or os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    xyz = sb.read_ply_xyz(ply); _, _, _, floor = sb.build_occ(xyz)
    print(f"[demo] 场景 {args.session}  start={tuple(np.round(env.start,2))}", flush=True)

    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    ck_name = os.environ.get("CKPT", "goalnav_gate.pt")
    ck = torch.load(os.path.join(RUNS, ck_name))
    model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
    model._status_encoding.load_state_dict(ck["status_encoding"]); model.eval()
    print(f"[demo] 加载 {ck_name}", flush=True)

    # 任务腿: 去A取物 → 去B放置 → 返回A → DONE
    LEGS = [
        {"name": "去A: 取物", "desc": CABINET, "wait_msg": "已到柜子, 等待机械臂取物(固定时间占位)"},
        {"name": "去B: 放置", "desc": ARMTABLE, "wait_msg": "已到机械臂桌, 等待放置(固定时间占位)"},
        {"name": "返回A", "desc": CABINET, "wait_msg": None},   # 最后一腿到位即DONE
    ]
    wait_steps = int(args.wait_s * args.fps)
    dyaw = np.deg2rad(args.seek_turn_deg)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    mp4 = os.path.join(RUNS, f"manip_nav_{args.session}_{stamp}.mp4")
    canon = os.path.join(RUNS, f"manip_nav_{args.session}.mp4")
    writer = imageio.get_writer(mp4, fps=args.fps, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    jl = open(os.path.join(RUNS, f"manip_nav_{args.session}_{stamp}.jsonl"), "w")

    obs = env.reset(); t0 = time.time(); v_ema = np.zeros(3, np.float32); last_coll = False
    leg_i = 0; phase = "SEEK"; wait_ctr = 0
    seek_cmd = "SEARCH"; vlm_ans = "(未调用)"; vlm_step = -99; vlm_ok = True; arrive_streak = 0
    real_i = -1; done_legs = []
    PHASE_COL = {"SEEK": (0, 200, 255), "WAIT": (0, 255, 255), "DONE": (0, 255, 0)}

    def log_panel(h, lines):
        if _PANEL_FONT is not None:
            img = Image.new("RGB", (560, h), (0, 0, 0)); dr = ImageDraw.Draw(img); yy = 6
            for txt, cc in lines:
                dr.text((10, yy), txt, font=_PANEL_FONT, fill=(cc[2], cc[1], cc[0])); yy += 22
            return np.array(img)[:, :, ::-1].copy()
        pan = np.zeros((h, 560, 3), np.uint8); yy = 26
        for txt, cc in lines:
            cv2.putText(pan, txt, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cc, 1, cv2.LINE_AA); yy += 26
        return pan

    for k in range(args.steps):
        pose = obs["pose"]; fpv = np.ascontiguousarray(obs["fpv"][:, :, ::-1])
        x, y, yaw = pose; vx = vy = wz = 0.0; traj = None; goal_ego = np.zeros(2, np.float32)
        src = "-"; safe_clip = False
        rimg_cur, real_i = real.nearest(pose)   # 模拟真机相机(VLM+DD都看真图, 避免面墙灰雾)
        state = "DONE" if leg_i >= len(LEGS) else phase
        leg = LEGS[min(leg_i, len(LEGS) - 1)]

        if state == "SEEK":
            # 慢脑(robo2vlm友好): 先yes/no问到没到, 没到再问方位。期间沿用上次决策。
            if k % args.vlm_every == 0:
                ar, _, ok1 = vlm_decide([rimg_cur], q_arrived(leg["desc"]),
                                        [("YES", ["YES"]), ("NO", ["NO"])], max_new=6); vlm_step = k
                if ok1 and ar == "YES":
                    seek_cmd = "ARRIVED"; vlm_ok = True
                else:
                    sd, _, ok2 = vlm_decide([rimg_cur], q_side(leg["desc"]), OPT_SIDE, max_new=8)
                    vlm_ok = ok1 and ok2
                    seek_cmd = {"CENTER": "AHEAD", "LEFT": "LEFT", "RIGHT": "RIGHT",
                                "NONE": "SEARCH", None: "SEARCH"}.get(sd, "SEARCH")
                arrive_streak = arrive_streak + 1 if seek_cmd == "ARRIVED" else 0
                vlm_ans = {"ARRIVED": "到位", "AHEAD": "在前方->直行", "LEFT": "在左->左转",
                           "RIGHT": "在右->右转", "SEARCH": "没看见->转找"}.get(seek_cmd, "?") \
                    + ("" if vlm_ok else "(VLM挂)")
                if arrive_streak >= args.arrive_confirm:
                    # 到站: 进等待(最后一腿则完成)
                    done_legs.append(leg["name"])
                    if leg["wait_msg"] is None:
                        leg_i += 1; phase = "DONE"
                    else:
                        phase = "WAIT"; wait_ctr = 0
                    seek_cmd = "STOP"; arrive_streak = 0
            # 快脑: 按决策出动作(只用cruise+原地转, 绕开OOD的strafe)
            if seek_cmd == "AHEAD":
                goal_ego = np.array([args.cruise_ahead, 0.0], np.float32); src = "DD cruise(直行)"
                traj = dd_traj(model, rimg_cur, pose, goal_ego, xyz, floor, device, head_ego=(1.0, 0.0))
                vx, vy, wz = sf.traj_to_control(traj)
                svx, svy = sf.bev_safe_vel(obs["bev"], vx, vy); safe_clip = (svx != vx or svy != vy); vx, vy = svx, svy
            elif seek_cmd in ("LEFT", "RIGHT", "SEARCH"):
                sgn = 1.0 if seek_cmd in ("LEFT", "SEARCH") else -1.0   # 左+ 右- ; SEARCH默认左扫
                traj = dd_traj(model, rimg_cur, pose, np.zeros(2, np.float32), xyz, floor, device,
                               head_ego=(np.cos(sgn * dyaw), np.sin(sgn * dyaw)))
                wz = turn_wz(traj); vx = vy = 0.0; src = f"DD原地转({'左' if sgn>0 else '右'}找)"
            else:  # ARRIVED未确认 / STOP
                vx = vy = wz = 0.0; src = "停(待确认到位)"

        elif state == "WAIT":
            vx = vy = wz = 0.0; wait_ctr += 1; src = "等待机械臂"
            if wait_ctr >= wait_steps:
                leg_i += 1; phase = "SEEK"; seek_cmd = "SEARCH"; arrive_streak = 0

        v_ema = 0.6 * v_ema + 0.4 * np.array([vx, vy, wz], np.float32)
        vx, vy, wz = float(v_ema[0]), float(v_ema[1]), float(v_ema[2])

        col = PHASE_COL[state]; fv = fpv.copy()
        if traj is not None:
            sf.draw_traj_on_fpv(fv, pose, traj, col)
        Hh = fv.shape[0]
        # 真图inset(左下)
        iw = fv.shape[1] // 4; ih = int(iw * rimg_cur.shape[0] / rimg_cur.shape[1])
        ins = cv2.resize(rimg_cur, (iw, ih)); y0 = Hh - ih - 8; x0 = 8
        fv[y0:y0 + ih, x0:x0 + iw] = ins
        cv2.rectangle(fv, (x0, y0), (x0 + iw, y0 + ih), (0, 220, 255), 2)
        cv2.rectangle(fv, (x0, y0 - 20), (x0 + iw, y0), (0, 0, 0), -1)
        cv2.putText(fv, "VLM sees REAL cam", (x0 + 3, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

        W_ = (255, 255, 255); Y_ = (0, 220, 255); G_ = (0, 255, 120); R_ = (0, 80, 255); B_ = (255, 180, 0)
        target_name = "柜子桌(A)" if leg["desc"] == CABINET else "机械臂桌(B)"
        lines = [
            (f"step {k}   {state}   任务腿: {leg['name']}", col),
            (f"pose x={x:.2f} y={y:.2f} yaw={np.degrees(yaw):+.0f}deg", W_),
            ("== 慢脑 VLM(经理, 无地图导航) ==", B_),
            (f"  找目标: {target_name}", W_),
            (f"  VLM@{vlm_step}: {vlm_ans[:24]}", (R_ if not vlm_ok else Y_)),
            (f"  到位确认: {arrive_streak}/{args.arrive_confirm}", W_),
        ]
        if state == "WAIT":
            lines.append((f"  {leg['wait_msg'][:26]}", G_))
            lines.append((f"  等待 {wait_ctr}/{wait_steps} 步", G_))
        lines += [
            ("  已完成腿:", B_),
        ]
        for nm in done_legs[-3:]:
            lines.append((f"    ✓ {nm}", G_))
        lines += [
            ("== 快脑 DD (HOW) ==", B_),
            (f"  src={src}", Y_),
            (f"  FINAL vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}", G_),
            (f"  bev_safe={'裁剪!' if safe_clip else 'ok'} coll={'YES' if last_coll else 'no'}",
             (R_ if (safe_clip or last_coll) else W_)),
        ]
        pan = log_panel(Hh, lines)
        writer.append_data(np.hstack([fv, pan])[:, :, ::-1])
        jl.write(json.dumps({"step": k, "state": state, "leg": leg_i, "seek": seek_cmd,
                             "pose": [round(float(p), 2) for p in pose]}) + "\n"); jl.flush()
        if k % 10 == 0 or state == "DONE":
            print(f"  step{k:3d} [{state}] leg={leg_i}({leg['name']}) seek={seek_cmd} "
                  f"pose=({x:.2f},{y:.2f},{np.degrees(yaw):+.0f}) 到位{arrive_streak}", flush=True)
        if state == "DONE":
            print(f"[demo] 完成! step{k}  完成腿: {done_legs}", flush=True)
            obs, _, _ = env.step(0, 0, 0); break
        obs, term, info = env.step(vx, vy, wz)
        last_coll = bool(info["collided"])
        if last_coll:
            print(f"  step{k} 碰撞! pose=({x:.2f},{y:.2f}) leg={leg_i}", flush=True)

    writer.close(); jl.close(); shutil.copy(mp4, canon)
    print(f"[done] {time.time()-t0:.0f}s  完成腿{len(done_legs)}/{len(LEGS)}\n  {mp4}\n  {canon}", flush=True)


if __name__ == "__main__":
    main()
