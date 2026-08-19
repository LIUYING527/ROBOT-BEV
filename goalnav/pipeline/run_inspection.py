"""巡检闭环: VLM当【项目经理】(主任务=巡完走廊+拆检视子任务+记进度), 几何报设备坐标, 动作=DD巡航+控制器机动。
状态机: CRUISE(DD沿A*朝前) -> 到设备前沿 派发子任务 -> TURN_TO_FACE(控制器转正) -> VLM经理确认(A跳过/B执行)
   -> SCAN(控制器横扫到后沿) -> TURN_BACK(控制器转回) -> 记忆记录子任务 -> 继续 -> 下一设备 -> 主任务完成DONE。
⭐分工: 项目经理VLM=决策"该不该检视这台"+记住干到哪(语义/长时程, 它的长处);
        几何=报设备在哪+哪侧+前后沿(坐标, VLM不碰); DD=巡航HOW; 控制器=转正/横扫/转回(暂, 后续可交DD)。
   A3D锐利皮肤。
用法(空卡):
  CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python goalnav/pipeline/run_inspection.py --steps 600 \
    --skin_ply "$(pwd)/outputs/gs_a3d_colmapjoint_all_world_deg0.ply"
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
from train_goalnav import status_with_goal, GOAL_AHEAD, robot_kmeans_anchors
from train_goalnav_turn import status_gh   # 目标点+目标朝向(gate模型转向命令)

RUNS = os.path.join(ROOT, "goalnav", "runs")


TURN_DD_MIN = 0.9   # rad(~51°): 剩余转角>此值用gate-DD原地转(训练分布内), 以下切控制器收尾(DD在<54°是OOD)


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def turn_wz(traj):
    """gate-DD原地转轨迹(8,3=x,y,yaw)→角速度wz: 用DD预测的末端yaw(≈命令dyaw)乘比例增益。
    ⚠不能用sf.traj_to_control: 那按【距离】选前视点, 纯旋转xy≈0→选回起点(yaw≈0)→wz≈0=转不动。"""
    return float(np.clip(1.5 * float(traj[-1, 2]), -0.8, 0.8))


def dd_traj(model, fpv_bgr, pose, goal_ego, xyz, floor, device, head_ego=(1.0, 0.0)):
    # gate模型用 status_gh(目标点+目标朝向): CRUISE给前视目标点+朝向前(1,0);
    # 转向命令给目标点(0,0)+目标朝向(cos e,sin e)=命令门控关场景→干净原地转。
    st = status_gh(np.asarray(goal_ego, np.float32), np.asarray(head_ego, np.float32), device=device)
    cam = sb.camera_feature(fpv_bgr, device); bev = sb.lidar_feature_fov(xyz, pose, floor, device)
    torch.manual_seed(0)
    with torch.no_grad():
        return model({"camera_feature": cam, "lidar_feature": bev, "status_feature": st})["trajectory"][0].cpu().numpy()


class RealFrameLookup:
    """位姿 -> 最近真实采集帧(方案B): 仿真转头看设备是灰雾, VLM感知决策改喂真图。
    真机上本就是真相机供图, 这正好模拟真机。按(x,y,cos yaw,sin yaw)最近邻。"""
    def __init__(self, nav_ds=None):
        from scipy.spatial import cKDTree
        meta = nav_ds if (nav_ds and nav_ds.endswith(".npz")) else \
            os.path.join(nav_ds or os.path.join(ROOT, "outputs", "nav_ds_real"), "meta.npz")
        M = np.load(meta, allow_pickle=True)
        self.poses = M["poses"].astype(np.float32); self.fpv = [str(x) for x in M["fpv"]]
        p = self.poses
        self.feat = np.stack([p[:, 0], p[:, 1], 2.0 * np.cos(p[:, 2]), 2.0 * np.sin(p[:, 2])], 1)  # yaw加权
        self.tree = cKDTree(self.feat)
        self._cache = {}
    def nearest(self, pose):
        q = np.array([pose[0], pose[1], 2.0 * np.cos(pose[2]), 2.0 * np.sin(pose[2])], np.float32)
        _, i = self.tree.query(q)
        if i not in self._cache:
            self._cache[i] = cv2.imread(self.fpv[i])
        return self._cache[i], float(self.poses[i, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--skin_ply", default=None)
    ap.add_argument("--vlm_every", type=int, default=10, help="慢脑VLM每N步决策一次(异步)")
    ap.add_argument("--no_vlm", action="store_true", help="关掉VLM纯几何(对照)")
    ap.add_argument("--unify_occ", action="store_true", help="占据用皮肤同源几何(A3D), 治FPV/BEV不一致 + 居中A*")
    ap.add_argument("--occ_thr", type=int, default=12, help="A3D稠密点云占据阈值")
    ap.add_argument("--vlm_realimg", action="store_true",
                    help="⭐VLM驱动巡检: 巡航VLM见蓝泵进子任务; 横扫VLM见蓝泵离中心退出(+几何兜底)")
    ap.add_argument("--vlm_manager", action="store_true",
                    help="⭐VLM当项目经理(多选编排): CRUISE=INSPECT/CONTINUE/END, SCAN=KEEP/FINISH, 替代yes/no薄化FSM")
    ap.add_argument("--capcam_fpv", action="store_true",
                    help="⭐FPV用真实采集相机内参+gsplat渲染(capture-cam质量,锐且正), 替代discoverse sim相机(fovy不匹配显歪)")
    ap.add_argument("--follow_capture", action="store_true",
                    help="⭐巡航贴采集轨迹走(on-path=渲染最锐), 配普通渲染=平滑+锐; 室外单程场景推荐")
    ap.add_argument("--geo_vlm", action="store_true",
                    help="⭐B方案: 几何候选(device_clusters凸出物)定'在哪转'(无空转)+VLM面向时yes/no分类'是不是目标'(可靠·支持换目标)")
    ap.add_argument("--vlm_temporal", action="store_true",
                    help="⭐时序慢脑(配geo_vlm): 面向分类从单帧升级为滑窗多帧(window_last, 实测精确率+5.8pt)"
                         "+任务记忆注入prompt(已检站点x位置+判决); 流式=服务器不支持(实测), 短答先出替代")
    ap.add_argument("--speed", type=float, default=1.0, help="平移速度缩放(<1变慢, 转向wz不缩; 慢了记得加--steps)")
    ap.add_argument("--min_sub_gap", type=float, default=2.5,
                    help="两次检视子任务的最小间距(米), 治重复检视同一泵。泵密调小/稀调大")
    ap.add_argument("--vlm_view", choices=["sim", "real"], default="real",
                    help="VLM感知源(默认real=方案B): real=按位姿匹配真图, VLM进/退两头都看真图=忠实模拟真机真相机、零提前标; "
                         "sim=直接看仿真渲染(前视连贯但面墙灰雾, 退出会乱)")
    ap.add_argument("--dev_side", type=float, default=1.0, help="设备所在侧(此泵房全在左墙=+1)")
    ap.add_argument("--target", default="a blue water pump",
                    help="⭐要检视的设备语义描述(换prompt即换检视目标, 验证'换指令→检视不同设备'). "
                         "例: 'an electrical cabinet with a red indicator light inside'")
    ap.add_argument("--nav_ds", default=None,
                    help="真图数据集目录或meta.npz(RealFrameLookup用, 换场景改这个). 默认走廊 outputs/nav_ds_real")
    ap.add_argument("--clusters", default=None,
                    help="设备站位 .npy 路径 (K,5)=[x,y,side,x_lo,x_hi]. 默认走廊 outputs/device_clusters.npy")
    ap.add_argument("--occ_ply", default=None,
                    help="占据/导航专用ply(与skin_ply解耦): 锐皮肤高斯太密→占据过密挤偏导航时, "
                         "用更稀的同源ply(如358k版)做占据, skin_ply仍用锐版渲染. 默认=skin_ply(unify_occ)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def vlm_manager(frames, n_done, n_total):
        """慢脑=项目经理决策: 转正对着候选设备后, 结合主任务进度决定这个子任务【执行还是取消】。
        返回(原话, do_inspect, ok)。已验证'赶路A/检视B'二选一真图+渲染都可靠。
        ⚠️ok=False(VLM挂/超时)时【不吞错】: 上层降级到几何(信任几何候选照执行), 面板标红。"""
        pr = (f"You are a patrol robot. Main task: inspect all equipment along this corridor. "
              f"Progress: {n_done} of {n_total} equipment stations done so far. "
              f"You just turned to face something beside you. Decide the subtask: "
              f"A) it is just corridor / nothing to inspect -> keep patrolling; "
              f"B) it is equipment -> inspect it now. Answer only A or B.")
        try:
            ans, _ = sf.vlm_api(frames, pr, max_new_tokens=14, timeout=60)
        except Exception as e:
            return f"ERR:{str(e)[:18]}", False, False    # ok=False: 报错, 上层降级不当决策
        a = ans.strip()
        return a, (a[:2].upper().startswith("B") or "inspect" in a.lower()), True

    def vlm_yesno(img, prompt):
        """真图问答, 返回(yes布尔, ok)。VLM挂=ok False。"""
        try:
            ans, _ = sf.vlm_api(img, prompt, max_new_tokens=8, timeout=40)
            yes = "yes" in ans.strip().lower()
            if os.environ.get("VLM_DEBUG"):
                print(f"    [VLM] Q='{prompt[:40]}...' -> '{ans.strip()[:20]}' => {'YES' if yes else 'NO'}", flush=True)
            return yes, True
        except Exception:
            return False, False
    def vlm_decide(frames, question, options, max_new=12):
        """⭐VLM当项目经理:多选决策(替代yes/no)。options=[(key,[大写别名...]),...]。
        返回(选中key or None, 原话, ok)。VLM挂或没解析出→key=None(上层降级几何)。"""
        try:
            ans, _ = sf.vlm_api(frames, question, max_new_tokens=max_new, timeout=90)
        except Exception as e:
            return None, f"ERR:{str(e)[:16]}", False
        up = ans.strip().upper()
        for key, aliases in options:
            if any(a in up for a in aliases):
                return key, ans.strip(), True
        return None, ans.strip(), True

    TGT = args.target   # ⭐检视目标(可换prompt): 默认蓝泵; 可换"内有红灯的电柜"等
    Q_ENTER = (f"Is there {TGT} right beside you, close and large in view? Answer only yes or no.")
    Q_EXIT = (f"Look at the CENTER of your view. Is there {TGT} right in the middle/front? "
              "Answer only yes or no.")
    # ⭐VLM编排prompt(多选, 项目经理直接决定跳转)
    Q_MGR_CRUISE = ("You are a patrol robot. You ONLY inspect this specific target: " + TGT + ". "
                    "Look at the equipment beside you and choose ONE, answer with only the letter:\n"
                    f"A) INSPECT - the equipment right beside you IS clearly {TGT}, inspect it now;\n"
                    f"B) CONTINUE - the equipment beside you is a DIFFERENT kind of equipment (NOT {TGT}), "
                    "or it is just empty corridor; keep patrolling forward;\n"
                    "C) END - you have reached the end of the corridor, nothing more ahead.\n"
                    "Only answer A if it truly matches the target; otherwise B.")
    Q_MGR_SCAN = ("You are scanning a piece of equipment beside you while sliding past it. "
                  "Choose ONE and answer with only the letter:\n"
                  f"A) KEEP - {TGT} is still in your view, keep scanning it;\n"
                  "B) FINISH - you have fully passed it / it left the view, stop scanning.")
    OPT_CRUISE = [("INSPECT", ["A", "INSPECT", "PUMP"]), ("CONTINUE", ["B", "CONTINUE", "CORRIDOR"]),
                  ("END", ["C", "END", "DONE"])]
    OPT_SCAN = [("KEEP", ["A", "KEEP", "STILL"]), ("FINISH", ["B", "FINISH", "STOP", "PASSED"])]
    # ⭐转正后确认(面向设备=前视粗判后的精判, 也治过度触发): 是目标才检视, 不是就跳过
    Q_CONFIRM = (f"You have turned to face the equipment beside you. Is it {TGT}? "
                 f"Answer yes only if it clearly IS {TGT}, otherwise no. Answer yes or no.")
    OPT_YN = [("YES", ["YES"]), ("NO", ["NO"])]
    real = RealFrameLookup(args.nav_ds) if ((args.vlm_realimg and args.vlm_view == "real") or args.geo_vlm) else None
    capfpv = None
    if args.capcam_fpv:
        from capture_fpv import CaptureFPV
        capfpv = CaptureFPV(args.session)
        print("[巡检] FPV用capture-cam渲染(真实内参+gsplat)", flush=True)
    Q_CLASSIFY = f"You are now facing a piece of equipment beside you. Is it {TGT}? Answer only yes or no."

    # ⭐时序慢脑: 滑窗多帧+任务记忆(0727, 走廊实测window_last比单帧精确率+5.8pt/FP-24%)
    sb_vlm = None
    if args.vlm_temporal:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from vlm_temporal_slowbrain import TemporalSlowBrain
        sb_vlm = TemporalSlowBrain(target=TGT, win=4, push_every=2)
        print("[巡检] 时序慢脑ON: 滑窗K=4 window_last分类 + 任务记忆注入", flush=True)

    occ_ply = args.occ_ply or (args.skin_ply if (args.unify_occ and args.skin_ply) else None)
    env = CorridorTestEnv(skin_session=args.session, skin_ply=args.skin_ply,
                          occ_ply=occ_ply, occ_thr=args.occ_thr)
    ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    xyz = sb.read_ply_xyz(ply); _, _, _, floor = sb.build_occ(xyz)
    nav_path = sf.plan_path(env)
    if args.follow_capture or capfpv is not None:
        # ⭐贴着【采集轨迹】巡航(采集路径无障碍+3DGS在此最锐)→ on-path渲染最锐; 偏离则off-path糊。
        #   配普通discoverse渲染=平滑(连续位姿)+锐(on-path), 不用capcam回放取巧。
        wp_cap = env.Wd["waypoints_xy"].astype(np.float32)
        xpk = int(np.argmax(wp_cap[:, 0]))                # 只取去程(到最远点), 返程由状态机反向走
        nav_path = wp_cap[:xpk + 1]
        print(f"[巡检] nav_path改用采集轨迹 {len(nav_path)}点(贴路径保持on-path锐)", flush=True)
    clusters = np.load(args.clusters or os.path.join(ROOT, "outputs", "device_clusters.npy"))   # (K,5) x,y,side,x_lo,x_hi
    print(f"[巡检] A*路径{len(nav_path)}点; 设备簇{len(clusters)}个: "
          f"{[tuple(np.round(c[:2],1)) for c in clusters]}", flush=True)

    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    # ⭐gate模型(办法C): 从ft热启动+命令门控, 保留cruise/横扫又新增干净原地转→CRUISE+TURN单一快脑
    ck_name = os.environ.get("CKPT", "goalnav_gate.pt")
    ck = torch.load(os.path.join(RUNS, ck_name))
    model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
    model._status_encoding.load_state_dict(ck["status_encoding"]); model.eval()
    print(f"[巡检] 加载 {ck_name} (cruise+横扫+命令门控原地转)", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    mp4 = os.path.join(RUNS, f"inspection_{args.session}_{stamp}.mp4")
    canon = os.path.join(RUNS, f"inspection_{args.session}.mp4")
    writer = imageio.get_writer(mp4, fps=args.fps, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    jl = open(os.path.join(RUNS, f"inspection_{args.session}_{stamp}.jsonl"), "w")

    inspected = np.zeros(len(clusters), bool)
    state = "CRUISE"; target = -1; scan_along0 = None; scan_dir = 1.0; scan_face = 0.0
    direction = 1                      # ⭐+1去程(沿A*向goal) / -1返程(向start)
    around_target = 0.0                # TURN_AROUND目标yaw
    obs = env.reset(); t0 = time.time(); v_ema = np.zeros(3, np.float32); last_coll = False
    fpv_buf = []; vlm_ans = "(未调用)"; vlm_inspect = False; vlm_ok = True; vlm_step = -99   # 慢脑状态
    # ⭐项目经理(VLM)状态: 主任务 + 子任务记忆 + 当前决策
    MAIN_TASK = "巡完走廊(去程+返程), 检视全部设备"
    memory = []                        # 已处理子任务记忆: [(id, x, side, 结果)]
    sub_n = 0; cur_sub = None          # 子任务计数 / 当前活动子任务
    mgr_decision = "巡逻中(去程)"       # 当前经理决策(面板)
    enter_prev = False; last_sub_x = -99.0   # VLM进入去抖(方案B): 上升沿 + 最小间隔
    last_sub_xy = np.array([env.start[0], env.start[1]])  # 初始化为起点→开头需走够min_sub_gap才检视(治过早触发)
    rimg_buf = []                      # ⭐多帧真图缓冲(喂VLM时序上下文)
    real_x = -1.0                      # 当前喂VLM的真图对应x(面板披露)
    STATE_COL = {"CRUISE": (0, 200, 255), "TURN_TO_FACE": (180, 0, 255), "TURN_AROUND": (255, 120, 0),
                 "SCAN": (0, 255, 255), "TURN_BACK": (255, 0, 180), "DONE": (0, 255, 0)}

    def log_panel(h, lines):
        # ⭐PIL渲染=支持中文(cv2.putText的Hershey字体无CJK会变???)。颜色cc是BGR, PIL用RGB故翻转
        if _PANEL_FONT is not None:
            img = Image.new("RGB", (560, h), (0, 0, 0)); dr = ImageDraw.Draw(img)
            yy = 6
            for txt, cc in lines:
                dr.text((10, yy), txt, font=_PANEL_FONT, fill=(cc[2], cc[1], cc[0])); yy += 22
            return np.array(img)[:, :, ::-1].copy()          # RGB->BGR 对齐fv
        pan = np.zeros((h, 560, 3), np.uint8); yy = 26
        for txt, cc in lines:
            cv2.putText(pan, txt, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, cc, 1, cv2.LINE_AA); yy += 26
        return pan

    for k in range(args.steps):
        pose = obs["pose"]
        x, y, yaw = pose; vx = vy = wz = 0.0; traj = None; goal_ego = np.zeros(2, np.float32)
        if capfpv is not None:
            # ⭐全状态用 render(x,y,yaw): 按连续机器人位姿解析渲染(自然运动+转向跟随, 无抓帧跳变),
            #   已含采集pitch + 自动补该处场景roll(画面水平)。on-path锐; 远端/侧扫off-path糊=采集天花板不变。
            cap_rgb = capfpv.render(x, y, yaw)
            fpv = np.ascontiguousarray(cap_rgb[:, :, ::-1])                   # capture-cam(RGB)->BGR
        else:
            fpv = np.ascontiguousarray(obs["fpv"][:, :, ::-1])
        dd_raw = (0.0, 0.0, 0.0); src = "-"; safe_clip = False   # 部件输出记录

        # A* 前视目标点: 沿direction(去程向goal/返程向start)在路径上前视
        d = np.hypot(nav_path[:, 0] - x, nav_path[:, 1] - y); i0 = int(d.argmin())
        jj = i0; acc = 0.0
        while 0 <= jj + direction < len(nav_path) and acc < GOAL_AHEAD:
            acc += float(np.linalg.norm(nav_path[jj + direction] - nav_path[jj])); jj += direction
        c_, s_ = np.cos(yaw), np.sin(yaw)
        gx, gy = nav_path[jj] - pose[:2]
        cruise_goal = np.array([c_ * gx + s_ * gy, -s_ * gx + c_ * gy], np.float32)
        j2 = min(i0 + 3, len(nav_path) - 1)
        path_tan = float(np.arctan2(nav_path[j2, 1] - nav_path[i0, 1], nav_path[j2, 0] - nav_path[i0, 0]))
        travel_tan = path_tan if direction > 0 else wrap(path_tan + np.pi)  # 机器人行进朝向(返程反向)

        # ---- FPV时序缓冲(喂VLM多帧) ----
        if k % 3 == 0:
            fpv_buf.append(fpv.copy()); fpv_buf[:] = fpv_buf[-3:]

        # ⭐VLM感知源: sim=直接用当前仿真渲染(连贯); real=按位姿匹配真图(inset会跳)
        rimg_cur = None
        if args.vlm_realimg and args.vlm_view == "sim":
            rimg_cur = fpv; real_x = x                     # 仿真前视渲染(BGR), on-path够锐
        elif real is not None:
            rimg_cur, real_x = real.nearest(pose)          # 真图(vlm_realimg real / geo_vlm)
        # ⚠多帧反而让检测飘(不同位姿的帧不一致)→回退单帧
        vlm_frames = [rimg_cur] if rimg_cur is not None else None
        # ⭐时序慢脑滑窗: 每步推真图帧(内部按push_every抽样), 面向分类时用window_last判据
        if sb_vlm is not None and rimg_cur is not None:
            sb_vlm.push(rimg_cur, pose=(x, y))

        if state == "CRUISE":
            goal_ego = cruise_goal; src = "DD(fast)"
            traj = dd_traj(model, fpv, pose, goal_ego, xyz, floor, device)
            vx, vy, wz = sf.traj_to_control(traj); dd_raw = (vx, vy, wz)
            ga = float(np.arctan2(goal_ego[1], goal_ego[0])); wz = float(np.clip(wz + 1.2 * ga, -0.8, 0.8))
            svx, svy = sf.bev_safe_vel(obs["bev"], vx, vy); safe_clip = (svx != vx or svy != vy); vx, vy = svx, svy
            if args.vlm_realimg and args.vlm_manager:
                # ⭐VLM项目经理编排: 巡航中直接多选决定 INSPECT/CONTINUE/END(替代yes/no, FSM薄化为执行)
                if k % args.vlm_every == 0:
                    key, raw, vlm_ok = vlm_decide(vlm_frames, Q_MGR_CRUISE, OPT_CRUISE); vlm_step = k
                    if not vlm_ok:
                        key = None
                    vlm_ans = {"INSPECT": "经理:检视", "CONTINUE": "经理:继续巡航",
                               "END": "经理:走廊到头", None: "经理:未解析"}.get(key, "?") + ("" if vlm_ok else "(VLM挂降级几何)")
                    # ⭐上升沿去抖(改回): 仅VLM从非INSPECT跳变到INSPECT那下触发, 空走廊段(有CONTINUE)不连续触发→治空转。
                    #   +距离gap防同一设备重复。代价: 间距<gap的相邻设备可能并一个(此场景设备间距大, 无碍)。
                    if key == "INSPECT" and not enter_prev and np.hypot(x - last_sub_xy[0], y - last_sub_xy[1]) > args.min_sub_gap:
                        sub_n += 1; last_sub_xy = np.array([x, y])
                        cur_sub = {"id": sub_n, "x": float(x), "side": float(args.dev_side), "x_hi": None}
                        seg = "去程" if direction > 0 else "返程"
                        mgr_decision = f"VLM经理->检视({seg}) 子任务#{sub_n}@x{x:.0f}"
                        state = "TURN_TO_FACE"
                    elif key == "END":
                        # 去程到头→原地转180°返程; 返程到头(回到起点附近)→完成
                        if direction > 0 and x > 28.0:
                            around_target = wrap(yaw + np.pi)   # 目标=当前朝向反转180°
                            mgr_decision = "VLM经理->走廊到头, 原地转180°返程"
                            state = "TURN_AROUND"
                        elif direction < 0 and x < 8.0:
                            mgr_decision = "VLM经理->返程到起点, 主任务完成"
                            state = "DONE"
                    enter_prev = (key == "INSPECT")
            elif args.vlm_realimg:
                # 方案B(yes/no): VLM看真图发现蓝泵 -> 进子任务(上升沿+最小间隔3m去抖)
                if k % args.vlm_every == 0:
                    yes, vlm_ok = vlm_yesno(rimg_cur, Q_ENTER); vlm_step = k
                    vlm_ans = ("蓝泵-有" if yes else "蓝泵-无") + ("" if vlm_ok else "(VLM挂)")
                    if yes and not enter_prev and (x - last_sub_x) > 3.0:
                        sub_n += 1; last_sub_x = x
                        cur_sub = {"id": sub_n, "x": float(x), "side": float(args.dev_side), "x_hi": None}
                        mgr_decision = f"VLM发现蓝泵->派发子任务#{sub_n}@x{x:.0f}"
                        state = "TURN_TO_FACE"
                    enter_prev = yes
            else:
                # 几何模式: 到设备前沿派发子任务
                cand = [ci for ci in range(len(clusters)) if not inspected[ci]
                        and clusters[ci, 3] <= x <= clusters[ci, 4] and abs(clusters[ci, 1] - y) < 3.5]
                if cand:
                    target = min(cand, key=lambda ci: clusters[ci, 3])
                    sub_n += 1; cur_sub = {"id": sub_n, "x": float(clusters[target, 0]),
                                           "side": float(clusters[target, 2]), "x_hi": float(clusters[target, 4])}
                    mgr_decision = f"派发子任务#{sub_n}: 检视设备@x{cur_sub['x']:.0f}"
                    state = "TURN_TO_FACE"

        elif state == "TURN_TO_FACE":
            side = float(cur_sub["side"])                           # 几何/facility先验定侧
            face_yaw = path_tan + (np.pi / 2 if side > 0 else -np.pi / 2)   # 切向±90°对准设备侧
            e = wrap(face_yaw - yaw)
            # ⭐混合原地转: 大角度(训练分布|dyaw|∈[54°,90°]内)用gate-DD原地转=展示可控转向;
            #   剩余小角度(<~51°=DD的OOD区)切控制器wz=1.5*e收尾=稳。两者都强制vx=vy=0。
            if abs(e) > TURN_DD_MIN:
                ec = float(np.clip(e, -np.pi / 2, np.pi / 2))   # 夹到训练范围|dyaw|≤90°(超90°是OOD→符号翻转走长路)
                traj = dd_traj(model, fpv, pose, np.zeros(2, np.float32), xyz, floor, device,
                               head_ego=(np.cos(ec), np.sin(ec)))
                wz = turn_wz(traj); vx = vy = 0.0
                dd_raw = (0.0, 0.0, wz); src = "DD(原地转正)"
            else:
                wz = float(np.clip(1.5 * e, -0.8, 0.8)); vx = vy = 0.0; src = "控制器(转正收尾)"
            if abs(e) < 0.15:
                if args.vlm_realimg:
                    # 转正后直接进横扫(进入已由巡航VLM决定)。⚠曾试"转正后VLM再确认"但robo2vlm面向确认不稳→
                    #   乱拒绝→转正不扫就转回=横跳, 已回退。选择性靠巡航INSPECT判据, 不在这加二次门。
                    mgr_decision = f"子任务#{cur_sub['id']} 执行->横扫检视中"
                    state = "SCAN"; scan_along0 = np.array([x, y], float); scan_face = face_yaw
                else:
                    # 几何候选→转正后VLM【面向时yes/no分类】(可靠, 支持换目标): 是目标才扫, 不是就跳过。
                    #   --no_vlm 则自动执行(纯几何对照)。分类喂真图rimg_cur(面向时仿真是灰雾)。
                    if args.no_vlm:
                        vlm_inspect, vok = True, True; vlm_ans = "(no_vlm)"
                    elif sb_vlm is not None:
                        # ⭐时序慢脑: 滑窗window_last分类 + 任务记忆注入(已检站点x位置+判决)
                        yes, vok, lat = sb_vlm.classify_facing(cur_sub["x"]); vlm_step = k
                        vlm_inspect = yes
                        vlm_ans = (f"时序分类:是目标({lat:.1f}s)" if yes else f"时序分类:非目标({lat:.1f}s)") + ("" if vok else "(VLM挂)")
                        if vok:
                            sb_vlm.remember(cur_sub["x"], TGT, "确认是目标" if yes else "跳过(非目标)")
                    else:
                        cimg = rimg_cur if rimg_cur is not None else fpv
                        yes, vok = vlm_yesno(cimg, Q_CLASSIFY); vlm_step = k
                        vlm_inspect = yes; vlm_ans = ("分类:是目标" if yes else "分类:非目标") + ("" if vok else "(VLM挂)")
                    if vlm_inspect or not vok:
                        mgr_decision = f"子任务#{cur_sub['id']} VLM确认是目标 -> 检视" + ("" if vok else "(VLM挂降级)")
                        state = "SCAN"; scan_along0 = np.array([x, y], float); scan_face = face_yaw
                    else:
                        inspected[target] = True
                        memory.append((cur_sub["id"], cur_sub["x"], cur_sub["side"], "跳过(非目标)"))
                        mgr_decision = f"子任务#{cur_sub['id']} VLM判非目标, 跳过"
                        cur_sub = None; state = "TURN_BACK"

        elif state == "SCAN":
            # ⭐DD横扫(strafe真实运动类): 面向设备沿走廊切向横移。走廊切向前视点在ego帧=侧向目标→DD输出横移。
            #   ⚠关键: DD吃【真图】rimg_cur(DD本就在nav_ds_real真图上训练), 不能用仿真fpv——面墙时仿真=灰雾→
            #   DD退化前冲漂进墙(之前2/4的真因)。横扫是平移命令(门开g→1)依赖场景, 故必须给好观测=真图。
            #   朝向保持仍用轻量控制器纠偏(strafe不管朝向漂移); tangent方向投影防垂直漂。
            te = travel_tan - yaw   # 沿【行进方向】切向横移(返程反向)
            goal_ego = np.array([GOAL_AHEAD * np.cos(te), GOAL_AHEAD * np.sin(te)], np.float32)
            dd_img = rimg_cur if rimg_cur is not None else fpv
            traj = dd_traj(model, dd_img, pose, goal_ego, xyz, floor, device, head_ego=(1.0, 0.0))
            vx, vy, _ = sf.traj_to_control(traj)
            # 投影到行进切向: 只保留沿切向分量(防DD垂直漂进墙), 幅度仍由DD给
            spd = float(np.hypot(vx, vy))
            tvx = float(np.cos(yaw) * np.cos(travel_tan) + np.sin(yaw) * np.sin(travel_tan))   # 行进切向在ego
            tvy = float(-np.sin(yaw) * np.cos(travel_tan) + np.cos(yaw) * np.sin(travel_tan))
            proj = vx * tvx + vy * tvy                                    # DD速度在切向的投影(带符号)
            proj = float(np.clip(proj, 0.30, 0.7))                        # 扫描节奏≥0.3(匹配原控制器0.35), DD在此之上调速
            vx, vy = proj * tvx, proj * tvy
            wz = float(np.clip(1.2 * wrap(scan_face - yaw), -0.4, 0.4))   # 稳向: 保持侧对设备
            svx, svy = sf.bev_safe_vel(obs["bev"], vx, vy); vx, vy = svx, svy
            dd_raw = (vx, vy, wz); src = "DD(横扫strafe,真图)"
            along = float(np.hypot(x - scan_along0[0], y - scan_along0[1]))
            done_scan = False
            if args.vlm_realimg and args.vlm_manager:
                # ⭐VLM经理编排退出: 多选 KEEP/FINISH 决定横扫是否结束(几何兜底扫过3.5m)
                exit_vlm = False
                if along > 0.4 and k % args.vlm_every == 0:
                    key, raw, vok = vlm_decide(vlm_frames, Q_MGR_SCAN, OPT_SCAN); vlm_step = k
                    vlm_ans = {"KEEP": "经理:继续扫", "FINISH": "经理:扫完退出",
                               None: "经理:未解析"}.get(key, "?") + ("" if vok else "(VLM挂)")
                    exit_vlm = vok and (key == "FINISH")
                done_scan = exit_vlm or (along > 3.5)
            elif args.vlm_realimg:
                # 方案B(yes/no)退出: VLM判蓝泵是否离开视野中心; 几何兜底(扫过3.5m)
                exit_vlm = False
                if along > 0.4 and k % args.vlm_every == 0:
                    centered, vok = vlm_yesno(rimg_cur, Q_EXIT); vlm_step = k
                    vlm_ans = ("中心有蓝泵" if centered else "蓝泵已离中心->退出") + ("" if vok else "(VLM挂)")
                    exit_vlm = vok and (not centered)
                done_scan = exit_vlm or (along > 3.5)              # VLM退出 或 几何兜底
            else:
                done_scan = along > 3.5   # 几何/geo_vlm: 扫过定长退出(方向无关, 兼容返程)
            if done_scan:
                if target >= 0: inspected[target] = True
                if cur_sub is not None:
                    memory.append((cur_sub["id"], cur_sub["x"], cur_sub["side"], "✓完成"))
                    mgr_decision = f"子任务#{cur_sub['id']} ✓完成 -> 回主任务"; cur_sub = None
                state = "TURN_BACK"

        elif state == "TURN_AROUND":
            # ⭐去程到头: 原地转180°(hybrid, clamp±90循环两次达标)→切返程
            e = wrap(around_target - yaw)
            if abs(e) > TURN_DD_MIN:
                ec = float(np.clip(e, -np.pi / 2, np.pi / 2))
                traj = dd_traj(model, fpv, pose, np.zeros(2, np.float32), xyz, floor, device,
                               head_ego=(np.cos(ec), np.sin(ec)))
                wz = turn_wz(traj); vx = vy = 0.0; dd_raw = (0.0, 0.0, wz); src = "DD(原地转180返程)"
            else:
                wz = float(np.clip(1.5 * e, -0.8, 0.8)); vx = vy = 0.0; src = "控制器(转180收尾)"
            if abs(e) < 0.15:
                direction = -1; state = "CRUISE"; target = -1
                last_sub_xy = np.array([x, y])   # 重置去抖位置
                inspected[:] = False             # ⭐返程重新检视各设备(几何候选按leg重置)
                mgr_decision = "已转180°, 开始返程巡检"

        elif state == "TURN_BACK":
            e = wrap(travel_tan - yaw)   # 转回【行进方向】(返程=反向)
            if abs(e) > TURN_DD_MIN:                                     # 混合: 大角度gate-DD转回
                ec = float(np.clip(e, -np.pi / 2, np.pi / 2))           # 夹到训练范围, 短路径转回
                traj = dd_traj(model, fpv, pose, np.zeros(2, np.float32), xyz, floor, device,
                               head_ego=(np.cos(ec), np.sin(ec)))
                wz = turn_wz(traj); vx = vy = 0.0
                dd_raw = (0.0, 0.0, wz); src = "DD(原地转回)"
            else:                                                        # 小角度控制器收尾
                wz = float(np.clip(1.5 * e, -0.8, 0.8)); vx = vy = 0.0; src = "控制器(转回收尾)"
            if abs(e) < 0.15:
                state = "CRUISE"; target = -1; mgr_decision = "巡逻中(主任务)"

        # ⭐几何兜底(VLM END没触发时): 去程到goal→转180返程; 返程到start→DONE
        if state == "CRUISE":
            if direction > 0:
                dg = float(np.hypot(x - env.goal[0], y - env.goal[1]))
                if dg < 1.5:
                    around_target = wrap(yaw + np.pi)
                    mgr_decision = "到走廊尽头, 原地转180°返程"; state = "TURN_AROUND"
            else:
                ds = float(np.hypot(x - env.start[0], y - env.start[1]))
                if ds < 1.5:
                    mgr_decision = "返程回到起点, 主任务完成"; state = "DONE"

        v_ema = 0.6 * v_ema + 0.4 * np.array([vx, vy, wz], np.float32)
        vx, vy, wz = float(v_ema[0]), float(v_ema[1]), float(v_ema[2])
        vx *= args.speed; vy *= args.speed   # ⭐平移变慢(转向wz不缩, 免转太久)

        # 最近设备簇(慢脑WHERE诊断)
        cd = np.hypot(clusters[:, 0] - x, clusters[:, 1] - y); nci = int(cd.argmin())
        # 前方最近墙距(环境诊断)
        wallf = 9.0
        for dd_ in np.arange(0.1, 3.0, 0.05):
            px, py = x + dd_ * np.cos(yaw), y + dd_ * np.sin(yaw)
            gx2, gy2 = int((px - env.lo[0]) / env.res), int((py - env.lo[1]) / env.res)
            if 0 <= gy2 < env.occ.shape[0] and 0 <= gx2 < env.occ.shape[1] and env.occ[gy2, gx2]:
                wallf = dd_; break
        col = STATE_COL[state]; fv = fpv.copy()
        if traj is not None:
            sf.draw_traj_on_fpv(fv, pose, traj, col)
        Hh = fv.shape[0]
        # ⭐真图inset(左下角): 仅real模式(sim模式VLM就看主画面, 无需inset)
        if rimg_cur is not None and args.vlm_view == "real":
            iw = fv.shape[1] // 4; ih = int(iw * rimg_cur.shape[0] / rimg_cur.shape[1])
            ins = cv2.resize(rimg_cur, (iw, ih))
            y0 = Hh - ih - 8; x0 = 8
            fv[y0:y0 + ih, x0:x0 + iw] = ins
            cv2.rectangle(fv, (x0, y0), (x0 + iw, y0 + ih), (0, 220, 255), 2)
            cv2.rectangle(fv, (x0, y0 - 20), (x0 + iw, y0), (0, 0, 0), -1)
            cv2.putText(fv, f"VLM sees REAL img @x{real_x:.0f}", (x0 + 3, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
        W_ = (255, 255, 255); Y_ = (0, 220, 255); G_ = (0, 255, 120); R_ = (0, 80, 255); B_ = (255, 180, 0)
        prog = 100.0 * min(1.0, x / max(1e-3, env.goal[0]))            # 主任务沿廊进度
        seg_txt = "去程→" if direction > 0 else "←返程"
        n_done = sum(1 for m in memory if "完成" in m[3])
        vlm_src = ("VLM看仿真渲染" if args.vlm_view == "sim" else f"VLM看真图@x{real_x:.0f}") if args.vlm_realimg else "VLM看渲染"
        lines = [
            (f"step {k}   STATE={state}", col),
            (f"pose x={x:.2f} y={y:.2f} yaw={np.degrees(yaw):+.0f}deg  [{vlm_src}]", W_),
            ("== 慢脑 VLM(项目经理) ==", B_),
            (f"  主任务: {MAIN_TASK}", W_),
            (f"  进度: {seg_txt} 沿廊{prog:.0f}%  已检视子任务{n_done}", W_),
            (f"  当前决策: {mgr_decision[:26]}", (G_ if "执行" in mgr_decision or "完成" in mgr_decision or "发现" in mgr_decision else Y_)),
            (f"  VLM@{vlm_step}: {vlm_ans[:22]}",
             (R_ if not vlm_ok else (G_ if ("有" in vlm_ans and "无" not in vlm_ans) else Y_))),
            ("  记忆(已处理子任务):", B_),
        ]
        for (sid, sx, sside, res) in memory[-3:]:                     # 最近3条子任务记忆
            lines.append((f"    #{sid} @x{sx:.0f} {'左' if sside>0 else '右'} {res}",
                          (G_ if "完成" in res else Y_)))
        lines += [
            ("== 几何 WHERE(报坐标) ==", B_),
            (f"  最近设备c{nci}@({clusters[nci,0]:.1f},{clusters[nci,1]:.1f}) d={cd[nci]:.1f}m", W_),
            ("== 快脑DD / 控制器(HOW) ==", B_),
            (f"  src={src}  goal_ego=({goal_ego[0]:+.1f},{goal_ego[1]:+.1f})", Y_),
            (f"  FINAL vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}", G_),
            (f"  bev_safe={'裁剪!' if safe_clip else 'ok'} wall={wallf:.1f}m coll={'YES' if last_coll else 'no'}",
             (R_ if (safe_clip or last_coll or wallf < 0.6) else W_)),
        ]
        pan = log_panel(Hh, lines)
        writer.append_data(np.hstack([fv, pan])[:, :, ::-1])
        jl.write(json.dumps({"step": k, "state": state, "pose": [round(float(p), 2) for p in pose],
                             "target": int(target), "done": int(inspected.sum())}) + "\n"); jl.flush()
        n_done = sum(1 for m in memory if "完成" in m[3])
        if k % 10 == 0 or state == "DONE":
            print(f"  step{k:3d} [{state}] pose=({x:.1f},{y:.1f},{yaw:+.2f}) 已检视={n_done} 子任务发起={sub_n}", flush=True)
        if state == "DONE":
            print(f"[巡检] 完成! step{k} 到终点; 共检视{n_done}处(VLM发起子任务{sub_n})", flush=True)
            obs, _, _ = env.step(vx, vy, wz); break
        obs, term, info = env.step(vx, vy, wz)
        last_coll = bool(info["collided"])
        if last_coll:
            print(f"  step{k} 碰撞! pose=({x:.1f},{y:.1f}) state={state}", flush=True)

    writer.close(); jl.close(); shutil.copy(mp4, canon)
    print(f"[done] {time.time()-t0:.0f}s  检视{sum(1 for m in memory if '完成' in m[3])}处\n  {mp4}\n  {canon}", flush=True)
    if sb_vlm is not None:
        st = sb_vlm.stats()
        print(f"[时序慢脑] 调用{st['n_calls']}次 平均延迟{st['avg_latency']}s 记忆={st['memory']}", flush=True)


if __name__ == "__main__":
    main()
