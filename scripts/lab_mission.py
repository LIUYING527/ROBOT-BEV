"""室内110744指定路径任务 —— ⭐场景图RAG原生双脑(0730掰回):
  WHERE=3D场景图节点坐标(RAG检索, GT-free)直接出站位+朝向: 位置=on-path采集帧(渲染锐),
  朝向=atan2(节点-站位)纯几何+节点点簇投影精对准(graph_fine_aim), roll=auto按位置找平 —— 全程无bbox无伺服;
  慢脑VLM(robo2vlm): 到位后对实时水平渲染图做判断式yes/no确认(判no如实报, 不挽尊);
  快脑 DiffusionDrive(gate model): 生成运动(HOW=dd_traj→traj_to_control出vx,vy,wz);
  A* 仅给快脑喂粗前视目标点。
流程: 原位→铁柜(几何对准+VLM判断确认+等待)→显微镜(等待)→移液枪白桌(等待)→回原位。
用法: CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      ~/discoverse_venv/bin/python scripts/lab_mission.py [--dry] [--speed 0.5] [--targets outputs/lab_mission_targets_rag.json]
"""
import os, sys, json, argparse, time as _time, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
RUNS = os.path.join(ROOT, "goalnav", "runs")


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi


def free(env, x, y):
    return not env._occupied(x, y)


def find_standoff(env, tx, ty, wp, yaw, stand=1.2, verbose=True, avoid=()):
    """⭐RAG原生站位(0730掰回): 节点3D坐标直接出站位+朝向, 全程无bbox。
    位置=采集轨迹上"最正对该目标"的帧位置(on-path→渲染锐, 采集yaw只用于打分选位置);
    朝向=atan2(节点-站位)纯几何解析(不继承采集帧姿态——绕看采集带roll/pitch, 继承=画面歪)。
    距离约束[0.7,2.0]m偏好stand=1.2m: <0.7贴脸全帧糊(0729显微镜站0.11m的教训), >2.0目标太小。
    avoid=已选站位列表, 0.6m内排除(防两站选到同一点=横移腿退化)。"""
    best = None
    for i in range(len(wp)):
        sx, sy = float(wp[i, 0]), float(wp[i, 1])
        d = np.hypot(sx - tx, sy - ty)
        if d < 0.7 or d > 2.0:
            continue
        if any(np.hypot(sx - ax, sy - ay) < 0.6 for ax, ay in avoid):
            continue
        if not free(env, sx, sy):
            continue
        bearing = np.arctan2(ty - sy, tx - sx)
        berr = abs(wrap(bearing - yaw[i]))          # 采集帧yaw是否对准目标(选渲染最正的位置)
        score = berr + 0.25 * abs(d - stand)
        if best is None or score < best[0]:
            best = (score, sx, sy, bearing, d, np.degrees(berr))
    if best is None:                                 # 轨迹附近无满足约束的帧: 沿最近采集点方向外推到stand米
        d = np.linalg.norm(wp - [tx, ty], axis=1)
        i = int(d.argmin())
        dirv = np.array([wp[i, 0] - tx, wp[i, 1] - ty]); dirv /= (np.linalg.norm(dirv) + 1e-9)
        sx, sy = tx + dirv[0] * stand, ty + dirv[1] * stand
        return float(sx), float(sy), float(np.arctan2(ty - sy, tx - sx))
    _, sx, sy, fy, d, berrd = best
    if verbose:
        print(f"  standoff ({tx:.2f},{ty:.2f}) -> 位置({sx:.2f},{sy:.2f}) dist={d:.2f}m "
              f"yaw={np.degrees(fy):.0f}°(几何解析, 采集帧对准残差{berrd:.0f}°)", flush=True)
    return float(sx), float(sy), float(fy)


def main():
    import sim_fullchain as sf
    from test_env import CorridorTestEnv
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--dry", action="store_true", help="只算站位+规划路径, 不渲染视频")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--speed", type=float, default=0.5, help="平移速度缩放(放慢), 快脑输出乘此系数")
    ap.add_argument("--targets", default=None, help="目标坐标json(如RAG三角化版lab_mission_targets_rag_tri.json)")
    ap.add_argument("--fine_aim", action="store_true", help="节点簇投影精对准(⚠簇过合并时会帮倒忙, 三角化targets下不用开)")
    # ⭐观察策略参数化(0802, 需v4快脑): 时序历史BEV的窗口W/衰减γ按相位取——巡航短窗省算力, 对准/转向长窗记全周围
    ap.add_argument("--hist_w_cruise", type=int, default=2, help="巡航历史窗口(0=单帧=旧行为)")
    ap.add_argument("--hist_w_align", type=int, default=8, help="对准/转向历史窗口(原地扫视时记全周围)")
    ap.add_argument("--hist_gamma_cruise", type=float, default=0.7, help="巡航衰减γ")
    ap.add_argument("--hist_gamma_align", type=float, default=0.85, help="对准/转向衰减γ")
    # ⭐0803论文项落地(周报⑥清单): ⑧慢脑→快脑结构化参数包(下见NavCmd) + ⑩四段推理链confirm
    # (History/Scene/Progress/Action段段落盘可审计) + ⑪判no时key-frame复核(论文"按索引回原图";
    # 用接近过程已采帧复核证据, 机器人不动≠扫描挽尊)
    ap.add_argument("--no_review", action="store_true", help="⑪关掉判no时key-frame复核(退回0802一锤子行为)")
    ap.add_argument("--sb_log", default=None, help="慢脑推理链jsonl(默认outputs/lab_mission_slowbrain_0803.jsonl)")
    # ⭐⑨搜索行为demo(论文§4.1.3): 无目标纯快脑探索(mode=1进status[7], 需v5快脑), 死胡同应自己回退
    ap.add_argument("--search_demo", type=int, default=0, metavar="NSTEPS", help=">0则跑搜索demo(跳过任务腿)")
    # ⭐RELOC精定位闭环(0811, 高精度导航): 到站注入里程计漂移→GS render-and-compare重定位纠正→
    # believed位姿几何对准; 终误差用真位姿测量。对照=不纠正时误差≈漂移本身。
    ap.add_argument("--reloc_demo", action="store_true", help="到站RELOC精定位闭环demo")
    ap.add_argument("--drift_m", type=float, default=0.25, help="注入漂移上限(米), yaw漂移=drift_m*40度")
    # ⭐0811续(吸收HumanCLAW"持续自我跟踪"): 巡航途中里程计随机游走漂移+周期RELOC纠偏——
    # 控制目标用believed位姿(=真实机器人的漂移里程计), FPV/BEV用真位姿(=自车传感不漂),
    # 每~2m一次RELOC把believed拉回。全程记录"无纠正漂移vs有纠正残差"时间序列。
    ap.add_argument("--reloc_odom", action="store_true", help="巡航周期RELOC纠偏(需--reloc_demo)")
    # ⭐0813回归头RELOC(0805调研P1落地): reloc_head.pt单次前向(<10ms, Orin可跑)替代GS梯度优化(~43s/次)。
    # 安全门保留: 纠正后单帧渲染验证loss>0.08仍拒收; 纠正幅度>1m/40°直接拒收(头无loss信号, 双闸门)。
    ap.add_argument("--raw", default="data/lab/110744", help="GSReloc原始数据目录(zed npz)")
    ap.add_argument("--reloc_head", nargs="?", const=os.path.join(ROOT, "outputs", "reloc_head_0812", "reloc_head.pt"),
                    default=None, help="用回归头做RELOC(可选指定ckpt路径), 默认=GS梯度优化")
    # ⭐0811用户判"视角倾斜+没正对": -22°俯角的keystone效应让竖直线全歪~10°=观感倾斜。
    # 分相位俯角: 巡航保-22°(快脑训练分布安全), 到站对准/确认/等待用-12°(正脸观感+目标区干净)。
    ap.add_argument("--pitch_cruise", type=float, default=-22.0)
    ap.add_argument("--pitch_station", type=float, default=-12.0)
    # ⭐0811用户贴脸对接要求: 正对缝隙横向<1cm + 距面15cm(三目标同)。全程传感伺服(深度+线检测), 不读地图坐标——
    # 与真机完全同构(zed中央ROI深度 + FPV线检测), sim里渲染ED即"相机真值"。
    ap.add_argument("--precise_dock", action="store_true", help="confirm后贴脸精对接(距面0.15m+缝隙/质心居中<1cm)")
    ap.add_argument("--flow", action="store_true", help="⭐0812全流程面板版: 右侧渲染完整实际流程(8段管线实时值+任务清单+位姿), 关键决策点加停留帧, 片头片尾卡片; 用于逐帧审查流程问题")
    args = ap.parse_args()

    # ===== ⭐⑧ 慢脑→快脑结构化参数包(论文nav(L,τ,Φ), 周报清单⑧⭐⭐⭐P0) =====
    # 以前: goal点+vx+W/γ散落在cruise/face调用点硬编码 = "信息吸管";
    # 现在: 每腿决策打成一个NavCmd{L=模式, τ=(名字,坐标), 空间提示(方位/距离), Φ=(W,γ,vx)},
    # 模式→观察配置的映射集中在PHI(论文核心主张: 行为差异=观察策略差异), 逐段落盘可审计。
    PHI = {"CRUISE": (args.hist_w_cruise, args.hist_gamma_cruise, 0.5),
           "STRAFE": (args.hist_w_cruise, args.hist_gamma_cruise, 0.4),
           "ALIGN":  (args.hist_w_align,  args.hist_gamma_align,  0.0)}   # 转向/对准=长窗+vx=0(对齐训练)

    class NavCmd:
        """慢脑派发给快脑的一腿指令包(对应论文 nav(L,τ,Φ) 调用签名)。"""
        def __init__(self, L, tau_name, tau_xy, pos, phi):
            self.L = L                                   # 模式: CRUISE/STRAFE
            self.tau_name = tau_name                     # τ身份(语言目标)
            self.tau_xy = tuple(np.round(tau_xy, 2))     # τ位置(世界系)
            brg = np.degrees(np.arctan2(tau_xy[1] - pos[1], tau_xy[0] - pos[0]))
            self.bearing_deg = round(float(brg), 1)      # 空间提示: 方位
            self.dist_m = round(float(np.hypot(tau_xy[0] - pos[0], tau_xy[1] - pos[1])), 2)  # 空间提示: 距离
            self.W, self.gamma, self.vx = phi            # Φ观察配置+速度先验
        def as_dict(self):
            return {"type": "nav_cmd", "L": self.L, "tau": self.tau_name, "tau_xy": list(self.tau_xy),
                    "bearing_deg": self.bearing_deg, "dist_m": self.dist_m,
                    "phi": {"W": self.W, "gamma": self.gamma, "vx": self.vx}}
        def __str__(self):
            return (f"nav(L={self.L}, τ={self.tau_name}@{self.tau_xy}, "
                    f"brg={self.bearing_deg}° d={self.dist_m}m, Φ=(W{self.W},γ{self.gamma},vx{self.vx}))")

    _sb_log_path = args.sb_log or os.path.join(ROOT, "outputs", "lab_mission_slowbrain_0803.jsonl")
    def sb_record(rec):
        """慢脑决策落盘(审计链: ⑧参数包 + ⑩四段推理 + ⑪复核证据)。"""
        with open(_sb_log_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tj = args.targets or os.path.join(ROOT, "outputs", "lab_mission_targets_zed.json")   # ⭐优先ZED真值深度定位(距离准, 目标不挤)
    if not os.path.exists(tj):
        tj = os.path.join(ROOT, "outputs", "lab_mission_targets.json")
    T = json.load(open(tj))
    print(f"[mission] targets from {os.path.basename(tj)}:", {k: (v[:2] if isinstance(v, list) else v) for k, v in T.items()}, flush=True)

    env = CorridorTestEnv(skin_session=args.session)
    print(f"[mission] occ {env.occ.shape} free{100*(env.occ==0).mean():.0f}% start={env.start[:2]}", flush=True)

    # 站位: 贴采集轨迹(render锐)
    _cap = np.load(os.path.join(ROOT, "outputs", f"sim_world_{args.session}.npz"))
    wp_cap, yaw_cap = _cap["waypoints_xy"], _cap["yaw"]
    # ⭐站位=find_standoff(RAG原生: 节点坐标→位置选on-path采集帧, 朝向纯几何解析, 详见函数docstring)
    # 已废弃: ①--standoffs场景图最佳观测采集位姿json(继承采集roll/pitch=画面歪+显微镜站0.11m贴脸);
    #         ②standoff_at墙法线版(手搓几何, 被场景图替换); ③SRC源帧confirm(改为实时水平渲染图判断式)。
    # ⭐站位: targets带front_deg(三角化脚本从节点最大检测帧算的正面方向)时,
    # 站位=目标沿正面方向反推stand米(视线垂直目标正脸=用户要的"正对", 目标自动居中),
    # 沿轴滑动找free; 不带则退回find_standoff(采集帧选位置)。
    FRONT = T.get("front_deg", {})
    def standoff_front(name, stand=0.85):
        """⭐0811改: 正面轴上选【离采集轨迹最近】的free点(而非固定0.85m)——
        on-path=渲染锐(0730 doctrine), 白桌固定0.85m站进off-path溅射区整屏糊(用户0811判'效果不好')。
        r候选[0.7,1.6]m步进0.05, 打分=到采集轨迹最近距离 + 0.3×|r-stand|(偏好站距)。"""
        th = np.radians(FRONT[name]); d = np.array([np.cos(th), np.sin(th)])
        tx, ty = float(T[name][0]), float(T[name][1])
        best = None
        for r in np.arange(0.7, 1.65, 0.05):
            sx, sy = tx - d[0] * r, ty - d[1] * r
            if not free(env, sx, sy):
                continue
            dpath = float(np.min(np.hypot(wp_cap[:, 0] - sx, wp_cap[:, 1] - sy)))
            score = dpath + 0.3 * abs(r - stand)
            if best is None or score < best[0]:
                best = (score, float(sx), float(sy), r, dpath)
        if best is not None:
            print(f"  standoff_front {name}: 轴上r={best[3]:.2f}m 离采集轨迹{best[4]:.2f}m", flush=True)
            return best[1], best[2], float(th)
        return standoff_front_fallback(name)
    def standoff_front_fallback(name):
        return find_standoff(env, float(T[name][0]), float(T[name][1]), wp_cap, yaw_cap)
    so = {}
    _STATIONS = [k for k in T if k not in ("origin", "front_deg", "front_method")]  # ⭐0813晚: 站名数据驱动(走廊=4泵站)
    for name in _STATIONS:
        if name in FRONT:
            so[name] = standoff_front(name)
        else:
            so[name] = find_standoff(env, float(T[name][0]), float(T[name][1]), wp_cap, yaw_cap,
                                     avoid=[v[:2] for v in so.values()])
        d0 = np.hypot(so[name][0] - T[name][0], so[name][1] - T[name][1])
        print(f"  停靠点 {name}: 节点({T[name][0]:.2f},{T[name][1]:.2f}) -> 站位({so[name][0]:.2f},{so[name][1]:.2f}) 距目标{d0:.2f}m yaw{np.degrees(so[name][2]):.0f}°", flush=True)
    print("[mission] standoffs:", {k: (round(v[0], 2), round(v[1], 2)) for k, v in so.items()}, flush=True)

    def plan(sx, sy, gx, gy):
        env.start = (sx, sy, 0.0); env.goal = (gx, gy)
        return sf.plan_path(env)

    # 任务腿: origin→铁柜#490 ; 横移strafe到显微镜#509 ; A*横穿到移液枪桌#215(1.0,0.13) ; 回原位
    _origin = tuple(T.get("origin", [0.0, 0.0])[:2])
    legs = []
    if set(_STATIONS) == {"iron_cabinet", "microscope", "pipette_table"}:
        legs.append(("A*", _origin, so["iron_cabinet"][:2], "goto_cabinet"))
        legs.append(("STRAFE", so["iron_cabinet"][:2], so["microscope"][:2], "strafe_microscope"))
        legs.append(("A*", so["microscope"][:2], so["pipette_table"][:2], "goto_pipette"))
        legs.append(("A*", so["pipette_table"][:2], _origin, "return_origin"))
    else:                                            # ⭐0813晚通用链(走廊4泵站等): 起点→各站A*(json顺序)→回起点
        _prev = _origin
        for _nm in _STATIONS:
            legs.append(("A*", _prev, so[_nm][:2], f"goto_{_nm}")); _prev = so[_nm][:2]
        legs.append(("A*", _prev, _origin, "return_origin"))
    for mode, a, b, tag in legs:
        if mode == "A*":
            p = plan(a[0], a[1], b[0], b[1])
            ok = p is not None and len(p) > 1
            print(f"  leg {tag}: A* {tuple(np.round(a,2))}->{tuple(np.round(b,2))} : {'OK '+str(len(p))+'pts' if ok else 'FAIL'}", flush=True)
        else:
            print(f"  leg {tag}: STRAFE {tuple(np.round(a,2))}->{tuple(np.round(b,2))} dist={np.hypot(b[0]-a[0],b[1]-a[1]):.2f}m", flush=True)

    if args.dry:
        print("[mission] dry run done (no render)", flush=True)
        return

    # ---------------- 双脑执行 + 渲染 ----------------
    import cv2, imageio, torch, glob
    import stage_b_eval as sb
    from train_goalnav import GOAL_AHEAD, robot_kmeans_anchors
    from train_goalnav_turn import status_gh
    device = "cuda"
    SPEED = args.speed                                 # ⭐平移速度缩放(放慢)

    # ===== 快脑: DiffusionDrive gate model =====
    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    ck_name = os.environ.get("CKPT", "goalnav_gate_lab.pt")   # ⭐默认用实验室微调版快脑(避墙+转正); 走廊版=goalnav_gate.pt
    ck = torch.load(os.path.join(RUNS, ck_name))
    model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
    model._status_encoding.load_state_dict(ck["status_encoding"]); model.eval()
    print(f"[mission] 快脑 ckpt = {ck_name}", flush=True)
    ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    xyz = sb.read_ply_xyz(ply); _, _, _, floor = sb.build_occ(xyz)
    print("[mission] 快脑 goalnav_gate 已加载 (DiffusionDrive)", flush=True)

    # ===== ⭐间隙场安全层(治窄通道撞墙抽搐): 距最近障碍的距离(米), 双线性查询 =====
    import cv2 as _cv2
    _freem = (env.occ == 0).astype(np.uint8)
    _clear = _cv2.distanceTransform(_freem, _cv2.DIST_L2, 5).astype(np.float32) * env.res  # 每格到障碍距离(米)
    def clearance(x, y):
        ix = (x - env.lo[0]) / env.res; iy = (y - env.lo[1]) / env.res
        i0, j0 = int(np.floor(iy)), int(np.floor(ix))
        if i0 < 0 or j0 < 0 or i0 >= _clear.shape[0] - 1 or j0 >= _clear.shape[1] - 1:
            return 0.0
        return float(_clear[i0, j0])
    ROBOT_R = 0.30
    def safe_vel(x, y, yaw, vx, vy, wz, dt=0.25):
        """预测下一步位姿, 离障碍太近→减速+朝更空方向偏转(势场避障), 从源头不贴墙/不抽搐。"""
        c, s = np.cos(yaw), np.sin(yaw)
        nx = x + (vx * c - vy * s) * dt; ny = y + (vx * s + vy * c) * dt
        cl = clearance(nx, ny)
        need = ROBOT_R + 0.25
        if cl >= need:
            return vx, vy, wz, False
        # 太近: 采8方向找间隙最大的可行方向, 朝它偏转+大幅减速
        best_th, best_cl = None, -1
        for a in np.linspace(-np.pi, np.pi, 16, endpoint=False):
            px = x + 0.4 * np.cos(a); py = y + 0.4 * np.sin(a)
            clc = clearance(px, py)
            if clc > best_cl:
                best_cl, best_th = clc, a
        # 偏转朝best方向(世界)→机体wz
        e = wrap(best_th - yaw)
        vx2 = float(np.clip(vx, 0, 0.25)) * (cl / need)     # 严重减速
        wz2 = float(np.clip(1.2 * e, -0.8, 0.8))
        return vx2, 0.0, wz2, True

    # ===== ⭐观察策略参数化(0802): 时序历史位姿环形缓冲(世界系, 新→旧) =====
    # 入队条件=位移>0.4m(对齐训练历史步长)或转向>0.3rad(原地扫视时记全周围=长窗收益处); 留最近8个。
    _hist_buf = []
    def hist_update():
        p = (env.x, env.y, env.yaw)
        if _hist_buf:
            dx = p[0] - _hist_buf[0][0]; dy = p[1] - _hist_buf[0][1]
            dth = abs(wrap(p[2] - _hist_buf[0][2]))
            if dx * dx + dy * dy < 0.16 and dth < 0.3:
                return
        _hist_buf.insert(0, p)
        del _hist_buf[8:]

    def dd_traj(fpv_bgr, pose, goal_ego, head_ego=(1.0, 0.0), hist_w=0, hist_gamma=0.85, vx=0.4, mode=0.0):
        st = status_gh(np.asarray(goal_ego, np.float32), np.asarray(head_ego, np.float32), bs_vx=vx, device=device)
        st[0, 5] = hist_w / 8.0; st[0, 6] = hist_gamma          # ⭐观察策略条件(⚠v4快脑专用: 旧ckpt的status[5:8]输入权重未训,
        #                                                          配旧ckpt必须--hist_w_cruise 0 --hist_w_align 0才是旧行为)
        st[0, 7] = mode                                          # ⭐⑨mode: 0=goal-seek 1=search(⚠v5快脑专用; v4及以前该列没训过, 必须保持0)
        cam = sb.camera_feature(fpv_bgr, device)
        if hist_w > 0 and _hist_buf:
            bev = sb.lidar_feature_fov_hist(xyz, pose, floor, device, np.array(_hist_buf),
                                            W=hist_w, gamma=hist_gamma, seed=0)
        else:
            bev = sb.lidar_feature_fov(xyz, pose, floor, device)
        log["_hist"] = f"W{hist_w}γ{hist_gamma}" if hist_w > 0 else "W0(单帧)"
        torch.manual_seed(0)
        with torch.no_grad():
            return model({"camera_feature": cam, "lidar_feature": bev, "status_feature": st})["trajectory"][0].cpu().numpy()

    def turn_wz(traj):
        return float(np.clip(1.5 * float(traj[-1, 2]), -0.8, 0.8))

    from capture_fpv import CaptureFPV
    capfpv = CaptureFPV(args.session)
    # ⭐镜头修正(治"没正对"垂直下压+"左右晃"): 采集pitch-26.6°俯视过头→目标挤画面上方(v128 vs中心200);
    #   减到-14°→目标垂直居中(实测仍无白雾, 站位贴采集轨迹); roll固定0→不随位置变(auto-roll±8°=地平线来回摇=左右晃)。
    capfpv.pitch = np.radians(args.pitch_cruise)   # ⭐巡航俯角(快脑训练分布); 到站切pitch_station消keystone(0811)
    # ⭐roll: 固定0平着走(0730用户明示"视角为啥要动呢,就平着呗"——auto-roll逐位置查表=左右摇,已否)。
    #   ⚠代价: 场景自身静态倾斜±4°会在画面里(世界没untwist), 要治得扶平几何本体, 不靠相机补。
    def sim_fpv():
        return capfpv._render_V(capfpv._world_to_colmap_cam(env.x, env.y, env.yaw, roll_deg=0.0))   # 平相机

    # ===== ⭐RELOC精定位闭环(0811): 里程计漂移注入 → GS render-and-compare重定位 → believed位姿纠偏 =====
    # 真实场景复刻: 巡航靠轮速里程计会漂(本项目无odom话题, 指令积分+IMU), 到站时believed位姿带0.15~0.3m误差;
    # 站内用"深度观测 vs GS地图渲染ED"梯度优化出真位姿(render-and-compare, 与gs_relocate.py同管线),
    # 纠正后believed残差~8-13cm(真zed实测噪声底, 见outputs/gs_relocate_110744s2/eval_scaled_0811.log)。
    # sim内观测=真位姿渲染ED÷OBS_SCALE→"真实米"+zed式噪声(σ=1cm+1%d, 5%丢点, 0.3~12m窗)——
    # 非循环论证的兜底证据=真zed数据评测表(同一管线同一地图); 此处验证的是闭环机制+终 docking 精度。
    _reloc = None
    _drift = np.zeros(3)                                  # believed = true + drift (dx,dy,dyaw)
    _reloc_stats = []
    if args.reloc_demo:
        from gs_relocate import GSReloc
        _reloc = GSReloc(args.session, args.raw)
        print("[mission] RELOC模块已加载 (GS render-and-compare, obs_scale=%.2f)" % _reloc.obs_scale, flush=True)
    _reloc_head_net = None
    if args.reloc_demo and args.reloc_head:
        from reloc_head_eval import build_head
        _reloc_head_net = build_head("cuda", ckpt=args.reloc_head)
        print(f"[mission] RELOC回归头已加载: {args.reloc_head} (单次前向替代43s优化)", flush=True)

    def bel_pose():
        """believed位姿(机器人以为自己在哪)。"""
        return float(env.x + _drift[0]), float(env.y + _drift[1]), float(wrap(env.yaw + _drift[2]))

    def _cam_CR(x, y, yaw):
        """与capfpv._world_to_colmap_cam同一相机构造(roll=0, pitch=capfpv.pitch), 出(C_w, R_c2w)。"""
        C_w = np.array([x, y, capfpv.cam_height], np.float64)
        c, s = np.cos(yaw), np.sin(yaw); cp, sp = np.cos(capfpv.pitch), np.sin(capfpv.pitch)
        fwd = np.array([c * cp, s * cp, sp]); right = np.array([s, -c, 0.0])
        down = np.cross(fwd, right); down /= np.linalg.norm(down)
        return C_w, np.stack([right, down, fwd], axis=1)

    def reloc_correct(name, emit_n=8, smooth=False):
        """RELOC纠偏: 伪zed观测(真位姿) → 从believed位姿优化 → 纠正_drift。返回(漂移量, 纠正后残差)。
        emit_n=优化前后各写多少占位帧(巡航途中纠偏传0, 不臃肿视频)。
        smooth=True(途中纠偏): 纠正量写进_odom['slew'], 巡航循环按每步≤2cm/0.3°渐进施加——
        防believed瞬跳→前视点跳→转向猛踢(0812用户: 运动中突然顿一下)。"""
        import torch as _t
        import zlib
        rng = np.random.RandomState(zlib.crc32(str(name).encode()) & 0xffff)
        obs = {}
        with _t.no_grad():
            C_t, R_t = _cam_CR(env.x, env.y, env.yaw)
            _res = ((320, 180), (640, 360)) if _reloc_head_net is None else ((160, 90), (320, 180))
            for (Wo, Ho) in _res:
                V_t = _reloc.viewmat_from_world(_t.tensor(C_t, device="cuda", dtype=_t.float32),
                                                _t.tensor(R_t, device="cuda", dtype=_t.float32))
                ed, al = _reloc.render_ed(V_t, Ho, Wo)
                d_real = (ed / _reloc.obs_scale).cpu().numpy()          # 世界米→真实米(独立传感器口径)
                sig = 0.01 + 0.01 * d_real                               # zed式噪声: 1cm + 1%测距
                d_n = d_real + rng.normal(0, sig).astype(np.float32)
                m = (al.cpu().numpy() > 0.6) & (d_real > 0.3) & (d_real < 12.0) & (rng.rand(Ho, Wo) > 0.05)
                obs[(Wo, Ho)] = (_t.tensor(np.where(m, d_n * _reloc.obs_scale, 0.0).astype(np.float32), device="cuda"),
                                 _t.tensor(m, device="cuda"))
        bx, by, byaw = bel_pose()
        C_b, R_b = _cam_CR(bx, by, byaw)
        drift0 = (float(np.hypot(_drift[0], _drift[1])), abs(float(np.degrees(_drift[2]))))
        if _reloc_head_net is not None:
            # ===== 回归头模式(0813): 单次前向(<10ms) × 2次迭代(同obs重复查询合法——观测不依赖believed) =====
            from reloc_head_eval import head_correct
            log["state"] = "RELOC head"
            log["brain"] = f"RELOC-head drift={drift0[0]:.2f}m/{drift0[1]:.0f}° → forward..."
            pipe["cur"] = "3"
            t_h0 = _time.time()
            d90, m90 = obs[(160, 90)]
            bel1 = head_correct(_reloc_head_net, d90.cpu().numpy(), m90.cpu().numpy(),
                                [bx, by, byaw], n_iter=2)
            _ms = (_time.time() - t_h0) * 1000
            step = (float(np.hypot(bel1[0] - bx, bel1[1] - by)), abs(float(np.degrees(wrap(bel1[2] - byaw)))))
            pipe["3"] = f"believed drift {drift0[0]:.2f}m/{drift0[1]:.0f}d -> HEAD fwd {_ms:.0f}ms..."
            for _ in range(emit_n):
                emit(env._obs())
            if step[0] > 1.0 or step[1] > 40.0:                      # 闸门1: 纠正幅度离谱=头失效, 拒收
                print(f"    [RELOC-head] {name}: 纠正量{step[0]:.2f}m/{step[1]:.0f}°超界, 拒收→GS兜底", flush=True)
                log["brain"] = f"RELOC-head REJECT step {step[0]:.2f}m/{step[1]:.0f}d"
                pipe["3"] = f"HEAD REJECT step{step[0]:.2f}m/{step[1]:.0f}d>limit -> GS fallback..."
                vloss = None
            else:
                # 闸门2: 纠正位姿单帧渲染验证(同一Charbonnier/trim口径, 1次渲染~0.1s)
                with _t.no_grad():
                    C_v, R_v = _cam_CR(bel1[0], bel1[1], bel1[2])
                    V_v = _reloc.viewmat_from_world(_t.tensor(C_v, device="cuda", dtype=_t.float32),
                                                    _t.tensor(R_v, device="cuda", dtype=_t.float32))
                    ed_v, al_v = _reloc.render_ed(V_v, 180, 320)
                    d_o, m_o = obs[(320, 180)]
                    mm = m_o & (al_v > 0.6) & (ed_v > 0.05) & (ed_v < 15.0)
                    if int(mm.sum()) < 200:
                        vloss = 9.9                                   # 视野验证不了=不可信, 走拒收
                    else:
                        r = (ed_v - d_o)[mm]
                        thr = _t.quantile(r.abs(), 0.9)
                        vloss = float(_t.sqrt(r[r.abs() <= thr] ** 2 + 1e-4).mean())
            if vloss is not None and vloss <= 0.08:                   # 头通过双闸门: 直接采纳
                C1, R1 = np.array([bel1[0], bel1[1], capfpv.cam_height]), _cam_CR(*bel1)[1]
                trace = [vloss, vloss]
                print(f"    [RELOC-head] {name}: 前向{_ms:.0f}ms, 验证loss={vloss:.3f} ✓", flush=True)
            else:                                                     # 头被拒 → GS梯度优化兜底(快+稳兼得)
                if vloss is not None:
                    print(f"    [RELOC-head] {name}: 验证loss={vloss:.3f}>0.08, 拒收→GS兜底", flush=True)
                log["state"] = "RELOC optimizing"
                log["brain"] = f"RELOC drift={drift0[0]:.2f}m/{drift0[1]:.0f}° → GS fallback..."
                pipe["3"] = f"HEAD rejected -> GS render-compare optimizing..."
                for _ in range(emit_n):
                    emit(env._obs())
                if (640, 360) not in obs:                             # head模式没渲640x360, 补
                    with _t.no_grad():
                        V_t = _reloc.viewmat_from_world(_t.tensor(C_t, device="cuda", dtype=_t.float32),
                                                        _t.tensor(R_t, device="cuda", dtype=_t.float32))
                        ed, al = _reloc.render_ed(V_t, 360, 640)
                        d_real = (ed / _reloc.obs_scale).cpu().numpy()
                        sig = 0.01 + 0.01 * d_real
                        d_n = d_real + rng.normal(0, sig).astype(np.float32)
                        m = (al.cpu().numpy() > 0.6) & (d_real > 0.3) & (d_real < 12.0) & (rng.rand(360, 640) > 0.05)
                        obs[(640, 360)] = (_t.tensor(np.where(m, d_n * _reloc.obs_scale, 0.0).astype(np.float32), device="cuda"),
                                           _t.tensor(m, device="cuda"))
                C1, R1, trace = _reloc.optimize(0, C_b, R_b, obs=obs)
        else:
            log["state"] = "RELOC optimizing"
            log["brain"] = f"RELOC drift={drift0[0]:.2f}m/{drift0[1]:.0f}° → optimizing..."
            pipe["cur"] = "3"
            pipe["3"] = f"believed drift {drift0[0]:.2f}m/{drift0[1]:.0f}d -> GS render-compare optimizing..."
            for _ in range(emit_n):
                emit(env._obs())                                         # 优化期间占位帧(视频可读)
            C1, R1, trace = _reloc.optimize(0, C_b, R_b, obs=obs)
        if C1 is None or not trace:                             # ⭐观测不足(空渲染/贴墙/离分布视角)→跳过本次纠偏不崩(0813走廊巡航#2腿教训)
            print(f"    [RELOC] {name}: 观测不足, 跳过纠偏(保持believed, 漂移{drift0[0]:.2f}m)", flush=True)
            log["brain"] = "RELOC skip: insufficient obs"
            pipe["3"] = f"RELOC skip (insufficient obs), keep believed drift {drift0[0]:.2f}m"
            return drift0, drift0
        if trace and trace[-1] > 0.08:                              # 发散保护收紧: 0.12→0.08(0811: 0.095半收敛被放行的教训)
            print(f"    [RELOC] {name}: 终loss={trace[-1]:.3f}>0.08, 拒收纠正, 保持原believed", flush=True)
            log["brain"] = f"RELOC REJECT loss={trace[-1]:.2f}"
            pipe["3"] = f"REJECT loss{trace[0]:.2f}->{trace[-1]:.2f}>0.08, keep believed (drift {drift0[0]:.2f}m)"
            res = drift0
            for _ in range(emit_n):
                emit(env._obs())
            return drift0, res
        yaw1 = float(np.arctan2(R1[1, 2], R1[0, 2]))
        _newd = np.array([C1[0] - env.x, C1[1] - env.y, wrap(yaw1 - env.yaw)])
        if smooth:
            _odom["slew"] = _newd                                # 渐进施加(cruise循环逐步靠近)
        else:
            _drift[:] = _newd                                    # believed ← 优化位姿(站内静止, 直接纠)
        res = (float(np.hypot(_newd[0], _newd[1])), abs(float(np.degrees(_newd[2]))))
        print(f"    [RELOC] {name}: 漂移{drift0[0]:.2f}m/{drift0[1]:.0f}° → 纠正后残差{res[0]*100:.1f}cm/{res[1]:.1f}° "
              f"loss {trace[0]:.3f}→{trace[-1]:.3f}", flush=True)
        log["brain"] = f"RELOC {drift0[0]:.2f}m→{res[0]*100:.0f}cm residual"
        pipe["3"] = f"drift {drift0[0]:.2f}m/{drift0[1]:.0f}d -> residual {res[0]*100:.1f}cm/{res[1]:.1f}d loss{trace[0]:.2f}->{trace[-1]:.2f} ACCEPT"
        for _ in range(emit_n):
            emit(env._obs())
        return drift0, res

    # ===== ⭐巡航途中周期纠偏(0811续): 里程计随机游走 + 每~2m RELOC =====
    _odom = {"on": bool(args.reloc_demo and args.reloc_odom), "rng": np.random.RandomState(7), "series": []}
    def odom_random_walk():
        """每步believed漂移随机游走(累计上限0.35m/12°), 返回当前无纠正漂移量。"""
        r = _odom["rng"]
        _drift[0] += r.normal(0, 0.02); _drift[1] += r.normal(0, 0.02)
        _drift[2] = wrap(_drift[2] + np.radians(r.normal(0, 0.3)))
        n = np.hypot(_drift[0], _drift[1])
        if n > 0.35:
            _drift[0] *= 0.35 / n; _drift[1] *= 0.35 / n
        _drift[2] = float(np.clip(_drift[2], -np.radians(12), np.radians(12)))
        return float(n)
    def odom_checkpoint(leg_tag, step_i):
        """周期RELOC: 记录(无纠正漂移, 纠正后残差)时间序列供报告曲线。
        emit_n=0=途中纠偏零占位帧(治视频中每~1.7s节奏性顿挫), smooth=纠正量渐进施加。"""
        d0, res = reloc_correct(f"{leg_tag}#{step_i}", emit_n=0, smooth=True)
        _odom["series"].append({"leg": leg_tag, "step": step_i,
                                "drift_m": round(d0[0], 3), "res_cm": round(res[0] * 100, 1)})

    def station_metric(name, drift0, res):
        """对准完成后用【真位姿】测终docking误差: 横向偏轴(米)+方位角差(度)+距目标(米)。"""
        tx, ty = float(T[name][0]), float(T[name][1])
        th = np.radians(FRONT[name]); dirv = np.array([np.cos(th), np.sin(th)])
        pos = np.array([env.x, env.y])
        s = float((pos - [tx, ty]) @ dirv)
        lat = float(np.linalg.norm(pos - (np.array([tx, ty]) + s * dirv)))
        berr = abs(float(np.degrees(wrap(np.arctan2(ty - env.y, tx - env.x) - th))))
        ferr = abs(float(np.degrees(wrap(th - env.yaw))))
        rec = {"station": name, "drift_m": round(drift0[0], 3), "drift_deg": round(drift0[1], 1),
               "reloc_res_cm": round(res[0] * 100, 1), "reloc_res_deg": round(res[1], 2),
               "final_lat_m": round(lat, 3), "final_bearing_deg": round(berr, 2),
               "final_front_deg": round(ferr, 2), "final_dist_m": round(float(np.hypot(tx - env.x, ty - env.y)), 3)}
        _reloc_stats.append(rec)
        pipe["8"] = f"{name}: lat{lat*100:.1f}cm berr{berr:.1f}d front{ferr:.1f}d dist{rec['final_dist_m']:.2f}m (true-pose)"
        print(f"    [DOCK] {name}: 终误差 lat={lat*100:.1f}cm bearing={berr:.1f}° front={ferr:.1f}° "
              f"(对照: 无RELOC≈漂移{drift0[0]*100:.0f}cm)", flush=True)
        return rec

    TERMS = {"iron_cabinet": "a white cabinet with glass doors", "microscope": "a microscope",
             "pipette_table": "a white table"}   # 移液枪渲染里太小不可见→用桌子本体(bbox+confirm都靠它)
    LABEL = {"iron_cabinet": "IRON CABINET", "microscope": "MICROSCOPE",
             "pipette_table": "WHITE TABLE (pipettes)", "return_origin": "ORIGIN"}
    import re as _re
    def _term(nm):                                    # ⭐走廊等: "a blue water pump"(去尾号, Qwen具体单名词甜区0716)
        return TERMS.get(nm) or ("a " + _re.sub(r"_\d+$", "", nm).replace("_", " "))
    def _label(nm):
        return LABEL.get(nm) or nm.replace("_", " ").upper()
    _LEGACY = {"goto_cabinet": "iron_cabinet", "strafe_microscope": "microscope",
               "goto_pipette": "pipette_table", "return_origin": "return_origin"}
    def _leg_name(tag):
        return _LEGACY.get(tag) or tag[len("goto_"):]
    def _leg_lbl(tag):
        return _label(_leg_name(tag))
    _tag = "zed" if not args.targets else os.path.splitext(os.path.basename(args.targets))[0].replace("lab_mission_targets_", "")
    _ck_tag = os.path.splitext(os.environ.get("CKPT", "goalnav_gate_lab.pt"))[0].replace("goalnav_gate", "").strip("_") or "lab"
    _sb_tag = "" if args.no_review else "_sb2"          # ⭐⑩⑪新慢脑(推理链+复核)版标识, 防覆盖0802旧慢脑视频(版本化铁律)
    _mipre = "lab_mission_110744" if args.session == "colmapjoint_110744s2" else f"mission_{args.session}"          # ⭐⑩⑪新慢脑(推理链+复核)版标识, 防覆盖0802旧慢脑视频(版本化铁律)
    out = os.path.join(ROOT, "outputs", f"{_mipre}_{_tag}_geo0804_{_ck_tag}{_sb_tag}{'_reloc' if args.reloc_demo else ''}{'_odom' if args.reloc_odom else ''}{'_precdock0812' if args.precise_dock else ''}{'_head' if args.reloc_head else ''}{'_flow' if args.flow else ''}.mp4")   # 版本化不覆盖(铁律); geo0804=z场扶平+正脸线标注版; _precdock0812=p10/min(ED,GEO)伺服+盲爬版; _flow=全流程面板慢速版(0812)
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264",
                               output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    frames_written = [0]
    log = {"state": "", "vlm": "", "leg": "", "brain": ""}
    # ===== ⭐0812全流程面板数据(--flow): 8段管线各留一行实时值, 各阶段自己往里写; cur=当前高亮段 =====
    pipe = {"cur": "", "1": "", "2": "", "3": "", "4": "", "5": "", "6": "", "7": "", "8": "",
            "legs": [(("CRUISE -> " if m == "A*" else "STRAFE -> ") + _leg_lbl(tg), " ") for m, a, b, tg in legs]}
    _dwell = {"on": False}                                  # 停留帧期间: kf不捕获重复帧/hist不入队
    # ===== ⭐0812场景图面板v4(--flow): 真实场景图【ego第一人称】——与左栏FPV同相机同俯仰逐帧投影 =====
    # 用户: "从右边跟左边对应上"——右栏=机器人眼睛看到的场景图(语义层), 左栏=GS渲染(皮肤层),
    # 节点错tag/过合并/位置偏移=右栏标签与左栏实物对不上, 一眼可检。
    _sg = {"pts": None, "cols": None, "nodes": None, "path": None, "so": None}
    if args.flow:
        _pk = os.path.join(ROOT, "outputs", "cg_lab110744s2", "lab1", "sg_cache", "map",
                           "scene_map_cfslam_pruned.pkl.gz")
        if os.path.exists(_pk):
            import pickle as _pk_, gzip as _gz, zlib as _zl
            with _gz.open(_pk, "rb") as _f:
                _smap = _pk_.load(_f)
            _pts, _col, _nodes = [], [], []
            _rngsg = np.random.RandomState(0)
            _TGT_ID = {"iron_cabinet": 130, "microscope": 148, "pipette_table": 20}
            for _i, _o in enumerate(_smap):
                _pc = np.asarray(_o["pcd_np"], np.float64)
                if len(_pc) < 5:
                    continue
                if len(_pc) > 3000:                          # 每节点抽3000点(轮廓可辨)
                    _pc = _pc[_rngsg.choice(len(_pc), 3000, replace=False)]
                _h = _zl.crc32(str(_i).encode()) % 180       # 节点id稳定配色
                _c = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[_h, 170, 225]]]), cv2.COLOR_HSV2BGR)[0, 0])
                _pts.append(_pc); _col += [_c] * len(_pc)
                _tags = [t for t in _o.get("class_name", []) if isinstance(t, str)]
                _tag = max(set(_tags), key=_tags.count) if _tags else "?"
                _nodes.append((np.asarray(_o["pcd_np"], np.float64).mean(0),
                               f"{_tag[:14]}#{_i}", _i in _TGT_ID.values()))
            _sg["pts"] = np.vstack(_pts)
            _sg["cols"] = np.array(_col, np.uint8)
            _sg["nodes"] = _nodes
            print(f"[mission] 场景图ego面板: {len(_smap)}节点 {len(_sg['pts'])}点", flush=True)

    # ===== ⭐⑪ key-frame证据缓冲(论文"按索引回原图"): 接近过程中每6帧存一帧(小图+距目标+方位差),
    # confirm判no时挑top3复核——机器人不动, 只回查已有证据, ≠0730禁的扫描挽尊(那是不停移动找通过视角)。 =====
    _kf = {"target": None, "buf": []}          # target=(name,tx,ty); buf=[(frame_idx, bgr小图, dist, |berr|°)]
    def kf_set_target(name):
        _kf["target"] = None if name is None else (name, float(T[name][0]), float(T[name][1]))
        _kf["buf"] = []
    def kf_capture(fpv_bgr):
        if _kf["target"] is None or frames_written[0] % 6:
            return
        _, tx, ty = _kf["target"]
        dist = float(np.hypot(env.x - tx, env.y - ty))
        berr = abs(np.degrees(wrap(np.arctan2(ty - env.y, tx - env.x) - env.yaw)))
        small = cv2.resize(fpv_bgr, (320, 180), interpolation=cv2.INTER_AREA)
        _kf["buf"].append((frames_written[0], small, round(dist, 2), round(berr, 1)))
        del _kf["buf"][:-120]                  # 上限120帧≈20条/腿, 防爆内存
    def kf_topk(k=3):
        """按 距离+方位差 打分挑top-k, 帧间至少隔8帧(视角要有差异)。"""
        cand = sorted(_kf["buf"], key=lambda r: r[2] + 0.02 * r[3])
        pick = []
        for r in cand:
            if all(abs(r[0] - p[0]) >= 8 for p in pick):
                pick.append(r)
            if len(pick) == k:
                break
        return pick

    def panel(obs):
        fpv = np.ascontiguousarray(sim_fpv()[:, :, ::-1])     # capcam连续渲染(匹配pitch, 治白雾)
        if not _dwell["on"]:
            kf_capture(fpv)                                   # ⭐⑪证据帧捕获(复用本帧渲染, 零额外开销; dwell期不捕重复帧)
        H = fpv.shape[0]
        bev = cv2.resize(((1 - obs["bev"]) * 255).astype(np.uint8), (H, H), interpolation=cv2.INTER_NEAREST)
        bev = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        cv2.circle(bev, (H // 2, H // 2), 6, (0, 0, 255), -1)
        cv2.arrowedLine(bev, (H // 2, H // 2), (H // 2, H // 2 - 30), (0, 0, 255), 2)
        cv2.putText(fpv, f"[SLOW-VLM] {log['state']}  ->{log['leg']}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(fpv, f"[FAST-DD] {log['brain']} [{log.get('_hist','')}]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
        if log["vlm"]:
            cv2.putText(fpv, f"VLM: {log['vlm']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.putText(bev, "BEV (DD in)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # ===== ⭐向量标注(0730用户"把正对用向量标出来"): FPV=中心十字/目标投影/正面向量/水平基准线; BEV=目标星+正面向量 =====
        nm = _aim["name"]
        if nm in so:
            Wf = fpv.shape[1]
            # 人工水平线(相机pitch的水平基准, 白虚线): 桌沿跟它比=真歪还是透视一眼见
            v_hor = int(_cys - _fys * np.tan(abs(capfpv.pitch)))
            for x0 in range(0, Wf, 24):
                cv2.line(fpv, (x0, v_hor), (min(x0 + 12, Wf), v_hor), (255, 255, 255), 1)
            cv2.drawMarker(fpv, (Wf // 2, H // 2), (0, 255, 255), cv2.MARKER_CROSS, 22, 1)      # 中心十字(黄)
            # 目标3D点投影(绿X, 对正时应压中心十字) + 正面向量(青箭头)
            Vv = capfpv._world_to_colmap_cam(env.x, env.y, env.yaw, roll_deg=0.0)
            tp = np.array(T[nm], dtype=np.float64)[None, :]
            u, v, vis = _project_cluster(tp, Vv)
            if vis[0]:
                cv2.drawMarker(fpv, (int(u[0]), int(v[0])), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 26, 2)
            if nm in FRONT:
                th = np.radians(FRONT[nm])
                # ⭐0804改(用户判"正对的那个向量就不对"): FPV画"正脸线"=过目标点、在正脸平面内的
                # 水平线段(±0.3m, 方向⊥front)——正对时它应水平且压中心十字, 歪几度一眼可见。
                # 旧法线箭头(3D外法线投影)在正视时退化: 俯拍22°下投影成竖线扎进地板(显微镜站)
                # 或缩成点看不见(铁柜站)——3D对但视觉误导。法线箭头只留BEV(那里无歧义)。
                e = np.array([-np.sin(th), np.cos(th), 0.0]) * 0.3
                u2, v2, vis2 = _project_cluster(np.vstack([tp - e, tp + e]), Vv)
                if vis2[0] and vis2[1]:
                    cv2.line(fpv, (int(u2[0]), int(v2[0])), (int(u2[1]), int(v2[1])), (255, 255, 0), 2)
                # BEV: 目标星(黄)+正面向量(品红箭头), ego heading-up映射(win_m见test_env)
                dx, dy = T[nm][0] - env.x, T[nm][1] - env.y
                cy_, sy_ = np.cos(env.yaw), np.sin(env.yaw)
                f_ = dx * cy_ + dy * sy_; r_ = dx * sy_ - dy * cy_
                sc = H / env.win_m
                gc, gr = int(H / 2 + r_ * sc), int(H / 2 - f_ * sc)
                if 0 <= gc < H and 0 <= gr < H:
                    cv2.drawMarker(bev, (gc, gr), (0, 255, 255), cv2.MARKER_STAR, 16, 2)
                    # ⭐世界角th须先转ego系(fwd=cos(th-yaw), right=-sin(th-yaw))再取反=外法线;
                    # 之前直接cos(th)/sin(th)当ego分量=yaw≠0时全歪(0731修)
                    f2 = f_ - np.cos(th - env.yaw); r2 = r_ + np.sin(th - env.yaw)
                    cv2.arrowedLine(bev, (gc, gr), (int(H / 2 + r2 * sc), int(H / 2 - f2 * sc)), (255, 0, 255), 2, tipLength=0.3)
                err_f = np.degrees(wrap(th - env.yaw))
                cv2.putText(fpv, f"front={FRONT[nm]:.0f}deg yaw_err={err_f:+.1f}deg", (10, H - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                if _aim.get("edge") is not None:                    # ⭐检测到的物体真实边缘线(绿)——应与青线重合
                    _x1, _y1, _x2, _y2 = _aim["edge"]
                    cv2.line(fpv, (int(_x1), int(_y1)), (int(_x2), int(_y2)), (0, 255, 0), 2)
        if not args.flow:
            return np.hstack([fpv, bev])[:, :, ::-1]
        # ===== ⭐0812三栏全流程面板(--flow): [FPV][文字面板640][场景图3D 640] =====
        PW = 640
        P = np.zeros((H, PW, 3), np.uint8)
        F = cv2.FONT_HERSHEY_SIMPLEX
        def pt(img, s, x, y, sc=0.42, col=(200, 200, 200), th=1):     # BGR(末帧统一翻RGB)
            cv2.putText(img, s, (x, y), F, sc, col, th, cv2.LINE_AA)
        bx, by, byaw = bel_pose()
        dn = float(np.hypot(_drift[0], _drift[1])); dd = abs(float(np.degrees(_drift[2])))
        # -- 左上: BEV迷你 --
        bs = 200
        P[24:24 + bs, 6:6 + bs] = cv2.resize(bev, (bs, bs), interpolation=cv2.INTER_NEAREST)
        pt(P, "BEV ego view (DD input)", 8, 16, 0.42, (220, 180, 80))
        # -- 右上: 任务清单 + 当前PHASE --
        pt(P, "MISSION: 3-station inspect + precision dock", 222, 20, 0.44, (255, 255, 255))
        for i, (lg, st) in enumerate(pipe["legs"]):
            mk = {" ": "[ ]", ">": "[>>]", "x": "[OK]"}[st]
            col = (0, 255, 255) if st == ">" else ((0, 220, 0) if st == "x" else (140, 140, 140))
            pt(P, f"{mk} {lg}", 222, 44 + i * 20, 0.42, col)
        phase = log["state"] or "-"
        pt(P, f"PHASE: {phase[:38]}", 222, 134, 0.52, (0, 255, 255), 2)
        # POSE块(活值): believed位姿+注入漂移+距目标——诚实标注drift是注入的
        pt(P, f"POSE bel=({bx:+.2f},{by:+.2f},{np.degrees(byaw):+.0f}d)", 222, 160, 0.42)
        pt(P, f"drift now {dn:.2f}m/{dd:.0f}d (INJECTED, demo)", 222, 178, 0.42, (80, 120, 255))
        if _aim["name"] in T:
            _tx, _ty = float(T[_aim["name"]][0]), float(T[_aim["name"]][1])
            pt(P, f"node-dist {float(np.hypot(env.x-_tx, env.y-_ty)):.2f}m (world x0.70)", 222, 196, 0.42)
        pt(P, "sim obs: pseudo-zed = renderED/0.70 + noise", 222, 216, 0.38, (120, 120, 120))
        # -- 下: 8段管线 --
        cv2.line(P, (6, 232), (PW - 6, 232), (90, 90, 90), 1)
        SEC = [("1", "WHERE graph"), ("2", "PLAN A*"), ("3", "RELOC"), ("4", "FAST-DD"),
               ("5", "ALIGN"), ("6", "SLOW-VLM"), ("7", "DOCK"), ("8", "METRIC")]
        for i, (k, nm_) in enumerate(SEC):
            y = 252 + i * 19
            cur = pipe["cur"] == k
            val = pipe[k] or "..."
            col = (0, 255, 180) if cur else ((210, 210, 210) if pipe[k] else (110, 110, 110))
            pt(P, f"[{k}]{nm_:<12s}", 8, y, 0.42, (0, 255, 180) if cur else (150, 150, 150))
            pt(P, val[:72], 122, y, 0.40, col)
        # -- 第三栏: 真实场景图ego第一人称(与左栏FPV同相机同俯仰, 逐帧投影) --
        if _sg["pts"] is not None:
            Wf_ = fpv.shape[1]
            SG = np.zeros((H, Wf_, 3), np.uint8)
            V3 = capfpv._world_to_colmap_cam(env.x, env.y, env.yaw, roll_deg=0.0)   # 与sim_fpv同位姿
            _pc = capfpv.s_norm * (capfpv.Rz.T @ ((_sg["pts"] - capfpv.t_world) / capfpv.k).T).T + capfpv.center
            _pm = (V3[:3, :3] @ _pc.T).T + V3[:3, 3]
            _z = _pm[:, 2]
            _m = _z > 0.10
            _u = _fxs * _pm[:, 0] / np.where(_m, _z, 1) + _cxs
            _v = _fys * _pm[:, 1] / np.where(_m, _z, 1) + _cys
            _m &= (_u >= 0) & (_u < Wf_ - 1) & (_v >= 0) & (_v < H - 1)
            _ord = np.argsort(-_z)                                   # 远→近画家算法
            _keep = _m[_ord]
            _ui = _u[_ord][_keep].astype(int); _vi = _v[_ord][_keep].astype(int)
            _ci = _sg["cols"][_ord][_keep]
            for _du in range(3):                                     # 3px块(轮廓/完整体区分度)
                for _dv in range(3):
                    SG[np.clip(_vi + _dv, 0, H - 1), np.clip(_ui + _du, 0, Wf_ - 1)] = _ci
            # 节点标签(视野内, 贪心去重叠28px; 任务节点黄色强制)
            _placed = []
            for _c3, _txt, _big in _sg["nodes"]:
                _q = capfpv.s_norm * (capfpv.Rz.T @ ((_c3 - capfpv.t_world) / capfpv.k)) + capfpv.center
                _qc = V3[:3, :3] @ _q + V3[:3, 3]
                if _qc[2] < 0.15:
                    continue
                _lu = _fxs * _qc[0] / _qc[2] + _cxs; _lv = _fys * _qc[1] / _qc[2] + _cys
                if not (4 <= _lu < Wf_ - 90 and 12 <= _lv < H - 4):
                    continue
                if not _big and any((_lu - pu) ** 2 + (_lv - pv) ** 2 < 28 ** 2 for pu, pv in _placed):
                    continue
                cv2.putText(SG, _txt, (int(_lu) + 2, int(_lv) - 2), F,
                            0.44 if _big else 0.32, (0, 255, 255) if _big else (200, 200, 200), 1, cv2.LINE_AA)
                _placed.append((_lu, _lv))
            # 任务目标X + 名字(与左栏绿X同一投影数学=应重合)
            for nm_ in _STATIONS:                                   # ⭐0813晚: 站名数据驱动(走廊=4泵站)
                _parts = nm_.split("_")
                lb_ = ("".join(w[0] for w in _parts[:-1]) + _parts[-1]).upper() if _parts[-1].isdigit() else nm_[:3].upper()
                _q = np.array([T[nm_][0], T[nm_][1], T[nm_][2] if len(T[nm_]) > 2 else 0.9], dtype=np.float64)
                _qc_ = capfpv.s_norm * (capfpv.Rz.T @ ((_q - capfpv.t_world) / capfpv.k)) + capfpv.center
                _qm = V3[:3, :3] @ _qc_ + V3[:3, 3]
                if _qm[2] > 0.15:
                    _tu = int(_fxs * _qm[0] / _qm[2] + _cxs); _tv = int(_fys * _qm[1] / _qm[2] + _cys)
                    if 8 <= _tu < Wf_ - 40 and 8 <= _tv < H - 8:
                        cv2.drawMarker(SG, (_tu, _tv), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
                        cv2.putText(SG, lb_, (_tu + 8, _tv - 6), F, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
            # 水平基准线+中心十字(与左栏同位置, 方便左右对照)
            v_hor = int(_cys - _fys * np.tan(abs(capfpv.pitch)))
            for x0 in range(0, Wf_, 24):
                cv2.line(SG, (x0, v_hor), (min(x0 + 12, Wf_), v_hor), (90, 90, 90), 1)
            cv2.drawMarker(SG, (Wf_ // 2, H // 2), (0, 255, 255), cv2.MARKER_CROSS, 22, 1)
            cv2.putText(SG, "SCENE GRAPH ego view (same camera as FPV)", (8, 16), F, 0.4, (220, 180, 80), 1, cv2.LINE_AA)
            cv2.putText(SG, "label=tag#id  X=target", (8, H - 8), F, 0.34, (130, 130, 130), 1, cv2.LINE_AA)
        else:
            SG = np.zeros((H, 640, 3), np.uint8)
        return np.hstack([fpv, P, SG])[:, :, ::-1]

    def flow_card(title, lines, n=45):
        """片头/片尾卡: 左=压暗FPV+大标题, 右=说明行。emit n帧(读者阅读时间)。"""
        if not args.flow:
            return
        fpv = np.ascontiguousarray(sim_fpv()[:, :, ::-1])
        H = fpv.shape[0]
        left = (fpv * 0.25).astype(np.uint8)
        F = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(left, title, (20, H // 2), F, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        P = np.zeros((H, 1280, 3), np.uint8)                  # 片头/尾卡: 右侧=FPV+SG两栏宽
        for i, ln in enumerate(lines):
            cv2.putText(P, ln, (14, 30 + i * 22), F, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        fr = np.hstack([left, P])[:, :, ::-1]
        for _ in range(n):
            writer.append_data(fr); frames_written[0] += 1

    trace = []
    _aim = {"name": None}                                  # 当前腿目标(供panel画向量标注)
    def emit(obs, hist=True):
        if hist:
            hist_update()                                          # ⭐每步入队历史位姿(观察策略参数化)
        writer.append_data(panel(obs)); frames_written[0] += 1
        trace.append((env.x, env.y, env.yaw, log.get("_vx", 0.0), log.get("_vy", 0.0),
                      log.get("_wz", 0.0), log["state"]))
    def dwell(n=12):
        """⭐0812 --flow慢节奏: 决策点停留n帧给观众读面板。位姿不变→hist不入队/kf不捕获(防污染)。"""
        if not args.flow:
            return
        _dwell["on"] = True
        for _ in range(n):
            emit(env._obs(), hist=False)
        _dwell["on"] = False

    # ===== 快脑驱动: 沿A*粗路径, DD给前进速度; 朝向=平滑跟踪路径(A*已避障, 不用势场抢方向) =====
    def cruise(path, tol=0.40, maxs=300, phi=None):
        """CRUISE: A*粗路径定方向(WHERE, A*本身避开障碍), DiffusionDrive快脑给前进速度(HOW, 保留快脑在环)。
        ⭐朝向用平滑比例控制器对准前视点=灭抖(不再用DD噪声wz+bearing hack+safe_vel每帧抢方向);
        ⭐近终点直瞄终点+减速+朝向偏先转再走=灭orbit转圈(不再追侧向前视点绕着走)。
        phi=⑧参数包观察配置(W,γ,vx), 默认巡航配置。"""
        phi = phi or PHI["CRUISE"]
        obs = env._obs()
        wz_prev = 0.0
        for _i in range(maxs):
            # ⭐周期RELOC模式: 每步里程计随机游走; 控制用believed位姿(=真实机器人漂移里程计),
            # FPV/BEV仍用真位姿(=自车传感, 不漂); 每25步(~2m)一次RELOC纠偏
            if _odom["on"]:
                odom_random_walk()
                if _i > 0 and _i % 25 == 0:
                    odom_checkpoint(log.get("leg", "leg"), _i)
                _sl = _odom.get("slew")                           # ⭐纠偏渐进施加(每步≤2cm/0.3°), 防前视点瞬跳猛踢
                if _sl is not None:
                    _d = _sl - _drift
                    _drift[:] = _drift + np.array([np.clip(_d[0], -0.02, 0.02), np.clip(_d[1], -0.02, 0.02),
                                                   np.clip(_d[2], -np.radians(0.3), np.radians(0.3))])
                    if np.hypot(_d[0], _d[1]) < 0.005 and abs(_d[2]) < np.radians(0.2):
                        _odom["slew"] = None
            bx, by, byaw = bel_pose()                                # believed位姿(odom off时=真值)
            pose = (env.x, env.y, env.yaw)                           # 传感的自车位姿(BEV/FPV用, 不漂)
            fpv = np.ascontiguousarray(sim_fpv()[:, :, ::-1])   # capcam渲染喂快脑
            dist_goal = float(np.hypot(bx - path[-1, 0], by - path[-1, 1]))
            if dist_goal < tol:
                break
            # 前视点: 远处沿A*路径, 近终点(<1m)直接瞄终点→不再追侧向点绕圈
            if dist_goal > 1.0:
                look = min(GOAL_AHEAD, max(1.0, dist_goal))
                d = np.hypot(path[:, 0] - bx, path[:, 1] - by); i0 = int(d.argmin())
                jj = i0; acc = 0.0
                while jj + 1 < len(path) and acc < look:
                    acc += float(np.linalg.norm(path[jj + 1] - path[jj])); jj += 1
                aim = path[jj]
            else:
                aim = path[-1]
            # ⭐朝向: 平滑比例控制器对准aim(believed系), 主导; DD的wz只轻掺(保留快脑影响)
            head_err = wrap(np.arctan2(aim[1] - by, aim[0] - bx) - byaw)
            c_, s_ = np.cos(byaw), np.sin(byaw)
            gx, gy = aim - np.array([bx, by])
            goal_ego = np.array([c_ * gx + s_ * gy, -s_ * gx + c_ * gy], np.float32)
            traj = dd_traj(fpv, pose, goal_ego,                    # ⭐快脑仍在环; 巡航=短窗观察(省算力)
                           hist_w=phi[0], hist_gamma=phi[1], vx=phi[2])  # ⑧Φ来自本腿参数包(不再读全局)
            dd_vx, _, dd_wz = sf.traj_to_control(traj)
            wz = 0.75 * float(np.clip(1.5 * head_err, -0.7, 0.7)) + 0.25 * float(np.clip(dd_wz, -0.7, 0.7))
            # ⭐前进速度=DD给, 朝向偏大先减速转正(不边走边甩=不画龙), 近终点减速(不冲过头绕圈)
            turn_slow = float(np.clip(1.0 - abs(head_err) / 1.0, 0.15, 1.0))
            goal_slow = float(np.clip(dist_goal / 1.2, 0.35, 1.0))
            vx = float(np.clip(abs(dd_vx), 0.2, 1.0)) * SPEED * turn_slow * goal_slow
            # safe_vel降级=只减速(A*路径已避障, 不再抢方向→不抖)
            nx = env.x + vx * np.cos(env.yaw) * 0.25; ny = env.y + vx * np.sin(env.yaw) * 0.25
            need = ROBOT_R + 0.25; cl = clearance(nx, ny)
            if cl < need:
                vx *= float(np.clip(cl / need, 0.2, 1.0))
            wz = 0.5 * wz_prev + 0.5 * wz; wz_prev = wz              # 轻低通(输入已平滑)
            wz = float(np.clip(wz, -0.7, 0.7))
            log["_vx"], log["_vy"], log["_wz"] = vx, 0.0, wz
            log["brain"] = f"CRUISE v={vx:.2f} w={wz:.2f} d={dist_goal:.1f} he={np.degrees(head_err):.0f}"
            pipe["cur"] = "4"
            pipe["4"] = f"Phi(W{phi[0]},g{phi[1]},vx{phi[2]}) -> v={vx:.2f},w={wz:+.2f} d_goal={dist_goal:.1f}m he={np.degrees(head_err):+.0f}d"
            obs, _, _ = env.step(vx, 0.0, wz); emit(obs)
        return obs

    TURN_DD_MIN = 0.9   # rad(~51°): 以上用gate-DD原地转(展示可控转向, 训练分布|dyaw|∈[54°,90°]);
    #                     以下切控制器wz=1.5*e收尾——DD在<51°是OOD区, 输出wz正负乱翻→env每帧±wz·dt→
    #                     period-2极限环抖动(原地转不动+抽搐+收敛不到正对), 对齐run_inspection的TURN_TO_FACE。
    def face(target_xy, maxs=90):
        """TURN_TO_FACE 混合原地转: 大角度DD(可控转向) + 小角度控制器收尾(稳收敛不抖)。⚠转向不乘SPEED(要转完)。"""
        obs = env._obs()
        for _ in range(maxs):
            e = wrap(np.arctan2(target_xy[1] - env.y, target_xy[0] - env.x) - env.yaw)
            if abs(e) < 0.04:                                    # ⭐收紧退出阈值→目标水平居中(±2°, 之前0.12=6.9°停偏一侧)
                break
            pose = (env.x, env.y, env.yaw)
            wz = None; src = "控制器"
            if abs(e) > TURN_DD_MIN:                             # 大角度: gate-DD原地转
                ec = float(np.clip(e, -np.pi / 2, np.pi / 2))   # 夹到训练范围|dyaw|≤90°
                fpv = np.ascontiguousarray(sim_fpv()[:, :, ::-1])
                traj = dd_traj(fpv, pose, np.zeros(2, np.float32), head_ego=(np.cos(ec), np.sin(ec)),
                               hist_w=PHI["ALIGN"][0], hist_gamma=PHI["ALIGN"][1], vx=PHI["ALIGN"][2])  # ⭐转向=长窗+vx=0(⑧Φ对齐训练)
                wz = turn_wz(traj); src = "DD"
                if abs(wz) < 0.20:                              # ⭐DD在交接边界(~51°)输出太弱转不动→立刻交控制器(治e卡55°没正对)
                    wz = None; src = "控制器(DD停)"
            if wz is None:                                       # 小角度/DD停: 控制器比例收尾(不抖, 不乘SPEED, 保证收敛)
                wz = float(np.clip(1.5 * e, -0.8, 0.8))
            log["_vx"], log["_vy"], log["_wz"] = 0.0, 0.0, wz
            log["brain"] = f"TURN({src}) w={wz:.2f} e={np.degrees(e):.0f}°"
            obs, _, _ = env.step(0.0, 0.0, wz); emit(obs)
        efin = wrap(np.arctan2(target_xy[1] - env.y, target_xy[0] - env.x) - env.yaw)
        print(f"    [face] 收敛残余 e={np.degrees(efin):.1f}°", flush=True)
        return obs

    def align_front(name, max_iter=5):
        """⭐正对闭环(0730终版, 治"打bearing≠正对"实测偏7~12°): cruise到站有0.15m容差→偏离正面轴。
        1) 转到front yaw(正面向量, 不是目标方位角); 2) 横移闭环: 位置投影到正面轴(过target沿front的直线),
        横向残差用全向vy步进消掉(纯几何+clearance), 直到|bearing-front|<1.5°=站位在轴上+朝向正对+目标居中三合一。
        无front_deg的目标退回face()。"""
        if name not in FRONT:
            return face(T[name][:2])
        th = np.radians(FRONT[name]); dirv = np.array([np.cos(th), np.sin(th)])
        tx, ty = float(T[name][0]), float(T[name][1])
        obs = env._obs()
        for _ in range(90):                                        # 1) 转到front yaw(believed系, drift=0时=真值)
            e = wrap(th - bel_pose()[2])
            if abs(e) < 0.02:
                break
            wz = float(np.clip(1.5 * e, -0.7, 0.7))
            log["_vx"], log["_vy"], log["_wz"] = 0.0, 0.0, wz
            log["brain"] = f"ALIGN turn e={np.degrees(e):.0f}°"
            obs, _, _ = env.step(0.0, 0.0, wz); emit(obs)
        for it in range(max_iter):                                 # 2) 横移上轴闭环(believed系几何)
            bxp, byp, byawp = bel_pose()
            pos = np.array([bxp, byp])
            s = float((pos - [tx, ty]) @ dirv)
            closest = np.array([tx, ty]) + s * dirv
            lat = pos - closest
            L = float(np.linalg.norm(lat))
            berr = np.degrees(wrap(np.arctan2(ty - byp, tx - bxp) - th))
            print(f"    [ALIGN] {name} it{it}: 横向偏移{L:.2f}m bearing-front={berr:+.1f}°", flush=True)
            pipe["cur"] = "5"
            pipe["5"] = f"it{it}: lat-off {L:.2f}m, bearing-front {berr:+.1f}d (strafe on-axis loop)"
            if abs(berr) < 1.5 or L < 0.02:
                break
            step = -lat if L <= 0.18 else -lat * 0.18 / L          # 朝轴移动, 每步≤0.18m
            if clearance(bxp + step[0], byp + step[1]) < ROBOT_R + 0.10:
                print(f"    [ALIGN] {name}: clearance不足, 接受现状", flush=True)
                break
            dist = float(np.linalg.norm(step)); nsteps = max(1, int(dist / 0.06))
            c, sg = np.cos(byawp), np.sin(byawp)
            bx = float(step @ np.array([c, sg])); by = float(step @ np.array([-sg, c]))
            for _ in range(nsteps):                                # 世界step→body(vx,vy): 左=(-s,c)
                vx, vy = bx / max(dist, 1e-6) * 0.24, by / max(dist, 1e-6) * 0.24
                log["_vx"], log["_vy"], log["_wz"] = vx, vy, 0.0
                log["brain"] = f"ALIGN strafe lat={L:.2f}m"
                obs, _, _ = env.step(vx, vy, 0.0); emit(obs)
            for _ in range(30):                                    # 横移后yaw复对准front
                e = wrap(th - bel_pose()[2])
                if abs(e) < 0.02:
                    break
                wz = float(np.clip(1.5 * e, -0.6, 0.6))
                obs, _, _ = env.step(0.0, 0.0, wz); emit(obs)
        return env._obs()

    def align_dispatch(name):
        return align_front(name)

    # ===== ⭐场景图几何精对准(0730二判"不对准"的修法): face()打到节点bbox中心(抽象质心, 有偏),
    #   改为把【节点自己的3D点簇】投影进实时渲染, 转到你"看见的簇质量"水平居中——
    #   纯场景图几何(点簇是图的节点数据), 非VLM bbox非手搓检测。逐位置站位误差+质心偏差一起收。 =====
    NODE_ID = {"iron_cabinet": 130, "microscope": 148, "pipette_table": 20}
    _clusters = {}
    _cnpy = os.path.join(ROOT, "outputs", "cg_lab110744s2", "node_clusters_mission3.npz")
    if os.path.exists(_cnpy):
        _nz = np.load(_cnpy)
        _clusters = {n: _nz[f"pcd_{i}"] for n, i in NODE_ID.items() if f"pcd_{i}" in _nz}
        print(f"[mission] 节点簇 loaded: {[(k, len(v)) for k, v in _clusters.items()]}", flush=True)
    _K = capfpv.K[0].cpu().numpy().astype(np.float64)
    _fxs, _cxs = _K[0, 0] * (capfpv.W / capfpv.Wr), _K[0, 2] * (capfpv.W / capfpv.Wr)
    _fys, _cys = _K[1, 1] * (capfpv.H / capfpv.Hr), _K[1, 2] * (capfpv.H / capfpv.Hr)

    def _project_cluster(pts_w, V):
        """世界点簇→当前渲染像素(与diag验证同一套数学)。"""
        p_col = capfpv.s_norm * (capfpv.Rz.T @ ((pts_w - capfpv.t_world) / capfpv.k).T).T + capfpv.center
        p_cam = (V[:3, :3] @ p_col.T).T + V[:3, 3]
        z = p_cam[:, 2]; m = z > 0.05
        u = _fxs * p_cam[:, 0] / np.where(z == 0, 1, z) + _cxs
        v = _fys * p_cam[:, 1] / np.where(z == 0, 1, z) + _cys
        vis = m & (u >= 0) & (u < capfpv.W) & (v >= 0) & (v < capfpv.H)
        return u, v, vis

    def graph_fine_aim(name, max_iter=3, deadband=np.radians(2)):
        """把节点簇可见部分的像素质心转到画面水平中线(±2°死区)。返回最终角误差(度), 无簇则跳过。"""
        pts = _clusters.get(name)
        if pts is None or len(pts) < 50:
            return None
        rng = np.random.RandomState(0)
        sub = pts[rng.choice(len(pts), min(3000, len(pts)), replace=False)]
        err_fin = None
        for it in range(max_iter):
            V = capfpv._world_to_colmap_cam(env.x, env.y, env.yaw)   # 与sim_fpv同位姿同roll
            u, v, vis = _project_cluster(sub, V)
            if vis.sum() < 30:
                print(f"    [FINE-AIM] {name} it{it}: 簇可见点{vis.sum()}太少, 保持face()朝向", flush=True)
                break
            err = float(np.arctan((u[vis].mean() - _cxs) / _fxs))    # 簇质心偏右为正→需减小yaw
            err_fin = np.degrees(err)
            print(f"    [FINE-AIM] {name} it{it}: 簇质心u={u[vis].mean()/capfpv.W:.2f} err={err_fin:+.1f}°", flush=True)
            log["brain"] = f"FINE-AIM(图几何) it{it} e={err_fin:+.1f}°"
            if abs(err) < deadband:
                break
            target_yaw = env.yaw - err
            for _ in range(60):                                       # 控制器小角度转(与face收尾同参, 边转边emit)
                e = wrap(target_yaw - env.yaw)
                if abs(e) < 0.02:
                    break
                wz = float(np.clip(1.5 * e, -0.6, 0.6))
                log["_vx"], log["_vy"], log["_wz"] = 0.0, 0.0, wz
                obs, _, _ = env.step(0.0, 0.0, wz); emit(obs)
        return err_fin

    # ===== ⭐贴脸精对接(0811): 深度伺服到距面0.15m + 缝隙/质心居中(<1cm) =====
    # 全程只读"传感器"(渲染ED中央ROI=zed中央ROI / FPV线检测), 不读地图坐标 → 与真机同构。
    _dock_stats = []

    def _ed_face_depth(frac=0.3, q=10):
        """当前位姿渲染ED, 中央竖条ROI的q分位深度(米)=到前面距离。渲染分辨率,colmap单位×k。
        ⭐0812改p10: median会被'看穿玻璃门读背景'(铁柜站读2.8m实际面距0.85m)/近场溅射双峰拉扯;
        p10读最近可信面=保守不撞。⚠<0.4m近场p10本身也膨胀(0811实测+0.6m)→伺服只用到0.35m, 之后盲爬。"""
        V = capfpv._world_to_colmap_cam(env.x, env.y, env.yaw, roll_deg=0.0)
        _, dep = capfpv._render_rgbd_V(V)
        d = dep * capfpv.k / capfpv.s_norm            # colmap单位→世界米(⚠/s_norm, 0811漏了→面距读3.1m的bug)
        Hr_, Wr_ = d.shape
        roi = d[int(Hr_ * frac):int(Hr_ * (1 - frac)), int(Wr_ / 2 - Wr_ * 0.08):int(Wr_ / 2 + Wr_ * 0.08)]
        roi = roi[(roi > 0.02) & (roi < 3.0)]
        return float(np.percentile(roi, q)) if len(roi) > 50 else None

    def _dock_step(fwd=0.0, lat=0.0, dyaw=0.0, n=1):
        """精对接小步进: 绕过占据检查(贴脸段必进occ格; 对接走廊由对准+深度伺服保证安全)。lat>0=左。"""
        for _ in range(n):
            c, s = np.cos(env.yaw), np.sin(env.yaw)
            env.x += fwd * c - lat * s; env.y += fwd * s + lat * c
            env.yaw = float(wrap(env.yaw + dyaw))
            log["_vx"], log["_vy"], log["_wz"] = fwd / 0.25, lat / 0.25, dyaw / 0.25
            emit(env._obs())

    def _seam_offset_px(fpv_bgr, u_anchor=None):
        """FPV检测竖直缝(铁柜双门中缝): LSD近竖直长线, 取离锚点(u_anchor=节点投影)最近者, 无锚退回画面中线。
        ⭐0813晚修: 无锚"离中线最近"会锁柜门框/邻物竖线(0813v4run锁错缝→真横向+42cm);
        锚点窗外(>0.15W)的缝一律拒收→交几何兜底。"""
        g = cv2.cvtColor(fpv_bgr, cv2.COLOR_BGR2GRAY)
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
        if lines is None:
            return None, None
        Hf, Wf = g.shape
        u_ref = Wf / 2 if u_anchor is None else u_anchor
        best = None
        for l in lines[:, 0]:
            x1, y1, x2, y2 = l
            L = float(np.hypot(x2 - x1, y2 - y1))
            if L < 0.22 * Hf:
                continue
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180     # 近竖直
            if abs(ang - 90) > 12:
                continue
            xm = (x1 + x2) / 2
            if abs(xm - u_ref) > 0.15 * Wf:                               # 锚点窗(原0.30W中线窗太宽=锁错线根因)
                continue
            score = abs(xm - u_ref) - 0.002 * L                           # 近锚点优先, 长度微调
            if best is None or score < best[0]:
                best = (score, xm)
        if best is None:
            return None, None
        return float(best[1] - Wf / 2), float(best[1])

    # ===== ⭐0812边缘线对齐(用户方案: 沿物体真实边缘——柜上边/显微镜下桌沿/白桌沿) =====
    def _edge_line(fpv_bgr, v_target=None):
        """LSD近水平线→目标v带内长度加权融合成一条边缘(治LSD碎段)。返回(斜率角°, 代表线段, 总长度px) or None。"""
        g = cv2.cvtColor(fpv_bgr, cv2.COLOR_BGR2GRAY)
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
        if lines is None:
            return None
        Hf, Wf = g.shape
        cand = []
        for l in lines[:, 0]:
            x1, y1, x2, y2 = map(float, l)
            L = float(np.hypot(x2 - x1, y2 - y1))
            if L < 0.08 * Wf:                                       # 0812实测站位最长边线仅~0.19W(125px)→原阈值0.25W全灭
                continue
            a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            a = ((a + 90) % 180) - 90                               # 归一化(-90,90]→近水平
            if abs(a) > 6:                                          # 先验拒绝: 真水平3D边缘斜率=sin(pitch12°)×yaw误差<0.21×err,
                continue                                            # |a|>6°⇔err>28°不可能是精修对象(透视对角线/溅射纹, 白桌振荡教训0812)
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            if abs(xm - Wf / 2) > 0.40 * Wf or not (0.10 * Hf < ym < 0.85 * Hf):
                continue
            cand.append((a, ym, L, (x1, y1, x2, y2)))
        if not cand:
            return None
        if v_target is not None:                                    # 目标v带内优先(±0.15H), 落空退回全部
            band = [c for c in cand if abs(c[1] - v_target) < 0.15 * Hf]
            if band:
                cand = band
        Lsum = sum(c[2] for c in cand)
        a_w = sum(c[0] * c[2] for c in cand) / Lsum                 # 长度加权斜率(平行线族共享斜率)
        seg = max(cand, key=lambda c: c[2])[3]                      # 代表线段=最长者
        return (float(a_w), seg, float(Lsum))

    def align_edge(name, max_iter=5):
        """⭐沿物体真实边缘线伺服yaw: 图像里水平3D边缘的斜率≈sin(pitch)×垂直偏差角→斜率归零=真垂直。
        front_deg只是粗朝向; 本闭环纯视觉(渲染图), 真机同构=同代码跑zed左目RGB, 零地图依赖。
        试探步自校准增益符号(防推反); 收敛后把FRONT[name]更新为实测朝向(图面正脸线与物体边缘重合,
        并量化报告 地图front vs 视觉front 的系统差)。返回最终斜率 or None。"""
        if name not in FRONT:
            return None
        gain = 4.5                                   # ≈1/|sin(pitch12°)|; 符号由试探自校准
        yaw0 = env.yaw; flips = 0                    # ⭐0813: 进站yaw+振荡计数(锁错线回滚用——白桌伺服翻号2次留15.4°front污染的教训)
        s_prev = None; seg_fin = None; s_hist = []
        for it in range(max_iter):
            fpv = np.ascontiguousarray(sim_fpv()[:, :, ::-1])
            Vv = capfpv._world_to_colmap_cam(env.x, env.y, env.yaw, roll_deg=0.0)
            _tp = np.array(T[name], dtype=np.float64)[None, :]
            _u, _v, _vis = _project_cluster(_tp, Vv)
            vt = float(_v[0]) if _vis[0] else None
            r = _edge_line(fpv, v_target=vt)
            if r is None:
                print(f"    [EDGE] {name} it{it}: 边缘线未检出, 保持front对准", flush=True)
                pipe["5"] = pipe["5"].split(" | edge")[0] + " | edge: no line"
                break
            a, seg, _sc_ = r
            seg_fin = seg
            print(f"    [EDGE] {name} it{it}: 边缘斜率{a:+.2f}°", flush=True)
            pipe["cur"] = "5"
            pipe["5"] = pipe["5"].split(" | edge")[0] + f" | edge it{it}: slope {a:+.2f}d"
            dwell(6)
            s_prev = a                                          # ⭐先存再判(0812 off-by-one: break时s_prev是上轮值→柜站漏更新FRONT)
            if abs(a) < 0.3:
                break
            if s_hist and abs(a) > abs(s_hist[-1]) + 0.05:
                gain = -gain; flips += 1              # 变糟→方向反了, 翻号
                print(f"    [EDGE] {name}: 斜率变大→增益翻号 g={gain:+.1f}", flush=True)
            s_hist.append(a)
            dyaw = float(np.clip(np.radians(gain * a), -np.radians(6), np.radians(6)))
            tgt = env.yaw + dyaw
            for _ in range(30):                                  # 小步转(边转边emit)
                e = wrap(tgt - env.yaw)
                if abs(e) < 0.005:
                    break
                wz = float(np.clip(2.0 * e, -0.5, 0.5))
                log["_vx"], log["_vy"], log["_wz"] = 0.0, 0.0, wz
                log["brain"] = f"EDGE-ALIGN slope={a:+.1f}d"
                obs, _, _ = env.step(0.0, 0.0, wz); emit(obs)
        _dyaw = abs(float(np.degrees(wrap(env.yaw - yaw0))))
        # ⭐0813晚修: 收敛还不够——视觉front与地图front差>2°=锁错线/碎线族融合偏(渲染溅射区LSD不可信,
        # 0813v4run三站被伺服带偏-5.0/-6.4/-8.9°用户判"都没对准")→按未收敛处理回滚地图front(0731真图LSD+用户验证过)
        if s_prev is not None and abs(s_prev) < 0.5 and flips < 2 and _dyaw <= 2.0:
            old = FRONT[name]
            FRONT[name] = float(np.degrees(wrap(env.yaw)))          # 视觉实测front覆盖地图值(本run内; wrap到±180防-196°丑显)
            print(f"    [EDGE] {name}: 收敛(slope{s_prev:+.2f}°) front {old:.1f}°→视觉实测{FRONT[name]:.1f}° "
                  f"(地图-视觉差{np.degrees(wrap(np.radians(FRONT[name]-old))):+.1f}°)", flush=True)
        elif _dyaw > 0.5:
            # ⭐0813: 未收敛(锁错线/振荡翻号≥2/线丢失/视觉-地图差>2°)=不可信 → yaw回滚进站值, 防污染后续dock朝向
            print(f"    [EDGE] {name}: 未收敛(slope{s_prev if s_prev is None else f'{s_prev:+.2f}'}°, 翻号{flips}次, "
                  f"视觉-地图差{_dyaw:.1f}°) → yaw回滚 {np.degrees(wrap(env.yaw - yaw0)):+.1f}°", flush=True)
            pipe["5"] = pipe["5"].split(" | edge")[0] + " | edge: NOT converged, yaw rollback"
            for _ in range(40):
                e = wrap(yaw0 - env.yaw)
                if abs(e) < 0.005:
                    break
                wz = float(np.clip(2.0 * e, -0.5, 0.5))
                log["_vx"], log["_vy"], log["_wz"] = 0.0, 0.0, wz
                log["brain"] = "EDGE-ALIGN rollback (not converged)"
                obs, _, _ = env.step(0.0, 0.0, wz); emit(obs)
        _aim["edge"] = seg_fin                                  # panel叠加显示检测到的边缘线
        dwell(10)
        return s_prev

    def precise_dock(name, target_dist=0.15, trust_dist=0.35):
        """0812v2(v1实测穿模: ED-p10过读玻璃/溅射→多走~18cm, 最近距节点仅2cm):
        0)站位先居中(该距离簇/缝在FOV内; 0.35m处台面簇在-12°俯角FOV下缘之外=无簇根因)
        1)伺服到可信区trust_dist: 面距=min(ED-p10, 几何dist-FACE_OFF)——ED过读取几何兜底, ED欠读(溅射floater)取ED=安全方向; 跳涨>6cm拒信
        2)0.35m处再居中 3)盲爬到target 4)贴脸精居中一次 5)指标落盘。"""
        log["state"] = "PRECISE-DOCK"
        tx, ty = float(T[name][0]), float(T[name][1])
        FACE_OFF = 0.05                                    # 面在节点前方~5cm(0811 s-z剖面实测, 柜; 他站同量级)

        def _geo_face_dist():
            """几何面距: believed位姿(RELOC残差~0.5cm可信)到节点距离-FACE_OFF。真机同构=地图+believed+zed。"""
            return float(np.hypot(env.x - tx, env.y - ty)) - FACE_OFF

        # ---- 居中伺服(横移为主, 保持front yaw); 抽成闭包, 站位/0.35m/贴脸三处复用 ----
        _fn = np.radians(FRONT.get(name, 0.0)); _nx, _ny = -np.sin(_fn), np.cos(_fn)   # 面轴横向(左+)
        def _center(max_it, step_cap, fd_bel):
            """⭐0813晚修: 视觉特征(缝/簇)一律先用节点投影u_node锚定, 出窗(缝0.15W/簇0.20W)→几何横向兜底
            (env位姿+front轴, RELOC后残差~0.5cm; 真机同构=believed)。0813v4run无锚缝锁错→真横向+42cm的教训。
            ⭐0813深夜修2: 信号源整段粘性裁决——it0比较视觉off与几何off, 矛盾>0.10W则整段只信几何;
            否则逐帧"缝进窗用缝/出窗用几何"会在两信号间极限环(柜站左右横跳6-8次=用户看到的左右抖动)。"""
            off_px_fin, lat_fin = None, None
            use_vis = None                                                # 整段粘性: it0裁决一次
            for it in range(max_it):
                fpv = np.ascontiguousarray(sim_fpv()[:, :, ::-1])
                V = capfpv._world_to_colmap_cam(env.x, env.y, env.yaw, roll_deg=0.0)
                _un, _vn, _visn = _project_cluster(np.array([[tx, ty, capfpv.cam_height]], dtype=np.float64), V)
                u_node = float(_un[0])                                        # 节点(目标中心)投影x=锚点
                lat_geo = (env.x - tx) * _nx + (env.y - ty) * _ny
                geo_off_px = lat_geo * _fxs / max(fd_bel, 0.2)
                off_vis, src_vis = None, ""
                if name == "iron_cabinet":
                    off_vis, u_seam = _seam_offset_px(fpv, u_anchor=u_node)
                    src_vis = f"seam_x={u_seam:.0f}" if u_seam else "no seam(锚窗外)"
                else:
                    pts = _clusters.get(name)
                    src_vis = "no cluster"
                    if pts is not None and len(pts) > 50:
                        u, v, vis = _project_cluster(pts, V)
                        if vis.sum() > 30:
                            u_c = float(u[vis].mean())
                            if abs(u_c - u_node) <= 0.20 * capfpv.W:          # 簇质心锚窗(FOV裁切偏簇拒收)
                                off_vis = u_c - capfpv.W / 2; src_vis = f"centroid({vis.sum()}pt)"
                            else:
                                src_vis = f"centroid出锚窗({u_c - u_node:+.0f}px)"
                if use_vis is None:                                           # it0粘性裁决
                    use_vis = off_vis is not None and abs(off_vis - geo_off_px) <= 0.10 * capfpv.W
                    if off_vis is not None and not use_vis:
                        print(f"    [DOCK] {name}: 视觉{off_vis:+.0f}px vs 几何{geo_off_px:+.0f}px 矛盾>"
                              f"{int(0.10 * capfpv.W)}px → 本段只信几何", flush=True)
                if use_vis:
                    off_px, src = off_vis, src_vis
                else:
                    off_px, src = geo_off_px, src_vis + "→geo"
                lat_m = off_px / _fxs * fd_bel                            # px→横向米(该深度处)
                off_px_fin, lat_fin = off_px, lat_m
                print(f"    [DOCK] {name} 居中it{it}: {src} 偏{off_px:+.0f}px≈{lat_m*100:+.1f}cm", flush=True)
                log["brain"] = f"DOCK center {off_px:+.0f}px≈{lat_m*100:+.1f}cm"
                pipe["7"] = f"center it{it}: {src[:14]} off {lat_m*100:+.1f}cm"
                if abs(lat_m) < 0.008:                                    # <8mm收敛
                    break
                step = float(np.clip(-lat_m, -step_cap, step_cap))        # lat>0=左; 缝在右(off>0)→右移
                _dock_step(lat=step, n=max(1, int(abs(step) / 0.01)))
            return off_px_fin, lat_fin

        # ---- 0) 站位先居中(簇/缝在此距离可见) ----
        pipe["cur"] = "7"
        pipe["7"] = "phase0: center at standoff (seam/cluster in FOV here)"
        off_px_fin, lat_fin = _center(6, 0.03, max(_geo_face_dist(), trust_dist))
        pipe["7"] = f"phase0 done: lateral {None if lat_fin is None else round(lat_fin*100,1)}cm | servo to 35cm..."
        dwell(8)

        # ---- 1) 深度伺服到可信区: min(ED-p10, 几何) ----
        fd = None; fd_prev = None
        for it in range(12):
            fd_ed = _ed_face_depth()
            fd_geo = _geo_face_dist()
            if fd_ed is None:
                fd_new, src = fd_geo, "GEO"
            elif fd_ed <= fd_geo:
                fd_new, src = fd_ed, "EDp10"
            else:
                fd_new, src = fd_geo, f"GEO<ED{fd_ed*100:.0f}"            # ED过读(看穿玻璃/溅射)→几何下限
            if fd_prev is not None and fd_new > fd_prev + 0.06:
                print(f"    [DOCK] {name} 接近it{it}: 读数跳涨{fd_prev*100:.0f}→{fd_new*100:.0f}cm, 拒信", flush=True)
                fd_new, src = fd_prev, "KEEP"
            fd = fd_new
            err = fd - trust_dist
            print(f"    [DOCK] {name} 接近it{it}: 面距{fd*100:.1f}cm[{src}] (先到{trust_dist*100:.0f}cm可信区)", flush=True)
            log["brain"] = f"DOCK dist={fd*100:.0f}cm→{trust_dist*100:.0f}cm[{src}]"
            pipe["7"] = f"servo it{it}: face {fd*100:.0f}cm [{src}] -> {trust_dist*100:.0f}cm"
            if abs(err) < 0.01:
                break
            step = float(np.clip(0.65 * err, -0.05, 0.05))               # 步进≤5cm
            _dock_step(fwd=step, n=max(1, int(abs(step) / 0.02)))
            fd_prev = fd
        if fd is None:
            fd = _geo_face_dist()
        dwell(8)

        # ---- 2) 可信区再居中 ----
        pipe["7"] = f"phase2: re-center at {fd*100:.0f}cm"
        off2, lat2 = _center(8, 0.03, max(fd, trust_dist))
        if off2 is not None:
            off_px_fin, lat_fin = off2, lat2
        dwell(6)

        # ---- 3) 盲爬到target(近场ED不可信: <0.4m p10膨胀/玻璃看穿) ----
        crawl = fd - target_dist
        if crawl > 0.005:
            print(f"    [DOCK] {name} 盲爬{crawl*100:.0f}cm → 面距{target_dist*100:.0f}cm", flush=True)
            log["brain"] = f"DOCK blind-crawl {crawl*100:.0f}cm→{target_dist*100:.0f}cm"
            pipe["7"] = f"blind-crawl {crawl*100:.0f}cm -> {target_dist*100:.0f}cm (near-field ED untrusted)"
            _dock_step(fwd=0.02, n=int(round(crawl / 0.02)))
        fd_bel = target_dist                                             # believed面距(近场ED不读)
        dwell(6)

        # ---- 4) 贴脸精居中一次(缝隙此时最清晰; 小步封顶1.5cm防过度) ----
        pipe["7"] = "phase4: final fine-center at 15cm"
        off2, lat2 = _center(3, 0.015, fd_bel)
        if off2 is not None:
            off_px_fin, lat_fin = off2, lat2

        # ---- 5) 终指标 ----
        ferr = abs(float(np.degrees(wrap(np.radians(FRONT[name]) - env.yaw)))) if name in FRONT else None
        _lat_geo_end = float((env.x - tx) * _nx + (env.y - ty) * _ny)
        _dist_end = float(np.hypot(env.x - tx, env.y - ty)) - FACE_OFF
        rec = {"station": name,
               "face_dist_cm": round(fd_bel * 100, 1),
               "geo_face_dist_cm": round(_dist_end * 100, 1),              # ⭐0813晚: 几何面距(trace复核用, believed可说谎)
               "geo_lateral_cm": round(_lat_geo_end * 100, 1),             # ⭐0813晚: 几何横向(同上)
               "seam_off_px": None if off_px_fin is None else round(off_px_fin, 1),
               "lateral_cm": None if lat_fin is None else round(lat_fin * 100, 1),
               "front_err_deg": None if ferr is None else round(ferr, 2)}
        _dock_stats.append(rec)
        pipe["7"] = f"DOCKED: face~{fd_bel*100:.0f}cm lat{rec['lateral_cm']}cm front{rec['front_err_deg']}d"
        dwell(16)
        print(f"    [DOCK] {name} 终: 面距{rec['face_dist_cm']}cm(believed) 横向{rec['lateral_cm']}cm "
              f"({rec['seam_off_px']}px) front偏{rec['front_err_deg']}°", flush=True)
        return rec

    def undock(name, back_to=0.6, from_dist=0.15):
        """对接完撤离到面距back_to(米)——0812改纯盲退(近场ED不可信, 0811版靠ED会误判早停)。"""
        log["state"] = "UNDOCK"
        n = int(round((back_to - from_dist) / 0.05))
        print(f"    [UNDOCK] {name} 盲退{n*5}cm → 面距~{back_to*100:.0f}cm", flush=True)
        for _ in range(n):
            log["brain"] = f"UNDOCK →{back_to*100:.0f}cm"
            _dock_step(fwd=-0.05, n=1)

    # ⭐0730已删: _vlm_bbox/_turn_to/servo_approach(bbox视觉伺服三件套)——
    # 场景图RAG就是替换这套手搓几何的, 0729捡回来=走回头路(假confirm+贴脸+歪)。对准=face()+graph_fine_aim。

    _srcframes = None   # 已废弃: 源帧confirm改为实时水平渲染图判断式(0730掰回RAG原生, bbox全链路剔除)
    def vlm_ask(img_bgr, prompt, max_new_tokens=6):
        try:
            ans, _ = sf.vlm_api(img_bgr, prompt, max_new_tokens=max_new_tokens, timeout=40)
            return (ans or "").strip()
        except Exception:
            return ""
    def vlm_confirm(name):
        """⭐⑩四段推理链慢脑confirm(论文§4.3结构化推理, 周报清单⑩) + ⑪判no时key-frame复核。
        四段: History(任务上下文, 我方已知) / Scene(VLM短描述画面) / Progress(几何量: 距离+front差, 我方算)
              / Action(VLM严格yes/no)——分段独立不串联(弱VLM喂回自己的话会糊, 0716/0727教训),
              段段落盘→出错能定位是哪段错(治白桌单问太脆反复翻车)。Scene说看到目标而Action判no=矛盾也会记。
        ⚠不再用bbox存在性(0729假confirm根因); Action判no不立刻结案→⑪回查接近过程top3证据帧,
        ≥2帧yes才翻案(单帧yes不翻=防侥幸), 全no则如实NOT-CONFIRMED并附证据链。"""
        tx, ty = float(T[name][0]), float(T[name][1])
        dist = float(np.hypot(env.x - tx, env.y - ty))
        berr = float(np.degrees(wrap(np.arctan2(ty - env.y, tx - env.x) - env.yaw)))
        ferr = float(np.degrees(wrap(np.radians(FRONT[name]) - env.yaw))) if name in FRONT else None
        # --- History(段1, 已知上下文) ---
        hist = {"leg": log["leg"], "dist_m": round(dist, 2), "berr_deg": round(berr, 1),
                "front_err_deg": None if ferr is None else round(ferr, 1)}
        print(f"    [慢脑⑩] History: {hist}", flush=True)
        pipe["cur"] = "6"
        pipe["6"] = f"chain: d={dist:.2f}m berr={berr:+.1f}d front_err={None if ferr is None else round(ferr,1)}d | asking Scene..."
        dwell(8)
        # --- Scene(段2, VLM短描述; 具体短答=Qwen甜区) ---
        img = np.ascontiguousarray(sim_fpv()[:, :, ::-1])
        scene = vlm_ask(img, "What is the main object in the center of this image? Answer in a few words.",
                        max_new_tokens=16)[:80]
        print(f"    [慢脑⑩] Scene: {scene or '(err)'}", flush=True)
        pipe["6"] = f"Scene='{(scene or '(err)')[:30]}' | Progress: arrived={dist<1.3} facing={ferr is not None and abs(ferr)<3.0} | asking Action..."
        dwell(8)
        # --- Progress(段3, 几何量, 不用VLM) ---
        prog = {"arrived": dist < 1.3, "facing_front": (ferr is not None and abs(ferr) < 3.0)}
        print(f"    [慢脑⑩] Progress: {prog}", flush=True)
        # --- Action(段4, 严格yes/no, 与0802同prompt同判据) ---
        ans = vlm_ask(img, f"Is there {_term(name)} in this image? Answer yes or no.")
        ok = ans.lower().startswith("y")
        rec = {"type": "confirm", "station": name, "history": hist, "scene": scene,
               "progress": prog, "action_raw": ans[:40], "action_ok": ok, "review": []}
        print(f"    [慢脑⑩] Action: {ans or '(err)'} -> {'YES' if ok else 'NO'}", flush=True)
        pipe["6"] = f"Scene='{(scene or '(err)')[:24]}' Action={'YES' if ok else 'NO'} | primary verdict"
        dwell(10)
        decided = "primary"
        # --- ⭐⑪ 判no→key-frame复核(接近过程top3证据帧, ≥2帧yes才翻案) ---
        if not ok and not args.no_review:
            picks = kf_topk(3)
            nyes = 0
            for fi, fimg, fd, fb in picks:
                rans = vlm_ask(fimg, f"Is there {_term(name)} in this image? Answer yes or no.")
                rok = rans.lower().startswith("y"); nyes += rok
                rec["review"].append({"frame": fi, "dist_m": fd, "berr_deg": fb,
                                      "ans": rans[:40], "ok": rok})
                print(f"    [慢脑⑪] 复核f{fi}(d={fd}m,berr={fb}°): {rans or '(err)'} -> {'YES' if rok else 'NO'}", flush=True)
                pipe["6"] = f"Action=NO -> keyframe review f{fi}(d{fd}m): {'YES' if rok else 'NO'} ({nyes}yes so far)"
                dwell(8)
            if nyes >= 2:
                ok = True; decided = f"review({nyes}/{len(picks)})"
            else:
                decided = f"review-fail({nyes}/{len(picks)})"
        rec["final"] = "CONFIRMED" if ok else "NOT-CONFIRMED"
        rec["decided_by"] = decided
        # ⑩审计点: Scene自称看到目标类而Action/复核全否=矛盾记录(白桌病例的病灶定位)
        tkey = _term(name).split()[-1] if _term(name).split() else ""
        rec["scene_action_conflict"] = bool(scene) and (tkey.lower() in scene.lower()) and not ok
        sb_record(rec)
        log["vlm"] = f"{_label(name)}? -> {'YES' if ok else 'NOT-CONFIRMED'}({decided})"
        pipe["6"] = f"VERDICT {_label(name)}: {'YES' if ok else 'NOT-CONFIRMED'} ({decided}) Scene='{(scene or '')[:18]}'"
        dwell(14)
        print(f"    [慢脑] confirm {name}: {log['vlm']}" +
              (f" ⚠Scene/Action矛盾!" if rec["scene_action_conflict"] else ""), flush=True)
        return ok

    def wait(sec=2.0):
        log["state"] = "WAIT"; log["brain"] = "WAIT (holding still)"    # ⭐刷新brain标签, 别显示上一条turn的陈旧e
        log["_vx"], log["_vy"], log["_wz"] = 0.0, 0.0, 0.0
        for _ in range(int(sec * args.fps)):
            emit(env._obs())

    # ===== ⭐⑨搜索行为demo(论文§4.1.3落地证据): 无目标纯快脑探索, 死胡同应自己回退 =====
    if args.search_demo > 0:
        out_search = os.path.join(ROOT, "outputs", f"lab_search_demo_{_ck_tag}_0804.mp4")   # 0804=修hist_update漏调+turn-then-go版
        writer.close()
        writer = imageio.get_writer(out_search, fps=args.fps, codec="libx264",
                                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        env.start = (0.0, 0.0, 0.0); obs = env.reset()
        cmd = NavCmd("SEARCH", "(none)", (env.x, env.y), (env.x, env.y), (8, 0.9, 0.45))   # ⑧搜索参数包: 长窗高γ=覆盖式历史
        print(f"  [⑧] {cmd}", flush=True); sb_record(cmd.as_dict())
        log["leg"] = "SEARCH(mode=1)"; log["state"] = "SEARCH"
        log["brain"] = "SEARCH Φ=(W8,γ0.9)"
        for i in range(args.search_demo):
            pose = (env.x, env.y, env.yaw)
            fpv = np.ascontiguousarray(sim_fpv()[:, :, ::-1])
            hist_update()                                           # ⭐每步入队历史位姿(0803首跑漏了=W8γ0.9状态+单帧BEV离分布, 30步就楔死)
            traj = dd_traj(fpv, pose, np.zeros(2, np.float32), head_ego=np.zeros(2, np.float32),
                           hist_w=8, hist_gamma=0.9, vx=0.45, mode=1.0)     # ⭐goal/heading全零+mode=1=无目标探索
            # 跟DD自己的搜索轨迹: ⭐turn-then-go(训练数据yaw恒朝行走方向, 折返GT=先原地转回去再走,
            # 旧控制器只vx>0+比例wz→回退轨迹被译成贴墙打转); 前视点(1m)定目标朝向
            look = traj[min(3, len(traj) - 1), :2]
            head_err = float(np.arctan2(look[1], look[0]))
            if abs(head_err) > 0.6:                                 # 前视点在背后/大角度: 先原地转(执行折返)
                vx, wz = 0.0, float(np.clip(1.2 * head_err, -0.6, 0.6))
            else:
                wz = float(np.clip(1.5 * head_err, -0.5, 0.5))
                vx = float(np.clip(np.linalg.norm(traj[-1, :2]) / 4.0, 0.0, 0.45))   # 无0.10下限(楔死时要能停)
            vx, vy, wz, esc = safe_vel(env.x, env.y, env.yaw, vx, 0.0, wz)   # 间隙场安全层(贴障→减速+朝空方向转)
            log["_vx"], log["_vy"], log["_wz"] = vx, vy, wz
            log["brain"] = f"SEARCH v={vx:.2f} w={wz:.2f}{' ESC' if esc else ''} [{log.get('_hist','')}]"
            obs, _, _ = env.step(vx, vy, wz); emit(obs)
            if i % 60 == 0:
                print(f"  [search] step{i} pos=({env.x:.2f},{env.y:.2f}) yaw={np.degrees(env.yaw):.0f}°", flush=True)
        writer.close()
        print(f"[mission] ✅ {out_search} ({frames_written[0]} frames)", flush=True)
        return

    # ===== 慢脑VLM编排的任务流程 =====
    if os.path.exists(_sb_log_path):
        os.remove(_sb_log_path)                                  # 本次任务新审计链(不与旧run混)
    env.start = (0.0, 0.0, 0.0)   # ⭐从【原位】出发! (上面各腿规划循环的最后一次plan把env.start改成了移液枪站位→必须复位, 否则首帧在漂移区左扭)
    obs = env.reset()
    # ⭐0812片头卡: 完整实际流程一览(诚实标注: 漂移是注入演示/sim观测是伪zed)
    flow_card("ACTUAL PIPELINE (no mock)", [
        "1 WHERE : scene-graph node (ConceptGraphs, 185 nodes)",
        "          tau = node xy + front_deg (LSD on real imgs)",
        "2 PLAN  : A* on occupancy from GS skin",
        "3 RELOC : regress-head 1-fwd (<10ms) / GS optim fallback",
        "          sim obs = renderED/0.70 + zed-like noise",
        "          drift shown = INJECTED (demo of correction)",
        "          real-zed eval: 5.6cm median (independent)",
        "4 FASTDD: DiffusionDrive, NavCmd(L,tau,Phi) in-loop",
        "5 ALIGN : front-yaw turn + lateral on-axis closed loop",
        "6 VLM   : 4-part chain (Hist/Scene/Prog/Action)",
        "          + keyframe review if primary says NO",
        "7 DOCK  : min(ED-p10, geo) servo 35cm, re-center,",
        "          blind crawl to 15cm, fine-center",
        "8 METRIC: final errors measured in TRUE pose",
        "",
        "world scale = 0.70x real (cam_height issue, patched)",
    ], n=60)
    for leg_i, (mode, a, b, tag) in enumerate(legs):
        name = _leg_name(tag)
        pipe["legs"][leg_i] = (pipe["legs"][leg_i][0], ">")
        if leg_i > 0:
            pipe["legs"][leg_i - 1] = (pipe["legs"][leg_i - 1][0], "x")
        log["leg"] = _label(name)
        log["vlm"] = ""                                             # ⭐清空上一站confirm标签(防跨站残留误导)
        _aim["name"] = name if name in so else None                 # panel向量标注目标
        _aim["edge"] = None                                         # 上一站边缘线标注清除
        # ⭐⑧ 本腿结构化参数包(nav(L,τ,Φ))——慢脑→快脑的唯一接口, 落盘可审计
        cmd = NavCmd("CRUISE" if mode == "A*" else "STRAFE", name, b, (env.x, env.y),
                     PHI["CRUISE" if mode == "A*" else "STRAFE"])
        print(f"  [⑧] {cmd}", flush=True)
        sb_record(cmd.as_dict())
        log["state"] = "CRUISE" if mode == "A*" else "STRAFE"
        fr = FRONT.get(name)
        pipe["cur"] = "1"
        if name in so:
            pipe["1"] = f"tau={name}@({T[name][0]:.2f},{T[name][1]:.2f}) front={fr if fr is not None else '-'}d -> standoff({b[0]:.2f},{b[1]:.2f})"
        else:
            pipe["1"] = f"tau={name} -> ({b[0]:.2f},{b[1]:.2f})"
        dwell(12)
        kf_set_target(name if name in so else None)                 # ⭐⑪本腿证据帧登记(站位腿才需要复核)
        p = plan(env.x, env.y, b[0], b[1])                          # A*粗路径(仅前视点)
        _sg["path"] = p if (p is not None and len(p) > 1) else None   # 场景图面板: 当前路径+站位
        _sg["so"] = b
        pipe["cur"] = "2"
        pipe["2"] = (f"A* {len(p)}pts len{float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))):.1f}m"
                     if p is not None and len(p) > 1 else "STRAFE direct (short leg)")
        dwell(10)
        if p is not None and len(p) > 1:
            # ⭐站位腿用小到达阈值(真开到head-on正前方站位, 不停在0.4m前=斜视); 回原位用大阈值(防绕origin转圈)
            obs = cruise(p, tol=(0.15 if name in so else 0.40),
                         phi=(cmd.W, cmd.gamma, cmd.vx))            # ⭐快脑生成运动(Φ=⑧参数包)
        if name in so:                                              # 慢脑: 对准+确认+等待
            log["state"] = "TURN/FACE"
            _drift0 = _res0 = None
            if _reloc is not None:                                  # ⭐RELOC闭环: 漂移注入→重定位纠偏
                # ⭐0813: RELOC提到切站位俯角之前——回归头训练分布=采集真实俯仰(-20~-29°)+巡航-22°,
                # 站位-12°俯角观测离分布会拒收/失效; RELOC吃深度几何不需要正视, 巡航俯角下先做。
                if not _odom["on"]:                                 # 周期纠偏模式下漂移已在巡航中累积, 不再注入
                    import zlib
                    _rngd = np.random.RandomState(zlib.crc32(name.encode()) & 0xffff)
                    _rr = args.drift_m * (0.6 + 0.4 * _rngd.rand()); _aa = _rngd.uniform(0, 2 * np.pi)
                    _drift[:] = [_rr * np.cos(_aa), _rr * np.sin(_aa),
                                 np.radians(_rngd.uniform(4, 4 + args.drift_m * 40) * _rngd.choice([-1, 1]))]
                    print(f"    [RELOC] {name}: 注入漂移 {np.hypot(_drift[0],_drift[1]):.2f}m/{abs(np.degrees(_drift[2])):.0f}°", flush=True)
                _drift0, _res0 = reloc_correct(name)
            capfpv.pitch = np.radians(args.pitch_station)           # ⭐到站换正视俯角(消keystone倾斜观感)
            obs = align_front(name)                                 # ⭐正对闭环: front yaw转向+横移上轴(零bbox)
            if _reloc is not None:
                station_metric(name, _drift0, _res0)                # 真位姿测终docking误差
                dwell(12)
                _drift[:] = 0.0                                     # 下一站重新注入(腿间巡航仍用真位姿)
            align_edge(name)                                        # ⭐0812边缘线视觉精对齐(柜顶/桌沿斜率→0)
            if args.fine_aim:
                graph_fine_aim(name)                                # 可选: 节点点簇投影居中(簇过合并时别开)
            vlm_confirm(name)                                       # ⭐⑩四段推理链+⑪key-frame复核(0803论文项)
            kf_set_target(None)                                     # 本腿证据冻结(复核已取)
            if args.precise_dock:                                   # ⭐贴脸精对接(0811): 距面15cm+居中<1cm
                precise_dock(name)
            wait(2.0)                                               # 等待一段
            if args.precise_dock:
                undock(name)                                        # 撤离到0.6m(下一腿A*起点要在free区)
            capfpv.pitch = np.radians(args.pitch_cruise)            # 出站台恢复巡航俯角(快脑分布)
    pipe["legs"][-1] = (pipe["legs"][-1][0], "x")
    # ⭐0812片尾卡: 精度汇总(全部真位姿口径)
    _oc = ["RESULTS (true-pose metrics):"]
    for s_ in _reloc_stats:
        _oc.append(f"  {s_['station']:<14s} drift-inj {s_['drift_m']*100:.0f}cm -> "
                   f"reloc res {s_['reloc_res_cm']:.1f}cm | final lat {s_['final_lat_m']*100:.1f}cm "
                   f"front {s_['final_front_deg']:.1f}d")
    for d_ in _dock_stats:
        _oc.append(f"  {d_['station']:<14s} DOCK face~{d_['face_dist_cm']}cm "
                   f"lat {d_['lateral_cm']}cm front {d_['front_err_deg']}d")
    if _odom["series"]:
        _oc.append(f"  en-route reloc x{len(_odom['series'])}: res median "
                   f"{float(np.median([s_['res_cm'] for s_ in _odom['series']])):.1f}cm")
    _oc.append("")
    _oc.append("caveat: sim closed-loop residuals are self-consistent;")
    _oc.append("independent real-zed eval = 5.6cm median horizontal.")
    flow_card("SUMMARY", _oc, n=75)
    writer.close()
    if _reloc_stats or _odom["series"] or _dock_stats:                 # ⭐RELOC/对接精度落盘
        _rj = os.path.join(ROOT, "outputs", f"{_mipre}_reloc{'_odom' if args.reloc_odom else ''}{'_precdock' if args.precise_dock else ''}{'_head' if args.reloc_head else ''}{'_flow' if args.flow else ''}_0813.json")
        payload = {"stations": _reloc_stats, "odom_series": _odom["series"], "dock": _dock_stats}
        with open(_rj, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        if _reloc_stats:
            lat = np.array([r["final_lat_m"] for r in _reloc_stats]); brg = np.array([r["final_bearing_deg"] for r in _reloc_stats])
            drf = np.array([r["drift_m"] for r in _reloc_stats])
            print(f"[mission] RELOC汇总: 漂移中位{np.median(drf)*100:.0f}cm → 终横向误差中位{np.median(lat)*100:.1f}cm "
                  f"bearing中位{np.median(brg):.1f}°", flush=True)
        if _odom["series"]:
            d0 = np.array([s["drift_m"] for s in _odom["series"]]); rs = np.array([s["res_cm"] for s in _odom["series"]])
            print(f"[mission] 周期纠偏: {len(d0)}次 无纠正漂移中位{np.median(d0)*100:.0f}cm(峰值{d0.max()*100:.0f}cm) "
                  f"→ 纠正后残差中位{np.median(rs):.1f}cm", flush=True)
        print(f"[mission] RELOC数据 -> {_rj}", flush=True)
    # ⭐调试: 每帧位姿+速度落盘, 供帧级诊断(抖/圈/斜)
    try:
        _tr = np.array([(t[0], t[1], t[2], t[3], t[4], t[5]) for t in trace], dtype=np.float32)
        _st = np.array([t[6] for t in trace])
        np.savez(os.path.join(ROOT, "outputs", f"{_mipre}_trace_{_tag}.npz"),
                 xyyaw_vxvywz=_tr, state=_st,
                 targets=np.array([T[n][:2] for n in _STATIONS], np.float32),
                 standoffs=np.array([so[n][:2] for n in _STATIONS], np.float32))
        print(f"[mission] trace -> outputs/{_mipre}_trace_{_tag}.npz", flush=True)
    except Exception as e:
        print("trace save err", e, flush=True)
    # 轨迹图: 机器人实际走过的路 + 目标 + 站位
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, trimesh
        tr = np.array([(t[0], t[1]) for t in trace])
        pc = trimesh.load(os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply"), process=False)
        P = np.asarray(pc.vertices); mm = (P[:, 2] > 0.1) & (P[:, 2] < 2.0); Q = P[mm]
        if len(Q) > 120000: Q = Q[np.random.choice(len(Q), 120000, replace=False)]
        fig, ax = plt.subplots(figsize=(9, 10))
        ax.scatter(Q[:, 0], Q[:, 1], c='lightgray', s=0.5, alpha=0.4)
        ax.plot(tr[:, 0], tr[:, 1], '-', c='blue', lw=2, alpha=0.7, label='robot path')
        ax.scatter([0], [0], c='lime', s=160, marker='s', zorder=6, label='origin')
        _cyc = ["red", "magenta", "orange", "cyan", "yellow", "lime"]
        cols = {n: _cyc[i % len(_cyc)] for i, n in enumerate(_STATIONS)}
        for name in cols:
            ax.scatter([T[name][0]], [T[name][1]], c=cols[name], s=240, marker='*', edgecolor='k', zorder=6, label=name)
            ax.scatter([so[name][0]], [so[name][1]], c=cols[name], s=90, marker='^', zorder=5)
        ax.set_aspect('equal'); ax.legend(fontsize=8); ax.set_title("robot actual path (line) + targets(*) + standoffs(^)")
        plt.tight_layout(); plt.savefig(os.path.join(ROOT, "outputs", "recon_verify", "查看目标", "D_机器人实际轨迹.png"), dpi=110)
        moved = float(np.sum(np.linalg.norm(np.diff(tr, axis=0), axis=1)))
        print(f"[mission] robot traveled {moved:.2f}m total", flush=True)
    except Exception as e:
        print("traj plot err", e, flush=True)
    print(f"[mission] ✅ {out}  ({frames_written[0]} frames @ {args.fps}fps)", flush=True)


if __name__ == "__main__":
    main()
