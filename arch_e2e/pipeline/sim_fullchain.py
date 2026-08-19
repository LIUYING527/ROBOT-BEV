"""活仿真全链闭环(双脑真跑, 不是回放): CorridorTestEnv 渲真实FPV →
   慢脑(我们自研 VLM, 走 HTTP API 10.0.0.18:8152)读图出语义 → answer→cond →
   快脑 DiffusionDrive(条件微调版)出轨迹 → 轨迹转(v,ω)真驱动机器人 → 循环。

⭐单进程跑在 discoverse_venv: 渲染(mujoco/3DGS)+ 快脑(本地 torch)+ 慢脑(requests 发 API)。
慢脑走远程 → 本地不需要 transformers, 一个进程一体闭环, 也正是未来"慢脑走 API"的架构。

产物(arch_e2e/runs/):
  sim_fullchain_<session>.mp4    左FPV(叠预测轨迹) | 右BEV占据窗(叠轨迹)
  sim_fullchain_<session>.jsonl  每帧实时日志: 慢脑输入prompt/输出answer→cond, 快脑cond→轨迹, (v,ω), 碰撞/到达

用法(卡7):
  CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=7 PYOPENGL_PLATFORM=egl \
    PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python arch_e2e/pipeline/sim_fullchain.py --session colmapjoint_all --steps 240
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, json, time, argparse
import numpy as np
import cv2
import requests
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # robot_bev_sim
sys.path.insert(0, os.path.join(ROOT, "scripts"))            # test_env, sim_walk_common
sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))  # stage_b_eval(_paths), train_conditional, gen_conditional_data

from test_env import CorridorTestEnv                          # 活仿真环境(渲FPV+占据碰撞+到达判定)
import stage_b_eval as sb                                     # 快脑: load_model / 特征
from train_conditional import robot_anchors                   # 机器人尺度锚点
from gen_conditional_data import project_ground, ego_to_world # FPV 地面投影

RUNS = os.path.join(ROOT, "arch_e2e", "runs")
API_URL = os.environ.get("VLM_URL", "http://10.0.0.18:8152/api/generate")   # 本地Qwen3.5服务时设 VLM_URL=http://127.0.0.1:8152/api/generate

# 4 语义条件(与训练一致); cond_id = index
CONDITIONS = ["cruise", "inspect_left", "inspect_right", "strafe_left", "strafe_right",
              "turn_left", "turn_right", "pause", "stop"]
COND_COLOR = {"cruise": (0, 200, 255), "inspect_left": (0, 255, 0), "inspect_right": (255, 128, 0),
              "strafe_left": (0, 255, 180), "strafe_right": (255, 0, 180),
              "turn_left": (180, 255, 0), "turn_right": (180, 0, 255),
              "pause": (150, 150, 150), "stop": (0, 0, 255)}  # BGR

# 默认 mission(可 --mission 覆盖)
MISSION_DEFAULT = ("Patrol down the corridor and keep moving; veer toward a device only when one is "
                   "clearly beside you to inspect. Do not stop unless the path is physically blocked.")

# ⭐慢脑=场景+任务推理器: 看多帧FPV, 出 WHERE(target)+HOW(action, 全9词表) 结构化JSON。
# ⚠️禁止按画面糊/暗判断(3DGS偏软会误触发, 历史踩过"blur→pause"坑); 只看场景内容。
def slow_prompt(mission):
    """⚠️实测(探针): 这个微调VLM只在【简单3选一方向题】上给随图变化的答案;
    选全9词表/复杂选项/带括号解释都会 collapse 到单一(pause/inspect_left)。
    故 prompt=go_straight/veer_left/veer_right, 映射到 cruise/inspect_left/inspect_right。"""
    return ("Inspection robot front view. Choose one: go_straight, veer_left, veer_right. One word.")


def parse_action(ans):
    """慢脑 go_straight/veer_left/veer_right -> cond_id(cruise/inspect_left/inspect_right)。"""
    tl = (ans or "").lower()
    if "veer_left" in tl or "left" in tl:
        return CONDITIONS.index("inspect_left"), "inspect_left"
    if "veer_right" in tl or "right" in tl:
        return CONDITIONS.index("inspect_right"), "inspect_right"
    if "straight" in tl or "cruise" in tl:
        return CONDITIONS.index("cruise"), "cruise"
    return CONDITIONS.index("cruise"), "cruise(default)"

T_HORIZON = 8      # 轨迹路点数
DT_TRAJ = 0.5      # 轨迹每步秒(与 make_traj 一致) -> 4s 视界


# ---------------- 慢脑: HTTP API ----------------
def vlm_api(frames_bgr, prompt, max_new_tokens=32, timeout=120):
    """frames_bgr: 单张 或 list(多帧时序). 多帧=robot2vlm原生格式, 定位更准。"""
    if not isinstance(frames_bgr, list):
        frames_bgr = [frames_bgr]
    files = []
    for j, fb in enumerate(frames_bgr):
        ok, buf = cv2.imencode(".jpg", fb)
        files.append(("images", (f"f{j}.jpg", buf.tobytes(), "image/jpeg")))
    data = {"prompt": prompt, "max_new_tokens": str(max_new_tokens), "do_sample": "false"}
    r = requests.post(API_URL, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j.get("answer", ""), j.get("meta", {})


def region_to_cond(ans):
    """VLM 定位的目标区域(left/center/right) -> cond_id。目标偏侧=横移贴过去(strafe), center=直行。"""
    t = (ans or "").lower()
    if "left" in t or "左" in t:
        return CONDITIONS.index("strafe_left")
    if "right" in t or "右" in t:
        return CONDITIONS.index("strafe_right")
    return CONDITIONS.index("cruise")


def answer_to_cond(ans):
    """自研 VLM 自由文本 answer -> cond_id。关键词映射(中英)。"""
    t = (ans or "").lower()
    if any(k in t for k in ["stop", "black", "blur", "block", "dark", "停"]):
        return CONDITIONS.index("stop")
    if "pause" in t or "暂停" in t:
        return CONDITIONS.index("pause")
    if any(k in t for k in ["left", "左"]):
        return CONDITIONS.index("inspect_left")
    if any(k in t for k in ["right", "右"]):
        return CONDITIONS.index("inspect_right")
    return CONDITIONS.index("cruise")


# ---------------- FPV 轨迹叠加(同 full_chain) ----------------
def cam_pose_from_robot(x, y, yaw, cam_h=1.2):
    R_cb = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    return np.array([x, y, cam_h]), Rz @ R_cb


def draw_traj_on_fpv(img, pose, traj, color, fovy=75.0):
    H, W = img.shape[:2]
    cpos, cmat = cam_pose_from_robot(*pose)
    wxy = ego_to_world(traj, pose)
    uv = project_ground(cpos, cmat, fovy, W, H, wxy)
    uv0 = project_ground(cpos, cmat, fovy, W, H, [ego_to_world(np.zeros((1, 3)), pose)[0]])[0]
    chain = [uv0] + uv
    for a, b in zip(chain[:-1], chain[1:]):
        if a and b:
            cv2.line(img, a, b, color, 4, cv2.LINE_AA)
    if uv[-1]:
        cv2.circle(img, uv[-1], 6, color, -1)


# ---------------- 轨迹 -> (vx, vy, wz) 全向(4轮底座能横移, 同 svtrobot_cmd) ----------------
def traj_to_control(traj, lookahead=1.0, vx_max=0.7, vy_max=0.5, wz_max=0.8):
    """ego 轨迹(8,3)=(x前,y左,yaw) -> (vx前, vy左, wz转)。
    ⭐机器人是4轮全向底座(svtrobot vx,vy,wz), 轨迹的横向分量当【横移vy】执行, 不是转弯,
    所以 inspect 那种贴向侧面设备的轨迹能直接横着过去, 不会原地打转。"""
    xy = traj[:, :2]
    d = np.hypot(xy[:, 0], xy[:, 1])
    idx = int(np.argmax(d >= lookahead)) if np.any(d >= lookahead) else int(d.argmax())
    tx, ty = xy[idx]
    Ttime = (idx + 1) * DT_TRAJ                                 # 到该前视点的轨迹时间
    vx = float(np.clip(tx / Ttime, -vx_max, vx_max))           # 前向
    vy = float(np.clip(ty / Ttime, -vy_max, vy_max))           # 横移(4轮能走)
    wz = float(np.clip(float(traj[idx, 2]) / Ttime, -wz_max, wz_max))  # 缓慢对齐轨迹朝向
    return vx, vy, wz


# ---------------- 反应式避障: 运动方向近处有墙则限速(治无约束横移滑穿墙) ----------------
def bev_safe_vel(bev, vx, vy, win_m=8.0, margin_m=0.55):
    """bev: (n,n) 局部占据(row小=前, col大=右, 中心=机器人)。
    前/左/右方向 margin_m 内有占据 → 该方向速度归零, 到墙边自停。"""
    n = bev.shape[0]; c = n // 2; ppm = n / win_m
    mrg = max(2, int(margin_m * ppm))
    if vx > 0 and bev[max(0, c - mrg):c, c - 2:c + 3].max() > 0:
        vx = min(vx, 0.03)                                  # 前方有墙: 几乎停前进
    if vy > 0 and bev[c - 2:c + 3, max(0, c - mrg):c].max() > 0:
        vy = 0.0                                            # 左侧有墙: 停左移
    if vy < 0 and bev[c - 2:c + 3, c:c + mrg].max() > 0:
        vy = 0.0                                            # 右侧有墙: 停右移
    return vx, vy


# ---------------- 走廊护栏: 侧向余量门控 + 卡住脱困(治单向veer撞墙) ----------------
def _blocked(bev, direction, margin_m=0.7, win_m=8.0):
    """机器人某方向 margin_m 内是否有占据。bev: row小=前, col大=右, 中心=机器人。"""
    n = bev.shape[0]; c = n // 2; m = max(2, int(margin_m * n / win_m))
    if direction == "front":
        return bev[max(0, c - m):c, c - 2:c + 3].max() > 0
    if direction == "left":
        return bev[c - 2:c + 3, max(0, c - m):c].max() > 0
    if direction == "right":
        return bev[c - 2:c + 3, c:c + m].max() > 0
    return False


def _more_open_dir(bev):
    """前方左/右象限谁更空 -> 脱困转向 wz 符号(+左 / -右)。占据均值小=更空。"""
    n = bev.shape[0]; c = n // 2; top = max(0, c - int(n * 0.4))
    left = float(bev[top:c, :c].mean()); right = float(bev[top:c, c:].mean())
    return +1.0 if left < right else -1.0


def follow_the_gap(bev, win_m=8.0, max_range=3.6, half_fov=80.0, dstep=0.06):
    """⭐纯感知反应式走廊跟随(Follow-The-Gap)。只吃 BEV 占据(来自深度/点云传感器, 真机也有),
    不用任何预录参考路径。从机器人向前方扇形射线扫, 每个角度量到最近占据的自由距离,
    返回【最深自由方向的角度(+左, rad)】+ 该方向自由距离 + 正前方自由距离。
    朝最深自由方向走 = 天然沿走廊、窄口处自动朝那条通道, 全程感知驱动。"""
    n = bev.shape[0]; c = n // 2; ppm = n / win_m
    angs = np.deg2rad(np.arange(-half_fov, half_fov + 1, 4.0))
    free = np.full(len(angs), max_range, np.float32)
    for i, a in enumerate(angs):
        ca, sa = np.cos(a), np.sin(a)                       # a: 从正前(+x前)量, +左
        for rr in np.arange(0.2, max_range, dstep):
            fwd = rr * ca; lft = rr * sa                    # ego: 前 / 左
            row = int(c - fwd * ppm); col = int(c - lft * ppm)  # 上=前(row小), 左=col小
            if row < 0 or row >= n or col < 0 or col >= n or bev[row, col] > 0:
                free[i] = rr; break
    # 深度做角向平滑(避免抓到单像素缝), 取最深自由方向
    k = np.ones(3) / 3.0
    fs = np.convolve(free, k, mode="same")
    best = float(angs[int(fs.argmax())])
    front = float(free[np.abs(angs).argmin()])              # 正前方自由距离
    return best, float(fs.max()), front


# ---------------- BEV 占据窗可视化(叠轨迹) ----------------
def render_bev(bev_win, traj, cond_color, size=400, win_m=8.0):
    """bev_win: (n,n) 占据(朝上=前). 画机器人+朝向+预测轨迹。"""
    n = bev_win.shape[0]
    img = cv2.resize(((1 - bev_win) * 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    c = size // 2
    ppm = size / win_m                                        # 像素/米
    # 预测轨迹(ego): 上=前(x), 右=右(-y)
    pts = [(c, c)]
    for ex, ey in traj[:, :2]:
        pts.append((int(c - ey * ppm), int(c - ex * ppm)))
    cv2.polylines(img, [np.array(pts, np.int32)], False, cond_color, 3, cv2.LINE_AA)
    cv2.circle(img, (c, c), 6, (0, 0, 255), -1)              # 机器人
    cv2.arrowedLine(img, (c, c), (c, c - 40), (0, 0, 255), 2)  # 朝向(上)
    cv2.putText(img, "BEV occ (DiffusionDrive in)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return img


# ---------------- 全局导航层: A* 在 SLAM 占据地图上规划到目标(AD的route对应物) ----------------
# ⚠️合法: 占据地图=机器人SLAM传感器地图(真机必有), goal来自mission(巡检到走廊尽头), A*=自己算的路。
#    不是抄预录ref轨迹。这补上了纯反应式缺的"全局朝向"。
def astar_grid(occ, s_rc, g_rc, step=3, clear_w=2.0, clear_target_m=1.0, res=0.05):
    """8连通 A* on 占据栅格(降采样step提速)。occ>0=障碍。返回 [(r,c),...] 或 None。
    ⭐clear_w>0: 加【离墙间隙代价】(distance transform)——离最近障碍越近代价越高,
    使 A* 走走廊中线而非抄近道贴墙(治"贴左墙"左偏 + 贴墙触发bev_safe龟速)。"""
    import heapq
    H, W = occ.shape
    # 降采样占据(取块内max=保守): 障碍膨胀过, 走廊仍通
    ds = occ[::step, ::step]
    hh, ww = ds.shape
    # 间隙场: 每个free格到最近障碍的距离(米)。贴墙→小间隙→高代价。
    if clear_w > 0:
        free = (ds == 0).astype(np.uint8)
        dist_cells = cv2.distanceTransform(free, cv2.DIST_L2, 3)  # 单位=降采样格
        clear_m = dist_cells * (step * res)                       # 转米
        clear_target = clear_target_m
        # 惩罚 = clear_w * max(0, 目标间隙 - 实际间隙)/目标间隙, 间隙>=目标则0
        clear_pen = clear_w * np.maximum(0.0, clear_target - clear_m) / clear_target
    else:
        clear_pen = np.zeros_like(ds, np.float32)
    sr, sc = s_rc[0] // step, s_rc[1] // step
    gr, gc = g_rc[0] // step, g_rc[1] // step
    sr = min(max(sr, 0), hh - 1); sc = min(max(sc, 0), ww - 1)
    gr = min(max(gr, 0), hh - 1); gc = min(max(gc, 0), ww - 1)
    def h(r, c): return ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5
    openq = [(h(sr, sc), 0.0, sr, sc)]; came = {}; g = {(sr, sc): 0.0}
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while openq:
        _, gc0, r, c = heapq.heappop(openq)
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in came:
                r, c = came[(r, c)]; path.append((r, c))
            path.reverse()
            return [(rr * step, cc * step) for rr, cc in path]   # 回原分辨率
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < hh and 0 <= nc < ww and ds[nr, nc] == 0:
                move = (1.414 if dr and dc else 1.0)
                ng = gc0 + move + float(clear_pen[nr, nc])        # ⭐间隙代价推向走廊中线
                if ng < g.get((nr, nc), 1e18):
                    g[(nr, nc)] = ng; came[(nr, nc)] = (r, c)
                    heapq.heappush(openq, (ng + h(nr, nc), ng, nr, nc))
    return None


def plan_path(env):
    """A* 规划 start->goal 世界路径(米)。返回 (N,2) 或 None。"""
    lo, res = env.lo, env.res
    def w2c(x, y): return (int((y - lo[1]) / res), int((x - lo[0]) / res))
    s = w2c(env.start[0], env.start[1]); gcell = w2c(env.goal[0], env.goal[1])
    cells = astar_grid(env.occ, s, gcell, step=2, res=res, clear_w=2.0, clear_target_m=1.0)
    if not cells:
        return None
    wp = np.array([[lo[0] + c * res, lo[1] + r * res] for r, c in cells])
    # 抽稀 + 极轻平滑(别把路径抹圆切进墙)
    wp = wp[::3] if len(wp) > 6 else wp
    wp[1:-1] = 0.6 * wp[1:-1] + 0.2 * (wp[:-2] + wp[2:])
    return wp


def pursue(path, pose, lookahead=2.0, v_nom=0.85):
    """纯追踪全局路径 -> (vx前, wz转, done)。wz增益适中防震荡; cos下限高=大弯也保速。"""
    xy = path; px, py, yaw = pose
    d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
    i0 = int(d.argmin()); j = i0
    while j < len(xy) - 1 and np.hypot(xy[j, 0] - px, xy[j, 1] - py) < lookahead:
        j += 1
    tx, ty = xy[j]
    dyaw = np.arctan2(ty - py, tx - px) - yaw
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi                  # wrap
    vx = v_nom * float(np.clip(np.cos(dyaw), 0.45, 1.0))
    wz = float(np.clip(1.0 * dyaw, -0.7, 0.7))
    done = (i0 >= len(xy) - 2)
    return vx, wz, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--vlm_every", type=int, default=8, help="每 N 步调一次慢脑(≈2s), 其余步复用cond")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--demo_modes", type=int, default=0,
                    help=">0: 忽略VLM, 每N步脚本切换cond(cruise→left→cruise→right循环), 在闭环里演示可控性")
    ap.add_argument("--localize", action="store_true",
                    help="⭐新架构: 慢脑不做左右决策, 只【定位目标在左/中/右】(多帧时序), region->cond, 快脑导航")
    ap.add_argument("--target", default="blue pump or machine", help="定位目标(仅--localize模式用)")
    ap.add_argument("--mission", default=MISSION_DEFAULT, help="任务指令(默认结构化prompt模式的mission)")
    ap.add_argument("--skin_ply", default=None, help="覆盖FPV皮肤ply(如ArtiFixer3D锐利皮肤 gs_a3d_..._world.ply)")
    ap.add_argument("--nav", action="store_true",
                    help="⭐全局导航层: A*在SLAM占据图规划到goal+纯追踪(导航主干), VLM语义只加有界侧倾(AD做法)")
    args = ap.parse_args()
    SLOW_PROMPT = slow_prompt(args.mission)                 # ⭐默认模式: 结构化 WHERE+HOW prompt
    DEMO_SEQ = [CONDITIONS.index(c) for c in                # 演示全套新模式(含原地转)
                ["cruise", "turn_left", "turn_right", "strafe_left", "strafe_right", "stop", "cruise", "inspect_left"]]
    # 定位式 prompt(感知, 非决策; VLM擅长): 输出目标水平区域
    LOC_PROMPT = (f"These images are consecutive frames in time order (oldest first). In the LAST (current) frame, "
                  f"where is the most prominent {args.target} located horizontally? "
                  f"Answer only one word: left, center, or right.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- 活仿真环境 ----------
    print(f"[sim] 建 CorridorTestEnv(skin={args.session}) ...", flush=True)
    env = CorridorTestEnv(skin_session=args.session, skin_ply=args.skin_ply)
    nav_path = None
    if args.nav:
        nav_path = plan_path(env)
        print(f"[nav] A* 全局路径: {'成功 %d 点' % len(nav_path) if nav_path is not None else '失败(无路)'}", flush=True)
    ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    xyz = sb.read_ply_xyz(ply)
    _, _, _, floor = sb.build_occ(xyz)
    print(f"[sim] start={tuple(round(x,2) for x in env.start[:2])} goal={tuple(round(x,2) for x in env.goal)}", flush=True)

    # ---------- 快脑(条件微调版) ----------
    ckpt_name = "nav_finetuned.pt" if args.nav else "cond_finetuned.pt"   # --nav=导航策略(含goal_mlp)
    print(f"[fast] 加载 DiffusionDrive + {ckpt_name} ...", flush=True)
    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_anchors(), device=device)
    model._trajectory_head.load_state_dict(
        torch.load(os.path.join(RUNS, ckpt_name))["trajectory_head"], strict=False)
    model.eval()
    st = torch.zeros((1, 8), device=device); st[0, 4] = 0.4    # 中性 status(vx)

    # ---------- 慢脑健康检查 ----------
    try:
        h = requests.get("http://10.0.0.18:8152/health", timeout=10).json()
        print(f"[slow] VLM API health: ok={h.get('ok')} loaded={h.get('loaded')}", flush=True)
    except Exception as e:
        print(f"[slow] ⚠️ VLM API 健康检查失败: {e} (仍会尝试请求)", flush=True)

    import imageio, shutil
    stamp = time.strftime("%Y%m%d_%H%M%S")                      # ⭐每次带时间戳存, 永不覆盖丢失
    mp4 = os.path.join(RUNS, f"sim_fullchain_{args.session}_{stamp}.mp4")
    jsonl = os.path.join(RUNS, f"sim_fullchain_{args.session}_{stamp}.jsonl")
    canon_mp4 = os.path.join(RUNS, f"sim_fullchain_{args.session}.mp4")   # 同名=最新副本(方便引用)
    canon_jsonl = os.path.join(RUNS, f"sim_fullchain_{args.session}.jsonl")
    writer = imageio.get_writer(mp4, fps=args.fps, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    flog = open(jsonl, "w", encoding="utf-8")

    obs = env.reset()
    cid, answer, meta = 1, "", {}
    frame_buf = []                                             # 定位模式: 最近若干tick帧(多帧时序)
    t0 = time.time()
    if args.localize:
        print(f"[slow] 定位模式(新架构): 目标='{args.target}', 慢脑只定位left/center/right->cond", flush=True)
    for k in range(args.steps):
        pose = obs["pose"]
        fpv_bgr = np.ascontiguousarray(obs["fpv"][:, :, ::-1])  # env 出 RGB -> BGR

        # ----- 慢脑: 脚本演示 / 定位模式(新架构) / VLM决策 -----
        if args.demo_modes > 0:
            cid = DEMO_SEQ[(k // args.demo_modes) % len(DEMO_SEQ)]  # 脚本切换cond演示可控性
            vlm_fired = (k % args.demo_modes == 0)
            answer = f"[scripted]{CONDITIONS[cid]}"
        elif args.localize:                                    # ⭐慢脑只定位目标, region->cond
            vlm_fired = (k % args.vlm_every == 0)
            if vlm_fired:
                frame_buf.append(fpv_bgr); frame_buf[:] = frame_buf[-3:]  # 最近3帧时序
                try:
                    answer, meta = vlm_api(frame_buf, LOC_PROMPT, max_new_tokens=8)
                    cid = region_to_cond(answer)
                except Exception as e:
                    answer, meta = f"[API err]{e}", {}
        else:                                                  # ⭐默认: 结构化 WHERE+HOW prompt, 全9词表
            vlm_fired = (k % args.vlm_every == 0)
            if vlm_fired:
                frame_buf.append(fpv_bgr); frame_buf[:] = frame_buf[-3:]  # 最近3帧时序
                try:
                    answer, meta = vlm_api(frame_buf, SLOW_PROMPT, max_new_tokens=12)
                    cid, act = parse_action(answer)
                except Exception as e:
                    answer, meta = f"[API error] {e}", {}
                    # cid 维持上次

        # ----- 快脑: cond(+goal) -> 轨迹 -----
        cam = sb.camera_feature(fpv_bgr, device)
        bev = sb.lidar_feature_fov(xyz, pose, floor, device)
        feats = {"camera_feature": cam, "lidar_feature": bev, "status_feature": st,
                 "cond_id": torch.tensor([cid], device=device)}
        if args.nav and nav_path is not None:
            # ⭐AD式: A*(SLAM地图+mission)只给【目标方向】(粗route,合法), 训练好的策略生成实际轨迹(会沿走廊居中避墙)
            d = np.hypot(nav_path[:, 0] - pose[0], nav_path[:, 1] - pose[1]); i0 = int(d.argmin())
            acc = 0.0; jj = i0
            while jj + 1 < len(nav_path) and acc < 5.0:
                acc += float(np.linalg.norm(nav_path[jj + 1] - nav_path[jj])); jj += 1
            c_, s_ = np.cos(pose[2]), np.sin(pose[2]); gx, gy = nav_path[jj] - pose[:2]
            ge = np.array([c_ * gx + s_ * gy, -s_ * gx + c_ * gy]); ge = ge / (np.linalg.norm(ge) + 1e-6)
            feats["goal_dir"] = torch.tensor(ge, device=device).float().view(1, 2)
        with torch.no_grad():
            traj = model(feats)["trajectory"][0].cpu().numpy()

        # 控制 = 策略轨迹 -> (vx,vy,wz) (nav模式下策略已朝goal沿走廊, 不再纯追踪)
        vx, vy, wz = traj_to_control(traj)
        vx, vy = bev_safe_vel(obs["bev"], vx, vy)          # 反应式避障兜底
        if abs(vy) < 0.03 and vx < 0.10:                   # 防卡死: 前方空保底前进
            _n = obs["bev"].shape[0]; _c = _n // 2; _m = int(0.6 * _n / 8.0)
            if obs["bev"][max(0, _c - _m):_c, _c - 3:_c + 4].max() == 0:
                vx = 0.45
        if args.nav and nav_path is not None:              # 接近目标平稳减速
            dg = float(np.hypot(pose[0] - env.goal[0], pose[1] - env.goal[1]))
            if dg < 2.0:
                vx *= max(0.15, dg / 2.0); vy *= max(0.15, dg / 2.0)

        # ----- 实时日志 -----
        rec = {"step": k, "pose": [round(float(p), 3) for p in pose],
               "vlm_fired": vlm_fired,
               "vlm_answer": (answer if vlm_fired else None),
               "vlm_gen_s": (round(meta.get("generate_seconds", 0), 2) if vlm_fired else None),
               "cond": CONDITIONS[cid], "cond_id": cid,
               "traj_end_ego": [round(float(x), 3) for x in traj[-1]],
               "vx": round(vx, 3), "vy": round(vy, 3), "wz": round(wz, 3)}

        # ----- 可视化: 左 FPV(叠轨迹) | 右 BEV -----
        col = COND_COLOR[CONDITIONS[cid]]
        fpv_vis = fpv_bgr.copy()
        draw_traj_on_fpv(fpv_vis, pose, traj, col)
        cv2.rectangle(fpv_vis, (0, 0), (fpv_vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(fpv_vis, f"VLM:{CONDITIONS[cid]} -> DD -> vx={vx:.2f} vy={vy:+.2f} wz={wz:+.2f}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)
        bev_vis = render_bev(obs["bev"], traj, col)
        H = fpv_vis.shape[0]
        bev_vis = cv2.resize(bev_vis, (H, H))
        frame = np.hstack([fpv_vis, bev_vis])
        writer.append_data(frame[:, :, ::-1])                 # BGR -> RGB

        # ----- 推进仿真(全向) -----
        obs, term, info = env.step(vx, vy, wz)
        rec["collided"] = bool(info["collided"]); rec["reached"] = bool(info["reached"])
        flog.write(json.dumps(rec, ensure_ascii=False) + "\n"); flog.flush()

        if vlm_fired or k % 10 == 0:
            tag = f" | VLM'{answer}'->{CONDITIONS[cid]}" if vlm_fired else ""
            print(f"  step{k:3d} pose=({pose[0]:.1f},{pose[1]:.1f}) cond={CONDITIONS[cid]:<12} "
                  f"vx={vx:.2f} vy={vy:+.2f} wz={wz:+.2f}{tag}", flush=True)
        if term:
            print(f"[sim] 终止 step{k}: {info}", flush=True)
            break

    writer.close(); flog.close()
    shutil.copy(mp4, canon_mp4); shutil.copy(jsonl, canon_jsonl)   # 同名副本=最新, 时间戳版永久留存
    print(f"[done] {time.time()-t0:.0f}s\n  版本(永久) -> {mp4}\n  同名(最新) -> {canon_mp4}\n  日志 -> {jsonl}", flush=True)


if __name__ == "__main__":
    main()
