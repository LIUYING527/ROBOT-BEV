#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_lab_mission_real.py — 真机版 lab_mission（实验室 110744 已知场景，直接跑 M2）。

移植自 scripts/lab_mission.py，只换「手和眼睛」，不动「大脑」：
  env.step(vx,vy,wz) → /svtrobot_cmd Twist 发布
  sim_fpv()           → ZED 真图（/dev/shm 读取）
  渲染 ED             → zed 真实深度
  believed 位姿        → 指令积分（无 odom）+ reloc HTTP 周期纠偏
  占据/目标/采集轨迹   → 离线资产（世界 ply + targets json + sim_world npz，拷到 Orin）

状态机逻辑（cruise/face/align_front/precise_dock/undock/reloc/vlm_confirm）原样保留。

⚠ 安全默认：--dry 默认开（不发指令 + mock 观测），真机跑必须显式 --no-dry。
⚠ 真机前必过：RELOC 三级冒烟（真 zed 重定位 <10cm/<2°），见 docs/真机部署操作手册_0904.md §3.3。
"""
import os, sys, json, argparse, time as _time, heapq
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
RUNS = os.path.join(ROOT, "goalnav", "runs")

DT_TRAJ = 0.5          # 轨迹时间步（与 run_diffusiondrive_policy --control-dt 对齐）
GOAL_AHEAD = 4.0
ROBOT_R = 0.30


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ============================================================
# A* 规划（自包含，纯 numpy/heapq/cv2，与 sim_fullchain.astar_grid 同口径）
# ============================================================
def astar_grid(occ, s_rc, g_rc, step=2, res=0.05, clear_w=2.0, clear_target_m=1.0):
    """8连通 A* on 占据栅格(降采样 step 提速)。occ>0=障碍。返回 [(r,c),...] 或 None。
    ⭐clear_w>0: 离墙间隙代价(distance transform)——走走廊中线, 治贴墙(与 sim_fullchain.astar_grid 同口径)。"""
    H, W = occ.shape
    ds = occ[::step, ::step]                          # 降采样(每隔 step 采样)
    hh, ww = ds.shape
    if clear_w > 0:
        free = (ds == 0).astype(np.uint8)
        dist_cells = cv2.distanceTransform(free, cv2.DIST_L2, 3)
        clear_m = dist_cells * (step * res)
        clear_pen = clear_w * np.maximum(0.0, clear_target_m - clear_m) / clear_target_m
    else:
        clear_pen = np.zeros_like(ds, np.float32)
    sr, sc = s_rc[0] // step, s_rc[1] // step
    gr, gc = g_rc[0] // step, g_rc[1] // step
    sr = min(max(sr, 0), hh - 1); sc = min(max(sc, 0), ww - 1)
    gr = min(max(gr, 0), hh - 1); gc = min(max(gc, 0), ww - 1)
    def h(r, c):
        return ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5
    openq = [(h(sr, sc), 0.0, sr, sc)]; came = {}; g = {(sr, sc): 0.0}
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while openq:
        _, gc0, r, c = heapq.heappop(openq)
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in came:
                r, c = came[(r, c)]; path.append((r, c))
            path.reverse()
            return [(rr * step, cc * step) for rr, cc in path]
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < hh and 0 <= nc < ww and ds[nr, nc] == 0:
                move = (1.414 if dr and dc else 1.0)
                ng = gc0 + move + float(clear_pen[nr, nc])
                if ng < g.get((nr, nc), 1e18):
                    g[(nr, nc)] = ng; came[(nr, nc)] = (r, c)
                    heapq.heappush(openq, (ng + h(nr, nc), ng, nr, nc))
    return None


def plan_path(env):
    """A* 规划 start->goal 世界路径(米)。返回 (N,2) 或 None。"""
    lo, res = env.lo, env.res
    def w2c(x, y):
        return (int((y - lo[1]) / res), int((x - lo[0]) / res))
    s = w2c(env.start[0], env.start[1]); g = w2c(env.goal[0], env.goal[1])
    cells = astar_grid(env.occ, s, g, step=2, res=res, clear_w=2.0, clear_target_m=1.0)
    if not cells:
        return None
    wp = np.array([[lo[0] + c * res, lo[1] + r * res] for r, c in cells])
    wp = wp[::3] if len(wp) > 6 else wp
    if len(wp) >= 3:
        wp[1:-1] = 0.6 * wp[1:-1] + 0.2 * (wp[:-2] + wp[2:])
    return wp


def traj_to_control(traj, lookahead=1.0, vx_max=0.7, vy_max=0.5, wz_max=0.8):
    """ego 轨迹(8,3)=(x前,y左,yaw) -> (vx前, vy左, wz转)。全向底座: 横向分量当横移 vy。"""
    xy = traj[:, :2]
    d = np.hypot(xy[:, 0], xy[:, 1])
    idx = int(np.argmax(d >= lookahead)) if np.any(d >= lookahead) else int(d.argmax())
    tx, ty = xy[idx]
    Ttime = (idx + 1) * DT_TRAJ
    vx = float(np.clip(tx / Ttime, -vx_max, vx_max))
    vy = float(np.clip(ty / Ttime, -vy_max, vy_max))
    wz = float(np.clip(float(traj[idx, 2]) / Ttime, -wz_max, wz_max))
    return vx, vy, wz


# ============================================================
# 占据栅格构建（自包含移植 test_env.build_occupancy + CorridorTestEnv._clear_path，
# 与仿真同口径：闭运算 + fill_holes + 去孤立噪块 + 膨胀 + 沿采集轨迹清路）
# ============================================================
def build_occupancy_real(merged_ply, res=0.08, robot_r=0.3, traj=None, thr=2, close_k=9):
    import trimesh
    from scipy import ndimage
    P = np.asarray(trimesh.load(merged_ply, process=False).vertices)
    mid = P[(P[:, 2] > 0.3) & (P[:, 2] < 2.0)]
    pts2 = mid[:, :2] if traj is None else np.vstack([mid[:, :2], traj])
    lo = pts2.min(0) - 1.5
    hi = pts2.max(0) + 1.5
    W = int(np.ceil((hi[0] - lo[0]) / res)); H = int(np.ceil((hi[1] - lo[1]) / res))
    occ = np.zeros((H, W), np.uint8)
    ix = np.clip(((mid[:, 0] - lo[0]) / res).astype(int), 0, W - 1)
    iy = np.clip(((mid[:, 1] - lo[1]) / res).astype(int), 0, H - 1)
    cnt = np.zeros((H, W), np.int32); np.add.at(cnt, (iy, ix), 1)
    occ[cnt >= thr] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    occ = ndimage.binary_fill_holes(occ).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 40:
            occ[lbl == i] = 0
    occ = cv2.dilate(occ, np.ones((int(robot_r / res) * 2 + 1,) * 2, np.uint8))
    return occ, lo, res


def clear_path(occ, lo, res, wp, r):
    rad = int(r / res)
    for a, b in zip(wp[:-1], wp[1:]):
        ga = (int((a[0] - lo[0]) / res), int((a[1] - lo[1]) / res))
        gb = (int((b[0] - lo[0]) / res), int((b[1] - lo[1]) / res))
        cv2.line(occ, ga, gb, 0, thickness=2 * rad)


# ============================================================
# ZED 真图源（/dev/shm 读取，从 run_diffusiondrive_policy 先例简化）
# ============================================================
class ZedSource:
    """读 web_control/server.py 写的 /dev/shm/zed_color.raw + zed_depth.raw。
    dry-run 时用 mock（黑图 + 全 0 深度），不连真机。"""

    def __init__(self, dry=True, W=1280, H=720):
        self.dry = dry
        self.W, self.H = W, H
        if dry:
            print("[zed] dry-run: mock ZED（黑图 + 0 深度）", flush=True)
        else:
            print(f"[zed] 读 /dev/shm/zed_color.raw + zed_depth.raw ({W}x{H})", flush=True)

    def capture(self):
        """返回 (rgb HxWx3 uint8, depth HxW float32 米)。dry-run 返回 mock。"""
        if self.dry:
            rgb = np.zeros((self.H, self.W, 3), np.uint8)
            depth = np.full((self.H, self.W), 1.0, np.float32)
            return rgb, depth
        # 真机: 读 shm（与先例 SharedMemoryZEDSource.capture 同款，超 3s 不新鲜需重启相机）
        try:
            rgb = np.fromfile("/dev/shm/zed_color.raw", np.uint8).reshape(self.H, self.W, 3).copy()
            depth = np.fromfile("/dev/shm/zed_depth.raw", np.float32).reshape(self.H, self.W).copy()
            return rgb, depth
        except Exception as e:
            raise RuntimeError(f"[zed] shm 读取失败: {e}（检查 web_control/server.py 是否在跑）")


# ============================================================
# Twist 发布（rclpy；dry-run 为 no-op）
# ============================================================
class TwistPublisher:
    """发布 /svtrobot_cmd（geometry_msgs/Twist）。0.5s 超时红线，控制环须 ≥2Hz。
    dry-run 只打印不发。"""

    def __init__(self, dry=True, auto_stop_timeout=0.5):
        self.dry = dry
        self.auto_stop = auto_stop_timeout
        self.pub = None
        self._last = _time.time()
        if not dry:
            import rclpy
            from geometry_msgs.msg import Twist
            rclpy.init()
            node = rclpy.create_node("run_lab_mission_real")
            self.pub = node.create_publisher(Twist, "/svtrobot_cmd", 10)
            self._node = node
            self._Twist = Twist
            print("[twist] rclpy 就绪 -> /svtrobot_cmd", flush=True)

    def publish(self, vx, vy, wz):
        vx, vy, wz = float(vx), float(vy), float(wz)
        now = _time.time()
        if now - self._last > 0.4:                    # 心跳检查（0.5s 超时前必须重发）
            pass
        self._last = now
        if self.dry:
            print(f"    [DRY] cmd (vx={vx:+.2f}, vy={vy:+.2f}, wz={wz:+.2f})", flush=True)
            return
        m = self._Twist()
        m.linear.x = vx; m.linear.y = vy; m.angular.z = wz
        self.pub.publish(m)


# ============================================================
# RealEnv：复刻 CorridorTestEnv 接口（状态机靠这套接口跑，逻辑不变）
# ============================================================
class RealEnv:
    """真机环境抽象。位姿 = 指令积分（无 odom），占据 = 离线世界 ply 烘的栅格。
    提供与 CorridorTestEnv 一致的接口: x/y/yaw, step, _obs, _occupied, occ/lo/res, start/goal, win_m。
    """

    def __init__(self, zed, twist, xyz, floor, occ, lo, res, win_m=8.0, dry=True):
        self.zed = zed
        self.twist = twist
        self.xyz = xyz          # 世界 ply 点云 (N,3)（BEV 用，与仿真同口径）
        self.floor = floor
        self.occ = occ
        self.lo = lo
        self.res = res
        self.win_m = win_m
        self.dry = dry
        self.start = (0.0, 0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.x, self.y, self.yaw = self.start
        self.t = 0

    def reset(self):
        self.x, self.y, self.yaw = self.start
        self.t = 0
        return self._obs()

    def step(self, vx, vy=0.0, wz=0.0, dt=0.25):
        """发 Twist + 指令积分更新 believed 位姿。返回 (obs, term, info)。"""
        self.twist.publish(vx, vy, wz)
        # 指令积分（全向）: 世界系位移
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.x += (vx * c - vy * s) * dt
        self.y += (vx * s + vy * c) * dt
        self.yaw = float(wrap(self.yaw + wz * dt))
        self.t += 1
        return self._obs(), False, {"t": self.t}

    def _obs(self):
        rgb, depth = self.zed.capture()
        return {"fpv": rgb, "depth": depth, "pose": (self.x, self.y, self.yaw)}

    def _occupied(self, x, y):
        ix = int((x - self.lo[0]) / self.res); iy = int((y - self.lo[1]) / self.res)
        if iy < 0 or iy >= self.occ.shape[0] or ix < 0 or ix >= self.occ.shape[1]:
            return True
        return self.occ[iy, ix] > 0


# ============================================================
# RELOC HTTP 客户端（真机: 发点云求纠正；dry-run: 跳过）
# ============================================================
class RelocClient:
    def __init__(self, server, dry=True, cam_h=1.2):
        self.server = server
        self.dry = dry
        self.cam_h = cam_h
        self.requests = None
        if not dry:
            import requests
            self.requests = requests
            print(f"[reloc] HTTP 客户端 -> {server}", flush=True)

    def correct(self, depth, x, y, yaw_deg, pitch_deg=0.0):
        """深度(H,W 米) + believed 位姿 → 纠正位姿。返回 (x,y,yaw_deg,accepted) 或 None(跳过)。"""
        if self.dry:
            print(f"    [DRY] reloc skip (believed {x:.2f},{y:.2f},{yaw_deg:.0f}°)", flush=True)
            return None
        import io
        # 深度 → xyzrgba npz（与采集同格式，float16 mm）——真实 zed 点云可由 camera_driver 直接取，
        # 这里用深度反投影近似（内参需按 ZED 实测填，见 TODO 标注）。
        H, W = depth.shape
        fx = fy = (W / 2.0) / np.tan(np.deg2rad(79.0 / 2.0))   # TODO: 换 ZED 实测内参
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        d = depth.astype(np.float32)
        m = (d > 0.3) & (d < 12.0)
        xc = (u - cx) * d / fx; yc = -(v - cy) * d / fy; zc = d
        pc = np.stack([xc, yc, zc], -1) * 1000.0              # 米 → mm
        rgb = np.zeros((H, W, 3), np.float32)
        xyzrgba = np.concatenate([pc, rgb], -1).astype(np.float16)
        buf = io.BytesIO()
        np.savez(buf, xyzrgba=xyzrgba)
        try:
            r = self.requests.post(f"{self.server}/reloc",
                                   params=dict(x=x, y=y, yaw=yaw_deg, pitch_deg=pitch_deg, cam_h=self.cam_h),
                                   data=buf.getvalue(), timeout=180)
            out = r.json()
        except Exception as e:
            print(f"    [reloc] HTTP 失败: {e}", flush=True)
            return None
        if not out.get("ok"):
            print(f"    [reloc] 拒绝: {out.get('err')}", flush=True)
            return None
        return float(out["x"]), float(out["y"]), float(out["yaw_deg"]), bool(out.get("accepted"))


# ============================================================
# 站位计算（与 lab_mission 同口径，纯几何 + 采集轨迹）
# ============================================================
def find_standoff(env, tx, ty, wp, yaw, stand=1.2, avoid=()):
    best = None
    for i in range(len(wp)):
        sx, sy = float(wp[i, 0]), float(wp[i, 1])
        d = np.hypot(sx - tx, sy - ty)
        if d < 0.7 or d > 2.0:
            continue
        if any(np.hypot(sx - ax, sy - ay) < 0.6 for ax, ay in avoid):
            continue
        if env._occupied(sx, sy):
            continue
        bearing = np.arctan2(ty - sy, tx - sx)
        berr = abs(wrap(bearing - yaw[i]))
        score = berr + 0.25 * abs(d - stand)
        if best is None or score < best[0]:
            best = (score, sx, sy, bearing, d)
    if best is None:
        d = np.linalg.norm(wp - [tx, ty], axis=1)
        i = int(d.argmin())
        dirv = np.array([wp[i, 0] - tx, wp[i, 1] - ty]); dirv /= (np.linalg.norm(dirv) + 1e-9)
        sx, sy = tx + dirv[0] * stand, ty + dirv[1] * stand
        return float(sx), float(sy), float(np.arctan2(ty - sy, tx - sx))
    return float(best[1]), float(best[2]), float(best[3])


def standoff_front(env, tx, ty, front_deg, wp, stand=0.85):
    th = np.radians(front_deg); d = np.array([np.cos(th), np.sin(th)])
    best = None
    for r in np.arange(0.7, 1.65, 0.05):
        sx, sy = tx - d[0] * r, ty - d[1] * r
        if env._occupied(sx, sy):
            continue
        dpath = float(np.min(np.hypot(wp[:, 0] - sx, wp[:, 1] - sy)))
        score = dpath + 0.3 * abs(r - stand)
        if best is None or score < best[0]:
            best = (score, float(sx), float(sy), r)
    if best is not None:
        return best[1], best[2], float(th)
    return None


# ============================================================
# 主流程（移植 lab_mission 状态机）
# ============================================================
def main():
    import torch
    import stage_b_eval as sb
    from train_goalnav_turn import status_gh

    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--targets", default=None, help="默认 lab_mission_targets_zed_front_perp_lvl.json")
    ap.add_argument("--ckpt", default="goalnav_gate_lab.pt")
    ap.add_argument("--reloc_server", default="http://10.0.0.20:8321")
    ap.add_argument("--vlm_url", default=None, help="默认本地 Qwen http://127.0.0.1:8152/api/generate")
    ap.add_argument("--cam_h", type=float, default=1.2, help="相机离地高(真机实测, 默认1.2; 根治尺度债后1.71)")
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--dry", dest="dry", action="store_true", default=True, help="默认开: 不发指令+mock观测")
    ap.add_argument("--no-dry", dest="dry", action="store_false", help="真机模式: 发指令+读ZED+真reloc")
    ap.add_argument("--precise_dock", action="store_true", help="confirm 后贴脸精对接")
    ap.add_argument("--reloc", action="store_true", help="到站 + 巡航周期 RELOC 纠偏")
    args = ap.parse_args()

    # ---------- 资产加载（离线资产，拷到 Orin） ----------
    ply = os.path.join(ROOT, "outputs", f"gs_vggto_{args.session}_world.ply")
    xyz = sb.read_ply_xyz(ply)
    _, _, _, floor = sb.build_occ(xyz)          # 只拿 floor（地面 z）
    tj = args.targets or os.path.join(ROOT, "outputs", "lab_mission_targets_zed_front_perp_lvl.json")
    T = json.load(open(tj))
    _cap = np.load(os.path.join(ROOT, "outputs", f"sim_world_{args.session}.npz"))
    wp_cap, yaw_cap = _cap["waypoints_xy"], _cap["yaw"]
    # 占据栅格: 与仿真 CorridorTestEnv 同口径（build_occupancy + 沿轨迹清路），否则 A* 堵死
    occ, lo, res = build_occupancy_real(ply, res=0.08, robot_r=0.3, traj=wp_cap)
    clear_path(occ, lo, res, wp_cap, ROBOT_R + 0.45)
    print(f"[real] 资产: ply {xyz.shape}, occ {occ.shape} free{100*(occ==0).mean():.0f}%, targets {list(T)[:6]}", flush=True)

    # ---------- 真机接口 ----------
    zed = ZedSource(dry=args.dry)
    twist = TwistPublisher(dry=args.dry)
    env = RealEnv(zed, twist, xyz, floor, occ, lo, res, dry=args.dry)
    reloc = RelocClient(args.reloc_server, dry=args.dry, cam_h=args.cam_h)

    # ---------- 快脑 ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sb.load_model(device)
    from train_goalnav import robot_kmeans_anchors
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    ck = torch.load(os.path.join(RUNS, args.ckpt), map_location="cpu")
    model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
    model._status_encoding.load_state_dict(ck["status_encoding"]); model.eval()
    print(f"[real] 快脑 ckpt={args.ckpt} device={device}", flush=True)

    # ---------- 站位 ----------
    FRONT = T.get("front_deg", {})
    _STATIONS = [k for k in T if k not in ("origin", "front_deg", "front_method")]
    _origin = tuple(T.get("origin", [0.0, 0.0])[:2])
    so = {}
    for name in _STATIONS:
        if name in FRONT:
            r = standoff_front(env, float(T[name][0]), float(T[name][1]), FRONT[name], wp_cap)
            so[name] = r if r else find_standoff(env, float(T[name][0]), float(T[name][1]), wp_cap, yaw_cap,
                                                 avoid=[v[:2] for v in so.values()])
        else:
            so[name] = find_standoff(env, float(T[name][0]), float(T[name][1]), wp_cap, yaw_cap,
                                     avoid=[v[:2] for v in so.values()])
        print(f"  站位 {name}: ({T[name][0]:.2f},{T[name][1]:.2f}) -> ({so[name][0]:.2f},{so[name][1]:.2f}) "
              f"yaw{np.degrees(so[name][2]):.0f}°", flush=True)

    # ---------- 任务腿 ----------
    legs = []
    if set(_STATIONS) == {"iron_cabinet", "microscope", "pipette_table"}:
        legs = [("A*", _origin, so["iron_cabinet"][:2], "goto_cabinet"),
                ("STRAFE", so["iron_cabinet"][:2], so["microscope"][:2], "strafe_microscope"),
                ("A*", so["microscope"][:2], so["pipette_table"][:2], "goto_pipette"),
                ("A*", so["pipette_table"][:2], _origin, "return_origin")]
    else:
        _prev = _origin
        for _nm in _STATIONS:
            legs.append(("A*", _prev, so[_nm][:2], f"goto_{_nm}")); _prev = so[_nm][:2]
        legs.append(("A*", _prev, _origin, "return_origin"))

    # ---------- 快脑特征构造（与 lab_mission.dd_traj 同口径） ----------
    def dd_traj(fpv_bgr, pose, goal_ego, head_ego=(1.0, 0.0), hist_w=0, hist_gamma=0.85, vx=0.4, mode=0.0):
        st = status_gh(np.asarray(goal_ego, np.float32), np.asarray(head_ego, np.float32), bs_vx=vx, device=device)
        st[0, 5] = hist_w / 8.0; st[0, 6] = hist_gamma; st[0, 7] = mode
        cam = sb.camera_feature(fpv_bgr, device)
        bev = sb.lidar_feature_fov(xyz, pose, floor, device)     # 已知场景: BEV 从离线世界 ply 取点（同仿真）
        torch.manual_seed(0)
        with torch.no_grad():
            return model({"camera_feature": cam, "lidar_feature": bev, "status_feature": st})["trajectory"][0].cpu().numpy()

    def sim_fpv():
        rgb, _ = zed.capture()
        return rgb[:, :, ::-1].copy()                            # BGR（与 lab_mission fpv 口径）

    # ---------- clearance 间隙场 ----------
    _freem = (env.occ == 0).astype(np.uint8)
    _clear = cv2.distanceTransform(_freem, cv2.DIST_L2, 5).astype(np.float32) * env.res
    def clearance(x, y):
        ix = (x - env.lo[0]) / env.res; iy = (y - env.lo[1]) / env.res
        i0, j0 = int(np.floor(iy)), int(np.floor(ix))
        if i0 < 0 or j0 < 0 or i0 >= _clear.shape[0] - 1 or j0 >= _clear.shape[1] - 1:
            return 0.0
        return float(_clear[i0, j0])

    def safe_vel(x, y, yaw, vx, wz, dt=0.25):
        c, s = np.cos(yaw), np.sin(yaw)
        nx = x + vx * c * dt; ny = y + vx * s * dt
        cl = clearance(nx, ny)
        if cl >= ROBOT_R + 0.25:
            return vx, wz, False
        best_th, best_cl = None, -1
        for a in np.linspace(-np.pi, np.pi, 16, endpoint=False):
            clc = clearance(x + 0.4 * np.cos(a), y + 0.4 * np.sin(a))
            if clc > best_cl:
                best_cl, best_th = clc, a
        e = wrap(best_th - yaw)
        return float(np.clip(vx, 0, 0.25)) * (cl / (ROBOT_R + 0.25)), float(np.clip(1.2 * e, -0.8, 0.8)), True

    # ---------- 状态机 ----------
    SPEED = args.speed
    def cruise(path, tol=0.40, maxs=300):
        obs = env._obs()
        wz_prev = 0.0
        for _i in range(maxs):
            bx, by, byaw = env.x, env.y, env.yaw
            pose = (env.x, env.y, env.yaw)
            fpv = np.ascontiguousarray(sim_fpv()[:, :, ::-1])
            dist_goal = float(np.hypot(bx - path[-1, 0], by - path[-1, 1]))
            if dist_goal < tol:
                break
            if dist_goal > 1.0:
                look = min(GOAL_AHEAD, max(1.0, dist_goal))
                d = np.hypot(path[:, 0] - bx, path[:, 1] - by); i0 = int(d.argmin())
                jj = i0; acc = 0.0
                while jj + 1 < len(path) and acc < look:
                    acc += float(np.linalg.norm(path[jj + 1] - path[jj])); jj += 1
                aim = path[jj]
            else:
                aim = path[-1]
            head_err = wrap(np.arctan2(aim[1] - by, aim[0] - bx) - byaw)
            c_, s_ = np.cos(byaw), np.sin(byaw)
            gx, gy = aim - np.array([bx, by])
            goal_ego = np.array([c_ * gx + s_ * gy, -s_ * gx + c_ * gy], np.float32)
            traj = dd_traj(fpv, pose, goal_ego, hist_w=0, hist_gamma=0.85, vx=0.5)
            dd_vx, _, dd_wz = traj_to_control(traj)
            wz = 0.75 * float(np.clip(1.5 * head_err, -0.7, 0.7)) + 0.25 * float(np.clip(dd_wz, -0.7, 0.7))
            turn_slow = float(np.clip(1.0 - abs(head_err) / 1.0, 0.15, 1.0))
            goal_slow = float(np.clip(dist_goal / 1.2, 0.35, 1.0))
            vx = float(np.clip(abs(dd_vx), 0.2, 1.0)) * SPEED * turn_slow * goal_slow
            nx = env.x + vx * np.cos(env.yaw) * 0.25; ny = env.y + vx * np.sin(env.yaw) * 0.25
            if clearance(nx, ny) < ROBOT_R + 0.25:
                vx *= float(np.clip(clearance(nx, ny) / (ROBOT_R + 0.25), 0.2, 1.0))
            wz = 0.5 * wz_prev + 0.5 * wz; wz_prev = wz
            wz = float(np.clip(wz, -0.7, 0.7))
            if _i % 10 == 0:
                print(f"    [cruise] d={dist_goal:.1f}m v={vx:.2f} w={wz:+.2f} he={np.degrees(head_err):.0f}°", flush=True)
            obs, _, _ = env.step(vx, 0.0, wz)
        return obs

    TURN_DD_MIN = 0.9
    def face(target_xy, maxs=90):
        obs = env._obs()
        for _ in range(maxs):
            e = wrap(np.arctan2(target_xy[1] - env.y, target_xy[0] - env.x) - env.yaw)
            if abs(e) < 0.04:
                break
            wz = float(np.clip(1.5 * e, -0.8, 0.8))       # 控制器比例（真机先转后走）
            obs, _, _ = env.step(0.0, 0.0, wz)
        return obs

    def align_front(name, max_iter=5):
        if name not in FRONT:
            return face(T[name][:2])
        th = np.radians(FRONT[name]); dirv = np.array([np.cos(th), np.sin(th)])
        tx, ty = float(T[name][0]), float(T[name][1])
        obs = env._obs()
        for _ in range(90):                                # 1) 转到 front yaw
            e = wrap(th - env.yaw)
            if abs(e) < 0.02:
                break
            obs, _, _ = env.step(0.0, 0.0, float(np.clip(1.5 * e, -0.7, 0.7)))
        for it in range(max_iter):                         # 2) 横移上轴闭环
            bxp, byp, byawp = env.x, env.y, env.yaw
            pos = np.array([bxp, byp])
            s = float((pos - [tx, ty]) @ dirv)
            closest = np.array([tx, ty]) + s * dirv
            lat = pos - closest
            L = float(np.linalg.norm(lat))
            berr = np.degrees(wrap(np.arctan2(ty - byp, tx - bxp) - th))
            print(f"    [ALIGN] {name} it{it}: 横向{L:.2f}m bearing-front={berr:+.1f}°", flush=True)
            if abs(berr) < 1.5 or L < 0.02:
                break
            step = -lat if L <= 0.18 else -lat * 0.18 / L
            if clearance(bxp + step[0], byp + step[1]) < ROBOT_R + 0.10:
                break
            c, sg = np.cos(byawp), np.sin(byawp)
            bx = float(step @ np.array([c, sg])); by = float(step @ np.array([-sg, c]))
            dist = float(np.linalg.norm(step)); nsteps = max(1, int(dist / 0.06))
            for _ in range(nsteps):
                obs, _, _ = env.step(bx / max(dist, 1e-6) * 0.24, by / max(dist, 1e-6) * 0.24, 0.0)
        return env._obs()

    def reloc_correct(name):
        """到站 RELOC 纠偏（HTTP）。返回纠正量 (drift_m, residual_cm) 或 None。"""
        rgb, depth = zed.capture()
        bx, by, byaw = env.x, env.y, env.yaw
        res = reloc.correct(depth, bx, by, np.degrees(byaw))
        if res is None:
            print(f"    [RELOC] {name}: 跳过", flush=True)
            return None
        nx, ny, nyaw, accepted = res
        dx, dy = nx - bx, ny - by
        drift0 = float(np.hypot(dx, dy))
        if not accepted:
            print(f"    [RELOC] {name}: 拒收(loss>0.08), 保持 believed drift={drift0:.2f}m", flush=True)
            return None
        env.x, env.y = nx, ny
        env.yaw = float(wrap(np.radians(nyaw)))
        print(f"    [RELOC] {name}: drift={drift0:.2f}m → 纠正后 believed=({nx:.2f},{ny:.2f},{nyaw:.1f}°)", flush=True)
        return drift0, 0.0

    def vlm_confirm(name):
        """四段推理链 confirm（HTTP VLM）。真机用真图。简化: Scene + Action 两段 VLM。"""
        rgb, _ = zed.capture()
        img = np.ascontiguousarray(rgb[:, :, ::-1])
        tx, ty = float(T[name][0]), float(T[name][1])
        dist = float(np.hypot(env.x - tx, env.y - ty))
        term = name.replace("_", " ")
        ans = vlm_ask(img, f"Is there {term} in this image? Answer yes or no.")
        ok = ans.lower().startswith("y")
        print(f"    [VLM] {name}: d={dist:.2f}m → {ans or '(err)'} -> {'YES' if ok else 'NO'}", flush=True)
        return ok

    def vlm_ask(img_bgr, prompt, max_new_tokens=16):
        import requests
        url = args.vlm_url or os.environ.get("VLM_URL", "http://127.0.0.1:8152/api/generate")
        try:
            ok, buf = cv2.imencode(".jpg", img_bgr)
            files = [("images", ("f.jpg", buf.tobytes(), "image/jpeg"))]
            data = {"prompt": prompt, "max_new_tokens": str(max_new_tokens), "do_sample": "false"}
            r = requests.post(url, files=files, data=data, timeout=40)
            return (r.json().get("answer", "") or "").strip()
        except Exception:
            return ""

    # ---------- 主执行 ----------
    if args.dry:
        print("[real] DRY-RUN：验证加载 + 站位 + A* 规划，不执行运动", flush=True)
    env.start = (0.0, 0.0, 0.0); env.reset()

    for leg_i, (mode, a, b, tag) in enumerate(legs):
        name = tag
        if mode == "A*":
            env.start = (a[0], a[1], 0.0); env.goal = (b[0], b[1])
            p = plan_path(env)
            print(f"  leg {tag}: A* {tuple(np.round(a,2))}->{tuple(np.round(b,2))} "
                  f"{'OK '+str(len(p))+'pts' if p is not None else 'FAIL'}", flush=True)
        else:
            print(f"  leg {tag}: STRAFE dist={np.hypot(b[0]-a[0],b[1]-a[1]):.2f}m", flush=True)
        if args.dry:
            continue                                   # dry-run 只验证规划，不进运动循环
        # ---- 真机执行 ----
        if mode == "A*" and p is not None and len(p) > 1:
            cruise(p, tol=(0.15 if tag.startswith("goto_") else 0.40))
        st_name = tag.replace("goto_", "").replace("strafe_", "")
        if st_name in _STATIONS:
            if args.reloc:
                reloc_correct(st_name)
            align_front(st_name)
            vlm_confirm(st_name)

    print("[real] 完成", flush=True)


if __name__ == "__main__":
    main()
