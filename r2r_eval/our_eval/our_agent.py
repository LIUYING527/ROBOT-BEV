#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
我们的方法(VLM-WHERE + 几何反投影 + 条件DiffusionDrive-HOW)在 LHPR-VLN 上的评测代理。

与 DualVLN 共用 LH-VLN 官方 SceneSimulator/动作空间/指标(同一 run_eval 骨架),
差异只在 agent 本体 —— 同协议公平对照, zero-shot(我们ckpt训练域=实验室走廊, 未见过HM3D)。

数据流(每步):
  habitat front RGB ──► camera_feature(1024x256) ─┐
  habitat front depth ─► 反投影世界点累积图 ─► lidar_feature_fov(±32m官方直方图) ─► 快脑 → traj
  VLM(robo2vlm) "Detect {target}" → bbox ─► bbox中心深度反投影 ─► goal_ego(status[0:2])
  traj → traj_to_control → (vx,vy,wz) → 量化: |wz|>0.25→turn, else forward
  目标不可见且无缓存 → turn_left 扫描; goal距离<1.05m → stop(交官方1m+60°FoV判定)

坐标系转换(habitat y-up 右手系 → 我们 z-up 平面, x前y左):
  我们(x,y,z) = (habitat x, -habitat z, habitat y); th = atan2(-fwd_z, fwd_x), fwd = R@(0,0,-1)。
  ⚠0825探针实证: y 必须取 -hab_z, 否则平面镜像(左转th反号/左点ey=-1); floor=position.y
  (habitat agent 原点在地面, 传感器在其上 SENSOR_H=1.0m, 勿再减)。
"""
import os
import sys
import re
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_LHVLN = os.path.dirname(_THIS)
_ARCH = '/data/DongBaorong/BEVLOOK/robot_bev_sim/arch_e2e/pipeline'
_SCRIPTS = '/data/DongBaorong/BEVLOOK/robot_bev_sim/scripts'
_GOALNAV = '/data/DongBaorong/BEVLOOK/robot_bev_sim/goalnav/pipeline'
# _THIS 也加进去(0914): escape_v3.py 与本文件同目录, 不依赖调用方是否已把 our_eval 入 path
for p in (_THIS, _LHVLN, _ARCH, _SCRIPTS, _GOALNAV):
    sys.path.insert(0, p)

import stage_b_eval as sb  # noqa: E402
import sim_fullchain as sf  # noqa: E402  vlm_api(默认自研10.0.0.18)
from train_goalnav_turn import status_gh  # noqa: E402
# 脱困模块(0914移植自 best_fastbrain_0913/escape_v3.py, 纯numpy零依赖)
from escape_v3 import Escape, EscapeV2, EscapeV3  # noqa: E402

RUNS = '/data/DongBaorong/BEVLOOK/robot_bev_sim/goalnav/runs'

# ⭐0920 诊断开关: habitat 的 rgbd_agent 实际 sensor position[1] = 1.25
#   (habitat/config/default_structured_configs.py:1370)。我们训练+推理一直用 1.0 ⇒
#   点云 z 系统性偏低 0.25m；而 sim_bev 走 use_ground_plane=False + (ez>0.2) 出图
#   ⇒ 真实高度 0.2~0.45m 的矮障碍被整体丢弃（实测占有效点 5.8%）。
#   默认 1.0 = 历史行为逐位不变；SENSOR_H_OVERRIDE=1.25 可做 A/B 诊断。
SENSOR_H = float(os.environ.get('SENSOR_H_OVERRIDE', '1.0'))
HFOV = 90.0
STOP_DIS = 1.05      # 官方判定 1m, 略留余量
WZ_TURN_TH = 0.25    # |wz|超过则执行离散转向(30°)
SCAN_EVERY = 1       # 无目标时每步都扫描转向


def parse_bbox(ans):
    """bbox 解析 -> (cxn, cyn, wn, hn) 归一化, 或 None(沿用 locate_target 铁律: 只信bbox)。

    0914: 兼容两种慢脑后端的【不同输出格式】—— 此前只认 robo2vlm 的, 换成 Qwen3.5-4B 后
    定位 100% 返回 None(实测 600/600), 整条真链路形同没接慢脑:
      ① robo2vlm(自研, 10.0.0.18):      直接吐归一化 [x1,y1,x2,y2] (0~1)
      ② Qwen3.5-4B(本地对照, 127.0.0.1): ```json [{"bbox_2d":[...], "label":"door"}] ```,
         坐标是 Qwen-VL 惯例的 0~1000 空间, 且外面裹着 JSON/markdown —— 旧的
         "取前4个数字 + 必须<=1.5" 会直接判 None。
    """
    if not ans:
        return None
    s = str(ans)
    # ② 先显式抓 Qwen 的 bbox_2d 字段: 避免把 label 文本里的数字也当成坐标抓进来
    m = re.search(r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)', s)
    if m:
        x1, y1, x2, y2 = [float(v) for v in m.groups()]
        if max(x1, y1, x2, y2) <= 1.5:            # 万一某天它改成 0~1, 也认
            pass
        else:
            x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
    else:
        # ① robo2vlm 原始路径(行为保持不变)
        nums = re.findall(r"[-+]?\d*\.?\d+", s)
        if len(nums) < 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in nums[:4]]
        if max(x1, y1, x2, y2) > 1.5:
            return None
    x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
    w, h = x2 - x1, y2 - y1
    if w < 0.01 or h < 0.01 or w > 0.98 or h > 0.98:
        return None
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2), w, h)


class OurAgent:
    def __init__(self, ckpt='goalnav_gate_lab_v4.pt', device='cuda', vlm_url=None,
                 hist_w=0, hist_gamma=0.85, vx=0.25, vlm_every=3, stop_head=None,
                 ground_head=None, e2e=False, stop_thr=0.40, stop_votes=2,
                 stop_mode='head', stop_eps=0.5, hfov=90.0,
                 locator='vlm', infinitevl_url='http://localhost:8123',
                 escape=3, v3_noprog=2, v3_minclear=1.5, v3_unknown=2.5,
                 v3_levelsat=False, v3_fwd_v2=False, turn_deg=30.0,
                 geo_aux=False, geo_aux_d=1.0):
        import torch
        self.torch = torch
        self.locator = locator                      # 'vlm' | 'infinitevl'（慢脑 InfiniteVL HTTP）
        self.infinitevl_url = infinitevl_url
        # ---- 脱困(0914移植): 与 best_fastbrain_0913/oracle 探针同配方 ----
        # escape: 0=关 / 1=v1原语 / 2=v2最空方向 / 3=v3规划式(默认, 全量79.6%的配方)
        self.escape_mode = escape
        # 0914修: escape=0 时 .get() 返回 None, 再 () 会 TypeError —— 关脱困这条路径必须能走通
        self.escape = None if escape == 0 else {1: Escape, 2: EscapeV2, 3: EscapeV3}.get(escape, EscapeV3)()
        EscapeV3.NO_PROG_N = v3_noprog          # 连续N次脱困无改善→大范围重定位(2=定版)
        EscapeV3.MIN_CLEAR = v3_minclear        # "这方向走得通"的最小净空
        EscapeV3.UNKNOWN_CLEAR = v3_unknown     # "无观测"方向的等效净空(不能当最空)
        EscapeV3.LEVEL_SAT = bool(v3_levelsat)  # True=消融②: level退回v2饱和
        EscapeV3.FWD_V2 = bool(v3_fwd_v2)       # True=消融④: fwd退回v2步数
        # ⚠️ TURN_DEG 保持 30: habitat vln_r2r.yaml 实际 15°/步, 但历史数字全在 30 下产出,
        # 且 0913 实测 30 优于 15(+6.0pt@100条单层失败集)。不要当 bug 去"修正"。
        Escape.TURN_DEG = float(turn_deg)
        # ---- 判停兜底(0914) ----
        # 0914 实测: chain_done 后若不判停会一直"前方3m"走下去 → 到过目标却停不下
        # (OSR 80% 而 SR 44%, 36pt 全丢在这)。geo_aux = 用"最后一次到达地标"的世界坐标
        # 做几何兜底: 链走完 + confirm 未过 + 离该点 < geo_aux_d → 停。
        self.geo_aux = bool(geo_aux)
        self.geo_aux_d = float(geo_aux_d)
        self.device = device
        self.model = sb.load_model(device)
        from train_goalnav import robot_kmeans_anchors
        self.model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
        ck = torch.load(os.path.join(RUNS, ckpt), map_location='cpu')
        # ⭐1005 anchor 数可变: 优先用 ckpt 自带的 plan_anchor(训练时真实 anchor)。
        #   形状不同则重建 Parameter。旧代码在此会 RuntimeError size mismatch
        #   (旧路径默认 24 锚点, 从未遇到 20 锚点的 ckpt)。
        #   对既有 24 锚点 ckpt 与旧行为【逐位等价】: load_state_dict 本就会覆盖 plan_anchor。
        _pa = ck['trajectory_head'].get('plan_anchor', None)
        if _pa is not None and tuple(_pa.shape) != tuple(self.model._trajectory_head.plan_anchor.shape):
            self.model._trajectory_head.register_parameter(
                'plan_anchor', torch.nn.Parameter(_pa.detach().float().to(device), requires_grad=False))
        # ⭐1015: 形状不符的键跳过(典型: text_proj 当 ckpt 是用别的 PWS_TEXT_DIM 训的)。
        #   不加这层过滤时, in_features 不符会直接 RuntimeError size mismatch。
        #   形状全匹配时该过滤不生效 ⇒ 旧行为逐位不变。(与 train_goalnav_r2r_dense.py:552-557 同写法)
        _cur = self.model._trajectory_head.state_dict()
        _th = {k: v for k, v in ck['trajectory_head'].items()
               if k not in _cur or tuple(_cur[k].shape) == tuple(v.shape)}
        _drop = [k for k in ck['trajectory_head'] if k not in _th]
        if _drop:
            print(f'[our_agent] 跳过形状不符的键: {_drop}', flush=True)
        self.model._trajectory_head.load_state_dict(_th, strict=False)
        self.model._status_encoding.load_state_dict(ck['status_encoding'])
        self.model.eval()
        self.hist_w = hist_w
        self.hist_gamma = hist_gamma
        self.vx = vx
        self.hfov = hfov                  # 0902: dense ckpt训练用79, 旧ckpt=90(训练部署一致)
        self.vlm_every = vlm_every          # VLM节流: 每N步查一次(目标世界点有缓存)
        self.stop_thr = stop_thr            # stop头阈值(0827定版0.40; e2e每步评需更严)
        self.stop_votes = stop_votes        # stop头连续票數(0827定版2; e2e假停率高→3)
        self.stop_mode = stop_mode          # e2e判停: head=stop头 / traj=轨迹幅度(原生到达信号)
        self.stop_eps = stop_eps            # traj模式: 末waypoint位移<e判到点(米)
        if vlm_url:
            os.environ['VLM_URL'] = vlm_url
        # stop 头(0827 路B): R2R train split ref_path 终点监督训的判停二分类,
        # 替换 VLM confirm(0826诊断实锤: 瓶颈=STOP通路, VLM元问题全no/接地问假停)
        self.stop_head = None
        if stop_head and os.path.exists(stop_head):
            import timm
            sh = timm.create_model('resnet34', pretrained=False, num_classes=1)
            ck2 = torch.load(stop_head, map_location='cpu')
            sh.load_state_dict(ck2['model'])
            sh.eval().to(device)
            self.stop_head = sh
            print(f'[agent] stop_head loaded: {stop_head} '
                  f'auc={ck2.get("val_auc"):.3f} thr={self.stop_thr} votes={self.stop_votes}', flush=True)
        # 接地头(0828 T1): LH-VLN 官方轨迹+深度连通域监督, (图,类名)->bbox+存在性。
        # 对照实锤: Qwen bbox 召回仅 0.02(98%答 none)+cerr 0.50, 头 R0.60@P0.25+cerr 0.118;
        # 5ms/次 vs Qwen 2.3s/次, 且 VLM 服务器挂了也能定位
        self.ground_head = None
        self.ground_vocab = {}
        if ground_head and os.path.exists(ground_head):
            ck3 = torch.load(ground_head, map_location='cpu')
            sys.path.insert(0, _THIS)
            from train_grounding_head import GroundHead
            gh = GroundHead(len(ck3['vocab']))
            gh.load_state_dict(ck3['model'])
            gh.eval().to(device)
            self.ground_head = gh
            self.ground_vocab = ck3['vocab']
            print(f'[agent] ground_head loaded: {ground_head} '
                  f'{len(self.ground_vocab)} classes', flush=True)
        # T5 e2e(0829): BERT 指令编码器(冻结), 与 train_inst_il_r2r 同口径(max_length=64, [CLS])
        self.e2e = e2e
        self._bert_tok = self._bert = None
        if e2e:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            from transformers import AutoTokenizer, AutoModel
            self._bert_tok = AutoTokenizer.from_pretrained('bert-base-uncased')
            self._bert = AutoModel.from_pretrained('bert-base-uncased').to(device).eval()
            for p_ in self._bert.parameters():
                p_.requires_grad = False
            print('[agent] e2e mode: BERT instruction encoder loaded', flush=True)
        self.reset_episode()

    def reset_episode(self):
        self.world_pts = np.zeros((0, 3), np.float32)   # 累积世界点图 (x=hx, y=hz, z=hy)
        self.target_world = None                          # VLM+几何定位的目标世界点
        self.miss_count = 0
        self.near_count = 0                               # 连续近距帧数(stop消抖, 0803复核经验)
        self.step_i = 0
        self.pos_log = []                                 # 卡死检测用位姿日志(仅前进帧, 0825修: 转向不计)
        self.unstick_left = 0                             # 剩余解卡转向步数(_act_impl LH-VLN模式仍用)
        self._last_act = None
        # ---- 脱困状态(0914) ----
        if self.escape is not None:
            self.escape.reset()
        self.n_escape = 0                                 # 被逃脱序列接管的累计步数
        self._prev_xy = None                              # moved口径: 本步位姿-上步位姿(与探针一致)
        self._esc_goal_ego = None                         # 上一帧goal_ego缓存(escape吃的是上一步信息)
        self._esc_dtg = None                              # 上一帧到目标距离(仅供v3无进展计数)
        self._last_reached_xy = None                      # 最后一次"到达地标"的世界坐标(判停兜底用)
        # R2R 模式状态
        self.r2r_subgoal = None                           # 慢脑给的当前地标(文本)
        self.r2r_stop_votes = 0                           # 连续 stop 票数(2票才停)
        self.r2r_history = []                             # 最近慢脑决策(防复读)
        self.r2r_slow_every = 8                           # 慢脑节拍(步)
        # v1(0826): 到达事件与复读熔断 —— 修 smoke 全败根因(206/207复读GOTO, 0 STOP)
        self.r2r_arrivals = {}                            # 子目标名 -> 到达次数
        self.r2r_banned = set()                           # confirm否决后禁止再GOTO的目标
        # v2(0826): 地标链慢脑 —— v1多选决策被robo2vlm复读病废掉(GOTO复读→TURN复读),
        # 回归已验证甜区: bbox定位+yes/no确认(0715走廊geo_vlm经验)
        self.r2r_chain = None                             # 地标链(首步VLM一次性解析)
        self.r2r_li = 0                                   # 当前地标下标
        self.r2r_scan = 0                                 # 当前地标连续未定位次数
        self.r2r_vantage = 0                              # 当前地标换视角次数(≥2找不到→跳过)
        self.r2r_confirm_cd = 0                           # 终态confirm冷却到该步
        self._tw_is_landmark = False                      # target_world来源: True=地标定位/False=虚拟前视点
        self._tw_steps = 0                                # 当前target_world已追逐步数(超时放弃用)
        self._loc_cand = None                             # 定位候选(ego点), 连续2次一致才采信(v5)
        self._loc_cand_n = 0
        self._cf_votes = 0                                # confirm连续yes票(2票才停, v5)
        # T5 e2e(0829): 指令BERT[CLS]->text_proj->每层FiLM, 无慢脑无goal_ego
        self._e2e_tf = None                             # 当前episode指令的BERT[CLS](cuda fp32)
        self._e2e_ins = None

    def _vlm(self, rgb, q, **kw):
        """sf.vlm_api 的容错封装(0828: robo2vlm 远端宕机→500 扩散毒化 61/300 episode;
        VLM 挂了应退化为'无回答'(agent 继续扫描/探索), 不能让整条 episode 报废)"""
        try:
            return sf.vlm_api(rgb, q, **kw)
        except Exception as e:
            if not getattr(self, '_vlm_down_logged', False):
                print(f'[agent] VLM unreachable, degrade to no-answer: {e}', flush=True)
                self._vlm_down_logged = True
            return None, ''

    def vlm_confirm(self, rgb, target):
        """到位确认(四段推理链经验的精简版): 近景是目标才准stop。≥1票yes。"""
        q = (f"Look at the image. Is there a {target} right in front of the camera, "
             "within arm's reach (about 1 meter)? Answer yes or no.")
        ans, _ = self._vlm(rgb, q, max_new_tokens=16, timeout=40)
        return bool(ans) and ans.strip().lower().startswith('yes')
    # ---------- 坐标转换 ----------
    @staticmethod
    def _rot_matrix(rotation):
        """agent_state.rotation -> 3x3 (兼容 magnum.Quaternion 与 np.quaternion)"""
        if hasattr(rotation, 'to_matrix'):
            return np.array(rotation.to_matrix())[:3, :3]
        import quaternion as npq
        return npq.as_rotation_matrix(rotation)

    @staticmethod
    def _pos_xyz(position):
        """agent_state.position -> (x,y,z) (兼容 magnum.Vector3 与 ndarray)"""
        return float(position[0]), float(position[1]), float(position[2])

    @classmethod
    def hab_pose_to_ours(cls, position, rotation):
        """habitat agent state -> 我们 (x, y, th, floor)。forward = R @ (0,0,-1)
        0825探针: y=-hab_z 才不是镜像; habitat agent 原点=地面, floor=position.y"""
        R = cls._rot_matrix(rotation)
        fwd = R @ np.array([0.0, 0.0, -1.0])
        th = np.arctan2(-fwd[2], fwd[0])         # 我们平面 (x=hx, y=-hz)
        px, py, pz = cls._pos_xyz(position)
        return px, -pz, float(th), py            # py=地面高度(传感器在其上SENSOR_H)

    def depth_to_world(self, depth, position, rotation, stride=4):
        """front depth(H,W米) → 世界点(我们坐标系)。habitat相机: x右 y上 z后(看向-Z)"""
        H, W = depth.shape
        fx = (W / 2.0) / np.tan(np.deg2rad(self.hfov / 2.0))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
        d = depth[::stride, ::stride].reshape(-1)
        u = us.reshape(-1).astype(np.float32)
        v = vs.reshape(-1).astype(np.float32)
        m = (d > 0.3) & (d < 12.0)
        d, u, v = d[m], u[m], v[m]
        # 相机系(OpenGL: -Z前, +Y上)
        xc = (u - cx) * d / fx
        yc = -(v - cy) * d / fx
        zc = -d
        R = self._rot_matrix(rotation)
        pts_cam = np.stack([xc, yc, zc], 1)
        px, py, pz = self._pos_xyz(position)
        pts_hab = pts_cam @ R.T + np.array([px, py + SENSOR_H, pz])
        ours = np.stack([pts_hab[:, 0], -pts_hab[:, 2], pts_hab[:, 1]], 1).astype(np.float32)
        return ours

    def head_locate(self, rgb, depth, target):
        """接地头定位: (图,类名)->bbox中心深度反投影 ego(x前,y左) 或 None(未检出/类未见)。
        与 vlm_locate 同几何, 只是 bbox 来源换成训练头"""
        tid = self.ground_vocab.get(target)
        if tid is None or self.ground_head is None:
            return None, 'no_vocab'                      # 类未涵盖 → 调用方退回 VLM
        import cv2
        im = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        t = self.torch.from_numpy(im.copy()).float().permute(2, 0, 1)[None] / 255.0
        mean = self.torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std = self.torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        with self.torch.no_grad():
            pb, pl = self.ground_head((t.to(self.device) - mean) / std,
                                      self.torch.tensor([tid], device=self.device))
        if float(self.torch.sigmoid(pl)[0]) < 0.3:       # th0.3: R0.60@P0.25, 靠2帧一致杀FP
            return None, 'absent'
        x1, y1, x2, y2 = [float(v) for v in pb[0].cpu().numpy()]
        cxn, cyn = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        H, W = depth.shape
        uu, vv = int(cxn * W), int(cyn * H)
        wn, hn = max(x2 - x1, 0.02), max(y2 - y1, 0.02)
        du, dv = max(2, int(0.15 * wn * W)), max(2, int(0.15 * hn * H))
        patch = depth[max(0, vv - dv):vv + dv, max(0, uu - du):uu + du]
        patch = patch[(patch > 0.3) & (patch < 12.0)]
        if patch.size == 0:
            return None, 'no_depth'
        d = float(np.median(patch))
        fx = (W / 2.0) / np.tan(np.deg2rad(self.hfov / 2.0))
        y_left = -(uu - (W - 1) / 2.0) * d / fx
        return np.array([d, y_left], np.float32), 'ok'

    # ---------- VLM 定位 ----------
    def infinitevl_locate(self, rgb, depth, target, room=None):
        """慢脑 InfiniteVL 定位：把子任务目标写成指令 → HTTP 拿归一化像素 → 深度反投影 → ego(x前,y左)。

        与 VLM 路径的区别：直接出像素坐标（不要 bbox），是慢脑训练的原生接口。
        ⚠️ 跨域提醒：InfiniteVL 在 640×480/fov79/h=1.30 口径上训练，而 LH-VLN 是
           512×512/fov90/h=1.0 —— 反投影必须用【当前相机】的 hfov(90)，但模型输出的
           归一化 ny 带有训练口径的系统偏差（见 docs/slowbrain-deep-audit-0910.md P1）。
        """
        import base64
        import requests
        where = f" in the {room}" if room else ""
        instruction = f"go to the {target}{where}"
        try:
            r = requests.post(self.infinitevl_url + '/locate', json={
                'rgb': base64.b64encode(np.ascontiguousarray(rgb).tobytes()).decode(),
                'shape': list(rgb.shape), 'instruction': instruction}, timeout=60).json()
        except Exception as e:
            if not getattr(self, '_ivl_down_logged', False):
                print(f'[agent] InfiniteVL unreachable: {e}', flush=True)
                self._ivl_down_logged = True
            return None
        if not r.get('visible') or r.get('nx') is None:
            return None
        H, W = depth.shape
        u = int(round(r['nx'] * W)); v = int(round(r['ny'] * H))
        u = max(0, min(u, W - 1)); v = max(0, min(v, H - 1))
        patch = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        patch = patch[(patch > 0.3) & (patch < 12.0)]
        if patch.size == 0:
            return None
        d = float(np.median(patch))
        fx = (W / 2.0) / np.tan(np.deg2rad(self.hfov / 2.0))
        y_left = -(u - (W - 1) / 2.0) * d / fx
        return np.array([d, y_left], np.float32)

    def vlm_locate(self, rgb, depth, target, room=None):
        """VLM bbox + 深度反投影 -> 目标 ego(x前,y左) 或 None。
        0825: 带 room 上下文消歧(LH-VLN 目标多实例, 只用类名会锁错实例)"""
        if getattr(self, 'locator', 'vlm') == 'infinitevl':
            return self.infinitevl_locate(rgb, depth, target, room)
        where = f" in the {room}" if room else ""
        q = (f"Detect the {target}{where}. Output its bounding box as [x1,y1,x2,y2] normalized 0-1. "
             "If not visible, answer none.")
        ans, _ = self._vlm(rgb, q, max_new_tokens=48, timeout=40)
        bb = parse_bbox(ans)
        if bb is None:
            return None
        cxn, cyn, wn, hn = bb
        H, W = depth.shape
        uu, vv = int(cxn * W), int(cyn * H)
        du, dv = max(2, int(0.15 * wn * W)), max(2, int(0.15 * hn * H))
        patch = depth[max(0, vv - dv):vv + dv, max(0, uu - du):uu + du]
        patch = patch[(patch > 0.3) & (patch < 12.0)]
        if patch.size == 0:
            return None
        d = float(np.median(patch))
        fx = (W / 2.0) / np.tan(np.deg2rad(self.hfov / 2.0))
        x_fwd = d
        y_left = -(uu - (W - 1) / 2.0) * d / fx
        return np.array([x_fwd, y_left], np.float32)

    # ---------- R2R 路径指令模式(慢脑=指令跟踪, 快脑=现有几何+轨迹) ----------
    def vlm_r2r_parse_chain(self, rgb, instruction):
        """v5: 两个聚焦问题(终点+中途地标)替代一次复合问(VLM复合问不稳: 出过动词短语/ hallucinate)。
        返回 [中途..., 终点], 空=解析失败"""
        # 0914修: 原 prompt 强制要求说出"最终地点", 但像 "Go down the stairs and stop." 这种
        # 指令【根本没有终点物体】, 模型被逼着编一个(实测 →'sofa'; "stop just passed the
        # doorway" →'piano')。允许它答 none, 幻觉才有出口。
        q1 = (f"Navigation instruction: \"{instruction}\"\n"
              "What is the FINAL place or object the robot must stop at? "
              "Answer with ONE short noun phrase (like 'pool', 'kitchen table'), no verbs. "
              "If the instruction only tells the robot to stop (e.g. \"go down the stairs and stop\") "
              "without naming any place or object at the end, answer exactly: none")
        a1, _ = self._vlm(rgb, q1, max_new_tokens=24, timeout=40)
        final = (a1 or '').strip().strip('.').splitlines()[0].strip() if a1 else ''
        q2 = (f"Navigation instruction: \"{instruction}\"\n"
              "Before reaching the final destination, which objects or places explicitly named "
              "in the instruction must the robot pass or go to first? List them IN ORDER, "
              "one short noun phrase per line. If none, answer none.")
        a2, _ = self._vlm(rgb, q2, max_new_tokens=48, timeout=40)
        out = []
        for ln in (a2 or '').strip().splitlines():
            ln = ln.strip().strip('-.•*0123456789) ').strip()
            for part in ln.split(','):
                part = part.strip()
                if part and len(part) < 40 and part.lower() not in ('none', 'n/a', 'and', 'no'):
                    out.append(part)
        if final and len(final) < 40 and final.lower() not in ('none', 'n/a'):
            out.append(final)
        return out[:6]

    def vlm_r2r_decide(self, rgb, instruction):
        """慢脑决策: 指令+当前视图 -> ('GOTO', landmark)|('FORWARD',)|('TURN_LEFT',)|('TURN_RIGHT',)|('STOP',)
        v1(0826): 注入到达事件上下文+ban列表, 破"复读GOTO永不说STOP"死循环"""
        hist = '; '.join(self.r2r_history[-3:]) or 'none'
        arrived = ', '.join(f'{k}×{v}' for k, v in self.r2r_arrivals.items()) or 'none'
        banned = ', '.join(sorted(self.r2r_banned)) if self.r2r_banned else None
        q = (f"You are guiding a robot through a house. Navigation instruction: \"{instruction}\"\n"
             f"Recent decisions: {hist}\n"
             f"Places the robot has ALREADY REACHED: {arrived}\n"
             + (f"Do NOT go to these places again: {banned}\n" if banned else "")
             + "Look at the current camera view. Choose the ONE best immediate action:\n"
             "- GOTO <object or place visible in the image that lies on the instructed path>\n"
             "- FORWARD (the path ahead is clear and follows the instruction)\n"
             "- TURN_LEFT or TURN_RIGHT (must reorient to follow the instruction)\n"
             "- STOP (the robot has reached the final destination of the instruction; "
             "if it just arrived at the place the instruction ends at, choose STOP)\n"
             "Answer with exactly one line.")
        ans, _ = self._vlm(rgb, q, max_new_tokens=48, timeout=40)
        if not ans:
            return ('FORWARD',)
        line = ans.strip().splitlines()[0].strip().upper()
        if line.startswith('GOTO') and len(line) > 5:
            return ('GOTO', line[4:].strip(' .:'))
        for k in ('STOP', 'TURN_LEFT', 'TURN_RIGHT', 'FORWARD'):
            if line.startswith(k):
                return (k,)
        return ('FORWARD',)

    def vlm_r2r_confirm(self, rgb, instruction):
        """终态确认(v3修): 不问"任务完成没"(抽象元问题,VLM全答no, 0826实证站在GT上7cm也no),
        改问接地近距问题=终地标是否就在眼前(LH-VLN模式vlm_confirm同款, 已验证可靠)"""
        landmark = self.r2r_chain[-1] if self.r2r_chain else None
        if landmark:
            q = (f"Look at the image. Is a {landmark} visible right in front of the camera, "
                 "nearby (within about 2 meters)? Answer yes or no.")
        else:
            q = (f"Navigation instruction: \"{instruction}\"\n"
                 "Look at the current camera view. Has the robot REACHED the final destination "
                 "and completed this instruction? Answer yes or no.")
        ans, _ = self._vlm(rgb, q, max_new_tokens=16, timeout=40)
        return bool(ans) and ans.strip().lower().startswith('yes')

    def stop_head_prob(self, rgb):
        """stop头打分: rgb uint8 -> P(到点可停)。与 train_stop_head 同预处理"""
        import cv2
        im = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        t = self.torch.from_numpy(im.copy()).float().permute(2, 0, 1) / 255.0
        mean = self.torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = self.torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        with self.torch.no_grad():
            logit = self.stop_head(((t - mean) / std).unsqueeze(0).to(self.device))
        return float(self.torch.sigmoid(logit)[0, 0])

    def _confirm_stop(self, rgb, instruction, near_geo=False):
        """N帧连续yes才准停(⑪key-frame复核经验: 单帧confirm会假yes, 0826 ep0第2步假停实证)。
        stop头在则用头(推理~5ms, 调用方不再冷却, 每步都评); 否则退回VLM confirm。
        阈值定版(0827 三档 smoke 实证): 0.40/2票 全场景统一 —— 0.55 不触发(近距均分仅0.42,
        ep1606 走满500步); 混合(无先验0.55/3票)SR 归0; 0.40档假停存在但<=VLM confirm 的
        退化假停(v5 ep1596 23步/ep1297 32步), 且探索更充分 SPL 最优"""
        if self.stop_head is not None:
            p = self.stop_head_prob(rgb)
            thr, need = self.stop_thr, self.stop_votes
            self._cf_votes = self._cf_votes + 1 if p >= thr else 0
            return self._cf_votes >= need
        if self.vlm_r2r_confirm(rgb, instruction):
            self._cf_votes += 1
            if self._cf_votes >= 2:
                return True
        else:
            self._cf_votes = 0
        return False

    def _world_pts_ego(self, x, y, th):
        """累积世界点图 → ego系局部点云(x前, y左), 下采样::6。
        与 probe_oracle_goal.py 喂 EscapeV2/V3 的构造逐位一致(同一个 world_pts, 同一步长)。"""
        wp = self.world_pts
        if len(wp) == 0:
            return None
        ws = wp[::6]
        dd = ws[:, :2] - np.array([x, y], np.float64)
        cc, ss = np.cos(th), np.sin(th)
        return np.stack([cc * dd[:, 0] + ss * dd[:, 1],
                         -ss * dd[:, 0] + cc * dd[:, 1]], 1).astype(np.float32)

    def _traj_to_act(self, traj, bend_th=15.0):
        # ⭐1012 PWS_BEND_TH: 动作阈值可调。默认不设 => 逐位不变。
        #   动机: 真实评测显示 22.5% 的步“该转却判成直行”(选中角落在 ±15° 内)。
        bend_th = float(os.environ.get('PWS_BEND_TH', bend_th))
        """0902: bend量化替代traj_to_control的yaw阈值。实测模型yaw通道输出≈0,
        转向全表达在轨迹横向位移(|vy|中位0.18), 旧量化把它当strafe丢弃 → 0/230假阴性。
        看末端waypoint弯曲角(y左为正), |bend|>15°转(habitat turn=15°)。"""
        # ⭐1008 解码器可切换(默认 tail = 旧行为, 逐位不变)。
        #   动机: agent 每步只走 0.25m, 而旧解码器用【1.6m 末端】点的角度决定动作 ——
        #   尺度不匹配, 且失败解剖显示失败组"59% 的步在只转不走"(多转)。
        #   实测(3420步): 末端解码 52.9% 的步在转; 近场(0.2m)解码只有 40.0%; 两者 14.1% 的步决策不同。
        _dec = os.environ.get('PWS_DECODER', 'tail')
        if _dec == 'head':
            _p = traj[0]
        elif _dec == 'seg1':
            _p = traj[0]
        elif _dec == 'pp':
            # 纯追踪: 沿轨迹找"离起点最近且弧长 >= L"的那个点
            _L = float(os.environ.get('PWS_PP_L', '0.6'))
            _seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 2), np.float32), traj[:, :2]], 0), axis=0), axis=1)
            _arc = np.cumsum(_seg)[1:]
            _k = int(np.searchsorted(_arc, _L))
            _k = min(max(_k, 0), len(traj) - 1)
            _p = traj[_k]
        else:
            _p = traj[-1]
        bend = float(np.degrees(np.arctan2(_p[1], max(_p[0], 1e-6))))
        if bend > bend_th:
            return 'turn_left', bend
        if bend < -bend_th:
            return 'turn_right', bend
        return 'move_forward', bend

    def act_r2r(self, observations, position, rotation, instruction):
        action, dbg = self._act_r2r_impl(observations, position, rotation, instruction)
        self._last_act = action
        return action, dbg

    def _act_r2r_impl(self, observations, position, rotation, instruction):
        rgb = np.ascontiguousarray(observations['color_sensor_f'][:, :, :3])
        depth = np.ascontiguousarray(observations['depth_sensor_f']).astype(np.float32)
        x, y, th, floor = self.hab_pose_to_ours(position, rotation)
        pts = self.depth_to_world(depth, position, rotation)
        self.world_pts = np.concatenate([self.world_pts, pts])[-600000:]
        self.step_i += 1

        # ---- 脱困优先(0914移植自 best_fastbrain_0913/oracle 探针) ----
        # 吃【上一步】的动作+实际位移; 命中则本步跳过慢脑与快脑, 直接执行逃脱序列。
        # 口径与 probe_oracle_goal.py 逐条对齐: moved=本步位姿-上步位姿(不论上步是什么动作),
        # escape 判定在模型之前, goal_ego 取上一帧缓存(escape 吃的本来就是上一步的信息)。
        # 旧逻辑(15帧位移<0.08 累积4次→直接giveup停)已被替换 —— 它只会放弃, 不会自救。
        moved = 0.0 if self._prev_xy is None else float(
            np.hypot(x - self._prev_xy[0], y - self._prev_xy[1]))
        self._prev_xy = (x, y)
        if self.escape is not None and self._last_act is not None and self._esc_goal_ego is not None:
            if self.escape_mode in (2, 3):
                e = self.escape.update(self._last_act, moved, self._esc_goal_ego, self.step_i,
                                       ego_pts=self._world_pts_ego(x, y, th), dtg=self._esc_dtg)
            else:
                e = self.escape.update(self._last_act, moved, self._esc_goal_ego, self.step_i,
                                       dtg=self._esc_dtg)
            if e is not None:
                self.n_escape += 1
                return e, f'escape#{self.n_escape} L{self.escape.level} {e}'

        # ================= v2 慢脑: 地标链 + bbox定位 + yes/no confirm =================
        # (v1逐节拍多选决策弃用: robo2vlm复读病, GOTO/TURN/FORWARD轮换复读, 0 STOP)
        if self.r2r_chain is None:
            self.r2r_chain = self.vlm_r2r_parse_chain(rgb, instruction)
            self.r2r_history.append('CHAIN: ' + ' -> '.join(self.r2r_chain or ['<empty>']))
        chain_done = self.r2r_li >= len(self.r2r_chain)

        if chain_done:
            # 链走完: confirm 判停(stop头=每步评, VLM=8步冷却防连问), 2票制, 未过则前进探索
            if self.stop_head is not None or self.step_i >= self.r2r_confirm_cd:
                if self._confirm_stop(rgb, instruction, near_geo=False):
                    return 'stop', 'r2r chain done + confirm YES×2'
                if self.stop_head is None:
                    self.r2r_history.append('confirm no @chain_done')
                    self.r2r_confirm_cd = self.step_i + 8
            # 0914 判停兜底: confirm 未过, 但慢脑已把最后一个地标判为"到达"且我们仍在其附近
            # → 停。修的是"到了 0.10~0.12m 却停不下来、继续"前方3m"走开"这个 36pt 的黑洞。
            # 只在真的走到过地标(而非跳过)时才可能成立 —— 距离条件天然保证。
            if self.geo_aux and self._last_reached_xy is not None:
                d_lm = float(np.hypot(x - self._last_reached_xy[0], y - self._last_reached_xy[1]))
                if d_lm < self.geo_aux_d:
                    return 'stop', f'chain done + geo-aux stop d_landmark={d_lm:.2f}'
            if self.target_world is None:
                c, s = np.cos(th), np.sin(th)
                self.target_world = np.array([x + 3.0 * c, y + 3.0 * s], np.float32)
                self._tw_is_landmark = False
                self._tw_steps = 0
        else:
            cur = self.r2r_chain[self.r2r_li]
            # 当前地标定位(节流)。v5: 追踪中直接刷新; 新目标要连续2次世界系一致(<1m)才采信(杀单发幻觉bbox)
            if self.step_i % self.vlm_every == 0:
                g = self.vlm_locate(rgb, depth, cur)
                if g is not None:
                    c, s = np.cos(th), np.sin(th)
                    gw = np.array([x + c * g[0] - s * g[1], y + s * g[0] + c * g[1]], np.float32)
                    if self.target_world is not None:
                        self.target_world = gw
                        self._tw_is_landmark = True
                        self._tw_steps = 0
                        self.r2r_scan = 0
                    elif self._loc_cand is not None and np.hypot(*(gw - self._loc_cand)) < 1.0:
                        self.target_world = gw                       # 候选复核通过, 采信
                        self._tw_is_landmark = True
                        self._tw_steps = 0
                        self.r2r_scan = 0
                        self._loc_cand = None
                    else:
                        self._loc_cand = gw                          # 首见候选, 等下帧复核
                else:
                    self._loc_cand = None
                    if self.target_world is None:
                        self.r2r_scan += 1
            if self.target_world is None:
                # 未定位: 原地扫描; 扫8次(≈360°)找不到→前进3m换视角; 换2次还找不到→跳过该地标
                if self.r2r_scan >= 8:
                    self.r2r_vantage += 1
                    self.r2r_scan = 0
                    if self.r2r_vantage >= 2:
                        self.r2r_history.append(f'SKIP {cur} (not found)')
                        self.r2r_li += 1
                        self.r2r_vantage = 0
                        self._loc_cand = None
                        return 'turn_left', f'skip {cur}, next landmark'
                    c, s = np.cos(th), np.sin(th)
                    self.target_world = np.array([x + 3.0 * c, y + 3.0 * s], np.float32)
                    self._tw_is_landmark = False
                    self._tw_steps = 0
                    return 'move_forward', f'vantage#{self.r2r_vantage} fwd3m (locate {cur} failed)'
                return 'turn_left', f'scan {cur} (miss {self.r2r_scan}/8)'

        if self.target_world is None:
            return 'turn_left', 'r2r no target'

        # goal_ego + 到达判定(R2R 无对象真值, 目标到 1m 内即完成)
        c, s = np.cos(th), np.sin(th)
        dx, dy = self.target_world[0] - x, self.target_world[1] - y
        goal_ego = np.array([c * dx + s * dy, -s * dx + c * dy], np.float32)
        dist = float(np.hypot(*goal_ego))
        # 脱困缓存(0914): 供【下帧】escape 使用(escape 判定发生在慢脑之前, 那一刻只有上一步信息)。
        # ⚠️ 口径差异: 探针里 dtg = 到终点的距离(上帝视角), 这里 = 到【当前地标】的距离 ——
        #    它只喂给 v3 的"无进展计数"(dtg 改善 0.3m 就清零), 用当前地标推进表达同样语义。
        self._esc_goal_ego = goal_ego
        self._esc_dtg = dist
        self._tw_steps += 1
        is_final = (self.r2r_li == len(self.r2r_chain) - 1)
        # 终地标近距直通confirm: "stop before X"语义, bbox中心天然过冲(0826 ep0实证偏3.5m)
        if (self._tw_is_landmark and is_final and dist < 2.0
                and (self.stop_head is not None or self.step_i >= self.r2r_confirm_cd)):
            if self._confirm_stop(rgb, instruction, near_geo=True):
                return 'stop', f'r2r final near d={dist:.2f} + confirm YES×2'
            if self.stop_head is None:
                self.r2r_history.append(f'confirm no @final d={dist:.2f}')
                self.r2r_confirm_cd = self.step_i + 8
        # 目标超时(45步未到达=隔墙/隔家具不可达): 末地标且已近按接近即达, 否则放弃计locate失败
        if self._tw_steps > 45:
            was_landmark = self._tw_is_landmark
            self.target_world = None
            self._tw_is_landmark = False
            self._tw_steps = 0
            if was_landmark and is_final and dist <= 2.5:
                self.r2r_li += 1               # 接近即达 → 下步走 chain_done confirm
                return 'turn_left', f'final timeout-near d={dist:.2f}, as arrived'
            if was_landmark:
                self.r2r_scan += 1
            return 'turn_left', f'target timeout d={dist:.2f}, abandon'
        if dist < 1.0:
            was_landmark = self._tw_is_landmark
            self.target_world = None
            self._tw_is_landmark = False
            if was_landmark:
                sg = self.r2r_chain[self.r2r_li]
                self.r2r_arrivals[sg] = self.r2r_arrivals.get(sg, 0) + 1
                self.r2r_history.append(f'ARRIVED {sg}')
                # 0914: 记下到达点 —— 链走完后用它做几何兜底判停(慢脑说"到了"的地方就是目标处)
                self._last_reached_xy = np.array([x, y], np.float32)
                self.r2r_li += 1
                self.r2r_scan = 0
                self.r2r_vantage = 0
                self._loc_cand = None
                return 'turn_left', f'arrived {sg}, -> next'
            # 虚拟前视点到达: 继续(下步重新定位/扫描/confirm)
        # ========================================================================

        # 快脑出轨迹(与 LH-VLN 模式同)
        import cv2
        cam = sb.camera_feature(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.device)
        pose = np.array([x, y, th], np.float32)
        bev = sb.lidar_feature_fov(self.world_pts, pose, floor, self.device)
        k = max(abs(goal_ego[0]) / 5.0, abs(goal_ego[1]) / 2.0, 1.0)
        st = status_gh(goal_ego / k, np.array([1.0, 0.0], np.float32), bs_vx=self.vx, device=self.device)
        st[0, 5] = self.hist_w / 8.0
        st[0, 6] = self.hist_gamma
        st[0, 7] = 0.0
        self.torch.manual_seed(0)
        with self.torch.no_grad():
            traj = self.model({'camera_feature': cam, 'lidar_feature': bev,
                               'status_feature': st})['trajectory'][0].cpu().numpy()
        act, bend = self._traj_to_act(traj)
        return act, f'r2r d={dist:.2f} g=({goal_ego[0]:.1f},{goal_ego[1]:.1f}) bend={bend:.1f} sg={self.r2r_subgoal}'

    # ---------- T5 e2e: 指令直接调制策略, 无慢脑/无goal_ego ----------
    def act_r2r_e2e(self, observations, position, rotation, instruction):
        """T5(0829): 指令BERT[CLS]->text_proj->每层FiLM + cam/BEV -> 轨迹 -> 量化 + stop头。
        部署口径必须与 train_inst_il_r2r.run_batch 逐位一致:
        status 全零 + [4]=vx + [6]=hist_gamma (goal槽置0, 条件全走text_feature)。"""
        rgb = np.ascontiguousarray(observations['color_sensor_f'][:, :, :3])
        depth = np.ascontiguousarray(observations['depth_sensor_f']).astype(np.float32)
        x, y, th, floor = self.hab_pose_to_ours(position, rotation)
        pts = self.depth_to_world(depth, position, rotation)
        self.world_pts = np.concatenate([self.world_pts, pts])[-600000:]
        self.step_i += 1

        # 卡死检测(仅前进帧, 与_act_r2r_impl同口径)
        if self._last_act == 'move_forward':
            self.pos_log.append((x, y))
        if self.unstick_left > 0:
            self.unstick_left -= 1
            self._last_act = 'turn_left'
            return 'turn_left', f'unstick({self.unstick_left})'
        if len(self.pos_log) > 15:
            self.pos_log = self.pos_log[-15:]
            moved = np.hypot(self.pos_log[-1][0] - self.pos_log[0][0],
                             self.pos_log[-1][1] - self.pos_log[0][1])
            if moved < 0.08:
                self.stuck_cycles = getattr(self, 'stuck_cycles', 0) + 1
                self.pos_log = []
                if self.stuck_cycles >= 4:
                    self._last_act = 'stop'
                    return 'stop', f'giveup: 连续{self.stuck_cycles}轮卡死'
                self.unstick_left = 2
                self._last_act = 'turn_left'
                return 'turn_left', f'stuck#{self.stuck_cycles}, unstick'
        else:
            self.stuck_cycles = 0

        # 指令编码(每episode一次, 与训练同: max_length=64, [CLS], fp32)
        if self._e2e_tf is None or self._e2e_ins != instruction:
            batch = self._bert_tok([instruction], padding=True, truncation=True,
                                   max_length=64, return_tensors='pt').to(self.device)
            with self.torch.no_grad():
                self._e2e_tf = self._bert(**batch).last_hidden_state[:, 0].float()  # (1,768)
            self._e2e_ins = instruction

        # 快脑: cam+BEV+status(goal槽0, 条件全走text)+text_feature
        import cv2
        cam = sb.camera_feature(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.device)
        pose = np.array([x, y, th], np.float32)
        bev = sb.lidar_feature_fov(self.world_pts, pose, floor, self.device)
        st = self.torch.zeros(1, 8, device=self.device)
        st[0, 4] = self.vx
        st[0, 5] = self.hist_w / 8.0
        st[0, 6] = self.hist_gamma
        self.torch.manual_seed(0)
        with self.torch.no_grad():
            traj = self.model({'camera_feature': cam, 'lidar_feature': bev,
                               'status_feature': st,
                               'text_feature': self._e2e_tf})['trajectory'][0].cpu().numpy()
        # 判停放在轨迹之后: traj=轨迹幅度(oracle轨迹终点监督→近目标预测位移收缩=原生到达信号,
        # 假停双向误诊的stop头在e2e裸评下被实证弃用); head=stop头每步评
        if self.stop_mode == 'traj':
            tmag = float(np.linalg.norm(traj[-1, :2] - traj[0, :2]))
            self._cf_votes = self._cf_votes + 1 if tmag < self.stop_eps else 0
            if self._cf_votes >= self.stop_votes:
                self._last_act = 'stop'
                return 'stop', f'e2e traj-stop tmag={tmag:.2f}×{self.stop_votes} @step{self.step_i}'
        elif self._confirm_stop(rgb, instruction, near_geo=False):
            self._last_act = 'stop'
            return 'stop', f'e2e confirm YES×{self.stop_votes} @step{self.step_i}'

        vx, vy, wz = sf.traj_to_control(traj)
        if wz > WZ_TURN_TH:
            act = 'turn_left'
        elif wz < -WZ_TURN_TH:
            act = 'turn_right'
        else:
            act = 'move_forward'
        self._last_act = act
        return act, f'e2e step={self.step_i} v=({vx:.2f},{vy:.2f},{wz:.2f})'

    # ---------- 主接口: 给一帧观测, 出一个 LH-VLN 动作 ----------
    def act(self, observations, position, rotation, target_name, room=None):
        action, dbg = self._act_impl(observations, position, rotation, target_name, room)
        self._last_act = action
        return action, dbg

    def _act_impl(self, observations, position, rotation, target_name, room=None):
        """返回 ('move_forward'|'turn_left'|'turn_right'|'stop', debug str)"""
        rgb = np.ascontiguousarray(observations['color_sensor_f'][:, :, :3])
        depth = np.ascontiguousarray(observations['depth_sensor_f']).astype(np.float32)

        # 子任务切换检测(0828P0修): 目标变了必须清缓存——否则沿用上一子任务的
        # target_world, dist 天然小 → 秒假停(smoke 实证 steps=[9,1,1] 全假停)
        cur = (target_name, room)
        if getattr(self, '_cur_target', None) != cur:
            self._cur_target = cur
            self.target_world = None
            self.near_count = 0
            self.miss_count = 0
            self.pos_log = []
            self.stuck_cycles = 0
            self._loc_cand = None
            self._tw_is_landmark = False

        x, y, th, floor = self.hab_pose_to_ours(position, rotation)
        pts = self.depth_to_world(depth, position, rotation)
        self.world_pts = np.concatenate([self.world_pts, pts])[-600000:]
        self.step_i += 1

        # 0) 卡死检测与解卡(仅统计"想走却走不动"; 原地转向/扫描不算卡死, 0825修)
        if self._last_act == 'move_forward':
            self.pos_log.append((x, y))
        if self.unstick_left > 0:
            self.unstick_left -= 1
            return 'turn_left', f'unstick({self.unstick_left})'
        if len(self.pos_log) > 15:
            self.pos_log = self.pos_log[-15:]
            moved = np.hypot(self.pos_log[-1][0] - self.pos_log[0][0],
                             self.pos_log[-1][1] - self.pos_log[0][1])
            if moved < 0.08:
                self.stuck_cycles = getattr(self, 'stuck_cycles', 0) + 1
                self.pos_log = []          # 清零重计(原地转向不改变x,y, 否则永远再触发)
                if self.stuck_cycles >= 3:
                    return 'stop', f'giveup: 连续{self.stuck_cycles}轮卡死'
                self.unstick_left = 2      # 原地转60°找新路/新视角
                return 'turn_left', f'stuck#{self.stuck_cycles}, unstick'
        else:
            self.stuck_cycles = 0

        # 1) 定位当前子任务目标(节流: 每N步; 目标点世界缓存持续有效)。
        # 0828 T1: 接地头优先(5ms, R0.60), 类未涵盖/未检出才退回 VLM(2.3s, R0.02);
        # 新目标要连续2次世界系一致(<1m)才采信(杀头的单发FP, 与R2R v5同款复核)
        goal_ego = None
        if self.step_i % self.vlm_every == 0 or self.target_world is None:
            if self.ground_head is not None and target_name in self.ground_vocab:
                goal_ego, how = self.head_locate(rgb, depth, target_name)
                if goal_ego is None and how == 'no_vocab':
                    goal_ego = self.vlm_locate(rgb, depth, target_name, room)
            else:
                goal_ego = self.vlm_locate(rgb, depth, target_name, room)
        if goal_ego is not None:
            c, s = np.cos(th), np.sin(th)
            gx = x + c * goal_ego[0] - s * goal_ego[1]
            gy = y + s * goal_ego[0] + c * goal_ego[1]
            gw = np.array([gx, gy], np.float32)
            if self.target_world is not None:
                self.target_world = gw                # 追踪中: 直接刷新
                self._tw_is_landmark = True
                self.miss_count = 0
            elif self._loc_cand is not None and np.hypot(*(gw - self._loc_cand)) < 1.0:
                self.target_world = gw                # 候选复核通过, 采信
                self._tw_is_landmark = True
                self.miss_count = 0
                self._loc_cand = None
            else:
                self._loc_cand = gw                   # 首见候选, 等下帧复核
        else:
            self._loc_cand = None
            self.miss_count += 1

        # 2) 无目标 → 原地扫描; 扫满一圈(12×30°)找不到 → 前进2.5m换视角再扫
        # (0828 T1实证: 接地头比robo2vlm严格, 出生地不可见时原地转永不检出,
        #  VLM靠宽松bbox蒙混, 头需要移动到能看见的位置=R2R vantage逻辑移植)
        if self.target_world is None:
            if self.miss_count > 0 and self.miss_count % 12 == 0:
                c, s = np.cos(th), np.sin(th)
                self.target_world = np.array([x + 2.5 * c, y + 2.5 * s], np.float32)
                self._tw_is_landmark = False
                return 'move_forward', f'vantage#{(self.miss_count//12)} fwd2.5m (scan {self.miss_count} misses)'
            return 'turn_left', f'scan miss={self.miss_count}'

        # 3) goal_ego(世界点 -> 当前ego)
        c, s = np.cos(th), np.sin(th)
        dx, dy = self.target_world[0] - x, self.target_world[1] - y
        goal_ego = np.array([c * dx + s * dy, -s * dx + c * dy], np.float32)
        dist = float(np.hypot(*goal_ego))

        # 4) 近距精化+直停(0828 P0: 数据分析 NE1-2m 有54子任务0成功=确认否决丢对目标+远距锁点误差;
        #    NE<1m 的56个里50成功=几何直停本就可靠 → 弃VLM confirm(白桌掷硬币级)+LH-VLN stop头
        #    (AUC0.65小物体信号太弱), 改: 2.5m内逐步重定位精化(近距bbox误差小), 0.9m直停)
        if dist < 2.5:
            g_close = None
            if self.ground_head is not None and target_name in self.ground_vocab:
                g_close, how = self.head_locate(rgb, depth, target_name)
                # 在词表内且头判 absent → 信头不追问 VLM(0828v2: 否则近距每步2.3s Qwen,
                # 500步拖成17min/ep; 头的 absent 在在词表类上可靠)
                if g_close is None and how == 'no_vocab':
                    g_close = self.vlm_locate(rgb, depth, target_name, room)
            else:
                g_close = self.vlm_locate(rgb, depth, target_name, room)
            if g_close is not None:
                gx = x + c * g_close[0] - s * g_close[1]
                gy = y + s * g_close[0] + c * g_close[1]
                self.target_world = np.array([gx, gy], np.float32)
                dx, dy = self.target_world[0] - x, self.target_world[1] - y
                goal_ego = np.array([c * dx + s * dy, -s * dx + c * dy], np.float32)
                dist = float(np.hypot(*goal_ego))
        if dist < 0.9:
            if not self._tw_is_landmark:
                # 虚拟vantage点到达: 不停, 清目标继续扫描(0828 vantage补丁)
                self.target_world = None
                self.near_count = 0
                return 'turn_left', 'vantage reached, rescan'
            self.near_count += 1
            if self.near_count >= 2:                       # 连续2帧近距消抖
                return 'stop', f'reach d={dist:.2f} (geometric, 0828P0)'
        else:
            self.near_count = 0

        # 5) 快脑出轨迹 → 量化离散动作
        import cv2
        cam = sb.camera_feature(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.device)
        pose = np.array([x, y, th], np.float32)
        bev = sb.lidar_feature_fov(self.world_pts, pose, floor, self.device)
        # 保方向饱和: status_gh 内部按 SX=5/SY=2 独立 clip 会扭曲远目标方位角, 先等比缩到框内
        k = max(abs(goal_ego[0]) / 5.0, abs(goal_ego[1]) / 2.0, 1.0)
        st = status_gh(goal_ego / k, np.array([1.0, 0.0], np.float32), bs_vx=self.vx, device=self.device)
        st[0, 5] = self.hist_w / 8.0
        st[0, 6] = self.hist_gamma
        st[0, 7] = 0.0
        self.torch.manual_seed(0)
        with self.torch.no_grad():
            traj = self.model({'camera_feature': cam, 'lidar_feature': bev,
                               'status_feature': st})['trajectory'][0].cpu().numpy()
        act, bend = self._traj_to_act(traj)
        return act, f'd={dist:.2f} g=({goal_ego[0]:.1f},{goal_ego[1]:.1f}) bend={bend:.1f}'
