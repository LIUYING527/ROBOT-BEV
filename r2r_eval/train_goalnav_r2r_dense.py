#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R2R 快脑稠密扩训(0831, 追平DualVLN Phase3): dense_data/*(0.25m插值+jitter)流式训练。

与 train_goalnav_r2r.py 的差异:
  ① 数据源=dense_data(每帧含jitter位姿+dense_path), gt=相对**jitter位姿**的未来路径
     —— 治"永远直走"病(0831 oracle探针实锤: 旧数据朝向恒沿路径, 模型忽略goal条件)
  ② HFOV 90→79(与R2R渲染/评测一致; 旧管线90是失真, 训练部署一致故隐性, 重训一并修)
  ③ 流式: BEV预计算落盘bevs.npy(fp16, 增量累积O(n)), rgb训练时从npz解码,
     不再全量feats缓存(400k样本×1.7MB=680GB不可行)
  ④ DataLoader workers并行

用法(lhvln env):
  # 预计算BEV(CPU多进程, 数据生成完后跑):
  python train_goalnav_r2r_dense.py --mode precompute --workers 10
  # 训练(GPU):
  CUDA_VISIBLE_DEVICES=2 python train_goalnav_r2r_dense.py --mode train --epochs 12 --batch 32
"""
import os
import sys
import json
import glob
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_feat import instr_feature      # ⭐0930 指令编码(零外部依赖, 确定性)
# ⭐1006: PWS_TEXT_ENC=vlm 时改用 InfiniteVL-4B 的学习到的词嵌入(真语义); 默认仍用 crc32 词袋哈希 => 旧行为逐位不变
if os.environ.get('PWS_TEXT_ENC', '') == 'vlm':
    from text_feat_vlm import instr_feature
elif os.environ.get('PWS_TEXT_ENC', '') == 'ctx':
    from text_feat_ctx import instr_feature


_THIS = os.path.dirname(os.path.abspath(__file__))
_ARCH = '/data/DongBaorong/BEVLOOK/robot_bev_sim/arch_e2e/pipeline'
_SCRIPTS = '/data/DongBaorong/BEVLOOK/robot_bev_sim/scripts'
_GOALNAV = '/data/DongBaorong/BEVLOOK/robot_bev_sim/goalnav/pipeline'
for p in (_ARCH, _SCRIPTS, _GOALNAV):
    sys.path.insert(0, p)

RUNS = '/data/DongBaorong/BEVLOOK/robot_bev_sim/goalnav/runs'
HFOV = 79.0          # ⭐与R2R渲染/评测一致(旧90为失真)
GOAL_AHEAD = 4.0
T, DS = 8, 0.2


# ---------- 自包含几何(numpy only, 与 our_agent 逐行一致, 仅HFOV改79) ----------
def quat_wxyz(qxyzw):
    import quaternion as npq
    x, y, z, w = qxyzw
    return npq.quaternion(w, x, y, z)


def rot_matrix(q):   # q = quaternion module object
    import quaternion as npq
    return npq.as_rotation_matrix(q)


def densify_ours(ref_ours, ds=0.25):
    """稀疏路点 → 0.25m 稠密插值(与 eval_r2r_escape.densify_ours 同实现)。⭐1009 DAgger 用。"""
    pts = [np.array(ref_ours[0], np.float64)]; acc = 0.0
    for i in range(1, len(ref_ours)):
        a = np.array(ref_ours[i-1], np.float64); b = np.array(ref_ours[i], np.float64)
        seg = float(np.linalg.norm(b - a))
        if seg < 1e-6: continue
        d = ds - acc
        while d < seg:
            pts.append(a + (b - a) * (d / seg)); d += ds
        acc = seg - (d - ds)
    pts.append(np.array(ref_ours[-1], np.float64)); return np.stack(pts)


def hab_pose_to_ours(position, rotation):
    """habitat(pos,rot) → 我们系 (x,y,th,floor)。与 OurAgent.hab_pose_to_ours 一致: y=-hab_z"""
    R = rot_matrix(rotation)
    fwd = R @ np.array([0, 0, -1.0])
    th = np.arctan2(-fwd[2], fwd[0])
    return float(position[0]), float(-position[2]), float(th), float(position[1])


def depth_to_world(depth, position, R, stride=4):
    """front depth(米) → 世界点(我们系 xyz)。our_agent.depth_to_world 同逻辑, HFOV=79"""
    H, W = depth.shape
    fx = (W / 2.0) / np.tan(np.deg2rad(HFOV / 2.0))
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    d = depth[::stride, ::stride].reshape(-1)
    u = us.reshape(-1).astype(np.float32)
    v = vs.reshape(-1).astype(np.float32)
    m = (d > 0.3) & (d < 12.0)
    d, u, v = d[m], u[m], v[m]
    xc = (u - cx) * d / fx
    yc = -(v - cy) * d / fx
    zc = -d
    pts_cam = np.stack([xc, yc, zc], 1)
    px, py, pz = position[0], position[1], position[2]
    # ⭐0920: 同 our_agent —— habitat 真实 sensor height=1.25, 历史用 1.0(点云 z 偏低 0.25m)
    SENSOR_H = float(os.environ.get('SENSOR_H_OVERRIDE', '1.0'))
    pts_hab = pts_cam @ R.T + np.array([px, py + SENSOR_H, pz])
    return np.stack([pts_hab[:, 0], -pts_hab[:, 2], pts_hab[:, 1]], 1).astype(np.float32)


def fov_select_ego(xyz, pose, floor, fov_deg=110.0, R=32.0):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    dx = xyz[:, 0] - x; dy = xyz[:, 1] - y
    ex = c * dx + s * dy; ey = -s * dx + c * dy
    ez = xyz[:, 2] - floor
    ang = np.degrees(np.arctan2(ey, ex))
    m = (ex > 0.2) & (np.abs(ang) < fov_deg / 2) & (np.hypot(ex, ey) < R)
    return np.stack([ex[m], ey[m], ez[m]], 1)


def resample_ego(ref_ours, pose, vi):
    """dense_path[vi:](我们系) → ego系 8点×0.2m + 前瞻4m goal点。同 train_goalnav_r2r。"""
    x, y, th = pose
    fut = ref_ours[vi:]
    c, s = np.cos(th), np.sin(th)
    d = fut - np.array([x, y])
    ego = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
    seg = np.linalg.norm(np.diff(ego, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    targets = (np.arange(T) + 1) * DS
    out = np.zeros((T, 2), np.float32)
    for i, t in enumerate(targets):
        if t >= total:
            out[i] = ego[-1]
        else:
            j = int(np.searchsorted(arc, t))
            j = max(1, min(j, len(ego) - 1))
            r = (t - arc[j - 1]) / max(1e-9, arc[j] - arc[j - 1])
            out[i] = ego[j - 1] * (1 - r) + ego[j] * r
    if total <= GOAL_AHEAD:
        gp = ego[-1]
    else:
        j = max(1, int(np.searchsorted(arc, GOAL_AHEAD)))
        r = (GOAL_AHEAD - arc[j - 1]) / max(1e-9, arc[j] - arc[j - 1])
        gp = ego[j - 1] * (1 - r) + ego[j] * r
    # *0929 P2 重设计: 把 GT 从「沿 ref_path 的 1.6m 片段」改成「朝 4m 目标方向的 1.6m 直线」。
    #   动机(离线实测): 旧 GT 下有 12.1% 的帧是「1.6m 内直、4m 处要转弯」却被标成直走,
    #     而 _traj_to_act 正是用 traj[-1](1.6m 处) 判 turn => 短视界直接导致动作判错。
    #   本开关让 1.6m 处那个点「提前转向」, 且 **anchor 完全不动**(仍是 1.6m/8点/0.2m)。
    #   默认关 => 旧行为逐位不变。
    if os.environ.get("PWS_GT_4MDIR", "0") == "1":
        _n = float(np.linalg.norm(gp))
        if _n > 1e-6:
            _d = gp / _n
            out = (((np.arange(T) + 1) * DS)[:, None] * _d[None, :]).astype(np.float32)
    return out, gp.astype(np.float32)


def sample_turn_ang(sm):
    """样本的 goal 侧方角(度)=|atan2(gy,gx)|, 衡量"该帧需不需要转弯"。"""
    x, y, th, floor = hab_pose_to_ours(sm['pos'], quat_wxyz(sm['rot']))
    _, gp = resample_ego(sm['dense'], (x, y, th), sm['vi'])
    return float(abs(np.degrees(np.arctan2(gp[1], gp[0]))))


# ---------- 预计算BEV: 每ep顺序读npz, 增量累积世界点 → bevs.npy(fp16) ----------
def precompute_ep(job):
    epd, frames = job
    import sim_bev
    world = np.zeros((0, 3), np.float32)
    bevs = np.zeros((len(frames), 256, 256), np.float16)
    for fi, fr in enumerate(frames):
        z = np.load(os.path.join(epd, f"f{fr['fi']:03d}.npz"))
        R = rot_matrix(quat_wxyz(fr['rot']))
        pts = depth_to_world(z['depth'], np.asarray(fr['pos'], np.float64), R)
        world = np.concatenate([world, pts])[-600000:]
        x, y, th, floor = hab_pose_to_ours(fr['pos'], quat_wxyz(fr['rot']))
        ego = fov_select_ego(world, (x, y, th), floor)
        # *0928 BEV 收窄实验: lidar_range 可配(默认 32.0 逐位不变)。室内重训建议 4~6。
        #   动机: 默认 ±32m 时 0.25m/px, 室内场景包围盒中位仅 6.6m ⇒ 99.4% 像素是空的;
        #   range=6 时 0.047m/px, 有效像素涨 ~28 倍(sim_bev.py 注释亦建议"室内重训用 4~6")。
        _LR = float(os.environ.get('LIDAR_RANGE', '32.0'))
        bevs[fi] = sim_bev.official_lidar_feature(ego, lidar_range=_LR)[0].astype(np.float16)
    # ⭐0920: BEV_OUT 环境变量控制输出文件名/子目录 —— 默认 bevs.npy 逐位不变。
    #   诊断用 SENSOR_H_OVERRIDE=1.25 + BEV_OUT=bevs_h125.npy 并列生成, 不覆盖原数据。
    _out = os.environ.get('BEV_OUT', 'bevs.npy')
    _p = os.path.join(epd, _out)
    os.makedirs(os.path.dirname(_p), exist_ok=True) if os.path.dirname(_p) != epd else None
    np.save(_p, bevs)
    return epd, len(frames)


def collect_samples(data_dirs, stride):
    """返回 samples 列表: dict(npz, epd, fi, pos, rot, dense_ours, vi, scene)"""
    samples = []
    for d in data_dirs:
        for mp in glob.glob(os.path.join(d, 'meta.jsonl')):
            base = os.path.dirname(mp)
            for ln in open(mp):
                e = json.loads(ln)
                dp = np.array(e['dense_path'], np.float64)
                dense_ours = np.stack([dp[:, 0], -dp[:, 2]], 1)
                epd = os.path.join(base, f"ep{e['episode_id']}")
                frames = [{'fi': fi, 'vi': fr['vi'], 'pos': fr['pos'], 'rot': fr['rot'], 'dist': float(fr.get('dist_to_goal', -1.0))}
                          for fi, fr in enumerate(e['frames']) if fr.get('vi', -1) >= 0]
                for fr in frames[::stride]:
                    samples.append({'epd': epd, 'fi': fr['fi'], 'pos': fr['pos'],
                                    'rot': fr['rot'], 'dense': dense_ours, 'vi': fr['vi'],
                                    'goal_y': float(dp[-1, 1]),
                                    # ⭐0924: 该 ep 的 y跨度(垂直落差)。供 --height_oversample 按高度加权采样。
                                    #   实测分布偏移: 训练集 y跨度>2m 占 4.7%, 评测集占 9.5%(2倍),
                                    #   而该类 ep 八模型全票成功率仅 43%(其他类 ~93%)。
                                    'dy': float(dp[:, 1].max() - dp[:, 1].min()),
                                    'scene': e['scene'],
                                    # ⭐0930 语义条件: R2R 指令文本(dense_data 的 meta.jsonl 自带)。
                                    #   在此之前快脑**完全不吃指令**(text_proj 通路在位但从未传值)
                                    #   ⇒ 它是一个"朝目标点走"的控制器, 不是 VLN 模块。
                                    'instr': e.get('instruction', ''),
                                    'dist': fr.get('dist', -1.0), 'npz': os.path.join(epd, f"f{fr['fi']:03d}.npz")})

    # ⭐1009 DAgger: 追加【闭环 on-policy】的 OOD 状态样本。
    #   动机: 训练帧离参考路径中位仅 7cm/max 20cm, 模型从没见过偏离状态;
    #         实测失败时 agent 偏到 1~1.4m 外(36% 的 OOD 帧 >1m)。
    #   参考路径必须取【评测集 val_unseen_dense】(训练集 dense_data 的 episode_id 独立编号, 会张冠李戴)。
    #   默认不设 PWS_OOD_DIR ⇒ 完全不执行 ⇒ 旧行为逐位不变。
    _od = os.environ.get('PWS_OOD_DIR', '')
    if _od and os.path.isdir(_od):
        _meta = os.environ.get('PWS_OOD_META',
            '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/val_unseen_dense/meta.jsonl')
        _ref = {}
        for _ln in open(_meta, encoding='utf-8'):
            _e = json.loads(_ln); _rp = np.array(_e['ref_path'], np.float64)
            _ref[str(_e['episode_id'])] = densify_ours(np.stack([_rp[:, 0], -_rp[:, 2]], 1))
        _n0 = len(samples)
        for _ed in sorted(os.listdir(_od)):
            _r = _ref.get(_ed[2:])
            if _r is None: continue
            for _f in sorted(os.listdir(os.path.join(_od, _ed))):
                _p = os.path.join(_od, _ed, _f)
                try: _z = np.load(_p)
                except Exception: continue
                _x, _y, _th, _fl = hab_pose_to_ours(_z['pos'], quat_wxyz(_z['rot']))
                _dd = np.linalg.norm(_r - np.array([_x, _y]), axis=1)
                _vi0 = int(_dd.argmin())
                # ⚠️1009修: 必须【把 agent 当前位置插到路径最前】再交给 resample_ego。
                #   否则 ego[0]=ref[vi]-pose ≠ 0 ⇒ 整条目标轨迹被平移(实测偏 0.5m、方向被带偏),
                #   模型学到的是"从投影点往前走"而不是"从你所在处走回路径" ⇒ 实测 −20pt。
                _dense = np.concatenate([np.array([[_x, _y]], np.float64), _r[_vi0:]], 0)
                samples.append({'epd': os.path.join(_od, _ed), 'fi': int(_f[1:5]),
                                'pos': _z['pos'], 'rot': _z['rot'], 'dense': _dense,
                                'vi': 0, 'goal_y': float(_r[-1, 1]),
                                'dy': float(_r[:, 1].max() - _r[:, 1].min()),
                                'scene': '', 'instr': '', 'npz': _p, 'ood_npz': _p})
        print(f'[DAgger] 追加 OOD 样本 {len(samples) - _n0} 条 (共 {len(samples)})', flush=True)
    return samples


def mode_precompute(cli):
    jobs = []
    for d in cli.data:
        for mp in glob.glob(os.path.join(d, 'meta.jsonl')):
            base = os.path.dirname(mp)
            for ln in open(mp):
                e = json.loads(ln)
                epd = os.path.join(base, f"ep{e['episode_id']}")
                if os.path.exists(os.path.join(epd, os.environ.get('BEV_OUT', 'bevs.npy'))):
                    continue
                frames = [{'fi': fi, 'vi': fr['vi'], 'pos': fr['pos'], 'rot': fr['rot']}
                          for fi, fr in enumerate(e['frames']) if fr.get('vi', -1) >= 0]
                if frames:
                    jobs.append((epd, frames))
    print(f'[precompute] {len(jobs)} eps to do', flush=True)
    from multiprocessing import Pool
    done = 0
    with Pool(cli.workers) as pool:
        for epd, nf in pool.imap_unordered(precompute_ep, jobs):
            done += 1
            if done % 100 == 0:
                print(f'  {done}/{len(jobs)}', flush=True)
    print('[precompute] ALL DONE', flush=True)


def mode_train(cli):
    import cv2
    import torch
    from torch.utils.data import Dataset, DataLoader
    import stage_b_eval as sb
    from train_goalnav import robot_kmeans_anchors, with_yaw, smooth_traj
    from train_goalnav_turn import status_gh, turn_gt

    # ⭐0924 随机种子(默认 -1 = 不设 ⇒ 旧行为逐位不变)。
    #   动机: 实测同配置不同 run 的 SR 波动 ~±4pt —— h7b 四个 epoch 82.7/77.0/79.3/77.3,
    #   E3(b1wtd) 77.0/78.0/81.0/78.3。而同一 ckpt 的评测是确定性的(逐 ep 复现过),
    #   故波动来自训练侧。⇒ 跨 run 对照(如 h7b vs h7c)混入了数据顺序噪声, 无法分辨真实效应。
    #   原有随机源: ①训练循环每 batch 的 flip 增广 np.random.rand()<0.5 ②DataLoader shuffle
    #   ③cuDNN 自动算法选择。固定种子可消掉 ①②③。
    #   ⚠️ 必须在 load_model 之前设, 否则模型内任何随机初始化(如 clean_head 重建的 head)不可复现。
    _seed = int(getattr(cli, 'seed', -1))
    if _seed >= 0:
        import random as _rnd
        _rnd.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)
        # benchmark=False: 关掉 cuDNN 自动算法选择(它会随 warmup 结果漂移)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f'[seed] seed={_seed} 已固定 (random/np/torch/cuda + cudnn.deterministic)',
              flush=True)
    device = 'cuda'
    samples = collect_samples(cli.data, cli.stride)
    scenes = sorted({s['scene'] for s in samples})
    val_scenes = set(scenes[-cli.val_scenes:])
    print(f'[data] {len(samples)} samples, {len(scenes)} scenes '
          f'(val hold-out {len(val_scenes)})', flush=True)

    # ⭐0921: bevs.npy 全量预加载到内存。
    #   实测瓶颈 = HDD 随机 IO —— worker 8→24 时吞吐仅 16→31 样本/s(已饱和),
    #   加 worker 无效。而全部 bevs 合计约 68GB,本机 429GB 空闲内存足以容纳。
    #   主进程一次加载, fork 后各 worker 经 CoW 共享(只读不写) ⇒ 彻底绕开 HDD。
    #   ⚠️ 纯数据搬运, 不改变任何训练语义。
    bev_mem = None
    if cli.bev_mem:
        import time as _t
        _t0 = _t.time()
        _bevfile = os.environ.get('BEV_OUT', 'bevs.npy')
        bev_mem = {}
        for sm in samples:
            k = sm['epd']
            if k not in bev_mem:
                bev_mem[k] = np.load(os.path.join(k, _bevfile))
        _gb = sum(v.nbytes for v in bev_mem.values()) / 2 ** 30
        print(f'[bev_mem] 预加载 {len(bev_mem)} eps / {_gb:.1f} GB / {_t.time()-_t0:.0f}s',
              flush=True)

    class R2RDense(Dataset):
        def __init__(self, samples, idxs, turnaround_prob=0.0, bev_mem=None,
                     gt_lookahead=0.0, gt_lookahead_deg=60.0):
            self.samples = samples
            self.idxs = idxs
            self.turnaround_prob = turnaround_prob
            # ⭐0921 GT前瞻(见 __getitem__ 注释): 0 = 关(默认, 行为与旧版逐位相同)
            self.gt_lookahead = gt_lookahead
            self.gt_lookahead_deg = gt_lookahead_deg
            # ⭐0921: bev_mem 非空 ⇒ 直接查内存(首选); 否则退回 mmap 句柄缓存。
            self.bev_mem = bev_mem
            # ⭐0921: 缓存 bevs.npy 的 mmap 句柄。原代码每个样本都 np.load(mmap_mode='r')
            #   一次 —— 实测 HDD 上单次 open+header 解析 ≈112ms，是 dataloader 的真瓶颈
            #   (GPU util 全程 0%)。句柄复用后同一 ep 的连续帧不再重复 open。
            self._bevm = {}

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            sm = self.samples[self.idxs[i]]
            # ⭐0930 语义条件: 指令 -> 768 维固定向量(喂 DD 的 text_proj)。
            #   无论开关如何都照常返回, 保持样本元组形状稳定; 只是 PWS_TEXT=0 时不传给模型。
            _tx = torch.from_numpy(instr_feature(sm.get('instr', '')))
            z = np.load(sm['npz'])
            rgb = cv2.resize(z['rgb'], (1024, 256),  # camera_feature净效果=resize(RGB)
                             interpolation=cv2.INTER_LINEAR)
            cam = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            # ⭐1009 DAgger: OOD 样本【没有 bevs.npy】(它对应的 ep 是评测集虚拟目录),
            #   跳过 mmap 加载, BEV 直接从样本自带的 npz 取(见下)。默认无该键 ⇒ 旧行为逐位不变。
            if not sm.get('ood_npz'):
                if self.bev_mem is not None:
                    _arr = self.bev_mem[sm['epd']]          # ⭐内存路径(首选)
                else:
                    # ⭐0921: 复用 mmap 句柄(见 __init__ 注释)。BEV_OUT 支持退回默认 bevs.npy。
                    _bkey = (sm['epd'], os.environ.get('BEV_OUT', 'bevs.npy'))
                    _arr = self._bevm.get(_bkey)
                    if _arr is None:
                        if len(self._bevm) >= 64:          # 简单上限,防跨 ep 句柄堆积
                            self._bevm.clear()
                        _arr = np.load(os.path.join(_bkey[0], _bkey[1]), mmap_mode='r')
                        self._bevm[_bkey] = _arr
            # ⭐1009 DAgger: OOD 样本自带上文的 BEV(采集时从模型手里拿的),
            #   因为它对应的 ep 没有 bevs.npy。默认无该键 ⇒ 旧行为逐位不变。
            if sm.get('ood_npz'):
                _zo = np.load(sm['ood_npz'])
                bev = torch.from_numpy(_zo['bev'][0, 0].astype(np.float32)).unsqueeze(0)
            else:
                bev = torch.from_numpy(np.array(_arr[sm['fi']]).astype(np.float32)).unsqueeze(0)
            x, y, th, floor = hab_pose_to_ours(sm['pos'], quat_wxyz(sm['rot']))
            # ⭐0923 高度实验: 目标相对高度(ep终点y - 当前帧地面y)。开关默认关 ⇒ 下面 st[7] 仍填0, 旧行为逐位不变。
            _dyv = float(sm.get('goal_y', 0.0)) - float(floor)
            gt, gp = resample_ego(sm['dense'], (x, y, th), sm['vi'])
            # ⭐0909 掉头增广: 以概率把样本改成"goal侧后方 + 原地转GT", 治组2掉头难(数据掉头样本仅0.78%)。
            # 掉头是图像无关的 → 复用 cam/bev, 只改 goal 点方向 + GT 原地转(复用 turn_gt)。
            if self.turnaround_prob > 0 and np.random.rand() < self.turnaround_prob:
                dyaw = np.random.choice([-1.0, 1.0]) * np.random.uniform(np.deg2rad(120), np.pi)
                gp = np.array([np.cos(dyaw), np.sin(dyaw)], np.float32) * float(np.linalg.norm(gp))
                gt = turn_gt(dyaw)                       # (8,3) 原地转, x=y=0 yaw爬±dyaw
                return (cam, bev, torch.from_numpy(gt), torch.from_numpy(gp),
                        torch.tensor([_dyv], dtype=torch.float32), _tx,
                        torch.tensor([_dist2mode(sm.get('dist', -1.0))], dtype=torch.long))
            # ⭐0921 GT前瞻: 输出空间 / anchor / 模型结构**全不动**, 只让 GT 在"前方需要转弯"时提前朝目标偏。
            #   动机: 转弯是头号失败源(按路径转弯率分箱 SR 98%→39%, 转弯密集 ep 占失败 71%),
            #         而 1.6m 视界下 76.8% 的帧 GT 是直走的 —— goal 信息进不了训练信号。
            #   直路上 |4m转角|≈0 ⇒ alpha=0 ⇒ gt 与原来**逐位相同**(不破坏已有能力);
            #   弯道处 alpha 随转角增大 ⇒ GT 平滑提前偏, 把 goal 注入训练信号。
            #   ⚠️ 不换 anchor(换 anchor 实测净亏 25pt)。
            if self.gt_lookahead > 0:
                _L4 = float(np.hypot(gp[0], gp[1]))
                if _L4 > 1e-6:
                    # ⭐判据(两版试错后的定稿):
                    #   ① 用 4m **绝对**转角 ⇒ 96% 的帧都被改(4m 转角中位就有 12°, 路径总有微弯)
                    #   ② 用「超出量」= 4m−1.6m ⇒ 仍 94%(平缓转弯的 4m 转角天然大于 1.6m)
                    #   ③ 定稿: 只在「**1.6m 内还基本是直的** (<15°) 且 **4m 后确实要转弯** (>20°)」
                    #      时才介入 —— 这正是那 13.5% 的"短视界看不到转弯"帧。
                    #      (1.6m 内已在转弯的帧不用改, 模型本来就看得到)
                    _b16 = abs(float(np.arctan2(gt[-1, 1], max(float(gt[-1, 0]), 1e-6))))
                    _b4 = abs(float(np.arctan2(gp[1], gp[0])))
                    if _b16 < np.deg2rad(15.0) and _b4 > np.deg2rad(20.0):
                        _a = min(1.0, (_b4 - np.deg2rad(20.0))
                                 / np.deg2rad(self.gt_lookahead_deg)) * self.gt_lookahead
                    else:
                        _a = 0.0
                    if _a > 1e-6:
                        _dirv = (gp / _L4).astype(np.float32)
                        _gt_goal = (np.arange(1, T + 1, dtype=np.float32)[:, None] * DS) * _dirv[None, :]
                        gt = ((1.0 - _a) * gt + _a * _gt_goal).astype(np.float32)
            return (cam, bev, torch.from_numpy(with_yaw(smooth_traj(gt))), torch.from_numpy(gp),
                    torch.tensor([_dyv], dtype=torch.float32), _tx,
                    torch.tensor([_dist2mode(sm.get('dist', -1.0))], dtype=torch.long))

    tr_idx = [i for i, sm in enumerate(samples) if sm['scene'] not in val_scenes]
    va_idx = [i for i, sm in enumerate(samples) if sm['scene'] in val_scenes]
    # ⭐0924: 固定种子时给 DataLoader 显式 generator / worker_init_fn。
    #   generator ⇒ shuffle 顺序可复现; worker_init_fn ⇒ worker 内的 np.random(转弯增广用)可复现
    #   (DataLoader 只给 worker 播 torch 种子, 不播 numpy)。
    #   默认 -1 时 _ld_kw 为空 dict ⇒ DataLoader 收到的参数与旧版逐位相同。
    _ld_kw = {}
    if _seed >= 0:
        def _worker_init(worker_id):
            _ws = torch.initial_seed() % (2 ** 32)
            np.random.seed(_ws)
            import random as _rr
            _rr.seed(_ws)
        _ld_kw = {'generator': torch.Generator().manual_seed(_seed),
                  'worker_init_fn': _worker_init}
    if cli.turn_oversample > 0 or cli.height_oversample > 0:
        # ⭐0909 转弯过采样: goal侧方(>20°)样本按 --turn_oversample 倍加权采样。
        # 治"cls系统性低估转弯anchor"(ep1600/991/1772/444 型: 转弯anchor常驻top3但选不上)。
        # ⭐0924 高度过采样(新增): y跨度 > --height_thresh 的 ep 按 --height_oversample 倍加权。
        #   治「垂直跨越」硬骨头 —— 8模型全票失败的37条中 y跨度>2m 占多数, 该类 SR 仅 43%(其他类 ~93%);
        #   且实测**训练集 y跨度>2m 占 4.7%、评测集占 9.5%(2倍分布偏移)**。
        #   默认 height_oversample=0 ⇒ 下面第二个 if 不进入, 权重与旧版逐位相同(tr_w 初值为1, *= turn 等价 = turn)。
        from torch.utils.data import WeightedRandomSampler
        print(f'[oversample] 计算权重(turn×{cli.turn_oversample}, height×{cli.height_oversample}) ...', flush=True)
        tr_w = np.ones(len(tr_idx), np.float64)
        n_turn = n_hgt = 0
        for k, i in enumerate(tr_idx):
            if cli.turn_oversample > 0 and sample_turn_ang(samples[i]) > 20.0:
                tr_w[k] *= cli.turn_oversample
                n_turn += 1
            if cli.height_oversample > 0 and samples[i].get('dy', 0.0) > cli.height_thresh:
                tr_w[k] *= cli.height_oversample
                n_hgt += 1
        print(f'[oversample] 转弯样本 {n_turn}/{len(tr_idx)} = {100*n_turn/max(len(tr_idx),1):.1f}%'
              f' | y跨度>{cli.height_thresh}m 样本 {n_hgt}/{len(tr_idx)} = {100*n_hgt/max(len(tr_idx),1):.1f}%',
              flush=True)
        tr_ld = DataLoader(R2RDense(samples, tr_idx, cli.turnaround_prob, bev_mem, cli.gt_lookahead, cli.gt_lookahead_deg), batch_size=cli.batch,
                           sampler=WeightedRandomSampler(tr_w, num_samples=len(tr_idx),
                                                         replacement=True,
                                                         generator=(torch.Generator().manual_seed(_seed)
                                                                    if _seed >= 0 else None)),
                           num_workers=cli.workers, drop_last=True, pin_memory=True,
                           persistent_workers=True, **_ld_kw)
    else:
        tr_ld = DataLoader(R2RDense(samples, tr_idx, cli.turnaround_prob, bev_mem, cli.gt_lookahead, cli.gt_lookahead_deg), batch_size=cli.batch, shuffle=True,
                           num_workers=cli.workers, drop_last=True, pin_memory=True,
                           persistent_workers=True, **_ld_kw)
    va_ld = DataLoader(R2RDense(samples, va_idx, bev_mem=bev_mem, gt_lookahead=cli.gt_lookahead,
                                gt_lookahead_deg=cli.gt_lookahead_deg), batch_size=cli.batch, shuffle=False,
                       num_workers=2, pin_memory=True, persistent_workers=True, **_ld_kw)

    model = sb.load_model(device)
    if cli.clean_head:
        # ⭐0905 干净训头: 旧 R2R 系列 ckpt 的 plan_anchor 因历史 requires_grad=True 的 bug 全漂移
        # (goalnav_r2r_dense 漂 2.08m), head 与脏 anchor co-adapted。换干净 k24 anchor 后 warm-start
        # head 8ep+小lr 学不动(SR 50→3.4 实锤)。故【重新构造】 trajectory_head(随机初始化, 用官方
        # __init__ 的正确 init: cls bias=bias_init_with_prob(0.01)/reg 随机/scene_gate 零init),
        # 只保留 backbone(NAVSIM pretrained) + status_encoding(R2R warm-start, 不依赖 anchor)。
        from navsim.agents.diffusiondrive.transfuser_model_v2 import TrajectoryHead
        assert cli.anchor, '--clean_head 必须配合 --anchor 指定新 anchor npy'
        new_head = TrajectoryHead(
            num_poses=model._trajectory_head._num_poses,
            d_ffn=model._trajectory_head._d_ffn,
            d_model=model._trajectory_head._d_model,
            plan_anchor_path=cli.anchor,
            config=model._config,
        ).to(device)
        model._trajectory_head = new_head
        ck = torch.load(os.path.join(RUNS, cli.ckpt_in), map_location='cpu')
        model._status_encoding.load_state_dict(ck['status_encoding'])
    elif cli.partial_warmstart:
        # ⭐0909 半warm-start: 换正确anchor + warm去噪骨干(plan_anchor_encoder/time_mlp/diff_decoder
        # 除task_decoder外) + 随机init task_decoder(cls/reg输出, anchor相关)。
        # 破"坏anchor天花板"(dense plan_anchor 18/24塌缩-74°) 且避免 clean_head 完全随机init 20ep去噪塌陷。
        # 去噪骨干是anchor无关的(噪声anchor特征→轨迹恢复), 可warm; 只有cls/reg是anchor相关的需重学。
        from navsim.agents.diffusiondrive.transfuser_model_v2 import TrajectoryHead
        assert cli.anchor, '--partial_warmstart 必须配合 --anchor 指定新 anchor npy'
        new_head = TrajectoryHead(
            num_poses=model._trajectory_head._num_poses,
            d_ffn=model._trajectory_head._d_ffn,
            d_model=model._trajectory_head._d_model,
            plan_anchor_path=cli.anchor,
            config=model._config,
        ).to(device)
        ck = torch.load(os.path.join(RUNS, cli.ckpt_in), map_location='cpu')
        new_sd = new_head.state_dict()
        ck_sd = ck['trajectory_head']
        n_warm = 0
        # ⭐0922 变量隔离(默认关 ⇒ 原行为逐位不变)：
        #   _warm_td=True  ⇒ task_decoder 也 warm(不再随机 init)
        #   _keep_anchor=True ⇒ plan_anchor 用 ckpt 原值(不换 anchor)
        # 依据: goalmlp_ep5 / k24 / p1ae 的 plan_anchor 形状全是 (24,8,2), 只是值不同
        # ⇒ 「task_decoder 随机init」是设计选择而非形状限制 ⇒ 两个变量可独立控制。
        _warm_td = getattr(cli, 'pws_warm_td', False)
        _keep_anchor = getattr(cli, 'pws_keep_anchor', False)
        for k in new_sd:
            # ⭐0921修: 与 :324/:329 同根因 —— `'plan_anchor' not in k` 是【子串匹配】,
            #   会误杀 `plan_anchor_encoder.*` 的 6 个权重(实测 256×512 等),
            #   导致"warm 去噪骨干"实际是「anchor 编码器随机初始化」→ 去噪链路输入错乱。
            #   ⚠️ 0908 只修了主分支,漏了这里 ⇒ 0909 pwarm 实验(峰值 SR 29.3)的可信度存疑。
            if k == 'plan_anchor':
                continue
            if (not _warm_td) and 'task_decoder' in k:
                continue
            if k in ck_sd and new_sd[k].shape == ck_sd[k].shape:
                new_sd[k] = ck_sd[k]
                n_warm += 1
        if _keep_anchor:
            if 'plan_anchor' in ck_sd and new_sd['plan_anchor'].shape == ck_sd['plan_anchor'].shape:
                new_sd['plan_anchor'] = ck_sd['plan_anchor']   # 不换 anchor: 用 ckpt 原值
                n_warm += 1
            else:
                raise RuntimeError('--pws_keep_anchor: ckpt plan_anchor 与新建 head 形状不一致')
        new_head.load_state_dict(new_sd)
        model._trajectory_head = new_head
        model._status_encoding.load_state_dict(ck['status_encoding'])
        print(f'[partial-warmstart] anchor={"保持ckpt原值(不换)" if _keep_anchor else cli.anchor}, '
              f'warm {n_warm} 参数, task_decoder={"warm" if _warm_td else "随机init"}', flush=True)
    else:
        # ⭐0904修v2: ckpt 的 plan_anchor 是 24 个(旧anchor训练时赋的)，load_model 默认 20 个 →
        # load_state_dict 直接 size mismatch([24,8,2] vs [20,8,2])。先排除 plan_anchor 加载其余参数，
        # 再【重建】plan_anchor 为新 anchor(24个) Parameter 并冻结。
        ck = torch.load(os.path.join(RUNS, cli.ckpt_in), map_location='cpu')
        if cli.keep_anchor:
            # ⭐0908: 保持 ckpt_in 原 anchor(共适配好的), 不换 anchor。goal_mlp 实验专用。
            # 关键: plan_anchor 必须在 cuda 上(否则 forward add_noise 报 cpu/cuda 混)。
            pa = ck['trajectory_head']['plan_anchor'].to(device)
            # ⭐1007: 形状不符的键跳过(text_proj 等)。形状全匹配时该过滤不生效 ⇒ 旧行为不变。
            _cur = model._trajectory_head.state_dict()
            sd = {k: v for k, v in ck['trajectory_head'].items()
                  if k != 'plan_anchor' and (k not in _cur or tuple(_cur[k].shape) == tuple(v.shape))}
            _drop = [k for k in ck['trajectory_head'] if k not in sd and k != 'plan_anchor']
            if _drop: print(f'[load] 跳过形状不符的键: {_drop}', flush=True)
            model._trajectory_head.load_state_dict(sd, strict=False)
            model._trajectory_head.register_parameter(
                'plan_anchor', torch.nn.Parameter(pa, requires_grad=False))
        else:
            # ⭐1007: 形状不符的键跳过(text_proj 等)。形状全匹配时该过滤不生效 ⇒ 旧行为不变。
            _cur = model._trajectory_head.state_dict()
            sd = {k: v for k, v in ck['trajectory_head'].items()
                  if k != 'plan_anchor' and (k not in _cur or tuple(_cur[k].shape) == tuple(v.shape))}
            _drop = [k for k in ck['trajectory_head'] if k not in sd and k != 'plan_anchor']
            if _drop: print(f'[load] 跳过形状不符的键: {_drop}', flush=True)
            model._trajectory_head.load_state_dict(sd, strict=False)
            anchor = np.load(cli.anchor).astype(np.float32) if cli.anchor else robot_kmeans_anchors()
            model._trajectory_head.register_parameter(
                'plan_anchor', torch.nn.Parameter(torch.tensor(anchor, device=device), requires_grad=False))
        model._status_encoding.load_state_dict(ck['status_encoding'])
    for p in model.parameters():
        p.requires_grad = False
    # ⭐1006: 可学习码本(默认关) —— 让 plan_anchor 参与优化但用独立的小 lr。
    #   动机: 换码本值的代价(3~7pt)来自"全链共适应", 让码本自己参与适应应能消掉该代价;
    #   历史那次漂移2.08m 是【无约束】(lr 与 head 同)所致, 故此处独立小 lr。
    if getattr(cli, 'learn_anchor', 0):
        assert cli.anchor, '--learn_anchor 必须配 --anchor 指定初值'
    if cli.full_finetune:
        # ⭐0911 全解冻: backbone + head + status_encoding 一起训, 唯一冻结 = plan_anchor(codebook)。
        # 动机: 历史上所有崩掉的案例都是"半动"(动一部分参数, 其他部分没跟上新分布):
        #   换anchor不动head / 加goal_mlp只动主体 / 改数据不动模型 → 全崩(3.4~39.7)。
        #   "全动"才可能是唯一自洽的形态 —— 让全部参数一起重新适配 anchor + 数据分布。
        for p in model.parameters():
            p.requires_grad = True
        model._trajectory_head.plan_anchor.requires_grad = False
    elif cli.goalmlp_only:
        # ⭐方案A(0908): 只训 goal_mlp(零init), 冻结其他所有 → 最干净验证"goal显式通道能不能救侧向转弯"
        for p in model._trajectory_head.goal_mlp.parameters():
            p.requires_grad = True
    elif cli.partial_warmstart:
        # ⭐0909: 只训 task_decoder(随机init的cls/reg头), 冻结去噪骨干(warm)+plan_anchor+status_encoding。
        for name, p in model._trajectory_head.named_parameters():
            if 'task_decoder' in name:
                p.requires_grad = True
            elif cli.unfreeze_ae and 'plan_anchor_encoder' in name:
                # ⭐0921: 额外解冻 anchor 编码器。它是在【旧塌缩 anchor】上训练的
                #   (基线 ckpt 的 plan_anchor 18/24 挤在 -74°、最近邻末点距离仅 15mm),
                #   换 anchor 后要它编码全新的 k24 分布, 却既不训也不改
                #   ⇒ 很可能是"换 anchor 净亏"的一环。默认关(行为与旧版逐位相同)。
                p.requires_grad = True
        # ⭐0923 高度实验必须额外解冻 status_encoding —— 否则新填的 st[7] 永远学不会。
        #   实证: 上一版漏了这步, h7_ep0..ep3 的 status_encoding.weight[:,7] 全程为 0(从未更新),
        #   整个 E4 训练作废。此处修正。
        # PWS_UNFREEZE_SE: 只解冻 status_encoding 但不喂高度 —— 分离「多训一层」vs「高度信息」的对照
        if (os.environ.get('PWS_GOAL_HEIGHT', '0') == '1'
                or os.environ.get('PWS_UNFREEZE_SE', '0') == '1'):
            for p in model._status_encoding.parameters():
                p.requires_grad = True
            _n_se = sum(p.numel() for p in model._status_encoding.parameters())
            print(f'[height] status_encoding 已解冻 requires_grad=True ({_n_se} 参数)', flush=True)
    else:
        # 原逻辑: 冻结backbone, 训整个head + status_encoding
        for p in model._trajectory_head.parameters():
            p.requires_grad = True
        for p in model._status_encoding.parameters():
            p.requires_grad = True
        # ⭐0904修: 冻结 plan_anchor(codebook 不该被训)。旧代码把它 requires_grad=True 混进
        # optimizer, 实测 dense 训练把它漂移了 2.08m → anchor 失去聚类语义。
        model._trajectory_head.plan_anchor.requires_grad = False
    if cli.full_finetune:
        # 分层 lr: backbone 用 1/10 —— 全解冻最大风险是灾难性遗忘(把 R2R 已学会的冲掉),
        # 视觉/激光编码器该动得慢, 头该动得快。
        lr_bb = cli.lr_backbone if cli.lr_backbone > 0 else cli.lr / 10.0
        bb = [p for n, p in model.named_parameters()
              if p.requires_grad and n.startswith('_backbone')]
        rest = [p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith('_backbone')]
        n_bb = sum(p.numel() for p in bb)
        n_rest = sum(p.numel() for p in rest)
        print(f'[full-ft] 分层lr: backbone {len(bb)}组/{n_bb/1e6:.1f}M 参数 lr={lr_bb:g} | '
              f'其余 {len(rest)}组/{n_rest/1e6:.1f}M 参数 lr={cli.lr:g}', flush=True)
        opt = torch.optim.AdamW([{'params': bb, 'lr': lr_bb},
                                 {'params': rest, 'lr': cli.lr}])
    else:
        if getattr(cli, 'learn_anchor', 0):
            pa = model._trajectory_head.plan_anchor
            pa.requires_grad = True
            rest = [p for p in model.parameters() if p.requires_grad and p is not pa]
            print(f'[learn-anchor] 码本可学习: plan_anchor lr={cli.anchor_lr:g} (主 lr={cli.lr:g})', flush=True)
            opt = torch.optim.AdamW([{'params': rest, 'lr': cli.lr},
                                     {'params': [pa], 'lr': cli.anchor_lr}])
        else:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cli.lr)

    _GOAL_HEIGHT = os.environ.get('PWS_GOAL_HEIGHT', '0') == '1'   # ⭐0923: 默认关
    # ⭐0924: flip 增广时是否同步镜像 camera(见 run_batch 内注释)。默认关 ⇒ 旧行为逐位不变。
    _FLIP_CAM = os.environ.get('PWS_FLIP_CAM', '0') == '1'
    # ⭐0930 语义条件开关: 默认关 ⇒ 不传 text_feature ⇒ 旧行为逐位不变。
    #   开 = 把指令向量经 text_proj 注入 time_embed(每层 FiLM), 让快脑真正读到指令。
    _TEXT = os.environ.get('PWS_TEXT', '0') == '1'
    if _GOAL_HEIGHT:
        # ⭐0923 高度实验: 旧训练里 status[7] 恒输入 0 ⇒ _status_encoding 第7列权重的梯度恒为0,
        #   从未被更新(保持 init 随机值)。现在喂真实高度, 若不清零则「随机权重×高度」= 噪声。
        #   清零 ⇒ 等价「零初始化新通道」, 让它从头学(有梯度: dL/dW7 = dL/dh * st7, st7≠0)。
        with torch.no_grad():
            model._status_encoding.weight[:, 7].zero_()
        _w7 = float(model._status_encoding.weight[:, 7].abs().sum())
        print(f'[height] status_encoding.weight[:,7] 已清零 (|w7|_sum={_w7:.3g}) ⇒ 该通道从0学', flush=True)

    # ⭐1013 PWS_MODE_LOSS: 给【闲置的 mode_head】加监督(距离分档 4 类)。
    #   模型输出本就有 mode_logits(继承 NAVSIM)但从未被监督 ⇒ 闲置。默认 0 ⇒ 旧行为逐位不变。
    _MODE_LOSS = os.environ.get('PWS_MODE_LOSS', '0') == '1'
    _MODE_W = float(os.environ.get('PWS_MODE_W', '0.5'))

    def _dist2mode(d):
        if d < 0: return -1
        if d > 10.0: return 0
        if d > 3.0: return 1
        if d > 1.0: return 2
        return 3

    def run_batch(cam, bev, gt, gd, hy, tx, md, flip, train=True):
        cam = cam.to(device, non_blocking=True)
        bev = bev.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True).clone()
        gd_np = gd.numpy().copy()
        if flip:
            bev = torch.flip(bev, dims=[-1])
            gt[:, :, 1] = -gt[:, :, 1]
            gd_np[:, 1] = -gd_np[:, 1]
            # ⭐0924 修 bug: cam 原先【没有】跟着镜像 ⇒ 翻转样本里 BEV/GT/goal 全是左右镜像,
            #   只有 camera 是原图, 两者空间上互相矛盾("BEV说目标在左, camera显示在右, 监督说往左")
            #   ⇒ 模型的最优策略是【学会忽略 camera】。
            #   实证吻合: abl_cam0(相机涂黑)只掉 0.67pt、基线失败的57条里45条涂黑后仍失败。
            #   PWS_FLIP_CAM=1 时把 cam 一起镜像(宽度维 = 最后一维)。默认关 ⇒ 旧行为逐位不变。
            if _FLIP_CAM:
                cam = torch.flip(cam, dims=[-1])
        st = status_gh(gd_np, np.tile([1.0, 0.0], (len(gd_np), 1)).astype(np.float32),
                       bs_vx=0.25, device=device)
        st[:, 5] = 0.0
        st[:, 6] = 0.85
        st[:, 7] = hy.reshape(-1) if _GOAL_HEIGHT else 0.0
        # ⭐goal_mlp: goal_dir = ego系goal点归一化(和status[0:2]同口径 SX=5/SY=2)
        goal_dir = torch.tensor(gd_np / np.array([5.0, 2.0], np.float32), device=device)
        _feats = {'camera_feature': cam, 'lidar_feature': bev, 'status_feature': st,
                  'goal_dir': goal_dir}
        if _TEXT:
            # ⭐0930 语义条件: 指令向量经 text_proj(768->d_model, 零初始化) 加进 time_embed。
            #   PWS_TEXT=0(默认) ⇒ 完全不传 ⇒ 旧行为逐位不变。
            _feats['text_feature'] = tx.to(device, non_blocking=True)
        out = model(_feats, {'trajectory': gt})
        loss = out['trajectory_loss']
        if _MODE_LOSS and 'mode_logits' in out and md is not None:
            import torch.nn.functional as _F
            _lab = md.reshape(-1)
            _ok = _lab >= 0
            if bool(_ok.any()):
                _ml = out['mode_logits'].reshape(_lab.shape[0], -1)
                loss = loss + _MODE_W * _F.cross_entropy(_ml[_ok], _lab[_ok].to(device))
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach())   # detach: 避免 requires_grad 张量转 scalar 的 warning

    best = 1e9
    for ep in range(cli.epochs):
        model.train()
        tot, nb = 0.0, 0
        for cam, bev, gt, gd, hy, tx, md in tr_ld:
            tot += run_batch(cam, bev, gt, gd, hy, tx, md, flip=bool(np.random.rand() < 0.5))
            nb += 1
            if nb % 500 == 0:
                print(f'  ep{ep} step{nb} loss={tot/nb:.4f}', flush=True)
        # val(⚠保持train()模式+no_grad: 扩散头只在train算loss)
        model.train()
        with torch.no_grad():
            vt, vb = 0.0, 0
            for cam, bev, gt, gd, hy, tx, md in va_ld:
                vt += run_batch(cam, bev, gt, gd, hy, tx, md, flip=False, train=False)
                vb += 1
                if vb >= 50:
                    break
        vl = vt / max(vb, 1)
        print(f'[ep{ep}] train={tot/max(nb,1):.4f} val={vl:.4f}', flush=True)
        # ⭐监控: goal_mlp 权重 norm(看通道是否"活了"; 若长期≈0 = 没学起来非假设错)
        with torch.no_grad():
            gnorm = sum((p.detach() ** 2).sum().item()
                        for p in model._trajectory_head.goal_mlp.parameters()) ** 0.5
        print(f'  goal_mlp_norm={gnorm:.6f}', flush=True)
        if vl < best:
            best = vl
            _sv = {'trajectory_head': model._trajectory_head.state_dict(),
                   'status_encoding': model._status_encoding.state_dict()}
            if _MODE_LOSS: _sv['mode_head'] = model._mode_head.state_dict()
            torch.save(_sv, os.path.join(RUNS, cli.ckpt_out))
            print(f'  saved best {cli.ckpt_out} val={best:.4f}', flush=True)
        # ⭐每 ep 存带 ep 号的 ckpt: 断点续训(用最新 _epN.pt 当 --ckpt_in 训剩余 ep)
        #    + 随时拿中间 _epN.pt 跑探针看 SR 趋势(不必等 50ep 全训完)。
        ep_ckpt = cli.ckpt_out.replace('.pt', f'_ep{ep}.pt')
        _sv2 = {'trajectory_head': model._trajectory_head.state_dict(),
                'status_encoding': model._status_encoding.state_dict()}
        if _MODE_LOSS: _sv2['mode_head'] = model._mode_head.state_dict()
        torch.save(_sv2, os.path.join(RUNS, ep_ckpt))
        print(f'  saved ep ckpt {ep_ckpt}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['precompute', 'train'], default='train')
    ap.add_argument('--data', nargs='+',
                    default=[f'dense_data/s{i}' for i in range(6)])
    ap.add_argument('--ckpt_in', default='goalnav_r2rft.pt')
    ap.add_argument('--ckpt_out', default='goalnav_r2r_dense.pt')
    ap.add_argument('--learn_anchor', type=int, default=0,
                    help='⭐1006: 让 plan_anchor 参与优化(默认0=冻结,行为逐位不变). 需配 --anchor 指定初值')
    ap.add_argument('--anchor_lr', type=float, default=1e-4,
                    help='⭐1006: 可学习码本的学习率(比 head 小一个量级, 防历史那种无约束漂移)')
    ap.add_argument('--anchor', default=None, help='自定义anchor npy路径(默认robot_kmeans_anchors)')
    ap.add_argument('--clean_head', action='store_true',
                    help='干净训头: 重新构造 trajectory_head(随机init), 不 warm-start 脏 head')
    ap.add_argument('--goalmlp_only', action='store_true',
                    help='方案A: 只训 goal_mlp(冻结其他所有), 最干净验证 goal 显式通道')
    ap.add_argument('--full_finetune', action='store_true',
                    help='0911 全解冻: backbone+head+status 全训(仅冻结 plan_anchor), 分层lr')
    ap.add_argument('--lr_backbone', type=float, default=0,
                    help='全解冻时 backbone 的 lr(0 = cli.lr/10)')
    ap.add_argument('--keep_anchor', action='store_true',
                    help='warm-start 保持 ckpt_in 原 anchor(不换), 与 --goalmlp_only 配合验证 goal 通道')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--turn_oversample', type=float, default=0,
                    help='转弯样本(goal侧方>20°)过采样倍数, 0=关(默认)')
    ap.add_argument('--height_oversample', type=float, default=0,
                    help='⭐0924: 垂直落差(y跨度 > --height_thresh)的 ep 过采样倍数, 0=关(默认⇒旧行为逐位不变)。'
                         '动机: 该类 ep 八模型全票成功率仅 43%%(其他类~93%%), 且训练集占比 4.7%% vs 评测集 9.5%%。')
    ap.add_argument('--height_thresh', type=float, default=2.0,
                    help='⭐0924: 上述"垂直落差"的 y跨度阈值(米), 默认 2.0')
    ap.add_argument('--turnaround_prob', type=float, default=0,
                    help='掉头样本增广比例(goal侧后方+原地转GT), 治组2掉头难, 建议0.15')
    ap.add_argument('--partial_warmstart', action='store_true',
                    help='半warm-start: 换正确anchor + warm去噪骨干 + 随机init task_decoder(破坏anchor天花板)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--val_scenes', type=int, default=8)
    ap.add_argument('--bev_mem', action='store_true',
                    help='⭐0921: bevs.npy 全量预加载到内存(绕开 HDD 随机 IO；~68GB 内存)')
    ap.add_argument('--unfreeze_ae', action='store_true',
                    help='⭐0921: partial_warmstart 时额外解冻 plan_anchor_encoder'
                         '(它在旧塌缩 anchor 上训练, 换 anchor 后可能编码失效)。默认关')
    ap.add_argument('--pws_keep_anchor', action='store_true',
                    help='⭐0922 变量隔离: partial_warmstart 时 plan_anchor 用 ckpt 原值(即"不换 anchor")。'
                         '实测 goalmlp_ep5 / k24 / p1ae 的 plan_anchor 形状全是 (24,8,2), 只是值不同 '
                         '⇒ 「task_decoder 随机init」是设计选择而非形状限制, 故可与「换anchor」拆开。')
    ap.add_argument('--pws_warm_td', action='store_true',
                    help='⭐0922 变量隔离: partial_warmstart 时 task_decoder 也 warm(不再随机 init)。'
                         '与 --pws_keep_anchor 组成 2×2, 用于归因「换anchor」vs「输出头重建」各自的代价。')
    ap.add_argument('--gt_lookahead', type=float, default=0.0,
                    help='⭐0921 GT前瞻: GT 朝 4m 目标方向的最大混合比(0=关,默认; 建议 0.3)。'
                         '输出空间/anchor/模型全不动, 只把 goal 信息注入训练信号')
    ap.add_argument('--gt_lookahead_deg', type=float, default=30.0,
                    help='GT前瞻: 「超出量」(4m转角 − 1.6m转角)达到该角度时混合比取满(默认 30°)')
    ap.add_argument('--seed', type=int, default=-1,
                    help='⭐0924: 固定随机种子(默认 -1 = 不设 ⇒ 行为与旧版逐位相同)。'
                         '固定后同配置两次 run 的 loss/ckpt 可复现, 消除跨 run 的 ±4pt SR 波动。')
    cli = ap.parse_args()
    if cli.mode == 'precompute':
        mode_precompute(cli)
    else:
        mode_train(cli)


if __name__ == '__main__':
    main()
