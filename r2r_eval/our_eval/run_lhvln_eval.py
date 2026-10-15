#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LH-VLN 评测入口:把自研 OurAgent 接到 LH-VLN 官方仿真 (habitat_base.SceneSimulator) 上。

官方仓库没带 eval 入口,本脚本按官方 train.py / utils/agent.py 的 validate 流程补齐:
  构造 config -> SceneSimulator -> 重置到 Start pos -> 循环(观测->OurAgent->actor)
  -> 收集 return_results() -> NavigationMetrics(ISR/CSR/CGT/SR/OSR/SPL/NE)

数据:  lhvln/data/episode_task/batch_{6,7,8}.json.gz
       {"episode_0": {"lh_task": {...}, "st_task": [...]}}
场景:  lhvln/data/hm3d/{train,val}/<Scene>/<short>.{basis.glb,basis.navmesh,semantic.glb,semantic.txt}
用法:
  GIT_PYTHON_REFRESH=quiet /data/DongBaorong/miniconda3/envs/lhvln/bin/python run_lhvln_eval.py \
      --batch 6 --max_episodes 3 --gpu 0 --out /tmp/lhvln_b6_smoke.jsonl

---------------- 已知环境坑与绕过(全部在本脚本内完成,不改仓库既有文件) ----------------
1) habitat_base/config.py:44 把 gpu_device_id 写死 1,且:12,26,48-49 会设置
   scene_dataset_config_file = data/hm3d/hm3d_annotated_basis.scene_dataset_config.json,
   该文件在本机不存在 —— 本脚本用 monkeypatch 替换 simulation.make_setting/make_cfg,
   改为「程序化构造 StageAttributes 模板」(见 _build_config)。仿真层一行未改。
2) habitat_sim 需要语义标注才能算 geodesic distance(get_coord -> sim.semantic_scene)。
   HM3D 官方 info_semantic.json 未随包下载,只有 <short>.semantic.txt。
   实测 habitat-sim 0.3.1 的 loadSemanticSceneDescriptor 能直接解析该 txt(首行
   "HM3D Semantic Annotations"),但查找路径是 <fileDirectory>/<stage_handle>.scn,
   且 fileDirectory 只能由 CWD 决定 => 本脚本在临时目录里放 <Scene>.scn 软链,构造
   Simulator 时临时 chdir 过去(_StageTemplateCache)。
3) 由 txt 描述的语义 AABB 在 glb 原始坐标系(z 朝上)里,而轨迹/navmesh 是 habitat 的
   y 朝上系。实测 floor/ceiling 的 z 分量才是高度,故 hab=(x, z, -y)。
   官方有 info_semantic.json 时其 OBB 已是 y-up,无需该变换 —— 见 _patch_get_coord,
   切回官方语义标注后应设 --no_coord_frame_fix。
4) gpu: torch 用 --gpu 指定 cuda:N,habitat 用 gpu_device_id=N;不做 CUDA_VISIBLE_DEVICES,
   避免 EGL 与 torch 的可见设备编号不一致。
5) VLM: OurAgent 默认走 sim_fullchain.vlm_api(robo2vlm/Qwen http),服务挂掉时
   agent 内部 _vlm() 已容错返回 None -> 退化为原地扫描/换视角,不会崩 episode。
   本脚本统计 vlm 不可用次数并在汇总里报告。--locator oracle 用仿真真值目标点
   代替 VLM 定位,用于验证整条链路(不含 VLM)能否跑出成功。
"""
import argparse
import glob
import gzip
import json
import math
import os
import shutil
import sys
import tempfile
import types

os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')
# habitat_base/visualization.py 顶层 import PyQt5(只用来设 QT_QPA 插件路径),本机没装;
# 打桩即可,display_env 本身不用 Qt。
if 'PyQt5' not in sys.modules:
    _stub = types.ModuleType('PyQt5')
    _stub.__file__ = '/tmp/_PyQt5_stub/__init__.py'
    sys.modules['PyQt5'] = _stub

import numpy as np  # noqa: E402

BEV = '/data/DongBaorong/BEVLOOK'
LH = f'{BEV}/lhvln'
OUR_EVAL = os.path.dirname(os.path.abspath(__file__))
for p in (LH, OUR_EVAL):
    if p not in sys.path:
        sys.path.insert(0, p)

import habitat_sim  # noqa: E402
import habitat_sim.simulator as hs  # noqa: E402
import magnum as mn  # noqa: E402
import habitat_base.simulation as sim_mod  # noqa: E402
from utils.metrics import NavigationMetrics  # noqa: E402

HM3D = f'{LH}/data/hm3d'
EPISODE_DIR = f'{LH}/data/episode_task'
TASK_DIR = f'{LH}/data/task'


# ============================ habitat 配置构造(绕过 config.py) ============================
def _scene_paths(scene):
    """LH-VLN Scene 字段 -> (split, scene_dir, short)"""
    short = scene.split('-')[-1]
    split = 'train' if int(scene[2]) < 8 else 'val'
    sdir = f'{HM3D}/{split}/{scene}'
    return split, sdir, short


class _StageTemplateCache:
    """每个场景建一次 StageAttributes 模板 + 全局 .scn 语义描述符目录。

    habitat 找语义描述符的路径是 `<fileDirectory>/<stage_handle>.scn`,而 fileDirectory
    python 侧只读(file_directory 只读,set('fileDirectory',..) 拒收字符串),实际由进程
    CWD 决定。所以这里建一个平铺目录,里面每个场景放一个 <Scene>.scn 软链(内容就是官方
    HM3D semantic.txt —— 实测 habitat-sim 0.3.1 的 loadSemanticSceneDescriptor 能直接解析
    首行 "HM3D Semantic Annotations" 的 txt),并在 main() 里把进程 CWD 固定到该目录。
    """

    def __init__(self):
        self._models = {}                       # scene -> MetadataMediator
        self.desc_dir = tempfile.mkdtemp(prefix='lhvln_sem_desc_')

    def prepare(self, scene):
        """返回 metadata_mediator;同场景复用。"""
        if scene in self._models:
            return self._models[scene]
        split, sdir, short = _scene_paths(scene)
        sem_txt = f'{sdir}/{short}.semantic.txt'
        if not os.path.exists(sem_txt):
            raise FileNotFoundError(f'no semantic annotations: {sem_txt}')
        link = f'{self.desc_dir}/{scene}.scn'
        if not os.path.exists(link):
            # 描述符文件名必须是 <stage_handle>.scn;内容就是官方 HM3D semantic.txt
            os.symlink(sem_txt, link)

        mm = hs.MetadataMediator()
        sam = mm.stage_template_manager
        t = sam.create_template(scene, False)
        t.render_asset_handle = f'{sdir}/{short}.basis.glb'
        t.collision_asset_handle = f'{sdir}/{short}.basis.glb'
        t.navmesh_asset_handle = f'{sdir}/{short}.basis.navmesh'
        t.semantic_asset_handle = f'{sdir}/{short}.semantic.glb'
        sam.register_template(t)
        assert scene in sam.get_template_handles(), f'stage template not registered: {scene}'
        self._models[scene] = mm
        return mm

    def close(self):
        shutil.rmtree(self.desc_dir, ignore_errors=True)


_STAGES = _StageTemplateCache()


def _make_setting(args, scene_file, robot):
    """替换 habitat_base.config.make_setting:只保留 SceneSimulator 自己用到的键。"""
    return {'scene': scene_file, 'robot': robot, 'default_agent': 0, 'gpu': int(args.gpu)}


def _make_cfg(settings):
    """替换 habitat_base.config.make_cfg:直接建 Configuration,不走 scene_dataset_config_file。"""
    scene = settings['scene']
    mm = _STAGES.prepare(scene)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene                 # 模板 handle,不是文件路径
    sim_cfg.gpu_device_id = settings['gpu']  # config.py 原本写死 1
    sim_cfg.enable_physics = False

    sensors = {
        'color_sensor_f': (habitat_sim.SensorType.COLOR, mn.Vector3(0, 1.0, 0), mn.Vector3(0, 0, 0)),
        'color_sensor_l': (habitat_sim.SensorType.COLOR, mn.Vector3(0, 1.0, 0), mn.Vector3(0, math.pi / 3, 0)),
        'color_sensor_r': (habitat_sim.SensorType.COLOR, mn.Vector3(0, 1.0, 0), mn.Vector3(0, -math.pi / 3, 0)),
        'depth_sensor_f': (habitat_sim.SensorType.DEPTH, mn.Vector3(0, 1.0, 0), mn.Vector3(0, 0, 0)),
        'depth_sensor_l': (habitat_sim.SensorType.DEPTH, mn.Vector3(0, 1.0, 0), mn.Vector3(0, math.pi / 3, 0)),
        'depth_sensor_r': (habitat_sim.SensorType.DEPTH, mn.Vector3(0, 1.0, 0), mn.Vector3(0, -math.pi / 3, 0)),
        'semantic_sensor': (habitat_sim.SensorType.SEMANTIC, mn.Vector3(0, 1.0, 0), mn.Vector3(0, 0, 0)),
    }
    specs = []
    for uuid, (stype, pos, ori) in sensors.items():
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [512, 512]
        s.position = pos
        s.orientation = ori
        specs.append(s)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    agent_cfg.action_space = {
        'move_forward': habitat_sim.agent.ActionSpec('move_forward', habitat_sim.agent.ActuationSpec(amount=0.25)),
        'turn_left': habitat_sim.agent.ActionSpec('turn_left', habitat_sim.agent.ActuationSpec(amount=30.0)),
        'turn_right': habitat_sim.agent.ActionSpec('turn_right', habitat_sim.agent.ActuationSpec(amount=30.0)),
    }
    # CWD 已固定在 _STAGES.desc_dir(main 里 chdir),habitat 才能找到 <Scene>.scn
    return habitat_sim.Configuration(sim_cfg, [agent_cfg], metadata_mediator=mm)


def _patch_get_coord():
    """txt 描述的语义 AABB 在 glb 原始 z-up 系;换成 habitat y-up(实测 floor/ceiling 验证)。

    官方 HM3D info_semantic.json 的 OBB 已是 y-up,拿到后可用 --no_coord_frame_fix 关掉。
    """
    def get_coord(self, obj_target):
        scene = self.sim.semantic_scene
        coord_list = []
        index = self.target.index(obj_target)
        region_id = self.region[index]
        for region in scene.regions:
            if region.id[1:] != region_id:
                continue
            for obj in region.objects:
                if obj.category.name() == obj_target:
                    c = obj.aabb.center
                    coord_list.append(np.array([float(c[0]), float(c[2]), -float(c[1])], np.float32))
        if not coord_list:
            return 0
        return coord_list

    sim_mod.SceneSimulator.get_coord = get_coord


def _patch_display_env():
    """SceneSimulator.actor 每步调 display_env 拼 PIL 图,我们用不到(obs 不是给 agent 的)。"""
    sim_mod.display_env = lambda *a, **k: []


# ============================ 数据 / config 构造 ============================
def load_batch(batch):
    path = f'{EPISODE_DIR}/batch_{batch}.json.gz'
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def build_config(lh_task, batch):
    """对齐 lhvln/utils/dataset.py:119-137 的 config 构造。"""
    obj, region = [], []
    for sub in lh_task['Subtask list']:
        if 'Move_to' in sub:
            name, rid = sub[9:-2].split('_')
            obj.append(name)
            region.append(rid)
    rooms = [r[1].split(': ')[1] for r in lh_task['Object']]
    return {
        'Scene': lh_task['Scene'],
        'Robot': lh_task['Robot'],
        'Object': obj,
        'Region': region,
        'Room': rooms,
        'Task instruction': lh_task['Task instruction'],
        'Batch': batch,
        'gt_step': list(lh_task['gt_step']),
    }


def quat_from_hab_yaw_deg(phi_deg):
    """绕 habitat Y 轴转 phi(度) 的四元数 (w,x,y,z);habitat forward = (-sin phi, 0, -cos phi)。"""
    phi = math.radians(phi_deg)
    return [math.cos(phi / 2.0), 0.0, math.sin(phi / 2.0), 0.0]


def quat_from_yaw_deg(yaw_deg):
    """GT trial 的 yaw(度)-> habitat 四元数。实测 GT forward=(sin y,0,cos y) => phi = y - 180。"""
    return quat_from_hab_yaw_deg(yaw_deg - 180.0)


def find_start_yaw(lh_task, batch):
    """从官方 task/<batch>/…/success/trial_*/task.json 拿起点朝向(度)。找不到返回 None。

    trial_0 = 第一个子任务的轨迹,pos[0] 与 lh_task['Start pos'] 一致。
    """
    ins = lh_task['Task instruction']
    scene = lh_task['Scene']
    start = np.array(lh_task['Start pos'], np.float32)
    pat = f'{TASK_DIR}/batch_{batch}/*/{glob.escape(ins)}/success/trial_*/task.json'
    for f in sorted(glob.glob(pat)):
        try:
            tj = json.load(open(f))
        except Exception:
            continue
        tr = tj.get('trial', {}).get('trial_0')
        if not tr or not tr.get('pos'):
            continue
        if np.linalg.norm(np.array(tr['pos'][0], np.float32) - start) > 0.05:
            continue
        if tj.get('Scene') and tj['Scene'] != scene:
            continue
        yaw = tr.get('yaw')
        if yaw:
            return float(yaw[0])
    return None


# ============================ oracle 定位(仅用于验证链路) ============================
def _ego_of_target(agent, sim, target_name):
    """仿真真值目标点 -> agent ego 系 (x前, y左)。与 OurAgent.hab_pose_to_ours 同口径。"""
    coords = sim.get_coord(target_name)
    if not coords:
        return None
    pos, rot = sim.return_state()
    pp = np.array([float(pos[0]), float(pos[1]), float(pos[2])], np.float32)
    coord = min((np.asarray(c, np.float32) for c in coords), key=lambda c: float(np.linalg.norm(c - pp)))
    R = agent._rot_matrix(rot)
    fwd = R @ np.array([0.0, 0.0, -1.0])
    th = math.atan2(-fwd[2], fwd[0])
    d = coord - pp
    dd = np.array([d[0], -d[2]], np.float32)          # 我们系平面 (x=hab_x, y=-hab_z)
    c, s = math.cos(th), math.sin(th)
    return np.array([c * dd[0] + s * dd[1], -s * dd[0] + c * dd[1]], np.float32)


def install_oracle_locator(agent, sim):
    """用仿真真值目标点替代 VLM 定位:只为验证 harness(步进/子目标推进/判停/指标)通不通。"""
    def locate(rgb=None, depth=None, t=None, room=None):
        ego = _ego_of_target(agent, sim, t)
        return (ego, 'oracle') if ego is not None else (None, 'absent')

    agent.head_locate = lambda rgb, depth, t: locate(t=t)              # -> (ego|None, how)
    agent.vlm_locate = lambda rgb, depth, t, room=None: locate(t=t)[0]  # -> ego|None
    return agent


# ============================ 单 episode 评测 ============================
def run_episode(agent, args, batch, key, lh_task, verbose=True):
    from types import SimpleNamespace
    config = build_config(lh_task, batch)
    sim_args = SimpleNamespace(scene=f'{HM3D}/', scene_dataset=None, success_dis=1.0,
                               max_step=int(args.max_step), gpu=int(args.gpu),
                               episode_data=f'batch_{batch}')
    rec = {'episode': key, 'batch': batch, 'scene': config['Scene'],
           'instruction': config['Task instruction'], 'targets': config['Object'],
           'regions': config['Region'], 'status': 'ok', 'steps': 0, 'actions': {},
           'vlm_down': False}

    sim = sim_mod.SceneSimulator(sim_args, config)
    try:
        if args.locator == 'oracle':
            install_oracle_locator(agent, sim)
        # --- 重置到 Start pos (官方 lh task 是随机可导航点;这里固定起点保证可复现) ---
        start = np.array(lh_task['Start pos'], np.float32)
        yaw_deg = None
        if args.start_pose == 'gt':
            yaw_deg = find_start_yaw(lh_task, batch)
        if yaw_deg is None:
            yaw_deg = float(args.start_yaw_deg)
            rec['start_yaw_source'] = 'fixed'
        else:
            rec['start_yaw_source'] = 'gt'
        rec['start_pos'] = [float(v) for v in start]
        rec['start_yaw_deg'] = yaw_deg
        hab_yaw_deg = None
        if args.debug_start_near_target > 0:
            # harness 自检:把起点挪到当前子目标附近并朝向它,验证「几何判停 -> 子目标推进 ->
            # 指标统计」这条链路。不代表算法能力,只用于自测。
            coords = sim.get_coord(sim.target[0])
            if coords:
                tgt = np.asarray(coords[0], np.float32)
                tgt_pl = np.array([tgt[0], -tgt[2]], np.float32)          # 我们系平面
                cur_pl = np.array([start[0], -start[2]], np.float32)
                v = cur_pl - tgt_pl
                n = float(np.linalg.norm(v)) or 1.0
                new_pl = tgt_pl + v / n * float(args.debug_start_near_target)
                start = np.array([new_pl[0], start[1], -new_pl[1]], np.float32)
                d = tgt - start                                            # habitat xyz
                hab_yaw_deg = math.degrees(math.atan2(-float(d[0]), -float(d[2])))
                rec['start_yaw_deg'] = hab_yaw_deg
                rec['start_yaw_source'] = 'debug_near_target'
                rec['start_pos'] = [float(v) for v in start]
        quat = (quat_from_hab_yaw_deg(hab_yaw_deg) if hab_yaw_deg is not None
                else quat_from_yaw_deg(yaw_deg))
        sim.set_state(start.tolist(), quat)
        sim.observations = sim.sim.get_sensor_observations()
        sim.info = sim.get_info()          # 距离/目标点需与 teleport 后的位姿一致
        if not isinstance(sim.info, dict):
            rec['status'] = 'invalid_target_coord'
            return rec, None

        sim.gt_step = sim.count_gt_step(False)

        # --- 起手:官方 validate 的第一拍 actor('stop') (step -1 -> 0) ---
        obs, done, info = sim.actor('stop')
        agent.reset_episode()

        prev_action = None
        while not sim.episode_over and sim.step < int(args.max_step):
            if not isinstance(sim.info, dict):
                rec['status'] = 'invalid_target_coord'
                break
            target_name = sim.target[sim.stage]
            room = config['Room'][sim.stage] if sim.stage < len(config['Room']) else None
            action, dbg = agent.act(sim.observations, sim.info['agent position'],
                                    sim.info['agent rotation'], target_name, room=room)
            if action not in ('move_forward', 'turn_left', 'turn_right', 'stop'):
                rec['status'] = f'bad_action:{action}'
                break
            rec['actions'][action] = rec['actions'].get(action, 0) + 1
            rec['last_dbg'] = str(dbg)
            rec['steps'] = sim.step
            prev_action = action
            obs, done, info = sim.actor(action)
        else:
            rec['steps'] = sim.step

        if prev_action is None:
            rec['status'] = 'no_action'

        res = sim.return_results()
        rec['results'] = {
            'successes': [bool(x) for x in res['successes']],
            'oracle_successes': [bool(x) for x in res['oracle_successes']],
            'navigation_steps': [int(x) for x in res['navigation_steps']],
            'navigation_errors': [float(x) for x in res['navigation_errors']],
            'gt_step': [int(x) for x in (res['gt_step'] or [])],
            'gt_path': [float(x) for x in (res['gt_path'] or [])],
            'stage_reached': int(sim.stage),
        }
        rec['vlm_down'] = bool(getattr(agent, '_vlm_down_logged', False))
        if verbose:
            print(f"  [ep {key}] {rec['status']} steps={rec['steps']} "
                  f"successes={rec['results']['successes']} "
                  f"nav_err={[round(x, 2) for x in rec['results']['navigation_errors']]} "
                  f"acts={rec['actions']}", flush=True)
        return rec, res
    finally:
        try:
            sim.close()
        except Exception:
            pass


# ============================ 主流程 ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', default='6', help='6/7/8 或逗号分隔,如 6,7,8')
    ap.add_argument('--max_episodes', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max_step', type=int, default=500)
    ap.add_argument('--ckpt', default='goalnav_gate_lab_v4.pt')
    ap.add_argument('--hfov', type=float, default=90.0)
    ap.add_argument('--vlm_url', default=None, help='覆盖 VLM_URL 环境变量')
    ap.add_argument('--ground_head', default=None, help='接地头 ckpt(存在则优先于 VLM)')
    ap.add_argument('--locator', default='ours', choices=['ours', 'oracle', 'infinitevl'],
                    help='ours=OurAgent 自带(VLM/接地头); oracle=仿真真值(仅验证链路); '
                         'infinitevl=慢脑 InfiniteVL HTTP 定位(推荐)')
    ap.add_argument('--infinitevl_url', default='http://localhost:8123',
                    help='慢脑 InfiniteVL 定位服务器地址(--locator infinitevl 时用)')
    ap.add_argument('--start_pose', default='gt', choices=['gt', 'fixed'],
                    help='gt=用官方 trial 起点朝向;fixed=用 --start_yaw_deg')
    ap.add_argument('--start_yaw_deg', type=float, default=0.0)
    ap.add_argument('--no_coord_frame_fix', action='store_true',
                    help='语义标注已是 y-up(有官方 info_semantic.json)时开启')
    ap.add_argument('--debug_start_near_target', type=float, default=0.0,
                    help='harness 自检:起点放到首个子目标外 N 米并朝向它(0=关)。'
                         '只为验证判停/子目标推进/指标链路,不代表算法能力')
    ap.add_argument('--out', default=None, help='结果 jsonl 路径')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    sim_mod.make_setting = _make_setting
    sim_mod.make_cfg = _make_cfg
    _patch_display_env()
    if not args.no_coord_frame_fix:
        _patch_get_coord()

    # habitat 的语义描述符查找走 CWD(fileDirectory 只读);把 CWD 固定到平铺描述符目录。
    # 其余路径(权重/数据/输出)全部用绝对路径,故常驻切换是安全的。
    for a in ('ground_head', 'out'):
        v = getattr(args, a)
        if v and not os.path.isabs(v):
            setattr(args, a, os.path.abspath(v))
    os.chdir(_STAGES.desc_dir)
    print(f'[eval] semantic descriptor dir (cwd): {_STAGES.desc_dir}', flush=True)

    batches = [b.strip() for b in str(args.batch).split(',') if b.strip()]
    episodes = []
    for b in batches:
        d = load_batch(b)
        for k in sorted(d.keys(), key=lambda s: int(s.split('_')[-1])):
            episodes.append((b, k, d[k]['lh_task']))
    if args.max_episodes and args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    print(f'==== LH-VLN eval: {len(episodes)} episodes, batch={batches}, gpu={args.gpu}, '
          f'locator={args.locator}, ckpt={args.ckpt} ====', flush=True)

    from our_agent import OurAgent          # import 在这里: 放后面避免拖慢 --help
    print('[eval] loading OurAgent ...', flush=True)
    agent = OurAgent(ckpt=args.ckpt, device=f'cuda:{args.gpu}', vlm_url=args.vlm_url,
                     ground_head=args.ground_head, hfov=args.hfov,
                     locator=('infinitevl' if args.locator == 'infinitevl' else 'vlm'),
                     infinitevl_url=args.infinitevl_url)
    print('[eval] OurAgent ready', flush=True)

    out_path = args.out or f'{OUR_EVAL}/lhvln_eval_b{args.batch}_{args.locator}.jsonl'
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fout = open(out_path, 'w', encoding='utf-8')

    metrics = NavigationMetrics()
    n_ok = n_fail = 0
    for (b, key, lh_task) in episodes:
        print(f'---- {b}/{key} scene={lh_task["Scene"]} targets={lh_task["Subtask list"]}', flush=True)
        try:
            rec, res = run_episode(agent, args, b, key, lh_task, verbose=not args.quiet)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec, res = ({'episode': key, 'batch': b, 'status': f'crash:{type(e).__name__}:{e}'}, None)
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        fout.flush()
        if res is None:
            n_fail += 1
            continue
        n_ok += 1
        metrics.add_sample(
            all(res['successes']),
            sum(res['gt_step']) if res['gt_step'] else 0,
            sum(res['navigation_steps']),
            all(res['oracle_successes']),
            (sum(res['navigation_errors']) / len(res['navigation_errors'])) if res['navigation_errors'] else 0,
            res['successes'], res['gt_step'], res['gt_path'], res['navigation_errors'])

    fout.close()
    m = metrics.compute()
    summary = {
        'episodes': len(episodes), 'scored': n_ok, 'crashed_or_skipped': n_fail,
        'SR': m['success_rate'], 'OSR': m['oracle_success_rate'], 'SPL': m['spl'],
        'NE': m['navigation_error'], 'ISR': m['independent_success_rate'],
        'CSR': m['conditional_success_rate'], 'CGT': m['conditional_path_length'],
        'TAR': m['tar'], 'out': out_path,
    }
    print('\n==== SUMMARY ====')
    for k, v in summary.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    _STAGES.close()
    return summary


if __name__ == '__main__':
    main()
