"""DISCOVERSE 真实仿真器:双层世界 = 干净几何物理层(墙/地/障碍盒,带碰撞) + 3DGS 可切换皮肤,
机器人主体用 DISCOVERSE 自带 MMK2 移动底座外观(底座+轮子,纯视觉) + 我们的 (v,ω) 可控底盘 + 碰撞盒。
被 sim_walk_discoverse.py(脚本视频)和 sim_walk_server.py(浏览器实时驱动)共用。

前置:
  scripts/align_gs_world.py    -> outputs/gs_vggto_<s>_world.ply + sim_world_<s>.npz
  scripts/fit_world_geometry.py -> outputs/world_geometry_<s>.json (墙/盒;缺失则退化为空场景)
运行环境:MUJOCO_GL=egl + PYTHONPATH=third_party/discoverse(本体未 pip 装)。

物理:控制写 mj_data.ctrl 配 velocity actuator(不是写 qvel——DISCOVERSE 每子步会覆盖 qvel,
接触力顶不住)。这样撞墙被挡住不穿墙;松手 ctrl=0 靠 damping 停(不做惯性滑行)。
双层切换:DISCOVERSE 内置 env.show_gaussian_img(True=高斯皮肤/False=MuJoCo几何)。
"""
import os
# headless EGL:必须在 import mujoco / OpenGL 之前设好(否则 PyOpenGL 选错平台 → PLATFORM_DEVICE 报错)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHDIR = os.path.join(ROOT, "third_party", "discoverse", "models", "meshes")

# MuJoCo 相机:看局部 -z,+x 右,+y 上。要它看机器人前向(+x_body)、up=+z_body:
#   cam→body 列 = [right=-y, up=+z, -forward=-x]
_CAM_M = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
_CAM_QUAT_XYZW = Rotation.from_matrix(_CAM_M).as_quat()
FWD_CAM_QUAT_WXYZ = _CAM_QUAT_XYZW[[3, 0, 1, 2]]   # wxyz for MJCF

# MMK2 移动底座外观(纯视觉, class="visual";位姿照搬 mobile_chassis/mmk2/mmk2.xml 的 agv_link)
_MMK2_AGV_VISUAL = """
      <body name="agv_visual" pos="0.02371 0 0">
        <geom mesh="mmk2_agv_0" material="mmk2_black" class="visual"/>
        <geom mesh="mmk2_agv_1" material="mmk2_copper" class="visual"/>
        <geom mesh="mmk2_agv_2" material="mmk2_grey" class="visual"/>
        <geom mesh="mmk2_agv_3" material="mmk2_black" class="visual"/>
        <geom mesh="mmk2_agv_4" rgba="0.592 0.9 0.9 1" class="visual"/>
        <geom mesh="rgt_front_wheel_link"  euler="0 0 1.5708" pos=" 0.13045 -0.089989 0.085" rgba="0.2 0.2 0.2 1" class="visual"/>
        <geom mesh="lft_front_wheel_link"  euler="0 0 1.5708" pos=" 0.13045  0.090011 0.085" rgba="0.2 0.2 0.2 1" class="visual"/>
        <geom mesh="rgt_behind_wheel_link" euler="0 0 1.5708" pos="-0.15755 -0.099989 0.085" rgba="0.2 0.2 0.2 1" class="visual"/>
        <geom mesh="lft_behind_wheel_link" euler="0 0 1.5708" pos="-0.15755  0.10001  0.085" rgba="0.2 0.2 0.2 1" class="visual"/>
        <geom mesh="lft_wheel_link" pos="-0.02371  0.16325 0.082" euler="1.5708 0 0" rgba="0.2 0.2 0.2 1" class="visual"/>
        <geom mesh="rgt_wheel_link" pos="-0.02371 -0.16325 0.082" euler="1.5708 0 0" rgba="0.2 0.2 0.2 1" class="visual"/>
      </body>"""

_MMK2_ASSETS = """
    <material name="mmk2_grey"   specular="0.5" shininess="0.5" rgba="0.93 0.93 0.93 1"/>
    <material name="mmk2_black"  specular="0.5" shininess="0.5" rgba="0.05 0.05 0.05 1"/>
    <material name="mmk2_copper" specular="0.5" shininess="0.5" rgba="0.54 0.54 0.54 1"/>
    <mesh name="mmk2_agv_0" file="mmk2/mmk2_agv_0.obj"/>
    <mesh name="mmk2_agv_1" file="mmk2/mmk2_agv_1.obj"/>
    <mesh name="mmk2_agv_2" file="mmk2/mmk2_agv_2.obj"/>
    <mesh name="mmk2_agv_3" file="mmk2/mmk2_agv_3.obj"/>
    <mesh name="mmk2_agv_4" file="mmk2/mmk2_agv_4.obj"/>
    <mesh name="lft_wheel_link" file="mmk2/lft_wheel_link.STL"/>
    <mesh name="rgt_wheel_link" file="mmk2/rgt_wheel_link.STL"/>
    <mesh name="rgt_front_wheel_link"  file="mmk2/rgt_front_wheel_link.obj"/>
    <mesh name="lft_front_wheel_link"  file="mmk2/lft_front_wheel_link.obj"/>
    <mesh name="rgt_behind_wheel_link" file="mmk2/rgt_behind_wheel_link.obj"/>
    <mesh name="lft_behind_wheel_link" file="mmk2/lft_behind_wheel_link.obj"/>"""


def _wall_box(a, b, height, thick, idx, side):
    """一面墙段 a->b 生成一对 box geom(collision+visual,绕 z 旋转对齐段方向)。"""
    ax, ay = a; bx, by = b
    cx, cy = (ax + bx) / 2, (ay + by) / 2
    dx, dy = bx - ax, by - ay
    length = float(np.hypot(dx, dy))
    if length < 1e-3:
        return ""
    yaw = float(np.arctan2(dy, dx))
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
    hl, ht, hh = length / 2 + thick / 2, thick / 2, height / 2   # +thick 让相邻段在拐角处接上
    q = f"{qw:.6f} 0 0 {qz:.6f}"
    name = f"wall_{side}_{idx}"
    return (
        f'      <geom name="{name}_c" class="collision" type="box" '
        f'size="{hl:.3f} {ht:.3f} {hh:.3f}" pos="{cx:.3f} {cy:.3f} {hh:.3f}" quat="{q}"/>\n'
        f'      <geom name="{name}_v" class="wallvis" type="box" '
        f'size="{hl:.3f} {ht:.3f} {hh:.3f}" pos="{cx:.3f} {cy:.3f} {hh:.3f}" quat="{q}"/>\n')


def _world_geometry_xml(session):
    """读 world_geometry_<s>.json 生成墙+障碍盒的 collision+visual geom。缺文件则返回空串。"""
    path = os.path.join(ROOT, "outputs", f"world_geometry_{session}.json")
    if not os.path.exists(path):
        print(f"[mjcf] 无 {path},退化为空几何场景")
        return ""
    g = json.load(open(path))
    xml = "    <!-- 几何物理层: 墙 + 障碍盒 -->\n"
    for w in g.get("walls", []):
        side = "L" if w["segs"] and w["segs"][0][0][0] < 0 else "R"
        for i, (a, b) in enumerate(w["segs"]):
            xml += _wall_box(a, b, w["height"], w["thick"], i, side)
    for j, bx in enumerate(g.get("boxes", [])):
        c, h = bx["center"], bx["half"]
        xml += (
            f'      <geom name="box_{j}_c" class="collision" type="box" '
            f'size="{h[0]:.3f} {h[1]:.3f} {h[2]:.3f}" pos="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}"/>\n'
            f'      <geom name="box_{j}_v" class="boxvis" type="box" '
            f'size="{h[0]:.3f} {h[1]:.3f} {h[2]:.3f}" pos="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}"/>\n')
    return xml


def build_mjcf(session, start_xy, start_yaw=0.0, cam_height=1.2, width=1280, height=720,
               fovy=75.0, simple_robot=False, skin_only=False):
    sx, sy = float(start_xy[0]), float(start_xy[1])
    q = FWD_CAM_QUAT_WXYZ
    # skin_only: 跳过几何物理层(无墙/盒/碰撞), 纯 3DGS 皮肤自由驾驶
    geom_xml = "" if skin_only else _world_geometry_xml(session)

    if simple_robot:                                   # 旧橙盒(回退)
        robot_visual = ('      <geom name="chassis" type="box" size="0.25 0.18 0.12" '
                        'pos="0 0 0.12" rgba="1 0.55 0.1 1"/>\n')
        mmk2_assets = ""
    else:
        robot_visual = _MMK2_AGV_VISUAL + "\n"
        mmk2_assets = _MMK2_ASSETS

    xml = f"""<mujoco model="vggt_walk_{session}">
  <compiler angle="radian" meshdir="{MESHDIR}" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"/>
  <visual>
    <global offwidth="{max(width,1920)}" offheight="{max(height,1080)}"/>
    <headlight ambient="0.45 0.45 0.48" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>

  <default>
    <default class="collision"><geom group="3" condim="3" contype="1" conaffinity="1" rgba="0.4 0.4 0.4 0" friction="1 0.05 0.001"/></default>
    <default class="visual"><geom group="2" contype="0" conaffinity="0"/></default>
    <default class="wallvis"><geom group="2" contype="0" conaffinity="0" material="wallmat"/></default>
    <default class="boxvis"><geom group="2" contype="0" conaffinity="0" material="boxmat"/></default>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.25 0.28" rgb2="0.30 0.34 0.38" width="512" height="512" mark="edge" markrgb="0.5 0.55 0.6"/>
    <material name="groundplane" texture="grid" texuniform="true" texrepeat="6 6" reflectance="0.1"/>
    <material name="wallmat" rgba="0.78 0.80 0.83 1" specular="0.2" shininess="0.3"/>
    <material name="boxmat" rgba="0.55 0.62 0.72 1" specular="0.3" shininess="0.4"/>{mmk2_assets}
  </asset>

  <worldbody>
    <light pos="0 0 18" dir="0 0 -1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <light pos="{sx} {sy} 6" dir="0 0 -1" diffuse="0.5 0.5 0.5" castshadow="false"/>
    <geom name="floor" type="plane" size="80 80 0.1" material="groundplane"/>
    <body name="background" pos="0 0 0" quat="1 0 0 0">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
    </body>
{geom_xml}
    <body name="robot" pos="{sx} {sy} 0">
      <joint name="jx" type="slide" axis="1 0 0" damping="10"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="10"/>
      <joint name="jyaw" type="hinge" axis="0 0 1" damping="3"/>
      <geom name="chassis_col" class="collision" type="box" size="0.22 0.20 0.13" pos="-0.015 0 0.14" mass="15"/>
{robot_visual}      <camera name="fpv" pos="0 0 {cam_height}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" fovy="{fovy}"/>
    </body>
  </worldbody>

  <actuator>
    <velocity name="act_vx"   joint="jx"   kv="200" ctrlrange="-3 3"  forcerange="-300 300"/>
    <velocity name="act_vy"   joint="jy"   kv="200" ctrlrange="-3 3"  forcerange="-300 300"/>
    <velocity name="act_vyaw" joint="jyaw" kv="50"  ctrlrange="-3 3"  forcerange="-60 60"/>
  </actuator>
</mujoco>"""
    path = os.path.join(ROOT, "outputs", f"_sim_walk_{session}.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def make_env(session="114830", width=1280, height=720, fps=24, cam_height=1.2, fovy=75.0,
             simple_robot=False, skin_only=False):
    """构建并返回 (env, world_npz)。需先设好 MUJOCO_GL/PYTHONPATH。"""
    from discoverse.envs import SimulatorBase
    from discoverse.utils.base_config import BaseConfig

    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{session}.npz"))
    start_xy = W["waypoints_xy"][0]
    start_yaw = float(W["yaw"][0]) if "yaw" in W else 0.0
    ply = os.path.join(ROOT, "outputs", f"gs_vggto_{session}_world.ply")
    mjcf = build_mjcf(session, start_xy, start_yaw, cam_height, width, height, fovy,
                      simple_robot, skin_only)

    class WalkCfg(BaseConfig):
        mjcf_file_path = mjcf
        headless = True
        sync = False
        decimation = 4
        timestep = 0.005
        render_set = {"fps": fps, "width": width, "height": height}
        obs_rgb_cam_id = [0]
        use_gaussian_renderer = True
        gs_model_dict = {"background": ply}
        enable_render = True

    class WalkEnv(SimulatorBase):
        def __init__(self, config):
            self.cmd = np.zeros(2)          # (v, omega)
            super().__init__(config)
            self.jx = self.mj_model.joint("jx").qposadr[0]
            self.jy = self.mj_model.joint("jy").qposadr[0]
            self.jyaw = self.mj_model.joint("jyaw").qposadr[0]
            self.a_vx = self.mj_model.actuator("act_vx").id
            self.a_vy = self.mj_model.actuator("act_vy").id
            self.a_vyaw = self.mj_model.actuator("act_vyaw").id

        def updateControl(self, action):
            # 写 ctrl(velocity actuator),不写 qvel:接触力才顶得住(撞墙被挡住)
            v, w = self.cmd
            yaw = self.mj_data.qpos[self.jyaw]
            self.mj_data.ctrl[self.a_vx] = v * np.cos(yaw)
            self.mj_data.ctrl[self.a_vy] = v * np.sin(yaw)
            self.mj_data.ctrl[self.a_vyaw] = w

        def set_pose(self, x, y, yaw):
            import mujoco
            self.mj_data.qpos[self.jx] = x
            self.mj_data.qpos[self.jy] = y
            self.mj_data.qpos[self.jyaw] = yaw
            self.mj_data.qvel[:] = 0
            self.mj_data.ctrl[:] = 0
            self.cmd[:] = 0
            mujoco.mj_forward(self.mj_model, self.mj_data)

        def get_pose(self):
            return (float(self.mj_data.qpos[self.jx]), float(self.mj_data.qpos[self.jy]),
                    float(self.mj_data.qpos[self.jyaw]))

        def frame(self):
            return self.img_rgb_obs_s[0]            # (H,W,3) uint8 RGB

        def post_load_mjcf(self): pass
        def getObservation(self): return self.img_rgb_obs_s
        def getPrivilegedObservation(self): return {}
        def checkTerminated(self): return False
        def getReward(self): return None

    env = WalkEnv(WalkCfg())
    return env, W
