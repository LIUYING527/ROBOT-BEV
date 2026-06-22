"""DISCOVERSE 真实仿真器:在 VGGT→3DGS 的世界帧场景里放一个差速机器人 + 车载相机,headless 渲染。
被 sim_walk_discoverse.py(脚本视频)和 sim_walk_server.py(浏览器实时驱动)共用。

前置:scripts/align_gs_world.py 已产出 outputs/gs_vggto_<s>_world.ply + sim_world_<s>.npz。
运行环境:MUJOCO_GL=egl + PYTHONPATH=third_party/discoverse(本体未 pip 装)。
"""
import os
# headless EGL:必须在 import mujoco / OpenGL 之前设好(否则 PyOpenGL 选错平台 → PLATFORM_DEVICE 报错)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MuJoCo 相机:看局部 -z,+x 右,+y 上。要它看机器人前向(+x_body)、up=+z_body:
#   cam→body 列 = [right=-y, up=+z, -forward=-x]
_CAM_M = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
_CAM_QUAT_XYZW = Rotation.from_matrix(_CAM_M).as_quat()
FWD_CAM_QUAT_WXYZ = _CAM_QUAT_XYZW[[3, 0, 1, 2]]   # wxyz for MJCF


def build_mjcf(session, start_xy, cam_height=1.2, width=1280, height=720, fovy=75.0):
    sx, sy = float(start_xy[0]), float(start_xy[1])
    q = FWD_CAM_QUAT_WXYZ
    xml = f"""<mujoco model="vggt_walk_{session}">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="Euler"/>
  <visual><global offwidth="{max(width,1920)}" offheight="{max(height,1080)}"/></visual>
  <worldbody>
    <light pos="0 0 20" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="80 80 0.1" pos="0 0 0" rgba="0.25 0.27 0.27 1"/>
    <body name="background" pos="0 0 0" quat="1 0 0 0">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
    </body>
    <body name="robot" pos="{sx} {sy} 0">
      <joint name="jx" type="slide" axis="1 0 0"/>
      <joint name="jy" type="slide" axis="0 1 0"/>
      <joint name="jyaw" type="hinge" axis="0 0 1"/>
      <geom name="chassis" type="box" size="0.25 0.18 0.12" pos="0 0 0.12" rgba="1 0.55 0.1 1" mass="6"/>
      <camera name="fpv" pos="0 0 {cam_height}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" fovy="{fovy}"/>
    </body>
  </worldbody>
</mujoco>"""
    path = os.path.join(ROOT, "outputs", f"_sim_walk_{session}.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def make_env(session="114830", width=1280, height=720, fps=24, cam_height=1.2, fovy=75.0):
    """构建并返回 (env, world_npz)。需先设好 MUJOCO_GL/PYTHONPATH。"""
    from discoverse.envs import SimulatorBase
    from discoverse.utils.base_config import BaseConfig

    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{session}.npz"))
    start_xy = W["waypoints_xy"][0]
    ply = os.path.join(ROOT, "outputs", f"gs_vggto_{session}_world.ply")
    mjcf = build_mjcf(session, start_xy, cam_height, width, height, fovy)

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
            self.vx = self.mj_model.joint("jx").dofadr[0]
            self.vy = self.mj_model.joint("jy").dofadr[0]
            self.vyaw = self.mj_model.joint("jyaw").dofadr[0]

        def updateControl(self, action):
            v, w = self.cmd
            yaw = self.mj_data.qpos[self.jyaw]
            self.mj_data.qvel[self.vx] = v * np.cos(yaw)
            self.mj_data.qvel[self.vy] = v * np.sin(yaw)
            self.mj_data.qvel[self.vyaw] = w

        def set_pose(self, x, y, yaw):
            import mujoco
            self.mj_data.qpos[self.jx] = x
            self.mj_data.qpos[self.jy] = y
            self.mj_data.qpos[self.jyaw] = yaw
            self.mj_data.qvel[:] = 0
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
