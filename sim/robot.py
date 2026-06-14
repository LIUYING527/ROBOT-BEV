"""箭头智能体：差速运动学。

车体简化为箭头，状态量 (x, y, theta)。坐标单位为栅格格子，theta 为弧度。
"""
import numpy as np


class ArrowRobot:
    def __init__(self, x=0.0, y=0.0, theta=0.0, dt=0.1):
        self.state = {"x": x, "y": y, "theta": theta}
        self.dt = dt

    def step(self, v, omega):
        """差速模型积分一步。v: 线速度（格/步），omega: 角速度（弧度/步）。"""
        s = self.state
        s["theta"] += omega * self.dt
        s["x"] += v * np.cos(s["theta"]) * self.dt
        s["y"] += v * np.sin(s["theta"]) * self.dt
        return self.pose

    @property
    def pose(self):
        s = self.state
        return np.array([s["x"], s["y"], s["theta"]])

    def reset(self, x, y, theta=0.0):
        self.state = {"x": x, "y": y, "theta": theta}
