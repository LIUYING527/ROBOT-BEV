"""主环境类（gym 风格接口）。

第一版只做运动学推进 + 边界保护，碰撞检测预留接口（见 4.4 简化决策）。
后续上层 Rule+VLM 通过 step() 的 (v, omega) 驱动，或扩展 action 空间。
"""
import numpy as np

from .robot import ArrowRobot


class BEVSimEnv:
    def __init__(self, grid, dt=0.1, start=None):
        """grid: (H, W) uint8 占据栅格，1=障碍。"""
        self.grid = grid
        self.H, self.W = grid.shape
        self.dt = dt
        sx, sy = (self.W / 2, self.H / 2) if start is None else start
        self.robot = ArrowRobot(x=sx, y=sy, theta=0.0, dt=dt)

    def reset(self, start=None):
        sx, sy = (self.W / 2, self.H / 2) if start is None else start
        self.robot.reset(sx, sy, 0.0)
        return self._obs()

    def step(self, v, omega):
        prev = self.robot.pose.copy()
        pose = self.robot.step(v, omega)
        # 边界保护：越界则回退（碰撞检测 TODO）
        if not (0 <= pose[0] < self.W and 0 <= pose[1] < self.H):
            self.robot.reset(prev[0], prev[1], pose[2])
        obs = self._obs()
        reward, done, info = 0.0, False, {}
        return obs, reward, done, info

    def is_collision(self, x, y):
        """占据格碰撞检测（预留，第一版未启用）。"""
        ix, iy = int(x), int(y)
        if 0 <= ix < self.W and 0 <= iy < self.H:
            return bool(self.grid[iy, ix])
        return True  # 越界视为碰撞

    def _obs(self):
        """返回当前观测。后续可替换为 BEV 局部切片（车前方扇形/矩形）。"""
        return {"pose": self.robot.pose}
