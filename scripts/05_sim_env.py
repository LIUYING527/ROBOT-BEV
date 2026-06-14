"""Pygame 箭头仿真（键盘手动控制）。

加载 outputs/bev_map.npy，渲染占据栅格 + 箭头智能体，方向键控制。
用于周二 demo：展示 x/y/θ 的变化轨迹。

用法：
    cd robot_bev_sim
    python scripts/05_sim_env.py
    方向键：↑前进 ↓后退 ←左转 →右转   ESC 退出
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.env import BEVSimEnv  # noqa: E402
from sim.renderer import PygameRenderer  # noqa: E402

import pygame  # noqa: E402

MAP_PATH = "outputs/bev_map.npy"


def main():
    if not os.path.exists(MAP_PATH):
        print(f"[WARN] 未找到 {MAP_PATH}，先跑 04_pcd_to_bev.py。")
        print("       这里用一张空白占据图占位，方便先调通仿真。")
        grid = np.zeros((200, 200), dtype=np.uint8)
    else:
        grid = np.load(MAP_PATH)

    env = BEVSimEnv(grid)
    renderer = PygameRenderer(grid.shape, scale=4)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        v, omega = 0.0, 0.0
        if keys[pygame.K_UP]:    v = 5.0
        if keys[pygame.K_DOWN]:  v = -5.0
        if keys[pygame.K_LEFT]:  omega = -1.0
        if keys[pygame.K_RIGHT]: omega = 1.0

        env.step(v, omega)
        renderer.render(grid, env.robot.state)

    renderer.close()


if __name__ == "__main__":
    main()
