"""Pygame 可视化：占据栅格 + 箭头智能体。"""
import numpy as np
import pygame


class PygameRenderer:
    def __init__(self, grid_shape, scale=4, fps=30):
        self.H, self.W = grid_shape
        self.scale = scale
        self.fps = fps
        pygame.init()
        self.screen = pygame.display.set_mode((self.W * scale, self.H * scale))
        pygame.display.set_caption("BEV Sim — Arrow Agent")
        self.clock = pygame.time.Clock()

    def _draw_grid(self, grid):
        # 障碍=深色，自由=浅灰
        img = ((1 - grid.T) * 220).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.stack([img] * 3, axis=-1))
        surf = pygame.transform.scale(surf, (self.W * self.scale, self.H * self.scale))
        self.screen.blit(surf, (0, 0))

    def _draw_arrow(self, state):
        cx, cy = state["x"] * self.scale, state["y"] * self.scale
        theta = state["theta"]
        L = 20
        tip = (cx + L * np.cos(theta), cy + L * np.sin(theta))
        left = (cx + 10 * np.cos(theta + 2.5), cy + 10 * np.sin(theta + 2.5))
        right = (cx + 10 * np.cos(theta - 2.5), cy + 10 * np.sin(theta - 2.5))
        pygame.draw.polygon(self.screen, (255, 0, 0), [tip, left, right])

    def render(self, grid, state):
        self._draw_grid(grid)
        self._draw_arrow(state)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
