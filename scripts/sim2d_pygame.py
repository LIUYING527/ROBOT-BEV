"""抽象 2D 仿真器(pygame 交互版)—— WASD 开着箭头 agent 跑,带激光射线。

加载 walls_tsdf_111450.json(sim2d_real.py 生成的真实墙线段),也可换任意墙。
控制: W/S 前进后退, A/D 转向, ESC 退出。需要有显示环境(本地跑,非无头服务器)。

用法(本地): python scripts/sim2d_pygame.py [walls.json]
"""
import sys
import json
import math
import numpy as np

try:
    import pygame
except ImportError:
    sys.exit("需要 pygame: pip install pygame")

WALLS_JSON = sys.argv[1] if len(sys.argv) > 1 else "outputs/walls_111450.json"
PX_PER_M = 28          # 像素/米
N_RAYS = 48; FOV = 200; LIDAR_MAX = 6.0
FLOOR_A = (159, 196, 232); FLOOR_B = (143, 184, 224); WALL_C = (122, 31, 43)


def ray_hit(px, py, ang, walls, rmax):
    dx, dy = math.cos(ang), math.sin(ang); best = rmax
    for x1, y1, x2, y2 in walls:
        ex, ey = x2 - x1, y2 - y1; den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        t = ((x1 - px) * ey - (y1 - py) * ex) / den
        u = ((x1 - px) * dy - (y1 - py) * dx) / den
        if t > 0 and 0 <= u <= 1 and t < best:
            best = t
    return best


def main():
    d = json.load(open(WALLS_JSON))
    walls = d["walls"]; x0, y0, x1, y1 = d["bounds"]
    W = int((x1 - x0) * PX_PER_M); H = int((y1 - y0) * PX_PER_M)
    pygame.init(); screen = pygame.display.set_mode((W, H)); clock = pygame.time.Clock()

    def w2s(x, y):  # 世界→屏幕
        return int((x - x0) * PX_PER_M), int(H - (y - y0) * PX_PER_M)

    ax, ay, ath = (x0 + x1) / 2, (y0 + y1) / 2, 0.0
    run = True
    while run:
        dt = clock.tick(30) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            run = False
        v = (keys[pygame.K_w] - keys[pygame.K_s]) * 2.0
        om = (keys[pygame.K_a] - keys[pygame.K_d]) * 1.5
        ath += om * dt; ax += v * math.cos(ath) * dt; ay += v * math.sin(ath) * dt

        # 棋盘地面
        for iy in range(int(y1 - y0) + 1):
            for ix in range(int(x1 - x0) + 1):
                sx, sy = w2s(x0 + ix, y0 + iy + 1)
                pygame.draw.rect(screen, FLOOR_A if (ix + iy) % 2 else FLOOR_B,
                                 (sx, sy, PX_PER_M + 1, PX_PER_M + 1))
        # 激光
        for a in np.radians(np.linspace(-FOV / 2, FOV / 2, N_RAYS)) + ath:
            r = ray_hit(ax, ay, a, walls, LIDAR_MAX)
            pygame.draw.line(screen, (230, 200, 0), w2s(ax, ay),
                             w2s(ax + r * math.cos(a), ay + r * math.sin(a)), 1)
        # 墙
        for x1w, y1w, x2w, y2w in walls:
            pygame.draw.line(screen, WALL_C, w2s(x1w, y1w), w2s(x2w, y2w), 5)
        # 箭头 agent
        sx, sy = w2s(ax, ay)
        ex, ey = w2s(ax + 1.0 * math.cos(ath), ay + 1.0 * math.sin(ath))
        pygame.draw.circle(screen, (27, 108, 255), (sx, sy), 7)
        pygame.draw.line(screen, (17, 17, 17), (sx, sy), (ex, ey), 4)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
