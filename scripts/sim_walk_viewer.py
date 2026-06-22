"""本地桌面窗口版:在自己的电脑(带显示器 + NVIDIA 显卡)上亲手开仿真器。

和 sim_walk_server.py 同一套真仿真器(MuJoCo 物理 + 差速机器人 + VGGT→3DGS 背景),
只是把"推流到浏览器"换成"pygame 桌面窗口直接显示 + 键盘实时驱动"。

依赖: torch(cuda) + gsplat + mujoco + gaussian_renderer + discoverse(PYTHONPATH) + pygame + opencv
见 docs/LOCAL_SIM_SETUP.md。

用法(在仓库根目录):
  PYTHONPATH=third_party/discoverse python scripts/sim_walk_viewer.py 114830c
  # 可选: --width 960 --height 540 --fovy 75
键位: W/S 前后 · A/D 左右转 · Shift 加速 · R 回起点 · Esc/关窗 退出
渲染后端: 默认 MUJOCO_GL=egl(无头离屏,NVIDIA 上可用);若本地报错可试 MUJOCO_GL=glfw。
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "discoverse"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from sim_walk_common import make_env   # noqa: E402  (内部已 setdefault MUJOCO_GL=egl)

import pygame  # noqa: E402

SPEED = 1.2     # m/s,前进基准速度
TURN = 1.2      # rad/s,转向基准角速度
BOOST = 1.8     # Shift 加速倍率


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", default="114830c")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fovy", type=float, default=75.0)
    args = ap.parse_args()

    print(f"[viewer] 构建仿真器 session={args.session} ...", flush=True)
    env, W = make_env(args.session, width=args.width, height=args.height, fps=30, fovy=args.fovy)
    wp, yaw = W["waypoints_xy"], W["yaw"]
    start = (float(wp[0, 0]), float(wp[0, 1]), float(yaw[0]))
    env.set_pose(*start)
    env.render()
    print("[viewer] 就绪。窗口里按 W/S/A/D 开,Shift 加速,R 回起点,Esc 退出。", flush=True)

    pygame.init()
    pygame.display.set_caption(f"VGGT→3DGS→DISCOVERSE  walk  [{args.session}]")
    screen = pygame.display.set_mode((args.width, args.height))
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    env.set_pose(*start)
                    env.cmd[:] = 0

        k = pygame.key.get_pressed()
        v = (1 if k[pygame.K_w] else 0) - (1 if k[pygame.K_s] else 0)
        w = (1 if k[pygame.K_a] else 0) - (1 if k[pygame.K_d] else 0)
        boost = BOOST if (k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]) else 1.0
        env.cmd[:] = (v * SPEED * boost, w * TURN)

        env.step()
        img = env.frame()                                   # (H,W,3) uint8 RGB
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        screen.blit(surf, (0, 0))

        x, y, th = env.get_pose()
        hud = f"v={env.cmd[0]:+.1f} w={env.cmd[1]:+.1f}  pos=({x:.1f},{y:.1f}) yaw={np.degrees(th):.0f}deg  {clock.get_fps():.0f}fps"
        screen.blit(font.render(hud, True, (255, 255, 0)), (8, 6))
        screen.blit(font.render("W/S 前后  A/D 转  Shift 加速  R 回起点  Esc 退出", True, (180, 220, 255)),
                    (8, args.height - 24))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("[viewer] 已退出。", flush=True)


if __name__ == "__main__":
    main()
