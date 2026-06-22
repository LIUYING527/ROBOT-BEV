"""干净的 2D 仿真世界定义（矢量地图：墙线 + 目标点）。

设计目标：像游戏俯视图一样规整、能一眼看懂。墙用线段表示，
目标按类别放置。后续可把真实数据提炼出的墙线塞进 walls 即可复用渲染。
单位：米。
"""
import numpy as np


class World2D:
    def __init__(self, bounds, walls, targets):
        self.bounds = bounds          # (xmin, ymin, xmax, ymax)
        self.walls = walls            # list[(x1,y1,x2,y2)]
        self.targets = targets        # list[(x, y, category)]

    @staticmethod
    def default_map():
        """一个规整的多房间巡检场地（参考小红书那张图的布局风格）。"""
        W, H = 16.0, 14.0
        b = (0.0, 0.0, W, H)
        walls = []

        def seg(x1, y1, x2, y2):
            walls.append((x1, y1, x2, y2))

        # 外墙（四周）
        seg(0, 0, W, 0); seg(0, H, W, H); seg(0, 0, 0, H); seg(W, 0, W, H)

        # 内部隔墙，留门洞(gap)分成几个房间
        # 竖墙 x=6，从下到上，中间留门
        seg(6, 0, 6, 5.0); seg(6, 7.0, 6, H)
        # 竖墙 x=11，上半段
        seg(11, 6.5, 11, H)
        # 横墙 y=8（左半），留门
        seg(0, 8.5, 2.5, 8.5); seg(4.5, 8.5, 6, 8.5)
        # 横墙 y=6.5（右侧房间下沿）
        seg(11, 6.5, W, 6.5)
        # 右下小隔间
        seg(11, 3.5, W, 3.5); seg(13.5, 0, 13.5, 3.5)

        # 目标点：三类（蓝/绿/红），模拟不同任务对象
        targets = []
        rng = np.random.RandomState(7)  # 固定种子，可复现
        # 蓝色一簇（左上房间）—— 比如一组待巡检设备
        for _ in range(6):
            targets.append((rng.uniform(0.8, 5.2), rng.uniform(9.0, 13.2), "blue"))
        # 绿色散布（中部开阔区）—— 比如路径观测点
        for _ in range(11):
            targets.append((rng.uniform(2.0, 13.5), rng.uniform(4.0, 12.5), "green"))
        # 红色（下部）—— 比如电箱/重点目标
        for _ in range(6):
            targets.append((rng.uniform(6.5, 13.0), rng.uniform(0.5, 3.0), "red"))
        return World2D(b, walls, targets)

    @staticmethod
    def default_route():
        """一条穿过门洞、不撞墙的巡检路线（路点，米）。
        路线：左下 → 过 x=6 门洞(y5~7) → 中部 → 上行 → 过 y=8.5 门洞(x2.5~4.5)
              → 左上蓝色目标房间。"""
        return np.array([
            (2.5, 2.0), (3.0, 5.8), (5.5, 6.0), (7.2, 6.0), (9.0, 4.8),
            (9.5, 8.0), (8.0, 10.0), (6.5, 9.5), (4.0, 8.5), (3.5, 11.0),
            (2.0, 12.5), (4.5, 12.8),
        ], dtype=float)
