#!/usr/bin/env python3
"""frontier_explore.py — M0/M1 无图自主探索模块 (0901 骨架, P-1仿真先行)
设计见 docs/真机部署方案v2_渐进式部署_0901.md §5。

思路(VLFM式, 参考 bdaiinstitute/vlfm MIT):
  在线占据栅格(ego累积, 指令积分位姿拼接, 容忍漂移——探索只要局部一致)
  → frontier = 自由/未知边界连通域(cv2, 不手搓)
  → 打分: 距离代价 + 覆盖增益(M1建图) / VLM价值(M0找目标) + 视角多样性(采集质量!)
  → 最优frontier → 快脑goal_ego → 执行 → 新观测融入
  → frontier耗尽=死胡同 → 回退最近未访问分支(0803骨架DFS经验)
终止: 目标confirm(M0) / 覆盖率>COV_STOP且帧数达标(M1) → 触发auto_map_pipeline.sh

⚠️ 本文件是骨架: 核心数据结构+打分写好, 与test_env/真机的接线处标TODO(P-1)。
铁律: 占据/边界提取用cv2现成函数, 别手搓([[prefer-official-code-not-handroll]])。
"""
import numpy as np
import cv2

RES = 0.05          # m/px, 与occupancy_bev.py一致
WIN = 400           # ego滚动窗 400px = ±10m
UNKNOWN, FREE, OCC = 0, 1, 2
# 采集质量约束(0720/0623教训): 探索打分必须带视角项, 采集密度决定重建天花板
MIN_FRAMES = 500    # M1首探帧数下限(510帧/18×12m成功先例)
COV_STOP = 0.85     # M1覆盖率终止阈(P-1验收线)


class OnlineOccGrid:
    """ego系滚动占据栅格: 每帧深度→点云→世界系累积(位姿来自指令积分/RELOC)。
    漂移处理: 只承诺局部一致; 长期漂移由M2的RELOC兜底(M0/M1本来就没有绝对定位)。"""

    def __init__(self, res=RES, win=WIN):
        self.res, self.win = res, win
        self.grid = np.full((win, win), UNKNOWN, np.uint8)   # 世界系栅格
        self.origin = np.zeros(2)                            # grid[0,0]的世界xy
        self._origin_set = False
        self.hits = np.zeros((win, win), np.uint16)          # 观测计数(采集密度统计用)

    def update(self, pts_world_xy, free_rays_xy=None, robot_xy=None):
        """pts_world_xy: (N,2)障碍物点(深度反投影, 世界系); free_rays: 射线经过的自由空间采样点。
        首次调用必须给robot_xy: 以机器人为中心定origin(否则负坐标全被裁)。
        TODO(P-1): 接 zed npz backproject(真机) / test_env渲染ED(仿真)。
        TODO(P-1): 走出窗口时recenter(整体平移grid, 或直接换成无限map dict)。"""
        if robot_xy is not None and not self._origin_set:
            self.origin = np.asarray(robot_xy, float) - self.win * self.res / 2
            self._origin_set = True
        ij = np.floor((pts_world_xy - self.origin) / self.res).astype(int)
        ok = (ij >= 0).all(1) & (ij < self.win).all(1)
        self.grid[ij[ok, 0], ij[ok, 1]] = OCC
        self.hits[ij[ok, 0], ij[ok, 1]] += 1
        if free_rays_xy is not None:
            fj = np.floor((free_rays_xy - self.origin) / self.res).astype(int)
            fok = (fj >= 0).all(1) & (fj < self.win).all(1)
            m = self.grid[fj[fok, 0], fj[fok, 1]] == UNKNOWN
            self.grid[fj[fok, 0][m], fj[fok, 1][m]] = FREE

    def coverage(self):
        known = (self.grid != UNKNOWN).sum()
        free = (self.grid == FREE).sum()
        return free / max(known, 1)


def extract_frontiers(grid, robot_ij, min_size=20):
    """frontier = 与未知相邻的自由格连通域。返回 [(centroid_world_xy, size)]。
    cv2.connectedComponentsWithStats 现成, 不手搓洪水填充。"""
    free = (grid == FREE).astype(np.uint8)
    unknown_dil = cv2.dilate((grid == UNKNOWN).astype(np.uint8), np.ones((3, 3), np.uint8))
    frontier_mask = free * unknown_dil
    n, lab, stats, cent = cv2.connectedComponentsWithStats(frontier_mask, connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out.append((cent[i], stats[i, cv2.CC_STAT_AREA]))
    return out


def score_frontier(cent_ij, size, robot_ij, mode, vlm_value=0.0, view_novelty=0.0):
    """打分 = 覆盖/尺寸增益 - 距离代价 + 模式项。
    mode='find'(M0找目标): vlm_value主导; mode='map'(M1建图): 覆盖+视角新颖度主导。
    ⚠️ view_novelty 是采集质量命门: 只冲最近frontier会采成单视角穿行, 重建必糊。"""
    dist = np.hypot(*(np.asarray(cent_ij) - robot_ij)) * RES
    base = np.log1p(size) - 0.5 * dist
    if mode == 'find':
        return base + 2.0 * vlm_value
    return base + 1.0 * view_novelty


class FrontierExplorer:
    """探索决策器: 每步给快脑一个goal_ego。死胡同回退=frontier耗尽时回最近未访问分支栈。"""

    def __init__(self, mode='map'):
        assert mode in ('find', 'map')
        self.mode = mode
        self.occ = OnlineOccGrid()
        self.backtrack = []   # (world_xy) 未访问分支栈(骨架DFS)
        self.n_frames = 0

    def step(self, pose_xy, pts_world, vlm_value_fn=None):
        """返回 goal_world_xy 或 None(终止)。
        TODO(P-1): 接 test_env 闭环——goal_world→ego系→R2R快脑status→轨迹→执行。
        TODO(真机): 接 run_lab_mission_real 状态机, 位姿=指令积分(接受漂移)。"""
        robot_ij = np.floor((pose_xy - self.occ.origin) / RES).astype(int)
        fronts = extract_frontiers(self.occ.grid, robot_ij)
        if not fronts:
            if not self.backtrack:
                return None                      # 探索完成
            return np.asarray(self.backtrack.pop())   # 死胡同回退
        best, bs = None, -1e9
        for cent, size in fronts:
            vv = vlm_value_fn(cent) if vlm_value_fn else 0.0
            s = score_frontier(cent, size, robot_ij, self.mode, vlm_value=vv)
            if s > bs:
                bs, best = s, cent
        return best * RES + self.occ.origin

    def done_mapping(self):
        return self.occ.coverage() >= COV_STOP and self.n_frames >= MIN_FRAMES


if __name__ == '__main__':
    # TODO(P-1): test_env 闭环验收——走廊/lab两图, 覆盖率>85%零碰撞。
    # 接线: env=CorridorTestEnv(...); 每步 exp.step(believed_xy, ed_backproject)
    #       → goal→R2R快脑(goalnav_r2r_dense.pt)status→轨迹→env.step(vx,vy,wz)
    print('skeleton only — P-1 在仿真 test_env 实现闭环, 勿直接上真机')
