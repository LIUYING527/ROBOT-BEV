#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""慢脑数据透视投影 —— world goal -> 相机 -> 像素坐标（纯 numpy，可单测）。

Habitat 相机约定：
  - rotation 为 [x,y,z,w] 四元数，表示「相机 -> 世界」的旋转 R。
  - 相机 forward = R @ (0,0,-1)，即相机看向 -z；点在相机前方 <=> 相机坐标 z < 0。
  - depth 传感器返回的是沿 forward 的正距离 = -z_cam。
  - 像素：u = cx + fx * x/(-z), v = cy - fy * y/(-z)，fx=fy=(w/2)/tan(hfov/2)（正方形像素）。

输出归一化像素坐标 (u/w, v/h) ∈ [0,1]²；目标不可见（后方或越界）返回 (-1,-1)。
"""
import numpy as np


def quat_to_rot(q):
    """四元数 [x,y,z,w] -> 3x3 旋转矩阵（相机->世界）。"""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def world_to_cam(point, position, quat):
    """世界坐标 point -> 相机坐标。"""
    R = quat_to_rot(quat)
    return R.T @ (np.asarray(point, np.float64) - np.asarray(position, np.float64))


def focal_from_hfov(width, hfov_deg):
    return (width / 2.0) / np.tan(np.deg2rad(hfov_deg / 2.0))


def project_goal_to_pixel(goal, position, quat, width, height, hfov_deg, margin=0.0):
    """把世界坐标 goal 投影到当前相机画面。

    返回 (u, v, visible):
      u,v 为像素坐标（浮点）；visible=False 表示目标在相机后方或超出画面（含 margin 收缩）。
    """
    cam = world_to_cam(goal, position, quat)
    zc = cam[2]
    if zc >= 0.0:                       # 在相机后方（forward=-z）
        return -1.0, -1.0, False
    fx = focal_from_hfov(width, hfov_deg)
    fy = fx                              # 正方形像素
    u = width / 2.0 + fx * cam[0] / (-zc)
    v = height / 2.0 - fy * cam[1] / (-zc)
    visible = (margin <= u < width - margin) and (margin <= v < height - margin)
    return u, v, visible


def project_goal_normalized(goal, position, quat, width, height, hfov_deg, margin=0.0):
    """归一化像素坐标 (u/w, v/h) ∈ [0,1]²；不可见返回 (-1,-1, False)。"""
    u, v, vis = project_goal_to_pixel(goal, position, quat, width, height, hfov_deg, margin)
    if not vis:
        return -1.0, -1.0, False
    return u / width, v / height, True


if __name__ == "__main__":
    # --- 单测：goal 在正前方应投影到图像中心 ---
    pos = np.array([0.0, 0.0, 0.0])
    quat = np.array([0.0, 0.0, 0.0, 1.0])   # 单位四元数, 相机 forward = -z
    goal_front = np.array([0.0, 0.0, -2.0])   # 正前方 2m
    u, v, vis = project_goal_to_pixel(goal_front, pos, quat, 640, 480, 79)
    print(f"[正前方] u={u:.1f} (期望 320) v={v:.1f} (期望 240) visible={vis}")
    assert vis and abs(u - 320) < 0.1 and abs(v - 240) < 0.1

    # --- 右前方 30°（在 79° 视野内）：u 应 > 320, v≈240 ---
    goal_right = np.array([2.0 * np.tan(np.deg2rad(30)), 0.0, -2.0])
    u, v, vis = project_goal_to_pixel(goal_right, pos, quat, 640, 480, 79)
    print(f"[右前方30°] u={u:.1f} (期望>320) v={v:.1f} (期望≈240) visible={vis}")
    assert vis and u > 320 and abs(v - 240) < 0.1

    # --- 右前方 45°（超出 79° 半角 39.5°）：应越界 visible=False ---
    goal_out = np.array([2.0, 0.0, -2.0])
    u, v, vis = project_goal_to_pixel(goal_out, pos, quat, 640, 480, 79)
    print(f"[右前方45°越界] u={u:.1f} (期望>640) visible={vis} (期望 False)")
    assert not vis and u > 640

    # --- 后方：不可见 ---
    goal_back = np.array([0.0, 0.0, 2.0])
    u, v, vis = project_goal_to_pixel(goal_back, pos, quat, 640, 480, 79)
    print(f"[后方] u={u} v={v} visible={vis} (期望 False)")
    assert not vis

    # --- 归一化坐标 ---
    nu, nv, vis = project_goal_normalized(goal_front, pos, quat, 640, 480, 79)
    print(f"[归一化] ({nu:.3f},{nv:.3f}) 期望 (0.500,0.500) visible={vis}")
    assert vis and abs(nu - 0.5) < 1e-3 and abs(nv - 0.5) < 1e-3

    print("\n所有投影单测通过 ✅")
