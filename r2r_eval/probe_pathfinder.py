#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 habitat-sim pathfinder.find_path 签名/返回格式。"""
import os
import sys
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from run_r2r import make_env  # noqa: E402


def main():
    env = make_env('train', 640, 480, 79)
    eps = list(env.episodes)
    ep = eps[0]
    env.episodes = [ep]
    env.reset()
    ref = ep.reference_path
    goal = np.array(ep.goals[0].position, np.float64)
    pf = env.sim.pathfinder
    print(f'ep={ep.episode_id} ref={len(ref)}pts goal={goal}')

    import habitat_sim
    a = np.array(ref[1], np.float64)
    b = goal
    print('\n=== find_path(ShortestPath) ===')
    try:
        sp = habitat_sim.ShortestPath()
        sp.requested_start = a
        sp.requested_end = b
        found = pf.find_path(sp)
        print('found =', found)
        print('dir:', [m for m in dir(sp) if not m.startswith('_') and not callable(getattr(sp, m, None))])
        pts = np.asarray(sp.points)
        print(f'points shape={pts.shape}')
        print(f'first={pts[0]} last={pts[-1]}')
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        print(f'段间距 min/mean/max = {seg.min():.3f}/{seg.mean():.3f}/{seg.max():.3f}')
        print('geodesic_distance =', sp.geodesic_distance)
        direct = float(np.linalg.norm(a - b))
        print(f'绕行比 = geodesic/direct = {sp.geodesic_distance/direct:.3f}')
    except Exception as e:
        import traceback
        traceback.print_exc()

    env.close()


if __name__ == '__main__':
    main()
