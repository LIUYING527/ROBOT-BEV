import torch, numpy as np, os
RUNS = '/data/DongBaorong/BEVLOOK/robot_bev_sim/goalnav/runs'
ref_new = np.load('/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/anchors_r2r_dense_k24.npy')
ref_old = np.load('/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/anchors_car20_plus4_robot.npy')
print('新anchor k24 shape=%s | 旧anchor car20 shape=%s' % (ref_new.shape, ref_old.shape))
for name in ['goalnav_r2r_dense_fullft_k24_ep0.pt', 'goalnav_r2r_dense.pt']:
    p = os.path.join(RUNS, name)
    if not os.path.exists(p):
        print('%s : 不存在' % name); continue
    ck = torch.load(p, map_location='cpu')
    pa = ck['trajectory_head']['plan_anchor'].numpy()
    d_new = np.abs(pa - ref_new).max() if pa.shape == ref_new.shape else float('nan')
    d_old = np.abs(pa - ref_old).max() if pa.shape == ref_old.shape else float('nan')
    verdict = 'k24(新)' if d_new < 1e-5 else ('car20(旧)' if d_old < 1e-5 else '两个都不是!')
    print('%-40s plan_anchor%s | 与k24差=%.2e 与car20差=%.2e  => %s' % (
        name, tuple(pa.shape), d_new, d_old, verdict))
