#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""0903 PPT 素材: 关键数据图(中文Noto Sans CJK JP)。输出到 outputs/ppt_素材_0903/"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
OUT = '/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/ppt_素材_0903'

C_BLUE = '#2E5B9A'; C_ORANGE = '#E88A1A'; C_GRAY = '#8A8A8A'
C_RED = '#C0392B'; C_GREEN = '#27AE60'

# ---------- 图1: R2R-CE val_unseen SR 对标 ----------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
names = ['我们\nv5慢脑\n(0831)', '我们\ndense快脑\n+oracle目标', '我们\nhybrid全量\n(0903)',
         'DualVLN\nrepro', 'DualVLN\n官方', 'Qwen-\nRobotNav 4B']
vals = [5.3, 62, 56.0, 58.7, 64.3, 66.9]
colors = [C_GRAY, C_GREEN, C_BLUE, C_ORANGE, C_ORANGE, C_RED]
bars = ax.bar(names, vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 1.2, f'{v}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('成功率 SR (%)', fontsize=12)
ax.set_ylim(0, 76)
ax.set_title('R2R-CE val_unseen 成功率对标', fontsize=14, fontweight='bold')
ax.axhline(64.3, ls='--', color=C_ORANGE, lw=1, alpha=0.6)
ax.axhline(66.9, ls='--', color=C_RED, lw=1, alpha=0.6)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/图1_R2R_SR对标.png', dpi=130); plt.close()

# ---------- 图2: 132条失败解剖 ----------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
sizes = [22, 36, 36, 38]
labels = ['A类: 到3m没停好\n(停车问题)', 'B类: 擦边3-5m\n(走位差1-2m)',
          'B类: 中迷5-8m', 'B类: 深迷≥8m']
colors2 = [C_RED, C_ORANGE, '#E8C44A', C_GRAY]
wedges, _, autotexts = ax.pie(sizes, labels=labels, colors=colors2, autopct='%1.0f%%',
                              startangle=90, counterclock=False,
                              textprops={'fontsize': 11})
for at in autotexts:
    at.set_color('white'); at.set_fontweight('bold'); at.set_fontsize(12)
ax.set_title('hybrid 300ep 失败 132 条解剖（成功 168 条）', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUT}/图2_失败解剖.png', dpi=130); plt.close()

# ---------- 图3: anchor 匹配质量 ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
old = np.load('/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/anchors_car20_plus4_robot.npy')
new = np.load('/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/anchors_r2r_dense_k24.npy')
for ax, a, t in [(axes[0], old, '旧 anchor（车端轨迹缩放）'), (axes[1], new, '新 anchor（R2R 数据聚类）')]:
    for i in range(len(a)):
        ax.plot(a[i, :, 0], a[i, :, 1], '-o', ms=2.5, lw=1.2, alpha=0.8)
    ax.set_title(t, fontsize=12, fontweight='bold')
    ax.set_xlabel('前向 x (m)'); ax.set_ylabel('左 y (m)')
    ax.axis('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 2.0); ax.set_ylim(-1.0, 1.0)
fig.suptitle('anchor 轨迹对比：R2R 轨迹到最近 anchor 平均距离 0.187 → 0.022（+88%）',
             fontsize=12.5, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f'{OUT}/图3_anchor对比.png', dpi=130, bbox_inches='tight'); plt.close()

# ---------- 图4: 差距四层(从当前56追到上限) ----------
fig, ax = plt.subplots(figsize=(7.0, 4.0))
layers = ['①慢脑\n(目标/STOP不准)', '②终段停车\n(A类22条)', '③接口损耗\n(官方vs repro)', '④天花板\n(单目SOTA线)']
pts = [46, 7.6, 5.6, 8.0]
colors4 = [C_RED, C_ORANGE, '#E8C44A', C_GRAY]
bars = ax.bar(layers, pts, color=colors4, width=0.55)
for b, v in zip(bars, pts):
    ax.text(b.get_x()+b.get_width()/2, v+1, f'{v}pt', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('SR 差距 (pt)', fontsize=12)
ax.set_ylim(0, 52)
ax.set_title('差距四层定量（论文叙事框架）', fontsize=13, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/图4_差距四层.png', dpi=130); plt.close()

# ---------- 图5: LH-VLN(LHPR-VLN) ISR 对标 ----------
fig, ax = plt.subplots(figsize=(6.6, 4.0))
names = ['FantasyVLN\n(online)', 'MGDM\n(offline)', '我们\nzero-shot', 'DualVLN']
vals = [11.01, 4.69, 4.25, 2.98]
colors5 = [C_RED, C_ORANGE, C_GREEN, C_GRAY]
bars = ax.bar(names, vals, color=colors5, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel('ISR (%)', fontsize=12)
ax.set_ylim(0, 13)
ax.set_title('LH-VLN（长程VLN）独立子任务成功率 ISR', fontsize=12.5, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/图5_LHVLN对标.png', dpi=130); plt.close()

print('=== 5 张图完成 ===')
import os
for f in sorted(os.listdir(OUT)):
    print(' ', f)
