#!/usr/bin/env python3
"""对比图: Infinite VLM(自研) vs Qwen3.5-4B, 走廊蓝泵巡检.
产出 fig4_compare.png: (a)延迟对比 (b)检测指标对比 (c)延迟→决策盲走距离."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs/infinite_vlm_eval")
S = json.load(open(os.path.join(OUT, "compare_summary.json")))
inf, qw = S["infinite"], S["qwen"]
CRUISE_V = 0.07 * 2  # m/step * 2Hz近似 -> m/s 量级(仅示意决策盲走)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) 延迟
ax = axes[0]
names = ["Infinite VLM\n(自研)", "Qwen3.5-4B\n(对照)"]
med = [inf["lat_median_s"], qw["lat_median_s"]]
p90 = [inf["lat_p90_s"], qw["lat_p90_s"]]
x = np.arange(2)
ax.bar(x - 0.18, med, 0.36, label="中位延迟", color="#4c72b0")
ax.bar(x + 0.18, p90, 0.36, label="P90延迟", color="#a3bce0")
for i, (m, p) in enumerate(zip(med, p90)):
    ax.text(i - 0.18, m + 0.03, f"{m:.2f}s", ha="center", fontsize=10)
    ax.text(i + 0.18, p + 0.03, f"{p:.2f}s", ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("秒"); ax.set_title(f"单次决策延迟（自研快 {qw['lat_median_s']/inf['lat_median_s']:.1f}×）")
ax.legend()

# (b) 检测指标
ax = axes[1]
labs = ["准确率", "精确率", "召回率", "F1"]
iv = [inf["accuracy"], inf["precision"], inf["recall"], inf["f1"]]
qv = [qw["accuracy"], qw["precision"], qw["recall"], qw["f1"]]
x = np.arange(4)
ax.bar(x - 0.18, [v * 100 for v in iv], 0.36, label="Infinite VLM", color="#c44e52")
ax.bar(x + 0.18, [v * 100 for v in qv], 0.36, label="Qwen3.5-4B", color="#dd9ea0")
for i, (a, b) in enumerate(zip(iv, qv)):
    ax.text(i - 0.18, a * 100 + 1, f"{a*100:.0f}", ha="center", fontsize=9)
    ax.text(i + 0.18, b * 100 + 1, f"{b*100:.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0, 100)
ax.set_ylabel("%"); ax.set_title("检测指标（召回=不漏泵, 巡检更看重）")
ax.legend()

# (c) 决策盲走距离 = 延迟 * 巡航速度
ax = axes[2]
blind = [inf["lat_median_s"] * CRUISE_V * 100, qw["lat_median_s"] * CRUISE_V * 100]  # cm
bars = ax.bar(names, blind, color=["#55a868", "#a8cbb0"])
for b, v in zip(bars, blind):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.0f} cm", ha="center", fontsize=11)
ax.set_ylabel("cm"); ax.set_title("决策期间机器人盲走距离\n(延迟×巡航速度, 越小越可控)")

fig.suptitle("Infinite VLM（自研）vs Qwen3.5-4B — 走廊蓝水泵巡检 (263帧, 同prompt/同GT)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_compare.png"), dpi=130)
print("saved fig4_compare.png")
print("盲走: inf %.0fcm qwen %.0fcm" % (blind[0], blind[1]))
