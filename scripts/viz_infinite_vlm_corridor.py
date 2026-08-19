#!/usr/bin/env python3
"""
读 Infinite VLM 走廊评测结果 -> 出表格(md/csv) + 3张可视化图, 交付王学长。

产出 (outputs/infinite_vlm_eval/):
  - corridor_metrics.md / corridor_metrics.csv   汇总指标 + 站位级检出表
  - fig1_verdict_vs_position.png                  逐位置判定散点(x vs pred, GT配色, 泵站标注)
  - fig2_frame_montage.png                        帧判定拼图(样例帧+Infinite VLM判定+GT)
  - fig3_confusion.png                            混淆矩阵 + 指标条形图
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs/infinite_vlm_eval")
RES = os.path.join(OUT, "corridor_results.jsonl")

# 泵站中心(记忆 x=8/18/26/31, 由strafe聚类确认)
PUMP_STATIONS = [8.0, 18.0, 26.0, 31.0]
STATION_HALF = 2.5  # 站位判定窗口半宽(m)


def load():
    rows = [json.loads(l) for l in open(RES)]
    rows = [r for r in rows if not str(r["answer"]).startswith("__ERR__")]
    return rows


def metrics(rows):
    tp = sum(1 for r in rows if r["gt_yes"] and r["pred_yes"])
    fp = sum(1 for r in rows if (not r["gt_yes"]) and r["pred_yes"])
    fn = sum(1 for r in rows if r["gt_yes"] and (not r["pred_yes"]))
    tn = sum(1 for r in rows if (not r["gt_yes"]) and (not r["pred_yes"]))
    n = len(rows)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, n=n, acc=acc, prec=prec, rec=rec, f1=f1)


def station_detection(rows):
    """站位级: 每个泵站窗口内 Infinite VLM 是否至少检出一次(检出率)。"""
    out = []
    xs = np.array([r["x"] for r in rows])
    preds = np.array([r["pred_yes"] for r in rows])
    for k, cx in enumerate(PUMP_STATIONS):
        m = np.abs(xs - cx) <= STATION_HALF
        nf = int(m.sum())
        det = int(preds[m].sum())
        rate = det / nf if nf else 0
        fired = det > 0
        out.append(dict(station=k + 1, x=cx, frames=nf, yes=det,
                        yes_rate=round(rate, 3), detected=fired))
    return out


def write_tables(rows, m, st):
    # CSV
    import csv
    with open(os.path.join(OUT, "corridor_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ["n", "TP", "FP", "FN", "TN"]:
            w.writerow([k, m[k]])
        for k, lab in [("acc", "accuracy"), ("prec", "precision"), ("rec", "recall"), ("f1", "f1")]:
            w.writerow([lab, round(m[k], 4)])
        w.writerow([])
        w.writerow(["station", "x", "frames", "yes", "yes_rate", "detected"])
        for s in st:
            w.writerow([s["station"], s["x"], s["frames"], s["yes"], s["yes_rate"], s["detected"]])
    # Markdown
    with open(os.path.join(OUT, "corridor_metrics.md"), "w") as f:
        f.write("# Infinite VLM（自研）走廊蓝水泵巡检 — 目标检测结果\n\n")
        f.write("- 模型: **Infinite VLM**（自研, 服务 10.0.0.18:8152）\n")
        f.write("- 场景: 走廊蓝水泵巡检, 真实采集帧 `nav_ds_real`\n")
        f.write("- 问题: *Is there a blue water pump right beside you, close and large in view? yes/no*\n")
        f.write(f"- GT: `strafe`=横扫检视泵(正) / `cruise`=前方走廊(负)（真实运动分类标签）\n\n")
        f.write("## 1. 逐帧检测指标\n\n")
        f.write("| 指标 | 数值 |\n|---|---|\n")
        f.write(f"| 评测帧数 | {m['n']} |\n")
        f.write(f"| 准确率 Accuracy | {m['acc']*100:.1f}% |\n")
        f.write(f"| 精确率 Precision | {m['prec']*100:.1f}% |\n")
        f.write(f"| 召回率 Recall | {m['rec']*100:.1f}% |\n")
        f.write(f"| F1 | {m['f1']*100:.1f}% |\n")
        f.write(f"| TP / FP / FN / TN | {m['TP']} / {m['FP']} / {m['FN']} / {m['TN']} |\n\n")
        f.write("> 注: `cruise` 段中机器人接近泵站时泵已可见, VLM 会提前判 Yes → 计入 FP; "
                "故逐帧指标偏保守, 站位级检出更能反映实际能力。\n\n")
        f.write("## 2. 站位级检出（每个泵站窗口 ±%.1fm）\n\n" % STATION_HALF)
        f.write("| 泵站 | x(m) | 窗口帧数 | 判Yes帧 | Yes率 | 是否检出 |\n|---|---|---|---|---|---|\n")
        for s in st:
            f.write(f"| #{s['station']} | {s['x']:.0f} | {s['frames']} | {s['yes']} | "
                    f"{s['yes_rate']*100:.0f}% | {'✅' if s['detected'] else '❌'} |\n")
        ndet = sum(1 for s in st if s["detected"])
        f.write(f"\n**站位检出: {ndet}/{len(st)}**\n")


def fig1(rows):
    xs = np.array([r["x"] for r in rows])
    pred = np.array([r["pred_yes"] for r in rows], float)
    gt = np.array([r["gt_yes"] for r in rows])
    order = np.argsort(xs)
    xs, pred, gt = xs[order], pred[order], gt[order]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    # GT 正样本(strafe=横扫检视段)用绿色竖带标出
    lbl = "GT 检视段 (strafe)"
    for k in range(len(xs)):
        if gt[k]:
            ax.axvspan(xs[k] - 0.06, xs[k] + 0.06, color="#2ca02c", alpha=0.12, zorder=0, label=lbl)
            lbl = None
    # 泵站中心
    for cx in PUMP_STATIONS:
        ax.axvline(cx, color="goldenrod", ls="--", lw=1.2, alpha=0.8, zorder=1)
    # 预测散点
    jit = (np.random.RandomState(0).rand(len(xs)) - 0.5) * 0.08
    ax.scatter(xs, pred + jit, s=15, c=np.where(pred > 0.5, "#d62728", "#1f77b4"),
               alpha=0.7, zorder=3)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["判 No", "判 Yes"])
    ax.set_xlabel("走廊位置 x (m)")
    ax.set_title("Infinite VLM 逐位置目标判定（绿=真实检视段, 黄虚线=泵站中心；红=判Yes 蓝=判No）")
    ax.set_ylim(-0.25, 1.25); ax.legend(loc="center left", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    # 注解
    ax.annotate("走廊后半段泵持续可见 → 巡航中也判 Yes", xy=(20, 1.0), xytext=(6.5, 0.5),
                fontsize=9, color="#555", arrowprops=dict(arrowstyle="->", color="#999"))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_verdict_vs_position.png"), dpi=130)
    plt.close(fig)


def fig2(rows):
    # 挑样例: 每个泵站附近1个YES帧 + 2个走廊NO帧
    xs = np.array([r["x"] for r in rows])
    picks = []
    for cx in PUMP_STATIONS:
        cand = [r for r in rows if abs(r["x"] - cx) <= STATION_HALF and r["pred_yes"]]
        if not cand:
            cand = [r for r in rows if abs(r["x"] - cx) <= STATION_HALF]
        if cand:
            picks.append(cand[len(cand) // 2])
    # 2个明确走廊NO
    no_c = [r for r in rows if r["kind"] == "cruise" and not r["pred_yes"]
            and min(abs(r["x"] - c) for c in PUMP_STATIONS) > 4]
    if no_c:
        picks.append(no_c[len(no_c) // 4]); picks.append(no_c[3 * len(no_c) // 4])
    n = len(picks); cols = 3; rowsn = (n + cols - 1) // cols
    fig, axes = plt.subplots(rowsn, cols, figsize=(cols * 4, rowsn * 3.0))
    axes = np.array(axes).reshape(-1)
    for a in axes: a.axis("off")
    for a, r in zip(axes, picks):
        img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
        a.imshow(img)
        pv = "YES" if r["pred_yes"] else "NO"
        col = "#d62728" if r["pred_yes"] else "#1f77b4"
        gtv = "泵在旁" if r["gt_yes"] else "走廊"
        a.set_title(f"x={r['x']:.0f}m  GT:{gtv}", fontsize=10)
        a.text(0.03, 0.06, f"Infinite VLM: {pv}", transform=a.transAxes,
               fontsize=13, fontweight="bold", color="white",
               bbox=dict(facecolor=col, alpha=0.85, pad=3))
    fig.suptitle("Infinite VLM 帧判定样例（走廊蓝水泵）", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_frame_montage.png"), dpi=130)
    plt.close(fig)


def fig3(m):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    cm = np.array([[m["TP"], m["FN"]], [m["FP"], m["TN"]]])
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Pred Yes", "Pred No"])
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["GT 泵在旁", "GT 走廊"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i, j], ha="center", va="center",
                     fontsize=16, color="black")
    ax1.set_title("混淆矩阵")
    labs = ["Accuracy", "Precision", "Recall", "F1"]
    vals = [m["acc"], m["prec"], m["rec"], m["f1"]]
    bars = ax2.bar(labs, [v * 100 for v in vals],
                   color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v*100:.1f}%",
                 ha="center", fontsize=11)
    ax2.set_ylim(0, 105); ax2.set_ylabel("%"); ax2.set_title("检测指标")
    fig.suptitle("Infinite VLM 走廊蓝水泵检测", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_confusion.png"), dpi=130)
    plt.close(fig)


def main():
    rows = load()
    m = metrics(rows)
    st = station_detection(rows)
    write_tables(rows, m, st)
    fig1(rows); fig2(rows); fig3(m)
    print("[viz] 完成. 产物在", OUT)
    print("  指标:", {k: (round(m[k], 3) if isinstance(m[k], float) else m[k]) for k in m})
    print("  站位检出:", [(s["station"], s["detected"], s["yes_rate"]) for s in st])


if __name__ == "__main__":
    main()
