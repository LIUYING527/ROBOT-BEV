"""场景图 RAG 效果可视化 (0729)。

图1: 185节点3D场景图(实例色点云) + 三目标RAG top1命中(大标记) + 几何真值(星)
     左=斜视3D 右=俯视BEV
图2: 三查询的top1节点外观卡片(建图时存的最佳视角crop) + tag + 距真值

输出: outputs/cg_lab110744s2/cg_scenegraph_rag_viz_0729.png
"""
import gzip
import json
import os
import pickle

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CG = os.path.join(ROOT, "outputs/cg_lab110744s2")
MAP = os.path.join(CG, "lab1/sg_cache/map/scene_map_cfslam_pruned.pkl.gz")
DEBUG_CROPS = os.path.join(CG, "lab1/sg_cache/cfslam_captions_llava_debug")
GT_JSON = os.path.join(ROOT, "outputs/lab_mission_targets_zed.json")
SMOKE = os.path.join(CG, "cg_rag_smoke_0729.json")
OUT = os.path.join(CG, "cg_scenegraph_rag_viz_0729.png")

QUERIES = {"microscope": "microscope", "iron_cabinet": "iron cabinet", "pipette_table": "white table"}
HIT_COLOR = {"microscope": "#e6194b", "iron_cabinet": "#f58231", "pipette_table": "#4363d8"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.image as mpimg

    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.style.use("dark_background")

    with gzip.open(MAP, "rb") as f:
        scene_map = pickle.load(f)
    gt = json.load(open(GT_JSON))
    smoke = json.load(open(SMOKE))

    # 每节点子采样点(聚焦任务区窗口内)
    gt_all = np.array([v for k, v in gt.items() if k != "origin"])
    ctr = gt_all.mean(0)
    xlim = (ctr[0] - 2.2, ctr[0] + 2.2)
    ylim = (ctr[1] - 2.6, ctr[1] + 2.6)
    node_pts = []
    for o in scene_map:
        p = np.asarray(o["pcd_np"])
        if len(p) > 300:
            p = p[np.random.default_rng(0).choice(len(p), 300, replace=False)]
        node_pts.append(p)

    def in_win(p):
        return (p[:, 0] >= xlim[0]) & (p[:, 0] <= xlim[1]) & (p[:, 1] >= ylim[0]) & (p[:, 1] <= ylim[1])

    def tag_of(o):
        return str(o.get("caption_dict", {}).get("response", {}).get("object_tag", "?")).lower()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 10))
    for ax in (axL, axR):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    # ---- 左: 灰底 + 三查询top1高亮(点+bbox线框) + GT星 ----
    for p in node_pts:
        w = in_win(p)
        axL.scatter(p[w, 0], p[w, 1], c="#666666", s=1.5, alpha=0.5)
    for name, q in QUERIES.items():
        hit = smoke[name][0]
        idx = hit["node"]
        col = HIT_COLOR[name]
        p = np.asarray(scene_map[idx]["pcd_np"])
        axL.scatter(p[:, 0], p[:, 1], c=col, s=6, alpha=0.95)
        b = np.asarray(scene_map[idx]["bbox_np"])
        x0, x1, y0, y1 = b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()
        axL.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=col, lw=2)
        g = np.array(gt[name])
        axL.scatter(g[0], g[1], marker="*", s=1500, c=col, edgecolors="white", linewidths=1.5, zorder=10)
        axL.annotate(f"{name}\nnode{idx} \"{hit['tag']}\"  err={hit['dist_to_gt']:.2f}m",
                     (x1, y1), textcoords="offset points", xytext=(6, 6),
                     color=col, fontsize=13, weight="bold")
    axL.set_title("RAG retrieval: query -> top-1 node (gray = other nodes, star = GT)", fontsize=13)

    # ---- 右: 按 object_tag 语义类别着色 ----
    CAT = [
        (("microscope",), "#e6194b", "microscope"),
        (("table", "desk", "workbench"), "#4363d8", "table/desk"),
        (("computer", "laptop", "monitor"), "#42d4f4", "computer"),
        (("chair", "stool"), "#3cb44b", "chair/stool"),
        (("shelf", "bookshelf", "bookcase", "cabinet", "closet"), "#ffe119", "shelf/cabinet"),
        (("window",), "#f58231", "window"),
    ]
    import matplotlib.patches as mpatches
    leg = {}
    n_table = 0
    for i, o in enumerate(scene_map):
        tag = tag_of(o)
        col, label = "#555555", "other"
        for keys, c, lb in CAT:
            if any(k in tag for k in keys):
                col, label = c, lb
                break
        if label == "table/desk":
            n_table += 1
        p = node_pts[i]
        w = in_win(p)
        axR.scatter(p[w, 0], p[w, 1], c=col, s=2.5, alpha=0.85)
        leg[label] = col
    axR.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in leg.items()],
               loc="upper right", fontsize=11, framealpha=0.6)
    axR.set_title(f'semantic map by object_tag  ({n_table} table/desk nodes -> "white table" 歧义)', fontsize=13)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print("saved:", OUT)

    # 图2: 三个top1节点的外观卡片
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, q) in zip(axes, QUERIES.items()):
        hit = smoke[name][0]
        crop = os.path.join(DEBUG_CROPS, f"{hit['node']}.png")
        if os.path.exists(crop):
            ax.imshow(mpimg.imread(crop))
        _sim = f"sim={hit['sim']:.3f}  " if hit.get('sim') is not None else ""
        ax.set_title(f"query \"{q}\" → node{hit['node']} \"{hit['tag']}\"\n"
                     f"{_sim}err_to_GT={hit['dist_to_gt']:.2f}m",
                     fontsize=12, color=HIT_COLOR[name], weight="bold")
        ax.axis("off")
    fig2.suptitle("RAG top-1 nodes: best-view appearance (captured at mapping time)", fontsize=14)

    fig.tight_layout()
    fig2.tight_layout()
    out1 = OUT
    out2 = OUT.replace(".png", "_crops.png")
    fig.savefig(out1, dpi=130)
    fig2.savefig(out2, dpi=130)
    print("saved:", out1)
    print("saved:", out2)


if __name__ == "__main__":
    main()
