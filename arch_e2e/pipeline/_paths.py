"""统一路径 + 依赖注入 —— 让 arch_e2e 自成一体, 不再 sys.path 到学长 robot_orig_code。

目录约定:
  arch_e2e/
    pipeline/   本文件所在, 三个流水线脚本
    vendor/     从学长处复制来的解耦依赖 (navsim / robot_data_prep / nuplan-shim)
    weights/    复制来的模型权重 (best-val-loss.ckpt / kmeans / resnet34)
    runs/       本架构自己的产物 (视频/指标/中间观测)
  repo/         上一级 robot_bev_sim, 仅 Stage A 渲染器与 3DGS 资产复用(我们自己的, 非学长)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))   # arch_e2e/pipeline
ARCH = os.path.dirname(HERE)                          # arch_e2e
REPO = os.path.dirname(ARCH)                          # robot_bev_sim
VENDOR = os.path.join(ARCH, "vendor")
WEIGHTS = os.path.join(ARCH, "weights")
RUNS = os.path.join(ARCH, "runs")


def inject_vendor():
    """把 vendor(navsim+robot_data_prep+nuplan-shim) 放到 import 路径最前。
    只依赖复制来的 vendor, 绝不指向 robot_orig_code。"""
    for p in (VENDOR, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)
