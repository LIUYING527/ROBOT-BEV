#!/usr/bin/env bash
# Stage B: 模型推理评测(artifixer env, 只用 vendor/weights, 不碰 robot_orig_code).
# 产物 -> arch_e2e/runs/eval_nonreactive_<session>.{mp4,json}
set -e
SESSION="${1:-colmapjoint_all}"
GPU="${2:-2}"
MODE="${3:-cruise}"   # VLM 桩: cruise/turn/inspect/exit
ARCH="$(cd "$(dirname "$0")" && pwd)"
CUDA_VISIBLE_DEVICES=$GPU \
  /home/DongBaorong/micromamba/envs/artifixer/bin/python \
  "$ARCH/pipeline/stage_b_eval.py" --session "$SESSION" --mode "$MODE"
