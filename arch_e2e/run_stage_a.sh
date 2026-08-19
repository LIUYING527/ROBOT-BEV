#!/usr/bin/env bash
# Stage A: 渲染 env(discoverse_venv). 产物 -> arch_e2e/runs/obs_seq_<session>/
set -e
SESSION="${1:-colmapjoint_all}"
N="${2:-60}"
GPU="${3:-2}"
ARCH="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$ARCH")"
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU \
  PYTHONPATH="$REPO/third_party/discoverse" \
  ~/discoverse_venv/bin/python "$ARCH/pipeline/stage_a_gen_obs.py" --session "$SESSION" --n "$N"
