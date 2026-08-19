#!/usr/bin/env bash
# 下 VLM 慢脑到 arch_e2e/vlm/ —— 走魔搭 ModelScope(阿里官方, 国内直连, 绕开 HF 防火墙)。
# 用法: ./download_vlm.sh [型号]   默认 Qwen/Qwen3.5-4B(原生多模态, transformers5.5.0支持)
set -e
MODEL="${1:-Qwen/Qwen3.5-4B}"
PY=/home/DongBaorong/micromamba/envs/artifixer/bin/python
ARCH="$(cd "$(dirname "$0")" && pwd)"
LOCAL="$ARCH/vlm/$(basename "$MODEL")"

$PY -c "import modelscope" 2>/dev/null || \
  /home/DongBaorong/micromamba/envs/artifixer/bin/pip install -q modelscope

echo "[下载] $MODEL -> $LOCAL"
$PY - "$MODEL" "$LOCAL" <<'PY'
import sys
from modelscope import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
echo "[完成] $LOCAL"
echo "接着跑: CUDA_VISIBLE_DEVICES=2 $PY $ARCH/pipeline/stage_vlm_slowbrain.py --session colmapjoint_all"
