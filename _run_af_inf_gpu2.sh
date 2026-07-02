#!/bin/bash
ENVDIR=/home/DongBaorong/micromamba/envs/artifixer
PY=$ENVDIR/bin/python
export CUDA_HOME="$ENVDIR" PATH="$ENVDIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com TORCH_CUDA_ARCH_LIST=8.0
export LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCENE=/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/artifixer/corridor
CKPT=/data/DongBaorong/artifixer-checkpoints/artifixer-14b.pt
SAVE=/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/artifixer/corrected
cd /data/DongBaorong/BEVLOOK/robot_bev_sim/third_party/artifixer
# local_attn_size 21→9 : KV cache 由注意力窗口大小决定,减小窗口砍显存(单卡80GB放得下)
exec $PY -m model_eval.run_inference \
  --evalset reconstructed_colmap --checkpoint_pt "$CKPT" \
  --save_dir "$SAVE" --split_path "$SCENE/split.json" --render_trajectory trajectory \
  --local_attn_size 9 --sink_size 5
