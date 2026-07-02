#!/bin/bash
# 等某张 GPU 空出 >=NEED_FREE MiB 就跑 ArtiFixer 14B 推理(60帧 off-path)。
# VAE 已开 tiling/slicing,峰值显存大降,故阈值可降到 ~55GB。失败自动重试。
set -u
ENVDIR=/home/DongBaorong/micromamba/envs/artifixer
PY=$ENVDIR/bin/python
export CUDA_HOME="$ENVDIR" PATH="$ENVDIR/bin:$PATH" HF_ENDPOINT=https://hf-mirror.com TORCH_CUDA_ARCH_LIST="8.0"
export LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCENE=/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/artifixer/corridor
CKPT=/data/DongBaorong/artifixer-checkpoints/artifixer-14b.pt
SAVE=/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/artifixer/corrected
NEED_FREE=60000   # MiB (num_views2清prope+attn9缩KV cache后峰值~55GB,留余量防加载期被抢)
LOCAL_ATTN=9      # 默认21→9,KV cache砍约57%(清 _initialize_kv_cache 那处OOM)
SINK=5
NUM_VIEWS=2       # 0701新增: 默认6→2,直接缩 prope 的 key_neighbor fp32 张量(清 transformer.py:898 那处OOM)

run_once() {
  # 找空闲显存最大的卡
  line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1)
  idx=$(echo "$line" | cut -d, -f1 | tr -d ' ')
  free=$(echo "$line" | cut -d, -f2 | tr -d ' ')
  if [ "$free" -lt "$NEED_FREE" ]; then echo "[wait] $(date +%H:%M) 最空闲 GPU$idx=${free}MiB < ${NEED_FREE},继续等"; return 9; fi
  export CUDA_VISIBLE_DEVICES=$idx
  echo "[run] $(date) GPU$idx 空闲 ${free}MiB,开跑"
  cd /data/DongBaorong/BEVLOOK/robot_bev_sim/third_party/artifixer
  $PY -m model_eval.run_inference \
    --evalset reconstructed_colmap --checkpoint_pt "$CKPT" \
    --save_dir "$SAVE" --split_path "$SCENE/split.json" --render_trajectory trajectory \
    --num_views $NUM_VIEWS --local_attn_size $LOCAL_ATTN --sink_size $SINK
  return $?
}

echo "[wait] $(date) 等待 >=${NEED_FREE}MiB GPU,失败自动重试 ..."
for attempt in $(seq 1 600); do   # 每分钟一次,最多 ~10h
  run_once && { echo "[done] $(date) 推理成功"; break; }
  rc=$?
  if [ "$rc" = "9" ]; then sleep 60; else echo "[retry] $(date) 推理 EXIT=$rc(多半抢卡/OOM),60s 后重试"; sleep 60; fi
done
echo "=== 输出产物 ==="; find "$SAVE" -type f \( -name "*.mp4" -o -name "*.png" \) 2>/dev/null | head -30
