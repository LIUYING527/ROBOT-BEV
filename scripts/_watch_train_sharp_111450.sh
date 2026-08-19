#!/bin/bash
# 看门狗: 等到某张GPU空闲>=NEED_MIB, 自动训最锐111450皮肤(~1M高斯,105731同规格)+align。
# 用法: bash scripts/_watch_train_sharp_111450.sh   (建议 nohup 后台)
set -u
cd "$(dirname "$0")/.."
NEED_MIB=${NEED_MIB:-42000}
POLL=${POLL:-120}
LOG=outputs/_train3dgs_111450_sharp.log
MARK=outputs/_sharp_111450.done
echo "[watch] $(date '+%H:%M') 等待GPU空闲>=${NEED_MIB}MiB (每${POLL}s轮询)..." | tee -a "$LOG"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
  | awk -F', ' -v need="$NEED_MIB" '{ free=$3-$2; if (free>=need && $4<40) { print $1, free; exit } }'
}

while true; do
  RES=$(pick_gpu)
  if [ -n "$RES" ]; then
    GPU=$(echo "$RES" | awk '{print $1}'); FREE=$(echo "$RES" | awk '{print $2}')
    echo "[watch] $(date '+%H:%M') GPU$GPU 空闲${FREE}MiB >= ${NEED_MIB} -> 启动锐化训练" | tee -a "$LOG"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ITERS=20000 SH_DEGREE=3 UPSAMPLE=1.0 DEPTH_SUP=1 DEPTH_W=0.4 ANTIALIAS=1 REG_SCALE=0.01 \
    APPEARANCE=1 MASK_OVEREXP=1 OVEREXP_MIN=225 \
    CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      ~/discoverse_venv/bin/python scripts/train_3dgs_vggt.py colmapjoint_111450 >> "$LOG" 2>&1
    RC=$?
    echo "[watch] $(date '+%H:%M') 训练退出码=$RC" | tee -a "$LOG"
    if [ $RC -ne 0 ]; then
      echo "[watch] 训练失败(可能OOM竞争), 60s后重试等卡" | tee -a "$LOG"; sleep 60; continue
    fi
    echo "[watch] $(date '+%H:%M') align中..." | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      ~/discoverse_venv/bin/python scripts/align_gs_world.py colmapjoint_111450 >> "$LOG" 2>&1
    echo "[watch] $(date '+%H:%M') 用锐皮肤重跑巡检demo..." | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
      ~/discoverse_venv/bin/python goalnav/pipeline/run_inspection.py \
      --session colmapjoint_111450 \
      --skin_ply "$(pwd)/outputs/gs_vggto_colmapjoint_111450_world.ply" \
      --geo_vlm --nav_ds outputs/nav_ds_111450 --clusters outputs/device_clusters_111450.npy \
      --target "a chain-link metal wire fence" --dev_side -1 \
      --unify_occ --occ_thr 12 --steps 500 >> "$LOG" 2>&1
    cp goalnav/runs/inspection_colmapjoint_111450.mp4 goalnav/runs/inspection_111450_sharp.mp4 2>/dev/null
    echo "SHARP_DONE $(date '+%F %H:%M') gpu=$GPU" > "$MARK"
    echo "[watch] $(date '+%H:%M') ✅ 锐化皮肤+align+demo完成 -> $MARK (视频 inspection_111450_sharp.mp4)" | tee -a "$LOG"
    break
  fi
  sleep "$POLL"
done
