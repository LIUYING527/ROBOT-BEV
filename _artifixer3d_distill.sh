#!/bin/bash
# ArtiFixer3D 蒸馏: 把已修好的 2D 锐利帧(corrected_full) 蒸进 3DGRUT 3D 高斯 -> 仿真可任意视角渲。
# 先短跑(STEPS 可调)验证全管线, 通了再加长。卡7专属。
set -u
ENVDIR=/home/DongBaorong/micromamba/envs/artifixer; PY=$ENVDIR/bin/python
export CUDA_HOME="$ENVDIR" PATH="$ENVDIR/bin:$PATH" HF_ENDPOINT=https://hf-mirror.com TORCH_CUDA_ARCH_LIST="8.0"
export LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export MOGE_MODEL_PATH=/data/DongBaorong/moge-2-vitl-normal/model.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7

R=/data/DongBaorong/BEVLOOK/robot_bev_sim
S=$R/outputs/artifixer/corridor
PRED=$R/outputs/artifixer/corrected_full/artifixer-14b/distilled_views_reconstructed_colmap_2_evenly_spaced_sink5_trajectory/corridor/frames/batch_0000/pred
BASE=$S/3dgrut_runs/corridor/corridor/ours_10000/ckpt_10000.pt
STEPS=${1:-20000}   # ⚠️必须 > base(10000): 否则3DGRUT认为已超目标不训, 写不出ours_<steps>ckpt报错

cd $R/third_party/artifixer
echo "[a3d] $(date) 蒸馏开始 steps=$STEPS pred=$PRED"
$PY -m data_processing.run_artifixer3d \
  --scene_root "$S" \
  --artifixer_frames_dir "$PRED" \
  --split_path "$S/split.json" \
  --base_checkpoint "$BASE" \
  --artifixer3d_steps "$STEPS" \
  --phases distill,render \
  --output_root "$S/artifixer3d"
echo "[a3d] $(date) EXIT=$?"
echo "=== 产物(ckpt/ply/render) ==="
find "$S/artifixer3d" -type f \( -name "*.pt" -o -name "*.ply" -o -name "*.mp4" -o -name "*.png" \) 2>/dev/null | head -20
