#!/bin/bash
set -e
ENVDIR=/home/DongBaorong/micromamba/envs/artifixer; PY=$ENVDIR/bin/python
export CUDA_HOME="$ENVDIR" PATH="$ENVDIR/bin:$PATH" HF_ENDPOINT=https://hf-mirror.com TORCH_CUDA_ARCH_LIST=8.0
export LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOGE_MODEL_PATH=/data/DongBaorong/moge-2-vitl-normal/model.pt
export CUDA_VISIBLE_DEVICES=6
R=/data/DongBaorong/BEVLOOK/robot_bev_sim
SCENE=$R/outputs/artifixer/corridor
CKPT=$SCENE/3dgrut_runs/corridor/corridor/ours_10000/ckpt_10000.pt
TRAJ=$R/outputs/artifixer/offpath_long80.json
cd $R/third_party/artifixer
echo "[1/2] render+scale $(date)"
$PY -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir $R/outputs/_colmap_joint_all_singlecam --output_root "$SCENE" \
  --selected_image_names_file $R/outputs/artifixer/train_images_95.txt \
  --trajectory_path "$TRAJ" --reconstruction_checkpoint "$CKPT" --phases render,scale --replace
echo "[2/2] inference $(date)"
$PY -m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt /data/DongBaorong/artifixer-checkpoints/artifixer-14b.pt \
  --save_dir $R/outputs/artifixer/corrected_long --split_path "$SCENE/split.json" --render_trajectory trajectory \
  --num_views 2 --local_attn_size 9 --sink_size 5
echo "[done] $(date)"; find $R/outputs/artifixer/corrected_long -name "*.mp4" 2>/dev/null
