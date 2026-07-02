#!/bin/bash
ENVDIR=/home/DongBaorong/micromamba/envs/artifixer
PY=$ENVDIR/bin/python
export CUDA_HOME="$ENVDIR" PATH="$ENVDIR/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2 HF_ENDPOINT=https://hf-mirror.com TORCH_CUDA_ARCH_LIST=8.0
export LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export MOGE_MODEL_PATH=/data/DongBaorong/moge-2-vitl-normal/model.pt
SCENE=/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/artifixer/corridor
CKPT=$SCENE/3dgrut_runs/corridor/corridor/ours_10000/ckpt_10000.pt
cd /data/DongBaorong/BEVLOOK/robot_bev_sim/third_party/artifixer
exec $PY -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir /data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/_colmap_joint_all_singlecam \
  --output_root "$SCENE" \
  --selected_image_names_file /data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/artifixer/train_images_95.txt \
  --trajectory_path /data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/artifixer/offpath_lateral035.json \
  --reconstruction_checkpoint "$CKPT" --phases render,scale --replace
