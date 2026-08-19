#!/bin/bash
# 把 ArtiFixer3D 蒸馏出的 3DGRUT ckpt 导成标准 .ply(高斯), 供后续对齐+DISCOVERSE渲。
set -u
ENVDIR=/home/DongBaorong/micromamba/envs/artifixer; PY=$ENVDIR/bin/python
export CUDA_HOME="$ENVDIR" PATH="$ENVDIR/bin:$PATH" TORCH_CUDA_ARCH_LIST="8.0"
export LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ENVDIR/lib:$ENVDIR/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=7
cd /data/DongBaorong/BEVLOOK/robot_bev_sim/third_party/3DGRUT-ArtiFixer-main
CKPT=/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/artifixer/corridor/artifixer3d/runs/corridor/corridor/ours_20000/ckpt_20000.pt
OUT=/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/artifixer3d_skin_colmapframe.ply
$PY - "$CKPT" "$OUT" <<'PY'
import sys, torch
from threedgrut.model.model import MixtureOfGaussians
ckpt_path, out = sys.argv[1], sys.argv[2]
ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
conf = ckpt["config"]
model = MixtureOfGaussians(conf)
model.init_from_checkpoint(ckpt, setup_optimizer=False)
print("num_gaussians:", int(model.num_gaussians))
model.export_ply(out)
print("[ok] exported ->", out)
PY
ls -la "$OUT" 2>/dev/null
