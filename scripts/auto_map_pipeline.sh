#!/bin/bash
# auto_map_pipeline.sh — M1首探数据 → M2地图资产 无人值守流水线 (0901)
# 用法: bash scripts/auto_map_pipeline.sh <session> --cam_height <实测米> [--stride 2] [--untwist] [--crop] [--gpu 4]
#   例: bash scripts/auto_map_pipeline.sh lab0905 --cam_height 1.71 --untwist --gpu 4
# 输入:  data/<session>/images/zed/*.jpg + data/<session>/pointcloud/zed/*.npz (xyzrgba mm)
# 产出:  outputs/vggto_colmapjoint_am<session>/  gs_vggto_colmapjoint_am<session>(_world).ply
#        outputs/sim_world_colmapjoint_am<session>.npz  world_geometry_*.json  occ_bev_*.npy  nav_ds_am<session>/
# 断点续跑: 每阶段产物存在即跳过; 失败即停, 阶段号写入 outputs/automap_<session>_VERDICT.txt
# ⚠️ cam_height 是全局尺度锚, 必须卷尺实测(量错=继承0.70×尺度债), 脚本拒绝默认值
set -u -o pipefail
cd "$(dirname "$0")/.." || exit 1
DPY=~/discoverse_venv/bin/python

# ---------- 参数 ----------
S=""; CAM_H=""; STRIDE=2; UNTWIST=0; CROP=0; GPU=4
while [ $# -gt 0 ]; do case "$1" in
  --cam_height) CAM_H="$2"; shift 2;; --stride) STRIDE="$2"; shift 2;;
  --untwist) UNTWIST=1; shift;; --crop) CROP=1; shift;; --gpu) GPU="$2"; shift 2;;
  *) [ -z "$S" ] && S="$1" || { echo "unknown arg $1"; exit 2; }; shift;; esac; done
[ -z "$S" ] && { echo "usage: auto_map_pipeline.sh <session> --cam_height <m> [--stride N] [--untwist] [--crop] [--gpu N]"; exit 2; }
[ -z "$CAM_H" ] && { echo "FATAL: --cam_height 必填(实测相机离地高度, 米)"; exit 2; }

SESS="colmapjoint_am${S}"          # 下游统一session名
V=outputs/automap_${S}_VERDICT.txt
LOG=outputs/automap_${S}.log
mkdir -p outputs; : > "$V"
say(){ echo "[automap $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
gate(){ # gate <阶段号> <名字> <条件...>; 失败写VERDICT退出
  local n="$1" name="$2"; shift 2
  if "$@"; then say "GATE$n $name PASS"; else say "GATE$n $name FAIL"; echo "FAIL stage$n $name" >> "$V"; exit 1; fi
}

# ---------- S0 输入检查 ----------
gate 0 输入布局 test -d "data/$S/images/zed"
NJPG=$(ls "data/$S/images/zed/"*.jpg 2>/dev/null | wc -l)
NNPZ=$(ls "data/$S/pointcloud/zed/"*.npz 2>/dev/null | wc -l)
say "S0: $NJPG jpg / $NNPZ npz"
gate 0 帧数充足 test "$NJPG" -ge 200      # 510帧/18×12m是成功先例, 200是下限
echo "stage0 input ok ($NJPG jpg, $NNPZ npz), cam_height=$CAM_H" >> "$V"

# ---------- S1 COLMAP (单相机+顺序匹配, O(n)) ----------
if [ ! -d "outputs/_colmap_${S}_sc/sparse" ]; then
  say "S1 COLMAP _recon_single_cam_seq $S stride=$STRIDE"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 $DPY scripts/_recon_single_cam_seq.py "$S" "$STRIDE" >> "$LOG" 2>&1 \
    || { echo "FAIL stage1 colmap" >> "$V"; exit 1; }
else say "S1 skip (产物在)"; fi
# 注册率+模型数门
$DPY - "$S" "$NJPG" >> "$LOG" 2>&1 <<'EOF' || { echo "FAIL stage1 colmap gate" >> "$V"; exit 1; }
import sys, glob, pycolmap
s, njpg = sys.argv[1], int(sys.argv[2])
recs = glob.glob(f"outputs/_colmap_{s}_sc/sparse/*")
assert recs, "无sparse模型"
best = max(pycolmap.Reconstruction(d).num_reg_images() for d in recs)
rate = best / max(njpg, 1)
print(f"[gate1] 模型数={len(recs)} 最大注册={best}/{njpg} rate={rate:.2f}")
assert rate >= 0.75, f"注册率{rate:.2f}<0.75"   # 110744先例86%, 走廊937/940
EOF
gate 1 COLMAP注册率 test -f "outputs/_colmap_${S}_sc/sparse/0/images.bin" -o -n "$(ls outputs/_colmap_${S}_sc/sparse/*/images.bin 2>/dev/null | head -1)"

# ---------- S2 导出vggto训练输入 (软链适配: build_colmap_session只吃_colmap_joint<tag>) ----------
if [ ! -f "outputs/vggto_${SESS}/cameras.npz" ]; then
  ln -sfn "$(readlink -f outputs/_colmap_${S}_sc)" "outputs/_colmap_joint_am${S}"
  say "S2 build_colmap_session am${S} (-> _colmap_${S}_sc)"
  $DPY scripts/build_colmap_session.py "am${S}" >> "$LOG" 2>&1 \
    || { echo "FAIL stage2 export" >> "$V"; exit 1; }
else say "S2 skip"; fi
gate 2 导出 test -f "outputs/vggto_${SESS}/cameras.npz"

# ---------- S3 3DGS (深度监督+外观建模; GPU) ----------
if [ ! -f "outputs/gs_vggto_${SESS}.ply" ]; then
  say "S3 train_3dgs_vggt $SESS (GPU$GPU, 约1-2h)"
  CUDA_VISIBLE_DEVICES=$GPU DEPTH_SUP=1 DEPTH_W=0.4 SH_DEGREE=3 APPEARANCE=1 \
    $DPY scripts/train_3dgs_vggt.py "$SESS" >> "$LOG" 2>&1 \
    || { echo "FAIL stage3 3dgs" >> "$V"; exit 1; }
else say "S3 skip"; fi
gate 3 3DGS test -f "outputs/gs_vggto_${SESS}.ply"

# ---------- S4 世界对齐 (cam_height=尺度锚; 可选--crop/--untwist) ----------
if [ ! -f "outputs/sim_world_${SESS}.npz" ]; then
  EXTRA=""; [ $UNTWIST = 1 ] && EXTRA="$EXTRA --untwist"; [ $CROP = 1 ] && EXTRA="$EXTRA --crop"
  say "S4 align_gs_world $SESS --cam_height $CAM_H$EXTRA"
  $DPY scripts/align_gs_world.py "$SESS" --cam_height "$CAM_H" $EXTRA >> "$LOG" 2>&1 \
    || { echo "FAIL stage4 align" >> "$V"; exit 1; }
else say "S4 skip"; fi
gate 4 对齐 test -f "outputs/sim_world_${SESS}.npz"

# ---------- S5 几何拟合 + 占据栅格 + nav_ds ----------
if [ ! -f "outputs/occ_bev_${SESS}.npy" ]; then
  say "S5 fit_world_geometry + occupancy_bev"
  PYTHONPATH=third_party/gs_playground/.venv/lib/python3.10/site-packages \
    $DPY scripts/fit_world_geometry.py "$SESS" >> "$LOG" 2>&1 \
    && $DPY scripts/occupancy_bev.py "$SESS" >> "$LOG" 2>&1 \
    || { echo "FAIL stage5 geometry/occ" >> "$V"; exit 1; }
else say "S5 skip"; fi
if [ ! -f "outputs/nav_ds_am${S}/meta.npz" ]; then
  say "S5b build_nav_ds am${S}"
  $DPY scripts/build_nav_ds.py "am${S}" >> "$LOG" 2>&1 || say "WARN nav_ds失败(不阻塞)"
fi

# ---------- S6 半自动环节提示 ----------
cat >> "$V" <<EOF
AUTO STAGES DONE ($SESS)
产物: outputs/gs_vggto_${SESS}{,_world}.ply / sim_world_${SESS}.npz / occ_bev_${SESS}.npy
待人工(半自动缺口, 见docs/真机部署方案v2_渐进式部署_0901.md §4.3):
  1. 目检 outputs/occ_bev_${SESS}.png + world_geometry_${SESS}_check.png
  2. targets生成 (lab_locate_clusters.py雏形 / 场景图cg三步, session名需参数化)
  3. 若roll漂移>3°需仿写apply_untwist_110744泛化版
  4. 拷Orin: occ_bev npz + targets json + (reloc_server换--session $SESS, OBS_SCALE=1.0)
EOF
say "ALL AUTO STAGES DONE → $V"
