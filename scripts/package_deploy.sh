#!/bin/bash
# package_deploy.sh — 打包真机部署需要的代码+资产到 deploy_pkg/（保持 robot_bev_sim 相对结构，Orin 解压即用）
# 用法: bash scripts/package_deploy.sh [--tar]
#   产物: robot_bev_sim/deploy_pkg/（文件夹）+ 可选 deploy_pkg.tar.gz
# ⚠ ply 是 symlink，用 cp -L 跟真身；依赖闭包含 vendor(navsim+nuplan+robot_data_prep)+weights。
set -u -e
cd "$(dirname "$0")/.."          # 到 robot_bev_sim
D=deploy_pkg
rm -rf "$D"; mkdir -p "$D"

echo "[pkg] 收集代码 ..."
mkdir -p "$D/scripts" "$D/arch_e2e/pipeline" "$D/goalnav/pipeline" "$D/goalnav/runs" "$D/outputs" "$D/docs"

# ---- 代码（保持相对结构，代码里 ROOT 用 dirname 计算依赖此结构） ----
cp scripts/run_lab_mission_real.py scripts/reloc_client.py \
   scripts/reloc_server.py scripts/vlm_server_local.py scripts/auto_map_pipeline.sh "$D/scripts/"
cp arch_e2e/pipeline/stage_b_eval.py arch_e2e/pipeline/_paths.py arch_e2e/pipeline/sim_bev.py "$D/arch_e2e/pipeline/"
cp -r arch_e2e/vendor arch_e2e/weights "$D/arch_e2e/"          # vendor: navsim+nuplan+robot_data_prep; weights: best-val-loss.ckpt+resnet34
cp goalnav/pipeline/train_goalnav.py goalnav/pipeline/train_goalnav_turn.py "$D/goalnav/pipeline/"
cp goalnav/runs/goalnav_gate_lab.pt "$D/goalnav/runs/"

# ---- 资产（ply 跟 symlink 真身，文件名保持，代码路径不变） ----
cp -L outputs/gs_vggto_colmapjoint_110744s2_world.ply "$D/outputs/"
cp outputs/sim_world_colmapjoint_110744s2.npz \
   outputs/lab_mission_targets_zed_front_perp_lvl.json \
   outputs/anchors_car20_plus4_robot.npy "$D/outputs/"

# ---- 文档 ----
cp DEPLOY_README.md docs/真机部署操作手册_0904.md "$D/"

# ---- manifest（清单 + 大小） ----
{
  echo "真机部署包 manifest — $(date '+%F %T')"
  echo "=================================================="
  echo "Orin 端(真机跑): scripts/run_lab_mission_real.py, scripts/reloc_client.py, goalnav/runs/goalnav_gate_lab.pt"
  echo "  资产: outputs/*.ply(世界点云), sim_world_*.npz(采集轨迹), *_targets_*.json, anchors_*.npy"
  echo "  依赖: arch_e2e/{pipeline,vendor,weights}, goalnav/pipeline"
  echo "服务器端(10.0.0.20 跑): scripts/reloc_server.py, scripts/vlm_server_local.py, scripts/auto_map_pipeline.sh"
  echo "=================================================="
  du -sh "$D"/scripts "$D"/arch_e2e "$D"/goalnav "$D"/outputs "$D"/docs 2>/dev/null
} > "$D/MANIFEST.txt"

echo "[pkg] 完成: $D"
du -sh "$D"

if [ "${1:-}" = "--tar" ]; then
  echo "[pkg] 打 tar（约 1.1GB，可能较慢）..."
  tar -czf deploy_pkg.tar.gz -C . deploy_pkg
  echo "[pkg] tar 完成: deploy_pkg.tar.gz"
  du -sh deploy_pkg.tar.gz
fi
