#!/usr/bin/env bash
# 在 ros2 env 内:启动 octomap_server + 喂点云 → 存标准 .pgm 占据图
# 用法: micromamba run -n ros2 bash scripts/octomap_run.sh <session>
set -e
SESS=${1:-111450}
cd /data/DongBaorong/BEVLOOK/robot_bev_sim
mkdir -p outputs

echo "[run] 启动 octomap_server_node"
ros2 run octomap_server octomap_server_node --ros-args \
  -p frame_id:=map -p resolution:=0.2 \
  -p sensor_model.max_range:=15.0 \
  -p pointcloud_min_z:=0.4 -p pointcloud_max_z:=8.0 \
  -p occupancy_min_z:=1.5 -p occupancy_max_z:=6.0 \
  -p filter_ground:=false -p base_frame_id:=map \
  > outputs/octomap_${SESS}_server.log 2>&1 &
OCTO=$!
sleep 4

echo "[run] 启动点云馈送"
python scripts/ros2_cloud_feeder.py ${SESS} > outputs/octomap_${SESS}_feeder.log 2>&1 &
FEED=$!

echo "[run] 累积建图 ~35s"
sleep 35

echo "[run] 当前 topics:"; ros2 topic list 2>/dev/null | grep -iE "map|cloud|octo" || true
echo "[run] 保存 /projected_map → outputs/octomap_${SESS}"
ros2 run nav2_map_server map_saver_cli -t /projected_map -f outputs/octomap_${SESS} \
  --ros-args -p save_map_timeout:=20.0 > outputs/octomap_${SESS}_saver.log 2>&1 || echo "[run] saver 失败,看日志"

kill $FEED $OCTO 2>/dev/null || true
sleep 1
echo "[run] 完成"; ls -la outputs/octomap_${SESS}.* 2>/dev/null || echo "  未生成pgm"
