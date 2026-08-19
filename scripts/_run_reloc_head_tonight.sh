#!/bin/bash
# 0812夜训: 精对准回归头全量数据+训练 (用户指示"长训放晚上跑")
cd /data/DongBaorong/BEVLOOK/robot_bev_sim
PY=/home/DongBaorong/discoverse_venv/bin/python
LOG=outputs/reloc_head_0812/train_$(date +%H%M).log
echo "start $(date)" > $LOG
CUDA_VISIBLE_DEVICES=1 $PY scripts/reloc_head_train.py gen >> $LOG 2>&1
CUDA_VISIBLE_DEVICES=1 $PY scripts/reloc_head_train.py train --epochs 200 >> $LOG 2>&1
echo "done $(date)" >> $LOG
