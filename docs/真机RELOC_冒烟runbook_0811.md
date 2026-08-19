# 真机 RELOC 冒烟 runbook（0811，周四交付用）

## 架构

```
机器人(Orin)                          GPU服务器(10.0.0.20)
zed点云(xyzrgba npz)  ──HTTP POST──▶  reloc_server.py (常驻, GPU4)
believed位姿(x,y,yaw,真实米)           GS render-and-compare位姿优化
纠正位姿(真实米)       ◀──JSON────     (accepted标志 + loss)
```
机器人端只有 numpy+requests 依赖；重计算全在服务器。**调用期间机器人静止**（每次约 20~40s）。

## 启动

**服务器端**（已在跑，若被抢卡重启即可）：
```bash
cd /data/DongBaorong/BEVLOOK/robot_bev_sim
OBS_SCALE=0.70 CUDA_VISIBLE_DEVICES=4 ~/discoverse_venv/bin/python scripts/reloc_server.py --port 8321
curl http://127.0.0.1:8321/health   # {"ok":true,"frames":510,"obs_scale":0.7}
```

**机器人端**（robot_control_current ws 内）：
```bash
python scripts/reloc_client.py --server http://10.0.0.20:8321 \
    --x <believed_x> --y <believed_y> --yaw <believed_yaw度> --cam_h <zed离地高,米>
# 离线单文件测试: --npz <path.npz> 替代实时抓取
```

## 坐标约定（重要）

- 客户端**全程真实米制**，原点 = 110744 采集起始点（实验室原位），轴向 = 采集起始朝向
- 0.70 尺度换算在服务器内部完成，客户端不用管
- zed 设置必须与采集一致：**HD720 + NEURAL 深度**（intrinsics 才匹配）
- `--cam_h` 传 zed 实际离地高（真实米，卷尺量；默认1.2）
- 机器人 believed 位姿来源：开机在原位置零后，cmd_twist 积分 + IMU（现有方案）

## 验证协议（由易到难，每步记录返回的 accepted/loss）

**A. 重复性**（不动机器人）：同一物理位置连测 5 次，believed 每次手动加 ±0.1~0.3m 不同扰动
→ 返回位姿应聚类（散布应 <10cm）；`accepted=true`（loss1≤0.08）

**B. 收敛域**：believed 故意偏 0.1 / 0.2 / 0.3 / 0.5m + yaw 偏 10°/25°
→ 各档返回应收敛到同一物理位姿（±10cm 内）；0.5m 档可能 REJECT=正常保护

**C. 物理精度**（要卷尺）：机器人停在铁柜正前方，卷尺量 zed 到柜面距离 d_real
→ reloc 返回 (x,y)，与节点坐标算距离对比（铁柜节点世界坐标 (-0.576,-3.555)，**真实坐标=世界÷0.70** = (-0.82,-5.08)）
→ |reloc距离 - 卷尺距离| 即物理精度，目标 <10cm

## 已验证（0811 晚，服务器端 selftest：采集npz+COLMAP GT 自动扰动走 HTTP 全链）

n=8 帧：**水平中位 10.2cm / p90 11.9cm / yaw 中位 1.35° / 接受率 88% / 平均 43s**。
（同管线离线评测 24 帧：水平 5.6cm / yaw 1.08°，收敛域 ±0.5m/±25°）

## 注意

- ⚠ 实验室布局须与 110744 采集时一致（家具大搬动会让 loss 升高→REJECT）
- ⚠ loss1>0.08 → accepted=false = 系统主动拒收（低纹理/视角差），**别用拒收的输出**，挪半步重试
- 若全程 REJECT：先查 zed 是否 HD720（npz shape 应为 720×1280×4）
- 服务器日志在 `outputs/reloc_server_0811.log`，每次调用有 loss 记录
