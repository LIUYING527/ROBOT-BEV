# 进展日志（PROGRESS LOG）

> 本文件每次工作后追加一条，倒序排列（最新在最上）。
> 格式固定：**做了什么 / 产出 / 结论 / 下一步 / 卡点**。
> 全过程叙事见 [`RECONSTRUCTION_JOURNEY.md`]，本文件只记“每次”的增量。

## 2026-09-01(午) —— 真机部署v2三层渐进式落盘 + dense重训自动接续开跑

- **做了什么**：①BEV precompute 全收(8118/8118)后 chain_train.sh 自动启训 dense 重训(GPU4, 333103样本/61场景, 12ep)；②用户定部署新方向"新场景不人工采集直接跑"→查证 DualVLN/NaVILA/VLFM/ViNT 部署形态后定**三层渐进式**：M0无图VLN即跑→M1自主首探+自动建图→热切换M2完整模式；③盘点重建流水线全部现成脚本后落盘自动化三件套。
- **产出**：`docs/真机部署方案v2_渐进式部署_0901.md`(总方案)；`scripts/auto_map_pipeline.sh`(COLMAP→3DGS→align→占据→nav_ds无人值守, 断点续跑+VERDICT+注册率门, 语法/门禁已测)；`scripts/frontier_explore.py`(在线占据+cv2 frontier+打分骨架, 冒烟过)。
- **结论**：**DOCK对接是目标相对的(zed深度+局部几何+正脸)全程不查全局图→无图也能cm级贴脸**；cm级绝对精度必须锚度量参考=物理约束；无图派(DualVLN/NaVILA)即插即用只做米级，我们的卖点=精度随建图渐进升级。build_colmap_session只吃`_colmap_joint<tag>`→流水线用软链trick适配不改源码。
- **下一步**：训完→oracle探针(转弯应恢复)→smoke10→300ep；部署侧=P-1 frontier接test_env仿真闭环(覆盖率>85%)；缺口=targets自动生成/场景图泛化/untwist泛化。
- **卡点**：M0/M2双ckpt特征构造等R2R快脑定型后一次做对；机器load高(149/80核)。

## 2026-08-29(午) —— 断连恢复三件套 + T5 e2e训完+部署+smoke10(SR10/OSR40)+300ep开跑

- **做了什么**: ①断连杀光进程后全恢复: Qwen(GPU3)+T1 b78评测续跑(GPU2, resume生效)+T5训练重启。②修今早 transfuser_model_v2.py 手术不完整(forward加text_feature但forward_train/forward_test签名没跟)。③T5训完: **goalnav_r2r_e2e.pt, best val 0.5133**(ep059; goal槽置0条件全走text, 比r2rft的0.364高=任务更难属预期)。④写e2e部署(此前只有训练): our_agent.act_r2r_e2e(BERT[CLS]每episode编一次+status全零[4]vx[6]gamma口径与train逐位一致+stop头每步0.40/2票+卡死检测同v5), our_server --e2e/model_settings.e2e透传。⑤smoke10(同seed0集)跑完→300ep全量开跑(GPU2, 无VLM依赖~90s/ep)。
- **产出**: `goalnav/runs/goalnav_r2r_e2e.pt`; our_agent e2e模式+our_server --e2e; `r2r_eval/results_our_e2e_smoke10.jsonl`; e2e服务器GPU3:8024。
- **结论**: smoke10 **e2e SR10/OSR40/SPL0.10/NE7.06 vs r2rft(v5慢脑) SR20/OSR60/SPL0.13/NE6.86 vs v5 zero-shot SR10/SPL0.07/NE7.25**。逐ep分解: 指令条件导航本身work(ep1569 31步真成功; **4条进3m圈=ep1297@0.04/ep915@2.26/ep1596@1.60都没停**, stop修好SR可到40)——但**stop头每步评裸奔暴露双向误诊**: 5条假停(ep799第5步8.6m外停/ep1606 11步, v5里同ep曾57步成功因头被地标上下文门控) + 3条近目标不停(0.04m反而OOD)。e2e无VLM=免疫robo2vlm宕机+快3倍, 是干净对照。
- **下一步**: 300ep出诚实对照数字(vs r2rft 300 SR5.3-5.9); stop通路改法候选=步数门/轨迹幅度当e2e原生stop信号/头补"贴脸目标"训练帧; T1 b78评测明早收。
- **卡点**: GPU只剩2+3(0/1/4/5=NingZijian, 6/7=WuKefei, WangBizi又挤2/3共120GB); robo2vlm仍宕。


## 2026-08-29(午后) —— e2e 300ep落盘(SR3.3, 82%假停实锤) + stop头阈值/票数调参证伪 + 轨迹幅度判停

- **做了什么**: ①e2e 300ep(stop头每步评0.40/2票)全量收: **SR3.3/OSR10.7/SPL2.6/NE9.75/步中位66**。②失败分解: **假停247/300=82%(163条<100步) / 进3m没停22条(潜在SR+7.3)**——stop头每步评估把逐帧假阳性复利放大(v5里被地标上下文门控只评几十次, e2e评500次)。③参数化stop_thr/stop_votes→smoke10测3票: **SR10→0更差**——ep799第6步(8.6m外)连3票照假停(头分数时间平滑=连发不独立), ep1569真到点(0.36m)反而凑不满3票漏停。阈值/票数调不出工作点=**头的置信区分能力在特定远景视角有不可约混淆区**。④改**轨迹幅度判停**(stop_mode=traj): oracle轨迹终点监督→近目标预测位移自然收缩=e2e原生到达信号, 末waypoint位移<0.5m连续2票→停, 判停移到轨迹计算之后。smoke10跑中。
- **产出**: `results_our_e2e_300.jsonl`(300ep); our_agent/our_server stop_thr/stop_votes/stop_mode/stop_eps参数化; `results_our_e2e_v3_smoke10.jsonl`(3票对照)。
- **结论**: ①e2e基线(头判停) SR3.3 vs r2rft 5.3-5.9 vs v5+stop头 5.4——但**这个数字量的是stop头假射率不是策略导航力**(OSR10.7 vs r2rft 15.9-17才是策略差距, 且82% ep被提前掐死步数中位66 vs r2rft 234)。②**stop头在e2e每步评估下无工作点**(2票假停83%/3票真停漏光, 时间平滑使连续票不独立=投票机制失效)——这是"同样的头在v5门控下好用"的反面实证, 判停信号必须与任务进度上下文耦合。③轨迹幅度是e2e范式判停的正解方向(训练信号原生, 零额外模块)。
- **下一步**: traj-stop smoke→若假停死光且真停存活→300ep重跑定版; 若eps=0.5工作点不行→扫eps或回到"traj+头AND门"。
- **卡点**: 同前(GPU 2+3; robo2vlm宕)。


## 2026-08-30(凌晨) —— LH-VLN官方test T1全量收: zero-shot ISR3.55/CSR2.81 全面超DualVLN

- **做了什么**: T1 b78 全量跑完(403ep/15h, Qwen单后端)→按官方 test 口径(batch_7+8, n=397, 排4崩溃+2测地inf, DualVLN同规则同集)官方 NavigationMetrics 复算; DualVLN 444 全量里切出 b7+8 子集同口径对照。
- **产出**: `docs/评测_LH-VLN官方test_T1全量_0830.md`; `our_eval/results_t1_b78.jsonl`。
- **结论**: **官方 test 上 zero-shot 全面超 VLN-CE 训练过的 DualVLN: SR0.25:0.00 / ISR3.55:2.62(+35%) / CSR2.81:1.34(翻倍) / CGT1.42:1.15 / TAR22.15:21.98; NE 11.75:10.87 略逊**(锁错实例)。vs FantasyVLN(训练过) ISR 11.01 仍差距大=诚实定位。这是**首个官方 test 单一口径全量**(0827 的 n=437 是 6/7/8 合并), 论文报数以此为准。T1 接地头(head 优先+Qwen 仅 OOV 兜底)全链 15h 无故障。
- **下一步**: 用户定 T5 stop 路 A/B/C; robo2vlm 恢复后 R2R 重跑; 周报素材(官方 test 表+e2e stop 三变体证伪)。
- **卡点**: 无新。


## 2026-08-29(凌晨) —— T1接地头端到端打平(诚实负结果)+GDINO对照自训头全胜+R2R 300ep补齐+转向T5

- **做了什么**: ①T1接地头集成三修(vantage探索/近距absent信头/子任务清缓存)后 val(batch_6) 端到端对照: **T1v3 ISR3.7/NE12.90/153步 vs P0(robo2vlm) ISR3.7/NE11.76/42步=打平**。②GroundingDINO-base zero-shot 同集(300 val帧)对照: P0.16/R0.60/cerr0.230/IoU0.142 vs **自训头 P0.25/R0.60/cerr0.118/IoU0.359** vs Qwen P0.50/R0.02/cerr0.50——自训头全面最优。③R2R 300ep 补齐: habitat reset 迭代器对齐坑(第一轮 topup 白跑 42 条重复)→显式 `env.episodes=[ep]` 修复→**终版 n=300 SR5.3/OSR17.0/SPL3.1/NE9.64**(Qwen 顶班 61 条 SR0 拉低, robo2vlm 段 SR5.9)。④Qwen 当 R2R 慢脑实测显著弱于 robo2vlm(SR 0/61 vs 14/239)。
- **产出**: our_agent vantage/门控/清缓存三修; `eval_grounding_dino.py`; run_r2r 显式对齐 resume; R2R 文档终版数字; `训练强化计划_0828.md` 增补(负结果+GDINO 表)。
- **结论**: ①头的离线优势(R0.60/cerr0.118)没转化成端到端收益——**接地"够用"后瓶颈=房屋级探索策略**(77%失败=目标在其他房间, 局部 vantage 不是过厅找门的搜索; head/VLM 类级检测锁错实例依旧); ②域内监督>通用开词表检测(GDINO); ③**T5(指令/子任务条件 IL)从消融升为主线**: oracle 轨迹教"怎么搜", R2R 快脑微调 OSR 翻倍已证此路, BERT/cond 注入点(_cond_offset)全备好。
- **下一步**: T5 指令条件 IL 训练(R2R 4000ep 数据已在盘, BERT text_proj→_cond_offset)→R2R smoke/300 对照; LH-VLN 侧子任务条件(target+room)探索策略同法; robo2vlm 恢复后 R2R 全量重跑(摆脱 Qwen 顶班损耗)。
- **卡点**: robo2vlm 仍宕; 6/7 卡不用。

## 2026-08-28(下) —— 对齐FantasyVLN路线落地: R2R快脑微调smoke四项全升(SR10→20/OSR10→60) + LH-VLN P0几何直停

- **做了什么**: ①用户定调"有官方数据必用官方+训练集不测试+看别人怎么写"→查实**官方split铁证**(configs/lh_vln.yaml: train=batch_1-5/val=batch_6/test=batch_7+8), FantasyVLN ISR11.01=训练过的数字(消融表为证), 我们zero-shot 4.25已居公开榜第二。②LH-VLN适配数据(gen_adapt_data_lhvln.py, 回放batch_1-5 oracle轨迹, 604ep/42.3k帧, pos=habitat(x,y,z)/st_task start-end切段口径探针实证)→stop头训练(AUC仅0.65, 小物体信号太弱, **弃用留档**)。③**转化缺口分析**(437ep逐子任务): NE<1m的56个50成功(几何直停可靠)/NE1-2m 54个0成功/77% NE>5m(实例接地); 官方判据=**仅geo_dis<1m**(simulation.py:110, 无FoV要求)。④LH-VLN P0改**2.5m逐步重定位精化+几何直停**(弃VLM confirm), 关键修=**子任务切换清target_world缓存**(否则秒假停steps=[9,1,1], 旧confirm否决路径其实是隐性保护)。⑤**R2R快脑微调**(train_goalnav_r2r.py: 复刻train_goalnav管线, 6666样本/59场景hold-out, warm-start v4, val 0.364)→smoke10。
- **产出**: `train_goalnav_r2r.py`(featurize带缓存/val须train()模式+no_grad=扩散头只在train算loss); `goalnav_r2rft.pt`; `gen_adapt_data_lhvln.py`+`adapt_lhvln/s0-s2`; `runs/stop_head_lhvln.pt`(弃用留档); our_agent P0/P1改动。
- **结论**: ①**R2R快脑微调显著有效**: smoke10 **SR 10→20/OSR 10→60/NE 7.25→6.86/SPL 0.07→0.13**, min_dtg 5.25→3.78(ep1469到0.38m/ep915到2.78m=微调前从未靠近)——域适配收益实锤, 剩余SR-OSR缺口(20 vs 60)=STOP通路仍拖后腿(ep1469到0.38m没停走飞); ②LH-VLN P0 val(batch_6): ISR 2.8→3.7/步数238→42(快5.7×), NE<1m转化4/4=100%, 但NE1-2m 6个仍0成功(锁点落在物面, geo_dis判到中心+snap), 试0.65m逼近阈值(p1)中; ③瓶颈未变=实例接地(NE>5m占76%)。
- **下一步**: R2R 300ep微调版全量(跑中); p1(0.65阈值)batch_6对照→定版后跑test batch_7+8全量; 实例接地(P2)是唯一剩余大杠杆。
- **卡点**: 无(6/7卡不用, 全在GPU2+3)。

## 2026-08-28(早) —— R2R 300ep 全量落盘(n=296): Ours 5.4 vs DualVLN 59.1, 主场结论定稿

- **做了什么**: 300ep 跑至 296/300(用户 08:50 要求让出 6/7 卡停跑, 后 4 条未跑)→与 DualVLN **完全同集**(sample seed0 val_unseen)对照→增补进 `docs/评测_DualVLN_R2R_baseline_0825.md`。GPU 6/7 我们进程全清(现占用者=WuKefei 训练, 与我们无关)。
- **产出**: `r2r_eval/results_our_stophead_300.jsonl`(296ep); 同集对照表: **Ours(v5慢脑+stop头 zero-shot) SR5.4/OSR8.4/SPL4.0/NE9.32/215步 vs DualVLN(R2R训练) SR59.1/OSR68.2/SPL52.6/NE4.38/120步**。
- **结论**: R2R=DualVLN 主场(它在该任务训练), 我们 zero-shot 个位数属预期; 失败主因=实例级接地(多数 ep min_dtg 未进 3m), stop 头只治"停"不治"找错目标"。**我们的优势战场是 LH-VLN**(0827 全量四项全超 DualVLN)——两条评测线叙事定稿: LH-VLN 比增量(zero-shot 即超), R2R 做诚实对照暴露接地短板=下一步改进方向。
- **下一步**: 实例级接地改进(场景图/多视消歧, 或 stop 头式"接地头"监督); 剩余 4 条 R2R ep 有卡时可补跑; LH-VLN 全量结果等用户验收后可写进周报。
- **卡点**: 6/7 卡暂不可用(用户指令, 现 WuKefei 占用)。

## 2026-08-27(晚, 自动模式过夜) —— LH-VLN全量我们zero-shot四项超DualVLN + stop头路B全链推进

- **做了什么**: ①汇总白天跑完的 LH-VLN our_eval 全量(b6/b7/b8a/b8b 共 444ep 全覆盖)→与 DualVLN **同 437 有效集**(444 按同一规则排 7 条数据侧不可跑, 取两结果集 valid 交集)官方 NavigationMetrics 复算。②R2R stop 头(路B): 修 gen_adapt_data(0826 深夜崩于 `env.sim.snap_point`→`pathfinder.snap_point`)+4 片×500ep 并行生成(GPU7, 2000ep/~19k 帧, 每 ep 3 正例+3-4 goal 周边 3-5m 硬负例)→训练两连修(fc 1000 类形状不符剔除 / **首训 valAUC 仅 0.61=标签噪声实锤**: 近 goal 但背对 goal 的正例帧与易负例视觉不可分→正例加视向 ±60° 过滤+LR 上调)→**valAUC 0.74**, 距离分段单调(dtg<1.5m 均分 0.42 vs 3-6m 0.17)。③stop 头集成 our_agent(`stop_head_prob`+`_confirm_stop` 优先用头每步评, 2 票制, VLM confirm 留 fallback)+our_server 传参; 重启服务器(GPU3, 旧进程是 8月25 启动的旧代码)。④释放 GPU6(kill 闲置 DualVLN server, 用户要求明早腾出)。
- **产出**: `docs/评测_我们方法_LHPR-VLN全量_0827.md`; `our_eval/results_our_merged.jsonl`(444ep); `adapt_data/s0-s3`(2000ep); `runs/stop_head_r2r.pt`(AUC0.74); our_agent/our_server stop 头集成; 全量对照表: **Ours zero-shot ISR4.25/CSR3.03/CGT1.95/TAR26.25/NE11.73 vs DualVLN ISR2.98/CSR1.43/CGT1.28/TAR22.20/NE10.93**(SR 双方全 0, 与榜单格局一致)。
- **结论**: ①**zero-shot 即超在 VLN-CE 训练过的 DualVLN**(ISR+43%/CSR 翻倍), NE 略逊=锁错实例走更远, 瓶颈仍是实例接地+STOP 非运动控制; ②stop 头单帧判"距 goal<3m"是弱信号任务(正对 goal 才部分可分), 阈值扫描 P@0.5=0.42/R=0.40——部署处在"已到终地标附近"的强先验下查询, 实战价值待 smoke 验证; ③smoke10 首试 thr=0.55 **翻车**: 阈值高于近距均分 0.42→永不判停, ep1606 min_dtg 1.10 走满 500 步 NE14.72(v5 同 ep 101 步成功 NE2.90)→降 thr=0.40 重跑中。
- **下一步**: smoke10(thr0.40)对照 v5(SR10%/NE7.25)→提升则 R2R 300ep 全量过夜(同 DualVLN seed0 子集 SR58.7); 备选增强=距离回归头替代二分类/stop 头与几何 dist 联合判停。
- **卡点**: stop 头绝对精度有限(P 天花板 ~0.6), 若 smoke 仍不如 v5 则退回"几何 dist<1.5m 直通+stop 头仅否决"的混合判停。

## 2026-08-26 —— R2R慢脑五代迭代(v0→v5): 复读病根因实锤→地标链重构, smoke SR 0→10%破零

- **做了什么**: 诊断 R2R smoke 全败(v0 SR=0)。diag_r2r_our.py 进程内单 episode 逐步实录+轨迹图, 五代慢脑迭代: v1 加到达事件/STOP confirm→**复读病换皮**(GOTO复读206次→TURN复读492次/FORWARD复读61次, min_dtg 0.59 走过头); v2 **弃多选改地标链**(指令两问解析终点+中途地标→逐地标 bbox 定位→链尾 confirm)=回归 robo2vlm 甜区(bbox+yes/no), 导航到 min_dtg 0.04m 但"指令完成了吗"元问题 confirm 站 GT 上 7cm 连答 52 次 no; v3 目标超时放弃+周期刷新; v4 confirm 改接地近距问→假停(第2步停 NE5.03/假定位瞬达); v5 定位两帧世界系复核采信+confirm 2票+解析两问。smoke10(与 v0 同 episode 集, sample seed0)经 our_server+run_r2r 跑完。
- **产出**: `our_agent.py` v5(act_r2r 地标链慢脑: parse_chain/两帧定位复核/目标45步超时/末地标近距confirm/skip+换视角探索); `r2r_eval/results_our_v5_smoke10.jsonl`; `diag_r2r_ep{0,1}_v{0..5}.*` 实录; `docs/评测_DualVLN_R2R_baseline_0825.md` 增补五代表; DualVLN R2R 300ep 终版数字落盘(SR58.7/SPL52.2/OSR67.7/NE4.48)。
- **结论**: ①**robo2vlm 多选一决策根本不可用**(复读自己最近输出, 换 prompt 只换复读内容), bbox+yes/no 甜区再次验证=慢脑定式; ②快脑+几何全程无责(v2 起多次 min_dtg<0.5m), 瓶颈全在 STOP 通路与 VLM 实例级接地(bbox中心锁水面中央偏3.5m/锁错桌实例); ③**v5 smoke SR 0→10%/NE 9.15→7.25/平均步数 500→161**(ep1606 停 101 步 NE2.90 首个完整成功), vs DualVLN 同链 ~60 仍差距大但纯机制迭代未训练。
- **下一步**: 用户定方向——R2R 继续(300 子集对照/接地改进) 或转 LH-VLN our_eval 全量 437ep(zero-shot NE 已≈DualVLN, ISR 11.1% smoke 底子在, 是我们更有利的战场)。
- **卡点**: 共享集群抢卡; our_server 跑完即 kill 释放(已执行)。

## 2026-08-25 —— LH-VLN全量跑分落盘 + 立项B(R2R)链通 + our_eval/our_server 搭好

- **做了什么**: ①LH-VLN 全量 DualVLN 评测收尾: 444ep 跑完(含 run_topup.py 补跑 8 条)→排除 7 条数据侧不可跑(5 条 HM3D 语义网格缺小物件几何 obj_centers 提不出 + 2 条 GT 测地 inf)→**有效 n=437, 官方口径复算 SR=0/ISR=2.98/CSR=1.43/CGT=1.28/TAR=22.2/NE=10.93**(compute_metrics.py 官方 NavigationMetrics 原样), 与"全场 SR=0、FantasyVLN ISR=11 最好"的榜单格局一致=链可信。②立项B(R2R): mp3d_r2r 数据备齐, 写 agent 无关客户端 run_r2r.py(habitat 官方 vln_r2r.yaml+官方 measurements+InternNav AgentServer 协议), DualVLN baseline 300ep(seed0 抽自 1839)跑完 6.7h。③我们方法两侧冒烟: our_eval/our_agent.py(VLM-WHERE bbox定位+几何反投影 goal_ego+条件DD快脑, habitat→我们坐标系探针实证 y=-hab_z 否则镜像/floor=position.y)+标准化 our_server.py(InternNav 协议兼容, target 与 instruction 双模式)。
- **产出**: `docs/评测_DualVLN_LHPR-VLN全量结果_0825.md`; `benchmarks/LH-VLN/dualvln_eval/{results_merged.jsonl,compute_metrics.py,run_topup.py}`; `benchmarks/InternNav/r2r_eval/{run_r2r.py,results_dualvln_300.jsonl}`; `benchmarks/LH-VLN/our_eval/{our_agent.py,our_server.py,run_our_eval.py,probe_our.py}`。
- **结论**: DualVLN R2R 实测 **SR58.7/SPL52.2/OSR67.7/NE4.48**(官方全量 64.3/58.5/70.7/4.05, 300 子集合理偏低)=R2R 链可信, 对照用 300 子集同 seed。我们 zero-shot(goalnav_gate_lab_v4.pt): **LH-VLN 侧先实锤两个坐标系 bug**(平面镜像须 y=-hab_z/floor=position.y 勿再减 SENSOR_H; 另修卡死只计前进帧/目标点等比预缩/VLM locate 加 room 消歧), smoke1 ISR0/TAR9% → **smoke2 ISR11.1%/TAR30.5%**(真导航 d3.7→0.5 到位 stop, 参照系=DualVLN 全量 ISR 仅 2.98); 剩余失败=锁错实例+探索弱。**R2R 侧 v0 smoke10 全败 SR=0/OSR10/NE9.15 撞 500 步上限**, 机制全通但慢脑指令跟踪是最简实现=最弱环, 待诊断。
- **下一步**: 0826 诊断 R2R 失败根因(diag_r2r_our.py)→修到 smoke 非零再子集对照; LH-VLN our_eval 全量 437ep 待排。
- **卡点**: 共享集群抢卡; 跑评测须 lhvln env。

## 2026-08-24 —— DualVLN×LHPR-VLN评测链: HM3D镜像数据搞定 + 模型服务器冒烟PASS

- **做了什么**: 接0821调研与0822用户已备的benchmarks/(LH-VLN+InternNav仓库、InternVLA-N1-DualVLN权重、lhvln env、episode数据), 把DualVLN baseline评测链搭到冒烟通过。①HM3D卡点破解: Matterport注册页403(墙)→hf-mirror找到**免token完整镜像zhuhu00/hm3d**(官方tar格式), 核对全部8个batch共153个场景被镜像(145 train+36 val)100%覆盖→后台下载43GB+自动解包流水线(解到LH-VLN/data/hm3d/{train,val}+生成根级合并scene_dataset_config)。②internvla env从零建(py3.10+torch2.8cu128+transformers4.51+flash_attn 2.8.3预编译轮abiTRUE+diffusers降到0.32.2修LuminaNextDiT gradient_checkpointing签名冲突+setuptools钉80.9.0因≥81删了pkg_resources+软链checkpoints)。③写适配层: LH-VLN/dualvln_eval/run_eval.py(纯HTTP客户端复刻AgentClient协议免装internnav/复用官方SceneSimulator+NavigationMetrics/动作映射0stop 1fwd 2left 3right/-1 noop防死锁)+InternNav服务器配置lhvln_dualvln_server_cfg.py(512×512 hfov90对齐LH-VLN前视/partial_async)。
- **产出**: `benchmarks/LH-VLN/dualvln_eval/{run_eval.py,smoke_client.py}`; `benchmarks/InternNav/scripts/eval/configs/lhvln_dualvln_server_cfg.py`; `benchmarks/hm3d_tars/{dl.sh,extract.sh}`; visualization.py的PyQt5改optional补丁。
- **结论**: 模型侧全链冒烟PASS(init 132s加载16G, S2首步9.1s, S1后续0.1s/步)。**端到端也通且目标真值已修正**: ①镜像缺scene_instance→`.semantic.txt`硬链成`.basis.scn`修复SSD加载; ②habitat 0.3.1认不出.semantic.glb(只找*_semantic.ply)→aabb全0→测地距离全撞同一点(症状nav_err[i]恒=gt_path[i+1]), 解法=extract_obj_centers.py从semantic.glb顶点COLOR_0(uint16 linear→sRGB EOTF)聚bbox中心; ③HM3D glb是z-up(orient_up=(0,0,1))→world=(x,z,-y), 漏转目标全埋地下5m。修后box测地2.84m/snap正常/nav_err逐子任务不同。⚠口径caveat: DualVLN按15°转角训练 vs LH-VLN官方30°动作空间; 无抓取→Grab/Release子任务天然失败。
- **下一步**: ~~HM3D下完~~→~~自动解包~~→~~batch_6冒烟~~→**全量评测已开跑(batch_6/7/8共444ep, 后台bp5gc5b3z, 约6h)**; 结果落dualvln_eval/results_full_b678.jsonl后汇总写对照文档。对照数字已备(FantasyVLN SR2.44/ISR11.01/CSR9.64/CGT8.99, 其余含MGDM全SR=0——DualVLN的SR=0是好公司, ISR/CSR/TAR/NE才是区分度)。用户定评测策略两原则: 跟DualVLN比要去它的主场VLN-CE R2R(需移植我们方法, 立项B); 用某评测就跟该评测已有数字比(引FantasyVLN/MGDM同口径)。
- **卡点**: 共享集群抢卡(服务器模型只在单卡, 被挤就要换卡重启——注意换卡必须重启服务器否则重复加载OOM)。

## 2026-08-21 —— 论文 baseline/benchmark 三轮调研 + 全模块技术盘点

- **做了什么**: ①用户要"继续调研，重点论文baseline和benchmark"→deep-research×3轮(共314agent/319声明/75条三票对抗验证, 69过6杀); ②用户七问盘点(单双目/ZED→3DGS创新点/baseline对照/场景图边界与更新/GT来源/VLM对比/DiffusionDrive改动量)+论文定位定稿(主题=具身机器人高精度定位与任务目标导航, 创新点20%目标表征+60%粗导航+20%精定位, **场景图留作另一篇**)→5路代码盘点agent全仓库核实(全带行号)。
- **产出**: `docs/调研_baseline与benchmark_0821.md`(benchmark=LHPR-VLN命名更正+GOAT-Bench/baseline=DualVLN+NavDP可复现/RELOC两铁证撑双闸门/语义可控性协议骨架=ControlNet++回读+P/R线+ICML2025 sanity checks/真机轻量协议先例); `docs/论文盘点_模块技术与创新点_0821.md`(七问全答: 单目+SDK深度,右目hq5lr证伪/每块沿用-改进-创新表/DD改动=新机制7+模块级8+配置级6/场景图四问/GT清单/VLM对比计划/20-60-20映射/诚实风险6条)。
- **结论**: 组合创新(语义可控扩散快脑+RELOC周期纠偏+真机全链)在已核实文献版图无竞品; "同目标换条件→轨迹分布变"在任何主流benchmark未被量化=空白可填; ⚠写作纪律=BEV±32m没改勿声称改了/GSplatLoc两篇同名勿踩/右目不是没用是实验证伪。
- **下一步**: 用户审盘点文档; 剩余缺口=VLN-CE官方榜数字/NoMaD真机协议细节/Diffusion Policy多模态实验精读/Habitat接口代码(零落地)。
- **卡点**: 真机无GT(无odom无mocap)是评测最短板, 建议AprilTag+卷尺终态协议或上公开benchmark借仿真器GT。

## 2026-08-19 —— 架构整合文档 + 真机部署方案文档

- **做了什么**: 用户要两份文档：①当前架构整合 ②真机部署怎么做。三路事实核查(仿真全链源码+outputs产物/robot_control_current运控+robot_orig_code已部署DiffusionDrive/docs现有文档盘点)后写就。
- **产出**: `docs/架构整合_当前全链_0819.md`(一页总览ASCII图/离线建图表/双脑NavCmd契约/RELOC双闸门/ALIGN→DOCK链/lab_mission CLI/环境产物速查/诚实残留6条/文档地图, 替代停在07-08的架构与现状_实时.md); `docs/真机部署方案_0819.md`(起点=robot_orig已在Orin实时跑通DiffusionDrive/硬件接口事实表含0.5s指令超时+无odom/Orin-vs-GPU服务器拓扑/逐模块移植清单/P0-P4分阶段/风险清单)。
- **结论**: 真机部署最大两个gap是**status 8维定义不一致**(真机版=模式onehot+vxvy+axay vs 我们lab版=goal_ego/head_ego/vx/hist_w/gamma/mode)和**BEV范围不一致**(±32m NAVSIM直方图 vs 局部占据窗+时序历史)→方案A=改真机节点特征构造适配我们ckpt(推荐), 方案B=真机数据重训; **无里程计(指令回显非实测/轮速10s才刷)正好是RELOC周期纠偏的对症场景**, 漂移被每~2m纠偏周期清零; 尺度债0.70×在真机新场景用正确cam_height一次避开。
- **下一步**: 用户审两份文档; P0冒烟(dry-run特征核对+RELOC runbook ABC)。
- **卡点**: 无。

---
## 2026-08-13(晚) —— 回归头v4双域+全流程; 用户判"都没对准"→trace复核三修全愈; 周报+架构图v4

- **做了什么**: v4双域头(真zed+机器人姿态42.7k)训完(val 5.1/6.4cm/2.4°)→真zed实测(all帧0.6-0.7cm/0.11°含记忆, val帧15.3cm/4.5°比v3的12-14cm略退=双域代价)→全流程跑(sim闸门accept仅2/9, GS兜底正确性不变; ⚠白桌假接收16.3cm一次=溅射区loss地形不可信, 被下游吸收)。**用户看视频判"都没对准尤其柜子"**→trace真位姿复核实锤: 柜dock横向+42cm(believed报0.0=缝检测锁错竖线)/三站yaw被EDGE伺服带偏-5.0/-6.4/-8.9°(渲染碎线族LSD融合≠地图front)/"front偏0.0°"=servo覆盖FRONT后自比同义反复。三修: ①缝/簇过节点投影u_node锚定窗(0.15/0.20W), 出窗→几何横向兜底; ②EDGE伺服|视觉-地图|>2°回滚地图front; ③落盘geo_face_dist/geo_lateral真值口径。
- **产出**: 修复版全流程mp4(正名, v4坏对准版_v4badalign留档); `docs/周报_0812-0813_贴脸对接与回归头RELOC.md`(三块PPT版); `outputs/arch_diagram_v4.png`(RELOC子系统实框+ALIGN/DOCK底排); `scripts/diag_edge_yaw.py`(yaw扫描渲染+LSD斜率诊断)。
- **结论**: 修复版trace真值=柜+0.4cm/0.74°、微-5.5cm/0.04°、桌+5.8cm/0.70°, 面距15cm believed≈几何; 三站EDGE全正确回滚(差>2°); 目检柜双门正脸居中无侧面(修复前露左侧面)。⭐教训: believed指标和视觉伺服都要独立几何复核, 溅射区渲染特征检测(缝/线/loss)一律不可信。
- **下一步**: 用户验收修复版视频; 真机runbook B/C级; 白桌confirm(渲染天花板)/头sim accept率/尺度0.70×根治=已知残留。
- **卡点**: 无。

---
## 2026-08-12 贴脸对接两连修（median穿模→p10→min(ED,GEO)）+ 交付版视频

- **做了什么**：复盘0811 precdock首版日志发现三站对接全翻车（median被"看穿玻璃门"拉扯：铁柜读2.8m实际0.85m/显微镜双峰48↔6.7振荡/白桌ED无效）→v1改ED-p10+跳涨拒信+0.35m可信区+盲爬，跑完**用trace真位姿复核发现仍穿模**（believed 15cm，实际距节点仅1.3~2.6cm——ED-p10近距系统性过读10~25cm）→v2三修：①面距=min(ED-p10, 几何dist-FACE_OFF)（ED过读取几何下限，believed位姿RELOC残差0.5cm可信）②站位先居中再接近（0.35m处台面簇在-12°俯角FOV下缘外=v1"无簇"根因）③undock纯盲退。重跑出片。
- **产出**：`outputs/lab_mission_110744_zed_front_perp_lvl_geo0804_lab_sb2_reloc_odom_precdock0812.mp4`（526帧35s：RELOC巡航纠偏+三站对准+⑩⑪confirm+15cm贴脸对接+撤离+回原点）；`lab_mission_reloc_odom_precdock_0812.json`；v1穿模版留档`*_v1OVERTRAVEL.*`（mp4/json/log三件套）。
- **结论**：三站对接全部 面距15cm+横向居中**-0.4/-0.7/-0.6cm**+front≤0.98°；铁柜缝LSD站位6轮收敛-0.1cm；显微镜簇1822pt/白桌4515pt居中收敛。RELOC全程：漂移中位13cm→终横向中位0.9cm/bearing 0.7°；周期纠偏6次残差中位0.2cm。⚠诚实残留：白桌confirm又NOT-CONFIRMED(review-fail 1/3)=0804起老问题（站位渲染溅射+VLM边界掷硬币，前一天同管线3/3 YES）。⭐教训：对接指标必须trace真位姿复核，believed自报不可信。
- **下一步**：周四交付（真机验证按runbook ABC三级）；可选=白桌confirm稳定性（多投票/换确认帧）、精对准回归头训练（GPU0/1空闲，脚本未写）。
- **卡点**：跑lab_mission必须`/home/DongBaorong/discoverse_venv/bin/python`（系统python缺glfw）；GPU2-7被WangBizi starVLA占满。

---
---

## 2026-08-11 GS重定位尺度根因实锤 + RELOC精定位闭环落地（周四交付冲刺）

- **做了什么**：①三假设鉴别诊断（`gs_reloc_diag.py`，GT零扰动回归r=a+b·d）→根因=**全局尺度**（b=-0.30跨10帧一致，散布0.06；align的cam_height=1.2m假设错，110744手持绕看采集真实~1.71m→世界仅真实0.70×）；②`gs_relocate.py`加OBS_SCALE=0.70修复；③两参数标定（`gs_reloc_calib.py`）证无全局常数偏移（o≈0.004），残留=逐帧内容相关偏差；④RELOC接入lab_mission（`--reloc_demo`：漂移注入→伪zed观测(渲染ED÷0.70+σ1cm+1%+5%丢点)→GS优化纠偏→believed位姿align_front→真位姿测终误差，终loss>0.12拒收保护）；⑤精读4篇论文原文(Vesta/Qwen-3D/HumanCLAW/RynnWorld-4D)核实分析文档。
- **产出**：`outputs/gs_relocate_110744s2/reloc_eval.json`+eval_scaled_0811.log（4档×24帧真zed评测）；`outputs/lab_mission_reloc_0811.json`+`lab_mission_110744_zed_front_perp_lvl_geo0804_lab_sb2_reloc.mp4`（357帧）；`scripts/gs_reloc_diag.py`/`gs_reloc_calib.py`/`gs_reloc_smoke_mission.py`。
- **结论**：重定位(真zed,24帧)：**收敛域±0.5m/±25°内初误差完全吸收**，终误差=噪声底：**水平中位5.6cm(p90 8.9)/yaw 1.08°**（垂直13cm弱可观，导航用不上）；闭环三站：漂移16~25cm注入→**终横向0.2/0.3/3.5cm、bearing≤2.1°、front≤0.9°，VLM三站全YES**。4篇论文无一要求改架构——Qwen-3D几何解码grounding印证我们场景图路线；HumanCLAW"具身自我意识缺失"是RELOC的论文级motivation；分析文档核心方案C(VLM输出3D坐标)被Qwen-3D原文Fig.2批判+违反实测铁律，且编造了第5篇(VISReg未上传)。
- **下一步**：周四晚前出三块PPT格式周报（尺度根因+精度表+闭环视频）；可选吸收=HumanCLAW执行前verifier、Vesta记忆模态消融(纯文本记忆学捷径→我们⑪key-frame已含图像)；彻底治尺度=重跑align cam_height=1.71+重建clusters/nav_ds（贵，暂缓）。
- **卡点**：白桌站渲染溅射（off-path采集天花板）；lab_mission需PYTHONPATH=third_party/discoverse；世界系距离全为真实0.70×（与真机交互必须换算）。

---
---

### 2026-08-06 —— 周报0802-0805 + 架构图v3更新

**做了什么**: 用户要本周工作总结→`docs/周报_0802-0805_论文改进与搜索行为.md`(用户定三块PPT结构: 一=架构优化分析(⑦时序历史/⑧NavCmd/⑩⑪确认结构化/⑨搜索行为, 各配"为什么"+证据)/二=闭环效果表(loss0.272/三站全YES/搜索y=-5.1m/z场1.5°)+翻车修复+诚实残留/三=调研三优先+当前问题5条+下周P0-P2); 架构图v2(0715走廊版)过时→`scripts/draw_arch_v3.py`重画(加离线建图带: 3DGS深度监督+z场扶平→场景图RAG185节点; 慢脑=四段confirm+keyframe复核; 接口=NavCmd{L,τ,Φ}; 快脑=scene_gate+时序历史Wγ+骨架DFS搜索; 控制器=turn-then-go+safe_vel; 待接=GS重定位虚线框放中带右侧空白)。
**产出**: 周报md(三块PPT版); `outputs/arch_diagram_v3.png`(目检五轮修重叠: 删"快脑(实时)"标签并入框内/待接框移出离线带/紫说明文字缩短右移避让NavCmd箭头/长虚线走底层z=2)。
**结论**: 本周论文改进⑦⑧⑩⑪⑨全落地闭环验证, 搜索demo真探索; 最大悬念=GS重定位深度系统性偏差(-0.24~-0.66m)待定位。
**下一步**: 下周P0=GS重定位深度偏差三假设验证(或转方案B zed融合点云配准); P1=搜索折返果断性/精对准回归头/白桌confirm多投票/建图粒度P1。
**卡点**: 同周报§9(GS重定位断点/白桌confirm掷硬币/角落折返不果断/自研VLM仍挂/GPU排队)。

---
---

### 2026-08-05 —— 高精度导航大调研(12路)+GS重定位落地试跑→撞见zed深度vs渲染ED系统性偏差(断点)

**做了什么**: ①用户要"能接入我们架构的高精度导航方法"→12路并行搜索(VLN SOTA/扩散导航/场景图RAG/3DGS定位/末段/拓扑/BEV/主动感知/探索/空间记忆/GRPO/视觉伺服)→`docs/调研_高精度导航方法_0805.md`(Tier1直接落地/Tier2训练侧/Tier3方向印证)。②用户拍板"先试成功率最高+精度最高的"→选**GSplatLoc式render-and-compare重定位**(免训练): 写`scripts/gs_relocate.py`(zed npz观测深度 vs gsplat渲染ED, 世界系SE(3)可微参数化→colmap viewmat, coarse-to-fine Adam+Charbonnier+trim, GT加扰动评测协议)。
**产出**: 调研文档; gs_relocate.py(可跑); outputs/gs_relocate_110744s2/(sanity json+viz)。
**结论**: gsplat 1.5.3 viewmat梯度可微✓(⚠相机在场景背后→alpha=0→grad=0假阴性); **但GT零扰动sanity漂到25.5cm/yaw2.12°=不合格**, 根因=渲染ED深度比zed度量深度系统性偏小(median -0.66/-0.26/-0.24m, 满幅均匀非局部artifact)。
**下一步**: 查三假设(按嫌疑): ①zed点云参考系≠左目光学系(原点常数平移→视角相关偏差, 恰好解释跨帧不一致); ②全局尺度误差(⚠此前backproject等几何验证全是同一换算自洽, zed深度是第一个独立度量源!); ③ED期望深度vs首面深度偏差。备选方案B=zed融合点云当配准地图(同传感器偏差自抵消)。
**卡点**: 上述偏差未定位前, 重定位精度不可用; 用户关机搬运, 现场已存memory(gs-reloc-depth-bias-breakpoint-0805)。

---
--- —— z场扶平+正脸线标注落地, v5任务重跑2/3primary(白桌=渲染天花板Qwen掷硬币); ⚠我两错: 地面带测错+忘带--targets

**做了什么**: 用户判"正对和地平面都有问题, 正对向量不对"→双修+重跑。
- **地平面**: 实测lvl世界残斜(逐waypoint局部平面)mean2.9°/median3.2°→写`apply_zlevel_110744.py`(逐waypoint平面拟合粗场→粗场±5cm残差带取真地面→0.25m格中位数精修场, z'=z-h(x,y)竖向反弯, world/render两ply+npz+clusters同步)→**median 3.17→1.52°**(尾部粗糙点是重建本底噪, 后处理极限)。⭐踩坑两连: ①首版|z|<0.2带测地面混进家具底→越扶越歪(回滚); ②扶平前后度量必须同一选点法(最低切片法在扶平后选到不同点子集=假读数), 一致口径=±0.15带逐waypoint。
- **正对向量**: 旧FPV法线箭头正视时退化(俯拍22°下投成竖线扎地/缩点)→改画**正脸线**(过目标⊥front的水平线段±0.3m, 正对时水平压中心)。
- **重跑(v5快脑+geo0804)**: 首跑显微镜NOT-CONFIRMED=⭐**我忘带--targets zed_front_perp_lvl.json**(默认zed.json无front_deg→align_front退回face()→站位1.66m远显微镜太小, Qwen判"tv"); 带对后**柜YES(primary)/微YES(primary, 0.85m站距berr-1.0°)/白桌NOT-CONFIRMED(review-fail 0/3, Scene/Action矛盾已标记)**。
- **白桌A/B对照实验**(同机位lvl vs zlvl渲染各一帧): 溅射水平几乎一致→**z场没把渲染变差, 白桌confirm翻车=站位渲染天花板+Qwen边界掷硬币**(0803过=复核帧运气好)。
**产出**: `lab_mission_110744_zed_front_perp_lvl_geo0804_lab_v5_sb2.mp4`(400帧) / apply_zlevel_110744.py / zlvl两ply+软链(旧lvl留盘) / `docs/架构详解_小白版_0804.md` / 审计lab_mission_slowbrain_0804.jsonl / A/B图ab_lvl.png ab_zlvl.png。错靶首跑视频改名_WRONGTARGETS留档。
**结论**: 运动层三站全准(距0.85~1.03m, berr≤1.9°, front_err≤1.2°); 地平面对视频中位1.5°可读; 正脸线标注直观; v5快脑任务腿无退化(搜索数据没污染巡航)。白桌confirm稳定性=渲染天花板问题, 可选=确认改多投票。
**下一步**: (待用户定)白桌confirm多投票/建图粒度P1/真机。
**卡点**: 无。

---
---

### 2026-08-04 —— ⭐v5搜索行为闭环demo: 首跑翻车根因=search循环漏调hist_update(W8状态+单帧BEV离分布); 修后400步原位→实验室最深处

**做了什么**: 接0803(v5快脑+探针全过)跑`lab_mission.py --search_demo 400`闭环搜索demo。
- **首跑翻车(0803版代码)**: 30步内机器人鼻子楔进障碍, 之后在~1m泡泡里原地打转400步(yaw 2°→580°, v≈0.01)。逐帧+逐行锤出**两个叠加根因**: ①⭐**search循环从没调`hist_update()`**(全repo只在cruise循环line367调)→_hist_buf恒空→`if hist_w>0 and _hist_buf`守卫回退单帧BEV→**status声称W8γ0.9但BEV是单帧=训练/测试输入严重离分布**(搜索训练样本全带W4-8覆盖式历史), 模型直接开进墙; ②**demo控制器只能vx>0前进**: 搜索GT含死胡同折返(训练数据yaw恒朝行走方向), 回退轨迹被"前视点比例转向"译成贴墙打转, 且vx有0.10下限楔死也停不下来。
- **修(lab_mission.py search块)**: 循环内加hist_update(); 控制器改**turn-then-go**(|head_err|>0.6=前视点在背后→先原地转执行折返, 否则比例跟), 去掉vx下限, 贴障走已有的safe_vel势场逃逸(减速+朝空方向转, ESC标记上HUD); 输出改名_0804(0803失败版留档不覆盖)。
**产出**: `outputs/lab_search_demo_lab_v5_0804.mp4`(400帧) + 0803失败版留档对照。
**结论**: ⭐修后**真探索**: 原位(0,0)→穿越整个任务区→实验室最深处(y=-5.1m, 铁柜/显微镜之外), FPV全程干净贴采集轨迹, goal=(0,0)+mode=1契约闭环成立(快脑自己覆盖式走位, 无慢脑无目标点)。**残留(诚实)**: 最深处角落口袋徘徊(v0.01~0.02, 折返大角度轮转与safe_vel逃逸互相打断, 净位移慢)——"死胡同果断折返"还不够脆, 可选改进=折返转向commit(转完才许平移)/更长前视/角落搜索数据加权。
**下一步**: (待用户定)用户看0804视频 / 折返果断性改进 / 建图粒度P1 / 真机。
**卡点**: 无。

---
---

### 2026-08-03 —— ⭐用户催论文改进项落地: ⑧结构化参数包+⑩四段推理链+⑪key-frame复核 一次闭环三站全YES

**做了什么**: 用户否"建图粒度剖析不痛不痒"→掰回Qwen-RobotNav论文改进清单(周报§6)优先。三项全落`scripts/lab_mission.py`:
- **⑧慢脑→快脑结构化参数包(论文nav(L,τ,Φ), ⭐⭐⭐P0)**: `NavCmd{L=模式, τ=(名,坐标), bearing/dist空间提示, Φ=(W,γ,vx)}`每腿派发+落盘; `PHI`字典集中"模式→观察配置"映射(巡航W2γ0.7vx0.5/转向W8γ0.85vx0.0), cruise改收phi参数(不再读全局args, 消灭日志与实际不一致)。
- **⑩四段推理链confirm(⭐⭐P1)**: History(任务上下文已知)/Scene(VLM短描述"main object in center?")/Progress(几何量dist+front_err)/Action(严格yes/no同0802判据)——分段独立不串联(弱VLM喂回自己的话会糊), 段段落盘, scene/action矛盾自动标(审计定位是哪段错)。
- **⑪判no时key-frame复核(⭐⭐P1)**: 接近过程每6帧存证据帧(panel复用渲染零开销), 判no挑"距离+方位差"top3(帧距≥8)逐帧同判据复问, **≥2帧yes才翻案**(单帧防侥幸); 机器人不动只回查已有证据≠0730禁的扫描挽尊。
- 输出版本化加_sb2后缀(不覆盖0802视频)。
**产出**: `outputs/lab_mission_110744_zed_front_perp_lvl_geo0731b_lab_v4_sb2.mp4`(392帧7.42m) / `outputs/lab_mission_slowbrain_0803.jsonl`(⑧包+⑩四段+⑪证据全链)。
**结论**: ⭐**白桌NOT-CONFIRMED被⑪治住**: Action仍no(站位直视镜头的渲染溅射是真的, Qwen判no诚实)但复核f324(berr45°)/f288(berr51°)两帧yes→2/3翻案CONFIRMED(review); 铁柜/显微镜primary直过(Scene自答"cabinet"/"microscope"=四段互证)。闭环三站全YES(v4快脑未动, 运动行为与0802逐帧同)。
**下一步**: ⑨骨架化搜索行为数据(快脑"找目标"行为空白, 需训练) / 用户看_sb2视频 / 建图粒度P1(诊断已完成: node130=SAM掩码溢出非合并阈值, node47怪物=BG_CLASSES缺RAM场景词)。
**卡点**: 自研VLM 10.0.0.18仍挂, 用本地Qwen3.5-4B; GPU全卡别人在训, ⑨训练要排队。

---
---
---

### 2026-08-02 —— ⭐观察策略参数化(论文最核心claim)落地: BEV-only参数化时序历史, v4快脑loss0.272闭环三站

**做了什么**: 用户拍板BEV-only(FPV单帧留给慢脑; token级FPV历史在±32m/8×8=4m/格下跨位姿对不齐被否)。
①`stage_b_eval.py`加`lidar_feature_fov_hist`: 历史位姿FOV锥点(世界系选择)→当前ego系单次变换→γ^lag种子确定性保留(衰减=遗忘)→拼接调一次官方points_to_bev_tensor(官方/模型架构零改动); W=0与旧版逐位一致(单测6项全过scripts/test_lidar_hist.py); 合并后重施hypot>0.2近距掩码(防ego原点假亮斑)不施FOV锥(身后点正是意义)。
②`gen_nav_dataset_lab_v4.py`: 基数据集与v3逐位一致(新随机性全走RandomState(1)), 每样本hist_poses(8,3@0.4m)/W/γ, P(W=0)=0.25(保护run_inspection等旧脚本单帧分支), 596样本。
③`train_goalnav_gate_v3.py`: (W,γ)走status[5]/status[6](全repo核实恒0)→暖启v3→`goalnav_gate_lab_v4.pt`(240ep loss0.272<v3的0.327, 门1.000/0.000); ref位姿FOV选点缓存; 转向行带源样本(W,γ)。
④`lab_mission.py`: 位姿环形缓冲(0.4m/0.3rad入队留8), dd_traj分相位观察参数(巡航W2γ0.7/转向W8γ0.85)+HUD显示[W2γ0.7]+顺手修v2遗留vx mismatch(巡航0.5/转向0.0)。
⑤`probe_obs_params.py`: 固定观测扫W×γ→8面板图(每面板=该参数实际BEV+轨迹)。
**产出**: goalnav_gate_lab_v4.pt / nav_ds_..._v4 / obs_param_probe_lab_v4.png / `lab_mission_110744_zed_front_perp_lvl_geo0731b_lab_v4.mp4`(396帧7.66m)。
**结论**: ⭐闭环三站全到全对准(bearing-front≤2.5°, 白桌yaw_err0.9°); 铁柜YES/显微镜YES/白桌NOT-CONFIRMED=⚠VLM是自研10.0.0.18又挂(Connection refused)切的本地Qwen3.5-4B(scripts/vlm_server_local.py GPU3, VLM_URL=127.0.0.1:8152)+白桌站位渲染溅射严重, 非运动问题(v3同站位robo2vlm判YES=Qwen严robo2vlm宽老差异)。探针: 目标寻求对观察参数不变(好事), 局部路径形状随W系统性平滑偏移(横向+0.12→+0.47m)=模型确实在用历史。
**下一步**: (待用户定)建图粒度P1(过合并拆node130/20) / 节点属性化 / 真机部署。
**卡点**: 自研VLM服务器10.0.0.18:8152又不在线(0711同款), 本次用本地Qwen顶替。

### 2026-07-31(续5) —— P1训练经验全落地: v3(条件连续化+减速GT+vx随机)闭环全YES + 周报文档

**做了什么**: `scripts/gen_nav_dataset_lab_v3.py`(起点xy±0.25m/yaw±0.4连续抖动, 目标±100°/1~6m连续采样, ⭐减速段GT前疏后密实测间距比0.08, vx~U(0.3,0.8)随机化存meta)→596样本(429cruise+167decel); train_goalnav_gate_v2加--ds_suffix+vx进status[4]→`goalnav_gate_lab_v3.pt`(240ep loss0.327, 门1.000/0.000); 闭环`..._geo0731b_lab_v3.mp4`(401帧, 三站全YES, 帧数多=减速GT到站更稳); P1-2分域归一化`outputs/traj_norm_scales.json`(lab1.56/走廊3.49/NAVSIM32m)。用户要"改动+意义写进周报"→`docs/周报_0727-0731_场景图RAG与论文训练经验.md`: §1场景图算法目标/框架/输入中间状态输出+3张当时实景图(用户指出节点标注图=3D点簇脏vs外观卡片=真实crops干净), §4六项已完成改动各配意义+诚实边界(观察策略参数化未动列下周)。
**产出**: goalnav_gate_lab_v3.pt / nav_ds_..._v3数据集 / traj_norm_scales.json / 周报md+主图week_summary_0727-0731_stations.png / v3闭环视频。
**结论**: 论文数据+接口层经验全部落地并闭环验证; 观察策略参数化(时序历史作条件=论文最核心)是下一个点上的活。
**下一步**: (待用户定)观察策略参数化(快脑加参数化历史) / 建图粒度P1 / 真机部署。
**卡点**: 无。

### 2026-07-31(续4) —— Qwen-RobotNav训练经验落地P0: 数据再平衡+条件dropout→快脑v2闭环全YES

**做了什么**: 用户供论文PDF精读37页(核实版分析+另一个AI多处编造纠正, 见memory qwen-robotnav-paper-verified-0731), 拍板按计划执行训练侧经验。P0-1+P0-2合一实现`goalnav/pipeline/train_goalnav_gate_v2.py`: ①数据再平衡(论文§4.1.2前进45%规则)→按GT末点偏航角分类: 直线(<10°,191个)只留45%, 中间/转向全留, 决策点(≥35°)复制×2, 选择先于featurize省算力(621→656); ②条件dropout(论文§4.1.5同轨迹多先验变体)→直线样本30%追加"目标点置零"变体(同obs同GT仅status目标=0, 直线时走廊跟随≈目标跟随监督不冲突); warm-start=goalnav_gate_lab.pt。
**产出**: `goalnav/runs/goalnav_gate_lab_v2.pt`(240ep loss 0.864→0.440, 门保持前向1.000/转向0.000) + 闭环`lab_mission_110744_zed_front_perp_lvl_geo0731b_lab_v2.mp4`(335帧): **三站VLM confirm全YES(v1版白桌NOT-CONFIRMED)**, ALIGN bearing-front≤1.9°, 巡航平滑(wz std 0.26)。旧版0731b改名_lab.mp4保留。
**结论**: 论文数据分布规则在我们pipeline直接有效; 条件dropout与gate机制兼容(门值没退化)。
**下一步**: P1-1条件连续化+随机化(Phase2数据生成器)→P1-2分域归一化。
**卡点**: 无。

### 2026-07-31(续3) —— ⭐渲染几何根本没扶平(CaptureFPV吃pre-align ply)→补同步渲染ply, 地面终平

**做了什么**: 用户判"虚线确实对于地面是歪的"→查出**续2的untwist只作用在world ply(占据栅格用), 而FPV视频是CaptureFPV渲染gs_vggto_<session>.ply(pre-align归一化模型)+npz刚性映射→画面一点没变**(0731a视频=没扶平)。写`apply_untwist_render_110744.py`: pre-align ply反归一化→colmap帧→刚性映射world帧→施同一warp场→逆变换回存`gs_vggto_colmapjoint_110744s2_lvl.ply`(软链改指,bak保留; quats用共轭旋转Rz⁻¹R_wRz)。LSD零交叉在扶平渲染上复测: 柜166.6/微168.7不变, 白桌LSD被溅射打败(沿用真图实测0°)。
**产出**: `outputs/lab_mission_110744_zed_front_perp_lvl_geo0731b.mp4`(317帧): 巡航帧全实验室水平线平行虚线✓ 显微镜台沿横平✓ 柜YES/微YES/白桌NOT-CONFIRMED(VLM如实判no, 绿X压桌上几何对, 白桌confirm老毛病=类别歧义)。
**结论**: ⭐⭐**改几何必须想清哪份文件进哪条渲染路径**: CaptureFPV=pre-align ply+npz映射(视频画面), world ply=占据栅格/BEV, 两者要同步扶平; quats共轭旋转别忘。LSD零交叉在干净渲染上可复现(166.6/168.7两版一致=方法稳)。
**下一步**: 用户确认0731b版(重点看地面)→P1建图粒度/真机部署。
**卡点**: 无。

### 2026-07-31(续2) —— 世界局部扶平(untwist)+LSD零交叉实测front+站距0.85m

**做了什么**: 用户判perp版"离得太远+还是没垂直桌子平行线"。两病根: ①**世界本身空间渐变扭曲**(逐waypoint测地面: 主区3.6~4.2°@az19°, 南端铁柜区0.25~1.9°——07/24版untwist没扶干净)→新写`apply_untwist_110744.py`(逐waypoint局部地面法线→最小旋正→空间kNN平滑→逐高斯近邻混合旋绕局部锚点, 同旋quats), 产物`gs_vggto_..._world_lvl.ply`+npz/targets/clusters同步变换(全有_PRELVL0731备份); ②**显微镜front的PCA方位角被邻物污染偏15°**(-177错)→⭐**LSD线检测零交叉实测**: 水平线倾角随yaw线性过零(柜166.6/微168), 白桌用真采集帧LSD≈0°→**front=柜167/微168/桌0**。站距1.2→0.85m。
**产出**: `outputs/lab_mission_110744_zed_front_perp_lvl_geo0731.mp4`(319帧): 三站0.85m, ALIGN bearing-front≤2°/横向≤0.03m, VLM confirm全YES, 台沿横平。
**结论**: ⭐台面/地面厚板SVD法线不可作微调依据(多层结构折中, 旋正后重测自相矛盾); **视觉零交叉(LSD线倾角vs yaw)才是"正"的直接度量**; PCA边方向须防邻物污染(显微镜93°错, 实测边~78°)。untwist残余: 台面拟合仍1~2.3°(重建非刚性扭曲, 平滑纠正是极限)。
**下一步**: 用户确认perp_lvl版→P1建图粒度/真机部署。
**卡点**: CG traj.txt未随untwist变换(重跑三角化才需要, 已记脚本docstring)。

### 2026-07-31(续) —— "正"=垂直于桌面平行线: front改点云PCA边法线, 三站全正+VLM全YES

**做了什么**: 用户判0731版"还是斜的"并定义"正=垂直于桌面的平行线"→0730检测射线圆均值front全错的原因坐实: **采集轨迹沿y走在房间西侧, 检测弧全在斜侧, 最大框射线均值继承斜视**(柜-135.7/微163.3/桌-38.1分别偏59/14/32°)。修法=点云紧裁PCA: 目标GT点附近小半径+台面/立面高度带→长轴=平行线方向(柜75°/微台93°/桌84°, 紧裁稳定ratio4~25), front=长边+90°取机器人侧→**柜165°/微-177°/桌-6°**。渲染验证三视角平行线横平后写`lab_mission_targets_zed_front_perp.json`重跑。
**产出**: `outputs/lab_mission_110744_zed_front_perp_geo0731.mp4`(332帧): ALIGN三站bearing-front≤0.5°/横向≤0.01m, **三站VLM confirm全YES(白桌正视图干净也YES了, 上版NOT-CONFIRMED)**。
**结论**: ⭐front正确求法=几何边法线(点云PCA), 不是检测射线统计(检测弧有偏就全偏); 紧裁(半径≤0.7m+窄高度带)是PCA稳定关键, 大裁必被邻桌/墙污染(白桌1m裁给出107.6°假边)。
**下一步**: 用户确认perp版→P1建图粒度/untwist/真机部署。
**卡点**: 无。

### 2026-07-31 —— 正面向量标注两bug修复(用户判"向量指的正对方向都不对")

**做了什么**: 抽0730定稿视频三站帧坐实: ①FPV青箭头从目标沿front_deg(=观察方向, 相机→目标)画→扎进目标内部, 应画正脸外法线front+180°; ②BEV品红箭头把世界系(cos th,sin th)直接当heading-up ego系分量, 少ego-yaw旋转→三站歪的方向各异。修lab_mission.py panel绘制(FPV取反/BEV增量=fwd-cos(th-yaw),right+sin(th-yaw))+输出固化_geo0731后缀。
**产出**: `outputs/lab_mission_110744_zed_front_geo0731.mp4`(316帧, BEV箭头两站都从黄星直指红ego点, FPV箭头指出柜面朝向观察者; 任务行为与0730完全一致=ALIGN收敛/白桌NOT-CONFIRMED如实)。
**结论**: 只是标注层bug, 运动层(站位/ALIGN/yaw_err≤1.1°)无错; front_deg语义=观察方向(机器人yaw), 画"正面"向量必须取反, BEV画世界向量必须先转ego系。
**下一步**: 用户确认0731版→P1建图粒度/untwist/真机部署(同0730断点)。
**卡点**: 无。

### 2026-07-30(续2) —— 正对闭环ALIGN+全向量标注定稿; ⚠三角化窄基线教训+我520/640检查bug

**做了什么**:
- **用户连否"还是歪/没对正"→逐层锤**: ①"正对"真义=视线垂直目标正脸(0722用户定义桌沿横平), 只打目标方位角不够; ②正面方向取自**节点最大检测帧的射线方位加权圆均值**(检测器在正脸视角框最大最稳, RAG-native): 铁柜-136°(门冲东北, 推翻墙法线180°假设)/显微镜163°/白桌-38°(与老管线手挑#214同向); ③cruise 0.15m容差→站位偏离正面轴, 打bearing≠front实测偏7~12°→**align_front闭环**: 转front yaw+位置投影到正面轴横向残差全向vy步进消除(纯几何零bbox), 三站收到横向偏移≤0.04m/bearing-front≤±1.8°。
- **用户"把正对用向量标出来"→panel全标注**: FPV=中心十字(黄)+目标3D投影(绿X)+正面向量(青箭头)+人工水平基准线(白虚线)+front/yaw_err数字; BEV=目标星+正面向量箭头。
- **⭐三角化教训(诚实研究结论)**: 用户判"标注都不准"深挖发现——铁柜检测来自单向窄弧(离散仅7.6°), 射线近平行→三角化点在深度方向系统性偏移(SW偏0.5m, 残差4.9cm只是自洽不是准); 穿刺法(射线×簇)也被过合并簇污染(38~73cm)。**三个目标当前建图粒度下图给不出干净中心点; zed GT(自动管线, 用户目检确认全对)胜出**→终版=GT点+检测正面方向。
- **⚠我的检查bug(道歉记录)**: FPV实际宽640(视频1040=640FPV+400BEV), 我抽帧按520裁→"绿X偏右15%"等判断全错, 标注本身是对的(单元测试: 正前方投影=中心±0, 左右±0.3m→u±39px✓)。**教训: 量产物画面尺寸先看合成逻辑别拍脑袋**。

**产出**: `outputs/lab_mission_110744_zed_front_geo0730.mp4`(316帧, 三站yaw_err≤1.1°, 绿X全压目标, 白桌站VLM如实NOT-CONFIRMED=画面杂判no) / `outputs/lab_mission_targets_zed_front.json` / `scripts/cg_triangulate_nodes.py`(含front_yaw) / align_front进lab_mission.py。

**结论**: 正对三要素(站位上轴+朝向front+目标居中)闭环达成; 窄基线三角化不可信(离散角<10°时), 深度须多方向观测或外部源。

**下一步**: (待定)①建图粒度P1(过合并拆 node130/20 区域); ②铁柜侧棱可站远到1.6m或修正门法线角; ③真机部署。

**卡点**: 无。

### 2026-07-30(续) —— 场景图可视化三轮 + 用户确认GT全对 + 节点多视三角化精化(GT-free出片)

**做了什么**:
- 用户二判视频"还是歪不对准"→实锤两病根: ①auto-roll逐位置查表=视角左右摇(用户明示"就平着呗"); ②打的是节点bbox中心, 而节点过合并(铁柜node130吞了扫地机器人+柜子, bbox 1.9×1.2m; 白桌node20是地板杂簇)。
- **场景图可视化三轮**(用户"看不清轮廓"连否两版): v1稀疏实例色点云环绕=彩纸屑; v2稠密3DGS底座+全节点=仍乱; v3只高亮3目标/三站锐利渲染+投影=`cg_stations_check_0730.mp4`(点簇)与`cg_stations_bbox_0730.mp4`(open3d官方OBB线框, 别手推棱边会连错)→**用户确认zed GT X全对, 节点框普遍过大错位=过合并是病根**。
- **⭐用户点醒"crops那么好为啥不用"→节点多视三角化**(`scripts/cg_triangulate_nodes.py`): 节点存了20次检测(image_idx/xyxy/conf=crops来源), bbox中心射线(该帧c2w+内参)最小二乘求交+0.15m修剪。vs用户确认的GT: 显微镜50→28cm/白桌88→38cm/铁柜57cm(=物体中心vs GT表面点参照差); 射线残差3.4~9.7cm=检测本身干净, 且残差可作节点质量诊断。渲染验证GT与三角化点落同一物体。
- **出片**: `lab_mission_110744_rag_tri_geo0730.mp4`(304帧, roll=0平走/三角化targets/站位avoid互斥0.6m防两站撞点/无bbox伺服): 铁柜0.00°居中/显微镜-1.07°正中/白桌-3.04°(场景本身歪, 世界没untwist), 三站判断式confirm全YES, 全程GT-free。

**产出**: `scripts/cg_triangulate_nodes.py`+`outputs/lab_mission_targets_rag_tri.json` / `cg_scenegraph_bev_mission_0730.png`(任务区节点编号图) / `cg_scenegraph_orbit{,_dense}_0730.mp4`(被否两版留档) / `cg_stations_{check,bbox}_0730.mp4` / `lab_mission_110744_rag_tri_geo0730.mp4`。

**结论**: 三角化是RAG-native的节点精化正解(图自己的检测数据, 无外部拐杖); zed GT(自动管线产出)经用户目检确认可当验证基准。

**下一步**: (待定)①白桌站-3°静态斜→untwist实验室世界扶平(坐标变, targets/clusters重变换); ②建图粒度P1课题(合并阈值太激进=节点过大); ③真机部署。

**卡点**: 无。

### 2026-07-30 —— 用户否掉0729版RAG闭环(假confirm+歪+贴脸)→掰回RAG原生几何链(零bbox)

**做了什么**:
- **用户判0729版`lab_mission_110744_rag.mp4`效果差**: 复核实锤——①站位姿态继承绕看采集帧(带roll/pitch)=画面歪, 三站err_cam_to_gt_deg=-19.2/+17.3/-17.6°全没正对; ②显微镜站位离节点0.11m=贴脸全帧糊; ③confirm=bbox存在性→铁柜站对着白桌也CONFIRMED(假YES); ④bbox大多帧没画上+confirm文字跨站残留, 不可审核。**用户点破架构漂移: 场景图RAG本是替换几何部分的(locate_target/bbox伺服/反投影), 0729深夜为让伺服收敛把旧拐杖捡回来=走回头路。**
- **掰回`scripts/lab_mission.py`**: 删`_vlm_bbox`/`_turn_to`/`servo_approach`(bbox三件套)+`--standoffs`(采集位姿json)+`standoff_at`墙法线(手搓几何)+SRC源帧confirm。`find_standoff`重做: 位置=采集轨迹上最正对目标的帧(on-path保渲染锐, 采集yaw只用于选位置打分), **朝向=atan2(节点-站位)纯几何解析**, 距离约束[0.8,2.0]m偏好1.2m。confirm=对实时水平渲染图(roll=0)做判断式yes/no, 判no如实NOT-CONFIRMED不挽尊。每腿清空VLM标签防跨站残留。
- **重跑**: 三站站位距目标1.07/0.91/1.32m(原0.11贴脸消除), face收敛残余-2.3/+2.1/-1.5°, 三站VLM判断式confirm全YES(这次是对着水平正视图判的), 7.65m/319帧。逐帧目检: 画面全水平、铁柜站正对白玻门柜、显微镜站正视桌面设备、白桌站对小桌, 无贴脸无歪。

**产出**: `outputs/lab_mission_110744_rag_geo0730.mp4` / 改版`scripts/lab_mission.py`。⚠首跑曾覆盖旧`lab_mission_110744_rag.mp4`(0729坏版, 反正被否, 但覆盖本身违反版本化铁律, 已在脚本输出名固化_geo0730后缀)。

**结论**: **RAG原生链路成立——节点3D坐标直接出站位+朝向, 环里零bbox**, 效果立杆见影(歪/贴脸/假confirm三病同治)。face()纯几何朝向收敛±2.3°内, 节点坐标0.5~0.6m误差在1.2m站距下目标仍居中区域, 判断式confirm够用。

**下一步**: (待用户看视频确认)若认可→此链路即真机部署模板(对齐`docs/真机部署路径_场景图RAG_0729.md`); 遗留=铁柜/显微镜节点仅1.1m间距的语义消歧仍靠判断式粗确认(P1空间消歧课题未动)。

**卡点**: 无。

### 2026-07-28 —— ConceptGraphs P0: 环境+数据+20帧检测验证跑通(场景图RAG方向)

**做了什么**:
- 续0727方案, P0官方ConceptGraphs落地。cg env补齐: Grounded-SAM@a4d76a2(GroundingDINO需`--no-build-isolation`)+segment_anything+RAM(Recognize_Anything-Tag2Text软链`ram/`进GSA根+fairscale)+gradslam@conceptfusion+chamferdist。坑: supervision必须0.14.0 / nvidia-cudnn-cu12装坏需force-reinstall / GDINO的bert-base-uncased走`HF_ENDPOINT=https://hf-mirror.com`。
- 三权重下载齐(GSA根): ram_swin_large_14m.pth(5.6GB)+groundingdino_swint_ogc.pth+sam_vit_h_4b8939.pth。
- 数据零新采集: `scripts/cg_prep_110744s2.py` 把110744s2(510帧)转Replica格式——cameras.npz w2c逆→c2w写traj.txt, ZED点云npz z通道(mm)→uint16深度PNG(scale1000), 510/510帧全有深度; dataconfig `lab/lab110744s2.yaml`。
- **20帧检测小验证(方案第七节第1步)通过**: RAM开集打标52类含microscope/cabinet/table/workbench/drawer/equipment/robot——家居→工业域差担忧解除, 目检框+SAM掩码质量正常。GPU2上~4.25s/帧。

**产出**: `outputs/cg_lab110744s2/lab1/`(results/+traj.txt+gsa_vis_ram_test20/+gsa_detections_ram_test20/+gsa_classes_ram_test20.json) / `scripts/cg_prep_110744s2.py`; 全量检测(stride2, 255帧)已后台开跑。

**结论**: 官方链路在我们数据上work, 域差不是问题; 可按方案继续P0后半。

**下一步**: 全量检测→incremental mapping(cfslam_pipeline_batch)→caption(LLaVA或换自研Infinite VLM)→build_scenegraph(openai调用→本地Qwen)→P1 RAG挂载lab_mission对比zed.json真值。

**卡点**: GPU全卡繁忙, 只能插空用GPU2(~32G空闲); LLaVA-7B老commit权重还没下。


### 2026-07-27 —— VLM时序慢脑(多帧+长记忆)走廊落地 + 能力边界实测 + 场景图RAG方案研究

**做了什么**:
- **能力边界探测**(走廊nav_ds_real实测, 自研Infinite VLM 10.0.0.18): ✅多帧ANY检测(K≤6)可靠(prompt须"visible in ANY"式) ✅window_last末帧聚焦可用 ✅文本任务记忆注入可用 ⚠pick-closest时序定位只近似 ❌相位3分类全答LEAVING ❌16帧长程枚举复读退化+37s ❌**流式=服务器不支持**(实测TTFB=总时长一次性返回, 替代=短答先出)。
- **离线评测** `scripts/eval_temporal_slowbrain.py`(219帧×3模式657查询): single精确0.344/召回0.824 → **window_last精确0.402(+5.8pt, FP 80→61)**/召回0.804; window_any无提升。分歧帧目检: 压掉的主要是"远处可见泵"真误报(对), 但上下文污染也会新增少量误报。判据措辞必须和单帧同义, 否则问法本身引入yes偏置(踩坑)。
- **接入闭环**: 新建`scripts/vlm_temporal_slowbrain.py`(TemporalSlowBrain: 滑窗/detect三模式/classify_facing带记忆/stats); `run_inspection.py`加`--vlm_temporal`(geo_vlm面向分类单帧→滑窗+记忆注入, 结束打印慢脑统计)。
- **走廊demo**: `--geo_vlm --vlm_temporal --unify_occ --occ_thr12 --speed0.7` → **1066步280s 8/8全检视DONE**, 慢脑8次调用均1.21s, 记忆8条全确认(返程复巡是既有架构行为, 记忆未改判决=4站全真泵)。
- **场景图RAG方案研究**(用户0724知识库两论文, 新方向=3D场景图替代几何部分): 查证**ConceptGraphs=ICRA2024**(笔记写CVPR2025有误)+**OpenFunGraph=CVPR2025 Highlight**; 读官方repo README(posed RGB-D输入/Grounded-SAM+LLaVA老commit/GPT-4建边)。写`docs/场景图RAG替代几何_方案研究_0727.md`: 替locate_target/lab_locate_clusters/device_clusters, 保留A*占据层; 110744s2=510帧posed RGB-D正是标准输入; P0官方ConceptGraphs跑通(GPT-4→本地Qwen)→P1 RAG挂载lab_mission→P2 OpenFunGraph功能边; 风险=家居→工业域差先20帧小验证。

**产出**: `goalnav/runs/inspection_geovlm_temporal_0727.mp4` / `scripts/vlm_temporal_slowbrain.py` / `outputs/vlm_temporal_eval/`(jsonl+summary) / `docs/场景图RAG替代几何_方案研究_0727.md`。

**结论**: 时序窗口对检测是真实但温和的提升(+5.8pt精确率), 非银弹; 记忆注入链路已通, 真收益要等场景图RAG(检索式记忆后端)。VLM时序能力边界已摸清, 不可用项别再试。

**下一步**: (待用户确认)场景图P0=官方ConceptGraphs跑110744s2。**卡点**: 8卡全占(GPU3挤22GB跑的demo); 代理7897死/17897活。

---

### 2026-07-21(晚) —— 室内110744断点续跑: 全帧剔糊帧重训(f-827) + capcam三方对比定论=stride2最佳

**做了什么**:
- **接断点**: 会话掉线时 `110744f`(全帧1020剔运动糊帧后剩827干净帧)的3DGS训练被杀在step3400、`.ply`未存。确认进程已死、session目录(`outputs/vggto_colmapjoint_110744f/`,827软链帧+cameras.npz+recon.ply软链复用主重建)完好→**GPU3按hqbest配方重新拉起训练**(`ITERS=20000 SH3 UPSAMPLE1.0 DEPTH_SUP DEPTH_W0.4 ANTIALIAS REG_SCALE0.01 APPEARANCE MASK_OVEREXP`)→致密N=1.076M剪枝后**531k高斯**(loss0.08~0.25)→`align --crop`地面1.73°(827wp span6.5×1.8m)。
- **⭐capcam三方对比判定采用哪版**: 写`scripts/_cmp_lab_versions.py`, **用原始逐帧真外参`_Vs[j]`渲染**(⚠踩坑: 先用`_Vs_smooth`平滑外参→110744是绕看转身采集, ±7帧平滑把差异大视角平均→假糊, 改原始外参才真实)。stride2缺备份norm.npy→复现训练归一化逻辑(相机中心均值+均距)重算, 与备份npz的center/s_norm完全一致=复现正确, 搭临时s2 session(强制sparse/0的510帧模型绕开CaptureFPV优先选sparse/1的76帧小模型bug)。

**产出**: `outputs/gs_vggto_colmapjoint_110744f{,_world}.ply`(531k, 实验记录非交付) / 对比图`outputs/recon_verify/lab_110744_stride2_only.png`(stride2锐) + `lab_110744_full_vs_f_rawV.png`(full/f都糊) / 脚手架s2 session(软链+重算norm.npy)。

**结论**: ⭐**stride2(596帧COLMAP→510注册, 1.1M高斯loss0.05)渲染明显最锐**(桌/黄设备/椅/墙照片清晰贴合真实图); full-1020(510k)和f-827(531k)全程发糊, f比full略好但都够不到stride2。**全帧回退真因=位姿质量**: stride1的1020帧COLMAP位姿更噪(帧多+快速转身糊帧→BA更难→漂移)光度一致性差=糊, 剔糊帧重训救不了; 596帧稀疏COLMAP位姿更紧才是stride2锐的根本→**加帧反而害位姿**。**决策: 室内实验室交付用stride2**(`gs_vggto_colmapjoint_110744_stride2_bak.ply`等)。

**下一步**: ①要跑巡检把canonical指回stride2(或用已搭好的s2 scaffolding); ②建device_clusters/nav_ds(绕看式采集→A*非走廊直行要重想); ③sim相机匹配采集pitch(capcam测≈-24~-27°, 是waypoint level渲染发白雾主因)。**卡点**: 无(重建线定论=用stride2)。

---

### 2026-07-22 —— stride2交付管线全跑通: canonical切换 + 锐利走查视频 + 仿真闭环冒烟

**做了什么**:
- **canonical 110744切到stride2**: full-1020差版归档`outputs/_exp_full1020/`, `gs_vggto_colmapjoint_110744{,_world}.ply`/`sim_world`/`.norm.npy`/`_colmap_joint_110744`软链到stride2备份。
- **建`110744s2`完整交付session**(避开canonical的vggto帧目录冲突: 该目录已被全帧1020版覆盖+f实验软链复用其recon.ply): `build_colmap_session.py 110744s2`(从stride2 sparse/0的510帧重建frames_zed) → `build_nav_ds.py 110744s2`(N=510)。搭s2 scaffolding时踩坑=CaptureFPV遍历sparse/2,1,0优先选到stride2的sparse/1(76帧小模型), 建`_colmap_joint_110744s2/sparse/0`只含sparse/0软链强制用510帧主模型; stride2丢的norm.npy按训练归一化逻辑(相机中心均值+均距)重算, 与备份npz的center/s_norm完全一致。
- **锐利capcam走查视频**`render_best_walkthrough.py`(PLY=s2 pre-align ply, MODEL=s2 sparse/0真外参, NFRAMES300 SS2.0)→`outputs/walk_lab_110744_stride2.mp4`。
- **仿真闭环冒烟**`test_env.py --skin_session colmapjoint_110744s2`(需PYTHONPATH=third_party/discoverse): 占据自由70%, A* start(0,0)→goal(0.35,-4.65), pursuit 28步reached零碰撞。

**产出**: `outputs/walk_lab_110744_stride2.mp4`(300帧走查, 工作台/货架/设备/窗全清晰) / `nav_ds_110744s2/` / `vggto_colmapjoint_110744s2/`(510帧) / `recon_verify/{walk_lab_sample,testenv_fpv_sample,lab_110744_scene_topdown}.png` / canonical 110744软链→stride2 + `_exp_full1020/`归档。

**结论**: **stride2重建可导航FPV锐利=交付达成**(走查视频=工作台/货架/黄设备全清晰, 仿真闭环通零碰撞, 在轨level相机FPV没严重白雾)。场景性质=中轴窄带(x跨1.4m/y跨5m/转身累计1436°≈4圈)边走边转身看两侧。

**下一步**: VLM巡检demo(需定target设备+巡逻式沿y轴中轴设站转向侧面; build_clusters_stations假设x单调要泛化y轴; 目标候选=黄色测试设备/工作台很显著)。**卡点**: 无。

---

### 2026-07-16(晚) 补: 111450锐皮肤(1.2M)+demo修复
- 看门狗等到GPU4空自动锐训(默认致密化→**1.2M高斯**,world ply 304MB)+align+重跑demo。
- **两坑已修**: ①重跑align坐标漂移(x峰27.54→29.15)→重建device_clusters对齐新坐标; ②1.2M占据太密(occ_thr12挤偏导航漏检)→occ_thr提到40(按密度3.4倍缩放)→导航恢复。run_inspection加`--occ_ply`(占据/皮肤解耦备用)。
- **结果**: `goalnav/runs/inspection_111450_sharp.mp4` 往返检视4处190s正常。
- **锐度结论**: 358K vs 1.2M公平对比(同前视on-path帧)差别很小, 顶/侧喷溅拉丝两版都在=off-axis采集天花板(单程前视), 加高斯救不了(印证0626)。VLM判定用真图不受影响。
- **教训**: 重跑align后必重建依赖坐标的clusters; 换皮肤密度必同比调occ_thr。



## 07-29（周二）ConceptGraphs 全链收尾 + RAM fallback 补丁 + RAG 冒烟

**07-29 深夜续（全链稀疏图救援实测 = 衰减实验终版）**
- s5 显微镜：CLIP top1=2.57m → VLM 精排连否 2 个错节点 → node35=**1.18m**（救援成功；真·最近节点不在 CLIP top5=合并粒度随机性）。
- s5 铁柜：修正地标后关系规则 → **0.69m**（2.97→0.69 级联修复）。s10：VLM 假阴性频发（真显微镜判 no）=节点太糙救不回，**建图密度是硬下限**。
- **最终部署结论（全链）：s2 级全目标≤0.9m 可靠；s5 级（102帧≈3分钟巡逻）≤1.2m 可用+VLM 到场兜底；s10 不可行。**



**07-29 深夜（视觉伺服收敛 + 巡逻密度衰减实验）**
- 对准攻坚收敛：根因=VLM bbox 在 off-path 糊渲染上不稳；解法=**站位改用场景图节点的最佳观测采集位姿**（节点最高 conf 检测帧→traj 取位姿，on-trajectory 渲染锐）→ servo 可靠收敛。confirm 改 bbox 存在性（检测>判断）。最终 `lab_mission_110744_rag.mp4`：铁柜 6 轮收敛 cx=0.51 / 显微镜 -2.9° / 白桌 -1.0°，三站实时帧 bbox 全 CONFIRMED。
- 真机部署路径研究 `docs/真机部署路径_场景图RAG_0729.md`：VLM 的病是仿真代餐特有；真机坐标系更干净（巡逻/任务同 SLAM 会话）；唯一真风险=SLAM 位姿漂移。
- **巡逻密度衰减实验**（模拟真机，stride2/5/10 各建图，`scripts/cg_decay_eval.py`）：节点 185→46→9；**几何覆盖三密度全≤0.34m（鲁棒），语义特异性先死**（s5/s10 图"microscope"标签 0 票消失）；检索误差非单调。**部署结论：稀疏巡逻够找区域、不够语义检索；s2 级（~250帧）可靠，s5 临界；真机=兴趣区采密+VLM 到场确认**。
- 环境坑记：user-site transformers 被升级污染 cg env（RAM 需 ≤4.44.2）；GPU0/1 被灌满 OOM 换 2/3；mapping 需目录内 PYTHONPATH=.。



**07-29 傍晚（⭐闭环跑通）**
- 场景图 RAG 接入 lab_mission（架构不变，仅 WHERE 坐标源替换，`--targets` 参数化，视频版本化）。
- 三站坐标全 GT-free：microscope=node148(CLIP top1,0.50m) / iron_cabinet=node130(VLM-yes+显微镜邻域,0.55m,tag被RAM漂移成microwave) / pipette_table=node20(白色像素占比top1,0.60m)。
- **`outputs/lab_mission_110744_rag.mp4`**（335帧，8.69m）：4腿全通，三站慢脑 VLM confirm 全 YES，抽帧目检三站 FPV 对准目标。
- ⚠ 环境修复：~/.local 的 torch/torchvision/scipy/matplotlib/hf_hub 曾被清，已装回（torch2.5.1+cu121 等，运行姿势见 memory）。
- 断点：真机 SLAM 位姿建图验证；"黄桌前白桌"关系消歧未硬编码（本次白属性探针）。



**07-29 下午续（三层检索验证完）**
- 更深诊断：过合并在建图增量阶段（post-merge 阈值 0.7→0.9 零变化，192→192）；几何覆盖完备（三目标 GT 全有 pcd_min≤0.22m 节点）；**真瓶颈=开集打标词表漂移**（铁柜位 RAM 叫 microwave/oven，白桌位叫 tool/computer，被桌上物带偏），纯文本查询救不了。
- 三层检索 `scripts/cg_rag_vlm_rerank.py`：①CLIP 粗排 top10（text+clip 特征+RAM 标签加成）②**VLM 存储视角精排**（候选 crop 画红框问自研 VLM yes/no，30 候选判否 12，有区分度）③**空间关系消歧**（铁柜=距显微镜地标最近候选 → 2.11m→**0.69m**）。
- 最终成绩：microscope **0.50m**✅ / iron_cabinet **0.69m**✅（显微镜当地标）/ pipette_table 🟡~1.5m（"黄桌"地标不可靠，覆盖节点 0.63m 存在，留到场 VLM 确认）。
- 产物：`cg_rag_vlm_rerank_0729.json`、`cg_rag_smoke_v2_185_0729.json`、灰底高亮版可视化（用户认可）。
- 下一步：接 lab_mission 状态机（前两站 RAG 直跑，白桌站导航到候选区+VLM 环视确认）。



**做了什么**
- 定位"昨晚断了"真相：0728晚全链实际已跑完（255帧检测→mapping 959obj合并185→caption→LLM打标），烂尾在最后一步——自研VLM做GPT-4式JSON整理，185个object仅~27个有效tag（75缺object_tag字段/75回invalid/8解析失败），剪枝后场景图只剩10节点且LLM幻觉标签（refrigerator/coffee machine）。
- 修复：`build_scenegraph_cfslam.py` 加 [PATCH ram_fallback]——LLM tag无效时回退该物体多视角RAM打标多数票（纯确定性，不再依赖VLM文本能力）。旧10节点图备份 `_10obj_bak`，重跑出 **185节点全保留**（microscope×4/cabinet×1/table×13）。边构建阶段（~184次VLM关系查询）对检索无用，kill跳过。
- RAG冒烟 `scripts/cg_rag_query.py`（CLIP ViT-H-14 text_ft+clip_ft均值，真值=lab_mission_targets_zed.json，同PREUNTWIST坐标系）。

**结论（照实）**
- microscope top-1 命中，距真值 **0.50m**（xy仅0.31m）→ 区分度物体可用场景图替代手搓几何定位。
- iron_cabinet top-1 偏2.11m，但microscope第2名 node124("equipment") 距铁柜真值仅0.44m——铁柜被检出但RAM打标叫equipment；铁柜与显微镜物理同簇（相距0.6m）。
- white table 失败（3.31m）= 类别歧义（13张桌子CLIP无区分度），需空间关系消歧或到场VLM确认，是P1课题非bug。
- 产物：`outputs/cg_lab110744s2/cg_rag_smoke_0729.json`；场景图 `sg_cache/map/scene_map_cfslam_pruned.pkl.gz`（185节点）。

**下一步（P1）**
- 歧义目标消歧（空间关系查询扩展/两阶段tag过滤+CLIP/到场VLM确认）。
- 接 lab_mission 状态机：RAG节点坐标替换 locate_target 手搓几何 WHERE。
- 可选：caption/edges 换本地 Qwen3.5-4B 提质（自研VLM纯文本JSON能力废，已实测）。


## 2026-07-16 —— 室外场景选型纠偏(105731弃→111450) + 111450铁网巡检demo全链跑通 + 锐皮肤看门狗

**做了什么**:
- **接断点"105731跑geo_vlm+换目标电柜"→诊断否决**: 画对齐轨迹俯视(`_viz_scene.py`)+密集contact sheet(`_contact_105731.py`)铁证105731 ①全程有黑衣操作员在镜头前挥手(重建污染+演不了发现设备) ②无明显配电箱(主体是高架桥下停车场+车+黄线) ③对齐轨迹紧凑(6.5m/12.9m/yaw累计)。**结论: 105731演不了换目标电柜**。
- **⭐用户两处纠正**: ①室外巡检目标不是配电箱, 是**电线杆右侧的铁丝网**; ②先探111450(同场景另一份)+剔掉有人帧。
- **✅改用111450全链重建**: contact sheet证人只在开头f0-95, 之后~1140帧干净(高架桥柱+停车场+黄线马路+右侧铁丝网)。`_recon_single_cam_seq.py`加`RECON_START/END`裁污染帧→窗口[96,1235)stride2取570张→COLMAP **566/570注册(主模型sparse/2)**→build_colmap_session。
- **⚠3DGS遇GPU竞争OOM**(高斯涨到1.3M撑爆26GB空闲)→重训加`GROW_GRAD2D=0.00035 REFINE_STOP_RATIO=0.35 expandable_segments`封顶358138高斯(loss0.09)→align(566wp **span28.4×9.1m=大线性场景**,非105731紧凑)。
- **✅参数化+建输入**: `run_inspection.py`加`--nav_ds/--clusters`(默认仍走廊,向后兼容); `build_nav_ds.py`(泛化)建`nav_ds_111450`; `build_clusters_stations.py`(室外点云无干净周期立柱=检测不可靠→沿路等距站位)建4站位side=-1(铁网在行进右侧)。
- **✅关键验证**: env+A*冒烟通(occ free99%,A*93点0→27.5); 采集相机大量侧看→RealFrameLookup对面向query返回**真正面向铁网真帧**(站位17=电线杆+配电箱+铁网+车); VLM(10.0.0.18 robo2vlm现0.6s)对铁网答Yes。
- **✅跑geo_vlm**: 往返检视4处step~490 DONE 118s, 面板`VLM@:分类:是目标`+记忆完成记录, 换prompt(泵→铁网)即换目标=机制室外复现。
- **皮肤偏糊(358K)**→用户选"等GPU空训最锐~1M"→写看门狗`_watch_train_sharp_111450.sh`(nohup后台轮询, 空卡≥42GB自动锐化训练+align+重跑demo→`inspection_111450_sharp.mp4`+写`_sharp_111450.done`)。备份358K皮肤防覆盖。挂每小时cron自动验证。

**产出**: `goalnav/runs/inspection_colmapjoint_111450.mp4`(铁网巡检demo,358K皮肤版) / `outputs/gs_vggto_colmapjoint_111450{,_world}.ply`+`_358k`备份 / `nav_ds_111450` `device_clusters_111450.npy` / 新脚本见上。

**结论**: 室外铁网巡检demo跑通=**"换指令→检视不同设备"机制在室外新大场景(28m)复现**; 111450侧看采集正合侧向巡检。**下一步**: 等GPU空出锐皮肤(看门狗自动)。**卡点**: 机器load410 8卡全占, 锐训要等卡。

---


## 2026-07-14(晚) —— 服务器修复后续接: gate训满+发现门不可复现加正则修复 / 191715 crab验证 / gate接进巡检闭环

**做了什么**:
- **191715 crab 3DGS 收尾**: 发现3DGS其实早跑完(step19999出`gs_vggto_colmapjoint_demo191715.ply` 426万高斯SH3,记忆"杀在19000"过时)。补跑`align_gs_world.py`(330 waypoints,span3.5×1.2m横向范围)+改`render_best_walkthrough.py`让MODEL可env覆盖→渲`walk_191715_crab.mp4`(240帧)。目检横扫全程锐利=**crab采集重建干净, 印证"crab失败"真因是单相机bug不是crab不行**。
- **⭐gate训练重跑发现真bug**: 重跑满240ep时同初始同数据却收敛到**不同门值**(一次理想分离/一次两门齐塌),两者loss几乎一样→**门被trajectory_loss欠约束(有goal驱动场景冗余),收敛靠`np.random.permutation`种子运气**,上次"漂亮门值"有运气成分对论文机制卖点不可接受。**修复=固定种子+加门正则`--gate_reg=0.3`(两层decoder监督:平移→门开/纯转向→门关)**。重跑ep005即到`前向0.986转向0.028`稳定,终门`前向1.000转向0.000`,loss0.88。
- **verify_turn(加env CKPT/OUT)验证**: `verify_turn_gate.png`证**±90°原地转3帧×左右两向全干净**(末点塌原点,yaw±88~89°),含之前turn-only"左转全废"的走廊帧889。⚠±45°是OOD不可靠(训练|dyaw|∈[54,90])。
- **⭐gate接进`run_inspection.py`闭环(几何--no_vlm)**: 载模型改`goalnav_gate.pt`+`dd_traj`改`status_gh`(目标点+朝向)+**混合TURN**(大角度gate-DD原地转+小角度控制器收尾)。踩两坑并修:①`traj_to_control`对纯旋转轨迹选点错→wz≈0卡死→`turn_wz`用DD末端yaw取角速度;②转角>90°是OOD符号翻转走长路→喂DD前夹`clip(e,±π/2)`。**结果4/4设备全检视step379到DONE,面板确认src=DD(原地转正)+纯旋转,转正后FPV见左墙蓝泵**。

**产出**: `goalnav/runs/goalnav_gate.pt`(正则版终门1.0/0.0)+`_train_gate_reg.log` / `outputs/gs_vggto_colmapjoint_demo191715_world.ply`+`walk_191715_crab.mp4` / `outputs/verify_turn_gate.png` / `goalnav/runs/inspection_gate_hybridturn_0714.mp4`。改动: `train_goalnav_gate.py`(seed+gate_reg) `verify_turn.py`(env CKPT/OUT) `render_best_walkthrough.py`(env MODEL) `run_inspection.py`(gate+status_gh+混合TURN+turn_wz+夹紧)。

**结论**: 办法C命令门控**机制成立且可复现**(正则锁定,非种子运气)=论文可控性卖点站得住;巡检全链CRUISE-DD→TURN-DD→SCAN→TURN_BACK×4闭环通,DD快脑统一驱动走+转。crab采集突破单相机bug天花板。

**下一步**(可选): ①VLM模式`--vlm_realimg`跑(需VLM服务10.0.0.18)验VLM发现蓝泵→DD转向全链;②191715 off-path定量对比证突破拖影天花板;③gate接进sim_fullchain全链真闭环。**卡点**: 无(几何闭环已通)。

---

## 2026-07-07(下) —— 拷入当前实时工作区 + 运控/采集对比总结 + VLM↔DiffusionDrive接缝文档

**做了什么**:
- **拷入** 机器人当前实时ROS2工作区到 `robot_control_current/`(rsync,403M)。三代理并行精读运控/采集/文档。
- **对比总结** `docs/当前机器人运控与采集_对比总结_0707.md`:①运控动作接口`/svtrobot_cmd`(vx,vy,wz)+`svtrobo_controller.py::publish_action`与DiffusionDrive输出**完美对齐**;②**⚠️`get_observation()`只有16维本体感(舵角/舵速/力矩/轮速),无相机/BEV/位姿**→感知obs要自建;③采集格式与`run_preprocess.py`~95%吻合(images/zed jpg+pointcloud npz mm+cmd jsonl+2Hz)但**缺.svo/positional tracking/odom/tf/曝光锁/task_modes**→轨迹GT只能cmd积分;④当前workspace**无run_diffusiondrive_policy.py**(只在orig)需移植;⑤新增双臂OpenArm+灵巧手LinkerO6+关节示教轨迹(与导航正交,README_trajectory是臂关节非导航)。
- **VLM接缝** 在 `docs/DiffusionDrive路径规划实现_0707.md` 新增§六:代码实证注入点(status[:4]→_status_encoding→keyval→ego_query→扩散头cross_ego_attn;ModeHead可视觉推mode;⭐ModulationLayer.global_cond空FiLM挂钩=注入解码器入口)+三接法(A零重训4-mode/B扩status/C走global_cond注入解码器⭐目标)+推荐(Phase1=A验证/Phase2=C重训)+VLM输出契约(结构化JSON)。
- **整理** `robot_control_current/.gitignore`(补egg-info/node_modules/*.so/*.log/*.svo/*.db3/办公锁/venv,验证无垃圾入库)。

**结论**:上车接线明确=`感知obs自建(ZED图→camera_feature/点云→lidar_feature/VLM mode→status) → DiffusionDrive → 轨迹→样条跟踪→publish_action`;VLM进decoder走`global_cond`(方案C,已有挂钩)。

**下一步**:(待定)按§五 Step1把真模型插闭环`step`(复用ckpt±32m先打通);或采集端补.svo/odom/曝光/task_modes。**卡点**:无位姿里程计影响BEV裁窗+轨迹反馈;VLM方案C需重训。

---

## 2026-07-07 —— BEV机制查证(真实NAVSIM数据) + 两份规范文档成文

**做了什么**:
- **BEV机制查证+真实数据**:读官方`transfuser_features.py`坐实**NAVSIM喂模型的BEV=LiDAR点云俯视直方图,非图像**(图像是另一路,backbone里才融合)。本机huggingface直连死→走`hf-mirror.com`镜像+`curl|tar --occurrence=1`**流式抽1帧真实pcd仅1.5MB**(不下200×3.3GB整包)。真实NAVSIM(98k点360°环视)与我们真实ZED深度(77万点但**只有前向锥**x∈[0.4,20])**过同一官方`points_to_bev_tensor`**并排出图。
- **文档一** `docs/数据采集规范_0707.md`:对齐上车`run_preprocess.py`的①目录/文件格式(硬约束)②传感器/topic(要录全,别只cmd)③采集协议(⭐别只朝前单程走/锁曝光防白雾/慢稳)④task_modes标注⑤训练超参约束⑥checklist。
- **文档二** `docs/DiffusionDrive路径规划实现_0707.md`:对齐`run_diffusiondrive_policy.py`。DiffusionDrive=条件轨迹生成器(20锚点截断扩散→未来4s/8路点(x,y,yaw)多模态选最高分),轨迹→样条跟踪(带反馈)/分段差分→[vx,vy,wz]+2Hz receding horizon+深度急停安全层。落地四步:复用ckpt打通→BEV洗干净→VLM慢脑接mode→重训。

**产出**:`_navsim_bev_demo.{py,png}`(机制演示)、`_real_bev_compare.{py,png}`(真实NAVSIM vs 真实ZED)、`_navsim_real/*.pcd`(真实样例留存)、两份docs。memory `real-navsim-vs-zed-bev-0706`。

**结论**:BEV域差=**视场覆盖**(环视稠密 vs 前向锥),非分辨率;复用学长ckpt须保±32m,室内±6m须重训。这是可视化铁证。

**下一步**:(待用户定)把±6m ZED BEV与重建占据栅格(OccBEV)叠图看能否补前向锥后方全黑;或按四步落地Step1把真模型插进闭环仿真`step`。

**卡点**:huggingface直连墙(用hf-mirror);真实NAVSIM metadata整包6.6GB未下(只看效果不需要)。

---

## 2026-07-02(下) —— ⭐整个架构端到端跑通(非反应式两阶段, 官方范式)

**目标**:跑一遍完整流程测整个架构(sim皮肤FPV+BEV+位姿 → DiffusionDrive快脑 → 轨迹)。

**查证定架构**:env冲突(sim渲染在discoverse_venv缺模型依赖/CUDA不同,模型需lightning/hydra/nuplan)→无法同进程。查NAVSIM(CoRL25伪仿真/arXiv2406.15349)=**官方本就非反应式**(预测一次4s轨迹→静态环境展开→算PDMS,策略与环境解耦非序贯,省6×)。据此=**两阶段解耦,天然绕开env冲突**。

**做了什么**:
- **模型env**:artifixer补装lightning/timm/shapely/pyquaternion(纯增量无降级);**vendor极小`_nuplan_shim`**(枚举+TrajectorySampling+一堆训练用空壳stub)绕开整个nuplan-devkit(它pin numpy1.23/hydra1.1会搞坏ArtiFixer)。**V2TransfuserModel直连**加载best-val-loss.ckpt(763权重全中)+假输入前向出trajectory[1,8,3]。
- **Stage A** `scripts/gen_obs_sequence.py`(discoverse_venv):沿参考轨迹渲FPV+存位姿→`outputs/obs_seq_<s>/`。
- **Stage B** `scripts/eval_diffusiondrive_nonreactive.py`(模型env):逐obs建features(FPV→camera_feature/**官方sim_bev前向FOV直方图→lidar_feature**/cruise status)→model.forward→轨迹→静态占据展开算无碰撞/进度→出`eval_nonreactive_<s>.mp4`(FPV+BEV叠轨迹)+json。

**结果**:40帧端到端跑通,可视化正确。**无碰撞率0.15/平均进度0.77m/多预测inspect=轨迹差**——**预期内**(学长ckpt训真实ZED,sim的3DGS FPV+几何BEV有domain gap)。**本次目标=验证整条架构数据流通+模型可被sim驱动,已达成**。真效果要重训(收窄BEV/ego坐标/真实轨迹标签)。

**产物/文件**:`_nuplan_shim/`(模型env免整装nuplan)、`scripts/gen_obs_sequence.py`、`scripts/eval_diffusiondrive_nonreactive.py`、`outputs/eval_nonreactive_colmapjoint_all.mp4`。**下一步**:重训原始DiffusionDrive on 我们数据;或先改进sim↔真实domain gap(皮肤用ArtiFixer/BEV参数)。

---

## 2026-07-01夜/07-02 —— ⭐ArtiFixer出片成功+效果拔群(全链路闭环打通)

**接0701 OOM根因定位**：等待器(正解配置 `--num_views 2 --local_attn_size 9 --sink_size 5`,单卡)在 7/1 21:35 集群终于空出卡时,一抢到就跑通。证明该配置成立——之前多次失败**全是抢卡竞态**(整板被占,加载33GB权重中途被别人挤爆),不是配置问题。

**产物**(40帧)：`outputs/artifixer/corrected/artifixer-14b/distilled_views_reconstructed_colmap_2_evenly_spaced_sink5_trajectory/corridor/`
- `videos/batch_0000_rendered.mp4` = 输入(hqbest 3DGS 在 off-path 横移轨迹上的渲染=拖糊的)
- `videos/batch_0000_pred.mp4` = **ArtiFixer 修正输出**
- `comparison.mp4` = 并排对比

**效果(实测对比帧00024)**：输入=糊成一团(墙面拖影、控制箱鬼影、结构不可辨)→ 输出=**锐利照片级**(控制箱清晰、右侧红色消防阀门/管路分明、地面绿底黄标线整齐、墙上标牌可读)。

**结论**：⭐验证 ArtiFixer 能修好 0626 判定的"采集物理下限"off-path 拖糊——**后处理(生成式扩散)确实能救视觉皮肤层**。原则不变=**只进皮肤层(FPV喂VLM/demo)不进几何层(生成=幻觉,导航几何继续用测量)**。

**下一步**：①渲更长 off-path/整条走廊轨迹出 demo 皮肤 ②ArtiFixer3D 蒸回 3DGRUT 做可自由漫游整条走廊皮肤 ③接进 FPV 喂 VLM。BEV 侧(另一线)已定=从测量几何洗干净的占据栅格→ego裁窗喂DiffusionDrive(见 memory bev-input-for-diffusiondrive-0701)。

---

## 2026-07-01（周三）—— ArtiFixer 推理 OOM 根因彻底钉死（更正0630）+ 找到正解配置，剩纯抢卡

**触发**：用户问"昨天跑通没有"。核实=**没跑通**（`corrected/`无mp4，等待器进程已不在）。今天沿OOM追到底。

**做了什么**：五次受控尝试，把OOM追成**两个串联瓶颈 + 抢卡竞态**（更正0630"local_attn9=解药"——那是错的）：
- ①单卡attn9→78.58GB OOM@prope ②单卡attn7→78.51GB（**注意力窗口对显存零影响**，因为死在prope之前根本没到KV cache）③2卡CP→`cp_frame_divisor=gcd(fpb,sink,attn)=gcd(7,5,9)=1`只许CP=1，assertion挂 ④2卡CP偶参数(gcd=2)→过assertion真进生成，但**每卡仍78GB OOM@prope**（CP切时序不切空间，neighbor每卡都要全存）。
- **根因**：`transformer.py:898`把`num_views`个参考帧key强转**fp32**喂prope投影einsum，这块"每帧空间token@1280×720×参考帧×fp32"固定张量一步撑满80GB；帧数/窗口/多卡切的全是时序维度，绕不开。模型**原生gcd(7,7,21)=7=作者按7~8卡多卡跑**，我们硬塞1~2卡方向错。
- ⑤**num_views=2起效**：OOM从prope**前移**到`_initialize_kv_cache`的K cache分配→此时local_attn9才**终于生效**。

**正解**：`--num_views 2 --local_attn_size 9 --sink_size 5` 单卡，两处OOM都清。

**卡点=纯集群争用**：该组合一启动就撞竞态（别的进程在我们加载33GB权重中途冲进来占49GB，权重挪不上差50MB）；整板反复被填满，最空卡常<30GB<33GB权重。**未得一次公平测试**。

**产出**：等待器`run_artifixer_inference_when_free.sh`已更新为正解配置(NUM_VIEWS=2/attn9/NEED_FREE=60G/失败重试)，后台轮询抢≥60GB窗口。日志`outputs/artifixer/infer_*.log`+`waiter_nv2attn9_0701.log`。

**下一步**：等waiter抢到干净卡出片→对比hqbest off-path拖糊；若竞态一直抢不到→降render分辨率(单卡门槛更低)或错峰。**只进皮肤层不进几何层**。

---

## 2026-06-28（周日）—— ArtiFixer 环境搭建（接 0627，补 0627 漏记的日志）

**触发**：上次（0627）工作只进了 memory(`artifixer-skin-repair-0627`)，**没回写本日志**，导致"上次进行到哪"对不上。本条补记 0627 + 推进 0628。

**0627 已完成（核实落地）**：
- 代码 `third_party/artifixer/` + `third_party/3DGRUT-ArtiFixer-main/`（codeload tarball，git clone 超时）。
- 权重全下完：`/data/DongBaorong/artifixer-checkpoints/artifixer-14b.pt`=**63GB**（curl 续传）；Wan2.1-T2V-14B base 仅 text_encoder(22G/5分片)+vae(485M)，**transformer 按计划排除省28G**。
- env `artifixer`(micromamba py3.12)：torch2.11.0+cu128 CUDA可用、slangtorch1.3.4、nvcc12.8(env内)。

**0628 做了什么（环境收尾，进行中）**：
- 查证 `Dockerfile.cuda12` 拿全依赖清单。**关键决策：A100(sm_80)砍掉全部 flash-attn(FA3/FA4)及 cuda-python 降级**——那是 Hopper/Blackwell 专用，A100 走 cuDNN SDPA（`model_training/net/transformer.py` 自动按算力选）。
- ✅ **slangc 2026.5.2** 装进 env 前缀(非 /usr/local，无 root)：`install_slangc.sh` 内置 wget 把 tarball 下截断(gzip EOF)→改 `curl -fL -C -` 续传 68M→`slangc -version` 通过。
- ✅ **3DGRUT requirements.txt**：fused-ssim 等 git/CUDA 扩展的 setup.py 构建期 import torch→隔离构建无 torch 失败→**必须 `--no-build-isolation`**(先装 ninja + `setuptools<72.1.0` + wheel)，fused_ssim wheel 编译通过。
- ✅ **3DGRUT `pip install -e .`**：editable 秒过(CUDA kernel 是运行时 slangtorch JIT，非构建期编译)，`import threedgrut` OK。
- ✅ **accelerate1.13/diffusers0.37.1/transformers5.5.0/ftfy + einops…torch-fidelity 批**：全过。
- ✅ **MoGe**：git clone 超时(已知坑)→codeload tarball `pip install /tmp/MoGe-main --no-build-isolation`→`from moge.model.v2 import MoGeModel` OK。
- ✅ **全 env 自检绿**：torch2.11+cu128 / A100(cc8.0) / 全模块 import 通过 / flash_attn 按预期缺席。

**环境彻底搭完**。prepare 接口已读：`--colmap_dir`(喂 `outputs/_colmap_joint_all`)/`--output_root`/`--trajectory_path`(transforms式c2w)/`--phases prepare,reconstruct,render,scale,caption`；**reconstruct=用我们COLMAP自训3DGRUT MCMC 10k步**(非复用hqbest gsplat)，scale=MoGe，caption=Wan text_encoder。

**0628 跑通 reconstruct(3DGRUT MCMC 10k 训练真正起跑)——踩了一长串坑**：
- **单相机**:我们 `_colmap_joint_all` 每图各一 SIMPLE_RADIAL(BA 逐图微调,fl_x 475~677/median546.76)→prepare 断言"one shared intrinsic"挂。写 `normalize_colmap_single_cam.py`(复用 artifixer COLMAP IO)压成单相机,且 3DGRUT 只吃无畸变→转 **PINHOLE 丢 k1**(median 0.0048 可忽略)→`_colmap_joint_all_singlecam`。
- **val 空**:默认选全部 1610 当训练→val=补集=空→`compute_spatial_extents` 崩。正确接口=传 `--selected_image_names_file`(训练子集),其余自动 val。生成每 20 帧留 1 张:train1529/val81 → `train_images_95.txt`。
- **tiny-cuda-nn 缺**:3DGRUT tarball 无 git 子模块→`thirdparty/tiny-cuda-nn` 空→JIT 缺 `common.h`。`git clone --recursive` 拉(含 fmt/cutlass/cmrc)。
- **TCNN_HALF_PRECISION**:clone 的是最新 main,要求显式定义→patch `threedgut_tracer/setup_3dgut.py` 加 `-DTCNN_HALF_PRECISION=1`(cflags+cuda_cflags)。
- **CUDA 头/库不全**:env 只装 cuda-nvcc(编译器),缺 `cusparse.h`(torch 头要)+ 链接缺 `-lnvrtc`。CUDA 库其实在 pip `site-packages/nvidia/*/{include,lib}`→把 70 个头软链进 `targets/x86_64-linux/include`、28 个无版本号 `.so` 软链进 `lib`(linker `-lxxx` 要无版本名)。
- **vgg16 卡死(元凶)**:LPIPS 感知损失从 torch hub 下 `vgg16-397923af.pth` **挂死在 0 字节**(urllib 卡;curl 同 URL 正常)→`curl` 预下到 `~/.cache/torch/hub/checkpoints/`(553MB)→重启命中缓存秒过。
- ✅ **重启后 GPU2 利用率 74→90%、显存增长,3DGRUT MCMC 10k 训练起跑**。MOGE_MODEL_PATH 指 `/data/DongBaorong/moge-2-vitl-normal`(1.3G 已下,供后续 scale)。

**下一步**:等 reconstruct 出 ckpt → render(造 off-path 轨迹 transforms json)+scale(MoGe)+caption(本地 Wan UMT5 编一句固定 prompt,跳过 30B Qwen3-VL)→ `run_inference`(artifixer-14b.pt) → 对比 hqbest off-path 拖糊。**只进皮肤层不进几何层**(生成=幻觉)。

**坑汇总**:github release wget 截断/git clone 超时→curl 续传 + codeload tarball;同 env 不并发 pip;含 torch import 的 sdist 一律 `--no-build-isolation`;pip CUDA 库要手工软链头+无版本 .so;torch hub 下载会挂→curl 预下到 cache;3DGRUT tarball 必须补 git 子模块;GPU 全板被抢→ reconstruct 用空出的 GPU2。

**0628→29 跑通 render/scale/caption + 推理打通到生成(OOM,已缓解)**:
- ✅ **off-path 轨迹**:`make_offpath_trajectory.py` 从 prepare 的 nerfstudio transforms.json 取中段(700起)沿相机 right 轴横移 0.35+偏航 12°→`outputs/artifixer/offpath_lateral035.json`(transforms 式 c2w,无 file_path)。正对准单程采集没覆盖、hqbest 拖糊的视角。
- ✅ **render**:`--phases render --trajectory_path ... --reconstruction_checkpoint ...`(复用 ckpt)渲出 RGB+opacity。**坑**:`write_eval_split`(写 split.json)需 render(trajectory)+scale 在**同一次调用**(trajectory 是局部变量),分开跑不写 split.json→render,scale 合并一次跑。
- ✅ **scale**:MoGe 估 metric_scale=3.525。**坑**:`MOGE_MODEL_PATH` 要指到 `model.pt` 文件(不是目录)。
- ✅ **caption(绕开 30B)**:`load_encoded_prompt` 只读 h5 第一个 dataset key,prompt_paths 空则用零 embedding,但 split.json 的 prompt_path 必须存在。写 `make_caption_h5.py` 用**本地 Wan UMT5**(免下 30B Qwen3-VL)编一句固定 prompt→`captions/corridor/caption.h5`(emb 45×4096)。
- ⚠️ **run_inference 打通但 OOM**:14B 加载成功(33GB)、生成已开始,但 **80帧 KV-cache 初始化 OOM**(满空 80GB 卡上仅差 480MB)。缓解:①轨迹缩到 **60 帧** ②`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- ⚠️ **真阻塞=GPU 争用**:WangBizi 等把整板抢满(每卡剩 24-37GB,14B 权重就要 33GB)。挂无人值守等待器 `run_artifixer_inference_when_free.sh`(轮询某卡空出≥76GB 即自动跑推理,最多等 8h)。

**当前状态**:ArtiFixer 全链路(env→reconstruct→render→scale→caption→inference)**已端到端打通并验证**,只差最终生成那步等一张空闲卡(等待器在跑,完成会通知)。产物落 `outputs/artifixer/corrected/`。新脚本:`normalize_colmap_single_cam.py`/`make_offpath_trajectory.py`/`make_caption_h5.py`/`run_artifixer_inference_when_free.sh`。

**0629 推理两次 OOM→定位+修复(60帧已过KV-cache,卡在VAE decode)**:
- 80帧→KV-cache OOM(满空80GB仅差480MB)→缩 **60帧** 解决,一路跑到**最后 VAE decode** 才挂。
- 60帧→VAE decode OOM:真因=**抢卡竞态**(等待器见 GPU0 空就启动,另一进程同时抢走 43GB,我们只剩 36GB,VAE 解全部帧差 170MB)。
- **修复①VAE tiling**:`kv_cache_pipeline.py decode_latents_to_video` 加 `vae.enable_tiling()+enable_slicing()`(幂等)→分块解码,峰值显存大降,不再需整张 80GB。
- **修复②等待器加固**:阈值 76GB→**55GB**(tiling 后够)+**失败自动重试**(不再一次放弃),每分钟轮询最多 10h。
- 教训:共享集群"见空就抢"有竞态;VAE 长视频解码必开 tiling/slicing。

**0630 OOM 真因定位(降帧数无效→真因是KV cache由注意力窗口定)**:
- 80/60/40 帧三次 OOM **完全相同**(都差 740MB、进程占~78GB)→**降帧数无效**,显存大头不随轨迹帧数变。
- 真因:`num_cache_frames = local_attn_size`(模型注意力窗口,默认**21**),KV cache = 21×frame_seq_length×层数,约 **50GB**;加 bf16 14B 权重 28GB → ~78GB,**单张 80GB A100 原生放不下**(这是 H100 多卡级模型,Dockerfile 本就多卡+context-parallel)。checkpoint 是 map_location=cpu+mmap 不占 GPU,排除。
- **修复=`--local_attn_size 9 --sink_size 5`**:注意力窗口 21→9,KV cache 砍约 57%→总显存 ~50GB,单卡放得下(质量略降可接受,做对比够用)。多卡 context-parallel 是另一路(torchrun+`--context_parallel_size`)但需≥2张空卡同时,集群争用下难凑。
- **当前阻塞=纯 GPU 争用**:GPU 空窗(GPU2 曾空 80GB)在改参数/渲染的 2 分钟内被别人抢走;板子反复填满(最大空闲常 ~22GB<28GB 权重)。等待器已用省显存配置(local_attn9)+阈值55GB+失败重试重新武装,抢下一个窗口。
- 教训:OOM 先看"差多少/进程占多少"是否随你改的量变化——不变=大头在别处(此处是注意力窗口非帧数)。
**下一步(出片后)**:对比 hqbest 原始 off-path 拖糊 vs ArtiFixer 生成;成则上 ArtiFixer3D 蒸回做整条走廊 demo 皮肤。**只进皮肤层不进几何层**。

---

## 2026-06-26（周五）—— 合成重训hq2 + 挑帧hq3：诊断"糊=采集几何天花板"(两实验证伪旋钮/帧选)

**触发**：用户嫌684k皮肤糊，要"先把重建做扎实再做BEV"，并指出"不只是旋钮、帧还少、反光没处理好"。

**查证(4项,排除伪因)**
- 684k实为 SH0 + 默认688×384低分辨率(ply只有f_dc无f_rest)；左目帧已用满(stride1,1610/1813=89%)；右目zed_right基线值**未记录**、左右目**19%不同步**、基线仅12cm。
- **源帧清晰度**:最锐帧Laplacian var=679照片级清晰→**不是GT糊**;最糊帧多是怼墙低纹理。
- **冒烟抓bug**:COLMAP的K本就是1280×720全分辨率,原计划UPSAMPLE=1.86会超采到2381×1339(假细节+3.4×浪费)→**正确是UPSAMPLE=1.0**。

**做了什么**
- **hq2合成重训**:1610帧+全旋钮(UPSAMPLE1.0原生1280×720 / SH3 / GROW_GRAD2D0.00015激进致密 / DEPTH_SUP0.4 / 严格遮罩225 / ANTIALIAS / REG_SCALE0.01)→**109万高斯**(205万剪枝),SH3。
- **hq3挑帧重训**:时序局部NMS(W5,ratio0.7)剔20%运动模糊帧→1291帧,同旋钮→**110万高斯**。子集目录`vggto_colmapjoint_all_sharp`(cameras子集+软链帧+共享recon.ply)。
- 同位姿对比(hq3用hq2轨迹驱动,同世界帧k7.44≈7.47):`_cmp_hq3_vs_hq2_f*.png`、`_compare_hq2_vs_684k.png`、`_cmp_full_f*.png`、`_single_vs_merged.png`、`_rawframe_sharp_vs_blur.png`、`_dropped_frames.png`。

**结论(诚实)**
- hq2 vs 684k:**on-path有提升**(水泵/管道/地标线更锐+SH视角高光+无发白),但**非本质**;侧边/远处仍拖糊。
- 单段114830c(旧配置)vs hq2合并:**边缘拖糊一样**→**证伪"合并是糊主因"**。
- hq3挑帧 vs hq2:同位姿**几乎一模一样,零可见提升**→**证伪"糊帧拖累/帧选有用"**。
- **真因=采集几何物理天花板**:单目前视单程,每表面只在狭窄前向视锥被看到,侧边掠射角缺横向视角→高斯约束不住→糊。旋钮/合并/帧选/帧数都动不了。
- 剩余真杠杆:①右目立体(补横向轴,但12cm小+几小时COLMAP重注册,只救近处);②**重采绕拍/多高度=真正的解(未来)**;③双层设计下FPV给VLM做识别不需razor-sharp,这层糊不挡主线。

**产出(版本号保留,未动684k定版)**:`gs_vggto_colmapjoint_all_hq2.ply`(+world+npz,271MB,**当前工作皮肤**)、`gs_vggto_colmapjoint_all_hq3.ply`(挑帧版,等价)、walk视频若干。

**坑**:训练中途另一用户(WangBizi)起VLA占满GPU3-7→hq3迁到GPU0(61G空)重跑;`align_gs_world`用同一session参数推ply和vdir→改名版本需`ln -sfn vggto_<base> outputs/vggto_<newtag>`软链。

**下一步**:用户定——accept hq2当皮肤转做**BEV(几何拟合两墙)** / 或赌右目立体。倾向前者(两实验已证天花板)。

**追加(同日,用户要求继续试深度+右目)**
- **查证深度用到哪**:深度**监督**一直开(DEPTH_W0.4全帧匹配,消针球);但**初始化**用的是COLMAP稀疏25.7万点,**没用ZED深度融合稠密云**。→ 这是真空档。
- **hq4 深度稠密初始化**:写`build_zed_init_merged.py`(按帧名A/B路由两段深度,复用fuse_zed的decode+k估)→`zed_fused.ply` **832万度量点**(密32×);`INIT_PLY=zed_fused INIT_PTS=1.2M DEPTH_W=0.2`(位置交初始化,调低监督防过平滑)重训→**118万高斯**。同位姿vs hq2:**又接近平手,无本质改善**。
- **右目几何反推(关键)**:从ZED深度+SIFT视差反推→**y视差中位0.1px=已rectified**、**基线119.6mm=标准ZED2 120mm**(紧分布);故右目位姿可解析:`R_R=R_L, t_R=t_L-[base_norm,0,0]`,**免COLMAP重注册**。
- **hq5lr 左+右合训**:`vggto_colmapjoint_all_lr`(1610左+1305同步右=2915视角,稀疏init隔离变量)同hq2旋钮→**93万高斯**(视角多致密更保守)。同位姿vs hq2:on-path**几乎一模一样**;f180转头看侧墙(横向视差最该起作用处)**两版一样糊成团**→**右目12cm对off-axis杯水车薪,证实无效**。

**三实验全证伪(hq3挑帧/hq4深度init/hq5lr右目)**:糊是单目前视单程采集天花板,后处理(旋钮/帧选/深度初始化/右目立体)**全部救不了**,唯一解=重采(绕拍/多高度)。产物:`gs_vggto_colmapjoint_all_{hq2,hq3,hq4,hq5lr}.ply`,**hq2当工作皮肤**,684k定版未动。对比图`_cmp_{hq4,hq5lr}_vs_hq2_f*.png`。坑:align的RANSAC地面估计跨run出不同k(5.68/6.88/7.47),同位姿对比改用"hq2轨迹×k比例"缩放驱动。

---

## 2026-06-25（周三）—— 用户三要求:反光/人像遮罩 + 用尽量多帧(1610) + 修BEV坐标轴

**① 反光/强光/人像预处理**
- **过曝/镜面反光**:`overexp_mask`纯阈值(min RGB>225 或 V>235&S<35,膨胀)。训练`MASK_OVEREXP=1`时这些像素**不参与光度+深度损失**。验证精准抓白爆/地面反光(`_overexposure_check.png`)。**这是之前"偏1m白爆"的元凶**。
- **人像**:⚠️ maskrcnn/yolo/transformers权重经pytorch.org/github/PyPI**全被本机防火墙挡死**,HOG不可靠。→ 改**帧范围排除**:114830有人段288-363(已避开)、113628开头0-30。

**② 用尽可能多的帧**(用户嫌"只用1/6")
- `colmap_dense_all.py`:1702帧,**sequential(O(n))+关键帧桥接跨段**,避exhaustive O(n²)的7小时。桥接`match_image_pairs`走faiss多线程BLAS**SIGSEGV**→`colmap_all_resume.py`**单线程**修。BA极慢(~4h,子模型逐个写sparse/,大模型sparse/0写出即kill)。**注册1610帧/25.7万点**(vs 426/8.8万,3.8×)。
- 训练`gs_vggto_colmapjoint_all`(1610帧+深度+过曝遮罩+抗锯齿):**~68万高斯,无针球无白爆,但偏软**。
- **诊断softness**:retrain去尺度正则+松剪枝(REG_SCALE=0,PRUNE_SCALE=0.5)→仍68万、仍软→**证明软不是剪枝,是1610帧覆盖35m大场景高斯密度低+深度监督平滑**(权衡:426版小范围更锐,1610版全覆盖更软)。
- 省显存:train图缓存uint8、深度f16,支撑1610帧不OOM。

**③ BEV坐标轴/自身箭头**:`test_env._bev_window`重写=ego heading-up向量化(前=上右=右),红箭头朝上=前进,与FPV一致。验证`_bev_fixed_frames.png`。

**最终定版(用户拍板回退)**:试了v2(放宽过曝遮罩240+增密)和v3(增密)想"更锐/消发白",用户实测都觉得**不如最初的684k版**(v2/v3发白更重;BEV因点云密而墙粗碎,且阈值调不动=被机器人半径膨胀主导)。**回退并永久保存684k版**(配置DEPTH_W0.4/严格遮罩225/默认密度→66万高斯):`gs_vggto_colmapjoint_all_684k.ply`(+_world+_684k.npz),FPV视频`outputs/walk_684k.mp4`。**教训:产物务必保留版本号**(原684k和视频被覆盖丢了,只能重训复现)。**BEV待办**:用户定了**从几何拟合两墙重建**BEV(decouple皮肤密度,fit_world_geometry的smooth_centerline/traj_frame/fit_wall_side可复用)——当前BEV仍是点云密度occupancy,未做。

**结论/权衡**:三件事全做。1610帧版**更完整**(整条走廊、无针球白爆)适合当测试场皮肤(机器人可在任意处测);**426版局部更锐**但覆盖小。几何/BEV导航底座两者一致,皮肤锐度只影响VLM的FPV观感。要1610更锐需~2-3倍高斯(更激进densify,更多算力)。

---

## 2026-06-24（周二，傍晚）—— ⭐深度监督+稠密426帧消针球(修正"物理上限"误判)

**触发**:用户质疑"走廊来回+侧扫这么多数据不够吗、有没有最新方法,都要查证"。查证发现**我严重没用全数据**。

**数据真相(用户确认)**:这条走廊**只有114830(630)+113628(1183)两段**(142522是**另一条**走廊、112633是**室外**,不能用)。两段1813帧我只用过165。**每帧ZED有度量深度**(pointcloud/zed/*.npz z通道,毫米),训练**从没用过**。

**做了(全跑通)**
- **稠密COLMAP BA**:`colmap_joint.py --a_end 288 --a_stride 2 --b_end 1183 --b_stride 4 --tag dense`(440帧,exhaustive~24min)→**注册426帧**(114830全144+113628 282/296)**8.8万点**。
- **深度监督**(`train_3dgs_vggt.py`新增env开关,向后兼容):渲染期望深度(gsplat `RGB+ED`)对齐ZED度量深度,训练前最小二乘估全局尺度m换算到归一化帧,alpha>0.5处Huber;`ANTIALIAS`(Mip式`rasterize_mode=antialiased`);`REG_SCALE`罚胖飘高斯。新`_build_pc_index`/`load_zed_depth`按时间戳查npz,排除20m量程饱和。
- 训练20k→**97.6万高斯**;对齐(k=7.0,跨度45.6×29.2m);穿行`sim_walk_colmapjoint_dense.mp4`(nonblack100%)。

**✅ 结果**:自检`colmapjoint_dense_freeview.png` vs 之前`colmapjoint_freeview.png`(前后对比`_compare_depth_before_after.png`):之前偏1m**白爆**、转头看墙**针球炸开**;现在偏1m走廊连贯、**转头看墙能看到真实管道/阀门/水泵**。**用户明确不想要的针球消掉**。从"轨迹锐+离轨迹灾难"→"**到处结构正确、只是糊**"。

**结论修正**:今早"观测覆盖物理上限"**是误判**,真因=只用165帧+训练零深度。深度监督把高斯压到表面是治针球直接杠杆。剩余糊=软上限(中线穿行非绕墙),要更锐需补绕拍采集,但已非"不可用针球"。

**没全帧的理由**:exhaustive O(n²)(440帧24min,1813帧≈7h),相邻帧~95%冗余,抽帧≠丢覆盖;真要全帧=PnP把余帧注册进已BA模型(O(n))不重跑exhaustive,等效果不够再做。

---

## 2026-06-24（周二，下午）—— COLMAP紧位姿训统一3DGS:轨迹上锐但自由视角B仍达不到(观测覆盖物理上限)

**做了什么**
- 承接上午"统一3DGS糊(loss0.2)→归因VGGT前馈位姿松"。上 **COLMAP联合BA紧位姿** 重训:`build_colmap_session.py`(修`im.cam_from_world`是方法要`()`调)导 `_colmap_joint/sparse` 最大模型(165帧/33233点)成训练格式 `vggto_colmapjoint/`(K已1280×720全分辨率)→ `train_3dgs_vggt.py colmapjoint`(ITERS20000/SH0,稀疏33k初始化**稠密化到157万→剪枝101万高斯**,66MB)→ `align_gs_world.py colmapjoint`(k=7.28,跨度12.2×29.3m)→ 穿行 `sim_walk_colmapjoint.mp4`(160帧,nonblack97-100%)+ 自由视角自检 `colmapjoint_freeview.png`(4视角)。

**结论(诚实)**
- ✅ **紧位姿有效**:轨迹上前视锐(好视角loss0.06=单段114830c同档)、穿行几乎无黑洞、**没再像统一VGGT那次整体糊成0.2**。COLMAP全局BA把两段绑进一个自洽坐标系成功。
- ❌ **但自由视角B仍达不到**:freeview图——轨迹上朝前锐;**偏左仅1m朝前→糊成白**;原地转头看墙→**针球/拉丝/炸开**(用户明确不想要的)。
- **根因=观测覆盖物理上限**,非位姿松(这次紧):114830/113628都沿走廊中线穿行,没贴墙/绕墙/多高度拍→3DGS只重建看过的视角,离采集带子就无约束→炸。113628横看帧有帮助(注册69/95)但也只一条线。**证伪"紧位姿能救B"**。
- → **双层方案才是"可信测试环境"正解**:①干净几何(占据/BEV)给DiffusionDrive,视角无关到处准;②3DGS皮肤只需轨迹附近锐给VLM当FPV。闭环测试依赖几何占据不依赖3DGS自由视角清晰度。真要B需重采(贴墙/绕拍/多高度)。

**git**:commit `7b2889d` 本地已存;push卡网络(github.com直连443超时+ghfast.top镜像只读)→打 bundle `robot_bev_sim_latest.bundle` 兜底,网络好/笔记本再推。

---

## 2026-06-24（周二）—— 用上深度 + 多采集合并 + COLMAP联合BA + 闭环测试场接口

**澄清的真目标**：仿真器=**闭环测试场**(现场实测难→仿真里测训好的模型),要点是**仿真精度够测试可信**(几何/观测对得上真实),物理碰撞不是难点。策略输入=BEV/占据(几何,视角无关)+VLM吃FPV-RGB。

**做了什么**
- **用上ZED深度等数据**(之前只用mono):ZED每帧度量点云96-98%有效+IMU+轮速里程计。`fuse_zed.py`(VGGT位姿融ZED深度,REFINE帧到模型ICP)、`zed_world.py`(→世界帧)、`dense_map.py`(Poisson网格)、`occupancy_bev.py`(几何→BEV占据)。**ZED真彩**:rgba打包在float16坏了→改从同名RGB图按像素对应取色(`decode_zed_npz`)。
- **多采集合并**:114830(朝前)+113628(横看侧面)是**同一走廊不同轨迹、有重叠、视角互补**。FGR+ICP配准成功(rmse6.5cm,`_reg_T_113628d_to_114830c.npy`),合并`corridor_merged.ply`。但**统一3DGS糊**(loss0.2,VGGT前馈位姿松~0.4m→多视角对不齐;`build_unified.py`)→证伪"训练能救松位姿"。
- **COLMAP联合BA(治糊关键)**:`colmap_joint.py` 对两段191帧exhaustive匹配+全局BA→**165帧注册进同一连通模型(114830:96/96,113628:69/95),厘米级紧位姿**。这正是VGGT前馈给不了的。`build_colmap_session.py`导成训练格式,待训锐利统一3DGS。
- **闭环测试场接口** `test_env.py` `CorridorTestEnv`:reset/step(v,ω)→obs{fpv:3DGS, bev:占据窗, pose}+碰撞/到达判定;脚本策略跑通闭环demo。模型插进来即可测。

**关键结论**：①3DGS锐利只来自单段自洽位姿(114830c loss0.06照片级);多段照片级需紧位姿=COLMAP-BA(已拿到)。②VGGT-Ω强在快/单段,弱在厘米级紧+跨段联合→COLMAP补这块,互补。③双层:单段锐3DGS皮肤(VLM)+合并几何(DiffusionDrive BEV,视角无关)。

**坑**：启server别同bash内`pkill -f sim_walk_server`(杀自己);EGL_BAD_ACCESS换没跑过的GPU+彻底清进程;CUDA_VISIBLE_DEVICES=N时MUJOCO_EGL_DEVICE_ID=0;COLMAP CPU匹配OPENBLAS_NUM_THREADS=1防崩。

**下一步**：用COLMAP紧位姿训锐利统一3DGS(逼近"到处清晰"的B);抠人;接DiffusionDrive/VLM真测。

---

## 2026-06-23（周一，下午）—— 双层仿真器跑通(干净几何物理层 + 3DGS可切换皮肤 + MMK2四轮车)

**背景**：用户看完 hq 后反馈「c 比 hq 好看」「还是要 DISCOVERSE 官网那种真仿真器」「为什么我们渲得差」。讲透根因=**3DGS 糊是单程前视采集欠观测的物理上限**(hq 过参数化反更糊),「没仿真感」=场景零碰撞几何。用户拍板**双层都要**(几何物理底座+3DGS皮肤),机器人用 DISCOVERSE 自带四轮车外观+我们(v,ω)底盘,只保碰撞不要动量。计划 `eager-dancing-unicorn.md`。

**做了什么**(A→D 四步全通)
- **A 几何拟合** `scripts/fit_world_geometry.py`：从 `gs_vggto_114830c_world.ply`(100万高斯)去噪(opacity>0.15+scale<0.3+统计离群+轨迹bbox裁剪)→**用「相对平滑中心线的带符号横向偏移」找两墙**(非全局x直方图,对走廊弯曲鲁棒)→重度平滑去抖。两墙各19段/高2.04/厚0.12,走廊宽3.6m;障碍盒 DBSCAN 聚类 + **剔除压在轨迹0.6m内的挡路噪声盒**(真实障碍贴墙在路侧)→9盒。出 `world_geometry_114830c.json` + `_check.png`。
- **B+C MJCF/物理** 重写 `scripts/sim_walk_common.py`：build_mjcf 读 json 生成墙/盒的 collision+visual geom(airbot class约定)+棋盘地+headlight环境光;机器人外观换 **MMK2 移动底座**(`models/meshes/mmk2/` 的 agv底座+4脚轮+2驱动轮 mesh,纯视觉,rm2_car mesh缺失改用mmk2)+碰撞盒;物理 `updateControl` 从写qvel改**写 mj_data.ctrl 配 velocity actuator**(DISCOVERSE每子步覆盖qvel→必须走ctrl接触力才顶得住)。
- **D 切换+交付** `sim_walk_server.py` 加 `/toggle` 路由翻 `show_gaussian_img`+HTML按钮/T键;`sim_walk_discoverse.py` 加 `--compare`(同位姿渲几何+皮肤左右拼接)。

**产出**：`outputs/sim_compare_114830c.mp4`(左几何/右皮肤对比,150帧)、`world_geometry_114830c.json`、`_compare_sheet.png`(汇报可用)。

**验证**：✅几何视图=干净游戏关卡走廊(两墙+棋盘地+障碍盒);✅3DGS皮肤照片级;✅**物理碰撞**(朝墙开8s只移2m被挡住不穿墙,朝前自由前进7.4m);✅server `/toggle` 交替0/1、`/cmd` 200。

**坑/解法**
- 起点全黑:npz `yaw[0]` 是噪声(首waypoint差分极小)→相机背对走廊。用「首个离起点≥1m的前方路点」定开局朝向(server已有);别给 robot body 加 `euler` 否则与 set_pose 的 jyaw **双重旋转**。
- align 走 SH 旋转分支缺 `einops`/`e3nn`(上午已装)。
- rm2_car mesh 不在 checkout(仅可选下载)→改用齐全的 mmk2 mesh。

**下一步**：用户验收双层效果;接 DiffusionDrive/VLM 把几何层当训练台;Phase E 重建兜底(全段-中间有人段)按需。

---

## 2026-06-23（周一）—— 高质量重建 114830hq 跑通(4 大糊源旋钮全开)

**做了什么**(承接 recon-quality-knobs 计划:把喂给仿真器的基础数据做"尽可能好",物理岔路暂缓)
- 新 session `114830hq`(隔离,不覆盖 114830c 好对比),走通整条 VGGT-Ω→3DGS→对齐→穿行:
  - **旋钮2 加密帧**:抽 **128 帧**(vs c 的 64)→ `run_vggto`(res512,峰值显存 16.8G)→ `recon.ply` **23.7M 点**(比 64 帧稠密 ~2×)。
  - **旋钮1 全分辨率训练**:`train_3dgs_vggt.py` 加 `UPSAMPLE` 支路,K 与渲染分辨率 688×384 → **1280×714**(读全分辨率 GT)。
  - **旋钮3 SH degree 3**:加 `sh0/shN` 参数 + SH 阶数 warmup;导出 f_dc+f_rest(通道优先,`shN.transpose(1,2)`)对齐 util_gau。
  - **旋钮4 剪枝**:导出前剔除 op<0.05 / scale>0.3 浮点高斯。
  - 训练 20k iters,INIT_PTS 70万 → 密化到 **214万 → 剪枝后 132万**高斯,loss~0.05–0.13。
  - `align_gs_world` 烘世界帧(k=9.52,跨度 10.1×35.5m);`sim_walk_discoverse` 渲 180 帧穿行。

**产出**:`outputs/gs_vggto_114830hq.ply`(326MB,SH3)、`gs_vggto_114830hq_world.ply`、`sim_world_114830hq.npz`、`sim_walk_114830hq.mp4`、**`compare_hq_vs_c.png`(hq上/c下 对比图,汇报可用)**。

**结论**
- ✅ 四个糊源旋钮全部跑通且有效:对比图里 hq 明显更锐(管道/墙体结构清晰、绿地面有视角相关反光高光=SH 生效),c 偏糊发雾、有重影。
- ✅ 顺带实测确认:**DISCOVERSE 的 MuJoCo 渲染路径确实按视角评估 SH**(hq 穿行 nonblack 96–100%,反光随视角变)——之前 recon-quality-knobs 里待实测的疑问解决。
- 注意:对比图两段轨迹帧数不同(c 64 / hq 128),非严格同位姿,只看质量趋势;要严格同位姿对比可让 c 的 ply 沿 hq 轨迹渲(下一步可选)。

**坑/解法**
- align 第一次走到 SH 旋转分支,缺 `einops` + `e3nn`(transform_shs 用 Wigner-D 旋转 SH 系数)→ pip 装进 `~/discoverse_venv` 解决。之前 64 帧无 SH 版 f_rest 为空,走不到这段所以没暴露。
- `train_3dgs_vggt.py` 保持 `SH_DEGREE=0` 与旧行为完全兼容(走 colors 预计算路径,不写 f_rest)。

**下一步**
- 可选:严格同位姿 hq-vs-c 对比图;把 hq 资产拷本地仿真器;更高 iters/更密帧继续压糊。
- 物理岔路(sim-no-physics-fork)仍待用户拍板:干净几何世界 vs 3DGS+隐形碰撞代理 vs 双层。

---

## 2026-06-22（周日）—— 去人版 114830c 全流程跑通 + 明确本地交互仿真需求

**做了什么**
- 用户要重跑一个"无人"的干净路径:仍是 114830,但删掉中间有人段 `1781149863800901--1781149901301656`(76帧)。
- 看 4 个边界帧确认:**114830 是连续长穿行(非来回)**,起点宽走廊→删除段前干净走廊→删除段后管道/机械区→终点楼梯间(终点还有个人)。删中间保两头会把场景切成两个无重叠区域,VGGT 拼不起来 → 经用户确认**只重建段前 288 帧(走廊段,全程无人)**。
- 新 session 名 `114830c`(隔离,不覆盖旧含人版):288 帧抽 64 → `run_vggto`(峰值显存 11.6G)→ `train_3dgs_vggt`(101万高斯,loss0.06)→ `align_gs_world`(k=9.94,轨迹 11.8×36.4m)→ `sim_walk_discoverse` 穿行视频。

**产出**:`outputs/gs_vggto_114830c_world.ply`(69MB)、`sim_world_114830c.npz`、`vggto_114830c/`、`outputs/sim_walk_114830c.mp4`(180帧,**全程无人**,忠实还原绿地走廊+两侧蓝色水泵/管道)。

**结论**:整条 VGGT-Ω→3DGS→真仿真器链路在去人数据上重跑通过。穿行画面有糊/黑边=3DGS 偏离训练视角的已知现象,亲手贴路走会清楚。

**下一步**:用户强调要"有显示器、能亲手操作、真仿真",且笔记本有 N 卡 → 打**本地包**:① MuJoCo 原生窗口版 `sim_walk_viewer.py`(弹真 3D 窗口,鼠标转+键盘开,最像"一个软件");②一页本地安装步骤(torch+gsplat wheel+mujoco+gaussian_renderer,~3-4G);③拷资产(_world.ply 69MB + npz + 脚本)。分工:服务器=重活(重建/训练),笔记本=亲手开仿真器。

---

## 2026-06-20（周六，下午）—— 汇报 PPT 视觉素材

**做了什么**
- PPT 文档 `docs/汇报_0619_PPT要点.md`:加 5 部分浓缩结构(① 巡检场景 ② 真实采数 ③ 仿真环境 ④ 双脑架构 ⑤ 闭环展望);P5 补完整 real2sim 流水线;P8 改"照片级仿真已落地"。
- 架构图 `scripts/draw_arch.py` → `outputs/arch_diagram.png`:DiffusionDrive 图(a)风格,重点双脑,可直接用(字体/框线不重叠、无"自行插入"提示、不标创新点;三路输入用箭头颜色区分)。
- 轨迹叠图 `outputs/traj_overlay_114830.png`:在 `data/114830/.../1781149750414963.jpg` 上画多模路线(蓝/青/黄,只要路线、去掉红椭圆/红箭头),贴架构图"多模轨迹"框。
- 修 mp4 不可播:`pip install imageio-ffmpeg` → H.264/yuv420p/faststart;`sim_walk_discoverse.py` 已改。

**坑**:matplotlib 中文用 `Noto Sans CJK JP`;cv2 mp4v 浏览器播不了,要 libx264。

---

## 2026-06-20（周六）—— 打通 VGGT-Ω→3DGS→真实仿真器(能走进去)

**做了什么**(链路三段全跑通,见 plan `starry-twirling-ripple.md`)
- **A. VGGT-Ω→3DGS**:`scripts/train_3dgs_vggt.py`(改自 train_3dgs.py):读 `cameras.npz`(world2cam+K,对应688×384)、`recon.ply`(未对齐,与cameras同帧)降采样40万初始化、`frames_zed` resize到688×384监督 → 训出 `outputs/gs_vggto_114830.ply`(63.5万高斯,loss0.06)。自检从训练位姿渲=照片级室内走廊。
- **对齐**:`scripts/align_gs_world.py`:复用 bev_topdown 重力三件套 + `gaussian_renderer.transform_gs_model.transform_gaussian`,把归一化ply烘焙成**世界帧**(重力z-up + 用相机离地~1.2m锚定米制 k=9.92 + 地面落z=0)→ `gs_vggto_114830_world.ply` + `sim_world_114830.npz`(轨迹waypoints,跨度13×34.5m)。
- **B. 真实DISCOVERSE仿真器**:`scripts/sim_walk_common.py`(WalkEnv:动态MJCF=background体+差速机器人jx/jy/jyaw+车载fpv相机+地面+重力;headless EGL)、`scripts/sim_walk_discoverse.py`:机器人沿真实轨迹(去程,返程3DGS前视欠观测会糊)穿行 → `outputs/sim_walk_114830.mp4`(180帧照片级走廊穿行)。
- **C. 浏览器亲手驱动**:`scripts/sim_walk_server.py`:MJPEG流+stdlib http.server(无websocket依赖),`/stream`持续step仿真渲帧推流、`/cmd`收WASD设(v,ω)。实测44帧/2s(~22fps),机器人物理步进、位姿推进、实时画面=走廊FPV。

**关键坑/解法**
- MuJoCo headless EGL 报 `PLATFORM_DEVICE`:必须在 import mujoco/OpenGL **之前**设 `MUJOCO_GL=egl`+`MUJOCO_EGL_DEVICE_ID=0`+`PYOPENGL_PLATFORM=egl`(已写进 sim_walk_common.py 顶部)。
- 坐标系一致性:gs训练必须用 recon.ply(未对齐)配 cameras.npz,**不能用 recon_aligned.ply**(被重力旋转过,与外参不同帧)。
- GSRendererMuJoCo 只按 body pos/quat 摆高斯**不缩放**→ 米制缩放必须烘焙进ply。
- imageio无ffmpeg→视频用 cv2.VideoWriter(mp4v)。

**怎么用(用户亲手走)**
```
cd /data/DongBaorong/BEVLOOK/robot_bev_sim
PYTHONPATH=third_party/discoverse ~/discoverse_venv/bin/python scripts/sim_walk_server.py 114830 --port 8000
# laptop: ssh -L 8000:localhost:8000 ...@10.0.0.20 ,浏览器开 http://localhost:8000 ,WASD 开机器人走
```

**下一步**:更密帧VGGT-Ω提画质;返程糊→可只发布去程或补反向帧重训;接 DiffusionDrive/VLM 把仿真器当训练台。

---

## 2026-06-19（周五）—— 0619汇报PPT要点 + DISCOVERSE 导入自有3DGS资产跑通

**做了什么**
- 写 0619 向王老师汇报的 **PPT 要点**（`docs/汇报_0619_PPT要点.md`）。用户两次修正定调：**主线 = 「真实巡检场景 → (A)任务驱动双层架构 + (B)真实数据仿真环境」的因果逻辑**；采数据这段讲"场景刻画驱动设计"，**不是**讲"缺了哪些数据"（缺数据弯路挪到备份页）。P4 是核心页。
- 复盘分析了全部 6 个 session 的真实结构（见下"数据事实"）。
- **任务2：试用"上一代平台" DISCOVERSE 导入我们自己的资产**——确认 GS-Playground README 明写其上一代 = DISCOVERSE；clone（GitHub 直连被墙，用 `https://ghfast.top/` 镜像）；建 venv（`~/discoverse_venv`，`--system-site-packages` 复用系统 torch2.5.1+cu124）；装 `gaussian_renderer 0.2.0 + mujoco 3.9.0`。
- 写两个脚本把我们的 `gs_111450.ply` 喂进 DISCOVERSE 渲染栈（headless，纯 gsplat 不碰 MuJoCo/显示）：
  - `scripts/discoverse_render_gs.py`：环绕视角 → **噪点**（印证记忆#14：3DGS 只在训练视角附近有效）。
  - `scripts/discoverse_render_drivethrough.py`：**用真实 COLMAP 训练位姿** → **照片级穿行帧**（路面/黄标线/建筑/行人清晰）。

**产出**（`outputs/discoverse_gs_111450/`）
- `drivethrough.gif`（40帧）+ `drivethrough_sheet.png`（3×3实景拼图，**汇报可用**）、`contact_sheet.png`（环绕噪点，反面对照）。

**结论**
- ✅ **我们能把自己重建的场景当资产放进上一代 DISCOVERSE 渲染器并 headless 渲出照片级帧**——这就是"把自己东西放进去变资产"的可行性验证（GS-Playground 的端到端导入仍未开源，DISCOVERSE 这条先通）。
- `gaussian_renderer.core.batch_render(gs, cam_pos, cam_xmat, H, W, fovy, y_up)` 是纯函数渲染接口；约定：`Tmat=[cam_xmat|cam_pos]`=cam→world，`viewmats=inv(Tmat)`。要对上 gsplat/COLMAP OpenCV world2cam：**`cam_pos=cn=(c-center)/s`、`cam_xmat=R.T`、`y_up=False`**。
- DISCOVERSE 包 PEP660 装不了 editable → 不装它本体，**PYTHONPATH 指仓库根 + 只装运行依赖**即可（纯 Python 包）。

**下一步（明天）**
- 可选：把场景接进 DISCOVERSE 的 MuJoCo Simulator（建名为 `background` 的 body + `gs_model_dict={"background":绝对路径ply}` + headless EGL），放机器人/相机做交互。
- 用更干净的 114830 / VGGT-Ω 结果重复同一导入。

---

## 2026-06-18（周四）—— VGGT-Ω 跑通 + 调研 GS-Playground

**做了什么**
- 解决昨天的卡点：拿到 VGGT-Ω 的 HF gated 权重（`vggt_omega_1b_512.pt`，4.5GB）。
- 在 session **114830** 上跑通 VGGT-Ω 重建：图像 → 点云 + 相机位姿 → 重力轴对齐 → BEV 俯视。
- 写了三个脚本：`run_vggto.py`（重建）、`bev_topdown.py`（出 BEV）、`render_views.py`（出多视角 3D 图）。
- 调研真实感仿真框架 **GS-Playground**（清华 DISCOVERSE 团队，RSS 2026）：clone + 建好 venv，torch/motrixsim 可 import。

**产出**（`outputs/vggto_114830/`）
- `recon.ply` / `recon_aligned.ply`（重力对齐后点云）、`cameras.npz`、`recon.glb`
- `bev_topdown_rgb.png`、`bev_topdown_height.png`、`bev_occupancy.png`、`views_3d.png`
- 效果：连贯的走廊俯视，**显著干净于**之前的 COLMAP / octomap 结果。

**结论**
- VGGT-Ω 这条路对**单段室内走廊**有效，是目前最干净的重建，可作为汇报主线。
- GS-Playground 是**软件框架**（物理引擎 MotrixSim + 自研 3DGS 渲染器 GaussianRenderer），不是单独 app；自带的是机械臂/四足预制场景。
- 想用它跑**我们自己的数据**，需走 `gs-real2sim` 把采集重建成 3DGS 资产再导入，而“资产打包导入”官方**未开源** → 它是**后续方向**，非当前可交付。
- 本机 headless（无 `DISPLAY`），它的实时窗口 demo 无法直接弹窗，需 Xvfb 或离屏渲染才能看。

**模型架构讨论（已落文档）**
- 确定双层双脑架构，写入 [`MODEL_ARCHITECTURE.md`] v0.1：
  - 上层 VLM(Qwen3-3B) 场景识别 → **离散任务 id**；下层 **DiffusionDrive** 任务条件轨迹。
  - 核心 framing = 把 DiffusionDrive 从车端 AD 迁移到任务驱动型机器人导航；创新点 = VLM 任务 id 注入级联解码器。
  - 4 迁移难点 + 两阶段训练 + Orin 部署约束，详见该文档 §9 待办表。

**下一步（待定）**
- 等用户从**工控机**拷数据过来 → 填架构里 🟡 待定项（底盘/控制接口/传感器/Orin型号/任务标签集/已有软件栈）。
- 备选：重跑更密帧 VGGT-Ω（~150 帧/2s, 约 5-10 分钟）；仿真线 GS-Playground 仍暂停。

**卡点 / 风险**
- GS-Playground Real2Sim 导入未开源；headless 渲染未验证。
- 明天（06-19 周五）向王老师汇报 → 主线用 VGGT-Ω 结果，GS-Playground 仅作一页“后续方向”。

---

## 之前里程碑（简记，详见 RECONSTRUCTION_JOURNEY.md）

- **06-17（周三）**：决定从 COLMAP/octomap 路线转向 **VGGT-Ω + DISCOVERSE 3D 路线**；d405 是手臂相机弃用，场景用 zed；环境/runner/抽帧就绪，卡在 HF gated 权重。
- **06-16（周二）**：ROS2/RoboStack 装 octomap 对比，从前视数据出占据图“一坨”，证伪“换标准工具就干净”；COLMAP CPU 匹配崩溃修复（`OPENBLAS_NUM_THREADS=1`+brute force）。
- **06-15（周日）**：重建实战发现——只有 142522/114830 适合做室内地图；111450 无里程计只能轨迹回放；朝向须 gyro 投影到重力轴；distance 传感器无效。早期用 **gsplat** 训过 3DGS（`gs_111450.ply`）。
- **06-14**：搭 `robot_bev_sim` 脚手架；发现真实数据是 ROS2/RealSense D405（rosbag+JPG+imu），与文档假设的逐帧 PNG 完全不同。

## 2026-08-31 —— R2R追平立项: 差距诊断=训练投入 + ⭐大发现快脑不会转弯(数据分布缺陷) + 稠密jitter数据6片开跑

- **做了什么**: ①用户定调"R2R必须追平甚至超越DualVLN"→失败结构分解(r2rft 300ep: min_dtg<3m仅17% vs DualVLN 68%; 5-10m偏航53%=主矛盾; stop只值7个SR点)。②DualVLN配方调研(84k token): S2=Qwen2.5-VL-7B全量微调≈百万样本/64卡, 4种相机形态增广, stop=S2输出文本末帧监督×5上采样无独立头 → 差距=训练投入(我们6.6k样本)非架构。③**Phase0 oracle探针(probe_oracle_goal.py, GT前瞻4m目标+几何停): r2rft 18/18全500步失败 → 逐步debug实锤快脑不会转弯**: goal在右前39°模型输出恒直线wz≈0, 40步航向0变化。④根因定位: gen_adapt_data teleport回放每帧朝向=yaw_towards(下一路径点)→训练分布ego系未来路径恒为直线→模型学成"永远直走", goal_ego条件被忽略(部署时v5慢脑显式turn动作掩盖)。⑤写gen_dense_data.py(0.25m稠密插值~40帧/ep+70%±25°/30%±120°朝向jitter+横向±0.2m)重启6片生成(GPU2+3, ~11h); 写train_goalnav_r2r_dense.py(流式: BEV预计算bevs.npy fp16增量累积, rgb训练时解码; 旧全量缓存400k样本=680GB不可行)。⑥HFOV失真一并修(90→79, 训练/部署同为90故隐性, R2R渲染是79)。⑦InternData-N1官方数据尝试: HF gated 403/ModelScope残缺→全自渲染路线。
- **产出**: `gen_dense_data.py`+6片跑中; `train_goalnav_r2r_dense.py`; `probe_oracle_goal.py`; memory r2r-catchup-diag-steering-0831。
- **结论**: ①**R2R 5.3 vs 58.7 = 训练投入差距(100x数据+微调慢脑), 链本身无问题**(DualVLN我们链上repro 58.7已证)。②**条件通道必须有数据分布支撑才生效**(goal_ego被忽略=0803 mode位休眠同源教训)。③旧"r2rft微调OSR翻倍"结论要重新解读: 提升来自近goal帧见过的域适配, 不是真会导航。
- **下一步**: 6片收→precompute BEV→重训→oracle探针复测(转弯应恢复)→smoke10→300ep; s6/s7补2701ep; Phase2慢脑微调(Qwen2.5-VL-7B基座待下, 7B全量vs3B/LoRA待定)。
- **卡点**: 8卡全被他人占, 我们用GPU2+3缝隙(各~20GB); robo2vlm仍宕。

## 2026-08-31(晚) —— oracle探针终版: r2rft SR=0/OSR=10 快脑转向残疾定量实锤

- **结果**: probe_oracle_r2rft 50ep(GT前瞻4m目标+1.5m几何停, 排除慢脑/stop两变量): **SR=0.0/OSR=10.0/NE=9.99, 50/50全撞500步**。比部署版(v5慢脑, OSR17)还差=慢脑显式turn动作确实在替快脑打补丁, "快脑不会转弯"定量坐实。
- **dense进度**: 6片~27%(360-380/1353 ep, 已8.7万帧), ETA ~00:10; s6/s7未补(GPU6/7仍WuKefei占)。
- **下一步**: dense收→precompute BEV→重训→oracle探针复测(判据: OSR应大幅回升, 航向能跟随goal)。

## 2026-09-02 —— ⭐转弯接缝bug实锤(用户直觉) + dense快脑oracle SR 0→62 + 部署smoke10 A/B

- **做了什么**: ①用户质疑"不会转弯是不是我们改造出的问题"→写offline_turn_probe.py(训练管线自身输入,绕开全部部署接缝)分层判决: 旧r2rft方向一致86.5%但幅度7.8°(GT44.8°,数据病真)/新dense 96.5%一致+41.6°满血(重训成功); 但两ckpt经traj_to_control量化出turn都0/230(yaw通道输出≈0,转向全在横向位移被R2R无strafe动作空间丢弃)=部署接缝假阴性实锤。②连带bug: our_agent HFOV=90 vs dense训练79。③修复: our_agent加_traj_to_act(末端弯曲角±15°)+hfov实例参数; our_server透传hfov; probe同步修。④oracle探针复测50ep: dense SR62/OSR66/NE3.53 vs r2rft SR2/OSR18(对照,同接缝)=数据病+接缝病同时坐实。⑤部署链smoke10 A/B(本地Qwen VLM代robo2vlm): dense SR10/OSR30/NE9.97/SPL0.061 vs r2rft SR10/OSR20/NE10.03/SPL0.042。
- **结论**: ①快脑已不再是瓶颈(oracle 62, 参照DualVLN全链58.7,不对等但说明上限)。②部署层瓶颈回到慢脑+stop: 7/10两臂共同os=0(部分ep NE逐位相同=慢脑脚本化行为主导), dense ep2/ep4明显优于r2rft, ep9 r2rft停住dense没停(stop头对dense待查)。③⚠混淆变量: robo2vlm仍宕, 本地Qwen3.5跑v5地标链不在甜区(scan miss 8/8 skip landmark), 慢脑被Qwen降级。
- **下一步**: 慢脑层修复(robo2vlm恢复重测 or Phase2微调)才能兑现快脑增益; dense 300ep待拍板; stop头对dense ckpt复查; e2e模式的同款wz量化未修(独立实验)。
- **产物**: offline_turn_probe.py; probe_oracle_dense_v2/r2rft_v2.jsonl; results_our_dense_smoke10.jsonl; results_our_r2rft_smoke10_v2.jsonl; memory r2r-turn-seam-bug-0902。

## 2026-09-02(晚) —— ⭐薅DualVLN慢脑混合体SR70 + 四方配对对照 + Infinite VLM复活实测

- **做了什么**: ①dualvlm_slow_server.py(InternVLA-N1 S2 4bit, internvla env)+run_r2r_hybrid.py(像素目标depth反投影→世界目标缓存→dense快脑bend量化), 慢脑HTTP解耦跨env。②发现habitat env.episodes顺序进程间不确定→run_r2r.py/run_r2r_hybrid.py都加--ep_ids锁配对。③配对10ep四方对照(同一批ep)。
- **四方配对结果(同10ep)**: v5+dense(Qwen) SR10/OSR30/NE9.97 | v5+r2rft(Qwen) SR10/OSR20/NE10.03 | v5+dense(Infinite复活) SR0/OSR10/NE9.56 | **DualVLN慢脑+dense快脑 SR70/OSR80/NE4.08/SPL0.522**。hybrid随机10ep另测SR60/OSR60/NE2.22。
- **结论**: ①瓶颈100%在慢脑实锤——同快脑只换慢脑SR 10→70, 混合体达DualVLN全系统(58.7)量级。②v5零样本地标链(无论Qwen还是Infinite VLM后端)在这批ep上都不work, prompt工程路线到顶了, Phase2慢脑微调是唯一正路。③Infinite VLM复活但v5链上SR0(一轮还踩了VLM_URL指向被kill的本地Qwen的坑, 二轮显式指10.0.0.18)。
- **坑记**: ①跨进程评测子集必须--ep_ids显式锁; ②VLM_URL默认值已变成127.0.0.1, 远端必须显式传; ③InternVLAN1Net构造要走get_config; depth_anything ckpt相对路径要求cwd=InternNav根。
- **下一步**: Phase2慢脑微调立项(dense_data切(观测,指令)→决策监督样本, 7B全量vs3B/LoRA待定, 基座权重待下); hybrid 300ep全量可挂(DualVLN慢脑+我们快脑=当前最强形态, 论文对照行); robo2vlm需用户找管理员重启服务(机器活服务死→现已复活)。
- **产物**: dualvlm_slow_server.py; run_r2r_hybrid.py(--ep_ids); results_hybrid_smoke10/paired10.jsonl; results_dense_robo_paired10.jsonl; Qwen-RobotNav对标表(Table1 monocular R2R SR66.9(4B)/65.7(8B), panoramic72.1, 权重未开源)。
