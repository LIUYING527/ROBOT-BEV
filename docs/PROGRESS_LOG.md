# 进展日志（PROGRESS LOG）

> 本文件每次工作后追加一条，倒序排列（最新在最上）。
> 格式固定：**做了什么 / 产出 / 结论 / 下一步 / 卡点**。
> 全过程叙事见 [`RECONSTRUCTION_JOURNEY.md`]，本文件只记“每次”的增量。

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
