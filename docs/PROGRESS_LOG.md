# 进展日志（PROGRESS LOG）

> 本文件每次工作后追加一条，倒序排列（最新在最上）。
> 格式固定：**做了什么 / 产出 / 结论 / 下一步 / 卡点**。
> 全过程叙事见 [`RECONSTRUCTION_JOURNEY.md`]，本文件只记“每次”的增量。

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
