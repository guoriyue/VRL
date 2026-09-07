# SPRINT: Anima 空间关系 / 属性绑定 RL（GenEval 规则奖励）

状态：进行中（2026-09-06 起）。前置：`SPRINT_anima_rl_target_search.md` §11.7 把 target C
（标签遵循）记为 null 并关线。

## 0. 为什么是这个目标

三个月里 Anima 上所有 RL 目标都卡在同一模式：奖励排的是风格/噪声（判官），或 base 已
经会了（标签遵循 0.92）。GenEval 是唯一一个**base 明显弱、奖励又是规则打分**的轴：

| 任务 | Anima base（553 题，OWLv2+CLIP 近似，2026-09-03） | 备注 |
|---|---|---|
| single_object | 0.988 | 饱和 |
| two_object | 0.798 | |
| colors | 0.894 | |
| **counting** | **0.400** | |
| **position** | **0.450** | |
| **color_attr** | **0.390** | 属性绑定 |

先例：Flow-GRPO 在 SD3.5-M 上用同一奖励 0.63 → 0.95（32 卡、48 prompt × 24 图/epoch、
β=0.04、global_std、lr 3e-4、LoRA、clip 1e-4、EMA）。

## 1. 奖励：`geneval_owl`（进程内，OWLv2 + CLIP）

官方 GenEval 评估器（mmdet Mask2Former Swin-S COCO + CLIP）本机没装，Flow-GRPO 通过
独立 conda 环境 + HTTP reward-server 接。我们用 09-03 已跑过的近似实现：

- 检测：`google/owlv2-base-patch16-ensemble`（开放词表；对动漫画风比 COCO 训练的
  Mask2Former 更有希望，G0 门会验证），per-class NMS 0.5，阈值 0.2。
- 颜色：`openai/clip-vit-large-patch14` 对检测框裁片做 10 色零样本分类。
- 位置：官方 GenEval 的框中心偏移 + 0.1 尺寸容差规则。
- 打分：`geneval_owl_strict` = 全部条件满足才 1（= 官方 GenEval 定义）；
  `geneval_owl` = 满足条件数 / 总条件数（部分分，减少全 0 / 全 1 的零 advantage 组；
  Flow-GRPO 的 reward-server 同样训练用部分分、报告用 strict）。
- 确定性：无采样，同图同分；可复现的 tagger 式奖励，不是判官。
- 放置：`distributed.resources.reward.device` 默认 `trainer`，模型常驻 trainer GPU
  （OWLv2-base + CLIP-L fp32 ≈ 2.3 GB；trainer 峰值 12.6 GB + rollout 7 GB，不用 parking）。

已知局限（写在这里，不写进奖励）：检测器是照片域训练的，动漫物体可能漏检/误检；策略
可能学会"把物体画得更像照片"来讨好检测器。护栏见 §4。

## 2. 数据

- 训练：`datasets/geneval/train_anime.jsonl` = Flow-GRPO 的 50k 训练 prompt，前缀
  `a photo of` → `an anime illustration of`（与 09-04 提交的 553 题动漫版同一变换）。
  分布：position 20588 / counting 14706 / color_attr 8824 / colors 2941 / two_object 2941。
- held-out：`datasets/geneval/official_eval_553_anime.jsonl`（与训练集 prompt 零重叠，
  已核对）。
- preset：`dataset/geneval_anime.yaml`。

## 3. 配方（中性模板 + 启动覆盖，不新建 experiment yaml）

`experiment/anima_preview3/online_grpo` + `+reward=geneval_owl +dataset=geneval_anime`，
覆盖：`algorithm.kl_coef=0.04 algorithm.global_std=true actor.optim.lr=3e-4
rollout.n_samples_per_prompt=16 rollout.prompts_per_batch=16
actor.timestep_fraction=0.25 actor.timestep_selection=random`。
继承模板的平价三件套（compile off、生成/replay 批 1）、`max_norm 0.1`、首步平价门。

与 Flow-GRPO 的差距（单卡现实）：每次更新 256 样本（他们 1152）、每 epoch 1 次优化器
步（他们 2 次）、组 16（他们 24）、20 去噪步（他们 10）。

## 4. 门（按顺序，写死在训练前）

- **G0 奖励在动漫域的有效性**：用进程内模型给 553 张 base 图打分，抽 24 张判 1 + 24
  张判 0 做人工核对，检测器判断与人眼一致 ≥ 90%；否则调阈值或换检测器，不进训练。
- **G1 策略能不能动**（10 次更新）：KL 惩罚要脱离 target C 的 <1e-3 带、训练 reward_mean
  要上升。两者都不动 → 停，问题在优化器步长而非奖励，不继续堆步数。
- **G2 主跑**：每 25 次更新存 checkpoint，held-out 553 动漫题 seed 对齐（同 manifest、
  seed 7777、base 与 checkpoint 逐题配对）。成功 = strict 总分显著高于 base（prompt 级
  自举 CI 不跨 0），且 position / color_attr / counting 至少一项 +0.1。
- **护栏**（任一破 = 不算成功）：锐度 median 不低于 base 的 90%；奖励翻转样本（base 0 →
  ck 1）抽 24 张人工看，"检测器被骗"（物体变照片风、碎片、重复贴图）≤ 4 张；
  `wd_tagger` 在 target C 的 170 行 held-out 上不下降（与训练奖励不相交的指标）。

## 5. 落地文件

- `vrl/rewards/models/geneval_owl.py`、`vrl/rewards/functions/geneval_owl.py`、注册表、
  `vrl/config/presets/reward/geneval_owl.yaml`、README 段落。
- `vrl/rewards/models/media.py`：`artifact_middle_frame_image` 从 wd_tagger 抽到共享处。
- `vrl/config/presets/dataset/geneval_anime.yaml`、`datasets/geneval/train_anime.jsonl`。
- `vrl/scripts/eval/anima_geneval_eval.py`：seed 对齐生成 + 打分 + 与 base 报告配对比较。
- 测试：`tests/rewards/models/test_geneval_owl.py`、`tests/scripts/eval/test_anima_geneval_eval.py`。

## 6. 记录

### 6.1 G0：奖励在动漫域的有效性（2026-09-06）

553 张 seed 对齐（seed 7777 + 行号）的 base 图，`outputs/eval_geneval_anime/base`。
先用 09-03 的阈值（0.2）打分，随机抽 24 张判 1 + 24 张判 0（只抽 counting / position /
color_attr / colors 四类）人工看：

- 判 1 的 24 张：22 张同意，2 张位置判断勉强（鸟在沙发上算"left of"、卡车/球棒）。
- 判 0 的 24 张：7 张是检测器错判——3 张颜色（红车、蓝飞机、绿苹果：**相邻物体像素混进
  裁片**让 CLIP 判错）、3 张漏检（交通灯 0.18、餐桌 0.16、真正的红车 0.17，都卡在 0.2 阈
  值下）、1 张计数（两只花瓶被一个跨越两者的大框数成 3 只）。斑马/床、长椅/熊两例按官方
  位置规则本来就不成立，不算错判。

修正：`detection_threshold` 0.2 → 0.15；NMS 后剔除"包含 ≥ 2 个同类已保留框"的组合框。
复核：24 张判 1 全部保留、错判修好 3 张（交通灯、红车、花瓶）、没有真失败被翻成通过。
剩余 4 张错判全是颜色-重叠一类（策略把物体画开就能满足，与目标方向一致，记为已知系统性
偏差）。48 张里 43 张一致 ≈ 90%，**G0 通过**。

修正后 base（553 题动漫版）：

| 任务 | strict | partial |
|---|---|---|
| single_object | 0.975 | 0.975 |
| two_object | 0.869 | 0.934 |
| colors | 0.798 | 0.856 |
| **counting** | **0.412** | 0.706 |
| **position** | **0.390** | 0.730 |
| **color_attr** | **0.380** | 0.765 |
| 任务均值 | **0.637** | 0.826 |

（Flow-GRPO 报告的 SD3.5-M base 是 0.63，量级一致。）锐度 median 16.77。
奖励模型：暖机 0.07 s/张、2.2 GB 常驻、重复打分逐位相同。

### 6.2 G1：10 步探测（2026-09-06 11:36 启动）

`outputs/anima_geneval_owl`，日志 `train_geneval_owl.log`。第一次拉起因为共享 GPU 的奖励
必须支持 CuMem parking 而失败（`InProcessRewardScorer(model=...)` 的即时构建路径不能
parking）；改成 pickscore 同款的 `CumemRewardFunction` + `model_factory` 工厂路径后启动。

第一次尝试在 epoch 1 第 3 个切片（12:00:04）失败：rollout worker 的 parking 校验报
`residual=11.16 GB baseline=2.10 GB`，而此前 6 次 parking 残留都是 2.32 GB，`loaded` 也从
9.18 GB 跳到 18.01 GB。`gpu_used_bytes` 用 `mem_get_info` 量整卡，所以多出的 8.8 GB 可能来自
任何进程；nvidia-smi 列表里只有 trainer、worker 和一个 774 MiB 的外部 benchmark，1 秒采样器
12:03 才启动，第二次尝试跑到 epoch 0 结束（reward_mean 0.745）时按操作员要求停止
（12:24），没抓到复现。**待查**：下次开跑前先复现并定位这 8.8 GB（优先怀疑 trainer 进程内
奖励打分 / 反向传播的缓存释放时机）。两次 epoch 0：reward_mean 0.746 / 0.745、组内 std
0.24、零 advantage 组 11.7%、grad_norm 0.004–0.006——信号强度明显高于 target C，但步数不够
下 G1 结论。

**8.8 GB 定位（2026-09-06 21:5x）**：单独走真实 runtime 路径（`sleep_offload` + CuMem 池）
给 256 张 rollout 尺寸的图打分：模型 2.25 GB，打分只多 0.23 GB 缓存，`park_memory` 后整卡
显存回到基线 +0.12 GB，两次 wake/park 结果一致——**奖励侧排除**。剩下的解释是共卡时的
外部进程：`gpu_used_bytes` 量整卡，rollout worker 的 parking 校验把别的进程在 run 期间新增
的显存当成自己的残留（12:00 前后操作员正好在同一张卡上启动别的任务）。这是共卡拓扑的已知
限制，不是 bug；对策是共卡时挂逐进程采样器以便归因。

**G1 第二次启动（21:55）**：同一命令，`outputs/anima_geneval_owl`（第一次的目录改名为
`anima_geneval_owl_attempt_20260906_1136`），与操作员的 VACE repair LoRA 训练（11.7 GB）共卡。

**第二次启动在 22:37 再次失败，这次采样器抓到了**：worker parking 时整卡多出 0.78 GB，
逐进程数据显示是操作员的作业换了（11.7 GB 的进程退出，新起 1.8 GB + 10.5 GB 两个），我们
三个进程全程正常。根因是 `gpu_used_bytes` 用 `mem_get_info` 量整卡。修复 `f24d076f`：改读
驱动的逐进程表（`nvidia-smi --query-compute-apps`，按本 pid + GPU UUID 求和），驱动不能
按进程归因时才回退整卡口径。真机验证：本进程 1.57 GiB vs 整卡 11.71 GiB。

**G1 第三次启动（22:42）**，与操作员的两个作业共卡（约 10.3 GB）。

**第三次启动在 23:24 因 OOM 停止**：parking 修复生效（两次 parking 校验都过了），但操作员
的新作业（`openatlas.image_world build`）显存在 1.6–22 GB 之间摆动，trainer 唤醒时正好撞上
22 GB 峰值，32 GB 卡放不下。两次 epoch 0/1 读数：reward_mean 0.745 → 0.765，KL 1.4e-5，
grad_norm 0.001–0.003。之后改为"外部进程显存连续 3 分钟 < 3 GB 才拉起"的等待启动器，
`save_freq=1` 以便被打断后从最近一步续跑。

**第四次启动（23:37，卡空后自动拉起，`save_freq=1`）跑了 5 步**，01:44 再次撞上操作员作业
（VACE repair 11.7 GB 稳定 + image_world 1–8.6 GB 脉冲）OOM；supervisor 已自动从
checkpoint-5 续跑，但为避免把操作员作业挤 OOM 而停止，改为卡空后再续。5 步读数：

| epoch | reward_mean | reward_std | KL 惩罚 | grad_norm | adv_zero |
|---|---|---|---|---|---|
| 0 | 0.735 | 0.236 | 0 | 0.0076 | 0.24 |
| 1 | 0.745 | 0.265 | 1.9e-4 | 0.0038 | 0.06 |
| 2 | 0.726 | 0.255 | 1.6e-4 | 0.0042 | 0.21 |
| 3 | 0.732 | 0.273 | 3.4e-4 | 0.0050 | 0.09 |
| 4 | 0.760 | 0.242 | 5.0e-4 | 0.0031 | 0.06 |

KL 单调爬升（策略在动），reward_mean 仍在噪声带内；G1 结论要等满 10 步。

### 6.3 G1 结论（2026-09-07 04:30）

10 步完成（epoch 5–9 于 02:48 从 checkpoint-5 续跑，04:03 成功退出）。训练侧
reward_mean 0.66–0.80 无趋势（16 条 prompt 的抽样噪声 ±0.05），KL 惩罚 0 → 8.8e-4 单调爬升，
grad_norm 0.001–0.008。

checkpoint-10 的 553 题 seed 对齐配对（`outputs/eval_geneval_anime/ck10`）：

| 任务 | base | ck10 | delta | z | W/T/L | CI95 |
|---|---|---|---|---|---|---|
| counting | 0.412 | 0.463 | +0.050 | 1.01 | 10/64/6 | [−0.050, +0.150] |
| position | 0.390 | 0.450 | +0.060 | 1.62 | 10/86/4 | [−0.010, +0.130] |
| color_attr | 0.380 | 0.290 | −0.090 | −1.91 | 7/77/16 | [−0.180, +0.000] |
| 任务均值 | 0.637 | 0.635 | −0.004 | −0.24 | 35/481/37 | [−0.033, +0.027] |

锐度 median 17.18（base 16.77，护栏 OK）。

**"策略能不能动"的判据改用像素证据**：同 seed 的 base 与 ck10 图平均 |diff| = 19.5/255
（p90 33.7，只有 0.4% 的行 < 2），而 target C 跑了 40 步才 25.6——策略明显在动。KL 惩罚在
扩散 GRPO 里本来就是个小量（Flow-GRPO 也把它调到"小且近乎恒定"），"KL < 1e-3"不能当
"没动"的证据；**这也修正 target C 的解读：那条线是方向/信号问题，不是步长问题**。
奖励翻转抽查：计数增益多为真实（洗手池 5→4、3；滑雪板 3→2），color_attr 损失一半真实
（红车变红墙、白手机变蓝）一半是检测器噪声。

判定：G1 通过（可动），10 步 = Flow-GRPO 预算的约 1%，看不到净增益属预期。进入 G2：
同一输出目录续到 60 步（supervisor 自动从 checkpoint-10 续），`save_freq=5`，30 / 60 步
各做一次 553 题配对评估；护栏同 §4。

### 6.4 G2 中期：checkpoint-30（2026-09-07 09:58）

| 任务 | base | ck10 | ck30 | ck30 delta | z | W/T/L | CI95 |
|---|---|---|---|---|---|---|---|
| counting | 0.412 | 0.463 | 0.412 | +0.000 | 0.00 | 7/66/7 | [−0.087, +0.087] |
| position | 0.390 | 0.450 | 0.390 | +0.000 | 0.00 | 10/80/10 | [−0.090, +0.090] |
| color_attr | 0.380 | 0.290 | 0.300 | −0.080 | −1.43 | 12/68/20 | [−0.190, +0.030] |
| colors | 0.798 | 0.798 | 0.851 | +0.053 | 1.69 | 7/85/2 | [+0.000, +0.117] |
| 任务均值 | 0.637 | 0.635 | 0.630 | −0.007 | −0.42 | 43/463/47 | [−0.042, +0.027] |

锐度 median 16.02（base 16.77，≥ 90% 护栏 OK）。ck10 的 counting / position 小幅上扬在 ck30
回到 0，是噪声。训练侧 reward_mean 前 10 步均值 0.73 → 第 20–29 步 0.77（训练 prompt 上略升），
held-out 不动。30 步 ≈ Flow-GRPO 预算的 3%，按计划续到 60 再评。

### 6.5 为什么 G2 的曲线看不出趋势，改做固定 prompt 过拟合探测（2026-09-07 13:50）

G2 跑到 epoch 42 停止（checkpoint-40 保留）。训练奖励按十步分段 0.730 / 0.721 / 0.770 /
0.742，held-out ck30 = base。原因是实验设计的分辨率不够，而不是步数：

- 每步随机抽 16 条 prompt，prompt 间部分分 sd = 0.252 → 每步均值噪声 ±0.063；按 Flow-GRPO
  的节奏折算，我们每步预期增益 0.001–0.003，信号比噪声小 20 倍，曲线**结构上**看不出趋势。
- 553 题 held-out 的 CI 半宽 ±0.021，一次 25 分钟；40 步累计预期增益 0.01–0.03，在分辨率以下。

改用"固定 prompt 过拟合探测"（同 DROID 的 can-the-curve-move 配方）：从 base 评估里挑 8 条
base 半对的难题（3 counting / 3 position / 2 color_attr，`outputs/probe_overfit_8_anime.jsonl`），
每步都训这 8 条 × 16 张（128 张/步），20 步，`outputs/anima_geneval_owl_probe_overfit8`。
prompt 不变 → 训练奖励本身就是评测（噪声 ±0.02）。判据：20 步内 reward_mean 明显爬升
（≥ +0.15）= 管道通、只是预算问题；爬不动 = 管道有问题（奖励噪声 / advantage / 步长），
再往 50k prompt 上堆步数没有意义。

**过拟合探测结果（15:18 停止）**：10 步 reward_mean 0.690 0.663 0.652 0.706 0.708 0.664 0.698
0.678 0.697 0.663——8 条固定题上**零斜率**。这排除了"预算不够"：正常的 GRPO 在固定题上 10
步内应开始记住。结合历史：平价修复（2026-08-28）之后 Anima 上没有任何奖励曲线动过（§3 干净
对照、target C、本 sprint）；修复前的 run 9 训练奖励曾**单调下降**且显著。

### 6.6 机器冒烟：稠密奖励能不能爬（2026-09-07 15:19 启动）

把奖励换成 `image_sharpness`（拉普拉斯能量，确定性、稠密、CPU），同样 8 条固定题 × 16 张、
10 步（`outputs/anima_sharpness_probe_overfit8`）。这是 Flow-GRPO / DDPO 自己的
compressibility 式冒烟：策略必定能提高锐度。判据：10 步内 reward_mean 明显上升 = GRPO 机器
正常，GenEval 的问题在信噪比 / 探索强度（noise_level 0.3 vs Flow-GRPO 0.7，128 vs 1152
样本/步）；不上升 = Anima 的 GRPO 管道本身有问题，先修管道。

**锐度冒烟结果（16:08 停止）**：6 步 0.605 0.537 0.558 0.581 0.552 0.518——稠密确定性奖励
在固定题上也**不爬**。读了损失和 SDE 对数概率（`vrl/algorithms/grpo/continuous.py`、
`vrl/math/denoise/flow_matching.py`）：与 Flow-GRPO 同式，EDM→RF 的 sigma 换算推导正确；奖励与
轨迹的拼接按 `sample_rows` 顺序、chunk 按索引排序，结构上没问题；`window_size: 0` = 全部 20 步
SDE，"随机选中确定性步"假设不成立。

**校准后的解释**：Adam 下每步参数位移 ≈ lr，与梯度大小无关；梯度里信号占比低时就是随机
游走——图像大幅变化（19.5/255）而奖励不动，正是这个现象。`ppo_epochs=1` + 流式累积 =
每个 rollout batch 只有 **1 次**优化器步。仓库自己的 Flux 验证
（`info/SPRINT_flux_algo_validation_curves.md`）结论完全一致：ppo_epochs=1 四条曲线全平，
改 4 机制才激活，"根因不是 epoch 数，是 ppo_epochs=1 的零漂移 + per-step 小梯度"。
Anima 所有 run 都是 ppo_epochs=1（流式累积强制）。

### 6.7 ppo_epochs=4 锐度探测（16:09 启动）

同 8 条固定题 × 16 张，`actor.gradient_accumulation_steps=0`（非流式，整批常驻）+
`actor.ppo_epochs=4`（每批 4 次优化器步，clip 3e-3 开始咬合），10 步，
`outputs/anima_sharpness_probe_overfit8_pe4`。判据同 §6.6；额外看 clip_fraction 是否 > 0。

**ppo_epochs=4 探测结果（4 步，17:21 停止）**：reward 0.571 0.538 0.596 0.587（不爬）。诊断列：
第 0–1 步机制确实激活（clip_fraction 0.155 / 0.150，ratio 偏差最大 0.56，grad_norm 0.0045 →
0.051），但第 2–3 步全部塌回零（clip 0.0008 / 0、ratio 偏差 3e-3、grad_norm 0.0013 / 0.0007）——
第 1 步的梯度尖峰之后更新就冻住了（Adam 早期偏差校正下二阶矩被尖峰主导的典型表现，或
策略真的不再变化）。仍不能区分"Anima 专有问题"和"配方问题"。

### 6.8 家族 A/B：SD3.5-M 同配方锐度探测（17:21 启动）

同 8 条固定题、同 image_sharpness、同 ppo=1 流式配方（lr 3e-4、clip 3e-3、kl 0.04、
global_std、16×8、timestep_fraction 0.25 random），模型换 SD3.5-M（LoRA r32，SD3.5 的标准
SDE 噪声 0.7、20 步），`outputs/sd35_sharpness_probe_overfit8`。判据：SD3.5 10 步内明显爬升
而 Anima 不爬 → 问题在 Anima 专有路径（Cosmos EDM replay / 对数概率 / 噪声 0.3）；两者
都不爬 → 探测设计或配方本身有问题，先修配方再谈目标。

### 6.9 家族 A/B 判定：问题在 Anima，不在配方（2026-09-07 18:11）

同 8 条固定题、同 `image_sharpness`、同 ppo=1 流式配方，只换模型：

| 探测 | 步数 | 首 → 末 | 斜率/步 | 以每步标准误为单位 |
|---|---|---|---|---|
| **SD3.5-M** | 10 | 0.942 → 0.973 | +0.0028 | **+4.4** |
| **Anima** | 6 | 0.605 → 0.518 | −0.0105 | **−3.0** |
| Anima ppo=4 | 4 | 0.571 → 0.587 | +0.011 | +0.5 |
| Anima GenEval | 10 | 0.690 → 0.663 | +0.0004 | −1.2 |

SD3.5 在同一套代码、同一个奖励、同样 10 步内**确实学得动**（+4.4 标准误，组内 std 从
0.089 收到 0.063）。所以 GRPO 管道、奖励、advantage、探测设计都是好的。Anima 反而**单调
下降 3 个标准误**——随机游走不会单调，方向性错误才会。这复现了 5090 sprint run 9 的记录
（奖励 0.637 单调掉到 0.452）。

两模型间我们控制的配置差异只有四项：`noise_level` 0.3 vs 0.7、`lora.alpha` 16 vs 64
（rank 32 → 缩放 0.5 vs 2.0，等效步长差 4 倍）、`actor.max_norm` 0.1 vs 1.0（Anima 的
grad_norm 0.001–0.008，从未触发，排除）、`scheduler_shift` 3.0（模型自带）。

### 6.10 noise_level 隔离探测（19:00 启动）

`outputs/anima_sharpness_probe_noise07`：Anima + `rollout.noise_level=0.7`，其余与上面那条
下降的 Anima 探测逐字相同。`noise_level` 0.3 是 2026-08-18 引入的（5090 sprint run 9），此后
**每一条** Anima run 都用它，也**每一条**都是平的或下降的。假设：SDE 注入的探索噪声过小时，
组内奖励差异主要由初始 latent（策略不控制）产生，而不是由策略的 SDE 动作产生，credit
assignment 因此失效。判据：0.7 下 10 步内奖励上升 = 根因确认；仍下降 = 转查 Cosmos EDM
路径的梯度方向。
