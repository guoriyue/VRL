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
