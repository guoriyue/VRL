# SPRINT: MiMo-V2.6 开源 RL 配方的借鉴评估

状态：**done（2026-09-27）：评估了四项，均未保留**。四项都先实现过，再因没有实测需求、只是防御性机器而撤回；
这里记下理由，免得重做。
来源：小米 MiMo-V2.6 技术报告（§4.2.1 监督评估、§4.3 组内评分）及其 verl fork `XiaomiMiMo/verl`
（`recipes/design/grader_service`、`recipes/design/webdev/group_reward.py`）。数据集 `MiMo-V2.6-RL-oss` 是 LLM agent
任务环境，对扩散模型无直接用处。

## 1. 单样本打分失败 → 屏蔽而不是整批抛错

现有生产奖励（HPSv3、PickScore、UnifiedReward 等）要么给分要么抛错，不会对单个样本弃权。为不存在的生产者在
reward → collector → trainer → 准入账本 → 离线评测五层维护一条每样本掩码通道不值。若将来某个 judge 的单样本弃权率
高到经常打断训练，先量弃权率，再在掩码、重试、整组中性分（MiMo 的 `group_ok=False`）之间选。

## 2. `gated_redistribution`（GAR：门决定符号，质量只在通过者间重分配正优势）

实测它与一行门控奖励 `safe * (1 + α·quality)` 再走默认组归一化给出相同的样本排序、几乎相同的优势值：λ 重分配与
cap 维护的"正优势总量守恒"会被随后的组 std 归一化抹掉（cap 从 2 改到 100 输出不变），`quality_floor` 就是 α，
`pass_below` 可由奖励组件自身定向代替。NSFW 合规率 97% 已饱和，门几乎总是全通过。需要硬门控时在奖励组合层加乘法门控即可。

## 3. `group_pick`：组内相对挑选的 VLM judge

动机是 Luna 逐点 judge 组内 60% 平分、38% 方差是噪声。但它从未在真实 VLM 上跑过，且把比较协议和评判准则
（写死的"整体视觉质量"提示词）绑在一个类里。若重做：准则走 `worker_config.rubric_path`（同 `unified_reward_video`），
rubric 哈希记进 diagnostics 与 reward card，先在 in-sampler 集上过 repeat / spread / 展示位置偏置检查
（MiMo 实测未旋转 judge 首位偏置 ±0.28），再进训练键。组身份已由 collector 的 `reward_group_id` 元数据提供。

## 4. rollout audit 资格门（组内奖励第一名是否盲评负例、末名是否盲评正例）

问题本身有价值，但仓库里没有盲标签，门跑不起来，而 reward card 为它另开了一条可选门路径。等有了盲标签、
reward card 真正开始用时再加。
