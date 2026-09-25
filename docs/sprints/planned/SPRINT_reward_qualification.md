# SPRINT: 奖励可用性认证（一个奖励"适合做 RL"到底要证明什么）

状态：**planned**（2026-09-25）。用户："i want to have a useful verifier for all my rewards. how I verify a reward is good for RL."

## 0. 一句话

"分数和人判断一致"只是必要条件之一。一个奖励能用于 GRPO，要同时证明五件事：**可复现、和人一致、钻不了空子、在训练采样下有组内差异、训练后人判断也在变好**。
其中第五件只能训练后验，前四件训练前必须过。把这五件做成每个奖励一张"奖励卡"，同一个命令、同一套关卡，学习型判官和可验证奖励各走各的路径。

## 1. 已经踩过的坑（每条对应下面一道关卡）

| 事故 | 缺的是哪一关 |
|---|---|
| object_move run4：ck20 学到了，ck40 奖励继续涨但盲评变差（放错位置↑、背景保住↓，Bench −0.067） | 训练后对照（关 5）；钻空子审计（关 3）没覆盖"移到别处" |
| local_edit keep 项：离线看着合理，盲评标签上判"改坏别处"只有 0.70，还误杀合法编辑 | 一致性关（关 2）必须用**本任务、本策略分布**的标注，不能借别的任务的数字 |
| EditReward：判"做没做"0.94，判"别处不动"0.65 | 一致性要**按维度**报，不能只报一个总 AUC |
| 512×4 探测：4 张图在 EditReward 眼里几乎一样（组内 sd 0.03–0.07） | 组内差异关（关 4）：AUC 高不等于有梯度 |
| ppo_epochs=1 / 12 vs 72 样本：曲线平不是奖励的错 | 关 4 要在**真实训练采样设置**（SDE、n 样本）下量，不是 ODE 探测 |
| 12 项奖励里 Cosmos+Kling 漂移在噪声内被当成学习 | 关 5 要有 CI，<2σ 不算 |

## 2. 两类奖励，两条路径

- **可验证奖励**（OCR 读字比对、RGBA 参考匹配、geneval_owl 数物体、目标 DINO 相似度这类"对着已知目标比"的）：
  有效性来自构造；要证明的是 **oracle 正确性**（合成已知答案的样本全对）、复现性和钻空子（能不能不做对也得分）。不需要偏好标注。
- **学习型判官**（EditReward、pickscore、hpsv3、aesthetic、kling、videoscore2、wd_tagger、ImgEdit_Judge……）：
  有效性只能靠**本任务标注**证明；上游 `reward` 包的 `fit`/`evaluate`（偏好对、源分组、holdout）就是为它们准备的。

## 3. 五道关（训练前四道，训练后一道）

**关 1 · 可复现**：同一输入重复打分，差异 ≤ 容差；服务/本地两种部署给同一分（上游 `Analysis.repeatability`、`qualify` 的原始轴一致检查已经做这个）。

**关 2 · 和人一致（学习型）/ oracle 正确（可验证）**
- 标注集必须是**这个策略自己生成的输出**（base 在训练采样设置下出的图，不是数据集的参考图），≥ 150 张，独立盲评，按维度标（做对 / 部分 / 没做 / 做错 / 重画；是否改坏别处；……）。
- 报**按维度**的排序 AUC：奖励想优化哪一维，那一维 ≥ 0.85；奖励声称覆盖但其实看不见的维度要明写"盲区"（EditReward 的盲区 = 改坏别处）。
- 用 `reward fit/evaluate` 时：偏好对按源图分组，calibration/holdout 不共享源图，报 holdout 准确率。
- 可验证奖励：合成 ≥ 50 个已知答案的样本（对 / 错 / 半对），oracle 全对。

**关 3 · 钻不了空子（stress / hack 审计）**
- 通用扰动（上游 `stress.build_stress_manifest` 已有）：模糊、噪声、棋盘格、全黑、alpha 破坏 —— 这些不能**涨分**。
- **任务专属作弊**（每个任务自己列，object_move 的 GATE A 是模板）：原图不动、整图平移/裁切、整图重画、把目标贴到别处、复制而不是移动、字变乱码……每种作弊的分数必须低于真正做对的样本（配对比较，报每种作弊的胜率）。
- 这一关的清单要在**训练后**回填：训练里出现的新作弊方式加进来（run4 的"放错位置"就是漏掉的一条）。

**关 4 · 训练采样下有组内差异**
- 在训练配方的采样设置（SDE、噪声、n 样本）下对 ≥ 60 条训练指令各出 n 张，报：base 成功率（要在 20–60% 带内）、有成有败的组的比例、组内奖励标准差、`adv_zero_rate`。
- 3 轮 dry run 顺便给出 parity / grad_norm / 显存 / 每轮耗时，这一关和 dry run 合并。

**关 5 · 训练后对照（奖励在涨 ≠ 在学）**
- 每 20 轮：held-out 上 base vs ck 出图，**盲评**（和关 2 同一套维度）+ 奖励差（bootstrap CI）+ 回归轴（和任务无关但该保住的：镜头、清晰度、文字）。
- 判定规则：奖励↑且盲评↑ = 学到；奖励↑盲评↔/↓ = 被钻空子（停，把作弊样本回填关 3）；都↔ = 平。<2σ 一律算"平"。

## 4. 产出物：每个奖励一张"奖励卡"

`docs/rewards/cards/<reward>.md`（或一个 JSON + 渲染页），固定字段：类型（可验证/学习型）、关 1–4 的数字与日期、盲区、任务专属作弊清单、已用于哪些 run 及关 5 结果。
没有卡的奖励可以跑 smoke，不能进正式 run；卡上任一关不过的奖励不能进训练键（可以留作观测）。

## 5. 复用什么、补什么

已有（上游 2026-09-22 合入）：`vrl.scripts.rewards.rescore_media`（独立重打分、内容哈希）、`reward analysis health/stress/ranking/paired/repeatability`、`reward review-export/import`（盲评页）、`reward fit/evaluate`（偏好对）、`reward qualify`（receipt）、`stress.build_stress_manifest`（通用扰动）、`image_checkpoint_eval`（同噪声出图、bootstrap CI）、rollout admission ledger（`adv_zero_rate`）。
要补：
1. **任务专属作弊清单**的声明格式 + 生成器（每个任务一个小函数：从源图/参考构造作弊样本），接进 `stress`。
2. **按维度的标注 schema** 和 `review-export` 的多维度版本（现在只有一维偏好）；Opus 盲评作为第一轮标注源，人工抽检 10%。
3. **奖励卡**的汇总命令：读 rescore/analysis/评测输出，填卡，判每一关。
4. **在采样设置下的组内差异探测**作为 `image_checkpoint_eval` 的一个报告项（现在要手算）。

## 6. 先拿哪些奖励开刀

按"正在用 / 即将用"排：EditReward（局部编辑，已有 400 张标签，关 2 可直接填）→ ImgEdit_Judge（候选"别处不动"判官，先过关 2）→ OCR（可验证路径的样板）→ pickscore / hpsv3 / aesthetic（正在跑的 SANA/anima 线）。

## 7. 非目标

- 不做"通用的奖励质量分"：每个奖励只对它声称的维度负责，卡上明写盲区。
- 不用奖励自己生成的分数当标注（GPT/VLM 判官的分不是人判断；上游 reliability 报告也这么写）。
- 不把 `qualify` receipt 当有效性证据：它只证明训练用的和离线验的是同一套打分。
