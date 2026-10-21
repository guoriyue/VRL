# SPRINT: 奖励可用性认证（一个奖励"适合做 RL"到底要证明什么）

状态：**planned**（2026-09-25；2026-10-21 按源码核对，结果见 §9）。用户："i want to have a useful verifier for all my rewards. how I verify a reward is good for RL."

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
  有效性只能靠**本任务标注**证明。原来为它们准备的 `fit`/`evaluate`（偏好对、源分组、holdout）已在 `f469d7055` 删除，现在用 `reward agreement` 或成对盲评的方向准确率（见 §5）。

## 3. 五道关（训练前四道，训练后一道）

**关 1 · 可复现**：同一输入重复打分，差异 ≤ 容差；服务/本地两种部署给同一分（上游 `Analysis.repeatability`、`qualify` 的原始轴一致检查已经做这个）。

**关 2 · 和人一致（学习型）/ oracle 正确（可验证）**
- 标注集必须是**这个策略自己生成的输出**（base 在训练采样设置下出的图，不是数据集的参考图），≥ 150 张，独立盲评，按维度标（做对 / 部分 / 没做 / 做错 / 重画；是否改坏别处；……）。
- 报**按维度**的排序 AUC：奖励想优化哪一维，那一维 ≥ 0.85；奖励声称覆盖但其实看不见的维度要明写"盲区"（EditReward 的盲区 = 改坏别处）。
- 用成对标签时：偏好对按源图（或 prompt）分组做 bootstrap，报方向准确率和区间（`docs/rewards/cards/` 里的卡都是这样报的）。
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

## 5. 复用什么、补什么（2026-10-21 按现状重写）

每道关现在对应的命令（都读 `rescore_media` 写出的 `Evaluation`，`vrl/rewards/evaluation.py`；子命令定义在 `reward/__main__.py`）：

| 关 | 命令 | 实现 |
|---|---|---|
| 0 打分 | `python -m vrl.scripts.rewards.rescore_media` | 独立重打分、内容哈希、可续跑；和训练用同一套 scorer transport |
| 1 可复现 | `reward repeat` | `Analysis.repeatability`（`reward/analysis.py`） |
| 2 和人一致 | `reward agreement --gate 0.85` | `reward/labels.py`：按维度的分类标签 + contrast，每个 contrast 报 AUC 和 bootstrap 区间；没过的 contrast 就是盲区 |
| 3 钻空子 | `reward stress-manifest` / `reward shortcut-manifest`，打分后 `reward stress` | `reward/stress.py`、`reward/shortcuts.py`，两者共用 `reward_stress` 元数据 |
| 4 组内差异 | `reward spread`、`reward health` | `Analysis.spread`：成功率带、有成有败的组的比例、组内标准差，以及每个 prompt 的 `success_share` / `within_std` / `score_range`；`health` 给每个 prompt 的极差和零极差 prompt 列表 |
| 5 训练后 | `vrl/scripts/eval/image_checkpoint_eval.py`、`reward paired`、`vrl/scripts/eval/score_report.py` | 同 seed 出图 + bootstrap CI；`paired` 做多 seed 配对，可用 `--stratify-by` 按 metadata 分层；`score_report --joint-min` 报联合通过率 |

训练里的 `adv_zero_rate` 和准入账本 `vrl/rollouts/admission.py`（每组的 reward 极值与去留）是关 4 在训练中的对应数据。
skill：`reward/skills/reward/{SKILL,qualification,scoring}.md`，`.claude/skills/reward` 是指向它的链接。

**已删除，不恢复**：
- `reward card`（`c14cd7f48`）：当时没写过卡，也没有代码读 `ready_for_training_key`。
- `fit/evaluate/apply`、`review-export/import`、`calibration.md`、`vrl/rewards/calibration.py`（`f469d7055`）：需要仓库里没有的盲偏好标签。
- `qualify` receipt（`5be40effa`，随训练侧的拟合组合绑定一起删除）。

五道关没有汇总成一个判定的代码，判定写在奖励卡里。

**奖励卡**：手写 Markdown，放在 `docs/rewards/cards/`（`f68a872fc`，2026-10-02），目前有五张，都是 Qwen-Image-2.1 文生图：
- `hpsv3_siglip_qwen21_t2i.md`：可进训练键，有关 5 结果；
- `qwen3vl_preference_32b_qwen21_t2i.md`：pairwise 模式可进训练键，pointwise 模式关 2 不过；
- `gpt56luna_codex_judge_qwen21_t2i.md`：只能做留出评测 judge，不能做组内 reward；
- `gpt_distilled_rm_qwen21_t2i.md`：还没认证；
- `offtheshelf_iqa_iaa_qwen21_t2i.md`：都不合格。

这些卡的关 2 用的都是成对盲评的方向准确率（组内 top-vs-random，或 base-vs-ck），由仓库外的一次性脚本计算，没有走 `reward agreement` 的分类标签路径。

**还没做**：§6 里 EditReward、OCR 的卡（需要 GPU 和本任务标签）。仓库里没有 ImgEdit_Judge 奖励的实现，只有 ImgEdit-Bench 的留出 manifest（`manifests/local_edit/heldout_imgedit_remove_add*.jsonl`）。

## 6. 先拿哪些奖励开刀

按"正在用 / 即将用"排：EditReward（局部编辑，已有 400 张标签，关 2 可直接填）→ ImgEdit_Judge（候选"别处不动"判官，先过关 2）→ OCR（可验证路径的样板）→ pickscore / hpsv3 / aesthetic（正在跑的 SANA/anima 线）。

## 7. 非目标

- 不做"通用的奖励质量分"：每个奖励只对它声称的维度负责，卡上明写盲区。
- 不用奖励自己生成的分数当标注（GPT/VLM 判官的分不是人判断；上游 reliability 报告也这么写）。
- 不把 `qualify` receipt 当有效性证据：它只证明训练用的和离线验的是同一套打分。

## 8. 补充（2026-10-20）：PyTorch Conference 证据与现状更正

**现状更正（HEAD a914644c6 核对）**：§5 里列的 `reward card`、`fit/evaluate`、`qualify`、`review-export/import`、`calibration.md` 都已删除。`c14cd7f48` 删除了 reward card，`f469d7055` 删除了离线偏好校准。现在的 `python -m reward` 只有 `health / stress / compare / paired / repeat / stress-manifest / shortcut-manifest / agreement / spread`（`reward/__main__.py:50-89`）。在代码里落地的阈值只有 `agreement --gate 0.85`，没有把五道关汇总成一个判定的地方。执行本 sprint 前，先把 §5 按现状重写。

**会上的三项补充**（004 Shopify judge 资格门、041 judge 置信度）：

1. **关 2 增加"标注员一致性上限"**：两名标注员对 ≥25 条随机样本做独立盲标，计算 Cohen's kappa。judge 和人的一致性不应高于人和人之间的一致性；kappa 太低，说明这个维度本身定义不清，应先修订标注协议，而不是继续调 judge。
2. **关 2 增加"历史 A/B 回测"**：对已有结论的 checkpoint 对比（盲评定过胜负的历史 run），检查 judge 的排序和历史结论是否一致。
3. **关 1 或关 2 增加"置信度校准"**：对能输出概率的 judge（例如 `qwen3vl_preference` 的 Yes 概率），按概率分箱，比较每箱的实际准确率（reliability diagram / ECE）。这一项对应 Luna judge 60% tie 的问题：置信度和准确率不相关的 judge，其组内差异大部分是噪声。

关 3 已经覆盖 Shopify 的"定向退化测试"（人为制造一种失败，确认只有对应维度下降），不另加。三项都只在下一个新 reward 上线时按实际需要加，并且加在 `reward/` 包已有的 `agreement` / `repeat` 报告里，不新建命令族。

## 9. 实施记录（2026-10-21）

逐项对照源码核对，基线是本地 main `546b0d871`。

| 项 | 状态 | 依据 |
|---|---|---|
| 关 1–4 的命令 | 已实现 | §5 表；CPU 测试在 `tests/reward/`（`test_analysis.py`、`test_labels.py`、`test_shortcuts.py`） |
| 关 5 的工具 | 已实现 | `image_checkpoint_eval`、`reward paired`、`score_report --joint-min` |
| §4 奖励卡 | 已有：`docs/rewards/cards/` 下五张手写卡；`reward card` 代码不恢复 | `f68a872fc`；删除见 `c14cd7f48` |
| §5 多维度分类标签的盲评页 | 不做 | `review-export/import` 已在 `f469d7055` 删除。之后的盲标签用仓库外的一次性标注页收集（`qwen3vl_preference` 卡里提到的 `label_page_hidden.json`），按"脚本是一次性的"规则不进库 |
| §6 EditReward / OCR 的卡 | 需要 GPU，未跑 | 命令见 `reward/skills/reward/qualification.md` §0–4，scorer YAML 见 `scoring.md` 里的 EditReward 例子。注意：`outputs/qwen_image_21_reward_study/visual_labels_*.json` 自己写明是"assistant visual inspection … not independent human labels"，不能当关 2 的标签 |
| §8 现状更正 | 本次完成 | §5 已重写；§2、§3 里 `fit/evaluate` 的引用改为已删除；`docs/rewards_offline_evaluation.md` 里"preference calibration"的过时说法已改 |
| §8.1 标注员一致性上限（kappa） | 未实现，需用户决定 | 见下 |
| §8.2 历史 A/B 回测 | 不需要新代码：现有 `reward paired` 就能做。打分要 GPU，未跑 | 命令见下 |
| §8.3 置信度校准（ECE） | 未实现，需用户决定 | 见下 |

**§8.1 和 §8.3 为什么这次不写**：
- §8 原文要求"三项都只在下一个新 reward 上线时按实际需要加"。
- 仓库里没有两名标注员各自独立给出的分类标签，也没有给概率型 judge 配好的分类标签。现有卡的关 2 标签都是成对偏好，而 `agreement` 读的是逐样本分类标签。现在动手写，就得先猜下一个 reward 用什么标签格式。
- 先例：2026-09-27 撤回 rollout-audit 门时（`5038f5004` 有记录），理由就是"没有盲标签，门跑不起来"。reward card 和偏好校准被删，也是因为没有消费者、没有标签。
- 现有的近似证据：
  - Luna 卡把 GPT 自身的顺序一致率 0.71（|gap| ≥ 2 时 0.91）当作蒸馏 RM 的可达上限；
  - hpsv3 卡里，三个模型 judge 在 30 对重叠样本上 26/30 一致。
  - 这两个数相当于"judge 之间的一致性上限"，但都不是两名人类标注员之间的 kappa。
- 真要加时放在哪里（不新建命令族）：
  - kappa：给 `agreement` 加第二份标签文件，在两份共有的 `sample_id` 上按维度算 Cohen's kappa，和 AUC 一起报；
  - ECE：对取值在 [0, 1] 的 axis 按概率分箱，每个 contrast 报每箱的正例比例和 ECE；
  - 如果下一个 reward 用的是成对标签，这两项应该加在成对报告上，而不是 `agreement`。

**§8.2 回测命令（需要 GPU，未运行）**。可以拿来对照的历史结论：
- hpsv3 卡关 5：`main_run1` 的 ck10 对 base，盲评偏好 72.5%；ck20 对 base 是 67.5%。
- Luna 卡：用户在 30 对里偏好 stage-1 ck30 的有 14 对；按 GPT rubric，anchored ck40 对 stage-1 ck30 是 22/30。

两个臂的 `media.jsonl` 必须是同一套 `sample_id`：prompt、seed 都相同，只有图不同。`paired` 会检查这一点。

```bash
python -m vrl.scripts.rewards.rescore_media --manifest <hist>/base/media.jsonl --config <judge>.yaml \
  --output-dir outputs/reward_evaluation/<judge>/backtest_<run>_base
python -m vrl.scripts.rewards.rescore_media --manifest <hist>/<ck>/media.jsonl --config <judge>.yaml \
  --output-dir outputs/reward_evaluation/<judge>/backtest_<run>_<ck>
python -m reward paired --baseline outputs/reward_evaluation/<judge>/backtest_<run>_base \
  --candidate outputs/reward_evaluation/<judge>/backtest_<run>_<ck> --axis <axis> \
  --output reports/<judge>/backtest_<run>_<ck>.json
```

判定规则：`directed_delta` 的符号和历史盲评结论一致，且 `source_bootstrap_95ci` 不跨 0，这一对才算回测通过；区间跨 0 记为"judge 分不出"。卡里的原图在 `/mnt/nvme/...`，这台机器没有挂载，所以这里只给命令。

**测试**：本次只改文档。`CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider -m "not gpu and not e2e" tests/reward tests/architecture tests/config/test_documented_launches.py` 结果是 27 passed、1 skipped（skip 的那条需要 `vdn-minimax-h3` submodule）。
