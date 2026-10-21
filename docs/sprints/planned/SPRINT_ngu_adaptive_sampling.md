# SPRINT: NGU 自适应采样 — 把 rollout 预算分配给尚未解决的问题

**日期**: 2026-09-17  
**状态**: PLANNED — 尚未实现；本文件记录方案与验收要求。2026-10-21 起 **blocked**。  
**解锁事件**：先在 GPU 上跑完 P0 的 OCR 基线（命令见文末“实施记录”），并且 owner 对 NGU 状态放在哪里做出决定（本文写的是 continuous producer，`SPRINT_rollout_admission_and_failure_attribution.md` §2/§5 写的是 `rollouts/admission/` 块）。两件事都完成后才开始 P1。  
**来源**: [Learning to Solve Hard Problems in RL for LLMs by Never Giving Up](https://arxiv.org/abs/2609.13443)，[作者博客](https://mnoukhov.github.io/posts/ngu/)。

## 目标与适用范围

实现默认关闭的 Never Give Up（NGU）采样策略：每个 prompt 先生成小组样本；
首组全成功则跳过本次训练，有成功和失败则形成训练组，全失败则以配置概率继续
采样。重试保留历史尝试，并设置资源预算，避免不可解问题无限占用算力。
跳过本次采样不等于永久删除该 prompt。

先在 OCR 任务上建立明确的文本匹配成功判定，并完成固定计算预算对比。
PickScore、美学分和一般视频连续奖励没有天然的“全对／全错”；其重试条件属于
后续方法扩展，不默认套用二值规则。不能把奖励方差很小直接解释成任务已解决。

论文的启示是区分已有能力的可靠性提升与难题突破，并审计真正参与更新的
样本分布；不预设 NGU 必然改善视觉生成任务。

## 当前代码与接入位置

以下基于本次代码阅读；开始实现前重新核对调用方、配置和测试。

| 位置 | 当前职责与计划 |
| --- | --- |
| `vrl/rollouts/orchestration/continuous/producer.py` | 已负责异步生成和评分收集。在 `_harvest_done()` 后引入采样决策，将重试与运行失败重试分开计数。 |
| 新增 `vrl/rollouts/orchestration/continuous/adaptive_sampling.py` | 管理逻辑 prompt 身份、重试决策、历史奖励统计、轨迹引用和预算；producer 驱动其状态转换。 |
| `vrl/rollouts/orchestration/continuous/consumer.py` | 当前 `_take_ready_groups()` 等待指定 batch 全部 slot 完成，且要求同一 policy version。NGU 需要新的有效组接收语义。 |
| `vrl/rollouts/orchestration/continuous/{schedule,thread,types,staleness,scored_queue}.py` | 接通补充 prompt、状态所有权、队列容量、实际生成版本和逐样本过期处理。 |
| `vrl/rollouts/orchestration/schedule.py` | 保留调度协议边界，明确 NGU 的适用模式与 capability 校验。 |
| `vrl/rollouts/collector/{core,requests,batch_builder}.py` | 复用生成、评分和组装接口；必要时补充尝试身份及 provenance，不在 collector 内藏训练策略。 |
| `vrl/algorithms/advantages.py` | 扩展历史 baseline 统计入口和过期样本后的 advantage 处理。 |
| `vrl/trainers/online/trainer.py` | `collect_training_batch()` 接收历史统计；检查变长逻辑组的 loss 权重、microbatch 和分布式同步。 |
| `vrl/rollouts/batch/core.py`、trajectory 类型 | 保留每次尝试的原始轨迹、行为策略 log-prob 和版本；避免为了逻辑分组强行拼接不兼容轨迹。 |
| `vrl/trainers/core/types.py` | 定义并验证显式配置，再投影到 continuous settings；默认关闭。 |

## 关键设计约束

### 1. 异步执行不等于自适应训练批次

现有 continuous consumer 要求当前 prompt batch 的每个 slot 恰好返回一组，
且整个 batch 同版本。仅在 producer 加重试循环仍会让训练等待最慢的 prompt。

NGU 必须允许已完成的有效逻辑组组成更新批次，未解决问题保留在有界重试队列，
跳过的简单题释放预算后补充其他 prompt。需要沿 `next_iteration()` 的调用链
核对数据迭代器、预取和 epoch 耗尽语义，不能仅放宽 consumer 的断言。
原有固定 batch 路径保持兼容；明确无有效组、预算耗尽和数据结束时的退出行为。

### 2. 身份、随机性与轨迹保真

- 用稳定的 example／采样周期身份关联历史，不能只用 prompt 字符串；相同文本
  可能有不同图像条件或 reward metadata。
- 每轮重试使用新的采样随机性，同时保留输入条件；记录 attempt 和实际生成版本。
- 同一逻辑组可以跨版本，但每个样本保留真实行为策略信息；禁止把旧样本统一
  标记为最新版本。复用算法对 off-policy staleness 的 capability 校验。
- diffusion 的 SDE window、时间步和 conditioning 等 replay 事实不能在合组时
  丢失。逻辑 advantage 分组与物理 replay 分块可以不同。
- 历史轨迹必须计入内存上限；退出、过期、取消和异常时释放。定义恢复时如何
  处理 pending 历史，禁止静默复用没有 provenance 的缓存。

### 3. 历史 baseline 与训练样本分开

按真实版本过滤过旧轨迹，允许保留其奖励计数用于组 baseline。
在二值奖励路径上实现并验证论文的 anchor-positive 处理：保留正样本优势，
调整仍参与训练的负样本优势，避免过滤旧负样本后无意改变组优势中心。
明确没有剩余负样本等边界行为。

当前 advantage 归一化／裁剪与论文 centered advantage 的区别必须显式处理；
不能把二值推导直接用于连续或多目标奖励。producer 不计算 advantage 或 loss。

## 执行计划

- [ ] **P0：建立基线。** 选择现有 OCR 实验，明确文本归一化和成功判定；固定
  独立评估集，按初始模型成功率分桶。记录每题成功率、实际训练组组成、零优势
  比例、生成／评分成本及所有丢弃样本。保存难度分桶规则和采样数。
- [ ] **P1：配置与调度。** 添加 NGU 开关、continuation probability、每次采样数、
  有界尝试／内存预算及过期策略；实现状态机、fresh/retry 公平调度和有效组补充。
  更新 producer/consumer/owner 协议及数据迭代器调用方，保持默认路径不变。
- [ ] **P2：历史训练信号。** 接入逐样本版本、历史奖励 baseline 与负样本 rescale；
  核对变长组的损失归一化、梯度累积、DDP/FSDP collective 次序和 replay 兼容性。
- [ ] **P3：比较与记录。** 在匹配计算预算下比较固定小组、固定大组和 NGU；
  使用多个随机种子，报告整体／各难度桶成功率、可靠性变化、成本和误差范围。
  配置与结果留在运行输出目录，结论写回本文件，决定是否扩大任务范围。

建议增加指标：各 prompt 尝试数、首次成功所需样本数、跳过／放弃数、有效组比例、
历史轨迹字节数、过期样本数、版本滞后、队列等待和每次成功的计算成本。
训练 reward 均值受采样分布变化影响，不能替代固定评估集。

## 验证与验收

- [ ] 状态机覆盖全成功、混合结果、全失败、概率退出、预算耗尽及零有效组退出。
- [ ] 两个相同文本但不同条件的 example 不混组；重试随机性变化且可复现。
- [ ] 可用组不被单个长重试 prompt 阻塞；数据耗尽、取消、weight-sync drain 和
  队列背压均能有界结束；评分服务错误不会被当成任务失败。
- [ ] 历史版本和 log-prob 保真；过期轨迹不进入 loss，允许的历史奖励统计保留。
- [ ] 用小型手算用例验证 baseline、anchor-positive 和变长组损失权重。
- [ ] 多 rank 不同重试次数、空组和不同组大小不会造成 collective 不匹配。
- [ ] NGU 关闭时现有 strict/continuous 行为与回归测试保持通过。
- [ ] GPU 实验将所有尝试、评分和丢弃成本计入预算；报告难题收益及简单题退化，
  不能只报告训练 reward 上升。无收益也如实记录，不据此默认启用。

## 架构取舍与非目标

**应该改变**：调度的有效组接收语义、历史状态和 provenance、advantage 历史统计
接口及相应配置。独立 `adaptive_sampling.py` 承载真实状态和共享决策，不是薄转发。

**保持不变**：family 生成接口、后端职责、replay evaluator 的概率计算，以及
collector 的生成／评分／组装边界。现有 schedule protocol 和 request adapter
虽薄，但提供协议边界及跨 family 一致性，保留。

**常量**：保留 `_DENOISE_FIELDS` 等 schema 字段集合和协议／metadata key 常量，
因为它们代表真实边界。NGU 概率、预算和成功阈值进入具名 typed config，避免
在 workflow 中新增 ALL_CAPS 参数表或任务词表。本 sprint 不顺带清理无关常量。

**非目标**：重写 GRPO loss、改模型架构、合并各 family 文件以减少行数、把 NGU
强加给 NFT/DPO、自动覆盖连续奖励语义、永久删除简单题，或默认开启 NGU。
跨 family 一致性优先于局部 LOC 缩减。

## 补充（2026-10-20）：PyTorch Conference 的相关输入

- **065 Modal（slime / PFSP）**：
  - 每个 prompt 只采 1 个样本，用"每难度一个 EMA baseline"代替 GRPO 组；
  - 按 `p(1−p)` 加权采样难度。
  - 本 sprint 的二元 OCR 范围可以拿 `p(1−p)` 作为重采概率的参考形式，但必须和"全对跳过、全错按概率重采"的现有设计比较后再选，不要叠加。
- **099 SkyRL**：继承 AReaL 的容量准入，生成前先申请名额，保证 `accepted + running ≤ (S + step)·B`。如果 NGU 的重采进入 continuous producer，预算上限应在 producer 侧准入，而不是消费时才丢弃。

## 实施记录（2026-10-21）

本轮没有改代码。原因见下表：P1–P3 被本文自己的顺序、admission sprint 的 knob
规则和一个未决的 owner 位置问题挡住；P0 需要 GPU。

### 现状核对（上文“当前代码与接入位置”表中的过期项）

- `continuous/staleness.py`、`continuous/scored_queue.py` 已不存在。continuous 现在只有
  `consumer.py`、`producer.py`、`schedule.py`、`thread.py`、`types.py`。
- consumer 没有 `_take_ready_groups()`。现在的接收入口是
  `ContinuousRolloutConsumer.collect_iteration` / `validate_ready_versions`
  （`vrl/rollouts/orchestration/continuous/consumer.py`）。“整批同一版本、每个 slot 恰好一组”
  的语义由 `PromptBatch`（`continuous/types.py`）承载。
- 准入记账已经有了：`vrl/rollouts/admission.py::AdmissionLedger`。每个组写一行
  `vrl.rollout-admission.v1`，包含 `prompt_key`（prompt 加输入 metadata 的哈希，所以文本相同、
  条件不同的 example 不会混在一起）、逐样本 reward、`rollout_policy_version`、
  `selected_rows`、decision/reason。P0 要的“实际训练组组成、零优势比例、所有丢弃样本”
  直接从这里读，不用新增记录。

### 各项状态

| 项 | 状态 |
| --- | --- |
| P0 基线 | **准备好，待 GPU**。成功判定已在代码里：`ocr_match` 是 `normalize_ocr_text` 之后的整句相等，取值 0/1（`vrl/rewards/models/ocr.py`）。独立评估集用 `manifests/ocr/test.txt`，训练集是 `train.txt`。命令见下；`--dry-run` 已在 CPU 上解析通过 |
| P1 配置与调度 | **不做（blocked）**。理由：①本文顺序是 P0 先于 P1，且“不预设 NGU 必然改善视觉生成任务”；②admission sprint §2 定了 knob 规则——没有 preset 使用或测量的旋钮不加；它的 §5 把 NGU 排在第 4 步，前面的第 2 步（用账本回看已有 run）和第 3 步（OCR 难度带固定预算对照）都还没做；③owner 位置冲突：本文写在 continuous producer，admission sprint 要求移到 `rollouts/admission/` 块并让 legacy 路径也能用，**需要用户决定**；④10-20 补充要求先把 `p(1−p)` 和“全对跳过、全错按概率重采”做对比再选，比较需要 P0 的每题成功率分布 |
| P2 历史训练信号（逐样本版本、历史 baseline、anchor-positive、变长组损失） | **不做**：依赖 P1 |
| P3 匹配预算对比 | **不做**：依赖 P1/P2，且需要多种子 GPU 运行 |
| 10-20 PFSP `p(1−p)` | **等 P0 数据**：要在 P0 的成功率分布上和现有设计对比，二选一，不叠加 |
| 10-20 SkyRL 容量准入 | **已有准入点，NGU 落地时接到这里**：`ContinuousRolloutProducer._admit`（`continuous/producer.py`）在提交前检查 `max_inflight_groups`。每个 `PromptBatch` 固定一个 policy version；`_store_result` 对超过 `max_stale_policy_versions` 的结果直接报错，不是消费时静默丢弃。NGU 的重采名额应该在 `_admit` 里计入，不要放到 consumer 侧 |
| “验证与验收”各条 | 全部依赖 P1/P2。只有最后一条（NGU 关闭时现有行为不变）现在自然成立 |

### P0 的 GPU 命令（未运行）

1. 先用 `experiment/sd3_5/online_grpo_ocr` 正常启动一次训练，得到 run 目录（里面有
   `resolved_config.yaml`）。这次训练的 `admission/*.jsonl` 也就是训练侧的组成记录。
2. 初始模型在独立评估集上每题采 16 张，只用 `ocr_match` 打分：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vrl.scripts.eval.image_checkpoint_eval \
  --run-dir <上一步的 run 目录> --base-only \
  --manifest manifests/ocr/test.txt --limit 256 --samples-per-prompt 16 \
  --eval-policy-override reward.kwargs.ocr.score_key=ocr_match \
  --output-dir outputs/ngu_p0_sd3_5_base_ocr_match
```

3. 按每题成功率分桶，分桶规则和采样数一起写进 `ngu_p0_buckets.json`：

```bash
.venv/bin/python - outputs/ngu_p0_sd3_5_base_ocr_match <<'EOF'
import collections, json, sys
from pathlib import Path
run = Path(sys.argv[1])
hits = collections.defaultdict(list)
for line in (run / "scores.jsonl").read_text().splitlines():
    row = json.loads(line)
    hits[row["prompt_index"], row["prompt"]].append(row["r_ocr"] == 1.0)
def bucket(rate):
    if rate == 0.0:
        return "unsolved"
    if rate == 1.0:
        return "solved"
    return "hard" if rate <= 0.25 else "medium" if rate < 0.75 else "easy"
prompts = [{"prompt_index": i, "prompt": p, "samples": len(f), "success_rate": sum(f) / len(f),
            "bucket": bucket(sum(f) / len(f))} for (i, p), f in sorted(hits.items())]
counts = collections.Counter(item["bucket"] for item in prompts)
report = {"success": "r_ocr == 1.0 (ocr_match: normalized whole-text equality)",
          "buckets": "unsolved p=0 | hard 0<p<=0.25 | medium 0.25<p<0.75 | easy 0.75<=p<1 | solved p=1",
          "bucket_counts": dict(counts), "prompts": prompts}
(run / "ngu_p0_buckets.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"prompts": len(prompts), "bucket_counts": dict(counts)}))
EOF
```

   这个分桶脚本已在合成的 `scores.jsonl`（20 题 × 16 样本）上跑通。

判定：`unsolved` 和 `hard` 两个桶加起来占比很小的话，NGU 在这个任务上没有可分配的
预算，就把这个结论写回本文。否则拿这个分布去比较 `p(1−p)` 和本文的重采规则，再决定
P1 怎么做。
