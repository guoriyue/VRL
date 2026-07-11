# SPRINT: Function-bag audit — 总览与结论

状态：in progress（2026-07-11）。核心模块的 artifacts 拆分与 dead-symbol sweep 已完成；
`scripts/diffusion/` 入口 body-level 审计已完成落地（2 个 FIX + 死协议面清除，见子
sprint §2.2），仅 perf fp8_math 重复 helper 待处置。子任务索引见 §5。

> 方法：22 个长期库核心文件，每个一个深度审计 agent（读全文 + grep 全仓调用点 + 按
> AGENTS.md 的"五种死代码形态 + thin-function 保留清单"逐符号定罪），每条"要改"判决再过一个
> **对抗性 verify agent**（专门反驳、找出让它其实该 KEEP 的理由）。另有一个 agent 扫
> `vrl/scripts/{perf,eval,data,diffusion}` 确认一次性生命周期。共 28 个 agent、~116 万 token。
> 偏置控制：用户历史上 revert 过两次过度扁平化，所以"错误的该改"比"漏报"更糟——凡是存在
> 跨家族一致性 / 协议边界 / lazy-import / 可独立测试的概念抽取等理由的，一律默认 KEEP。

## 0. 一句话

**"function-bag 遍地都是"这个直觉，被证据大幅推翻。** 22 个文件里 **15 个是真正内聚的**
（包括你点名的 `fsdp.py`），只有 **1 个真 grab-bag**（`trainers/data/artifacts.py`，你点名的
另一个 artifacts）+ **4 个零散小问题**（3 个死符号、1 个单调用者合并）。真正的系统性发现不是
"函数太散",而是**审计范围漏了一层**:`vrl/scripts/diffusion/` 下 7 个生产训练入口被当成
一次性脚本、从未按库核心审过。

## 1. 你点名的两个文件:一个平反,一个坐实

**`fsdp.py` —— 平反(cohesive-keep)。** "把这堆自由函数收进一个 FSDP applier 对象"的假设被
**证伪**:进程组生命周期那对函数(`init_/shutdown_training_process_group`)被 **FSDPStrategy 和
DDPStrategy 两个策略共用**(strategy.py:244/395 与 348/494)——收进 FSDP 对象反而破坏 DDP 复用。
其余每个函数都是一个独立的 collective 契约(mesh 构建、精度策略、wrapper 剥壳、model/optimizer
两对 gather/load),各自有活的非测试调用者、各自单元可测。它不是函数袋,是"把一个 handle 分片
并往返其全量 state"这一个关注点的最小 collective 层,坐落在 lazy-import 边界后。**不要动。**

**`artifacts.py` —— 坐实,但要认清是哪一个。** 你记忆里的"artifacts 问题"是
`vrl/trainers/data/artifacts.py`(451 行),它**确实是 grab-bag**:一个文件塞了两个零耦合的
落盘契约——① prompt-manifest 路径解析 + 溯源校验(和文件 docstring 相符),② SFT clean-latents
张量分片存取(`save_/load_sft_latents`,和上面不共享任何符号/常量/helper,docstring 里根本没提)。
拆分结果见 `../done/SPRINT_fbag_artifacts_split.md`。而 `vrl/utils/artifacts.py`(64 行)是干净的,唯一问题是
被 ① 抄了一份 `_coerce_data_root`。

## 2. 全量清单(22 文件 × 判决)

| 文件 | 判决 | 动作 |
|---|---|---|
| trainers/fsdp.py | cohesive-keep | 无(假设已证伪,见 §1) |
| trainers/checkpointing.py | cohesive-keep | 无(565 行=单一大职责,非袋) |
| trainers/activation_checkpointing.py | cohesive-keep | 无 |
| **trainers/data/artifacts.py** | **grab-bag-split** | **已完成**：拆 SFT-latents + 合并 `_coerce_data_root` → `../done/SPRINT_fbag_artifacts_split.md` |
| utils/artifacts.py | mostly-fine | 成为 `_coerce_data_root` 的唯一 owner |
| **utils/media.py** | minor | **已完成**：保留共享 `write_png`，删除 video_world 私拷 → `../done/SPRINT_fbag_dead_symbol_sweep.md` |
| utils/memory.py | mostly-fine | 无 |
| utils/config.py | cohesive-keep | 无(config 访问器 facade,合法) |
| config/validation.py | cohesive-keep | 无 |
| **config/builders.py** | minor | **已完成**：删死函数 `section_to_dataclass` → `../done/SPRINT_fbag_dead_symbol_sweep.md` |
| config/loading.py | cohesive-keep | 无 |
| config/precision.py | cohesive-keep | 无 |
| config/unknown_keys.py | cohesive-keep | 无 |
| trajectory/builders.py | cohesive-keep | 无 |
| trajectory/ops.py | cohesive-keep | 无(名字像袋,内容内聚) |
| trajectory/storage.py | cohesive-keep | 无 |
| **rollouts/collector/config.py** | minor | **已完成**：合并单调用者 `_has_sde_sampling` → `../done/SPRINT_fbag_dead_symbol_sweep.md` |
| **rollouts/batch/ops.py** | minor | **已完成**：删死函数 `shuffle_and_rebatch_batches` → `../done/SPRINT_fbag_dead_symbol_sweep.md` |
| nn/layers/attention/cache_rows.py | cohesive-keep | 无 |
| nn/modules/ar_attention_backends.py | cohesive-keep | 无(framework-adapter 跨家族一致形状,合法) |
| models/diffusion/build.py | cohesive-keep | 无 |
| models/utils.py | cohesive-keep | 无 |

## 3. 系统性发现:审计范围漏了生产入口层

sweep 判定 `vrl/scripts/{perf,eval,data}` 都是**正确的一次性生命周期**(procedural 函数袋在这里
合法,AGENTS.md 明说),但 **`vrl/scripts/diffusion/` 被错分**:它是生产训练/生成入口层,由 ~15 个
config preset 通过 `trainer.entrypoint` 点名字符串调度(`vrl.scripts.diffusion.<family>.train:...`)。
这 7 个文件是长期资产,却因为在 `scripts/` 下而被我这轮深度审计排除。跟进见
`SPRINT_fbag_scripts_diffusion_entrypoint_audit.md`。

另:`scripts/perf/common/fp8_math.py` 里 `amax_scale`/`tensorwise_fp8_matmul` 手抄了
`nn/quantization/fp8.py` 的量化核心序列(form-4/5),但它在一次性 perf 目录下,优先级低,并入上面那篇。

## 4. Non-goals(明确不做,及原因——防止下次又议)

- **不把 `fsdp.py` 收成对象**:pg 生命周期跨 FSDP/DDP 共用,收进去破坏复用(§1)。
- **不拆 `checkpointing.py`**:565 行是单一职责(checkpoint save/load/resume/RNG/state-seam),
  被两个真入口 + strategy seam + eval 消费,零死分支、无 gather/load 重复。大 ≠ 袋。
- **不动 `utils/config.py`、`nn/ar_attention_backends.py`、所有 `config/*` 与 `trajectory/*`**:
  分别是 config 访问器 facade、framework-adapter 跨家族一致形状、以及各自内聚的单关注点模块。
  这些正是用户以前 revert 扁平化时保护的"registry/convention 抽象"。
- **不为省行数合并任何 thin function**:15 个 cohesive-keep 里的每个 thin 函数都有命名理由
  (可独立测试的概念抽取、对称 read/write 公开对、递归必需的命名、跨策略共用)。
- **不 auto-derive `SOURCE_BACKED_VIDEO_WORLD_METADATA_FIELDS`**:它手写但注释说明了 tuple 顺序
  语义不可从 dataclass 派生——是 ALL_CAPS 规则里"合法保留"的一类。

## 5. 子 sprint 索引

- `../done/SPRINT_fbag_artifacts_split.md` —— **done**：trainers/data/artifacts.py 拆 SFT-latents + 合并 `_coerce_data_root`(唯一的真 grab-bag)
- `../done/SPRINT_fbag_dead_symbol_sweep.md` —— **done**：2 个死函数删除 + 1 个重复 helper 收敛 + 1 个单调用者合并
- `SPRINT_fbag_scripts_diffusion_entrypoint_audit.md` —— **入口审计完成**（2026-07-11）：2 个 FIX（wan-DPO 归一化反向、anima seed 虚构）+ definition seam 死协议面清除；余 perf fp8_math

## 引用

- 审计脚本:`scratchpad/fbag_audit.js`(一次性 workflow);逐 agent 结果 journal 在 workflow 转录目录
- 平反证据:`vrl/trainers/strategy.py:244,348,389,395,494`(pg 生命周期跨策略共用)
- 坐实证据:`../done/SPRINT_fbag_artifacts_split.md`；当前实现分别位于
  `vrl/trainers/data/artifacts.py` 与 `vrl/trainers/data/sft_latents.py`
- 范围漏洞:config `recipe/online/*.yaml` 的 `trainer.entrypoint` 指向 `vrl.scripts.diffusion.*`
