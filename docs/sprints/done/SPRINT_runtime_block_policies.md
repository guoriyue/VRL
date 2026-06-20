# SPRINT：Runtime block policies（done — resolved, negative result）

状态：**DONE / 已结案（negative result，2026-06-20 核实）**。这是一次 feasibility/perf spike，已得出并记录答案：
**当前 eager/no-compile 路径上 runtime block policies 没有性能收益**（T4 smoke：b16 比 b8 control 慢 ~3.9%），
故实验实现已**完整撤回**、零残留。核实证据：`vrl/generation/block.py` / `vrl/models/diffusion/common/block.py`
及对应 test 均不存在；`RuntimeBlockPolicy`/`TorchCompileBlockPolicy`/`runtime_blocks`/`block_policy`/
`decode_batch_size` 在 `vrl/`+`tests/` 零命中；各 family `decode_latents` 已回到无 batch 签名
（如 `vrl/models/diffusion/sd3_5/model.py:384`）。按 AGENTS.md「one-shot validation artifact」——价值是它产出的
答案（已记录），不是代码的继续存在，故归 `done/`。

目标：给 rollout 里的逻辑运行块提供统一配置面，让不同模型 family 可以声明并消费自己的 block policy。第一阶段只做 **serial executor 内的策略下发**，不做物理 stage pipeline、不加 Ray stage worker、不改 collector/reward 边界。

## 核心结论

这个方向作为 schema 形状仍然合理，但当前不应该继续作为 performance implementation 推进。
真实 SD3.5 OCR smoke 显示：`b16 denoise + decode_latents.batch_size=1` 能跑通，但同配置
`b8 decode1` control 更快。因此 block policy 目前只证明了容量/配置链路可行，没有证明吞吐收益。

当前决策：

```text
不要把 denoise batch 默认从 8 提到 16。
不要保留 runtime block implementation 作为当前代码路径。
保留本 sprint 作为未来需要容量旋钮 / physical pipeline 前的设计记录。
性能主线仍然是 rollout-only compile / warmup / recompile observability。
```

原始动机是：不要直接实现 `SPRINT_diffusion_rollout_stage_pipeline.md` 里的完整物理 pipeline，而是先把已经存在但分散的控制面收敛起来：

```text
rollout.sample_batch_size       -> denoise block batch size
rollout.denoise_compile         -> denoise block torch.compile policy
model.memory.vae_decode         -> VAE object memory policy; keep this boundary
```

这些都影响 diffusion rollout 的运行形态，但现在配置路径不统一。`tiling/slicing` 这类 VAE object 行为应该继续留在 `model.memory`，而 `denoise.batch_size`、`denoise.torch_compile`、`decode_latents.batch_size` 这类 rollout 执行策略应该统一进入 block policy。否则后续给 Wan / Cosmos / SD3.5 / AR family 扩展时会继续长出一批专用开关。

本 sprint 的方向是增加：

```yaml
rollout:
  blocks:
    denoise:
      batch_size: 16
      torch_compile:
        enable: true
        mode: reduce-overhead

    decode_latents:
      batch_size: 2
```

第一阶段等价映射：

```text
rollout.blocks.denoise.batch_size
  -> legacy rollout.sample_batch_size

rollout.blocks.denoise.torch_compile
  -> legacy rollout.denoise_compile

rollout.blocks.decode_latents.batch_size
  -> model runtime's VAE decode mini-batch
```

这样本来可以先解决一个真实问题：**denoise 想用更大 batch，但 VAE decode 可能 OOM**。但 smoke 之后的结论是：这个容量旋钮不能自动推出性能收益，必须先过同口径 profile gate。

## 为什么不是 stage pipeline

repo 里已经有 `ExecutionStage`，它现在是 planner-visible 的 sample chunk / profiler label：

```python
@dataclass(frozen=True, slots=True)
class ExecutionStage:
    """One planner-visible execution stage and profiler label."""

    prompt_index: int | None = None
    sample_start: int | None = None
    sample_count: int | None = None
    batch_group_key: tuple[Any, ...] = ()
```

它不是 `prompt_encode -> denoise -> vae_decode` 这种模型内部阶段。继续把这个新配置叫 `stage` 会制造概念冲突。

现有 diffusion executor 也已经不是完全没边界的 monolith：

```python
encoded = self.encode_prompt_for_chunk(...)
state = self.prepare_denoise_state(...)
denoise_result = self.run_denoise_steps(...)
chunk_result = self.decode_denoise_result(...)
```

所以第一阶段不需要创建 `vrl/generation/stages/*`，也不需要把这些方法包一层空 wrapper。需要的是给这些逻辑 block 一个稳定、可验证的 policy 配置面。

## Block 命名

用 `block`，不用 `stage`。

原因：

```text
stage
  已经被 ExecutionStage 占用，语义是 planner chunk/profiler label。

block
  表示 runtime 内部可施加策略的逻辑执行块。
  它不承诺物理 worker、队列、placement、relay。
```

Diffusion 第一批 block：

```text
prompt_encode
prepare_sampling
denoise
decode_latents
```

第一阶段只消费：

```text
denoise.batch_size
denoise.torch_compile
decode_latents.batch_size
```

其余 block 名先作为 profiling/reporting 对齐，不先加无用配置。

AR 未来可能的 block：

```text
prefill
decode_step
image_decode
```

不要为了 AR 未来形状先实现 AR 配置。先让 diffusion 的最小可用策略跑通。

## 配置边界

推荐最终配置：

```yaml
rollout:
  blocks:
    denoise:
      batch_size: 16
      torch_compile:
        enable: true
        mode: reduce-overhead

    decode_latents:
      batch_size: 2
```

保留 legacy 配置作为 alias：

```yaml
rollout:
  sample_batch_size: 8
  denoise_compile:
    enable: false
    mode: default
```

解析规则：

```text
1. 新路径 `rollout.blocks.*` 优先：该字段一旦由新路径设置,就采用它,完全不看 legacy alias。
2. legacy 路径(`sample_batch_size` / `denoise_compile`)仅在新路径未设该字段时兜底生效。
3. 不做新旧冲突报错。原因:base config 永远带 `sample_batch_size` / `denoise_compile` 默认,
   OmegaConf 合并后无法区分「用户显式设的 legacy」和「base 继承的默认」;若做冲突检测,任何
   用新路径的 experiment 都会撞上继承默认而误报,使新路径不可用。故采用「新赢、legacy 兜底」。
4. legacy 只在被真正采用(新路径未设)时才做 family 支持校验。
```

`model.memory.vae_decode.tiling/slicing` 保持在 `model.memory` 下，因为它们是 model build 时施加到具体 VAE object 的内存行为，不是 rollout scheduling 行为。

`decode_latents.batch_size` 是 rollout block policy，因为它决定一次 decode 几个 rollout rows。它**不**烤进 model：和 `denoise.batch_size` 一样走 `executor_kwargs` 到 generation executor，executor 在调用时把它作为参数传给 `model.decode_latents(latents, decode_batch_size=...)`。model 只暴露 decode 能力（含 `ChunkedLatentDecoder` 机制），batch 大小这个 policy 由 generation 在调用时决定，不再由 model 自存 `pipe.decode_batch_size`。**policy 归 generation，mechanism 归 model。**

## Source of truth

新增一个窄 parser，不新增大 framework：

```text
vrl/generation/block.py
```

建议数据结构：

```python
@dataclass(frozen=True, slots=True)
class TorchCompileBlockPolicy:
    enable: bool = False
    mode: str = "default"


@dataclass(frozen=True, slots=True)
class RuntimeBlockPolicy:
    batch_size: int | None = None
    torch_compile: TorchCompileBlockPolicy | None = None
```

允许 key 必须从 dataclass fields 派生，不手写一份 `_ALLOWED_KEYS`：

```python
frozenset(f.name for f in fields(RuntimeBlockPolicy))
```

这是为了避免配置 schema 和类型结构分叉。现有 `VaeDecodeMemory` 已经用这个模式：

```python
_VAE_DECODE_KEYS = frozenset(f.name for f in fields(VaeDecodeMemory))
```

block 不靠自动从源码里猜。正确来源是：

```text
FamilyCapability.execution_stages
executor 真实方法边界
family 明确声明哪些 policy 可用
```

也就是说，自动化只发生在 parser 侧：

```text
read family-supported blocks
validate rollout.blocks keys
reject unsupported block
route policy to the right runtime hook
```

判断一个东西是不是 block，用这几个条件：

```text
1. 有稳定 profiler / method boundary
2. 能独立接受 runtime policy
3. policy 不改变其他 block 的语义
4. family 显式声明支持它
5. 不是太小的内部 op
```

family 支持的 block 不要做全局大表。每个 capability 或 executor metadata 声明自己支持哪些 block：

```text
diffusion supports:
  prompt_encode
  prepare_sampling
  denoise
  decode_latents

AR supports later:
  prefill
  decode_step
  image_decode
```

Diffusion 的初始映射：

```text
prompt_encode
  from ExecutionStageCapability("prompt_encode")
  current policy: none / report-only

prepare_sampling
  from ExecutionStageCapability("prepare_sampling")
  current policy: none / report-only

denoise
  from ExecutionStageCapability("denoise_step")
  policy: batch_size, torch_compile
  note: user policy is named denoise because it controls the denoise loop /
        sample chunk shape, not one timestep in isolation.

decode_latents
  from ExecutionStageCapability("decode_latents")
  policy: batch_size
```

`reward_artifact` / `collector.reward_score` is a collector boundary today, not a
generation runtime block in this sprint. Keep reward scoring outside generation
until a later physical pipeline sprint proves that moving it is useful.

AR future mapping can follow the same rule without prebuilding AR policy code:

```text
prefill
  from ExecutionStageCapability("prefill")
  possible future policy: batch_size / torch_compile

decode_step
  from ExecutionStageCapability("decode_step")
  possible future policy: backend / kv_cache / batch_size

vq_decode
  from ExecutionStageCapability("vq_decode")
  possible future policy: batch_size
```

不支持的 block 出现在配置里要 fail fast。

## Prior art：sglang-omni 的 StageConfig

`~/Desktop/sglang-omni`（`sglang_omni/config/schema.py`）已经收敛到本 sprint 想要的 config 模式，可作为 schema 纪律的直接参照。它的核心是一个**统一的 per-unit 策略 schema** `StageConfig`——每个 stage 同一个形状，靠 `factory` / 字段值区分，而不是 per-family 散开关：

```python
class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")   # 未知 key 直接拒绝
    name: str
    factory: str
    factory_args: dict[str, Any] = ...
    runtime: StageRuntimeConfig = ...           # 嵌套 typed 子策略
    parallelism: ParallelismConfig = ...
    relay: RelayConfig | None = ...
    ...
    def model_post_init(self, _ctx=None):
        # tp_size vs parallelism.tp：都设且冲突 -> 报错点名两者；只设一个 -> 自动 reconcile
```

对应关系（说明本 sprint 的方向是成熟模式，不是新发明）：

```text
统一 RuntimeBlockPolicy（非 per-family 开关）  ~ 统一 StageConfig
frozenset(fields) 拒绝未知 key                 ~ ConfigDict(extra="forbid")
嵌套 TorchCompileBlockPolicy                   ~ 嵌套 RelayConfig/ParallelismConfig/StageRuntimeConfig
新+legacy alias reconcile                      ~ model_post_init 的 tp_size↔parallelism.tp reconcile
```

也就是说：**统一 typed schema + 拒绝未知 key + 嵌套 typed 子策略 + alias reconcile** 这套，sglang-omni 已经在用。本 sprint 抄的是这套 schema 纪律。差异是这里的新路径必须赢过 inherited legacy defaults，因为 OmegaConf 合并后无法可靠区分「用户显式 legacy」和「base 默认 legacy」。

校准（关键，决定 phase-1 边界）：`StageConfig` 之所以重（`next` / `wait_for` / `merge_fn` / `stream_to` / `gpu` / `process` / `relay` / `fused_stages`），是因为 sglang-omni 本身就是**物理多进程 pipeline**，那 ~90% 字段全是“跨进程 stage 之间怎么连”。它的通用性是被一个真正通用的 runtime 挣来的。本 sprint phase-1 **没有**物理 stage，所以：

```text
抄：  统一形状 / extra=forbid / alias reconcile / 嵌套 typed 子策略
不抄：next / wait_for / relay / process / gpu / fused_stages 这些物理 stage 字段
```

`RuntimeBlockPolicy` 现在只放执行 intent（`batch_size` / `torch_compile`），相当于 `StageConfig` 里 `runtime` / `factory_args` 那块，不引入 topology / transport / placement。等真正做物理 pipeline 时，`StageConfig`（含 `relay_backend: shm|nccl|nixl|mooncake`、`fused_stages`、`runtime_overrides`）才是该照抄的目标形状——那是另一个 sprint。

## Implementation Plan

### T0：增加 block policy parser

新增：

```text
vrl/generation/block.py
tests/generation/test_block.py
```

职责：

```text
parse `rollout.blocks`
derive allowed policy fields from dataclasses
validate batch_size is positive int when set
validate torch_compile block shape
merge legacy aliases with new-path-wins semantics
reject unsupported block names by family capability/metadata
```

Acceptance：

```text
rollout.blocks.denoise.batch_size parses to RuntimeBlockPolicy(batch_size=...)
rollout.blocks.denoise.torch_compile parses to TorchCompileBlockPolicy(...)
unknown policy key is rejected
unknown block name is rejected for the selected family
legacy-only config still works
new+legacy same value works
new path wins over inherited legacy defaults
```

### T1：把 denoise batch size 接到 block policy

当前 Ray launcher 只读：

```python
sample_batch_size = cfg_path(cfg, "rollout.sample_batch_size", None)
kwargs["sample_batch_size"] = int(sample_batch_size)
```

改成：

```text
resolve rollout block policies
read denoise.batch_size
fall back to rollout.sample_batch_size
write executor kwargs sample_batch_size
```

Acceptance：

```text
rollout.blocks.denoise.batch_size reaches DiffusionChunkExecutorBase.default_sample_batch_size
rollout.blocks.denoise.batch_size reaches request.sampling["sample_batch_size"] for planner chunking
legacy rollout.sample_batch_size remains compatible
new path wins over legacy sample_batch_size when both are present
existing runtime input tests pass
```

### T2：把 rollout-only compile 接到 denoise block policy

当前 launcher 只读：

```python
_ROLLOUT_COMPILE_CFG_PATH = "rollout.denoise_compile"
compile_cfg = cfg_path(cfg, _ROLLOUT_COMPILE_CFG_PATH, None)
```

改成：

```text
read rollout.blocks.denoise.torch_compile
fall back to rollout.denoise_compile
continue writing runtime model_config through torch_compile_model_config(...)
```

保持这个边界不变：

```text
rollout config path:
  rollout.blocks.denoise.torch_compile

runtime model build payload:
  model_config["torch_compile"]
```

Acceptance：

```text
rollout-only compile still applies only to worker runtime build payload
driver replay model is not compiled by this setting
families without supports_torch_compile still fail fast
legacy rollout.denoise_compile remains compatible
TORCH_LOGS=recompiles profiling path remains documented in SPRINT_rollout_performance.md
```

### T3：给 decode_latents 增加 batch_size policy

现有 latent decoder 已支持 `decode_batch_size`，现在由调用方在调用时传入：

```python
def decode_latents(self, latents, *, decode_batch_size=None):
    ...
    LatentDecodeSpec(..., decode_batch_size=decode_batch_size)
```

现有 VAE memory policy 只包含：

```python
class VaeDecodeMemory:
    tiling: bool = False
    slicing: bool = False
```

本阶段不要把 `batch_size` 加进 `VaeDecodeMemory`，因为 `tiling/slicing` 是 VAE object memory behavior，而 `decode_latents.batch_size` 是 rollout block policy。

实现（decode batch 由 generation 驱动，不烤进 model）：

```text
launcher: rollout.blocks.decode_latents.batch_size -> executor_kwargs["decode_batch_size"]
executor: 调用 model.decode_latents(latents, decode_batch_size=self.default_decode_batch_size)
family model.decode_latents(latents, *, decode_batch_size=None) 把它喂给现有 ChunkedLatentDecoder
不再有 pipe.decode_batch_size baking，也不再走 model_config["runtime_blocks"]
```

Acceptance：

```text
SD3.5 decode_latents uses rollout.blocks.decode_latents.batch_size
Wan/Cosmos existing tiling/slicing behavior unchanged
chunked latent decode parity test covers batch_size
invalid batch_size <= 0 is rejected
unsupported family/block combination errors before rollout starts
```

## Phase-1 implementation attempt（已撤回）

下面记录的是已经撤回的实验实现范围，不是当前代码状态：

```text
vrl/generation/block.py
  统一解析 rollout.blocks
  从 RuntimeBlockPolicy / TorchCompileBlockPolicy dataclass fields 派生允许 key
  从 FamilyCapability.execution_stages + capability flags 派生 family 支持的 block policy
  合并 legacy rollout.sample_batch_size / rollout.denoise_compile
  新路径赢过 inherited legacy defaults，legacy 仅兜底

vrl/rollouts/collector/config.py
  rollout.blocks.denoise.batch_size -> request sampling sample_batch_size
  保证 planner chunking 和 executor kwargs 使用同一个 denoise batch policy

vrl/generation/ray/launcher.py
  rollout.blocks.denoise.batch_size       -> executor_kwargs["sample_batch_size"]
  rollout.blocks.decode_latents.batch_size -> executor_kwargs["decode_batch_size"]
  rollout.blocks.denoise.torch_compile     -> model_config["torch_compile"]
  注：torch_compile 走 model_config 是因为它是 model-build 时编译 transformer 的配置，
      不是 per-call runtime 旋钮；batch 类 policy 全部走 executor_kwargs。

vrl/generation/diffusion/executor.py
  base 新增 default_decode_batch_size（来自 executor_kwargs）
  decode_denoise_result 调用 model.decode_latents(latents, decode_batch_size=...)
  chunk context 和 engine_counters 都写入 diffusion_decode_batch_size

diffusion family executors / models
  6 个 family executor __init__ 接受 decode_batch_size -> self.default_decode_batch_size
  SD3.5 / Wan / Cosmos Predict2 / Predict2.5 / Anima 的 decode_latents 加 decode_batch_size 参数，
    喂给现有 ChunkedLatentDecoder；不再读 pipe/self.decode_batch_size

已删除（model_config["runtime_blocks"] 注入通道整个废掉）
  vrl/models/interfaces/runtime.py: RUNTIME_BLOCKS_MODEL_KEY / runtime_blocks_model_config / RuntimeBuildSpec.runtime_blocks
  vrl/models/diffusion/common/block.py: 整个文件（apply_decode_latents_batch_size 等）
  各 family runtime: apply_decode_latents_batch_size(...) 调用 + pipe.decode_batch_size baking

vrl/generation/execution/worker.py
  runtime_debug ray chunk metrics 携带 chunk engine_counters / stage_durations / peak_memory_mb
```

实验实现当时的 YAML 迁移状态：

```text
现有 experiment YAML 仍保留 legacy rollout.sample_batch_size / rollout.denoise_compile。
命令行只设置 rollout.blocks.denoise.* 时，新路径赢过 legacy 默认值；不需要把
rollout.sample_batch_size override 成 null。

重要修复：早期 smoke 曾把 rollout.sample_batch_size=null 用来绕开旧冲突逻辑，结果
GenerationRequest.sampling["sample_batch_size"] 变成 None，scheduler 在 int(None)
崩溃。实验实现当时修在 source of truth：collector config projection 会把
rollout.blocks.denoise.batch_size 投影成 request sampling 的 sample_batch_size；
scheduler/planner 也对显式 None 做 fallback。

如果未来重新启用这条方向，再单独做 YAML migration，把默认配置整体迁到 rollout.blocks，
减少双路径心智负担。
```

历史验证：

```text
pytest tests/generation/test_block.py \
       tests/generation/ray/test_runtime_config.py \
       tests/models/diffusion/common/test_block.py \
       tests/models/diffusion/common/test_latent_decode.py \
       tests/rollouts/runtime/test_runtime_inputs.py

pytest tests/config/test_load_all_experiments.py tests/config/test_schema.py

ruff check changed implementation/test files
python -m py_compile changed implementation files
```

T4 smoke（SD3.5 OCR，bf16，真实 Ray rollout）：

```bash
python -u -m vrl.scripts.train \
  --config experiment/diffusion/sd3_5/online_grpo_ocr \
  trainer.total_epochs=1 \
  trainer.output_dir=outputs/sd3_5_ocr_block_policy_b16_decode1_20260608_222142 \
  trainer.save_freq=999999 \
  eval.enable=false \
  trainer.precision_drift_guard.mode="'off'" \
  trainer.debug.first_step=true \
  actor.gradient_accumulation_steps=1 \
  rollout.n=16 \
  rollout.rollout_batch_size=1 \
  rollout.blocks.denoise.batch_size=16 \
  rollout.blocks.decode_latents.batch_size=1
```

结果：

```text
exit code: 0
request sample_batch_size: 16
executor kwargs: {"sample_batch_size": 16, "decode_batch_size": 1}
legacy rollout.sample_batch_size in resolved YAML: 8

runtime_debug chunk:
  chunk_key: prompt:0:samples:0:16
  stage_id: ...:chunk:p0:s0:n16
  diffusion_sample_batch_size: 16
  diffusion_decode_batch_size: 1
  diffusion_num_denoise_steps: 10
  diffusion_rollout_transformer_dtype: bfloat16
  diffusion_denoise_mode: sde
  peak_memory_mb: 17690.0703125
  stage_durations_s:
    encode: 0.145s
    prepare_latent: 0.005s
    denoise: 7.272s
    decode: 0.642s

metrics.csv:
  group_size: 16.00
  trained_prompt_num: 1
  reward_mean / r_ocr: 0.4141
```

同配置 b8 control：

```text
output: outputs/sd3_5_ocr_block_policy_b8_decode1_control_20260608_222142
rollout.n: 16
rollout.blocks.denoise.batch_size: 8
rollout.blocks.decode_latents.batch_size: 1

b8 decode1:
  chunks: 2 x 8
  recorded stages total: 7.760s
  recorded stages / sample: 0.485s
  denoise / sample: 0.444s
  peak_memory_mb: 16874.567

b16 decode1:
  chunks: 1 x 16
  recorded stages total: 8.064s
  recorded stages / sample: 0.504s
  denoise / sample: 0.455s
  peak_memory_mb: 17690.070
```

结论：实验实现可以让 denoise 用更大的 sample batch（16）同时让 VAE decode 按 1
micro-batch 运行；planner、executor kwargs、runtime payload、chunk counters 四处都一致。
但同配置 control 下 b16 比 b8 慢约 3.9%，所以这不是当前 eager/no-compile 路径的性能改进。
因此实现已撤回，记录保留在 sprint 里。

### T4：Profiling gate

验证目标是回答：

```text
Can denoise use a larger batch while decode_latents uses a smaller batch?
Does larger denoise batch improve wall time or SM utilization after bf16/fp16 + compile?
Does decode mini-batch remove the VAE OOM observed at larger sample_batch_size?
```

Run matrix：

```text
precision: fp16 or bf16 production path
compile: rollout.blocks.denoise.torch_compile.enable=true
denoise.batch_size: 8, 16, 24, 32
decode_latents.batch_size: 1, 2, 4
distributed preset: ray_rollout or ray_rollout_cross_node
```

Record：

```text
rollout wall time
generation.denoise_step
generation.denoise_forward
generation.decode_latents
peak_memory_mb
OOM/retry count
NCU duration-weighted top GEMM / attention Compute(SM)
TORCH_LOGS=recompiles status
```

Success criteria：

```text
larger denoise batch no longer fails because of VAE decode OOM
compile graph is reused after warmup
no sustained recompiles
decode_latents batch size is visible in runtime metadata or engine counters
```

当前 gate 结果：

```text
capacity wiring: passed in the experimental implementation
performance improvement: failed for eager/no-compile b16 vs b8
current action: do not ship block implementation; keep as future design record
```

If denoise batch still OOMs after decode mini-batch, the blocker is denoise activations / trajectory buffers / CUDA graph pool, not VAE decode. Then this sprint stops and the next work is denoise memory accounting, not physical stage pipeline.

## What Should Stay Unchanged

Keep `ExecutionStage` unchanged. It is a planner/profiler boundary for sample chunks, not a runtime block policy object.

Keep `model.memory.vae_decode.tiling/slicing` unchanged. Those are model-build memory knobs applied to a concrete VAE object.

Keep `torch_compile_model_config(...)` as the runtime payload writer. The rollout config path can change, but the runtime build schema should still have a single writer for `model_config["torch_compile"]`.

Keep current collector/reward boundary unchanged. Reward scoring still happens after generation output is returned; this sprint does not move reward into generation.

Keep `forward_chunk_plan()` as the serial coordinator. Do not create duplicate `run_*_stage()` wrappers just to match the new names.

## Non-goals

```text
不做物理 stage pipeline
不新增 Ray stage workers
不新增 placement / relay / bounded queue runtime
不把 reward_score 移进 generation executor
不把 block policy 做成全局大表
不为尚未消费的 block 预留空文件或空 adapter
不重写 DiffusionExecutor 的现有逻辑边界
不把 `decode_latents.batch_size` 混进 `model.memory.vae_decode`
```

## References

```text
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/execution/planner.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/diffusion/executor.py
/home/mingfeiguo/Desktop/wm-infra/vrl/generation/ray/launcher.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/common/latent_decode.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/common/vae_decode_memory.py
/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/sd3_5/model.py
/home/mingfeiguo/Desktop/wm-infra/docs/sprints/info/SPRINT_rollout_performance.md
/home/mingfeiguo/Desktop/wm-infra/docs/sprints/parked/SPRINT_diffusion_rollout_stage_pipeline.md
```
