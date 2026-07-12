# SPRINT: generation memory policy

状态：done（第一阶段 build-time policy 已落地，commit 3bb1a33 在 main 上 — apply_generation_memory_policy + generation_memory_targets() 统一 5 个 family loader，架构/单元测试 19 passed；第二阶段 generation-wide report 已演进为独立 planned/SPRINT_memory_plan_full.md 的 L1，不计入本篇 scope）。

目标：把 generation 期间的内存控制从“每个模型 family 手工维护一份”改成按生命周期分层的统一 policy。第一阶段只收敛 build-time diffusion component memory，解决 VAE tiling/slicing 在 SD3.5、Wan、Anima、Cosmos Predict2 / Predict2.5 loader 里重复接线的问题；第二阶段只做 generation-wide memory report 和配置边界梳理，不创建一个到处 mutate 的大 manager。

## 0. 一句话

**policy 归 generation，mechanism 归 model。**

Model loader 负责加载、冻结、搬运 backend components；generation memory policy 负责把内存策略施加到这些 components 上，并把结果写进 runtime metadata。

当前散落形状：

```python
memory_metadata = apply_vae_decode_memory(
    pipeline.vae,
    memory_config=build.memory,
    owner="Wan VAE",
)
```

这个逻辑现在分别出现在 SD3.5、Wan、Anima、Cosmos Predict2、Cosmos Predict2.5 的 model loader 里。它不是 family-specific behavior；它是同一个 diffusion generation memory policy 对不同 backend object 的应用。

## 1. 现状证据

### 1.1 ModelBuild 已经携带 model.memory，不应该变成 parser

```python
@property
def memory(self) -> dict[str, Any] | None:
    """The whole ``model.memory`` block (consumer extracts its sub-block)."""
    return (self.model_config or {}).get("memory")
```

现有架构测试也明确要求：

```python
def test_runtime_interface_does_not_parse_model_memory_sections() -> None:
    """ModelBuild is a data contract, not a model.memory parser."""
```

因此本 sprint 不把 `ModelBuild` 扩成内存策略解析器。它继续只是 runtime build contract。

### 1.2 VAE policy 已有共享模块，但调用点仍然按 family 重复

现有共享模块：

```python
def apply_vae_decode_memory(
    vae: Any,
    *,
    memory_config: Mapping[str, Any] | None,
    owner: str,
) -> dict[str, Any]:
    """Apply ``model.memory.vae_decode`` and return bundle metadata."""
```

重复调用点：

```text
vrl/models/diffusion/sd3_5/model.py
vrl/models/diffusion/wan_2_1/model.py
vrl/models/diffusion/cosmos/anima/model.py
vrl/models/diffusion/cosmos/predict2/model.py
vrl/models/diffusion/cosmos/predict2_5/model.py
```

这说明第一层 abstraction 已经对了一半：`vae_decode_memory.py` 是正确的 policy primitive；缺的是一个更高一层的 generation memory application boundary。

### 1.3 不是所有 memory knob 都属于 model.memory

现有 sprint 已经给出边界：

```text
model.memory.vae_decode         -> VAE object memory policy; keep this boundary
rollout.sample_batch_size       -> denoise block batch size
rollout.denoise_compile         -> denoise block torch.compile policy
rollout.trajectory_storage      -> rollout artifact storage policy
Ray release_* flags             -> process/resource lifecycle policy
```

`decode_latents.batch_size` 也不应该烤进 model：

```text
policy 归 generation，mechanism 归 model。
```

本 sprint 延续这个结论：统一 memory policy 不等于把所有配置搬进 `model.memory`。

## 2. 决策

### 2.1 第一阶段：build-time diffusion memory policy

新增一个窄边界：

```text
vrl/models/diffusion/common/generation_memory.py
```

建议结构：

```python
@dataclass(frozen=True, slots=True)
class GenerationMemoryTargets:
    vae: Any | None = None
    pipeline: Any | None = None


def apply_generation_memory_policy(
    *,
    memory_config: Mapping[str, Any] | None,
    targets: GenerationMemoryTargets,
    owner: str,
) -> dict[str, Any]:
    ...
```

第一阶段只消费：

```text
model.memory.vae_decode.tiling
model.memory.vae_decode.slicing
```

实现上复用现有 `apply_vae_decode_memory(...)`。新模块的价值不是重新写 VAE 逻辑，而是把“哪个 backend object 接受 memory policy”这件事从每个 family loader 移出去。

### 2.2 Model 暴露 targets，不应用 policy

每个 diffusion model 增加一个小方法或属性：

```python
def generation_memory_targets(self) -> GenerationMemoryTargets:
    return GenerationMemoryTargets(vae=self.pipeline.vae, pipeline=self.pipeline)
```

Anima 这种 VAE 不在 `pipeline.vae` 上的 family 也能用同一合同：

```python
def generation_memory_targets(self) -> GenerationMemoryTargets:
    return GenerationMemoryTargets(vae=self.vae)
```

Model loader 只保留：

```text
load backend
freeze non-trainable modules
move components to device / dtype
return model
```

不再在 `from_build()` 内手动调用 `apply_vae_decode_memory(...)`。

### 2.3 Runtime builder 应用 policy，并统一 attach metadata

Family runtime builder 已经是模型构建后的统一边界。第一阶段改成：

```python
model = SD3_5Model.from_build(build)
memory_metadata = apply_generation_memory_policy(
    memory_config=build.memory,
    targets=model.generation_memory_targets(),
    owner="SD3.5",
)
```

然后 bundle metadata 继续通过现有 key 输出：

```python
metadata={
    ...
    **memory_metadata,
}
```

这样后续加 CPU offload、attention slicing、decode chunking metadata 时，只改 `generation_memory.py` 和 targets，不再扫每个 family loader。

### 2.4 第二阶段：generation-wide memory report，不做大搬家

第二阶段增加 report/facade，而不是全局 mutator：

```text
vrl/generation/memory.py
```

它汇总但不强行迁移配置来源：

```text
build_time:
  model.memory.vae_decode.*

execution_shape:
  rollout.sample_batch_size
  rollout.blocks.denoise.batch_size
  rollout.blocks.decode_latents.batch_size
  rollout.denoise_compile

artifact_storage:
  rollout.trajectory_storage

process_lifecycle:
  distributed.rollout.release_before_reward_model
  reward.release_after_score
  rollout.release_after_collect
```

Report 的用途：

```text
1. profile / smoke 输出一眼看清本次 run 的内存相关 knobs
2. 防止某个 family 偷偷引入自己的专用 memory key
3. 为未来 OOM auto-split / memory budget policy 留一个共同观察面
```

Report 不负责把 trajectory tensor 搬到 CPU，也不负责 release Ray actor。那些 mechanism 继续留在现有执行位置。

## 3. 命名

`ModelBuild` 的参数和局部变量统一使用 `build`。这个名字直接说明对象是模型构建输入，
并与已经表示运行期对象的 `RuntimeBundle` / `RuntimeModel` 保持清晰边界。

保留现有 public contract：

```text
ModelBuild
from_build(...)
```

原因：这是已有接口，重命名会产生大量和 memory boundary 无关的 churn。

新代码和新触达 call site 使用更具体的名字：

```python
def build_sd3_5_runtime_bundle(build: ModelBuild) -> RuntimeBundle:
    model = SD3_5Model.from_build(build)
    memory_metadata = apply_generation_memory_policy(
        memory_config=build.memory,
        targets=model.generation_memory_targets(),
        owner="SD3.5",
    )
```

命名规则：

```text
build       ModelBuild local variable inside runtime builders
memory      parsed or raw memory config only when the scope is already clear
targets     backend objects that policy may act on
owner       human-readable metadata/error prefix
```

不用：

```text
manager     太泛，暗示拥有复杂 lifecycle
context     不说明里面有什么
runtime     和 RuntimeBundle / RuntimeModel 现有概念冲突
component   只是在 target/object 后面加装饰词
```

## 4. 架构边界

### 4.1 应该改变

- 把 per-family `apply_vae_decode_memory(...)` 调用移到 `vrl/models/diffusion/common/generation_memory.py` 后面。
- 给 diffusion models 增加统一的 `generation_memory_targets()`。
- Runtime builder 在 model 构建完成后统一调用 `apply_generation_memory_policy(...)`。
- 更新 architecture test：family `model.py` 不应直接 import/call `vae_decode_memory`; 只有 `generation_memory.py` 能直接消费它。
- Bundle metadata 继续输出 `memory_policy`，但来源从 family loader 变成统一 policy boundary。

### 4.2 应该保持不变

- `ModelBuild.memory` 保持“返回整个 `model.memory` block”的 data contract，不解析具体 policy。
- `vrl/models/diffusion/common/vae_decode_memory.py` 保留。它是 VAE decode primitive，不是重复代码。
- `model.memory.vae_decode.tiling/slicing` 路径保持。它们是 build-time VAE object policy。
- `rollout.trajectory_storage` 的实际应用继续留在 collector/batch builder，因为它作用在 rollout artifact 上，不是 model build 行为。
- Ray release flags 的实际应用继续留在 `vrl/ray/resources.py` / generation runtime，因为它们是 process lifecycle 行为。
- Family runtime/model 文件保留。它们是 backend adapter 和 cross-family grep boundary，不为了减少行数而 flatten。

### 4.3 ALL_CAPS / thin function hygiene

保留：

```text
MEMORY_POLICY_METADATA_KEY
TORCH_COMPILE_MODEL_KEY
```

原因：它们是 metadata/schema protocol keys，属于真实边界。

不新增手写 duplicated key sets。`GenerationMemoryTargets` 不是配置 schema，不需要 allowed key set。未来如果新增 typed policy block，允许 key 必须从对应 dataclass fields 派生，沿用现有模式：

```python
frozenset(f.name for f in fields(VaeDecodeMemory))
```

不把 family `generation_memory_targets()` 压成数据表。这个薄方法是 backend adapter boundary：不同 family 的 VAE 位置不同，用方法显式暴露比靠属性猜测更稳。

## 5. 实施顺序

### T0. Add shared diffusion generation memory boundary

- 新增 `vrl/models/diffusion/common/generation_memory.py`。
- 定义 `GenerationMemoryTargets`。
- 定义 `apply_generation_memory_policy(...)`。
- 第一版只代理 `model.memory.vae_decode` 到 `apply_vae_decode_memory(...)`。
- 没有 VAE target 且配置了 `vae_decode` 时直接报错，避免 silent no-op。

### T1. Move SD3.5 / Wan / Cosmos model loaders to targets-only

- 删除 family model loader 内的 `apply_vae_decode_memory(...)` 调用。
- 删除 model constructor 里的 `memory_metadata` 字段，或仅在迁移期间保留但不再由 loader 写入。
- 添加 `generation_memory_targets()`。
- Runtime builder 统一 apply policy 并 attach metadata。

### T2. Architecture tests

- 更新 `tests/architecture/test_memory_policy_boundaries.py`：
  - 禁止 diffusion family `model.py` 直接 import `vae_decode_memory`。
  - 禁止 diffusion family `model.py` 直接调用 `apply_vae_decode_memory(...)`。
  - 保留禁止 inline `enable_tiling(...)` / `enable_slicing(...)`。
  - 保留 `ModelBuild` 不解析 `model.memory` 的测试。

### T3. Unit tests

新增：

```text
tests/models/diffusion/common/test_generation_memory.py
```

覆盖：

```text
vae target + tiling/slicing -> calls VAE methods and returns memory_policy metadata
no vae target + no vae_decode config -> empty/default metadata
no vae target + vae_decode config -> ValueError with owner
unknown vae_decode key -> existing ValueError still bubbles up
```

### T4. Generation-wide report draft

新增只读 report，不改变执行路径：

```text
vrl/generation/memory.py
```

第一版只汇总 metadata/config，不做 mutation：

```text
model.memory.vae_decode
rollout.sample_batch_size
rollout.blocks.*
rollout.trajectory_storage
distributed/reward release flags
```

Report 可以先只被 smoke/profile 工具消费；不要先接入训练主路径。

## 6. 验收

代码结构验收：

```text
rg "apply_vae_decode_memory" vrl/models/diffusion --glob model.py
  -> no matches

rg "enable_tiling\\(|enable_slicing\\(" vrl/models/diffusion --glob model.py
  -> no matches
```

测试：

```bash
python -m pytest \
  tests/models/diffusion/common/test_vae_decode_memory.py \
  tests/models/diffusion/common/test_generation_memory.py \
  tests/architecture/test_memory_policy_boundaries.py
```

回归：

```bash
python -m pytest tests/config/test_load_all_experiments.py tests/config/test_schema.py
```

行为验收：

```text
SD3.5 / Wan / Anima / Cosmos Predict2 / Predict2.5 configs with model.memory.vae_decode
still produce the same memory_policy metadata shape.

Adding a new diffusion family requires implementing generation_memory_targets(),
not copying apply_vae_decode_memory(...) into its loader.
```

## 7. 非目标

- 不重命名 `RuntimeBundle` / `RuntimeModel`；它们仍然准确表示构建完成后的运行期对象。
- 不把 `rollout.sample_batch_size`、`trajectory_storage`、Ray release flags 搬进 `model.memory`。
- 不实现 full physical stage pipeline。
- 不实现 OOM auto-tuner；这里只留出 future policy/report 边界。
- 不删除 family runtime/model adapters；一致的跨 family 形状比少几行代码更重要。

## 8. 参考

- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/common/vae_decode_memory.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/interfaces/runtime.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/sd3_5/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/wan_2_1/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/cosmos/anima/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/cosmos/predict2/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/models/diffusion/cosmos/predict2_5/model.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trajectory/storage.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/ray/resources.py`
- `/home/mingfeiguo/Desktop/wm-infra/tests/architecture/test_memory_policy_boundaries.py`
- `/home/mingfeiguo/Desktop/wm-infra/docs/sprints/parked/SPRINT_runtime_block_policies.md`
