# SPRINT: 让 composition root 拥有 generation gatherer

状态：**DONE（2026-07-30，CPU-only）**。

## 根因

`ChunkExecutorBase.gather_chunks()` 位于 family-neutral execution 层，却通过 lazy import
重新读取 `vrl.families.registry`：

```python
gatherer = get_model_family_entry(self.family).new_gatherer()
```

与此同时，`RayGenerationLauncher.from_cfg()` 已经持有同一个 `ModelFamilyEntry`，并为 driver
executor 构造 gatherer。这让一项 registry-owned binding 出现两个 construction site，也让
neutral base 通过字符串 family identity 反向寻找 owner。

## 落地

1. `RayGenerationLauncher.from_cfg()` 仍是 registry-backed construction site：
   `RayGenerationLaunchInputs` 保存唯一显式构造的 gatherer。
2. launcher 把同一个 typed launch input 交给 Ray worker；Ray 的 actor serialization 为每个
   worker 复制该无状态对象，但不会重新解释 family identity。
3. `RayGenerationLaunchInputs` 在 driver composition boundary 验证 launch contract、
   `ChunkGatherer` protocol、方法可调用性与整包 pickle；`GenerationWorkerCore` 在独立 worker
   process boundary 保留同样的 type/protocol 复验，再把 gatherer 作为 executor constructor
   kwarg 注入。两层验证对应两个可独立构造的真实边界，不是重复 helper。
4. `ChunkExecutorBase` 只保存并调用注入对象，不再 import `vrl.families`。没有 gatherer 的
   executor 仍可执行单个 `forward_chunk_plan()`；一旦调用 request-level `gather_chunks()`，
   会 fail loud。
5. rollout preview 与 real-checkpoint e2e 是另外两个持有 `ModelFamilyEntry` 的 composition
   root，因此也显式调用 `entry.new_gatherer()` 后注入。
6. `ChunkExecutorBase` 不再 nominally inherit consumer-facing
   `GenerationChunkExecutor(Protocol)`；具体类以 structural conformance 满足 worker boundary，
   不依靠 MRO 覆盖 Protocol stub。

永久测试覆盖：

- 注入对象就是 executor 使用的对象；
- request-level execution 缺少 gatherer 时抛出明确错误；
- real Ray worker build 保留同一个 gatherer identity；
- architecture gate 禁止 `executor_base.py` 重新 import `vrl.families`。

## 保持不变

- 保留 `ChunkGatherer` 薄 protocol。它是 collector 与 family-specific output assembly 之间的
  真实协议边界。
- 保留各 family/binding 的薄 gatherer class。diffusion、chunk-autoregressive denoise 与
  token-autoregressive payload 的排序、trajectory 和输出形状不同，不能 data-ize 成一张表。
- 保留 `ModelFamilyEntry.new_gatherer()`。它是 dotted-string registry 的 lazy import adapter，
  construction site 已收敛，但 adapter 本身仍有真实职责。
- 保留 `GenerationWorkerCore` 对 family registry 的读取。worker composition 仍需要 registry
  构建 model 与 executor；本 sprint 只移除 neutral executor base 的反向 lookup。
- 保留 `GenerationChunkExecutor` 的两成员 structural protocol，不把 constructor wiring 塞进
  consumer-facing contract。

## ALL_CAPS 与薄函数判决

本 sprint 不新增 ALL_CAPS 数据。registry 的 `FAMILY_REGISTRY` 与 gatherer dotted paths 是刻意
隔离的 family taxonomy / plugin protocol，继续保留。没有为了减少行数合并 facade、protocol、
Ray adapter 或 family gatherer。

## Non-goals

- 不改变 chunk payload、trajectory schema、Ray wire result 或 OOM retry。
- 不让 worker 根据 `family` 再构造第二个 gatherer。
- 不把 gatherer 合并进 executor class，也不让 driver executor持有 model。
- 不引入 `GathererManager`、通用 util 或新的 registry facade。

## 验证

```bash
.venv/bin/python -m pytest \
  tests/architecture/test_generation_rollout_boundaries.py \
  tests/generation/execution/test_chunk_gatherer.py \
  tests/generation/execution \
  tests/generation/bindings \
  tests/generation/ray \
  tests/models/families/causvid/test_replay_and_loading.py \
  tests/models/families/janus_pro/test_r1_model.py \
  -q -p no:randomly
```

触及的 Python 文件通过 scoped Ruff check/format；最终 diff 通过 `git diff --check`。

## References

- `vrl/generation/execution/executor_base.py`
- `vrl/generation/execution/worker.py`
- `vrl/generation/ray/launch_inputs.py`
- `vrl/generation/ray/launcher.py`
- `vrl/generation/ray/worker.py`
- `vrl/families/registry.py`
- `tests/generation/execution/test_chunk_gatherer.py`
- `tests/generation/ray/test_ray_resident_session.py`
- [[SPRINT_homeless_function_placement]]
