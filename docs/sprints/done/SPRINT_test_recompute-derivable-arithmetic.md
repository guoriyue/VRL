# SPRINT: 测试里冻结手算结果，应改为从输入重算（done）

状态：done（2026-06-21）。四个文件的手算冻结结果改为从 source 公式/模板重算:
`test_chunk_dispatch.py` 两处 `20.0` → `estimate_chunk_cost(...)` / `row sample_count*num_steps`(并把
request 提为局部变量);`test_oom_split.py` chunk_key 字面串 → `SampleChunk(...).chunk_key`(新增 `_key` helper);
`test_cosmos_predict25_kling_eval.py` `== 26` → `first == second` + 输入算术锚 `17+2*4+1`;
`test_diffusion_nft.py` 冻结优势张量 → `nft==grpo` 契约 + `group_relative_advantages(...)` 闭式重算(已数值
核验 atol=1e-6 重现旧 `-1.2247449`/`-0.5855400`)。真不变量(`sample_start==[0,2,4,6]`、split 递归顺序、
`first==second`、`nft_adv==grpo_adv`)全保留。pytest 四文件 39 passed,ruff 全绿,未改任何 `vrl/` source。
范围：清理一类**低优先级**测试坏味道 —— 测试把某个公式/格式的输出**手算成一个字面量钉死**，而那个公式/格式本身就活在 source（`estimate_chunk_cost`、`SampleChunk.chunk_key`、`_seed_for`、`group_relative_advantages`）。测试真正想守的不变量（成本被透传、seed 与 checkpoint 无关、nft==grpo 优势、split 递归顺序）另有断言；冻结的字面量只是把同一段算术再抄一遍 —— 公式/格式一改，字面量就因**与被测行为无关的原因**而报错。本 sprint 只把这些字面量改成**从 source 的同一个函数/模板重算**，不改任何被测行为，不动算法/调度/seed 逻辑本身。

> 与 [[SPRINT_segment_signal_dead_field_cleanup]] 删死字段不同：这里没有死代码，被测行为全是对的；要消灭的是「测试自带一份 source 公式的手算快照」这一**冗余 + 易腐**模式。判定一个字面量是否该改的唯一标准：**它是不是 source 里某个函数/f-string 模板在该测试输入下的输出**？是 → 改成调用那个函数/模板重算；否（是测试自己选的输入、或外部固定契约）→ 保留。

## 0. Core Decision（先看这一段）

裁决一切「字面量该不该改」的，是 AGENTS.md 的派生原则在测试侧的镜像：**source 公式是单一事实源，测试不得再抄一份它的手算结果**。本仓库已有一个金标准反例 —— 凡是把 `tuple(FAMILY_REGISTRY)` 这种东西手抄成字面量 key-list 的测试都是 bug（`registered_rollout_families() == tuple(FAMILY_REGISTRY)`，新增 `wan_2_2` 会静默漏覆盖）。算术冻结是同一个 bug 的「标量版」：把 `estimate_chunk_cost(request, chunk)`、`SampleChunk(...).chunk_key`、`_seed_for(...)`、`group_relative_advantages(...)` 的输出抄成 `20.0` / `"prompt:0:samples:0:8"` / `26` / `-1.2247449`。

判定三档：

1. **是 source 公式/模板在测试输入下的输出** → 改成调用 source 重算（`estimated_cost == 20.0`、`chunk_key` 字面串、`== 26`、NFT `expected` 张量）。**这是本 sprint 的全部目标。**
2. **是测试自己选的输入回显**（如 `_request(num_steps=10)` 里的 `10`、`SampleChunk(..., sample_count=8)` 里的 `8`、`sample_start == [0,2,4,6]` 的偏移序列）→ **保留**，这是测试控制的量，不是 source 的输出。
3. **是外部固定契约**（HF repo id、vLLM cache layout 的 `2`）→ **保留**。

关键护栏：每个被改的断言旁边，**真正的不变量必须仍然独立存在**。例如 chunk_dispatch 的 `sample_start == [0,2,4,6]`（gather 契约）、oom_split 的递归顺序、seed 的 `first == second`、NFT 的 `nft==grpo` —— 这些是测试存在的理由，**一行都不删**；只把它们旁边那条「再抄一遍算术」的冗余断言换成派生。

## 1. 现状实锤

逐条已开文件核对（行号、snippet 均来自实读，非 JSON 盲信）。

### 1.1 `tests/ray/test_chunk_dispatch.py:210,316` —— `estimated_cost == 20.0`（手算 `samples*steps`，出现两次）

`vrl/generation/execution/chunk_placement.py:66-77` 是唯一事实源：

```python
def estimate_chunk_cost(request: GenerationRequest, chunk: SampleChunk) -> float:
    steps = request.sampling.get("num_steps") or request.sampling.get("max_new_tokens")
    return float(chunk.sample_count * max(1, int(steps or 1)))
```

测试用 `_request(num_steps=10, samples=8, sbs=2)`（`test_chunk_dispatch.py:66,73`：`sample_batch_size=2`），8/2=4 个 chunk，每个 2 samples × 10 steps = `20.0`。两处冻结：

```python
# :209-210  dynamic planner
# 8 samples / sbs 2 = 4 chunks of 2 samples x 10 steps = cost 20 each.
assert [a.estimated_cost for a in plan.assignments] == [20.0] * 4
```

```python
# :316  executor end-to-end schedule row
assert row["estimated_cost"] == 20.0
```

**铁证**：同文件 `:223` 的兄弟测试 `test_estimate_chunk_cost_uses_steps_axis` 已经在用派生写法 `assert estimate_chunk_cost(request, chunk) == chunk.sample_count * 35`，且 `estimate_chunk_cost` 在 `:20` 已 import。`20.0` 就是 `sample_count*num_steps` 的手算；加个 CFG=2x 因子之类的成本公式改动会让这两条无关报错，而 `sample_start == [0,2,4,6]`（`:212`，gather 契约）才是这条测试的真不变量。

### 1.2 `tests/generation/ray/test_oom_split.py:174-178` —— `chunk_key` 字面串重抄 f-string 模板

`vrl/generation/execution/chunks.py:61-65` 是唯一事实源：

```python
@property
def chunk_key(self) -> str:
    return f"prompt:{self.prompt_index}:samples:{self.sample_start}:{self.sample_end}"
```

测试把 split 递归产生的 key 全手抄成字面串：

```python
# :173-178
splits = output.extra["ray_chunk_oom_splits"]
assert [row["chunk_key"] for row in splits] == [
    "prompt:0:samples:0:8",
    "prompt:0:samples:0:4",
    "prompt:0:samples:4:8",
]
```

`SampleChunk` 在 `:19` 已 import。真正被测的是 **split 递归顺序**（8 → [0:4]+[4:8] → 2-sample 叶子）；`covered` 断言（`:167-172`）覆盖叶子集合也是真不变量。但这三条字面串是 `chunk_key` 模板的手抄 —— `prompt:` 改成 `p:` 或加字段，这里就因纯格式原因全腐烂。注意 `:174` 的 `covered` 那组 `(\"prompt:0:samples:0:2\", 2)` 同理来自模板，但它表达的是「叶子覆盖集合」，可一并改为从 `SampleChunk(...).chunk_key` 派生。

### 1.3 `tests/scripts/test_cosmos_predict25_kling_eval.py:42` —— `first == second == 26`（手算 seed 公式）

`vrl/scripts/eval/cosmos_predict25_kling_eval.py:408-419` 是唯一事实源：

```python
def _seed_for(*, base_seed, checkpoint_index, prompt_index, sample_index, samples_per_prompt) -> int:
    del checkpoint_index
    return int(base_seed) + int(prompt_index) * int(samples_per_prompt) + int(sample_index)
```

`base_seed=17, prompt_index=2, sample_index=1, samples_per_prompt=4` → `17 + 2*4 + 1 = 26`：

```python
# :42
assert first == second == 26
```

真正被测的契约是 `first == second`（seed 与 `checkpoint_index` 无关 → reward 变化反映权重而非噪声），这条**已被完整捕获**。尾巴上的 `== 26` 是把 seed 公式手算了一遍：若 seeding 方案加个全局 offset，`first == second` 仍成立但 `26` 会腐烂。`26` 这个 magic number 不比「结构相等 + 从公式重算」多提供任何东西。

### 1.4 `tests/algorithms/test_diffusion_nft.py:61-93` —— NFT 优势快照张量（group-relative 闭式手算）

`vrl/algorithms/advantages.py:58-89`（`group_relative_advantages`）是闭式事实源：`(r - group_mean) / clamp(std, eps)`，`global_std=False` 用每组 std，`global_std=True` 用跨population std。测试 parametrize 把输出抄成魔法小数：

```python
# :61-76
@pytest.mark.parametrize(("global_std", "expected"), [
    (False, torch.tensor([-1.2247449, 0.0, 1.2247449, -1.2247449, 0.0, 1.2247449])),
    (True,  torch.tensor([-0.5855400, 0.0, 0.5855400, -0.5855400, 0.0, 0.5855400])),
])
def test_diffusion_nft_advantages_match_grpo_contract(global_std, expected):
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    ...
    assert torch.allclose(grpo_advantages, expected, atol=1e-6)
    assert torch.allclose(nft_advantages, grpo_advantages, atol=0.0, rtol=0.0)
```

**真正的契约是 `:93` 的 `nft_advantages == grpo_advantages`**（NFT 与 GRPO 共享同一组相对优势）—— 已完整捕获。`expected` 张量是 `group_relative_advantages` 已可派生的闭式输出的冗余重钉，把 std 约定（`unbiased=False`、eps 位置）冻结成魔法小数；归一化方式的正当改动会逼着手改这堆常量而非改一个公式。

> 备注（不在本 sprint 范围）：JSON 同主题清单里还有一批「重抄 dataclass 默认值 / 目录清单 / Literal 成员 / 协议方法名」的发现（如 `test_schema.py` 的 algorithm-kind 列表、`test_generation_rollout_boundaries.py` 的 `hub.py` 已腐烂目录快照、`test_family_registry.py` 的 alias dict）。它们与本 sprint 同根（抄 source 而非派生），但**不是「手算算术输出」**这一窄主题，属于「冻结 registry/目录/Literal 快照」的另一类，留给同族但独立的 sprint，本 sprint 不混入。

## 2. 落地方案

**金标准 derive-from-source 模式**（贯穿全 sprint）：测试断言的右侧若等于 source 某函数/属性在该输入下的输出，就**调用那个函数/属性**，让公式/格式只活在一处。

### A. `test_chunk_dispatch.py` —— 两处 `20.0` 改为 `estimate_chunk_cost(...)`

BEFORE（`:210`）：

```python
assert [a.estimated_cost for a in plan.assignments] == [20.0] * 4
```

AFTER（沿用同文件 `:223` 已有派生写法）：

```python
assert len(plan.assignments) == 4
assert [a.estimated_cost for a in plan.assignments] == [
    estimate_chunk_cost(request, a.chunk) for a in plan.assignments
]
```

BEFORE（`:316`，executor schedule row）：

```python
assert row["estimated_cost"] == 20.0
```

AFTER（schedule row 自带 `sample_count`，`request` 在 `:306` 构造时可提到外层变量；从 row 的 `sample_count` × `num_steps` 派生，把格式锁在 source 的成本定义里）：

```python
assert row["estimated_cost"] == row["sample_count"] * request.sampling["num_steps"]
```

> 注：`test_executor_round_robin_dispatches_per_plan_binding`（`:301`）目前是 `await executor.execute(_request(...))`，需把 `request = _request(num_steps=10, samples=8, sbs=2)` 提出为局部变量再 `execute(request)`，使 `:316` 能引用 `request.sampling`。`sample_count == 2`（`:315`）是 chunk 几何不变量，保留。`sample_start == [0,2,4,6]`（`:212`）保留不动。

### B. `test_oom_split.py` —— `chunk_key` 字面串改为 `SampleChunk(...).chunk_key`

BEFORE（`:173-178`）：

```python
splits = output.extra["ray_chunk_oom_splits"]
assert [row["chunk_key"] for row in splits] == [
    "prompt:0:samples:0:8",
    "prompt:0:samples:0:4",
    "prompt:0:samples:4:8",
]
```

AFTER（递归顺序仍是 8 → [0:4]+[4:8]，但格式从 `SampleChunk.chunk_key` 派生）：

```python
def _key(start: int, count: int) -> str:
    return SampleChunk(prompt_index=0, prompt="p", sample_start=start, sample_count=count).chunk_key

splits = output.extra["ray_chunk_oom_splits"]
assert [row["chunk_key"] for row in splits] == [_key(0, 8), _key(0, 4), _key(4, 8)]
```

`covered`（`:167-172`）同样从 `_key(...)` 派生其四个叶子 key，保留 `samples` 计数（`2`）这个测试自选的容量输入。递归顺序与叶子集合（真不变量）一字不动，只是格式不再手抄。

### C. `test_cosmos_predict25_kling_eval.py` —— `== 26` 改为从公式重算

BEFORE（`:42`）：

```python
assert first == second == 26
```

AFTER（保留 `first == second` 这个真契约；数值锚从 `_seed_for` 的公式按输入重算，而非钉死 `26`）：

```python
assert first == second
# Seed is checkpoint-independent: same (prompt, sample) → same draw.
assert first == 17 + 2 * 4 + 1  # base_seed + prompt_index*samples_per_prompt + sample_index
```

> 数值锚用「输入算术表达式」而非 magic `26`，使 seeding 方案若加全局 offset 时，这条仍跟随公式（或直接删数值锚，只留 `first == second` —— 二选一，本 sprint 取「保留可读的输入表达式」）。

### D. `test_diffusion_nft.py` —— `expected` 张量改为 `nft==grpo` + 从公式算锚

BEFORE（`:61-93`）：parametrize 的 `expected` 是手钉小数张量。

AFTER（真契约 `nft == grpo` 保留；若要数值锚，从 `(r - group_mean)/std` 现算，而非粘 `-1.2247449`）：

```python
def test_diffusion_nft_advantages_match_grpo_contract() -> None:
    """DiffusionNFT and GRPO share one group-relative advantage contract."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    for global_std in (False, True):
        grpo = GRPO(GRPOConfig(eps=1e-4, global_std=global_std))
        nft = DiffusionNFT(DiffusionNFTConfig(eps=1e-4, global_std=global_std))
        grpo_adv = grpo.compute_advantages_from_tensors(rewards, group_ids)
        nft_adv = nft.compute_advantages_from_tensors(rewards, group_ids)
        # The real contract: NFT reuses the GRPO group-relative advantage.
        assert torch.allclose(nft_adv, grpo_adv, atol=0.0, rtol=0.0)
        # Numeric anchor recomputed from the closed form, not a frozen snapshot.
        expected = _group_relative(rewards, group_ids, eps=1e-4, global_std=global_std)
        assert torch.allclose(grpo_adv, expected, atol=1e-6)
```

数值锚有两种等效实现，取其一：

- 直接调用 source 闭式：`from vrl.algorithms.advantages import group_relative_advantages`，`_group_relative` 即转调 `group_relative_advantages(rewards, group_ids, eps=eps, adv_clip_max=..., global_std=global_std)`（最贴近「source 是唯一事实源」）；
- 或在测试内按 `(r - group_mean) / clamp(std, eps)` 现算（`std` 按 `global_std` 取每组 / 跨population，`unbiased=False`）。

两者都让 std 约定只活在一处，归一化改动只需改公式不需改小数。

## 3. 验证（finishing criteria）

- `grep -n "20.0" tests/ray/test_chunk_dispatch.py` 不再出现在 `estimated_cost` 断言里（仅可能剩无关命中）。
- `grep -n '"prompt:0:samples' tests/generation/ray/test_oom_split.py` 零命中（所有 chunk_key 经 `SampleChunk(...).chunk_key` 派生）。
- `grep -n "== 26" tests/scripts/test_cosmos_predict25_kling_eval.py` 零命中。
- `grep -nE "1\.2247449|0\.5855400" tests/algorithms/test_diffusion_nft.py` 零命中（魔法小数已消除）。
- `pytest tests/ray/test_chunk_dispatch.py tests/generation/ray/test_oom_split.py -q` 全绿。
- `pytest tests/scripts/test_cosmos_predict25_kling_eval.py tests/algorithms/test_diffusion_nft.py -q` 全绿。
- 改动后**真不变量仍在**：`grep -n "sample_start.*\[0, 2, 4, 6\]" tests/ray/test_chunk_dispatch.py` 仍命中；oom_split 的递归顺序断言仍在；kling_eval 的 `first == second` 仍在；NFT 的 `nft_adv` vs `grpo_adv` 相等断言仍在。

## 4. 非目标 / Non-Goals

- **不改被测行为/source 公式**：`estimate_chunk_cost`、`chunk_key` 模板、`_seed_for`、`group_relative_advantages` 一行不动 —— 本 sprint 只动测试侧的字面量。
- **不删真不变量断言**：gather 契约的 `sample_start` 序列、split 递归顺序、`first == second`、`nft==grpo` 全部保留，仅替换其旁边那条冗余的「手算算术」断言。
- **不保留测试自选输入回显**为「待改」：`_request(num_steps=10)` 的 `10`、`SampleChunk(..., sample_count=8)` 的 `8`、`sample_count == 2`、`samples` 计数这些是测试控制的输入，**不是** source 输出，保持原样。
- **不混入「冻结 registry/目录/Literal/dataclass 默认值快照」主题**：`test_schema.py` 的 algorithm-kind/strategy/loader 列表、`test_generation_rollout_boundaries.py` 的目录清单（含已腐烂的 `hub.py`）、`test_family_registry.py` 的 alias dict、协议方法名元组等 —— 同根但不同窄主题（它们冻结的是集合/目录/类型枚举，不是「算术输出」），留待独立 sprint。本 sprint 严格限于「手算了一个公式标量/张量输出」的四个文件。

## References

- `tests/ray/test_chunk_dispatch.py:209-212,316`（两处 `20.0`）
- `tests/generation/ray/test_oom_split.py:167-178`（`chunk_key` 字面串）
- `tests/scripts/test_cosmos_predict25_kling_eval.py:25-42`（`== 26`）
- `tests/algorithms/test_diffusion_nft.py:61-93`（NFT 优势快照张量）
- `vrl/generation/execution/chunk_placement.py:66-77`（`estimate_chunk_cost` 事实源；兄弟测试 `test_chunk_dispatch.py:223` 已用派生）
- `vrl/generation/execution/chunks.py:61-65`（`SampleChunk.chunk_key` f-string 事实源）
- `vrl/scripts/eval/cosmos_predict25_kling_eval.py:408-419`（`_seed_for` 公式事实源）
- `vrl/algorithms/advantages.py:58-89`（`group_relative_advantages` 闭式事实源）
- 金标准反例：`vrl/rollouts/families/registry.py:361-364`（`registered_rollout_families() == tuple(FAMILY_REGISTRY)` —— 冻结 key-list 是 bug，标量版即本 sprint）
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]（删死字段；本 sprint 是「活行为 + 冗余冻结快照」的姊妹主题）
