# SPRINT: tier 政策、`real_cover` 缺口标注机制与车道可见性（轨道一 / 地基）

状态：**planned**。轨道顺序 1 / 6。风险：低。

> **本轨道交付的是机械，不是覆盖率。** 它注册一个 marker、给 marker 装一个会红的守卫、
> 把 21 个已经在跑的真实 GPU 测试接进 `-m gpu`，并把 3 处“今天没人验证过的手抄常量 /
> 派生包装器”换成真断言。后续五条轨道的每一条标注都落在这套机械上——**先有守卫，
> 再有标注**，否则标注会像本文 §0 的实例一样静默腐烂。

---

## 0. 为什么必须排第一（两条实测事实）

**(1) 现在写 marker 会直接炸 collection，不是 warning。**

`pyproject.toml:203` 有 `addopts = ["--strict-config", "--strict-markers"]`，
`markers` 表（204-211）只有 6 条：`e2e` / `slow_test` / `gpu` / `rollout_preview` /
`distributed` / `optional`。全仓 `grep -rn "real_cover" --include=*.py --include=*.toml
--include=*.md` **零命中**（已复核）。任何 sprint 先写标注再注册，整个文件的收集会
`ERROR` 中断。下游至少 6 条 finding（M9 残余、RW-04、S5、TNA-05、GR-07、TNA-10）各自
独立踩到这同一件事。

**(2) 散文标注已经腐烂了，而且腐烂点就在同一个文件里。**

`tests/generation/execution/test_worker_sleep.py:103` 的 `_FakeCuMem` docstring 写着：

```python
class _FakeCuMem:
    """Stand-in for vLLM's CuMemAllocator: records pool/sleep/wake calls.
    ...
    The allocator-missing branch is tested for real via
    test_sleep_offload_requires_cumem; a memory-effect twin belongs in a
    vLLM-equipped GPU lane when one exists.
    """
```

那个“when one exists”的孪生**就在同一文件下方 1090 行处**，早已存在：

```python
# tests/generation/execution/test_worker_sleep.py:1191
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cumem_one_shot_scope_sleep_wake_in_subprocess() -> None:
    ...
        assert residual <= baseline + 4 * 1024 * 1024, (baseline, residual)
```

没有消费者的标注一定会跟不上代码——这正是 AGENTS.md 点名的 dead-field 模式。
**守卫是标注不腐烂的唯一原因。**

---

## 1. 实测基线（本机，`.venv/bin/python -m pytest ... -p no:randomly`）

| 口径 | 命令 | 结果 |
|---|---|---|
| 全量 | `pytest tests -q` | **3792 passed / 25 skipped / 200.46s** |
| 本轨道触及区 | `pytest tests/config tests/ray tests/data tests/utils/test_profiling.py -q` | 706 passed / 31.44s |
| 量化区 | `pytest tests/nn/quantization -q` | 58 passed / 3.77s |
| `_FakeCuMem` / OOM / probe 区 | `pytest tests/generation/ray/test_oom_split.py tests/generation/execution/test_chunk_memory_shadow.py tests/generation/execution/test_worker_sleep.py -q` | 72 passed / 3.80s |
| 本机硬件 | — | RTX 5090（SM 12.0），CUDA 可用，`device_count() == 1`，torch 2.11.0+cu130 |

> 全量数比 tier policy 基线文档记的 188.88s 高约 6%，因为测量时本机同时在跑探针进程。
> 测试数一致（3792 vs 3791+1）。**skip 数是环境函数，不要当指标。**

**成本口径（全部实测，非估算）：**

| 动作 | 车道 | 实测 |
|---|---|---|
| AST 守卫（读 385 个文件 + 解析 15 个载体 + 解析 8 个目标） | 默认 | **32 ms** |
| S6 的默认车道断言 `nvfp4_available("cpu") is False` | 默认 | < 5 ms |
| CRD-06 ② 的两个真转换 | 默认 | 0.03 s × 2（原 monkeypatch 版同量级，净 ±0） |
| S6 的 GPU 车道断言（gate 与真 `_scaled_mm` 一致性） | gpu | 0.34 s |
| `_is_oom_error` 对真 torch OOM 消息的断言 | gpu | 0.20 s |
| TNA-01 打上 marker 后 `-m gpu` **新选中**的 21 个实例 | gpu | **3.0 s**（这是交付物，不是开销） |

**净 wall-clock：默认车道 +0.04 s（+0.02%）；GPU 车道 +0.54 s。**
`-m gpu` 从「量化区 0 tests collected（58 deselected）」变成能选中 21 个真实 GPU 实例。

---

## 2. 总表

| 测试路径 | 今天替身的是什么 | 变成 | 成本 |
|---|---|---|---|
| `pyproject.toml:204-211` | — | 注册 `real_cover` marker（第 7 条） | 0 |
| `tests/conftest.py`（新 hook） | — | `--real-cover-report` 打印登记册（目标 + **车道** + why + tracked_in） | 0（不跑测试时才触发） |
| `tests/architecture/test_real_cover_labels.py`（新） | — | 消费 marker 的 AST 守卫 | **+32 ms**，默认车道 |
| `tests/config/test_load_all_experiments.py:490/537/654/721/1061` | `_auto_visible_cuda_devices` 被整体替换成 `lambda: (0,)` / `lambda: (physical_device,)` | **T1**：490/537/654/1061 改打 torch 层跑真包装器；721 改走真实的 `visible_devices` config 分支 | ±0 |
| `tests/nn/quantization/test_fp4.py`（新增 2 个测试） | `nvfp4_available()` 全仓 10 处引用**全是 monkeypatch**，零真断言 | **T1 + gpu 车道 T1** | +5 ms / +0.34 s |
| `tests/generation/ray/test_oom_split.py`（新增 1 个测试） | `_OOM_MESSAGE` 是手抄的 torch wire format，无人验证 | **gpu 车道 T1** | +0.20 s |
| `tests/nn/quantization/test_fp8.py` ×8、`test_fp4.py` ×8 | 真 GPU 工作，但只有 `skipif`，`-m gpu` 选不中 | 加 `@pytest.mark.gpu`（**与能力 skipif 并存**） | GPU 车道 +2.9 s |
| `tests/trainers/online/test_state_restore.py:169/194` | 真 CUDA fp16 GradScaler，函数体内 `pytest.skip` | 加 `@pytest.mark.gpu` | GPU 车道 +0.02 s |
| `tests/scripts/perf/test_gpu_preflight.py:29/42` | 真 GPU 测峰值，只有 inline `skipif` | 加 `@pytest.mark.gpu` | GPU 车道 ±0 |
| `test_worker_sleep.py:1039/1192`、`test_in_process_runtime.py:437` | 已带 `gpu` marker 还挂 `skipif(not torch.cuda.is_available())` | 删除（conftest:56-61 已负责） | ±0 |
| `tests/ray/test_resources.py:1350` | 打 `torch.cuda.is_available/device_count` 造 4 卡拓扑 | **T3-ENV 标注**（保留） | 0 |
| `tests/ray/test_resource_cleanup.py:12/29`、`test_global_placement.py:355/384/437` | 按需注入 Ray `kill` / `remove_placement_group` / `pg.ready()` 失败 | **T3-ENV 标注，指向真 Ray happy path** | 0 |
| `tests/utils/test_profiling.py:41/51/61` | 计数版 `torch.cuda.nvtx.range_push/pop` | **T3-ENV 标注**（保留） | 0 |
| `tests/data/test_danbooru.py:168`、`test_setup.py:175` | `_http_download` / `fetch=` 注入 | **T3-ENV 标注**（保留） | 0 |
| `tests/data/test_setup.py:125`、`test_jrdb_import.py:100` | `_write_mp4` 替身 | **T3-ENV 标注**（保留）+ 一处 round-trip 清洁修复 | 0 |
| `tests/ray/test_chunk_dispatch.py` 模块级 | `_FakeRef` / `_FakeWorker` 控制 asyncio 完成顺序 | **标注指向 HEAD 已有真孪生** | 0 |
| `test_worker_sleep.py:587/1171` | `_FakeCuMem` 调用记录当内存效果 | **标注指向同文件 :1193 的真孪生** + 改写腐烂 docstring | 0 |
| `test_chunk_memory_shadow.py` ×6 | `fake_cuda` 固定 24GB/32GB 卡 | **T3-ENV 标注**（保留） | 0 |
| GR-08 四条（见 §6） | paged-attention 与 pipelined 的进程内替身 | **标注指向 4 个 gpu 门控的真对位** | 0 |
| `tests/ray/test_global_placement.py:611` | `pytestmark_slow = pytest.mark.slow_test`（命名错误、全仓零引用） | **删除死代码** | 0 |

**合计：28 个标注点 / 13 个文件；3 组真转换（5 个 monkeypatch 站点收敛 + 3 个新测试）；
20 个函数（21 个实例）接进 `-m gpu`；4 处删除/清洁修复。**

---

## 3. 机制：marker + 守卫 + 登记册（已原型验证）

### 3.1 注册（`pyproject.toml` markers 表加第 7 条）

```
"real_cover: this test stands in for something it cannot run in-process; names the real-lane test that covers it for real, or the tracked gap",
```

**已实测**：注册后 `pytest.mark.real_cover("...", why=..., tracked_in=...)` 在
`--strict-markers` 下正常收集；`pytestmark = [pytest.mark.real_cover(...)]` 模块级
形态同样有效（模块级 helper class 挂不上函数级装饰器，必须支持这一形态）。

### 3.2 两种写法

```python
@pytest.mark.real_cover(
    "tests/ray/test_ray_actor_pool.py::test_run_actor_jobs_awaits_real_object_refs",
    why="fake refs control asyncio completion ORDER, which real Ray cannot make deterministic",
)
```

```python
@pytest.mark.real_cover(
    None,
    why="torch reports the host GPU count; a 4-GPU topology cannot be manufactured in-process",
    tracked_in="docs/sprints/planned/SPRINT_tier-policy-and-real-cover-labels.md",
)
```

重复使用同一个理由时，按仓库既有先例（`tests/nn/quantization/test_fp8.py:34` 的
`requires_fp8 = pytest.mark.skipif(...)`）提成模块常量再复用，不要抄 6 遍字符串。

### 3.3 守卫（`tests/architecture/test_real_cover_labels.py`，新文件）

沿用 `tests/architecture/test_generation_rollout_boundaries.py` 已立的 AST 先例
（**不能复用它的 `_forbidden_imports`**——CRD-08 已实测该 helper 的 `_python_files`
走 `root.rglob`，传文件路径返回 `[]`，是个永久绿的空断言）。三条规则：

1. **字符串目标必须解析到磁盘上真实存在的函数**（或脚本文件）；
2. **目标必须真的活在一条真实车道里**：`gpu` / `e2e` / `distributed` / `slow_test` 之一；
3. **`real_cover(None, ...)` 必须同时带 `why=` 和磁盘上存在的 `tracked_in=`。**

**车道解析必须穿透 `IfExp`。** `tests/ray/test_ray_actor_pool.py:17` 写的是

```python
pytestmark = pytest.mark.slow_test if pytest is not None else ()
```

只看函数装饰器会在第一个真目标上直接失败。原型对 8 个真目标全部解析成功：

```
OK tests/ray/test_ray_actor_pool.py:79  test_run_actor_jobs_awaits_real_object_refs      lanes=['slow_test']
OK tests/ray/test_global_placement.py:615 test_owner_reserves_trainer_gpu_and_binds_...  lanes=['slow_test']
OK tests/nn/kernels/test_vllm_paged_attention_real_ops.py:18                             lanes=['gpu']
OK tests/generation/execution/test_worker_sleep.py:1193                                  lanes=['gpu']
OK tests/e2e/test_real_checkpoint_rl.py                                                  lanes=['e2e']
OK tests/generation/execution/test_chunks_pipelined_cuda.py  （模块级 pytestmark）        lanes=['gpu']
```

**性能设计（对 sprint 简报里 172–327 ms 的修正）：** 不要对全部 385 个文件做 AST walk。
先 `read_text()` 做 `"real_cover" in text` 子串过滤（385 文件 **7.5 ms**），只解析载体
文件（预计 15 个，**15.8 ms**）与被点名的目标文件（8 个，**8.0 ms**）。
**实测总计 32 ms**，而朴素的全树双次 AST walk 实测 607 ms。取 32 ms 的方案。

### 3.4 登记册必须打印**车道**（这是最关键的一条规则）

M10 的复核已证伪“`pytest -m real_cover -q` 会打印诚实缺口登记册”——实测只输出
`N passed, M deselected`，marker 参数不经任何内置报表暴露。必须自己加 hook。

而“打印车道”不是锦上添花，是**必需**。实测证据：

```
$ pytest tests/generation/bindings/token_autoregressive tests/nn/kernels \
         tests/generation/execution/test_chunks_pipelined_cuda.py -q -rs
SKIPPED tests/generation/bindings/token_autoregressive/test_janus_vllm_paged_attention_backend.py:27:
  vLLM paged-attention internals are unavailable: ... importing 'vllm.v1.worker.block_table' ...
SKIPPED tests/generation/bindings/token_autoregressive/test_nextstep_vllm_paged_attention_backend.py:39: (同上)
SKIPPED tests/nn/kernels/test_vllm_paged_attention_real_ops.py:23: (同上)
31 passed, 3 skipped in 1.61s
```

**这台机器有真 GPU，这三个 `gpu` 对位仍然全部 skip**——第二道门（vLLM/torch ABI）挡住了。
所以「已被真实覆盖」不仅能在 CPU 机器上纸面为真，**在 GPU 机器上也能纸面为真**。

由此定死一条语义规则，写进 marker 的注册说明：

> `real_cover` 记录的是**真对位住在哪里**，不是**它跑过**。登记册必须打印车道，
> 让读者自己判断这条覆盖在本机是否真的执行。

hook 已原型验证（`--real-cover-report`，`pytest_collection_modifyitems` +
`pytest_terminal_summary`），输出形如：

```
real_cover register  (double -> real counterpart / tracked gap)
  tests/ray/test_chunk_dispatch.py::test_dynamic_binds_at_dispatch_time
      -> tests/ray/test_ray_actor_pool.py::test_run_actor_jobs_awaits_real_object_refs   [lane: slow_test]
      why: fake refs control asyncio completion ORDER, which real Ray cannot make deterministic
  tests/ray/test_resources.py::test_fsdp_4x_l4_rank_mask_resolves_one_logical_gpu
      -> NO REAL COUNTERPART   [lane: -]
      why: torch reports the host GPU count; a 4-GPU topology cannot be manufactured in-process
      tracked_in: docs/sprints/planned/SPRINT_tier-policy-and-real-cover-labels.md
```

**为什么不是 docstring 约定：** 没有消费者，会腐烂（§0 已给出仓库内的实例）。
**为什么不是 `skip` / `xfail`：** 这些测试是通过的、有真实价值的；标注的是它们的**边界**，
不是关掉它们。

**Verify：**
`.venv/bin/python -m pytest tests/architecture -q -p no:randomly`（当前基线 19 tests / 0.90s）

---

## 4. TNA-01：让 opt-in 车道成为一个真去处

整个 T3 叙事的前提是「`-m gpu` 是个真去处」。今天不是。

### 4.1 实测：量化区的真 GPU 工作，`-m gpu` 一个都选不中

```
$ pytest tests/nn/quantization -q -m gpu --collect-only
no tests collected (58 deselected) in 0.42s

$ pytest tests/nn/quantization -q
58 passed, 14 warnings in 3.77s          # 本机有 RTX 5090，全部真跑
```

按 AST 精确统计，被 `requires_fp8` / `requires_nvfp4` / `requires_vllm_fp8` 门控的是
**16 个函数 / 17 个实例**（`test_fp8_linear_matches_bf16_within_tolerance` 带
`parametrize("recipe", ["rowwise","tensorwise"])`）。它们做的是真 `torch._scaled_mm`、
真 `torch.compile`、真 CUDA 缓存搬移，但只有 `skipif`，marker 表里查不到。

> **对简报的修正：** 简报写“16 fp8/NVFP4 测试 + 2 个 state-restore”。精确数是
> **17 个量化实例**（fp8 9 + fp4 8）**+ 2 个 state-restore + 2 个 gpu_preflight = 21 个实例**。

另外两处同形态：

- `tests/trainers/online/test_state_restore.py:169/194` — 真 CUDA fp16 GradScaler 存取
  往返，但用的是**函数体内的 `pytest.skip`**，连 `skipif` 都不是，完全不可选择：
  ```python
  def test_fp16_cuda_state_dict_round_trips_grad_scaler(self) -> None:
      if not torch.cuda.is_available():
          pytest.skip("CUDA is required for fp16 GradScaler")
  ```
- `tests/scripts/perf/test_gpu_preflight.py:29/42` — 真测 bf16 峰值 TFLOPS，只有 inline
  `skipif(not _HAS_CUDA)`。

**改法：加 `@pytest.mark.gpu`，能力 `skipif` 一律保留并存。**
折叠会静默丢掉 skip：`conftest.py:56-61` 的 gpu 分支只在**没有 CUDA** 时 skip，
它不知道 fp8 `_scaled_mm` 或 SM≥10。在一张 A100 上折叠后测试会真跑并真失败。
已实测两者并存的行为正确：

```
$ pytest -m gpu -rs                      # 一个能力具备、一个不具备
SKIPPED tests/test_coexist.py:9: capability absent
1 passed, 1 skipped
```

**实测新增：这 21 个实例合计 3.00 s（pytest 计时）/ 4.97 s（含解释器启动，两次运行
3.08 s / 2.99 s）。CPU 机器 +0 s（本来就 skip）。这 +3 s 正是交付物。**

### 4.2 删掉已带 marker 的冗余 inline skipif

```python
# tests/generation/execution/test_worker_sleep.py:1038-1039
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")   # <- conftest 已负责
```

全仓命中 **3 处**：`test_worker_sleep.py:1039`、`test_worker_sleep.py:1192`、
`tests/rewards/inference/test_in_process_runtime.py:437`。

> **对简报的修正：** 简报写“另有 9 处 inline CUDA skipif 挂在已经带 marker 的测试上”。
> 全仓 `grep -rn "skipif" tests/ | grep -i "cuda\|gpu\|device_count"` 只有 6 处，其中
> **3 处**才是“已带 marker 的冗余”。另外 3 处（`test_gpu_preflight.py:29/42`、
> `test_fp8.py:34`）是**没有 marker** 的，属于 §4.1 要补 marker 的对象，不是删除对象。
> `test_in_process_runtime.py:437` 上还叠着 `WM_RUN_REAL_MODEL_TESTS` 门，那一层保留。

### 4.3 `tests/conftest.py:64-69` 的注释只修一半

```python
    # NOTE: the `distributed` and `optional` gating branches below are reserved
    # vLLM-parity lanes (commit "vLLM-style marker gating") with no current
    # members — `@pytest.mark.distributed`/`@pytest.mark.optional` is unused
    # repo-wide. Keep them: ...
```

实测：`distributed` **有 2 个成员**——`tests/trainers/test_wan_fsdp_distributed.py:576`
与 `:612`；`optional` **确实为 0**。

改法：删掉 `distributed` 那半句，**保留 `optional` 的“保留勿删”理由**（vLLM 结构对等 +
未来 opt-in 落点）。那条理由仍然成立，不能连坐删掉。

**Verify：**
```
.venv/bin/python -m pytest tests/nn/quantization tests/trainers/online/test_state_restore.py \
  tests/scripts/perf/test_gpu_preflight.py -q -p no:randomly -m gpu
```
（期望：在本机选中并通过 21 个实例，而不是 `no tests collected`）

---

## 5. 两条被纠正为**真转换**、不是标注

### 5.1 CRD-06 ②：`_auto_visible_cuda_devices` 改打 torch 层

**今天：**

```python
# tests/config/test_load_all_experiments.py:490 / 537 / 654 / 1061
monkeypatch.setattr("vrl.ray.resources._auto_visible_cuda_devices", lambda: (0,))
```

被替换掉的是**我们自己的 10 行包装器**，不是环境：

```python
# vrl/ray/resources.py:1341
def _auto_visible_cuda_devices() -> tuple[int, ...]:
    try:
        import torch
    except Exception:
        return ()
    try:
        if not torch.cuda.is_available():
            return ()
        return tuple(range(int(torch.cuda.device_count())))
    except Exception:
        return ()
```

**证明它没在证明什么：** 断言里出现的 `(0,)` 是测试自己写进 lambda 的。包装器的
`is_available` 短路、`device_count` 取整、异常兜底三条分支，**这五个站点一条都没跑过**。

**改法（T1）：** 打 torch 层，让真包装器执行。5 处收进 `tests/conftest.py` 的共享 fixture：

```python
@pytest.fixture()
def cuda_devices(monkeypatch):
    """Pin the host GPU topology so `_auto_visible_cuda_devices` runs for real."""
    def _apply(count: int) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    return _apply
```

实测：`torch` 层 `device_count()==1 → (0,)`、`==4 → (0,1,2,3)`、`is_available()==False → ()`。

**⚠️ 对 CRD-06 修正版的再修正：它说“5 处收进一个共享 fixture”，第 5 处收不进去。**

`tests/config/test_load_all_experiments.py:721` 是
`@pytest.mark.parametrize("physical_device", [0, 1])` + `lambda: (physical_device,)`。
而 `_auto_visible_cuda_devices` 返回的永远是 `tuple(range(n))`——
**`(1,)` 在 auto 路径下是不可达状态**：

```
can torch-level patching ever yield (1,)?  ->  False
```

也就是说这个替身今天在喂一个**生产函数造不出来的返回值**。正确的转换不是换打桩层，
而是走它在生产里真正的来源：

```python
def test_masked_physical_ordinal_comes_from_the_config_knob_not_the_auto_path():
    """`(1,)` is UNREACHABLE through the auto path: it returns tuple(range(n)).

    The entrypoint rewrites `visible_devices` to the selected physical ordinal
    before resource resolution (tests/scripts/test_online_entrypoint.py:94).
    """
    cfg = load_config("experiment/wan_2_1/online_grpo_droid_fullparam_fsdp_4x_l4")
    OmegaConf.update(cfg, "distributed.resources.visible_devices", [1], force_add=True)
    validate_training_config(cfg)
    r = resolve_distributed_resources(cfg)
    assert r.trainer_devices == r.rollout_devices == (1,)
```

这条路径跑的是真实的 `_parse_devices` + `_dedupe_ints` 分支——今天在这个 recipe 上零覆盖。
**仓库自己已有这个先例**：`tests/ray/test_resources.py:1350` 的
`test_fsdp_4x_l4_rank_mask_resolves_one_logical_gpu` 就是 `"visible_devices": [rank]`，
注释写着 “The entrypoint rewrites this to the selected physical ordinal before resource
resolution.”

两个转换都已实测通过（0.03 s / 0.03 s，与 monkeypatch 版同量级）：

```
$ pytest test_conversions_proto.py -q --durations=8
0.03s call     ::test_auto_path_runs_the_real_wrapper
0.03s call     ::test_masked_physical_ordinal_comes_from_the_config_knob_not_the_auto_path
5 passed in 1.42s
```

**Verify：** `.venv/bin/python -m pytest tests/config/test_load_all_experiments.py -q -p no:randomly`
（基线 54 passed / 10.31s）

### 5.2 S6：给 `nvfp4_available()` 补上全仓第一个真断言

**今天**：`nvfp4_available` 在 `vrl/` 有 5 个生产消费者（`vrl/models/loader.py:94` 决定
是否走 NVFP4 加载；三个 perf 脚本的前置检查），在 `tests/` 有 **10 处引用，全部是
monkeypatch**：

```
tests/nn/quantization/test_fp4.py:269 / :286 / :295
tests/models/steps/denoise/common/test_lora_fp8_build.py:377 / :406
tests/models/steps/token/test_rollout_quantization.py:199 / :237
tests/scripts/perf/test_quantized_rollout_drift_probe.py:125
tests/scripts/perf/test_quantized_linear_benchmark.py:53
tests/scripts/perf/test_quantized_sd3_forward_profile.py:168
```

**这个门本身从没被断言过一次。** 那些“模拟不可用”的用例今天指不到任何东西。

而生产的门和测试自己的能力探针**是两套不同的判据，且无人校验它们一致**：

```python
# vrl/nn/quantization/fp4.py:37 —— 生产：能力号 >= 10
def nvfp4_available(device=None) -> bool:
    if not torch.cuda.is_available() or not hasattr(torch, "float4_e2m1fn_x2"):
        return False
    target = torch.device("cuda" if device is None else device)
    if target.type != "cuda":
        return False
    return torch.cuda.get_device_capability(target)[0] >= 10

# tests/nn/quantization/test_fp4.py:25 —— 测试：真跑一次 _scaled_mm
def _nvfp4_capable() -> bool: ...
```

**新增两个测试（落在 `tests/nn/quantization/test_fp4.py`）：**

```python
def test_nvfp4_gate_rejects_non_cuda_devices():
    """Default lane: a non-cuda device can never satisfy the packed-NVFP4 gate."""
    assert nvfp4_available("cpu") is False
    assert nvfp4_available("meta") is False


@pytest.mark.gpu
def test_nvfp4_gate_matches_the_hardware_it_claims_to_describe():
    """The >=SM10 gate must agree with an actual packed-NVFP4 _scaled_mm."""
    ...
    assert nvfp4_available() is empirical
    assert nvfp4_available("cpu") is False   # not a fallout of "no CUDA"
```

**注意：第二个测试只挂 `@pytest.mark.gpu`，绝不能加 `@requires_nvfp4`。**
`requires_nvfp4` 本身就是那个经验探针；用它当门会在“门说 True 而 kernel 失败”这一
唯一有意思的分歧上直接 skip，把测试变成同义反复。

实测（RTX 5090，SM 12.0）：`nvfp4_available() → True`，经验探针 `→ True`，一致；
`nvfp4_available("cpu") → False` 而同一进程里 `nvfp4_available() → True`，
证明走的确实是 `target.type != "cuda"` 那条分支，不是“没有 CUDA”的顺带结果。
成本：默认车道 < 5 ms；gpu 车道 0.34 s。

**Verify：** `.venv/bin/python -m pytest tests/nn/quantization -q -p no:randomly`（基线 58 / 3.77s）

---

## 6. GR-08：四条指向 HEAD 已有测试的标注

> **诚实说明：GR-08 的 finding 正文没有交到本轨道手上**（findings 清单里标 `MISSING`）。
> 下面四对是我按 HEAD 现场自己取证重建的，逐条给了证据；如果原 finding 的四条与此不同，
> 以原 finding 为准重新裁定，不要照抄本节。

四个 `gpu` 门控的真对位（车道已用 AST 原型解析确认）：

| # | 进程内替身 | 真对位（HEAD 已存在） | 车道 |
|---|---|---|---|
| A | `tests/nn/layers/test_paged_attention_contract.py`（整文件 66 行，只验形状契约，不碰任何 kernel） | `tests/nn/kernels/test_vllm_paged_attention_real_ops.py:18::test_vllm_paged_attention_writes_real_cuda_kv_cache` | `gpu` |
| B | `tests/generation/bindings/token_autoregressive/test_janus_paged_attention_one_step.py:47` 的 `_RecordingPagedBackend` | `.../test_janus_vllm_paged_attention_backend.py:20::test_janus_vllm_paged_attention_matches_hf_llama_one_step` | `gpu` |
| C | `tests/models/families/nextstep_1/test_runner.py:33` 的 `_RecordingAttentionBackend` | `.../test_nextstep_vllm_paged_attention_backend.py:32::test_nextstep_vllm_paged_attention_matches_hf_qwen_one_step` | `gpu` |
| D | `tests/generation/bindings/full_sequence_denoise/test_forward_plan_pipelined_equiv.py:95`（`device = cuda if available else cpu`，CPU 车道上 Event / side-stream 机制完全没跑） | `tests/generation/execution/test_chunks_pipelined_cuda.py`（模块级 `pytestmark = pytest.mark.gpu`，3 个测试） | `gpu` |

D 是第二个「散文标注、机器读不到」的实例，和 §0 的 `_FakeCuMem` 同形：

```python
# tests/generation/bindings/full_sequence_denoise/test_forward_plan_pipelined_equiv.py:1-4
"""End-to-end equivalence: ... (the mechanism's bit-exactness is in test_chunks_pipelined_cuda)."""
```

**B/C 的载体位置：** `_RecordingPagedBackend` / `_RecordingAttentionBackend` 是模块级
helper class，函数级装饰器挂不上，用模块级 `pytestmark = [pytest.mark.real_cover(...)]`。

**必须同时写进 why 的一句话（§3.4 已实测）：** A/B/C 三条对位即使在本机这张真 5090 上
也仍然 skip（vLLM `vllm.v1.worker.block_table` 与 torch ABI 不匹配）。它们是**真对位**，
但**不保证在任何一台具体机器上执行过**。登记册打印车道就是为了让这件事看得见。
D 是四条里唯一在本机真跑的（3 passed / 0.22s）。

---

## 7. 显式保留，并写明理由（不是默许，是判决）

以下三组是本轨道**明确判定保留**的替身，每条给出保留理由。它们仍然要打 `real_cover`
标注——保留和标注不冲突，标注是让保留可被审计。

### 7.1 `_ProbeExecutor` / `fake_cuda` 的字节算术：按设计就没有真孪生

```python
# tests/generation/execution/test_chunk_memory_shadow.py:187
@pytest.fixture
def fake_cuda(monkeypatch):
    """Fixed 24GB-free/32GB-total card. Kept as a fake on purpose: the probe's
    budget arithmetic asserts exact byte values, which no real GPU can pin
    (mem_get_info is machine- and load-dependent)."""
```

理由成立且已写在源上：被断言的是**精确字节值**（`10*GB + 2*GB*n` 的峰值模型、
`non_torch = (32-24) - 8 = 0` 的推导）。真 GPU 的 `mem_get_info` 随机器与负载变化，
换真硬件等于把一个确定性算术断言换成一个不可复现的近似断言。
→ `real_cover(None, why=..., tracked_in=...)`，6 个消费者（`:197/222/236/251/271/276`）。

### 7.2 fp8 / fp4 的能力 `skipif` 必须与 marker 并存

见 §4.1 的实测。折叠会在没有 fp8 `_scaled_mm` 的 CUDA 卡上把 skip 变成 fail。
**不打标注**——它不是替身，是能力门。

### 7.3 `_CapacityWorker` 的 OOM 字符串：原 `why=` 是错的，必须改写

```python
# tests/generation/ray/test_oom_split.py:26
_OOM_MESSAGE = "CUDA out of memory. Tried to allocate 4.00 GiB"
```

生产侧的匹配器是子串检查：

```python
# vrl/generation/ray/executor.py:450
return "out of memory" in message.lower()
```

**原来的 why=（“造不出按需 OOM”）站不住**：本机实测，真 OOM 0.20 s 就能造出来，
而且真消息**确实**以那个前缀开头：

```
CUDA out of memory. Tried to allocate 4096.00 GiB. GPU 0 has a total capacity of 31.33 GiB ...
_is_oom_error(real)         -> True
_is_oom_error(hand-copied)  -> True
```

正确的裁定是：`_CapacityWorker` **本身保留**（它替的是一个会 OOM 的 Ray worker，
不是 torch 的消息格式），但那个手抄的 wire format 今天**零验证**。所以：

1. 新增一个 gpu 车道的真孪生（0.20 s），落在同文件：

```python
@pytest.mark.gpu
def test_oom_matcher_accepts_the_real_torch_allocator_message():
    """The hand-copied `_OOM_MESSAGE` prefix is only honest if torch still emits it."""
    with pytest.raises(torch.OutOfMemoryError) as caught:
        torch.empty(1024 ** 4, dtype=torch.float32, device="cuda")
    real = str(caught.value)
    assert real.startswith("CUDA out of memory. Tried to allocate ")
    assert _is_oom_error(real) is True
```

2. `_CapacityWorker` 的两个依赖该消息格式的消费者
   （`test_oom_chunk_splits_until_it_fits:155`、`test_single_sample_oom_still_raises:178`）
   标 `real_cover("tests/generation/ray/test_oom_split.py::test_oom_matcher_accepts_the_real_torch_allocator_message", why="a Ray worker that OOMs on demand cannot be produced in-process; the message FORMAT it hand-copies is pinned by the gpu-lane twin")`。

**注意这不是 `None`。** 写 `None` 会把一个刚刚被证明可造的真对位记成“不存在”。

---

## 8. 清洁修复（顺手，零行为变化）

1. **删死代码。** `tests/ray/test_global_placement.py:611`
   ```python
   pytestmark_slow = pytest.mark.slow_test     # 全仓零引用；因未命名为 `pytestmark` 而完全无效
   ```
   下方两个测试各自带显式 `@pytest.mark.slow_test`（`:614` / `:615`），删除无行为变化。

2. **停止 round-trip 替身自己的格式化字符串。** `tests/data/test_setup.py:149-166`
   ```python
   def fake_video_writer(path, frames, fps):
       path.write_bytes(f"fake-video frames={len(frames)} fps={fps}".encode())
   ...
   assert (data_root / rows[0]["target_video"]).read_text().endswith("fps=15.0")   # 读回自己写的
   ```
   改为记录调用参数：`calls.append((path, len(frames), fps))`，断言 `calls[0][2] == 15.0`。
   这才让「per-episode `source_fps=15.0` 覆盖 CLI `fps=10.0`」这个真不变量显式可读。

3. **改写腐烂的 docstring。** `test_worker_sleep.py:103-111` 的
   “a memory-effect twin belongs in a vLLM-equipped GPU lane **when one exists**”
   → 点名同文件 `:1193` 那个已经存在的孪生；同时给 `:587` / `:1171` 打 `real_cover`
   指向它。

4. **压缩重复表述。** `tests/ray/test_chunk_dispatch.py:1-13` 的模块 docstring 已经用散文
   点名了真孪生。改成 marker 后把那段压缩，避免同一事实写两遍（marker 是机器可校验的
   那一份，docstring 只留“为什么是受控时钟而不是 Ray 协议替身”这半句）。

---

## NON-GOALS（本轨道明确不做，理由逐条）

- **不动 `tests/math/test_denoise_flow_matching.py:21` / `test_token_flow_matching.py:20`。**
  TNA-14 已复核：`_FakeScheduler` / `_FakeHead` 是**解析预言机替身**，不是环境阻塞——
  手工配对的 EDM/flow sigma 表**本身就是被测的同构关系**，`0.1*x` 让闭式解可手算。
  给它们打 `real_cover(None, why=<blocked>)` 是**写假文档**。而且 denoise 那个文件
  `:182-186` 已经在跑真实 `FlowMatchEulerDiscreteScheduler`。

- **不动 `tests/trainers/test_fsdp.py:464 / :242 / :1034`、`test_ddp.py:186`。**
  TNA-14 实测这四条**可以转成真的**（真 DCP 0.7 ms、真 FSDP2 `_mp_policy` 读取、
  真 `shutdown_training_process_group()` 12.4 ms）。把可转换的东西标成“永久受阻的保留”
  正是 owner 的指令要防止的结果。它们属于 trainers 轨道的**转换**项，不是本轨道的标注项。

- **不动 `tests/rewards/inference/test_in_process_runtime.py` 的 7 个 `_FakeCumemAllocator`
  用例（RW-05）。** RW-05 的核心前提在本机被证伪：vLLM 能 import，
  `CuMemAllocator.get_instance()` 成功，`_cumem_allocator()` 返回真 allocator；而
  `building` 属性**只存在于替身上**，真 allocator 没有它。这一区需要的是转换 + 探针
  重设计，不是盖章。本轨道只在 §4.2 删掉它 `:437` 那行冗余的 `torch.cuda.is_available`
  skipif（它上面的 `WM_RUN_REAL_MODEL_TESTS` 门保留）。

- **不动 `tests/rewards/service/test_service.py`、`tests/rewards/test_clip_reward_models.py`、
  `tests/rewards/kling_video_reward/`（RW-09 / RW-10）。** 两条复核都证明了那里的
  “不可制造”判断是错的（真 `snapshot_download` 离线解析 revision 0.1 ms；真
  `ClientConnectorError` 走死端口即可）。那是**转换**工作，归 rewards 轨道。
  本轨道只保证 marker 已注册，让那条轨道能落地。

- **不动 `tests/architecture/test_generation_rollout_boundaries.py` 现有的子串检查
  （CRD-08）。** 复核实测 AST 版在两处会变成永久绿的空断言，且抓不到
  `from vrl.generation.execution import (...)` 这个真逃逸路径。本轨道**新增**一个
  AST 文件，**不重写**已有的。

- **不做 `--strict-markers` 之外的 pytest 配置改动**，不重命名/搬移任何测试文件。

- **不删任何覆盖。** 本轨道只有一处删除，是 §8.1 那行零引用、命名错误因而完全无效的
  `pytestmark_slow`。

---

## HONEST GAPS（本轨道标为“进程内未覆盖”的东西）

**有真对位，只是今天没人指向它（本轨道补上指针）：**

| 替身 | 真对位 | 车道 | 本机实际执行？ |
|---|---|---|---|
| `_FakeRef` / `_FakeWorker`（`test_chunk_dispatch.py`） | `test_ray_actor_pool.py::test_run_actor_jobs_awaits_real_object_refs` | `slow_test` | ✅ |
| Ray 清理失败注入（`test_resource_cleanup.py`、`test_global_placement.py`） | `test_global_placement.py:615::test_owner_reserves_trainer_gpu_and_binds_roles_on_simulated_gpus` | `slow_test` | ✅ |
| `_FakeCuMem` 的 sleep/wake 记录 | `test_worker_sleep.py:1193::test_real_cumem_one_shot_scope_sleep_wake_in_subprocess` | `gpu` | ✅（本机有 CUDA） |
| paged-attention 契约 / recording backend（GR-08 A/B/C） | `test_vllm_paged_attention_real_ops.py` 等三条 | `gpu` | ❌ **本机也 skip**（vLLM/torch ABI） |
| pipelined 等价（GR-08 D） | `test_chunks_pipelined_cuda.py` | `gpu` | ✅ |
| `_CapacityWorker` 的 OOM wire format | **本轨道新建** `test_oom_matcher_accepts_the_real_torch_allocator_message` | `gpu` | ✅ |

**没有真对位，且这就是缺口本身（`real_cover(None, ...)`）：**

| 替身 | 为什么进程内造不出来（引用具体阻塞点） |
|---|---|
| `torch.cuda.is_available` / `device_count` 打桩（`test_resources.py:1353`） | torch 报告的是宿主机的卡数。实测：不打桩时 `resolve_training_context` 在这台 1 卡机上给出 `cuda:2`，断言 `cuda:0` 直接失败——4 卡拓扑造不出来。 |
| `torch.cuda.nvtx.range_push/pop` 计数（`test_profiling.py:41/51/61`） | 实测**即使 CUDA 可用**，`range_push` 返回 `-2`、`range_pop` 返回 `-2`——没有挂载 profiler 时 range 深度不可观测。CUDA 路径由 `vrl/scripts/perf/profile_smoke.py` 在 GPU 机器上跑，那是脚本不是测试。 |
| `_http_download` / `fetch=`（`test_danbooru.py:168`、`test_setup.py:175`） | 在 CI 打 `danbooru.donmai.us` 既不可复现也不免费。 |
| `_write_mp4`（`test_setup.py:125`、`test_jrdb_import.py:100`） | `imageio` / `imageio-ffmpeg` **已在 `pyproject.toml:20-21` 声明**，真编码技术上可行；但被测的是 clip 切分与 manifest 结构，测试从不读写出的内容，换真编码是**纯加时间零覆盖**。判定保留，理由改写为这一句（不要写成“依赖缺失”）。 |
| `fake_cuda` 的 24GB/32GB 固定卡（`test_chunk_memory_shadow.py` ×6） | 见 §7.1：断言的是精确字节值，`mem_get_info` 随机器与负载变化。 |

**登记册的元规则（写进 marker 注册说明）：**
`real_cover` 记录真对位**住在哪里**，不记录它**跑过**。上表第 4 行就是活例子——
`gpu` 门控在一台真 GPU 机器上依然被第二道 ABI 门挡住。`--real-cover-report` 打印车道，
就是为了不让“纸面为真”被读成“已验证”。

---

## 附：单独立项，本轨道不修（已实测，比简报的描述更精确）

`vrl/scripts/perf/gpu_preflight.py:96` 与 `vrl/scripts/perf/gemm_peak_probe.py:53` 直接写
legacy 的 `torch.backends.cuda.matmul.allow_tf32 = False`，而 `vrl/models/precision.py:48`
的 `apply_float32_precision` 在 torch ≥ 2.9 上走的是新的 `fp32_precision` 字符串 API。
`precision.py` 的 docstring 明说“The fallback keeps supported older releases on the legacy
bool API **without mixing both mechanisms in one process**”——脚本正在破坏这条不变量。

**简报说“在 torch 2.11 上是 torch 明确会 raise 的组合”，只对一半。** 三种顺序实测
（torch 2.11.0+cu130，子进程隔离）：

```
A  legacy-write -> new-write -> legacy-read :  RAISE  RuntimeError: ... you have used mix of
                                                      the legacy and new APIs ...
B  new-write    -> legacy-write -> legacy-read:  read -> False | fp32_precision -> ieee   (静默)
C  生产顺序 apply_float32_precision('tf32') 然后跑 measured_bf16_peak_tflops():
   effective state -> {'matmul': 'ieee', 'cudnn': 'tf32'}                                  (静默)
```

**更危险的是 C，不是 A：** 不报错，但把生产刚设好的 tf32 静默改回 ieee，并留下
matmul/cudnn 不一致的裂开状态。`measured_bf16_peak_tflops()` 返回 223.4 TFLOPS，看起来
一切正常。修法是让两个脚本改调 `apply_float32_precision("ieee")`。
**归精度轨道，本轨道不动**（它会碰 `vrl/scripts/perf/`，与本轨道的 tests-only 改动无关）。

---

## 验证与顺序

**必须串行的只有第一步。**

1. **注册 marker**（`pyproject.toml` markers 表第 7 条）。
   `.venv/bin/python -m pytest tests -q --collect-only 2>&1 | tail -2`
   （期望：3792 collected，无 `not found in markers configuration option`）
2. **落 conftest 的 `--real-cover-report` hook + 修 `:64-69` 注释。**
   `.venv/bin/python -m pytest tests/trainers/test_wan_fsdp_distributed.py -q --distributed --collect-only | tail -2`
3. **落 AST 守卫**（此时零标注，守卫应绿）。
   `.venv/bin/python -m pytest tests/architecture -q -p no:randomly`（基线 19 / 0.90s）
4. **2–3 之后，以下三组可并行：**
   - TNA-01 marker 补齐 + 删 3 处冗余 skipif
     `pytest tests/nn/quantization tests/trainers/online/test_state_restore.py tests/scripts/perf/test_gpu_preflight.py -q -m gpu`
   - 两条真转换（CRD-06 ② / S6）
     `pytest tests/config/test_load_all_experiments.py tests/nn/quantization -q -p no:randomly`
   - 28 个标注 + §8 清洁修复
     `pytest tests/config tests/ray tests/data tests/utils/test_profiling.py tests/generation -q -p no:randomly`
5. **全量 + 登记册。**
   ```
   .venv/bin/python -m pytest tests -q -p no:randomly
   .venv/bin/python -m pytest tests -q --real-cover-report
   .venv/bin/ruff check <touched> && .venv/bin/ruff format --check <touched>
   ```
   完成条件：3792 passed 不变（+3 个新测试 → 3795），默认车道 ≤ 基线 +0.1 s，
   `--real-cover-report` 打印 28 条、每条都有车道。

**Definition of done：** 在任意一个 `real_cover` 目标上故意打错一个字符（指向不存在的
函数、或指向一个不带真实车道 marker 的测试），`pytest tests/architecture` 必须变红。
守卫抓不到的标注等于没有标注。

---

## 与现场的偏差（必须先核对再执行）

- **简报说工作树有 `vrl/rewards/`、`vrl/families/registry.py`、`pyproject.toml` 的在飞编辑。
  已过期。** 复核时 `git status --porcelain` 只有三项，全在 tests：
  ```
   M tests/generation/execution/test_execute_request_pipelined.py
   M tests/models/families/flux/test_diffusion_nft_interface.py
  ?? tests/generation/execution/test_zzscratch_probe_real.py
  ```
  本轨道**不碰**这三个文件。执行前重跑 `git status` 确认。
- `tests/generation/execution/test_zzscratch_probe_real.py` 是未追踪的一次性验证产物
  （docstring 自称 “Scratch: real-Ray verification of the GR-06 proposal”），按 AGENTS.md
  的一次性/长期资产规则应由其作者在答案落到 sprint 文档后删除。**仅登记，本轨道不处理。**
- **`M11` 被路由进本轨道，但 finding 正文没有交付**，且本轨道的立项叙述里没有任何锚点
  提到它。本文**不为 M11 写计划**——凭猜测写出来的计划比没有计划更糟。它需要重新猎取
  后单独立项。
- CRD-06 的 ⑦ 与 GR-08 的四条都指向 HEAD 已有测试，已逐条复核存在。
  **创建孪生的 sprint 自己写自己的标注**——守卫对悬空目标会红，先标后建必然中断。

## References

- 标准与生命周期：`AGENTS.md`（Evidence-First Work / Architecture Hygiene / Long-term
  Assets vs One-shot Validation）、`docs/sprints/README.md`
- 前一轮审计：`docs/sprints/done/SPRINT_test_suite_tiny_real_and_fake_audit.md`（commit `84584d23`）
- 车道定义：`tests/conftest.py:1-18`（docstring）、`:56-82`（`pytest_collection_modifyitems`）
- marker 表：`pyproject.toml:200-211`
- AST 先例：`tests/architecture/test_generation_rollout_boundaries.py:351-406`
- tiny-real 样板：`tests/models/steps/denoise/fixtures.py`、`tests/models/steps/token/fixtures.py`
- 真实 e2e 落点：`tests/e2e/test_real_checkpoint_rl.py`（9 个 `case_id`）、
  `tests/models/families/sana/test_real_inference.py:11-13`
- 生产源：`vrl/ray/resources.py:1341-1351`、`vrl/nn/quantization/fp4.py:37-46`、
  `vrl/generation/ray/executor.py:450`、`vrl/models/precision.py:48-68`
