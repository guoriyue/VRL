# SPRINT: 轨道三 — 进程里已经造得出来的东西，别再手搓

状态：**planned**。Order 3 of 6（tier-policy 基线 sprint 之后，两条更贵的转换轨道之前）。Risk: low。

> **归并轴是 tier，不是目录。** 这一轨收的全部是「真对象在当前进程里已经免费可构造」的替身：真
> `RootConfig`、真 torch DCP、真 CPU `GradScaler`、真 `torch.compile`、真
> `DistributedTrainingContext`、真 `EMAModuleWrapper`、tmp_path 上的真 safetensors 文件、
> 127.0.0.1:0 上的真 aiohttp `RewardService`、真实验预设。不需要 tiny 模型、不需要新车道、
> 不需要下载、不需要网络。
>
> 判据见 [[SPRINT_test_tiers]]（tier policy 基线）。本文只在需要时链接，不复述。

---

## 0. 一句话

**把 10 条已复原的发现里的手搓替身换成同一个进程里本来就能构造的真对象，代价实测
`+0.36 s`（默认 lane，从 189 s 起算 = +0.19%），其中 `+0.35 s` 全部来自 S4 的两个新
`main()` 级测试；其余 9 条加起来 `+11 ms`。**

三条不是「换个类型」而是「不再是恒真式」，都有实测变异证据：

- **S8** 今天的 `seed(0, 2, 1) == seed(3, 2, 1)` 是**字面恒真**——局部 helper 收下
  `checkpoint_index` 后第一行就 `del`。实测：把 checkpoint label 折进 seed，新测试红、旧测试绿。
- **S4** 实测：把 `main()` 改成「调了闸门但丢弃返回值」，`tests/scripts/eval/test_sana_aesthetic_checkpoint_eval.py`
  **47 个测试全绿**。所以只写一个「闸门被调用」的测试不够，必须两个。
- **TNA-02** 今天断言的是测试自己写的列表推导，不是 torch 的答案；改成 delegate 给真 DCP 后
  `set(state)` 变成 `get_model_state_dict` 算出来的。

两条是做减法：**S7** 删掉一个与第一个用例逐位相同的 `parametrize`（并改掉引发它的错误测试名）；
**S11** 顺手删掉全仓仅一次出现的死字段 `rollout_base_precision`。

一处机制纠正必须记下：**TNA-11 的嵌套用例不能把真 `torch.compile` 放外层**——`OptimizedModule.__getattr__`
会穿透到 `_orig_mod`，让 compile 剥离变得不可观测。实测见 §2.5。

---

## 0.1 交付前必读 — 这份计划的证据完整度

**必须先说清楚：18 条 finding 里只有 2 条（CRD-03、RW-13）带着完整 payload 送到我手上，
其余 16 条只有 id。** 我按「grouping rationale 点名的真对象清单 + 仓库实证」把其中 13 条复原到
可执行程度，剩下 3 条只能给候选站点。凡是复原出来的，本文的行号、断言、成本都是我在当前工作树上
重新读过、重新跑过的，不是转述。

| 类别 | 数量 | 说明 |
|---|---|---|
| payload 完整送达 | 2 | `CRD-03`、`RW-13` |
| 由 rationale + 仓库实证复原、id 明确 | 8 | `TNA-02` `TNA-07` `TNA-11` `GR-05` `S4` `S7` `S8` `S11` |
| 由对象清单复原、**id 未随件送达** | 3 | §2.6 / §2.7 / §2.8（`DistributedTrainingContext` / 真 safetensors / `EMAModuleWrapper`），对应 `TNA-08/09/10/15` 中的三条，但**无法确定谁是谁** |
| **业主工作树里已经落地** | 2 | §0.2 |
| 只有候选站点、不足以立计划 | 3 | §0.3（`M1`/`M4` 中未落地的那条、`GR-04` 剩余部分、`S3`） |

**不要把这份文档当作 18 条的完整计划。** 它是 13 条的可执行计划 + 3 条的定位线索。

### 0.2 与业主未提交改动的冲突 —— 两条 finding 已经落地，不要重做

工作树当前有两个 test 文件被业主改过（`git status`）：

**（一）`tests/generation/execution/test_execute_request_pipelined.py`** —— 这正是对象清单里的
「真 `GenerationWorkerCore` 构造函数」（对应 `GR-04`）。改动已完成：

```python
# 改前
core = GenerationWorkerCore.__new__(GenerationWorkerCore)
core.worker_id = "w0"
core._policy_version = policy_version
core._memory_parking = WorkerMemoryParking(...)
core.load_policy = lambda: None                       # type: ignore[method-assign]
...
def _request(version):
    return SimpleNamespace(policy_version=version, request_id="r", metadata={})

# 改后（工作树现状）
core = GenerationWorkerCore("w0", GenerationRuntimeLaunchContract(..., policy_version=policy_version))
core.executor = executor
core._uses_versioned_slots = uses_slots
...
def _request(version):
    return GenerationRequest(request_id="r", family="sd3_5", task="t2i", inputs=["p"],
                             samples_per_prompt=1, policy_version=version)
```

**（二）`tests/models/families/flux/test_diffusion_nft_interface.py`** —— 这正是对象清单里的
「真 peft」（对应 `M1`/`M4` 之一）。改动已完成：`build_tiny_wan_transformer` →
`build_tiny_flux_transformer`，`_TINY_WAN_LORA_TARGETS` → FLUX 出厂 LoRA target 表，
`_build()` 从 `SimpleNamespace(lora={...})` 换成真 `ModelBuild`（让 `build.lora` 来自生产
property 而不是预烘焙 dict）。

**动作：这两条不进本 sprint 的执行清单。** 执行者只需在开工前 `git diff` 确认它们仍在，并在
最终验证时把这两个文件一起跑绿。另有一个未追踪文件
`tests/generation/execution/test_zzscratch_probe_real.py`（真 local Ray 的 GR-06 探针，
带 `pytestmark = pytest.mark.slow_test`）——属于另一条轨道的 one-shot 验证物，**本 sprint 不动它**。

### 0.3 payload 未送达、只有候选站点的三条

| id | 对象清单里的名字 | 我定位到的候选站点 | 为什么不足以立计划 |
|---|---|---|---|
| `S3`（推测） | 真 `resolve_distributed_resources` | `tests/scripts/test_wan_dpo_config.py:176-180` 把它换成 `lambda _cfg, **_kwargs: SimpleNamespace(trainer_torch_device="cpu")` | **实测它不属于这一轨**：真调用首次 `560 ms`（惰性 import 主导），而且对该 recipe 返回 `trainer_torch_device == "cuda:0"`，与替身写死的 `"cpu"` 相反。成本和环境依赖都要单独论证，不能按「零成本」收编。 |
| `GR-04` 剩余部分 | — | `tests/generation/execution/test_worker_sleep.py:508`、`test_worker_versioned_slots.py` 的 `_Model`/`_Executor` | §0.2 已落地的那部分之外，剩下的按 §3.1 判为**命名保留**。 |
| `M1`/`M4` 中未落地的那条 | 「tmp_path 上的真 safetensors 文件」？ | 见 §2.7 —— 但它在 `tests/trainers/`，前缀是 `TNA` 不是 `M`，所以对应关系存疑 | 需要原始 payload 才能确认是不是同一条。 |

**复原程序（如果业主要补齐）：** 对 `tests/models/`（`M`）、`tests/generation` + `tests/ray`（`GR`）
按对象清单剩余项 grep：`grep -rn "SimpleNamespace(" tests/models | grep -v "^.*:.*#"`，
再与 `git log` 里的 hunter 产物比对。

---

## 1. 汇总表

成本一律是**本机实测**（`.venv/bin/python -m pytest … -q -p no:randomly`，同一文件 A/B 各跑 2–3 次取中位数），
不是估算。

| # | 测试路径 | 今天假的是什么 | 变成什么 | tier | 实测成本 |
|---|---|---|---|---|---|
| 2.1 | `tests/config/test_precision.py:27-46` | `_cfg()` 返回 `SimpleNamespace`，喂给整个 precision resolver（约 35 个测试） | 真 `RootConfig(**top)`；另加一个 `rootconfig`/`dictconfig` 跨形状参数化 | T1 | **0**（142 passed，2.28 s vs 基线 2.25–2.35 s） |
| 2.2a | `tests/rewards/videoscore2/test_parsing.py:203-211` | 手写 `_approx` 类（`pytest.approx` 的重实现） | `pytest.approx(x, abs=1e-2)`，删 helper | T1 | **0** |
| 2.2b | `tests/rewards/functions/test_multi.py:483-503` | `_RemoteRuntime.ensure_ready` 往 list 里 append，断言 `checked == ["a","b"]` | 127.0.0.1:0 上的真 `RewardService` + 两个真 `HttpRewardRuntime` | T1 | **+2 ms** |
| 2.3 | `tests/trainers/test_fsdp.py:433-461` | `fake_get_model_state_dict` 用列表推导重实现 DCP 的过滤语义 | delegate 给真 `torch.distributed.checkpoint.state_dict.get_model_state_dict` | T1 | **+0.2 ms** |
| 2.4 | `tests/trainers/test_strategy.py:130-134` | `SimpleNamespace(_scale=…, _growth_tracker=…, _per_optimizer_states=…)` | 真 `torch.amp.GradScaler("cpu")`，跑一次 scale/step/update 预热 | T1（**类型保真，非覆盖增益**） | **+1 ms** |
| 2.5 | `tests/trainers/test_weight_sync.py:118-124,147-154` | `_CompiledLike`（手写 `_orig_mod` 壳） | 真 `torch.compile`，且**必须放内层**：`_DDPLike(torch.compile(inner))` | T1 | 套件内 **+2 ms**；单文件独跑 **+0.9 s**（dynamo 惰性 import） |
| 2.6 | `tests/trainers/test_checkpointing.py`（10 处） | `SimpleNamespace(is_primary=…, world_size=…)` | 真 `DistributedTrainingContext` | T1 | **0**（123 passed，中位 1.79 s vs 基线 1.88 s） |
| 2.7 | `tests/trainers/test_checkpointing.py:636,660,691,1005,1062` | `(path / "adapter_model.safetensors").write_text("stub")` | `safetensors.torch.save_file({...}, path)`，断言可加载 | T1 | **+1 ms** |
| 2.8 | `tests/trainers/test_checkpointing.py:1006,1153,1252,1323,1383,1428` | 6 份手写 `_EMA`（`copy_ema_to`/`copy_temp_to`） | 真 `EMAModuleWrapper` | T1 | **< +1 ms** |
| 2.9 | `tests/scripts/eval/test_sana_aesthetic_checkpoint_eval.py:108-109` | `_allow_minimal_protocol` 把 `_normalize_run_config` 换成 `lambda cfg: cfg` | **两个**新测试：闸门被调用 + 闸门产出被转发 | T1 | **+0.35 s**（本轨最大项） |
| 2.10 | `tests/scripts/perf/test_diffusion_runtime.py:19-71` | 无假替身；第二个 `parametrize` 用例与第一个逐位相同 | 删 `parametrize`，改测试名 | 减法 | **−2 ms** |
| 2.11 | `tests/scripts/test_cosmos_predict25_kling_eval.py:42-62` | 局部 `seed()` helper 收下 `checkpoint_index` 后 `del` → 恒真 | 主张上移到唯一生产消费者 `_generate_checkpoint_videos` | T1 | **+5 ms** |
| 2.12 | `tests/scripts/test_online_lifecycle.py:345-348` | `SimpleNamespace(rollout="float32", rollout_base_precision="float32")` | 真 `PrecisionPolicy`；删死字段 | T1 | **0**（25 passed，8.40 s vs 基线 8.45 s） |
| — | **合计** | — | — | — | **+0.36 s / 189 s = +0.19%** |

> **与简报数字的偏差要说清楚：** 简报写「实测总增量约 +0.2s」。我重测得 **+0.36 s**。差额几乎全在
> S4（§2.9）：它需要两个 `main()` 级测试，而单个 `main()` 测试实测 `0.25 s`。其余 12 项加起来
> `+11 ms`，与简报一致。**+0.36 s 仍然是「只凭成本就能整轨接受」的量级**，但账要记对。

---

## 2. 逐条

### 2.1 `CRD-03` — precision resolver 的输入形状没有生产对应物

**位置：** `tests/config/test_precision.py:27-46`（`_cfg`），约 35 个测试消费；生产入口
`vrl/config/validation.py:339`（传 `RootConfig`）、`vrl/config/builders.py:356`（传裸 `DictConfig`）。

**今天证明了什么：** **这些测试是有效的**，它们驱动真 resolver 并断言真结果：

```python
assert p == PrecisionPolicy(
    training=RolePrecision(dtype="bf16", float32_precision="tf32"), ...
)
```

问题不是「证明不了东西」，是**输入形状没有生产对应物**。`_cfg` 返回的是：

```python
return SimpleNamespace(
    precision=top.get("precision"),
    actor=SimpleNamespace(**{...}),
)
```

生产里没有任何调用点会给 `resolve_precision_policy` 喂一个 `SimpleNamespace`。讽刺的是同一个文件
`:283` / `:302` 两个测试已经在用 `OmegaConf.create` 了——**一个文件里并存两种输入约定**。

**替换（改动 1）：** `_cfg` 的 `return` 换成真 `RootConfig`：

```python
from vrl.config.schema import RootConfig

return RootConfig(**top)
```

`RootConfig` 是 13 个生产入口里的 12 个，且保持了 `_cfg` 今天已经正确的三条分支走向
（outer `getattr` / `.precision` dict / `.actor` `getattr`）。它能承载全部畸形输入而不被 pydantic
提前拦掉：`RootConfig.precision` 的注解是 `Annotated[Any, ConfigBlock(PrecisionConfig)]`
（`vrl/config/schema.py:886`），所以 scalar `False`、legacy key、畸形 block 都原样穿透给 resolver 报错；
`actor.optim` 能存活是因为 `_OnlineRuntimeSection` 用 `extra="allow"` 且 `optim` 在 `runtime_fields`
白名单里（`vrl/config/schema.py:551-564`）。

**替换（改动 2）：** 新增一个 `@pytest.mark.parametrize("shape", ["rootconfig", "dictconfig"])`
的小测试，覆盖 `_select`（`vrl/config/precision.py:397-413`）的四个分支点——`precision` 存在 /
缺失 / 标量 / 遗留 `actor.optim.allow_tf32`——断言两种真实形状解析出**同一个** `PrecisionPolicy`、
报**同一条**错。

正当理由不是「第一次接上 `getattr` 分支」（早就接上了），而是两个真实缺口：

1. 今天没有任何测试断言 `RootConfig` 与 `DictConfig` 两种形状解析出相同结果——而 `_select` 那套
   `Mapping`/`.get` 与 `getattr` 双路兜底存在的唯一意义就是这个，却没有守卫。
2. 三条错误分支只在 `test_precision.py` 里被 `_cfg` 喂过：
   `grep "top-level \`precision\` is required"` 与 `grep "scalar \`precision\`"` 全仓只命中
   `test_precision.py:317` / `:260`；`allow_tf32` 迁移错误只命中 `:248`。
   `tests/config/test_load_all_experiments.py:415` 的 `test_experiments_do_not_use_legacy_precision_fields`
   只断言 YAML 里没有 `allow_tf32`，从不驱动 resolver 的报错路径。

`dictconfig` 那一臂仍要留（它是 `builders.py:356` 兜底真正会拿到的形状，也是 `load_config` 的原生产物），
但要在测试里注明它对应**兜底路径**，不是主路径。

**实测：** 改动 1 落地后 **142 passed / 2.28 s**，基线 2.25–2.35 s，整文件运行时间不变；
`RootConfig(...)` 构造 1.98 µs × 50 实例 = +0.09 ms。改动 2 的 8 个实例 < 1 ms。默认 lane，
无新 fixture，无 RNG、无 I/O。

**验证：**
```bash
.venv/bin/python -m pytest tests/config/test_precision.py -q -p no:randomly
```

**附带发现，建议单独开票（不在本 sprint scope）：** `vrl/config/builders.py:356` 的
`precision = precision or resolve_precision_policy(cfg)` —— 唯一生产调用方 `build_configs`
（`builders.py:494`）永远传 `precision=precision`，这条 `or` 兜底只有
`tests/config/test_load_all_experiments.py:87` 和 `:762` 能触发。按 AGENTS.md 死代码第 2 形态
（活调用方、死语义），要么把 `precision` 改成必填、要么保留并注明它是测试入口。它正是
「`DictConfig` 看起来像生产形状」的唯一来源。

---

### 2.2 `RW-13` — 手搓 `_approx` + 用 recorder 冒充 preflight

#### (a) `tests/rewards/videoscore2/test_parsing.py:203-211`

```python
def _approx(value: float, tol: float = 1e-2) -> object:
    class _Approx:
        def __eq__(self, other: object) -> bool:
            return abs(float(other) - value) <= tol
        def __repr__(self) -> str:
            return f"~{value}"
    return _Approx()
```

八行手写类 == `pytest.approx(value, abs=1e-2)`。**替换：** 六个调用点
（`:139-141`、`:164-166`、`:183`）换成 `pytest.approx(X, abs=1e-2)`，删 helper，加 `import pytest`。
理由**不是省行数**，是失败诊断：`float(None)` 在手写版里抛 `TypeError`，`pytest.approx` 给出正常 diff。
实测 11 passed / 0.02 s，ruff clean，**0 ms**。

#### (b) `tests/rewards/functions/test_multi.py:483-503`

```python
async def ensure_ready(self) -> None:
    checked.append(self._name)
...
await reward.preflight()
assert checked == ["a", "b"]
```

**它守的规则是真的**（`vrl/rewards/base.py:220-222` 按属性存在与否 duck-type 分派，
`InProcessRewardRuntime` 确实没有 `ensure_ready`，「skips local ones」那一半是真的），
但「reached every remote」那一半断言的是替身自己记的账。

**替换（真服务器，不是关闭的端口）：** 在 `127.0.0.1:0` 起一个真 `RewardService`，两个真
`HttpRewardRuntime` 指向它——形状照抄 `tests/rewards/service/test_service.py::_running_service`。
可观测的后置条件是真实且**逐组件**的：`RewardFunction.external_accelerator_isolation_verified`
只有在该 client 自己的 `/ready` + `/info` 往返完成、且 `/info` 广告了 `generation_overlap_safe`
之后才从 `False` 翻成 `True`（`vrl/rewards/service/client.py:252-255`、
`vrl/rewards/service/server.py:151-153`）。没有任何东西是脚本化的；server access log 里能看到
2× `GET /ready` + 2× `GET /info`。

```python
class _NeverScoredRuntime:
    """Service-side scoring runtime; preflight only hits /ready and /info."""
    async def score_batch(self, request):
        raise AssertionError("preflight must not score")
    async def shutdown(self) -> None:
        return None
```

完整可落地补丁（`git apply --check` 已通过，本次复核再次确认）：
`/tmp/claude-1000/-home-mingfeiguo-Desktop-VRL/3d48dcbf-816b-47e2-9eb3-237c62e9083f/scratchpad/RW13_verified_patch.diff`
需要在 `test_multi.py` 补一个 import：`from vrl.rewards.service.server import RewardService`。

**实测：** `service.start()` 0.20–0.31 ms；`reward.preflight()` 1.88–2.13 ms；整文件
0.16–0.17 s（23 tests）vs 基线 0.16–0.20 s（22 tests），delta 低于噪声。OS 分配端口（无 TOCTOU）、
仅 loopback、无 RNG、无时序断言；15/15 绿（`-W error::ResourceWarning`）。
**变异敏感性：** 在第一个 `await fn.preflight()` 后插入 `break`，本测试**红**；今天的 recorder 版本**绿**。

**tier 再裁定：** 这是 **T1**，不是先前审计归的「process/wire boundary double」。真 aiohttp server、
真 client、真 socket、进程内、~2 ms。前一轮 sprint 对这条的 keep 裁定应改判为「已转真」。

**诚实的残留（要写进注释，不要藏）：**
- service 侧的打分 runtime（`_NeverScoredRuntime`）仍是替身，但 preflight 永远到不了 `score_batch`，
  且该 stub 自己会 assert 这一点。真实打分过线已由
  `tests/rewards/service/test_service.py::test_client_preflight_hands_session_to_continuous_owner_loop` 覆盖。
- 顺序断言 `checked == ["a", "b"]` 在真实路径上**不可恢复**（`/info` 不携带 reward name）。preflight
  对任一组件 fail-fast，顺序不是任何人依赖的保证，丢掉即可。若业主坚持保留，把 recorder 版并排留着
  （~0 ms）并标注「ordering only」。
- **不要**加「关闭端口」变体：它与 `tests/rewards/service/test_service.py:312` 重复，且对
  「跳过某个组件」这个变异全盲。

**验证：**
```bash
.venv/bin/python -m pytest tests/rewards/functions/test_multi.py tests/rewards/videoscore2/test_parsing.py -q -p no:randomly
```

---

### 2.3 `TNA-02` — DCP 的过滤语义被测试自己重实现了一遍

**位置：** `tests/trainers/test_fsdp.py:433-461`
（`test_fsdp_checkpoint_gather_asks_dcp_to_skip_frozen_base_when_possible`）。

**今天的断言：**

```python
def fake_get_model_state_dict(actual, *, options):
    observed.append(options.ignore_frozen_params)
    return {
        name: parameter
        for name, parameter in module.named_parameters()
        if not options.ignore_frozen_params or parameter.requires_grad
    }
...
assert set(state) == ({"weight", "bias"} if register_frozen else {"weight"})
```

那个 `set(state)` 断言的是**测试自己写的列表推导**，不是 torch 的答案。如果哪天
`ignore_frozen_params` 的语义在 torch 里变了，这份手抄件会静默继续通过。

**替换：** spy 保留（还需要观察 flag），但 body 改成 delegate：

```python
import torch.distributed.checkpoint.state_dict as dcp

real = dcp.get_model_state_dict

def spy(actual, *, options):
    assert actual is module
    observed.append(options.ignore_frozen_params)
    return real(actual, options=options)

monkeypatch.setattr(dcp, "get_model_state_dict", spy)
```

生产在函数体内 `from torch.distributed.checkpoint.state_dict import get_model_state_dict`
（`vrl/trainers/fsdp.py:307-310`），所以 patch 模块属性有效。

**为什么真 DCP 在单进程也能跑：** 实测，无 process group、裸 `nn.Linear`：

```
ignore_frozen_params=True  -> ['weight']
ignore_frozen_params=False -> ['bias', 'weight']
```

**为什么 flag 观察必须保留（labelled keep）：** `register_frozen=False` 那一臂里，
`gather_checkpoint_state_dict` 只遍历 `owned_names`（`vrl/trainers/fsdp.py:333-341`），
所以把 flag 变异成恒 `False`，输出**不变**——单进程里这一臂的 flag 选择没有输出后果。
它真正的收益（frozen base 永不进 all-gather）只有 ≥2 rank 才可观测。
`register_frozen=True` 那一臂则有后果：flag 恒 `True` 会让真 DCP 少返回 `bias`，
生产的 `missing` 检查抛 `ValueError`。

**实测：** 转换后 2 passed / 0.76 s（含 collection）。真 DCP 单次调用 **0.078 → 0.024 ms**；
`torch.distributed.checkpoint.state_dict` 的惰性 import 是 **211 ms**，但今天的测试 patch 的就是
这个模块、已经付过了。**净增 +0.2 ms。**

**验证：**
```bash
.venv/bin/python -m pytest "tests/trainers/test_fsdp.py::test_fsdp_checkpoint_gather_asks_dcp_to_skip_frozen_base_when_possible" -q -p no:randomly
```

---

### 2.4 `TNA-07` — `GradScaler` 的私有表面被手抄了三个字段

**位置：** `tests/trainers/test_strategy.py:130-134`。

```python
scaler = SimpleNamespace(
    _scale=torch.tensor(8.0),
    _growth_tracker=torch.tensor(2),
    _per_optimizer_states={0: {"found_inf_per_device": {"cpu": torch.tensor(0.0)}}},
)
```

生产 parking 用 `getattr(state.grad_scaler, attr, None)` 读这三个名字
（`vrl/trainers/strategy.py:273-280`）——它们是 torch 的**私有**属性，替身把它们的名字和类型手抄了一份。

**替换：** 真 `torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=2)`，跑一次
`scale/backward/step/update` 让 `_scale` / `_growth_tracker` 变成真张量。实测预热后
`_scale=tensor(8.)`、`_growth_tracker=tensor(1, dtype=torch.int32)`、`get_scale()==8.0`。

**这条明确是类型保真，不是覆盖增益——不得当作覆盖增益出售。** CPU 上 parking 的目标设备就是 CPU，
`_move_tensor_tree_in_place(..., cpu)` 是 no-op，所以**任何 CPU 断言都无法区分这条分支是否执行过**。
真正的行为覆盖在 gpu lane：`tests/trainers/test_strategy.py:227`
`test_cuda_training_state_parking_round_trip_preserves_all_live_state`（`@pytest.mark.gpu`），
它用真 `torch.amp.GradScaler("cuda")` 并断言三个字段确实落到了 CPU。**这个指向必须写进注释**（见 §4）。

**实测：** 真 `GradScaler` 构造 0.006 ms，`scale` 0.45 ms，`step` 0.51 ms，`update` 0.03 ms，
`backward` 50 ms（但该文件在此之前已经有 backward，autograd 已热）。**净增 ≈ +1 ms。**
文件基线 15 passed / 2 skipped / 0.88–0.90 s。

> **环境注记：** 本机跑该文件时那 2 个 `@pytest.mark.gpu` 测试是 **skip** 的。tier policy 基线已经
> 写明「skip 数是环境函数，不是稳定指标」——本条的 gpu 对位测试在有 CUDA 的机器上才真的跑。

**验证：**
```bash
.venv/bin/python -m pytest tests/trainers/test_strategy.py -q -p no:randomly
```

---

### 2.5 `TNA-11` — `torch.compile` 剥离：**嵌套顺序是机制问题，不是口味问题**

**位置：** `tests/trainers/test_weight_sync.py:118-124`（`_CompiledLike`）、`:138-144`、`:147-154`。

**今天的替身：**

```python
class _CompiledLike(torch.nn.Module):
    """Mimics torch.compile's OptimizedModule: inner under ``_orig_mod``."""
    def __init__(self, inner): super().__init__(); self._orig_mod = inner
```

`torch.compile` 在 CPU 上是**免费**的（它惰性，不调用就不 trace），所以这个手写壳没有存在理由。

**替换（单层，`:138`）：**
```python
state = flatten_trainable_module_state({"adapter": torch.compile(inner)})
```

**替换（嵌套，`:147`）—— 必须改顺序：**
```python
wrapped = _DDPLike(torch.compile(inner))          # 真 compile 在内层
assert any(key.startswith("module._orig_mod.") for key in wrapped.state_dict())
state = flatten_trainable_module_state({"adapter": wrapped})
assert set(state) == {"adapter.weight"}
```

**为什么不能把真 `compile` 放外层——实测：**

| 组装 | 真实现 | 变异（`unwrap` 不剥 `_orig_mod`） |
|---|---|---|
| `torch.compile(_DDPLike(m))`（外层 compile） | `['adapter.weight']` | `['adapter.weight']` ← **变异存活** |
| `_DDPLike(torch.compile(m))`（内层 compile） | `['adapter.weight']` | `['adapter._orig_mod.weight']` ← **变异被抓** |

根因：`OptimizedModule.__getattr__` 会穿透到 `_orig_mod`，所以外层 compile 时
`getattr(module, "module")` 直接落到 DDP 的 `.module`，compile 剥离这一步变成不可观测。
今天那个测试的 docstring 写着「unwraps fully **regardless of order**」——真要证明「regardless of
order」，唯一可观测的那一半就是内层 compile。

**`_DDPLike` 保持替身（labelled keep）：** 真 `DDP` 需要 gloo process group，这个文件不该为一个
key-prefix 断言长出一套进程组基础设施。真对位是 `tests/trainers/test_ddp.py` 的 4 个真 gloo 测试；
实测删掉 `.module` 剥离会让它们全红。

**实测成本：** 首次 `torch.compile` = **830 ms**（`torch._dynamo` 惰性 import），之后 **0.74 ms**。
`tests/trainers/test_fsdp.py:402` 已经在默认 lane 里 `torch.compile(inner)`，所以**整套跑时这笔
import 早已付过，净增 ≈ +2 ms**；但 `tests/trainers/test_weight_sync.py` **单文件独跑**会从
`0.01 s` 涨到 `0.9 s`。这个数字要写进 PR 描述，别让人以为文件坏了。

**验证：**
```bash
.venv/bin/python -m pytest tests/trainers/test_weight_sync.py -q -p no:randomly          # 单文件（会看到 +0.9s）
.venv/bin/python -m pytest tests/trainers -q -p no:randomly                              # 套件内（应看不出差别）
```

---

### 2.6 `DistributedTrainingContext` —— 10 处 `SimpleNamespace(is_primary=…, world_size=…)`

> **id 未随件送达。** 这条来自 grouping rationale 的对象清单「真 `DistributedTrainingContext`」，
> 我按仓库证据复原。它属于 `TNA-08/09/10/15` 中的一条，但无法确定是哪一条。

**位置：** `tests/trainers/test_checkpointing.py:456, 485, 530, 564, 598, 1145, 1237, 1311, 1369, 1421`。

```python
self.context = SimpleNamespace(is_primary=True, world_size=1)
```

**替换：**
```python
self.context = DistributedTrainingContext(
    strategy="fsdp", rank=0, world_size=1, device=torch.device("cpu"),
)
```

**为什么值得：** `is_primary` 在真类型里是**派生**的（`vrl/trainers/distributed.py:52-54`，
`return self.rank == 0`）。替身允许把它和 `world_size` 各写各的——而 `:530` 与 `:1421` 写的正是
`SimpleNamespace(is_primary=False, world_size=1)`，这是 `resolve_training_context` 永远产不出的状态
（`single_process` 恒返回 rank 0 / world 1 / primary；`fsdp`/`ddp` 会校验
`WORLD_SIZE == num_nodes * gpus_per_node`，`vrl/trainers/distributed.py:90-118`）。换成真类型后，
这两处必须写成 `rank=1, world_size=2`，替身的意图第一次变得自洽可读。

**副作用（要写进 PR）：** 转换后有两个假 strategy 需要补上 `all_ranks_succeeded(self, succeeded)`
方法——这本身就是信号：替身表面此前比真契约窄。

**实测：** 转换版 **123 passed**，2.02 / 1.79 / 1.72 s（中位 **1.79 s**）；基线
2.00 / 1.84 / 1.88 s（中位 **1.88 s**）。**净增 0**（在噪声内，甚至偏负）。

**验证：**
```bash
.venv/bin/python -m pytest tests/trainers/test_checkpointing.py -q -p no:randomly
```

---

### 2.7 真 safetensors 文件 —— `write_text("stub")` 写出来的不是 safetensors

> **id 未随件送达**（对象清单「tmp_path 上的真 safetensors 文件」）。

**位置：** `tests/trainers/test_checkpointing.py:636, 660, 691, 1005, 1062`。

```python
def save_pretrained(self, path, *, state_dict, selected_adapters):
    assert state_dict.keys() == {"weight"}
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_text("stub")
...
assert (tmp_path / "checkpoint-1" / LORA_WEIGHTS_NAME / "adapter_model.safetensors").exists()
```

断言只是「有个同名文件」。文件内容是 `"stub"` 五个字节，任何 safetensors reader 都读不动。

**替换：** `save_pretrained` 里改成
`safetensors.torch.save_file({"weight": state_dict["weight"]}, path / "adapter_model.safetensors")`，
断言从 `.exists()` 升级成「用 `safetensors.torch.load_file` 读回来、key 与张量值都对得上」。
`safetensors` 已经是仓库直接依赖（`tests/config/test_schema.py` 等 12 个文件在用）。

**注意范围：** `save_pretrained` 本身**仍是替身**（真 PEFT 的 `save_pretrained` 会带整套 adapter
config 落盘，那是另一条轨的事）。这条只把**产物**变真，不动调用者。真 PEFT 导出的对位覆盖在
`tests/models/families/flux/test_diffusion_nft_interface.py`（§0.2 业主刚改成真 peft 的那个文件）。

**实测：** `save_file` + `load_file` 一对 ≈ **0.2 ms**，5 处 **≈ +1 ms**。

---

### 2.8 真 `EMAModuleWrapper` —— 6 份手写 `_EMA`

> **id 未随件送达**（对象清单「真 `EMAModuleWrapper`」）。

**位置：** `tests/trainers/test_checkpointing.py:1006, 1153, 1252, 1323, 1383, 1428`，六份几乎同形的
`class _EMA: has_updates = True; copy_ema_to(...); copy_temp_to(...)`，其中至少两份互相漂移
（`:1006` 版记录 `self.parameters`，`:1153` 版不记）。

**替换：** 真 `EMAModuleWrapper`（`vrl/trainers/online/ema.py`）。实测：构造 + 一次 `step` 在
1-param `nn.Linear` 上是亚毫秒级，同一 API 表面（`has_updates` / `copy_ema_to(store_temp=True)` /
`copy_temp_to`）真类都有。`tests/trainers/test_strategy.py:207` 已经在默认 lane 里构造真
`EMAModuleWrapper`，先例现成。

**保留的部分：** 那句 `assert store_temp is True` 是**调用参数记录**，真对象记不了自己的入参。
若要保住它，用 delegating spy（`unittest.mock.patch.object(ema, "copy_ema_to", wraps=...)`）而不是
整个替身——这与 §2.3 的手法一致。

**实测：** **< +1 ms / 6 处**。与 §2.6、§2.7 同文件，三条应当作**一个 PR**落地，一次 A/B 计时。

---

### 2.9 `S4` — `main()` 从来没有证明过它调了协议闸门（本轨最大项）

**位置：** `tests/scripts/eval/test_sana_aesthetic_checkpoint_eval.py:108-109`。

```python
def _allow_minimal_protocol(monkeypatch) -> None:
    monkeypatch.setattr(checkpoint_eval, "_normalize_run_config", lambda cfg: cfg)
    ...
```

每一个走 `main()` 的测试（`:145`、`:261`、`:283`、`:348` …）都先调 `_allow_minimal_protocol`，
把协议闸门换成恒等函数。于是 `main()` 与闸门之间的接线**完全无覆盖**。

**实测变异（已跑，仓库已还原）：** 把 `vrl/scripts/eval/sana_aesthetic_checkpoint_eval.py:156` 从

```python
cfg = _normalize_run_config(load_config(config_path))
```

改成

```python
_normalize_run_config(load_config(config_path))     # 调了，但丢弃返回值
cfg = load_config(config_path)
```

→ **`47 passed in 3.74s`，全绿。**

**为什么需要两个测试：**

- **测试 A（闸门确实被调用）：** 造一个 run dir，其 `resolved_config.yaml` 相对注册协议有行为漂移
  （照抄 `test_historical_shape_normalization_rejects_behavioral_drift` 的任一 `path/value` 对，
  例如 `algorithm.kl_reward_coef=0.1`），**不打** `_normalize_run_config` 的补丁，断言
  `main()` 抛 `ValueError, match="does not match the registered SANA full-parameter protocol"`，
  且 `_generate_images` 从未被调用（用 `pytest.fail` 哨兵，照抄 `:223` 现有写法）。
- **测试 B（闸门的产出被转发）：** 把 `_allow_minimal_protocol` 的恒等补丁换成**委派 spy**——
  调真 `_normalize_run_config`、记下返回对象——再断言 `main()` 下游拿到的 cfg 就是那个对象
  （在 `fake_generate` 里捕获 `root`，或对 `_validate_training_metrics` 的第二个实参做 identity 断言）。
  **只做 A 的话，上面那个「调了但丢弃返回值」的变异照样存活。**

**实测成本：** 单个 `main()` 级测试 `0.25 s`（`test_main_writes_provenance_bound_report`），
单个闸门级测试 `0.09 s`（`test_fullparam_long_config_is_the_exact_registered_protocol`）。
测试 A 在闸门处就抛、不进生成，≈ `0.10 s`；测试 B 是完整 `main()`，≈ `0.25 s`。
**合计 +0.35 s**——本轨 97% 的成本在这一条。

**要不要因此把它踢出「零成本轨」？不要。** `+0.35 s` 对 189 s 是 `+0.19%`，而它买到的是
**唯一一条能发现「协议闸门被绕过」的测试**——这个脚本的全部价值就是它拒绝未注册的配置。
但成本要在 PR 里明写，不能藏在「零成本」的标题下。

**验证：**
```bash
.venv/bin/python -m pytest tests/scripts/eval/test_sana_aesthetic_checkpoint_eval.py -q -p no:randomly
```

---

### 2.10 `S7` — 第二个 `parametrize` 用例与第一个逐位相同

**位置：** `tests/scripts/perf/test_diffusion_runtime.py:19-71`。

```python
@pytest.mark.parametrize("use_lora", [False, True])
def test_build_runtime_preserves_the_resolved_lora_contract(monkeypatch, use_lora: bool) -> None:
    ...
    assert build_runtime(root, device, precision=precision) is runtime
    assert calls == {"resolver": (root, device, precision), "builder": resolved_build}
    assert root.model is not None
    assert root.model.use_lora is use_lora
```

**证据：** `build_runtime`（`vrl/scripts/perf/common/diffusion_runtime.py:24-39`）整个函数体是
八行——`root.model is None` 守卫、`get_model_family_entry`、`resolve_model_build`、`build_rollout`。
**它从不读 `use_lora`。** 两个 parametrize 实例走的是逐位相同的生产路径；唯一的差别是
`assert root.model.use_lora is use_lora`，那是 `parse_config` 把输入原样回显。

而这个恒真断言正是**错误测试名的产物**：「preserves the resolved LoRA contract」——这个测试跟
LoRA contract 没有任何关系。它实际证明的是：`build_runtime` 把 `(root, device, precision)`
原样交给 registry entry 的 `resolve_model_build`，再把返回的 build 交给 `build_rollout`。

**动作（减法）：**
1. 删 `@pytest.mark.parametrize("use_lora", [False, True])`，`use_lora` 固定为 `False`。
2. 删 `assert root.model.use_lora is use_lora`。
3. 改名为 `test_build_runtime_hands_the_resolved_build_to_the_family_rollout_builder`。
4. docstring 写不变量：「registry entry 是 `build_runtime` 的唯一出口；resolver 收到的必须是调用者
   给的那三个对象本身，rollout builder 收到的必须是 resolver 的返回值」。

**这不是删覆盖。** 覆盖量在删前删后完全相同——被删的是同一段生产代码的第二次执行。

**实测：** 文件基线 5 passed / 0.18–0.21 s；删一个实例 **≈ −2 ms**。

**验证：**
```bash
.venv/bin/python -m pytest tests/scripts/perf/test_diffusion_runtime.py -q -p no:randomly
```

---

### 2.11 `S8` — seed 网格的「checkpoint 无关性」是一句字面恒真

**位置：** `tests/scripts/test_cosmos_predict25_kling_eval.py:42-62`。

```python
def seed(checkpoint_index: int, prompt_index: int, sample_index: int) -> int:
    del checkpoint_index                      # ← 第一行就丢掉
    return eval_script._seed_for(
        base_seed=17, prompt_index=prompt_index,
        sample_index=sample_index, samples_per_prompt=4,
    )

assert seed(0, 2, 1) == seed(3, 2, 1)        # ← 同一个函数、同一组实参
```

`seed_for` 的签名（`vrl/scripts/eval/denoise_video_generation.py:13-22`）里**根本没有 checkpoint
维度**，所以在这一层「checkpoint 无关」是签名结构保证的，测不出任何东西；测试还额外用一个
`del` 掉的形参把恒真式伪装成对照实验。

**真正的风险在唯一的生产消费者**：`_generate_checkpoint_videos`
（`vrl/scripts/eval/cosmos_predict25_kling_eval.py:362-408`）——那里 `target` 在作用域内，
把 checkpoint label 折进 seed 只差一次手滑，而后果是每个 reward delta 都变成不同的 latent noise draw。

**替换：** 主张上移一层，驱动真 `_generate_checkpoint_videos`：

```python
def test_seed_grid_cell_is_identical_across_checkpoints(monkeypatch, tmp_path) -> None:
    """The generator must derive each seed from the (prompt, sample) cell only.

    ``target`` is in scope inside the generation loop, so folding the checkpoint
    label into the seed is a live edit away — and it would silently turn every
    reward delta into a different latent-noise draw instead of a weight effect.
    """
    monkeypatch.setattr(
        eval_script, "_generate_one_video",
        lambda model, *, prompt, seed, sampling: torch.zeros(3, 1, 2, 2),
    )
    monkeypatch.setattr(eval_script, "write_mp4", lambda tensor, path, *, fps: path.touch())

    def run(label: str):
        return eval_script._generate_checkpoint_videos(
            object(),
            eval_script.CheckpointTarget(label, tmp_path / label),
            ["p0", "p1"],
            samples_per_prompt=2, base_seed=17,
            output_dir=tmp_path / label, sampling={"fps": 16.0},
        )

    base, trained = run("base"), run("trained")
    assert [v.seed for v in base] == [v.seed for v in trained]
    assert len({v.seed for v in base}) == 4          # non-degeneracy
```

`_generate_one_video` / `write_mp4` 是 GPU 与 mp4 编码边界，属于合法保留（见 §3.4）；
seed 网格本身全程是真代码。

**实测变异（已跑，仓库已还原）：** 把 `:377-378` 的 `base_seed=base_seed` 改成
`base_seed=base_seed + len(target.label)` →

```
FAILED ::test_seed_grid_cell_is_identical_across_checkpoints - assert [21, 22...
1 failed, 1 passed
```

**新测试红，今天那个 `test_seed_grid_is_identical_across_checkpoints` 绿。**

**实测成本：** 新测试 call time < 5 ms（2.37 s 是该模块的 import，现有文件已经付过）。
`_seed_for` 那个恒真测试建议**保留但改名 + 改 docstring**（它仍然守着 non-degeneracy 与算术公式），
去掉那个 `del` 掉的 `checkpoint_index` 形参。

**验证：**
```bash
.venv/bin/python -m pytest tests/scripts/test_cosmos_predict25_kling_eval.py -q -p no:randomly
```

---

### 2.12 `S11` — 死字段 `rollout_base_precision` + 形状不对的 precision 替身

**位置：** `tests/scripts/test_online_lifecycle.py:345-348`。

```python
precision = SimpleNamespace(
    rollout="float32",
    rollout_base_precision="float32",
)
```

两个问题：

1. **`rollout_base_precision` 是死字段**——`grep -rn "rollout_base_precision" .`（含 `*.py`
   `*.yaml` `*.md`）**全仓仅这一次出现**。它不是 `PrecisionPolicy` 的字段，不被任何生产代码读，
   也不被任何断言读。按 AGENTS.md 的死字段规则直接删。
2. **`rollout` 的类型也是错的**——真 `PrecisionPolicy.rollout` 是 `RolePrecision`
   （`vrl/config/precision.py:216-219`），不是 `str`。而同一个文件 `:275` 已经在构造真
   `RolePrecision` 了，一个文件里两种约定并存。

**替换：**

```python
precision = PrecisionPolicy(
    training=RolePrecision(dtype="fp32", float32_precision="ieee", outer_autocast=False),
    rollout=RolePrecision(dtype="fp32", float32_precision="ieee", outer_autocast=False),
    diffusion_math="fp32",
    prompt_encoder_dtype="fp32",
)
```

import 从 `from vrl.config.precision import RolePrecision` 扩成
`from vrl.config.precision import PrecisionPolicy, RolePrecision`。

**实测：** **25 passed / 8.40 s**，基线 8.45 s。净增 0。（这个文件用真 local Ray，8 s 里绝大部分是
集群 spin-up，与本改动无关。）

**范围说明——不做更大的那一步。** scratchpad 里有一份实验，把整个 `build_configs` 换成
真实验预设（`load_config("experiment/sd3_5/online_grpo_ocr")` + 真 `build_configs`）。那是**另一条
更贵的转换**，会把 recipe 内容变成这个 lifecycle 测试的隐式依赖。本 sprint **不做**，
留给「真实验预设」那一轨单独论证。

**验证：**
```bash
.venv/bin/python -m pytest tests/scripts/test_online_lifecycle.py -q -p no:randomly
```

---

## 3. NON-GOALS — 本区里刻意保留的替身，逐条给理由

> 按业主的新方针：keep 只有在它是 **(b) 无法按需制造的环境条件** 或 **(c) 进程/线边界** 且
> **被标注**时才成立。下面每一条都点名了它是哪一类、以及真对位在哪。

### 3.1 `GR-05` — `_uses_versioned_slots` 手工赋值 + `_Model` / `_Executor` 替身｜**保留**

`tests/generation/execution/test_execute_request_pipelined.py:55` 与
`tests/generation/execution/test_worker_sleep.py:508` 里的 `core._uses_versioned_slots = uses_slots`
是**载荷性注入**，不是替身：这两个文件测的是「已经处于 slot 模式的 worker 如何处理 stale
request / sleep」，前置状态必须能直接摆到位。**这个字段的真实推导已经在同目录被真代码覆盖了**——
`tests/generation/execution/test_worker_versioned_slots.py:126-165` 用真
`GenerationWorkerCore("rollout-0", GenerationRuntimeLaunchContract(...))`、真 `update_weights`，
断言 `core._uses_versioned_slots is True/False` 是**生产算出来的**。所以这是「同一事实，一处真推导
+ 两处直接置位」，不是三处手搓。

`_Model` / `_Executor`（`test_worker_versioned_slots.py:23-40, 77-88`）同样保留：它们实现的是
`RuntimeModel` / executor 的**协议表面**（`has_trainable_state` / `activate_trainable_state` /
`forward_plan_pipelined`），而断言全部落在 worker 的调度决策上（哪个版本被激活、slot 是否被覆盖、
`load_calls` 是否为空），不落在替身的返回值上。换成真模型只会把这些仪器抹掉。

**动作：给 `:55` 与 `:508` 各加一行注释指向 `test_worker_versioned_slots.py` 的真推导测试。** 无代码改动。

### 3.2 `TNA-02` 的 flag 观察｜**保留并标注（T3-SCALE）**

`ignore_frozen_params` 在 `register_frozen=False` 那一臂**单进程里没有可观测的行为后果**（§2.3 实测）。
它的真实收益——frozen base 永不进 all-gather——只有 ≥2 rank 才可观测。
**真对位：** `tests/trainers/test_fsdp_gather_distributed.py` 的真 gloo 多进程测试（默认 lane 里已经在跑）。
这条要标注，不是默默留着。

### 3.3 `TNA-11` 的 `_DDPLike`｜**保留（T3-SCALE）**

真 `torch.nn.parallel.DistributedDataParallel` 需要初始化 process group。为一个 key-prefix 断言在
`test_weight_sync.py` 里长出进程组基础设施是错的投资。**真对位：** `tests/trainers/test_ddp.py`
的 4 个真 gloo 测试——实测删掉 `.module` 剥离会让它们全红。

### 3.4 `S8` 的 `_generate_one_video` / `write_mp4`｜**保留（T3-SCALE / T3-ENV）**

前者需要真 diffusion 模型 + GPU，后者需要真 mp4 编码器。seed 网格的主张不需要它们真实。
**真对位：** `tests/e2e/test_real_checkpoint_rl.py` 的真 checkpoint 用例。

### 3.5 `S4` 的 `_materialize_model_snapshot` / `_materialize_reward_model_snapshots`｜**保留（T3-ENV）**

`_allow_minimal_protocol` 里其余几个补丁（`:110-125`）打的是 HF snapshot 物化——需要网络与多 GB 权重。
**本 sprint 只把 `_normalize_run_config` 那一行变真，其余三个补丁原样保留。** 这是刻意的窄改动。

### 3.6 `RW-13` 的 `_NeverScoredRuntime`｜**保留（payload 替身）**

service 侧的打分 runtime 仍是替身，但 preflight 永远到不了 `score_batch`，且该 stub 自己会 assert
这一点。**真对位：** `tests/rewards/service/test_service.py::test_client_preflight_hands_session_to_continuous_owner_loop`。

### 3.7 `TNA-07` 的 CPU 断言｜**保留但必须写明它证不了什么**

CPU 上 parking 分支不可观测（§2.4）。**真对位：** 同文件 `:227` 的 `@pytest.mark.gpu` 测试。

### 3.8 明确的非目标动作

- **不改文件名、不挪文件、不为整洁重排。**
- **不删任何测试**，除了 §2.10 那个与前一个用例逐位相同的 `parametrize` 实例（并已论证覆盖不变）。
- **不碰 `tests/generation/execution/test_zzscratch_probe_real.py`**（业主未追踪的另一轨探针）。
- **不碰 `vrl/`**——本 sprint 全部是测试侧改动。§2.1 末尾那条 `builders.py:356` 死语义单独开票。

---

## 4. HONEST GAPS — 本 sprint 标注为「进程内测不了」的东西

### 4.1 标注机制的前置条件（**硬阻塞，必须先解决**）

`pyproject.toml:202` 设了 `addopts = ["--strict-config", "--strict-markers"]`，而 `markers` 列表
（`:203-211`）只有 `e2e` / `slow_test` / `gpu` / `rollout_preview` / `distributed` / `optional`——
**`real_cover` 未注册**。实测：在 `tests/` 下放一个带 `@pytest.mark.real_cover(...)` 的探针文件
→ `ERROR collecting … 'real_cover' not found in markers configuration option`，
**整个文件收集失败，不是 warning。**

**因此：**

- `real_cover` marker 的注册 + AST meta-test 属于 **tier-policy 基线 / infra sprint**，不属于本轨。
- **本 sprint 不依赖那个 marker 也能整轨落地**：§3.1–§3.7 的标注先以**注释 + docstring** 形式落在
  替身定义处，并逐条点名真对位测试的完整路径。
- infra sprint 落地后，本轨的 7 条 keep 用一个后续 PR 换成 `@pytest.mark.real_cover(...)`。
  **换过去之前不要在本轨的 PR 里写这个 marker。**

### 4.2 本 sprint 声明「进程内不覆盖」的清单

| 事实 | 类别 | 真对位（完整路径） |
|---|---|---|
| `ignore_frozen_params` 让 frozen base 不进 all-gather | T3-SCALE | `tests/trainers/test_fsdp_gather_distributed.py`（真 gloo，默认 lane） |
| DDP `.module` 前缀剥离 | T3-SCALE | `tests/trainers/test_ddp.py`（真 gloo 4 测试，默认 lane） |
| GradScaler parking 真的把 `_scale`/`_growth_tracker`/`_per_optimizer_states` 搬到 CPU | T3-ENV（无 CUDA 就观测不到） | `tests/trainers/test_strategy.py:227`（`@pytest.mark.gpu`） |
| 真 diffusion 生成 + mp4 编码 | T3-SCALE | `tests/e2e/test_real_checkpoint_rl.py` |
| SANA 协议里 HF snapshot 物化 | T3-ENV（需网络 + 多 GB 权重） | **没有 e2e 对位——这就是缺口本身。** 记录在此，不假装有覆盖。 |
| reward service 的真实打分过线 | 已覆盖 | `tests/rewards/service/test_service.py::test_client_preflight_hands_session_to_continuous_owner_loop` |

**最后一行要单独强调：** SANA 协议的 snapshot 物化目前**没有任何真实对位**。本 sprint 不解决它，
只把它写进登记册。假装它被覆盖了，比不写还糟。

---

## 5. 校验与顺序

### 5.1 落地顺序（4 个 PR，互不依赖，可并行 review）

| PR | 内容 | 触及文件 | 实测增量 |
|---|---|---|---|
| **A** | §2.6 + §2.7 + §2.8 | `tests/trainers/test_checkpointing.py` | **≈ +2 ms** |
| **B** | §2.3 + §2.4 + §2.5 | `tests/trainers/test_fsdp.py` / `test_strategy.py` / `test_weight_sync.py` | **+3 ms**（单跑 `test_weight_sync.py` 会看到 +0.9 s，见 §2.5） |
| **C** | §2.1 + §2.2 | `tests/config/test_precision.py` / `tests/rewards/functions/test_multi.py` / `tests/rewards/videoscore2/test_parsing.py` | **+2 ms** |
| **D** | §2.9 + §2.10 + §2.11 + §2.12 | 4 个 `tests/scripts/` 文件 | **+0.35 s**（几乎全在 §2.9） |

PR **A/B/C 加起来 +7 ms**，可以只凭成本接受。**PR D 的 +0.35 s 必须在描述里明写**，
并附上 §2.9 的变异证据（47 passed 全绿）作为它值这个价的论据。

§3.1–§3.7 的注释标注**并入各自 PR**，不单独开。`real_cover` marker 的替换等 infra sprint（§4.1）。

### 5.2 每个 PR 的门禁

```bash
# 1. 触及文件的 ruff（只跑触及的文件，不做全仓 sweep）
.venv/bin/ruff check --fix <touched files> && .venv/bin/ruff format <touched files>
.venv/bin/ruff check <touched files> && .venv/bin/ruff format --check <touched files>

# 2. 触及目录先绿
.venv/bin/python -m pytest <touched dirs> -q -p no:randomly

# 3. 计时 A/B（改动前后各 3 次取中位数，别信单次）
.venv/bin/python -m pytest <touched files> -q -p no:randomly     # x3

# 4. 全量回归 + 总计时
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest -m "not e2e and not slow_test" -q
```

### 5.3 收尾门禁（整轨落地后）

- 全量：**3816 collected / 3791 passed**，与 tier-policy 基线一致（skip 数是环境函数，**不作为指标**）。
- 全量 wall-clock：`≤ 基线 + 0.5 s`（预算 +0.36 s，留 0.14 s 噪声余量）。
- Fast PR lane：`≤ 基线 + 0.4 s`。
- `git diff` 里**不得出现** `vrl/` 下的改动（§2.1 附带发现单独开票）。
- `git diff` 里**不得覆盖** §0.2 那两个业主未提交文件的改动——开工前先
  `git stash list && git diff --stat` 确认。

### 5.4 变异复验（reviewer 照做即可复现本文的三条主张）

```bash
# S8：把 checkpoint label 折进 seed
#   vrl/scripts/eval/cosmos_predict25_kling_eval.py:378
#   base_seed=base_seed  ->  base_seed=base_seed + len(target.label)
# 期望：新测试 FAILED，旧 test_seed_grid_is_identical_across_checkpoints PASSED

# S4：调了闸门但丢弃返回值
#   vrl/scripts/eval/sana_aesthetic_checkpoint_eval.py:156
#   cfg = _normalize_run_config(load_config(config_path))
#   ->  _normalize_run_config(load_config(config_path)); cfg = load_config(config_path)
# 期望（改前）：47 passed —— 变异存活；（改后）：测试 B FAILED

# TNA-11：unwrap 不剥 _orig_mod
#   vrl/models/weight_utils.py:41  删掉 getattr(module, "_orig_mod", module) 那一行
# 期望：_DDPLike(torch.compile(m)) 版 FAILED；torch.compile(_DDPLike(m)) 版仍 PASSED
```

**每个变异跑完都必须 `git checkout -- vrl/` 还原。** 本文所有变异实验都是这样做的，
复核结束时工作树只剩业主自己的两个未提交文件。

---

## 参考

- Tier policy 基线：`docs/sprints/planned/SPRINT_test_tiers.md`（本轨的判据来源）
- 前一轮审计：`docs/sprints/done/SPRINT_test_suite_tiny_real_and_fake_audit.md`（commit `84584d23`）
  —— 本文对 `RW-13` 的 keep 裁定**明确改判**为「已转真」（§2.2 tier 再裁定）
- 格式范本：`docs/sprints/done/SPRINT_homeless_function_placement.md`、
  `docs/sprints/done/SPRINT_deadcode_rewards.md`
- RW-13 现成补丁：
  `/tmp/claude-1000/-home-mingfeiguo-Desktop-VRL/3d48dcbf-816b-47e2-9eb3-237c62e9083f/scratchpad/RW13_verified_patch.diff`
