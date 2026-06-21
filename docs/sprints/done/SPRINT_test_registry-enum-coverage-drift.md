# SPRINT: 测试硬抄 family/enum/alias 清单与注册表源头漂移（done）

状态：done（2026-06-21）。优先级：high。六处硬抄清单全部改为从源头派生，落地实锤：
strategy 漂移补回 `ddp`（从 `Literal` 派生）；scheduler-parity 改为迭代注册表 diffusion family +
从 model config 读 path/revision，修正 `cosmos-anima`→`cosmos-predict2-anima` key 漂移、补回
`flux`/`qwen_image`（新覆盖的 flux/qwen 动态 shift scheduler 需传 `mu`，已在测试体处理）、
`wan_2_1_i2v` 无映射时显式 skip；alias 表改迭代 `entry.aliases`；`EXPECTED_ALGO_TYPE` 镜像 dict
删除、改断言 builder 派发稳定。六个测试文件 ruff 全绿、pytest 全绿（缓存缺失家族走 skip）。
逆向验证：临时改注册表/源头成员，派生断言随之跟进。
范围：把若干测试里**手抄的「合法成员集合」**（`Literal` 注解、`FAMILY_REGISTRY`、alias 表、kind→config-class 派发表）改成**从源头派生**。这些测试把一个 typed allow-list 的成员逐个重新打字进 `parametrize` 列表或 expected dict，然后断言「它们自己重抄的成员」。因为这份拷贝从不来自源头，所以源头新增一个成员时该成员**完全不被测试**，而一份已经走样的拷贝要么静默腐烂、要么因非行为原因报错。本 sprint 处理的就是 `registered_rollout_families() == tuple(FAMILY_REGISTRY)` 这个 canonical 例子的同类问题 —— 一个硬写 key-list 的测试本身就是 bug。**已有两处确认漂移**：strategy 测试漏了 `ddp`（`Literal` 里有），scheduler-parity 字典漏了 `flux`/`qwen_image`/`wan_2_1_i2v` 且把 key 写成 `cosmos-anima`（注册表实为 `cosmos-predict2-anima`）。

> 影响 6 个测试文件、6 条 finding。所有「可派生 / 已漂移」判定均已逐条打开测试文件取真实行号 + 打开 `vrl/` 源头核实派生路径（`typing.get_args` / `FAMILY_REGISTRY` 迭代 / `entry.aliases`），下文给出 BEFORE 冻结断言与 AFTER 派生断言的对照代码。

## 0. Core Decision（先看这一段）

**单一源头（typed structure）才是真相，测试不得再抄一份。** 三类源头各有标准派生姿势：

1. **`Literal` 注解** → `typing.get_args(Model.model_fields["<field>"].annotation)`，`Literal[...] | None` 要先剥掉 `NoneType` 再取内层 `Literal`。已实测三处源头：
   - `AlgorithmConfig.kind` → `('grpo','dance_grpo','flow_dppo','grpo_guard','token_grpo','token_grpo_multisegment','diffusion_dpo','diffusion_nft')`
   - `TrainingSection.strategy` → `('single_process','fsdp','ddp')`（测试只覆盖了前两个 → **漏 ddp**）
   - `DataConfig.loader`（`Literal[...] | None`）→ `('pickapic_preference','prompt_manifest','prompt_image_manifest')`
2. **`FAMILY_REGISTRY` / `entry.aliases`** → 迭代 `FAMILY_REGISTRY.items()`，alias 覆盖迭代 `entry.aliases`。canonical 不变量：`registered_rollout_families() == tuple(FAMILY_REGISTRY)`（同一份 `test_family_registry.py:20-32` 已经用 `registered_rollout_families()` 派生 family 集合做循环断言 —— 同文件里的 alias 测试却又手抄 12 条，自相矛盾）。
3. **派发函数（kind→config-class）** → 让 `build_algorithm_config` 自己当唯一真相，断言它的返回类型，而不是再维护一份镜像 dict。

**判据**：一个 `parametrize` 列表 / expected dict 如果只是把源头成员重抄一遍、再断言「成员 == 成员自己」，它**永远抓不到新增/缺失成员**，是纯冗余 + 漂移温床 → 改成从源头派生。
派生后**保留**每个成员原有的 per-member 行为分支（per-loader 构造、per-family scheduler 校验）—— 删的是「成员集合的硬拷贝」，不是「逐成员的行为覆盖」。

## 1. 现状实锤

### 1.1 `tests/config/test_schema.py:136-140` —— strategy 已漂移（漏 `ddp`）

源头 `vrl/config/schema.py:552`：

```python
strategy: Literal["single_process", "fsdp", "ddp"] = "single_process"
```

测试只覆盖两个：

```python
# tests/config/test_schema.py:136-140
@pytest.mark.parametrize("strategy", ["single_process", "fsdp"])
def test_valid_training_strategies_are_accepted(strategy: str) -> None:
    section = TrainingSection(strategy=strategy)
    assert section.strategy == strategy
```

`ddp` 在 `Literal` 里、却从不被这条「合法 strategy 都被接受」的测试验证 —— 正是 `wan_2_2 missing` 那类静默覆盖缺口。`docstring` 还写着「Both readiness strategies validate; the Literal is the only allow-list」，与实际三成员的 `Literal` 直接打架。

### 1.2 `tests/config/test_schema.py:48-60` —— kind 列表是 `Literal` 的逐字拷贝

源头 `vrl/config/schema.py:90-99`：

```python
kind: Literal[
    "grpo", "dance_grpo", "flow_dppo", "grpo_guard",
    "token_grpo", "token_grpo_multisegment", "diffusion_dpo", "diffusion_nft",
]
```

测试把这 8 个字符串原样抄进 `parametrize`（`test_schema.py:48-60`），再断言 `AlgorithmConfig(kind=k).kind == k`。它只是把 `Literal` echo 一遍，**永远无法发现**「`Literal` 新增了一个 kind 但没人改这份列表」。当前恰好对齐，但属于会静默腐烂的快照。

### 1.3 `tests/config/test_schema.py:297-299` —— loader 列表是 `Literal` 的逐字拷贝

源头 `vrl/config/schema.py:126`：

```python
loader: Literal["pickapic_preference", "prompt_manifest", "prompt_image_manifest"] | None = None
```

测试（`test_schema.py:297-299`）手抄 `["prompt_manifest", "prompt_image_manifest", "pickapic_preference"]`。`Literal` 新增 loader 不改这里就不被测。注意 `test_schema.py:303-321` 的 per-loader 构造分支是**真实行为覆盖**，要保留；只有成员清单是冗余拷贝。

### 1.4 `tests/models/diffusion/test_scheduler_logprob_parity.py:78-87` —— 已漂移 + 漏成员（最严重）

```python
# tests/models/diffusion/test_scheduler_logprob_parity.py:78-87
_FAMILY_SCHEDULERS = {
    "sd3_5": lambda: _hf_scheduler("stabilityai/stable-diffusion-3.5-medium"),
    "wan_2_1": lambda: _hf_scheduler("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
    "cosmos-predict2": lambda: _hf_scheduler("nvidia/Cosmos-Predict2-2B-Video2World"),
    "cosmos-predict2.5": lambda: _hf_scheduler("nvidia/Cosmos-Predict2.5-2B", revision="diffusers/base/post-trained"),
    "cosmos-anima": _anima_scheduler,
}
```

实测 `FAMILY_REGISTRY` 的 diffusion families（`entry.collector.kind == "diffusion"`）为：`sd3_5, flux, qwen_image, wan_2_1, wan_2_1_i2v, cosmos-predict2, cosmos-predict2.5, cosmos-predict2-anima`。对照得三处实锤：

1. **key 漂移**：字典写 `"cosmos-anima"`，注册表 canonical 名是 `"cosmos-predict2-anima"`（`registry.py` 实测 aliases 里 `anima`/`cosmos_anima` 才指向它）。
2. **漏 `flux` / `qwen_image`**：两者是新加的 flow-matching diffusion family，`vrl/models/diffusion/flux/runtime.py`、`qwen_image/runtime.py` 都加载 FlowMatch scheduler —— 正是 `wan_2_2 missing` 类缺口。
3. **漏 `wan_2_1_i2v`**。

且每个 repo id / revision（`test:79-85`）都是 family 的 `model.path` / `model.revision` 的逐字拷贝（实测：`configs/model/diffusion/sd3_5/medium.yaml:5`、`wan_2_1/1_3b.yaml:5`、`cosmos/predict2_2b.yaml:7`、`cosmos/predict2_5_2b.yaml:9-10` 完全一致）。config 改 checkpoint，这里仍测旧 repo 的 scheduler，静默不报。

### 1.5 `tests/rollouts/runtime/test_family_registry.py:52-65` —— alias 表是 `entry.aliases` 的手抄并集

```python
# tests/rollouts/runtime/test_family_registry.py:52-65
expected_aliases = {
    "flux_1_dev": "flux", "qwen-image": "qwen_image", "wan": "wan_2_1",
    "wan_i2v": "wan_2_1_i2v", "cosmos": "cosmos-predict2",
    "cosmos_predict2": "cosmos-predict2", "cosmos_predict2_5": "cosmos-predict2.5",
    "anima": "cosmos-predict2-anima", "cosmos_anima": "cosmos-predict2-anima",
    "janus": "janus_pro", "janus_r1": "janus_pro_r1", "nextstep": "nextstep_1",
}
for alias, expected in expected_aliases.items():
    assert normalize_rollout_family(alias) == expected
    assert get_rollout_family_entry(alias) is FAMILY_REGISTRY[expected]
```

源头 `vrl/rollouts/families/registry.py` 每个 entry 的 `aliases=(...)`，机械汇成 `_FAMILY_ALIASES`。实测两份完全一致 —— 但这正是 canonical frozen-registry bug 的 alias 孪生：任何 family 增/删一个 alias，这份手写 dict 立刻走样。讽刺的是**同文件 `test_family_registry.py:34` 已经用 `registered_rollout_families()` 迭代做 family 覆盖**，alias 测试却倒退回手抄。

### 1.6 `tests/config/test_load_all_experiments.py:32-43` —— kind→config-class 镜像了 builder 派发

```python
# tests/config/test_load_all_experiments.py:32-43
EXPECTED_ALGO_TYPE = {
    "grpo": GRPOConfig, "dance_grpo": GRPOConfig, "flow_dppo": FlowDPPOConfig,
    "grpo_guard": GRPOGuardConfig, "token_grpo": TokenGRPOConfig,
    "token_grpo_multisegment": MultiSegmentTokenGRPOConfig,
    "diffusion_dpo": DiffusionDPOConfig, "diffusion_nft": DiffusionNFTConfig,
}
```

这份 dict 手抄了 `vrl/config/builders.py:225-268` `build_algorithm_config` 的 kind→class 派发。唯一消费点 `test_load_all_experiments.py:343`：

```python
assert isinstance(algo_cfg, EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)])
```

而它**正上方 `:342`** 已经断言 `isinstance(algo_cfg, expected_type)`，`expected_type` 取自本地 `examples`（`:330-338`）逐 experiment 的预期类型。即 `:343` 断言的是「`build_algorithm_config` 返回了它自己一份拷贝声称的类型」。builder 改派发不改这份 dict → 要么 `KeyError`，要么静默断言一个 stale 类型。

## 2. 落地方案

canonical 派生模式（贯穿全篇）：**「源头是唯一真相，测试在 collection 期从源头算出成员集合」**。下面逐文件给 BEFORE/AFTER。

### A. `test_schema.py` 三处 `Literal` 派生（含修复 ddp 漂移）

在文件顶部加一处 helper（就近放在现有 import 下，不新建文件）：

```python
import typing

def _literal_args(annotation) -> tuple[str, ...]:
    """Flatten a Literal[...] or Literal[...] | None annotation into its members."""
    members = typing.get_args(annotation)
    # Optional[Literal[...]]: unwrap each non-None union member's Literal args.
    if any(m is type(None) for m in members):
        return tuple(
            a for m in members if m is not type(None) for a in typing.get_args(m)
        )
    return members
```

kind（BEFORE → AFTER）：

```python
# BEFORE  test_schema.py:48-60  (8 个字符串手抄 Literal)
@pytest.mark.parametrize("kind", ["grpo", "dance_grpo", ... , "diffusion_nft"])

# AFTER  从 Literal 注解派生，新增 kind 自动被覆盖
@pytest.mark.parametrize(
    "kind", _literal_args(AlgorithmConfig.model_fields["kind"].annotation)
)
def test_valid_algorithm_kinds_are_accepted(kind: str) -> None:
    assert AlgorithmConfig(kind=kind).kind == kind
```

strategy（修复漏 `ddp`）：

```python
# AFTER  test_schema.py:136-140  —— ddp 自动进入覆盖
@pytest.mark.parametrize(
    "strategy", _literal_args(TrainingSection.model_fields["strategy"].annotation)
)
def test_valid_training_strategies_are_accepted(strategy: str) -> None:
    assert TrainingSection(strategy=strategy).strategy == strategy
```

> 同步把该测试 docstring「Both readiness strategies」改成不写死成员数（例如「Every strategy in the Literal allow-list validates」）。

loader（保留 per-loader 构造分支）：

```python
# AFTER  test_schema.py:297-299  —— parametrize 派生，函数体内现有 if/elif 构造分支不动
@pytest.mark.parametrize(
    "loader", _literal_args(DataConfig.model_fields["loader"].annotation)
)
def test_valid_data_loaders_are_accepted(loader: str) -> None:
    if loader == "prompt_manifest":
        ...  # 现有分支保留
```

### B. `test_family_registry.py:52-65` alias 从 `entry.aliases` 派生

```python
# BEFORE  手抄 12 条 alias→family
expected_aliases = {"flux_1_dev": "flux", ...}
for alias, expected in expected_aliases.items():
    assert normalize_rollout_family(alias) == expected
    assert get_rollout_family_entry(alias) is FAMILY_REGISTRY[expected]

# AFTER  迭代注册表，每个声明的 alias 都断言「解析回其所属 entry」
def test_family_aliases_resolve_to_canonical_entries() -> None:
    seen = 0
    for family, entry in FAMILY_REGISTRY.items():
        # canonical name itself must resolve to its own entry
        assert normalize_rollout_family(family) == family
        for alias in entry.aliases:
            assert normalize_rollout_family(alias) == family
            assert get_rollout_family_entry(alias) is entry
            seen += 1
    assert seen > 0  # guard: registry must actually declare aliases
```

新 family / 新 alias 自动被覆盖；不再有 `cosmos-anima` 式 key 漂移可能（key 直接来自 entry）。

### C. `test_scheduler_logprob_parity.py:78-87` 从注册表派生 diffusion family + 从 config 读 path

把硬写 dict 换成「迭代注册表的 diffusion family → 从该 family 的 model config 读 `model.path`/`model.revision` → 缺 scheduler 缓存则 skip」。anima 因为没有可加载的 HF scheduler config（它在 `runtime.py` 里手搓），用注册表名 `cosmos-predict2-anima` 显式映射到 `_anima_scheduler`，其余从 config 派生：

```python
# AFTER  test_scheduler_logprob_parity.py
from vrl.config.loading import load_config
from vrl.rollouts.families.registry import FAMILY_REGISTRY

# anima 没有可下载的 scheduler config（runtime.py 手动构造），单独映射；
# 其余 diffusion family 的 scheduler 一律从其 model config 的 path/revision 派生。
_MANUAL_SCHEDULERS = {"cosmos-predict2-anima": _anima_scheduler}

# 每个 diffusion family → 其 model config 路径（loader 实际加载的那份）。
# key 由注册表派生，杜绝 cosmos-anima 式 key 漂移。
_FAMILY_MODEL_CONFIG = {
    "sd3_5": "model/diffusion/sd3_5/medium",
    "flux": "model/diffusion/flux/dev",
    "qwen_image": "model/diffusion/qwen_image/base",
    "wan_2_1": "model/diffusion/wan_2_1/1_3b",
    "cosmos-predict2": "model/diffusion/cosmos/predict2_2b",
    "cosmos-predict2.5": "model/diffusion/cosmos/predict2_5_2b",
}

def _diffusion_families() -> list[str]:
    return [f for f, e in FAMILY_REGISTRY.items() if e.collector.kind == "diffusion"]

def _scheduler_for(family: str):
    if family in _MANUAL_SCHEDULERS:
        return _MANUAL_SCHEDULERS[family]()
    cfg_name = _FAMILY_MODEL_CONFIG.get(family)
    if cfg_name is None:
        pytest.skip(f"no cached scheduler mapping for diffusion family {family!r}")
    model = load_config(cfg_name).model  # path/revision come from the YAML, not re-typed
    return _hf_scheduler(model.path, revision=getattr(model, "revision", None))

@pytest.mark.parametrize("sde_type", ["flow_grpo", "cps"])
@pytest.mark.parametrize("family", sorted(_diffusion_families()))
def test_family_scheduler_sample_replay_parity(family: str, sde_type: str) -> None:
    scheduler = _scheduler_for(family)
    ...  # 现有 ratio==1 + sigma 不变量断言保留
```

> 这样：注册表新增 flow-matching family 时，若已登记 model config 则**自动覆盖**，否则**自动 skip**（永不静默缺席）。`wan_2_1_i2v` 若无独立 scheduler config 也会被显式 skip 而非被遗忘。`_FAMILY_MODEL_CONFIG` 仍是手维护映射，但它是「family→config 路径」而非「family→repo id」—— repo id/revision 不再被重抄，且缺失映射会 skip 报出 family 名，逼出补登记。如果后续 `FamilyEntry` 暴露了 model-config 路径字段，这张表也应改为从 entry 直接派生。
>
> 实现时核对 `load_config` 能否单独加载 `model/...` 组（若 loader 要求完整 experiment context，则改为直接 `OmegaConf.load(configs/model/...yaml)` 读 `model.path`/`model.revision`，仍然是「从 config 读」而非重抄）。

### D. `test_load_all_experiments.py:32-43` 删镜像 dict，让 builder 当唯一真相

`:342` 已用本地 `examples`（`:330-338`）的 `expected_type` 断言每个 experiment 的预期类型 —— 这份才是有意义的 per-experiment 契约。`EXPECTED_ALGO_TYPE` 与 `:343` 是它的冗余镜像：

```python
# BEFORE
EXPECTED_ALGO_TYPE = {"grpo": GRPOConfig, ...}   # :32-43
...
assert isinstance(algo_cfg, expected_type)                              # :342  (keep)
assert isinstance(algo_cfg, EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)])  # :343  (drop)

# AFTER  —— 删 EXPECTED_ALGO_TYPE dict 与 :343；
# 让 build_algorithm_config 自己当 kind→class 的唯一真相，并验证派发稳定：
assert isinstance(algo_cfg, expected_type)                              # :342 保留
# build_algorithm_config 是 kind→class 的单一源头：同一 kind 两次构建类型一致
assert type(build_algorithm_config(cfg)) is type(algo_cfg)
```

> 删掉镜像 dict 后，未用到的 import（`GRPOConfig` 等若仅服务该 dict）按 ruff 提示一并清掉；`examples`（`:330-338`）保留 —— 它是「具体 experiment → 预期算法类型」的真实回归锚点，不是 builder 派发的镜像。

## 3. 验证（finishing criteria）

- 派生有效性（grep 确认硬抄清单已消失）：
  - `grep -n '"grpo"' tests/config/test_schema.py` 不再命中 `parametrize` 的 kind 列表（仅 helper/fixture 里的 `{"kind": "grpo"}` 这类构造保留）。
  - `grep -n '"single_process", "fsdp"\]' tests/config/test_schema.py` 零命中。
  - `grep -n 'cosmos-anima\|"flux_1_dev"\|expected_aliases' tests/ -r` 零命中（alias 表、漂移 key 已删）。
  - `grep -n 'EXPECTED_ALGO_TYPE' tests/config/test_load_all_experiments.py` 零命中。
  - `grep -n '_FAMILY_SCHEDULERS\|"cosmos-anima"' tests/models/diffusion/test_scheduler_logprob_parity.py` 零命中。
- 覆盖度自检（派生集合非空 + 命中漂移成员）：
  - `python -c "import typing; from vrl.config.schema import TrainingSection as T; assert 'ddp' in typing.get_args(T.model_fields['strategy'].annotation)"` 通过（确认 ddp 现已进入派生集合）。
  - `python -c "from vrl.rollouts.families.registry import FAMILY_REGISTRY as R; d=[f for f,e in R.items() if e.collector.kind=='diffusion']; assert {'flux','qwen_image','wan_2_1_i2v','cosmos-predict2-anima'} <= set(d)"` 通过。
- pytest 全绿：
  - `pytest tests/config/test_schema.py -q`（kind/strategy/loader 三处 parametrize 现按派生集合展开，`ddp` case 出现且通过）。
  - `pytest tests/config/test_load_all_experiments.py -q`。
  - `pytest tests/rollouts/runtime/test_family_registry.py -q`。
  - `pytest tests/models/diffusion/test_scheduler_logprob_parity.py -q`（缓存缺失的 family 走 skip，不报 fail）。
- 全量 `pytest -q` 与 `ruff check tests/` 零回归（删除镜像 dict 后无未用 import）。

## 4. 非目标 / Non-Goals

- **不改这些 allow-list 的源头本身**（`Literal` 成员、`FAMILY_REGISTRY` 条目、`entry.aliases`、builder 派发分支）—— 它们是源头，本 sprint 只让测试停止重抄。
- **不删 per-member 行为覆盖**：`test_schema.py` 的 per-loader 构造分支、scheduler-parity 的 ratio==1 + sigma 不变量、`test_load_all_experiments` 的 `examples` 逐 experiment 锚点全部保留。删的只是「成员集合的硬拷贝」。
- **不动 `tests/models/diffusion/registry.py:27-34`** 的 `hf-internal-testing/tiny-*` repo + commit-hash —— 经核实是 genuine external pin（固定上游 test-fixture commit），不是内部派生违例。
- **不扩展到本主题外的 finding**：同一审计 JSON 里 `literal_config_assertion`（断言 YAML 声明值）、`frozen_snapshot`（目录 listing 快照，如 `test_generation_rollout_boundaries.py` 的 `hub.py` 漂移）、`magic_number`、`brittle_string_match` 等属其它主题，由各自 sprint 处理；本 sprint 只收口「registry/enum/alias 成员集合漂移」这一类。
- **不为 anima 重构 `runtime.py` scheduler 构造**：C 节保留 `_anima_scheduler` 手搓 mirror（它本就无可下载的 HF config），仅修正其 key 为注册表 canonical 名。

## References

- `tests/config/test_schema.py:48-60,136-140,297-321`
- `tests/config/test_load_all_experiments.py:32-43,330-343`
- `tests/models/diffusion/test_scheduler_logprob_parity.py:71-99`
- `tests/rollouts/runtime/test_family_registry.py:18-68`
- `vrl/config/schema.py:90-99,126,552`
- `vrl/config/builders.py:220-268`
- `vrl/rollouts/families/registry.py`（`FAMILY_REGISTRY`、`_FAMILY_ALIASES`、`registered_rollout_families()`、`CollectorSpec.kind`、`FamilyEntry.aliases`）
- `configs/model/diffusion/{sd3_5/medium,flux/dev,qwen_image/base,wan_2_1/1_3b,cosmos/predict2_2b,cosmos/predict2_5_2b}.yaml`（`model.path` / `model.revision` 源头）
- 关联：[[SPRINT_segment_signal_dead_field_cleanup]]（同一「源头是唯一真相、消费者/源头定义说了算」的派生结构体审计准则）
