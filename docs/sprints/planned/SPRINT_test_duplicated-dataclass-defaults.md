# SPRINT: 测试重打 dataclass/builder 默认值，而非断言行为（planned）

状态：未开始（2026-06-21）。
范围：清理一类测试坏味道 —— 测试把 config dataclass 的字段默认值、或 builder 注入的结构体常量**手抄一份**，再断言「源码的值 == 我自己抄的那份」。这类断言不验证任何行为：默认值一旦被合理重调（换 paper 取值、调漂移阈值、改 dispatch），测试就为「无行为回归」的理由而红；dispatch 镜像 dict 还会 `KeyError` 或静默断言陈旧类型。优先级 medium。涉及 **4 个测试文件**（`test_dpo.py` / `test_logprob_mismatch.py` / `test_online_precision_bridge.py` / `test_load_all_experiments.py`）。**不**改任何被测源码、**不**删真正的行为型 no-op 测试与显式 override 测试。

> 与 [[SPRINT_test_frozen_registry_snapshots]] 同源 —— 那条收口「手抄目录/registry 清单」的 frozen snapshot；本条收口「手抄单个默认值/dispatch 映射」的 duplicated-default。规则同一条：typed structure 是唯一 source of truth，重复的常量在源类新增字段那一刻就开始烂。

## 0. Core Decision（先看这一段）

判一条断言是不是坏味道，看它**是否复述了一份它本可以直接 import 的源码默认值**。canonical 反例是 `wan_2_2 missing`：`registered_rollout_families() == tuple(FAMILY_REGISTRY)`，所以任何把 family key 手写成 literal list 的测试都是 bug —— 同理，把 `DiffusionDPOConfig.beta=5000.0` 抄进测试、把 `build_algorithm_config` 的 `kind->class` dispatch 镜像成一个 dict，都是同一个 bug 的小型变种。

三档处理：

1. **等于默认值的断言**（`cfg.beta == 5000.0`、`pc.rs_log_ratio_low == LN_HALF`）→ 它只复述了字段默认，不测任何行为。**删**，或改成「未设置的 config 与显式传入文档默认值的 config 产出相同结果」的行为断言。
2. **auto 派生策略的逐值断言**（split-precision 装上的 `truncate`/`seq_mean_k1`/`math.log(10.0)`/`9.0`）→ 真正契约是「rollout≠train 时才触发，且装上的就是 builder helper 会产出的那套」。**改成派生不变量**：断言触发条件 + 等于 `_apply_rollout_precision_defaults` 会产出的结构体，而不是逐个抄 builder 的字面常量。
3. **dispatch 镜像 dict**（`EXPECTED_ALGO_TYPE = {kind: Class}`）→ 它把 `build_algorithm_config` 自己拥有的映射又手抄了一遍，只为断言「dispatch 返回的类 == 一份 dispatch 拷贝声称的类」。**删 dict**，让 `build_algorithm_config` 当唯一 dispatch（同 kind 产出同类）。

非目标里保留两类真断言：行为型 no-op（`test_off_returns_none` 等）、显式 override（`test_explicit_precision_correction_is_respected_on_rollout_split`）—— 它们测的是行为，不是默认值的拷贝。

## 1. 现状实锤

### 1.1 `test_dpo.py:173-177` —— 重打 `DiffusionDPOConfig` 字段默认值

`tests/algorithms/test_dpo.py:173-177`：

```python
def test_dpo_config_defaults() -> None:
    """Checks DPO config defaults."""
    cfg = DiffusionDPOConfig()
    assert cfg.beta == 5000.0
    assert cfg.sft_weight == 0.0
```

source of truth `vrl/algorithms/dpo.py:32-33`：

```python
beta: float = 5000.0
sft_weight: float = 0.0  # optional auxiliary SFT-on-winner loss
```

整个测试只是把字段默认重写一遍。`beta` 的 docstring 明说 SD1.5=5000 / SDXL=2000 是 paper 取值，换 backbone 就该改默认 —— 一改这两行就红，但 `diffusion_dpo_loss` 的行为没任何变化。「configs are declarations, don't assert literal values」规则应用到 dataclass 默认上的标准案例。

### 1.2 `test_logprob_mismatch.py:80-81,89-92` —— 重打 RS band + RS/recompute 默认

`tests/algorithms/test_logprob_mismatch.py:80-81` 把源码默认表达式抄进本地常量：

```python
LN_HALF = math.log(0.5)
LN_TWO = math.log(2.0)
```

`:87-92` 用它断言 config 默认 == 自己的拷贝：

```python
def test_defaults_are_off_and_symmetric_band(self) -> None:
    pc = PrecisionCorrectionConfig()
    assert pc.rs_mode == "off"
    assert pc.recompute_old_logprob == "off"
    assert pc.rs_log_ratio_low == pytest.approx(LN_HALF)
    assert pc.rs_log_ratio_high == pytest.approx(LN_TWO)
```

source of truth `vrl/algorithms/logprob_mismatch.py:120,124-125,128`：

```python
rs_mode: str = field(default="off")  # "off" | "seq_mean_k1" | "seq_max_k1"
rs_log_ratio_low: float = field(default=math.log(0.5))
rs_log_ratio_high: float = field(default=math.log(2.0))
recompute_old_logprob: str = field(default="off")  # "off" | "on"
```

`rs_log_ratio_low/high` 的断言是把源码默认表达式 `math.log(0.5)/math.log(2.0)` 原样抄进 `LN_HALF/LN_TWO` 再自比 —— band 一被重调（注释说默认对齐 verl-omni LLM 预设，换预设就会变）即烂，行为无变化。
注意：`LN_HALF/LN_TWO` 同时是 `:128-132` RS-mask 测试的**合法固定输入**（那里它们是显式 band，不是默认拷贝），保留。
`rs_mode=="off"` / `recompute_old_logprob=="off"` 低风险些（「默认 off = 必须 opt-in」勉强算安全契约），但仍是等于默认值的复述，且 no-op 行为已被本文件的 `test_off_returns_none` 等行为测试覆盖。

### 1.3 `test_online_precision_bridge.py:84-91` —— 逐值重打 auto split-precision 策略

`tests/scripts/test_online_precision_bridge.py:84-91`：

```python
    assert trainer_config.precision_correction.tis_mode == "truncate"
    assert trainer_config.precision_correction.rs_mode == "seq_mean_k1"
    assert trainer_config.precision_correction.recompute_old_logprob == "off"
    assert trainer_config.precision_drift_guard.mode == "fail"
    assert trainer_config.precision_drift_guard.max_abs_log_ratio == pytest.approx(
        math.log(10.0),
    )
    assert trainer_config.precision_drift_guard.max_ratio_abs_dev == pytest.approx(9.0)
```

source of truth 是 builder helper `vrl/config/builders.py:21,34-51`（函数名实测为 `_apply_rollout_precision_defaults`，**不是** JSON 里写的 `_apply_split_precision_defaults`）：

```python
def _apply_rollout_precision_defaults(...):
    if precision.rollout == precision.train:
        return
    ...
    if not path_exists(cfg, "trainer.precision_correction"):
        payload["precision_correction"] = PrecisionCorrectionConfig(
            tis_mode="truncate",
            rs_mode="seq_mean_k1",
        )
    if not path_exists(cfg, "trainer.precision_drift_guard"):
        payload["precision_drift_guard"] = PrecisionDriftGuardConfig(
            mode="fail",
            max_abs_log_ratio=math.log(10.0),
            max_ratio_abs_dev=9.0,
            fail_on_nonfinite=True,
        )
```

测试逐字抄了 builder 注入的每个常量。真正契约是「rollout 精度 != train 且用户没给显式 block 时，装上一套 auto 修正策略」。一旦有人在 builder 里调默认漂移阈值/修正模式，测试就红，但它声称保护的 wiring 仍然工作。

**实锤纠偏（关键）**：`mode="fail"` / `max_abs_log_ratio=math.log(10.0)` / `max_ratio_abs_dev=9.0` **不是** `PrecisionDriftGuardConfig` 的 dataclass 默认 —— 实测 `vrl/trainers/core/types.py:110,112-113` 的默认是 `mode="auto"` / `1e-3` / `1e-3`。这三个值是 builder helper **注入**的。所以 AFTER 不能拿 `PrecisionDriftGuardConfig()` 裸默认去比（JSON 修复建议在这点上是错的），必须比「builder helper 会产出的那个结构体」。`PrecisionCorrectionConfig` 那侧 builder 只设了 `tis_mode/rs_mode`，其余（含 `rs_log_ratio_*`/`recompute`）才走 dataclass 默认。

显式 override 测试 `test_explicit_precision_correction_is_respected_on_rollout_split`（:94-）是真行为契约，保留。

### 1.4 `test_load_all_experiments.py:32-43` + `:343` —— `kind->class` dispatch 镜像 dict

`tests/config/test_load_all_experiments.py:32-43` 手抄了一份 dispatch：

```python
EXPECTED_ALGO_TYPE = {
    "grpo": GRPOConfig,
    "dance_grpo": GRPOConfig,
    "flow_dppo": FlowDPPOConfig,
    "grpo_guard": GRPOGuardConfig,
    "token_grpo": TokenGRPOConfig,
    "token_grpo_multisegment": MultiSegmentTokenGRPOConfig,
    "diffusion_dpo": DiffusionDPOConfig,
    "diffusion_nft": DiffusionNFTConfig,
}
```

唯一消费点 `:343`：

```python
        assert isinstance(algo_cfg, EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)])
```

source of truth `vrl/config/builders.py:220-268` `build_algorithm_config` 的 `if kind == ...` dispatch 自己就拥有这套 `kind->class` 映射。这个 dict 把它原样抄了一份，`:343` 实际断言的是「`build_algorithm_config` 返回的类 == 一份 `build_algorithm_config` 拷贝声称的类」。在 builders.py 加/改一个 kind 而不同步本 dict：要么 `KeyError`，要么静默断言陈旧类型。`:330-343` 的 `test_algorithm_config_dispatches_representative_kinds` 已经用 `examples` dict 自带了 `kind->expected_type` 并断言 `isinstance(algo_cfg, expected_type)`，`:343` 这一行是同一断言的镜像冗余。

## 2. 落地方案

canonical 派生模式（贯穿全篇）：**不要在测试里复述默认值，要么从源 import 后比较，要么断言「未设置 == 显式传文档默认」的行为等价**。dispatch 同理 —— 让 `build_algorithm_config` 当唯一 dispatch，测试只断言「同 kind 产出同类」。

### A. `test_dpo.py:173-177` —— 删等于默认值的断言

直接删 `test_dpo_config_defaults`：它无行为内容。`beta`/`sft_weight` 的真实作用已被本文件的 loss 行为测试（如 `test_sft_loss_is_winner_mse`、DPO loss 测试）覆盖。

若要保留「默认可用」的轻量保证，改成行为等价断言（默认 config 与显式传默认值的 config 产出同一 loss），而不是重打值：

```python
# BEFORE
cfg = DiffusionDPOConfig()
assert cfg.beta == 5000.0
assert cfg.sft_weight == 0.0

# AFTER（行为等价：未设默认 == 显式传 dataclass 默认）
default_cfg = DiffusionDPOConfig()
explicit_cfg = DiffusionDPOConfig(beta=default_cfg.beta, sft_weight=default_cfg.sft_weight)
# 用同一对输入跑 diffusion_dpo_loss，两份 cfg 的 loss 必须一致
assert torch.allclose(_loss_with(default_cfg), _loss_with(explicit_cfg))
```

首选直接删（行为已被覆盖）；上面的等价写法仅在 owner 想留一个 default-smoke 时采用。

### B. `test_logprob_mismatch.py` —— 删 band/默认 的等值断言，保留 mask 输入

保留 `LN_HALF/LN_TWO`（mask 测试的合法输入），删 `test_defaults_are_off_and_symmetric_band` 里 4 条等于默认值的断言：

```python
# BEFORE (:87-92)
def test_defaults_are_off_and_symmetric_band(self) -> None:
    pc = PrecisionCorrectionConfig()
    assert pc.rs_mode == "off"
    assert pc.recompute_old_logprob == "off"
    assert pc.rs_log_ratio_low == pytest.approx(LN_HALF)
    assert pc.rs_log_ratio_high == pytest.approx(LN_TWO)

# AFTER —— 删整个等值断言测试；off 的 no-op 行为已由 test_off_returns_none 覆盖。
# 若 owner 坚持要 pin「默认 band 对称」这一不变量，断言对称性（派生关系）而非具体值：
def test_default_band_is_symmetric_around_zero(self) -> None:
    pc = PrecisionCorrectionConfig()
    assert pc.rs_log_ratio_low == pytest.approx(-pc.rs_log_ratio_high)  # 不重打 ln2/ln0.5
```

`LN_HALF/LN_TWO` 仍由 `TestRejectSampleMask._cfg`（:128-132）作为显式 band 输入使用，不动。

### C. `test_online_precision_bridge.py:84-91` —— 断言派生不变量，不抄 builder 常量

改成「触发条件 + 等于 builder helper 会产出的结构体」。helper 是模块私有，从 `vrl.config.builders` import 后直接调用其产出来比较，让 builder 当唯一 source of truth：

```python
# BEFORE (:84-91) —— 逐字抄 builder 注入的 truncate/seq_mean_k1/fail/ln10/9.0
assert trainer_config.precision_correction.tis_mode == "truncate"
assert trainer_config.precision_correction.rs_mode == "seq_mean_k1"
assert trainer_config.precision_correction.recompute_old_logprob == "off"
assert trainer_config.precision_drift_guard.mode == "fail"
assert trainer_config.precision_drift_guard.max_abs_log_ratio == pytest.approx(math.log(10.0))
assert trainer_config.precision_drift_guard.max_ratio_abs_dev == pytest.approx(9.0)

# AFTER —— 派生不变量：split 时非 None，且等于 builder helper 注入的结构体
from vrl.algorithms.logprob_mismatch import PrecisionCorrectionConfig
from vrl.trainers.core.types import PrecisionDriftGuardConfig

# builder 在 rollout!=train 且无显式 block 时注入的正是这两个结构体（单一 source）
expected_correction = PrecisionCorrectionConfig(tis_mode="truncate", rs_mode="seq_mean_k1")
expected_guard = PrecisionDriftGuardConfig(
    mode="fail", max_abs_log_ratio=math.log(10.0), max_ratio_abs_dev=9.0, fail_on_nonfinite=True,
)
assert trainer_config.precision_correction == expected_correction
assert trainer_config.precision_drift_guard == expected_guard
```

注意：`PrecisionDriftGuardConfig` 的 `fail`/`ln10`/`9.0` **不是** dataclass 默认（默认是 `auto`/`1e-3`/`1e-3`），所以 AFTER 仍需写出 builder 注入的那套参数 —— 这不是「重打默认」，而是「重述 builder 的注入策略」。若想连这层重述都消掉，把 builder helper 重构成可调用的纯函数（返回 `(correction, guard)`），测试直接调它对比；但**这属于动源码**，超出本 sprint 范围（非目标）。本 sprint 内的最小修复：等值改 `==` 整体结构体比较，并补一条**触发条件**断言 —— rollout==train 时不应装策略：

```python
def test_no_split_means_no_auto_correction_policy() -> None:
    cfg = _with_precision("diffusion/sd3_5/online_grpo_ocr", {"train": "bf16", "rollout": "bf16"})
    trainer_config = build_configs(cfg)["trainer"]
    # rollout == train：builder 早 return，不注入 auto 策略（精确语义按现状源码核对后写断言）
```

（落地时先核对 `build_configs` 在无 split 时 `precision_correction` 的实际取值再写最终断言，避免凭空假设为 None。）

### D. `test_load_all_experiments.py:32-43,343` —— 删 dispatch 镜像 dict

删 `EXPECTED_ALGO_TYPE` 整块及 `:343` 的镜像断言。`:330-342` 的 `test_algorithm_config_dispatches_representative_kinds` 已用 `examples` dict 断言 `isinstance(algo_cfg, expected_type)`，足够；再补一条「同 kind 产出同类」的 dispatch 稳定性不变量，让 `build_algorithm_config` 当唯一 dispatch：

```python
# BEFORE (:32-43) 删整个 dict；(:343) 删这一行
assert isinstance(algo_cfg, EXPECTED_ALGO_TYPE[str(cfg.algorithm.kind)])

# AFTER —— dispatch 稳定性：同 kind 两次构造产出同一类（不镜像映射）
def test_algorithm_dispatch_is_stable_per_kind() -> None:
    for name in ("diffusion/sd3_5/online_grpo_ocr", "ar/janus_pro/online_grpo_ocr",
                 "diffusion/wan_2_1/offline_dpo_pickapic"):
        cfg = load_config(f"experiment/{name}")
        first = build_algorithm_config(cfg)
        second = build_algorithm_config(cfg)
        assert type(first) is type(second)  # build_algorithm_config 是唯一 dispatch
```

`examples` dict（:332-338）保留 —— 它是「这几个真实 experiment 解析出预期算法类」的行为锚点，不是 dispatch 镜像；它的 `expected_type` 来自人对 experiment 语义的判断，而非从 builders 抄映射。删 `EXPECTED_ALGO_TYPE` 后，相应删 `:343` 那一行即可，`:339-342` 的 `isinstance(algo_cfg, expected_type)` 留着。

## 3. 验证（finishing criteria）

- `grep -rn "EXPECTED_ALGO_TYPE" tests/` 零命中（dispatch 镜像 dict 已删）。
- `grep -rn "== 5000.0\|sft_weight == 0.0" tests/algorithms/test_dpo.py` 零命中。
- `grep -rn "rs_log_ratio_low == pytest.approx(LN\|rs_log_ratio_high == pytest.approx(LN" tests/algorithms/test_logprob_mismatch.py` 零命中；`LN_HALF`/`LN_TWO` 仍可在 mask 测试作为输入存在（`grep -n "LN_HALF\|LN_TWO" tests/algorithms/test_logprob_mismatch.py` 只剩定义 + mask `_cfg` 用法）。
- `grep -rn 'tis_mode == "truncate"\|rs_mode == "seq_mean_k1"\|max_abs_log_ratio == pytest.approx(' tests/scripts/test_online_precision_bridge.py` 零命中；改为结构体 `==` 比较。
- `pytest tests/algorithms/test_dpo.py tests/algorithms/test_logprob_mismatch.py -q` 全绿。
- `pytest tests/scripts/test_online_precision_bridge.py -q` 全绿（含保留的显式 override 测试 + 新增的 no-split 触发条件测试）。
- `pytest tests/config/test_load_all_experiments.py -q` 全绿（dispatch 稳定性测试 + 保留的 `examples` 行为锚点）。
- 全量 `pytest -q` 零回归（确认删 dict / 改断言未牵连其他 import）。

## 4. 非目标 / Non-Goals

- **不改任何被测源码**：`DiffusionDPOConfig` / `PrecisionCorrectionConfig` / `PrecisionDriftGuardConfig` 的字段默认、`build_algorithm_config` 的 dispatch、`_apply_rollout_precision_defaults` 的注入逻辑全部不动。本 sprint 只动测试。
- **不删行为型 no-op 测试**：`test_off_returns_none` 及 RS-mask 系列是真行为契约，保留。
- **不删显式 override 测试**：`test_explicit_precision_correction_is_respected_on_rollout_split` 验证「专家 block 覆盖 auto 默认」，是真契约，保留。
- **不为消除「重述 builder 注入策略」而重构 builder helper**：把 `_apply_rollout_precision_defaults` 提成纯函数供测试调用会改源码语义边界，超范围；本 sprint 的 C 仅把逐值断言收成整体结构体 `==` 比较 + 触发条件断言。
- **不扩展到 frozen-registry/目录快照那一类**：那是 [[SPRINT_test_frozen_registry_snapshots]] 的范围（`registered_rollout_families()`、reward `models/` 目录清单、`AlgorithmConfig.kind` Literal 等），本 sprint 只收口「重打单个默认值 / dispatch 镜像」这 4 个文件。

## References

- `tests/algorithms/test_dpo.py:173-177` —— DPO 默认值复述
- `vrl/algorithms/dpo.py:32-33`（`DiffusionDPOConfig.beta=5000.0` / `sft_weight=0.0`）
- `tests/algorithms/test_logprob_mismatch.py:80-81,87-92,128-132` —— RS band/默认复述 + 合法 mask 输入
- `vrl/algorithms/logprob_mismatch.py:120,124-125,128`（`rs_mode`/`rs_log_ratio_low`/`rs_log_ratio_high`/`recompute_old_logprob` 默认）
- `tests/scripts/test_online_precision_bridge.py:84-91,94-`（auto split 逐值断言 + 保留的显式 override 测试）
- `vrl/config/builders.py:21,34-51,220-268`（`_apply_rollout_precision_defaults` 注入策略 + `build_algorithm_config` dispatch）
- `vrl/trainers/core/types.py:110,112-113`（`PrecisionDriftGuardConfig` 真实默认 `auto`/`1e-3`/`1e-3` —— 证明 bridge 里那套是 builder 注入而非 dataclass 默认）
- `tests/config/test_load_all_experiments.py:32-43,330-343`（`EXPECTED_ALGO_TYPE` 镜像 dict + 消费点 + 保留的 `examples` 锚点）
- 关联：[[SPRINT_test_frozen_registry_snapshots]]（手抄 registry/目录清单的 frozen snapshot，同源坏味道的另一半）；[[SPRINT_segment_signal_dead_field_cleanup]]（同样以「消费者是 source of truth」为裁决依据的字段审计）
