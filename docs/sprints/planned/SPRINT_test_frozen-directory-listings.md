# SPRINT: 干掉穷举式冻结目录清单断言（planned）

状态：未开始（2026-06-21）。
范围：清理 `tests/architecture/test_generation_rollout_boundaries.py` 里 **4 处** `_module_filenames(dir) == {手敲全集}` 的穷举冻结目录清单断言。这类断言把「目录里有哪些文件」当成一份手抄的 `ls` 快照来 pin，文件系统一旦新增/重命名一个模块就立刻红，且红的原因与行为无关。已经发生 **一次 LIVE FAILURE**（详见 §1.1）。本 sprint 只动这一个测试文件，把穷举 `==` 换成它真正想表达的结构不变量（forbidden imports / not-exists 守卫 / 从 `_REWARD_REGISTRY` 派生的 reward 模块集），**不**改任何 `vrl/` 生产代码、**不**动该文件里已经正确的 forbidden-text / not-exists 断言。

> 本 sprint 的核心反模式与 `registered_rollout_families()` 一致：`vrl/rollouts/families/registry.py:364` 是 `return tuple(FAMILY_REGISTRY)` —— 注册表本身就是来源。任何「手敲一份 key 全集再断言 ==」的测试，都是把来源抄了一份副本，副本必然 rot（新增 family 不更新就红，或更新了也只是机械同步、抓不到真 bug）。目录清单是同一个 bug 的文件系统版本：**文件系统是来源**，断言要么从更上游的来源（注册表）派生，要么只断言「该在的在 / 不该在的不在」，绝不穷举抄一遍 `ls`。

## 0. Core Decision（先看这一段）

四处断言的真实意图早已由相邻的 forbidden-text / not-exists 断言守住，穷举 `==` 集合是冗余且会 rot 的那一半。逐处处理：

1. **`rewards/models/`（§1.1，severity high，LIVE FAILURE）** —— 这个目录是 reward 实现的落点，它的「该有什么」有上游来源：`vrl/rewards/functions/registry.py` 的 `_REWARD_REGISTRY`。每个注册的 reward 名 `<name>` 都对应一个 `<name>.py` 模块。把期望集**从注册表派生**（注册表 key → 模块文件名）+ 固定脚手架白名单（`__init__.py` / `base.py` / `hub.py`），断言「每个注册 reward 都有模块」+「目录里没有 misplaced 文件」，而不是穷举。这样 `hub.py` 这种纯增量的基础设施文件一次过，不用编辑测试。
2. **`rewards/ray/`（§1.2）、`rewards/`（§1.3 上半）、`generation/ray/`（§1.4）** —— 这三处没有更上游的注册表来源（它们是 adapter / 顶层脚手架目录，不映射任何 vocabulary）。它们的真实意图是「这个 adapter 保持精简：不泄漏 model-specific 代码、没有 `spec.py` / `inference` 目录、不 import `build_engine_plan` / `execution.chunks`」—— 而这些**已经**由各自下方的 forbidden-text / forbidden-import / not-exists 断言守住了。穷举 `==` 在这里纯属多余的会-rot 副本：直接删 `==`，把「必需脚手架在不在」降级为 **presence/subset 检查**（`required <= actual`），让增量文件不再误伤。
3. **`rewards/functions/`（§1.3 下半）** —— 与 §1.1 同源：reward function 模块集是 `_REWARD_REGISTRY` 的镜像。从注册表派生期望 reward 模块 + 固定脚手架（`__init__.py` / `base.py`? / `registry.py`），不再手敲。

**统一派生模式**（下文 §2 所有 reward 目录复用）：注册表是懒注册的（`_register_builtins()` 在 `MultiReward.from_dict` 时才填，见 `vrl/rewards/functions/registry.py:98`），测试里要先触发注册再读 key。最干净的做法是在测试顶部 `MultiReward.from_dict({}, device="cpu")` 触发一次注册（空 dict 不构造任何 reward 实例），再 `from vrl.rewards.functions.registry import _REWARD_REGISTRY` 读 key。reward 名到模块名是 identity 映射（`<name>.py`），无需第二份映射表。

## 1. 现状实锤

文件：`tests/architecture/test_generation_rollout_boundaries.py`。辅助函数 `_module_filenames`（`:254-255`）= `{path.name for path in root.glob("*.py")}`，即对目录做一次 `ls *.py`。下面四处把它的返回值 `==` 一个手敲全集。

### 1.1 `rewards/models/` —— confirmed LIVE FAILURE（severity high）

`test_reward_models_live_under_models`（`:95-111`）：

```python
# tests/architecture/test_generation_rollout_boundaries.py:98-108
assert _module_filenames(models_root) == {
    "__init__.py", "aesthetic.py", "base.py", "geneval.py",
    "kling_video_reward.py", "nsfw_safety.py", "ocr.py",
    "pickscore.py", "videocon_physics.py",
}
```

实地 `ls vrl/rewards/models/*.py` 多出一个 `hub.py`（新增，纯增量、非行为）：

```
aesthetic.py base.py geneval.py hub.py __init__.py
kling_video_reward.py nsfw_safety.py ocr.py pickscore.py videocon_physics.py
```

所以该断言**当前就是红的**：`Extra items in the left set: 'hub.py'`。这是教科书级穷举快照 rot —— `hub.py` 是 reward model 的下载/缓存基础设施（落在 `models/` 完全正确），但因为没人去手敲全集里补一行，测试把一个合法的增量改动判成失败。

真实意图（紧随其后的断言已守住）：

```python
# :109-111
assert not (VRL_ROOT / "rewards" / "kling_video_reward.py").exists()      # 没放错到 rewards 根
assert not (VRL_ROOT / "rewards" / "ray" / "kling_video_reward.py").exists()
assert not (VRL_ROOT / "rewards" / "scorers").exists()                     # 没有旧的 scorers/ 目录
```

上游来源已核实：`vrl/rewards/functions/registry.py:35-43` 的 `_REWARD_REGISTRY.update({...})` 注册的 7 个 key（`aesthetic / geneval / nsfw_safety / ocr / pickscore / kling_video_reward / videocon_physics`）与 `models/` 下的 7 个非脚手架模块**逐一对应**（`<key>.py`）。

### 1.2 `rewards/ray/` —— 同款穷举快照（severity medium）

`test_reward_ray_adapter_stays_lean`（`:67-92`）：

```python
# :70-75
assert _module_filenames(ray_root) == {
    "__init__.py", "model.py", "runtime.py", "worker.py",
}
```

实地 `ls` 当前恰好匹配（`__init__.py model.py runtime.py worker.py`），但这是穷举副本，加/改一个文件就红。该测试**真正**守的是精简不变量，已在下方 `:76-92`：`forbidden_text`（不得出现 `class RewardInferenceArtifact/Request/Result`、`VideoRewardArtifactStore`）、`model_specific`（不得泄漏 `KlingTeam / VideoVLMRewardInference / huggingface_hub`）、以及 4 条 not-exists（`ray.py` / `inference` / `video_inference` / 任意 `spec.py`）。`rewards/ray/` 无上游注册表来源 → 用 subset 守必需文件，不穷举。

### 1.3 `rewards/` 根 + `rewards/functions/` —— 两处穷举快照（severity medium）

`test_reward_function_implementations_live_under_functions`（`:128-149`）一次断言两个目录：

```python
# :131-138  rewards 根
assert _module_filenames(rewards_root) == {
    "__init__.py", "artifacts.py", "base.py",
    "inference.py", "runtime.py", "types.py",
}
# :139-149  rewards/functions
assert _module_filenames(rewards_root / "functions") == {
    "__init__.py", "aesthetic.py", "geneval.py", "kling_video_reward.py",
    "nsfw_safety.py", "ocr.py", "pickscore.py", "registry.py",
    "videocon_physics.py",
}
```

实地 `ls` 当前均匹配（rewards 根：`artifacts.py base.py inference.py __init__.py runtime.py types.py`；functions：上述 7 个 reward + `__init__.py registry.py`），但同 §1.1：`functions/` 下的 reward 模块就是 `_REWARD_REGISTRY` key 的镜像（identity 映射 + 脚手架 `__init__.py` / `registry.py`），应派生；`rewards/` 根是顶层脚手架目录，无注册表来源，应 subset。

### 1.4 `generation/ray/` —— 同款穷举快照（severity medium）

`test_generation_ray_adapter_stays_lean`（`:152-179`）：

```python
# :155-165
assert _module_filenames(ray_root) == {
    "__init__.py", "config.py", "executor.py", "launcher.py",
    "runtime.py", "pipeline_runner.py", "stage_worker.py",
    "weight_sync.py", "worker.py",
}
```

实地 `ls` 当前匹配，但同 §1.2：真实意图是下方 `:176-179` 的 forbidden-import（adapter 不得 `from vrl.generation.execution.planner import build_engine_plan`、不得 `from vrl.generation.execution.chunks import ...`），那才是真边界。无上游来源 → subset 守必需文件，不穷举。

> 对照组（**不动**）：同文件 `test_generation_execution_core_stays_flat_and_ray_neutral`（`:182-203`）已经是**正确范式** —— 它用 `for expected in (...): assert (execution_root / expected).exists()`（presence）+ `for ... : assert not (... ).exists()`（not-exists），**从不**穷举 `==`。本 sprint 就是把那四处对齐到这个已存在的范式。

## 2. 落地方案

仅改 `tests/architecture/test_generation_rollout_boundaries.py`。先加一个从注册表派生 reward 模块名的小辅助（紧挨现有 `_module_filenames` 放，复用其语义）。

### 通用：reward 模块名从 `_REWARD_REGISTRY` 派生

在文件辅助区新增：

```python
def _registered_reward_modules() -> set[str]:
    """Reward-impl filenames derived from the registry, the single source of truth.

    Each registered reward ``<name>`` owns a ``<name>.py`` module, so the
    expected module set is the registry keys — never a hand-typed ``ls``.
    Registration is lazy (``_register_builtins`` runs inside ``from_dict``),
    so trigger it once with an empty score dict before reading the keys.
    """
    from vrl.rewards.functions.registry import MultiReward, _REWARD_REGISTRY

    MultiReward.from_dict({}, device="cpu")  # populate _REWARD_REGISTRY
    return {f"{name}.py" for name in _REWARD_REGISTRY}
```

### A. `rewards/models/`（§1.1，修 LIVE FAILURE）

BEFORE（`:98-108`）：穷举 `==`。AFTER —— 派生 + 脚手架白名单 + 保留 misplacement 守卫：

```python
def test_reward_models_live_under_models() -> None:
    """Checks reward models live under models."""
    models_root = VRL_ROOT / "rewards" / "models"
    present = _module_filenames(models_root)
    # Every registered reward has a model module here (registry is the source).
    assert _registered_reward_modules() <= present
    # Only scaffolding may live alongside the per-reward modules.
    scaffolding = {"__init__.py", "base.py", "hub.py"}
    extras = present - _registered_reward_modules() - scaffolding
    assert not extras, f"unexpected modules under rewards/models/: {extras}"
    # Misplacement guards (unchanged — the real intent).
    assert not (VRL_ROOT / "rewards" / "kling_video_reward.py").exists()
    assert not (VRL_ROOT / "rewards" / "ray" / "kling_video_reward.py").exists()
    assert not (VRL_ROOT / "rewards" / "scorers").exists()
```

`hub.py` 落在 `scaffolding` 白名单（它是 reward-model 下载/缓存基础设施，非某个 reward 的实现）。注：保留一个收紧的 `extras` 检查（而非纯 subset），这样「把一个无关文件丢进 `models/`」仍会被抓到 —— 但新增一个**注册过的** reward 模块会自动通过（派生集随注册表长）。这就是 subset 与穷举 `==` 的关键差别：增量合法、misplacement 仍违法。

### B. `rewards/ray/`（§1.2）

BEFORE（`:70-75`）：穷举 `==`。AFTER —— 删 `==`，必需文件降级为 subset，forbidden-text / not-exists **原样保留**：

```python
def test_reward_ray_adapter_stays_lean() -> None:
    """Checks reward Ray adapter stays lean."""
    ray_root = VRL_ROOT / "rewards" / "ray"
    assert {"__init__.py", "model.py", "runtime.py", "worker.py"} <= _module_filenames(ray_root)
    forbidden_text = (...)        # :76-81 unchanged
    model_specific = (...)        # :82 unchanged
    for path in _python_files(ray_root):   # :83-88 unchanged
        ...
    assert not (VRL_ROOT / "rewards" / "ray.py").exists()        # :89-92 unchanged
    assert not (VRL_ROOT / "rewards" / "inference").exists()
    assert not (VRL_ROOT / "rewards" / "video_inference").exists()
    assert not list((VRL_ROOT / "rewards").rglob("spec.py"))
```

### C. `rewards/` 根 + `rewards/functions/`（§1.3）

BEFORE（`:131-149`）：两处穷举 `==`。AFTER —— 根用 subset，functions 用派生 + 脚手架：

```python
def test_reward_function_implementations_live_under_functions() -> None:
    """Checks reward function implementations live under functions."""
    rewards_root = VRL_ROOT / "rewards"
    required_root = {"__init__.py", "artifacts.py", "base.py", "inference.py", "runtime.py", "types.py"}
    assert required_root <= _module_filenames(rewards_root)

    functions = _module_filenames(rewards_root / "functions")
    assert _registered_reward_modules() <= functions
    scaffolding = {"__init__.py", "base.py", "registry.py"}
    extras = functions - _registered_reward_modules() - scaffolding
    assert not extras, f"unexpected modules under rewards/functions/: {extras}"
```

（`base.py` 是否存在于 `functions/` 实地为否 —— subset 容许其缺席，白名单只是「若出现则允许」，不会误伤。）

### D. `generation/ray/`（§1.4）

BEFORE（`:155-165`）：穷举 `==`。AFTER —— 删 `==`，必需文件降级为 subset，forbidden-import 块 **原样保留**：

```python
def test_generation_ray_adapter_stays_lean() -> None:
    """Checks generation Ray adapter stays lean."""
    ray_root = VRL_ROOT / "generation" / "ray"
    required = {
        "__init__.py", "config.py", "executor.py", "launcher.py", "runtime.py",
        "pipeline_runner.py", "stage_worker.py", "weight_sync.py", "worker.py",
    }
    assert required <= _module_filenames(ray_root)
    ray_adapter_files = (...)      # :166-175 unchanged
    for path in ray_adapter_files:  # :176-179 unchanged — the real boundary
        text = path.read_text(encoding="utf-8")
        assert "vrl.generation.execution.planner import build_engine_plan" not in text
        assert "vrl.generation.execution.chunks import" not in text
```

## 3. 验证（finishing criteria）

- `grep -nE "_module_filenames\([a-z_ /]+\) ==" tests/architecture/test_generation_rollout_boundaries.py` **零命中**（确认四处穷举 `==` 全部消失）。
- `grep -n "_registered_reward_modules\|<= _module_filenames\|required <=\|required_root <=" tests/architecture/test_generation_rollout_boundaries.py` 命中新增的派生/subset 断言。
- `python -m pytest tests/architecture/test_generation_rollout_boundaries.py -q` 全绿 —— 重点 `test_reward_models_live_under_models` 从「红（Extra items: hub.py）」转绿，且**未**在任何全集里手抄 `hub.py`（它走脚手架白名单）。
- 反向验证派生有效：临时在 `vrl/rewards/functions/registry.py` 注册一个假 reward（或临时往 `models/` 丢一个无关 `_scratch.py`）——前者应让 `test_reward_models_live_under_models` 因「缺对应模块」而红，后者应因 `extras` 而红，证明 subset 仍能抓真问题（验证后回滚临时改动）。
- `python -m pytest tests/architecture/ tests/rewards/ -q` 零回归。
- 解释器：本机用 `/home/mingfeiguo/miniconda3/bin/python`（仓库无 `.venv`）。

## 4. 非目标 / Non-Goals

- **不改任何 `vrl/` 生产代码** —— `hub.py` 落在 `rewards/models/` 是正确的，问题在测试侧的穷举断言，不在源码。
- **不动 forbidden-text / forbidden-import / not-exists 断言** —— 那些才是真边界，原样保留（§2 各处明确标 unchanged）。
- **不把 subset 放宽成纯 presence-without-extras-check** —— `rewards/models/` 与 `rewards/functions/` 仍保留收紧的 `extras` 守卫，确保「misplaced 无关文件」仍违法；只让「注册过的新 reward 模块」与「白名单脚手架」合法通过。
- **不递归审计本文件之外的其它测试** —— 本 sprint 仅收口这一个文件的 4 处目录清单快照。其余同类 frozen-snapshot（config Literal 镜像、registry alias 镜像、protocol 方法元组等）属各自 sprint 的范围，见 References 关联条目。
- **不为 `rewards/ray/` 与 `generation/ray/` 引入注册表派生** —— 它们是 adapter 脚手架目录，无 vocabulary 上游来源，subset 即是正确范式。

## References

- `tests/architecture/test_generation_rollout_boundaries.py:70-75,98-108,131-149,155-165`（四处穷举 `==`，本 sprint 改写）
- `tests/architecture/test_generation_rollout_boundaries.py:76-92,109-111,176-179,254-255`（保留的真边界断言 + `_module_filenames` 辅助）
- `tests/architecture/test_generation_rollout_boundaries.py:182-203`（已正确的 presence/not-exists 范式，对齐目标）
- `vrl/rewards/functions/registry.py:16,26-43,98`（`_REWARD_REGISTRY` 单一来源；`_register_builtins` 懒注册触发点）
- `vrl/rollouts/families/registry.py:361-364`（`registered_rollout_families() == tuple(FAMILY_REGISTRY)`：注册表即来源的范式，本反模式的根因对照）
- 实地目录：`vrl/rewards/models/`（多出 `hub.py`，触发 LIVE FAILURE）、`vrl/rewards/ray/`、`vrl/rewards/`、`vrl/rewards/functions/`、`vrl/generation/ray/`
- 关联（同源 frozen-snapshot / hardcoded-entity-list，属各自 sprint）：[[SPRINT_test_config-literal-mirrors]]（config schema 的 `Literal` 手抄副本，应 `typing.get_args` 派生）、[[SPRINT_test_registry-mirror-assertions]]（family alias / task / return_artifacts 手抄副本，应从 `FAMILY_REGISTRY` 派生）、[[SPRINT_test_protocol-method-tuples]]（`ReplayModel/RuntimeModel.__protocol_attrs__` 手抄副本）
