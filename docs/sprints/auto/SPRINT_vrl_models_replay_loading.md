# SPRINT(auto): vrl/models/replay_loading.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/replay_loading.py` (361 LOC)
角色判定: helper
结论: improve

## 0. 一句话
这个文件被各家族 runtime 大量复用（核心价值真实存在），但它把两条互不相关的关注点塞在一个 module 里：一是 "replay runtime-role 元数据 taxonomy + 校验"，二是 "具体的 diffusers/PEFT/torch 加载动作"；同时有三个公开符号只被测试触达、没有任何 production caller，需拆分并澄清这些声明的实际归属。

## 1. 现状（读代码得出）
文件名/docstring 声明它是 "Runtime metadata helpers for replay-only model loading"，但实际包含两组完全不同的东西。

第一组：runtime-role 元数据 taxonomy + 校验。
```python
RUNTIME_ROLE_KEY = "runtime_role"
LOADS_FULL_GENERATION_MODULES_KEY = "loads_full_generation_modules"
...
@dataclass(frozen=True, slots=True)
class ReplayModuleLoadingProfile:   # line 27
    ...
def full_generation_bundle_metadata(...): ...   # line 81
def minimal_replay_bundle_metadata(...): ...     # line 98
def module_loading_profile_from_metadata(...): ...  # line 114
def require_minimal_replay_bundle(...): ...      # line 144
def bundle_loads_full_generation_modules(...): ...  # line 135
```

第二组：和元数据毫无关系的具体后端加载/改写动作（diffusers / peft / torch）。
```python
def load_diffusers_transformer_component(...):  # line 158, `import diffusers`
def load_diffusers_scheduler_component(...):     # line 181
def load_flow_match_scheduler_component(...):    # line 207
def apply_lora_to_transformer(...):              # line 221, `from peft import ...`
def enable_transformer_full_finetune(...):       # line 253
def compile_transformer(...):                    # line 262, `import torch`
def resolve_torch_dtype(...):                    # line 272
```

引用情况（grep 实测，排除自身）：
- 重度复用、真核心：`compile_transformer` 20、`resolve_torch_dtype` 10、`load_diffusers_transformer_component` 8、`apply_lora_to_transformer` 6、`enable_transformer_full_finetune` 6、`load_diffusers_scheduler_component` 4、`load_flow_match_scheduler_component` 4、`full_generation_bundle_metadata` 14、`minimal_replay_bundle_metadata` 14、`bundle_loads_full_generation_modules` 2（含 `vrl/utils/memory.py`）。
- 只被测试触达、production 0 引用：`module_loading_profile_from_metadata`、`require_minimal_replay_bundle`、`ReplayModuleLoadingProfile`。实测仅出现在 `tests/models/test_minimal_replay_runtime_wiring.py` 与 `tests/models/test_replay_loading.py`，无任何 `vrl/` 内 caller。
- 所有 `*_KEY` ALL_CAPS 常量 + `RuntimeRole` 别名：`vrl/` 内非自身引用 0（仅供本文件内部 + `__all__` 导出）。

## 2. 质疑点 / 改进机会
1. 职责过载 / 文件名误导（god-ish file）。`replay_loading.py` 这个名字 + docstring 说的是 "replay metadata"，但一半内容是 generic 后端加载/编译动作（`compile_transformer` / `resolve_torch_dtype` / `apply_lora_to_transformer` 等）。这些动作和 "replay role 元数据" 没有概念关系，被各 family runtime 当通用工具箱用。`interfaces/runtime.py:65` 的 docstring 还点名让大家 "Build these fields through `vrl.models.replay_loading`"——把元数据契约和加载动作绑在一个模块名下，概念边界模糊。证据：line 158-293 全是后端动作，line 27-156 全是元数据。
2. parser/validator 路径只有测试在用。`module_loading_profile_from_metadata`(line 114)、`require_minimal_replay_bundle`(line 144) 以及公开的 `ReplayModuleLoadingProfile`(line 27) 在 production 0 引用——只有 `full_generation_bundle_metadata`/`minimal_replay_bundle_metadata` 两个 factory（内部 new 了 profile 再 `as_metadata()`）被真正调用。也就是说：写入侧（factory）活跃，读取/校验侧（parser）目前只有测试断言、没有运行时消费者。这不是死代码（测试在固化契约），但说明 "minimal replay loader" 这条管线尚未接通，留着一组没有 runtime caller 的 public API + 一个 public dataclass，属于 "声明先行、未接线" 的待澄清状态，应记录清楚而不是默默留着。
3. ALL_CAPS keys 是合法边界但导出过宽。`RUNTIME_ROLE_KEY` 等是写进 `RuntimeBundle.metadata` 的 schema key（AGENTS.md 明确 schema key 是真边界，保留正确）。但它们在 `__all__` 里全量导出却无任何外部 import（实测 0），且 key 字面值与 `RuntimeRole` 的 Literal 值在同文件手维护两份（"runtime_role" 字符串 + Literal 值），无 drift 风险但有冗余导出面。

## 3. 建议动作
- 拆分模块（consolidate by responsibility）：已落地——通用模型装载动作（`load_diffusers_transformer`、`load_diffusers_scheduler`、`load_flow_match_scheduler`、`apply_lora_to_transformer`、`enable_transformer_full_finetune`、`compile_transformer`）迁到 `vrl/models/loader.py`（沿用 vLLM/SGLang 的 `loader` 命名，去掉无意义的 `_component` 后缀；`resolve_torch_dtype` 留在 `vrl/models/dtypes.py`），`replay_loading.py` 只留 replay-role 元数据 taxonomy + 校验。各 family runtime 的 import 随之分成两行，语义更清晰。
- 对 parser 路径表态：要么尽快把 `require_minimal_replay_bundle` / `module_loading_profile_from_metadata` 接进真正的 minimal-replay 加载流程（trainer 侧），要么在 docstring 明确标注 "contract declared, runtime consumer pending"，避免读者误以为已接线。不建议删——测试在固化契约，且 factory 写入侧已 production 使用，parser 是其对称读取面。
- `__all__` 不必收紧 key 导出（schema key 作为模块公共词汇可保留），但拆分后应把 key 跟着元数据模块走。

## 4. 不动什么 / 为什么不是过度清理
- 不要拍平 / 数据化那 7 个后端动作函数：它们是跨 7 个 family runtime 复用的共享抽象（compile 20 / dtype 10 / transformer 8 次），符合 AGENTS.md "移除真实复杂度的共享抽象" 与 "跨家族一致性优先" —— 保留为独立 helper，只是换个名副其实的家。
- `*_KEY` ALL_CAPS 常量保留：它们是写入 `RuntimeBundle.metadata` 的 schema key，属 AGENTS.md 明列的真边界，不是手抄 typed 结构。
- 不删任何 parser/factory：factory 已 production 使用，parser 有测试覆盖且是契约对称面，证据不足以判 delete。本 sprint 是 improve（拆分 + 表态），不是 delete。

## 5. 验证
- 拆分后跑 `pytest tests/models/test_replay_loading.py tests/models/test_minimal_replay_runtime_wiring.py -q` 必须全绿（这两个文件直接 import 受影响符号）。
- `grep -rn "from vrl.models.replay_loading import" --include=*.py vrl/ tests/` 核对所有 import 行已按新拆分更新，无残留指向旧路径。
- `ruff check vrl/models/` 无新增告警；`python -c "import vrl.models.replay_loading"` 与新模块均可导入。
