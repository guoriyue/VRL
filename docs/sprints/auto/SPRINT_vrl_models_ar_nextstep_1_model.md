# SPRINT(auto): vrl/models/ar/nextstep_1/model.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/ar/nextstep_1/model.py` (573 LOC)
角色判定: core
结论: improve

## 0. 一句话
这是 NextStep-1 的核心模型 wrapper（真核心），但模块 docstring 仍把整个文件描述成"每个真实调用都标了 `# TODO(nextstep-binding)` 的 scaffolding"，与实际代码已严重不符；同时上游 repo 的 `sys.path` 引导逻辑被复制了两份。

## 1. 现状（读代码得出）
文件被 `vrl/rollouts/families/registry.py:278` 和 `vrl/scripts/ar/nextstep_1/train.py` 引用，是 production import graph 的一部分，提供 `NextStep1Model` / `NextStep1ReplayModel` / `NextStep1Config`。
`recompute_logprobs`、`replay_forward`、`decode_image_tokens`、`disable_adapter`、`_init_kv`、`_step_llm` 都已是完整可运行实现（调用 `flow_logprob_at` / `flow_sample_with_logprob`，二者均存在于 `vrl/math/ar/flow_matching.py`）。

模块 docstring 的 "UPSTREAM BINDING" 段（model.py:18-30）写：

```
This module is a *scaffolding* — every real call into the upstream
NextStep-1 package is marked ``# TODO(nextstep-binding)``. Once you've
done ``pip install -e .`` ... fill in:
    - ``_load_pipeline``     : ...
    - ``_run_llm_step``      : ...
    - ``_image_in_projector``: ...
    - ``_decode_via_vae``    : ...
```

但实测 `grep "TODO(nextstep-binding)"` 全文件只剩 **1 处** 真实代码级 TODO（model.py:425，在 `_init_kv` 内说明 KV-cache 类型），其余 docstring 里点名的 `_load_pipeline`/`_image_in_projector`/`_decode_via_vae`（实际叫 `decode_image_tokens`）都已实现，`_run_llm_step` 这个名字根本不存在（真实是 `_step_llm`）。

## 2. 质疑点 / 改进机会
1. **stale scaffolding 描述（one-shot vs long-term 规则）**：docstring 把长期资产说成一次性 scaffolding，且点名的函数名（`_run_llm_step`/`_decode_via_vae`）与实现不符（model.py:18-30 vs 实际 `_step_llm`/`decode_image_tokens`）。这会让读者误判文件成熟度，是 AGENTS.md「问题已记录却描述与现实脱节」的命名/状态腐烂。
2. **重复的上游路径引导**：`_load_pipeline`（model.py:168-187）与 `_load_nextstep_replay_model`（model.py:539-556）逐行复制了同一段 `import nextstep` → 定位 `repo_root` → 把 `inference/` 插入 `sys.path` → ImportError 文案。两份完全一致，源类型变动时会双份腐烂。应抽成一个 `_ensure_nextstep_on_path()` helper（一次性副作用 + 返回 inference_dir），二处共用。
3. （非问题，记录）`NEXTSTEP_DEFAULT_TOKEN_NUM/DIM/PIXEL_SIZE`（model.py:54-56）是真实模型架构维度（32x32 patch grid、f8ch16 通道、256 像素），属 AGENTS.md 允许保留的「模型架构维度」边界，且都被 `NextStep1Config` 字段默认值引用，不是手抄的 typed 结构副本 → 保留。

## 3. 建议动作
- 重写模块 docstring 的 "UPSTREAM BINDING" 段：删除"every real call is marked TODO / scaffolding"措辞，只保留真实剩余的 1 个 binding 风险点（`_init_kv` 的 KV-cache 类型依赖 HF Qwen2 的 `DynamicCache`，见 model.py:425），并修正函数名（`_step_llm`、`decode_image_tokens`）。
- 抽出 `_ensure_nextstep_on_path() -> str`（返回 inference_dir），`_load_pipeline` 与 `_load_nextstep_replay_model` 共用，消除 model.py:168-187 与 539-556 的重复。
- 不要为此重排类结构或改公共 API。

## 4. 不动什么 / 为什么不是过度清理
- `NextStep1Model` / `NextStep1ReplayModel` 的双类形状刻意对齐 `janus_pro`（generation bundle vs minimal replay bundle 两条路径，registry + train.py 各取所需），属跨家族一致性，不动。
- `recompute_logprobs` 与 `replay_forward` 看似职责接近但契约不同（前者裸张量入口、后者 `ReplayRequest`/`TrajectoryResolver` 入口），是 ReplayModel 协议边界，不要合并。
- ALL_CAPS 维度常量保留（见 2.3）。

## 5. 验证
- `ruff check vrl/models/ar/nextstep_1/model.py`。
- `grep -n "TODO(nextstep-binding)" vrl/models/ar/nextstep_1/model.py` 应只剩 `_init_kv` 一处。
- `grep -n "inference_dir" vrl/models/ar/nextstep_1/model.py` 应只在新 helper 内出现一次定义。
- 跑 nextstep_1 family 已有的单测（若有）或 `python -c "import vrl.models.ar.nextstep_1.model"` 确认导入无回归。
