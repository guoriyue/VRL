# SPRINT(auto): vrl/trainers/online/trainer.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/trainers/online/trainer.py` (899 LOC)
角色判定: core
结论: improve

## 0. 一句话
这是真正的核心训练循环，但 `_step_impl` 单方法塞进了大量一次性 grad-split / first-step 调试脚手架（含写 `/tmp` 文件、重复的 print+stderr+file 三连），把 long-term 训练逻辑和 one-shot probe 混在了同一条 import graph 里。

## 1. 现状（读代码得出）
- 核心管线清晰：`step → _step_impl`，collect→advantage→filter→train loop→ema/step（trainer.py:265-796），委托 `rollout_schedule` 做 collect/offload/sync（trainer.py:190-199, 295, 779）。这部分是 core，keep。
- 但 `_step_impl` 内嵌两块调试脚手架：
  - `cfg.debug.first_step` 块（trainer.py:467-511）做 log-prob parity 诊断。
  - `cfg.debug.grad_split` 块（trainer.py:513-530 + 601-680）反复出现同一模式：
    ```python
    print(_msg, file=sys.stderr, flush=True); print(_msg, flush=True); logger.info(...)
    try:
        with open("/tmp/grad_split_debug.log", "a") as _f: _f.write(_msg)
    except Exception: pass
    ```
    并用类属性单发标志 `OnlineTrainer._grad_split_already_fired`（trainer.py:605-611）。

## 2. 质疑点 / 改进机会
- one-shot probe 混入长期代码（AGENTS.md 规则 4）：grad-split 诊断写死 `/tmp/grad_split_debug.log`（trainer.py:527, 622, 675），是典型一次性 spike 产物，却长期留在主训练方法体内，没有 `*_probe` 隔离。
- 重复样板（规则 5 职责过载）：同一段 "print stderr + print stdout + logger.info + try-open-/tmp-append" 在文件里出现三次（trainer.py:516-530, 614-625, 661-678），应抽成一个本地诊断 helper 或独立模块 `trainer_debug.py`。
- 在热循环里用 `torch.autograd.grad(..., retain_graph=True)` 跑两次额外反向（trainer.py:633-659）即使有单发标志也带可观开销，应明确隔离到独立诊断路径，避免阅读核心 step 时被淹没。
- 命名/职责：`_step_impl` 既是训练核心又是诊断宿主，899 LOC 单文件偏大，可把 PhaseTimer + autocast + optimizer factory 这些独立 util 拆到 `trainer_support.py`，让 trainer.py 聚焦 CEA 管线。

## 3. 建议动作
- 把 grad-split 与 first-step parity 诊断抽到独立模块（如 `vrl/trainers/online/debug_probes.py`），在 `_step_impl` 里只保留一行 `if cfg.debug.grad_split: _emit_grad_split_probe(...)` 的调用点。
- 消除三处重复的 print/stderr/file 三连，统一为一个 `_dump_probe(msg, path)` helper；去掉硬编码 `/tmp` 路径，改用 `cfg.output_dir`（与 phase_events.jsonl trainer.py:738 一致）。
- 可选：把 `_create_optimizer` / `PhaseTimer` / `_get_autocast` 移到 support 模块，trainer.py 收缩到只剩管线。

## 4. 不动什么 / 为什么不是过度清理
- CEA 主管线（advantage 计算、zero-advantage 过滤、grad accumulation、ema.step、rollout_schedule 委托）是 justified 的核心逻辑，不重写，不"简化"。
- 不动 `TrainStepMetrics` 字段集合与早退分支（trainer.py:409-433）——这是真实业务语义。
- `_validate_ema_state_shapes`（trainer.py:867-899）是 load_state_dict 的正当校验逻辑，保留（只是不该被 package `__all__` 当 public 导出，见 __init__ sprint）。

## 5. 验证
- `pytest tests/trainers/test_online.py`（13 个 OnlineTrainer 用例）全绿。
- `grep -rn "/tmp/grad_split_debug.log" vrl/` 抽离后应为空或仅在 debug_probes 内。
- `ruff check vrl/trainers/online/trainer.py` 无新增告警；行数应明显下降。
