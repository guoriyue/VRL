# SPRINT: 退役有混淆变量的 Kling 探针（planned）

状态：**DONE（2026-07-28，`b205085a`）**。净删除 `107` 行、`1` 个 Python 文件，按计划落地。

> **执行偏差**：本文验收里的 `ruff format` 步骤**未执行**。`kling_reward_diagnosis_probe.py`
> 在 HEAD 时就已 format-dirty（用 `git show HEAD:` 版本过 `ruff format --check` 验证），
> 格式化会把 4 行 docstring 改动放大成 ~78 行无关重排，违反 AGENTS.md 的 diff 纪律。
> `ruff check` 通过；该文件的既有格式债留给单独的 formatting-only pass。
> 另：本文验收的 grep 覆盖 `docs/sprints/planned`，而本文自身当时就在那里并 4 次点名被删文件，
> 因此该 grep 在文档归档到 `done/` 前不可能干净。

## 目标

删除 `vrl/scripts/eval/kling_480p_discrimination_probe.py`。它把噪声、帧乱序、丢帧和 FPS 变化同时施加到视频，无法判断哪个变量导致 reward 变化。后继
`kling_reward_diagnosis_probe.py` 已把实验改成保持帧序与 FPS 的单轴 noise ladder，并产出最终结论：VQ 方向正确，MQ 反向，`overall_reward` 被错误加权。

这符合 one-shot 生命周期：初版问题已由更严格实验取代，结果已记录在 run 文档，保留脚本只会鼓励重跑一个已知有混淆变量的实验。

## 改动

1. 删除 `vrl/scripts/eval/kling_480p_discrimination_probe.py`。
2. 更新 `vrl/scripts/eval/kling_reward_diagnosis_probe.py` 的开头：保留“先前 mixed-degradation 实验有混淆”的 provenance，不再引用已删除文件名。
3. 更新：
   - `docs/runs/README.md`
   - `docs/runs/cosmos_predict25_nft_kling_480p33f_rbs16_20260620/README.md`

   两处都保留初版实验结果，但只把可重跑入口指向 `kling_reward_diagnosis_probe.py`。
4. `done/` 中的历史审计引用保持原样；它们记录删除前现场，不是活入口。

## 保持不变

- **保留 `inductor_cache_recompile_probe.py`。** 它有 cold/warm/control 三臂，可在 PyTorch、CUDA 或 GPU 变化后重跑；`SPRINT_compile_rollout_lifecycle.md` 也把它列为长期测量脚手架。文件自称 one-shot 不足以推翻实际可复用边界。
- **保留 `wan_i2v_base_sample.py`。** 它绕过 VRL family wrapper，直接驱动 upstream diffusers，是定位“上游 checkpoint 还是本仓包装层”问题的独立 adapter。生产 family parity probe 不能替代这条边界。
- 保留 `kling_reward_diagnosis_probe.py`；它是更严格的当前诊断入口。
- 不 sweep `vrl/scripts/perf` 或 `vrl/scripts/eval` 的其他 probe。

被删文件内 `_EVAL_NOISE` 与 `_PROMPT` 是该一次性实验的 fixture constants，随实验一起删除；不借此质疑其他脚本中作为协议、fixture 或模型维度边界的 ALL_CAPS 常量。

## 验收

```bash
rg -n 'kling_480p_discrimination_probe' \
  vrl tests README.md docs/runs docs/sprints/planned docs/sprints/parked
# expected: no matches

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  tests/rewards/kling_video_reward -q

.venv/bin/ruff check --fix vrl/scripts/eval/kling_reward_diagnosis_probe.py
.venv/bin/ruff format vrl/scripts/eval/kling_reward_diagnosis_probe.py
.venv/bin/ruff check vrl/scripts/eval/kling_reward_diagnosis_probe.py
.venv/bin/ruff format --check vrl/scripts/eval/kling_reward_diagnosis_probe.py
```

不需要 GPU，不重跑 Kling 模型。

## References

- `vrl/scripts/eval/kling_480p_discrimination_probe.py`
- `vrl/scripts/eval/kling_reward_diagnosis_probe.py`
- `docs/runs/README.md`
- `docs/runs/cosmos_predict25_nft_kling_480p33f_rbs16_20260620/README.md`
- `docs/sprints/done/SPRINT_compile_rollout_lifecycle.md`
- `vrl/scripts/perf/inductor_cache_recompile_probe.py`
- `vrl/scripts/eval/wan_i2v_base_sample.py`
