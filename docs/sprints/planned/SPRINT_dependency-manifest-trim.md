# SPRINT: 删除不可达的 Prometheus 与 RapidOCR 依赖声明（planned）

状态：**planned / CPU-only**。执行前以当时 HEAD 重取 `pyproject.toml` 与 `uv.lock` 基线。

## 目标

依赖清单应描述代码真实 import 的能力。本 sprint 只删除两条已失去生产消费者的声明：

| 目标 | 证据 | 连带动作 |
|---|---|---|
| core `prometheus-client` | `vrl/` 与 `tests/` 零 Python importer；旧 metrics endpoint 已删除 | 重生成 lock |
| `[ocr]` 的 `rapidocr-onnxruntime` | OCR runtime 只构造 `PaddleOCR`；当前真引擎测试却错误地用 RapidOCR 作 skip gate | 改 gate/测试名、修 preset 注释、重生成 lock |

这两项共用 manifest 与 lock，作为一个原子 dependency commit 落地。

## 改动

1. 从 `pyproject.toml` 删除直接依赖 `prometheus-client>=0.20.0`。
2. 从 `[project.optional-dependencies].ocr` 删除 `rapidocr-onnxruntime>=1.3.0`。
3. 在 `tests/rewards/functions/test_ocr.py`：
   - 删除 `_has_rapidocr` / `_skip_no_rapidocr`；
   - 真引擎用例内部用 `pytest.importorskip("paddleocr")` 钉住实际 dependency；
   - `rapidocr` 测试名与 docstring 改为 `paddleocr`。
4. 把 `vrl/config/presets/experiment/janus_pro/online_grpo_ocr.yaml` 的过期 `rapidocr` 注释改为 `paddleocr`。
5. 运行 `uv lock`，提交生成后的 `uv.lock`。

`pytest.importorskip` 是 test adapter，不新增生产 helper；这里无需为一处 gate 保留独立薄函数。

## 保持不变

- **不删 `pose` / `pose-gpu`。** 仓库仍保留 RTMW anime-anatomy probe，README 也把这两个 extra 当安装边界。当前问题是 extra 没声明 `rtmlib`，而 `rtmlib` 又会拉 CPU `onnxruntime` 与 GPU runtime 冲突；这需要单独设计可安装的 CPU/GPU contract，不能靠删 CPU extra 掩盖。
- 不删 `[ocr]` 的 `paddleocr`、`paddlepaddle`、`python-Levenshtein`；生产代码逐项使用。
- 不删真 OCR integration test；只修正它检查的 dependency。
- 不改 `vrl/rollouts/stats.py` 对未来 metrics sink 的架构注释。未来真的实现 sink 时再诚实加回 dependency。
- 不删 lock 中由 vLLM 间接带入的 Prometheus packages。
- 不改 `reading/` 中的历史 RapidOCR 叙述。

本簇没有需要迁移的 ALL_CAPS 业务表，也不借依赖清理重构 reward/runtime。

## 验收

```bash
uv lock
uv lock --check

rg -n 'rapidocr' vrl tests
# expected: no matches

rg -n '^[[:space:]]*(from|import)[[:space:]]+prometheus' vrl tests
# expected: no matches

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  tests/rewards/functions/test_ocr.py tests/config -q
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m vrl.config.lint

.venv/bin/ruff check --fix tests/rewards/functions/test_ocr.py
.venv/bin/ruff format tests/rewards/functions/test_ocr.py
.venv/bin/ruff check tests/rewards/functions/test_ocr.py
.venv/bin/ruff format --check tests/rewards/functions/test_ocr.py
```

检查最终 diff：`uv.lock` 只能反映上述两条直接声明的移除，不接受无关依赖升级。

## References

- `pyproject.toml`
- `uv.lock`
- `tests/rewards/functions/test_ocr.py`
- `vrl/rewards/models/ocr.py`
- `vrl/config/presets/experiment/janus_pro/online_grpo_ocr.yaml`
- `vrl/scripts/eval/probe_anime_anatomy_report.py`
- `vrl/scripts/eval/anime_probe_common.py`
