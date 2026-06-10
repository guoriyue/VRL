# SPRINT: Unified logging — kill per-module `_emit_*` hacks（vLLM-style init_logger）

状态：implemented（2026-06-09）。

落地记录：
- T1 ✅ `vrl/utils/logging.py`：`init_logger`（幂等装一个 stdout handler 于 "vrl" 命名空间，
  `propagate=False` 防双打印，`VRL_LOG_LEVEL` 控级别）+ `kv`（key=value，float `.3f`）。
  handler 用 emit 时解析 `sys.stdout` 的子类（跟随重定向；pytest capsys 可捕获）。
  测试 `tests/test_logging.py`（3 用例）。
- T2 ✅ 删除 `_emit_worker_log`（worker.py，3 调用点）与 `_emit_kling_load_log`
  （kling_video_reward.py，7 调用点），全部改 `logger.info("... %s", kv(...))`，print 移除。
- T3 ✅ 13 个 actor 常驻模块 `logging.getLogger` → `init_logger`（generation worker、
  全部 model family runtime/model、videocon、utils/profiling、utils/memory）。
  driver 侧（scripts/trainers/launcher/placement/producer）留待后续批次。
- **caplog 兼容**：`propagate=False` 会挡住 pytest caplog（靠向 root 传播）。
  `tests/conftest.py` 加 autouse fixture，测试期间临时恢复传播、teardown 还原。
- 验收：裸进程（等价 Ray worker）`logger.info` 直接可见；老写法对照确认被静默丢弃；
  非 scripts 目录 `print(flush=True)` 归零；全量 695 passed + ruff clean。

## 0. 一句话

Ray worker 进程里 `logger.info` 被静默丢弃（没人配置 handler），于是各模块各自长出
「`logger.info` + `print(flush=True)` 双发 + 手拼 key=value」的私有日志函数。
修根因：照 vLLM 的 `init_logger` 模式建一个 `vrl/utils/logging.py`，删掉所有私有 emit。

## 1. 根因与症状（代码证据）

- **根因**：`logging.basicConfig` 只在 driver 入口配置（`vrl/scripts/train.py:76`、
  `cosmos/anima/generate.py:124`）。Ray actor 是独立进程，从未配置 → Python 对无 handler
  的 INFO 记录直接丢弃 → worker 里 `logger.info` 失踪。
- **症状 1**：`vrl/rewards/ray/worker.py:127` `_emit_worker_log` —— `logger.info` + `print(flush=True)`
  双发（print 借 Ray 的 stdout 转发保证可见），手拼 `key=value`，float `.3f`。
- **症状 2**：`vrl/rewards/models/kling_video_reward.py:784` `_emit_kling_load_log` ——
  同一段逻辑的复制粘贴（差异仅 float 格式化）。
- **附带 bug**：双发意味着日志一旦被配置（单测 caplog、或 driver 端），同一条消息打两遍。
- **不是同类病**：`generation/ray/placement.py:178` `_log_placement` 在 driver 跑、标准
  logger、真实多行领域格式化 —— **保留**。

## 2. 设计（prior art: vLLM `vllm/logger.py`）

新建 `vrl/utils/logging.py`（与 utils/config.py、utils/profiling.py 同层），只两个公开函数：

```python
def init_logger(name: str) -> logging.Logger:
    """vLLM-style: return a namespaced logger; idempotently install ONE
    stdout StreamHandler + formatter on the "vrl" root logger
    (propagate=False). Safe to call at import time in every module —
    works identically in driver and Ray worker processes."""

def kv(**fields: Any) -> str:
    """Shared key=value formatter (floats rendered .3f). Composes with
    stdlib levels instead of wrapping the logger API."""
```

要点：
- handler 装在 `"vrl"` 命名空间 logger 上、`propagate=False`——driver 的 `basicConfig`
  root handler 不会再次打印（消除双发）；幂等守卫防止同进程多次 init 重复装 handler。
- 输出到 **stdout**（与现 print 行为一致，Ray 按 actor 名前缀转发）。
- 级别走环境变量 `VRL_LOG_LEVEL`（默认 INFO），仿 vLLM 的 `VLLM_LOGGING_LEVEL`。
- 不引入 loguru/structlog（额外依赖；vLLM/SGLang 都用 stdlib 就够）。
- 不建 logger 门面类/Protocol（模块依赖的就是 stdlib `Logger` 接口，已是抽象）。

调用点形态（before → after）：

```python
# before (kling_video_reward.py)
_emit_kling_load_log("resolving Kling VideoReward root",
                     reward_model_name=..., model_path=..., local_files_only=...)

# after
logger = init_logger(__name__)          # 模块顶部，替代 logging.getLogger
logger.info("resolving Kling VideoReward root %s",
            kv(reward_model_name=..., model_path=..., local_files_only=...))
```

SOLID 对照：SRP（格式化/输出/身份分离，业务模块不再自带管道）；OCP（要 JSON/文件/远端
sink 时只改 handler 安装处，调用点零改动）；DIP（模块依赖 stdlib Logger，不依赖私有 emit）。

## 3. 分步实施

### T1 `vrl/utils/logging.py` + 单测
- `init_logger` 幂等性测试（同进程多次调用只装一个 handler）、`kv` float `.3f`、
  caplog 下不双发。

### T2 收口两个症状模块
- `rewards/ray/worker.py`、`rewards/models/kling_video_reward.py`：删 `_emit_worker_log` /
  `_emit_kling_load_log`，调用点改 `logger.info("... %s", kv(...))`，删 print。
- **注意**：这两个文件当前有另一会话的在途改动（reward 放置重构），落地前先确认已合入。

### T3 Ray-actor 常驻模块统一换 `init_logger`
- `generation/execution/worker.py`、`vrl/ray/runtime.py` 等 actor 进程模块的
  `logging.getLogger(__name__)` → `init_logger(__name__)`（机械替换，25 处 getLogger
  里只动 actor 侧；driver 侧随后批次统一，保持跨家族一致 shape）。
- driver 入口的 `basicConfig` 保留（它还服务第三方库的 WARNING）。

## 4. 验收

- Ray worker 里 `logger.info` 不再需要 print 即可在 driver 控制台看到（跑一次
  kling reward smoke 对照启动日志）。
- 单测 caplog 同一消息只出现一次。
- `grep -rn "print(.*flush=True" vrl/ --include="*.py"` 在非 scripts 目录归零。

## 5. Non-goals

- 不引入第三方日志库、不做 JSON/结构化 sink（OCP 留好了位置，需要时一处改）。
- 不动 `_log_placement`（driver 侧合法的领域格式化）。
- 不动 `vrl/scripts/` 下 CLI 工具的 print（那是用户输出，不是日志）。
- 不做日志采样/限流/分级文件轮转（YAGNI）。

## 6. 参考

- `vrl/rewards/ray/worker.py:127`、`vrl/rewards/models/kling_video_reward.py:784`（待删）
- `vrl/scripts/train.py:76`（driver basicConfig）
- vLLM logger：https://github.com/vllm-project/vllm/blob/main/vllm/logger.py（`init_logger`、env 控级别）
