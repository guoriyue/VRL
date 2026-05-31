# SPRINT(auto): vrl/rewards/types.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rewards/types.py` (40 LOC)
角色判定: core
结论: improve

## 0. 一句话
`RewardTrajectoryStep` 这个 dataclass 从未被实例化，`RewardTrajectory.steps`/`seed` 在所有调用点都是死字段（恒为 `[]` 和 `0`），属于为不存在的需求预留的结构，应删掉收敛成实际在用的形状。

## 1. 现状（读代码得出）
本文件定义三个数据容器：`RewardTrajectoryStep`、`RewardTrajectory`、`RewardRollout`。

```python
@dataclass(slots=True)
class RewardTrajectoryStep:  # types.py:10
    timestep: int
    log_prob: Any
    noise_pred: Any
    new_log_prob: Any = None
    ref_log_prob: Any = None

@dataclass(slots=True)
class RewardTrajectory:  # types.py:21
    prompt: str
    seed: int
    steps: list[RewardTrajectoryStep]
    output: Any
```

`RewardRollout`/`RewardTrajectory` 被 reward 打分链路广泛使用（`vrl/rollouts/collector/rewards.py:71`、`vrl/rewards/artifacts.py`、`vrl/rewards/base.py`、多个 functions 和 tests）。

## 2. 质疑点 / 改进机会
- `RewardTrajectoryStep` 是死代码。`grep -rn "RewardTrajectoryStep(" --include=*.py` 在整个仓库无任何实例化结果；它只出现在 `types.py` 定义处和 `__init__.py`/`__all__` 的 re-export 里。注意：`ref_log_prob`/`new_log_prob` 的 grep 命中全部来自另一套类型 `vrl/rollouts/evaluators/types.py`，与本类无关。
- `RewardTrajectory.steps` 是恒空死字段。所有构造点都传 `steps=[]`：`vrl/rollouts/collector/rewards.py:74`、`vrl/scripts/data/danbooru.py`、以及全部 `tests/rewards/*`。没有任何代码读取 `trajectory.steps`（`grep "trajectory.steps\|\.steps"` 在 rewards/ 下无命中）。
- `RewardTrajectory.seed` 同样是死字段。所有构造点都传 `seed=0`，没有任何代码读 `trajectory.seed`（`grep "trajectory.seed"` 无命中）。reward 打分实际只用到 `trajectory.prompt` 和 `trajectory.output`（见 `artifacts.py:48,71` 和 `base.py:49,50`）。
- 结论：这是"为想象中的 per-step 打分预留的结构"，但 reward 链路只是按最终 output 打分。死结构会误导读者以为 reward 能看到逐步 trajectory，且 `steps` 类型还把 `RewardTrajectoryStep` 钉死在 import graph 里制造活性假象。

## 3. 建议动作
- 删除 `RewardTrajectoryStep` 类，并从 `vrl/rewards/__init__.py` 的 import 与 `__all__` 中移除（`__init__.py:7,21`）。
- 从 `RewardTrajectory` 删除 `steps` 和 `seed` 字段，只保留 `prompt: str` 和 `output: Any`。
- 同步更新所有构造点：移除 `steps=[]`、`seed=0` 实参——`vrl/rollouts/collector/rewards.py:71`、`vrl/scripts/data/danbooru.py:1475`、`tests/rewards/test_video_reward.py`、`test_video_reward_artifacts.py`、`test_video_reward_versioning.py`、`test_reward_function_local.py`、`test_multi.py`、`test_nsfw_safety.py`、`test_geneval_reward.py`、`test_ocr.py`。
- 若团队明确有 per-step reward（如 dense process reward）的近期规划，则改判 question 并在字段上加注释说明为何预留；但当前 import graph 里无任何消费者，证据指向删除。

## 4. 不动什么 / 为什么不是过度清理
- `RewardRollout` 与 `RewardTrajectory` 本身是真核心容器，跨 rewards/ + rollouts/ + scripts/ + tests 广泛使用，不动。
- `RewardTrajectory.prompt` / `output` 是实际被读取的字段，保留。
- 这不是为省几行而拍平 thin function；删的是从未被实例化的类和恒定值的死字段，属于 AGENTS.md 第 6 条死代码与"为不存在需求预留结构"的范畴。

## 5. 验证
- 删除后 `grep -rn "RewardTrajectoryStep\|\.steps\|\.seed" vrl/rewards tests/rewards vrl/rollouts/collector` 应无残留引用。
- 跑 `pytest tests/rewards -q` 全绿。
- `ruff check vrl/rewards/types.py` 无未使用 import 告警。
