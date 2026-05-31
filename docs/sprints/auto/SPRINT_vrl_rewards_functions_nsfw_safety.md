# SPRINT(auto): vrl/rewards/functions/nsfw_safety.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/rewards/functions/nsfw_safety.py` (47 LOC)
角色判定: thin-wrapper
结论: question

## 0. 一句话
reward wrapper 本体 justified，但两个 passthrough 方法（`probability_batch` / `_probability_from_classifier_result`）只为让测试经 reward 对象触达 model 内部而存在，把 model 私有实现泄漏到了 reward 公共表面——值得质疑是否该让测试直接打 model。

## 1. 现状（读代码得出）
类核心是标准薄 wrapper（构造 `NSFWSafetyRewardModel` + `LocalRewardRuntime`），但额外挂了两个转发方法：

```python
def probability_batch(self, images: list[Any]) -> list[float]:
    """Raw NSFW probabilities for image-level safety audits."""
    return self._model.probability_batch(images)

def _probability_from_classifier_result(self, result: Any) -> float:
    return self._model._probability_from_classifier_result(result)
```

grep 确认这两个方法的唯一调用者是 `tests/rewards/test_nsfw_safety.py:62` 和 `:75`，生产代码（trainer / multi-reward）只走 `score` / `score_batch`，从不调用这两个。

## 2. 质疑点 / 改进机会
- `_probability_from_classifier_result` 是纯转发到 model 同名私有方法（`return self._model._probability_from_classifier_result(result)`），零额外逻辑、零边界价值，唯一目的是给 test 一个经 reward 对象的入口。证据：`nsfw_safety.py:42-43`，调用者仅 `tests/rewards/test_nsfw_safety.py:62`。
- `probability_batch` 同理，唯一调用者 `tests/rewards/test_nsfw_safety.py:75`。docstring 说是 "image-level safety audits"，但 grep 不到任何 audit 工具/脚本调用它——若该用例只存在于设想中而非实际 import graph，则属于为未来留的死表面。
- 这违反封装：reward 的公共职责是 `score`，把 model 的概率内部 API 复制一层挂到 reward 上，等于两处维护同一签名，model 改了私有方法签名两边都要动。

## 3. 建议动作
- 倾向：删掉这两个 passthrough，让 `test_nsfw_safety.py` 直接构造 / 访问 `NSFWSafetyRewardModel`（reward `__init__` 已把 model 存在 `self._model`，测试可 `reward._model.probability_batch(...)`，或直接实例化 model）。这样 reward 表面只剩 `score` 语义。
- 若确认存在 reward-level safety-audit 的真实生产入口（需 grep 出调用者证据），则把 `probability_batch` 留为 public 并加 audit 调用方的引用注释；`_probability_from_classifier_result` 这种私有转私有无论如何应删。
- 不确定点（故判 question 而非 improve）：需要先确认 "image-level safety audits" 是否有计划中的生产消费者。证据不足，不直接判 delete。

## 4. 不动什么 / 为什么不是过度清理
- `NSFWSafetyReward` 类主体保留：registry builtin（`registry.py:36`），`inference_runtime != "local"` 的显式拒绝（`:21-24`）和 eager build（`:25-26`）是有意的构造时校验边界。
- `**kwargs` 透传 threshold/penalty_scale 给 model 是 justified 的 config 边界，不动。
- 与其它 reward wrapper 的整体形状一致性保留。

## 5. 验证
- 改完 `pytest tests/rewards/test_nsfw_safety.py` 全绿（若把测试改为直接打 model，断言不变）。
- `grep -rn "probability_batch\|_probability_from_classifier_result" vrl tests` 确认 reward 层不再有这两个方法、调用点已迁到 model。
- `pytest tests/rewards/test_multi.py` 确认 nsfw_safety 作为 multi-reward 组件仍可构造。
