# SPRINT(auto): vrl/math/ar/flow_matching.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/math/ar/flow_matching.py` (257 LOC)
角色判定: core
结论: question

## 0. 一句话
这是 NextStep-1 per-token flow-matching 的真核心数学（被 runner/model 实际调用），不是死代码也不是薄
wrapper；但它对上游 `image_head` 的 velocity 调用契约是「未确认绑定」（`# TODO(nextstep-binding)`），
存在数学静默跑错 upstream API 的风险，需要确认或加运行期校验。

## 1. 现状（读代码得出）
两个公开函数 `flow_sample_with_logprob` / `flow_logprob_at`，实现 per-token 高斯 log-prob 的
flow-matching 采样与重算，约定与 diffusion 侧 `sde_step_with_logprob` 一致（K-1 步确定性 Euler + 末步注入高斯噪声）。

它确实被生产路径调用，不是 dead：
- `vrl/models/ar/nextstep_1/runner.py:242` 调 `flow_sample_with_logprob(self.model.image_head, ...)`
- `vrl/models/ar/nextstep_1/model.py:298` 调 `flow_logprob_at(...)`
- 该 family 在 `vrl/models/ar/nextstep_1/__init__.py` 注册并 re-export（`NextStep1Model` 等）

velocity 调用契约靠 `getattr` / `hasattr` 猜：
```python
# flow_sample_with_logprob 默认路径（line 120-125）
def _velocity(xk, tk, c):
    if velocity_fn is not None:
        return velocity_fn(xk, tk, c)
    return image_head.net(xk, tk, c)   # 直接假设 .net 存在
```
而同文件 `flow_logprob_at` 的 fallback 链却不同（line 220-227）：先 `.net`，再 `.velocity`，再 `image_head(...)`。
token 维度也靠 `getattr(image_head, "input_dim", None)`（line 93）。

文件 docstring 自承未绑定：line 81-85
> "The exact velocity-call signature depends on NextStep-1's upstream implementation. Until we have
> stepfun-ai/NextStep-1 installed we cannot bind this — see the `# TODO(nextstep-binding)` markers."

## 2. 质疑点 / 改进机会
1. 未确认绑定 + 静默错配风险（question 的主因）：`flow_sample_with_logprob` 默认直接 `image_head.net(...)`，
   若上游属性名不是 `.net` 会 `AttributeError`；更糟的是若 upstream `net` 的入参顺序/语义不是 `(x, t, cond)`，
   数学会「跑通但跑错」，GRPO ratio 静默失真，难以察觉。证据：`flow_matching.py:125`、`:85`。
2. 两个函数的 velocity fallback 不一致：sample 路径只认 `.net`（line 125），logprob 路径认 `.net`/`.velocity`/`__call__`
   三选一（line 223-227）。同一文件同一契约写了两套，采样和重算可能解析到不同 callable，破坏 sample↔replay 一致性
   （这正是 GRPO old/new logprob 必须同源的前提）。应抽一个共享 `_resolve_velocity(image_head, velocity_fn)`。
3. one-shot 标识：`# TODO(nextstep-binding)` 已记录问题但代码长期留在 import graph 中且无 `*_spike`/guard 标记。
   按 AGENTS.md 第 4 条，问题已记录却还以「未验证」状态留在长期资产路径里，应当显式收口（要么补 binding 测试，
   要么在缺失属性时 fail-fast 并指向 modeling_nextstep.py）。

非问题（不要误判）：这不是薄 wrapper，也没有 ALL_CAPS 手抄结构，命名（flow_matching / 函数名）也准确，
放在 `vrl/math/ar/` 是合理的共享数学边界。

## 3. 建议动作
- 不删（有真实 caller，证据见 §1 的 runner.py:242 / model.py:298）。verdict 取 question 而非 improve，
  因为核心不确定点是「上游契约对不对」，需 upstream 信息才能定论。
- 统一 velocity 解析：把 `_velocity` 的解析逻辑抽成模块级 `_resolve_velocity_callable(image_head, velocity_fn)`，
  两个函数共用，消除 sample/replay 解析分叉。
- fail-fast：当既无 `velocity_fn` 又无 `.net`/`.velocity`/`__call__` 时抛带 `modeling_nextstep.py` 指引的
  `RuntimeError`，而不是让 `image_head.net` 直接 `AttributeError`（与 `input_dim` 缺失时的报错风格一致，见 line 95-99）。
- 落地一个绑定校验：用 fake `image_head`（test fake）跑一遍 `flow_sample_with_logprob` → `flow_logprob_at` 往返，
  断言 `saved_noise` 复用时 old logprob 可精确复现（std 已知，误差应为 0）。这能把「未确认」变成「契约被测试钉住」。

## 4. 不动什么 / 为什么不是过度清理
- 高斯 log-prob 公式（line 163-168 / 250-255）、SUM-over-D（非 mean）的选择、`std = noise_level*sqrt(dt)` 的
  SDE-from-ODE 参数化都是与 diffusion 侧刻意对齐的约定，属于跨家族一致性，保留不动。
- `flow_logprob_at` 三段 fallback 本身的「宽容」不是缺点（它要兼容多种 head），问题只在于和 sample 路径不一致；
  统一即可，不要为省行数砍掉对 `velocity_fn` override 的支持（那是合法的 framework adapter 注入点）。
- 不要把这文件并进 runner——它是模型无关的纯数学，独立于 `vrl/math/ar/` 是正确的边界。

## 5. 验证
- 新增/运行往返一致性测试（fake head，固定 generator）：`pytest tests/rollouts/ -k flow` 全绿，
  断言 `flow_logprob_at(saved_noise=x0)` 与采集时 `flow_sample_with_logprob` 返回的 log_prob 数值一致。
- `grep -rn "image_head.net\|_velocity\|velocity_fn" vrl/math/ar/flow_matching.py` 确认两函数走同一 `_resolve_*`。
- `ruff check vrl/math/ar/flow_matching.py`。
- 真接入 stepfun-ai/NextStep-1 后，跑一次 runner.py 的 AR step，确认无 AttributeError 且 GRPO ratio 量级合理。
