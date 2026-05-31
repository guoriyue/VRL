# SPRINT(auto): vrl/algorithms/dpo.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/algorithms/dpo.py` (182 LOC)
角色判定: core (纯函数损失) + thin-wrapper (DiffusionDPO 类)
结论: improve

## 0. 一句话
纯函数损失 `diffusion_dpo_loss` / `diffusion_sft_loss` 是真核心、被 offline trainer 实际调用；但 `DiffusionDPO` 这个 `Algorithm` 适配类在生产路径里从未被实例化，只活在测试里，文档声称的"shared algorithm adapter"用途并不存在。

## 1. 现状（读代码得出）
两个纯函数是真正干活的：`vrl/trainers/offline/dpo.py:25` `from vrl.algorithms.dpo import diffusion_dpo_loss, diffusion_sft_loss`，并在 `:331` / `:343` 调用。这部分移植自 Wallace et al. 论文，签名清晰、有完整 docstring，属于 keep-justified。

问题在 `DiffusionDPO(Algorithm)` 类（`dpo.py:39`）。它的 docstring 说：

```python
"""...The module keeps the pure functional loss for
offline trainers and exposes ``DiffusionDPO`` for the shared algorithm adapter."""
```

但 grep 全仓 `DiffusionDPO(` 的实例化结果只有一处，且在测试里：

```
tests/algorithms/test_dpo.py:185:    algo = DiffusionDPO(cfg)
```

生产侧 factory 明确把 dpo 当作 offline-only、不走 online adapter：

```
vrl/scripts/common/factory.py:230:    if kind == "diffusion_dpo":
vrl/scripts/common/factory.py:232:        "diffusion_dpo is an offline recipe and is not supported by common online recipe",
```

`config/builders.py:117-120` 也只构造 `DiffusionDPOConfig`，不构造 `DiffusionDPO`。

## 2. 质疑点 / 改进机会
- `DiffusionDPO` 类是一个没有生产调用方的 `Algorithm` 适配 wrapper。它的 `compute_advantages_from_tensors` 直接 `raise RuntimeError("DiffusionDPO is an offline preference objective")`（`dpo.py:59`），`compute_loss` 从 `metadata` 里掏 `model_pred/ref_pred/target` 再转调纯函数（`dpo.py:75`）。也就是说它把"明明是 offline、不该走 AlgorithmInput 协议"的东西硬塞进了 online `Algorithm` 协议——一个 advantage 方法注定抛异常，本身就说明这个抽象套错了。证据：`dpo.py:53-59`。
- 文档与现实矛盾：docstring（`dpo.py:9`）声称为 shared algorithm adapter 而存在，但没有任何 adapter 路径用它。属于 AGENTS.md 的 one-shot/装饰性抽象——已记录用途却无人引用。
- 这不是死代码到可直接 delete 的程度：测试覆盖了它，且作者可能有意保留 online-DPO 的接口占位。证据不足以判 delete，故判 improve。

## 3. 建议动作
二选一，倾向前者：

1. **删除 `DiffusionDPO` 类（推荐）**：DPO 本质是 offline preference，不该挂在 online `Algorithm` 协议上。删类、删 `__all__` 里的 `"DiffusionDPO"`（`dpo.py:177`），同步删 `tests/algorithms/test_dpo.py` 里仅测该类的用例（保留对两个纯函数的测试）。模块退化为纯函数损失库，与 `vrl/math/diffusion/nft.py` 同构。

2. **若确实计划做 online-DPO**：把 factory 接上（`scripts/common/factory.py` 的 `diffusion_dpo` 分支当前直接 raise），并修正 docstring 的"shared algorithm adapter"措辞使其名副其实；在那之前给类加 `# NOTE: not yet wired into any recipe` 并在 sprint 记录待办，而不是留一个 docstring 撒谎的占位。

## 4. 不动什么 / 为什么不是过度清理
- `diffusion_dpo_loss` / `diffusion_sft_loss` 两个纯函数不动：它们是被 offline trainer 真实调用的核心损失，docstring 详尽（论文出处 + tensor layout 约定），是 justified 的共享抽象。
- `DiffusionDPOConfig` 不动：被 `config/builders.py:118` 和 `scripts/diffusion/wan_2_1/train_dpo.py:94` 实际使用。
- 不要因为"省几行"去动纯函数；本 sprint 只针对那个用途虚假、生产无引用方的 `DiffusionDPO` 适配类。

## 5. 验证
- grep 确认无生产引用：`grep -rn "DiffusionDPO(" vrl/`（应为空，仅 class 定义行）。
- 删类后跑 `pytest tests/algorithms/test_dpo.py`（保留纯函数测试应通过）+ `pytest tests/trainers/offline`（offline DPO trainer 不受影响）。
- `ruff check vrl/algorithms/dpo.py` 确认无未用 import（删类后 `Algorithm` / `AlgorithmInput` / `TrainStepMetrics` import 应一并移除）。
