# SPRINT(auto): vrl/algorithms/diffusion_nft.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/algorithms/diffusion_nft.py` (305 LOC)
角色判定: core
结论: improve

## 0. 一句话
这是真核心算法（被 factory/builder 实际装配），但它的 `compute_advantages_from_tensors` 与 `GRPO` 那份几乎逐行重复，应抽成共享 helper；`compute_batch_timestep_loss` 也偏 god-method，可拆。

## 1. 现状（读代码得出）
被生产路径装配：`vrl/scripts/common/factory.py:213-226` 构造 `DiffusionNFT(algorithm_config)`，`config/builders.py:122-125` 构造 `DiffusionNFTConfig`。算法本体（loss 组装、previous-policy / reference 双分支 forward）是 diffusion-specific 的真复杂度，属核心资产。

per-group advantage 归一化在 `diffusion_nft.py:47-71`：

```python
for gid in torch.unique(group_ids):
    mask = group_ids == gid
    group_rewards = rewards[mask]
    if group_rewards.numel() <= 1:
        advantages[mask] = 0.0
        continue
    mean = group_rewards.mean()
    std = rewards.std() if cfg.global_std else group_rewards.std()
    denom = torch.clamp(std, min=cfg.eps)
    group_adv = (group_rewards - mean) / denom
    advantages[mask] = torch.clamp(group_adv, -cfg.adv_clip_max, cfg.adv_clip_max)
```

`GRPO.compute_advantages_from_tensors`（`vrl/algorithms/grpo/continuous.py:43-79`）是同一个算法、同样的 `eps/adv_clip_max/global_std` 字段、同样的逐组归一化+clamp，只有 `global_std` 分支对 `numel<=1` 的极小写法差异。

## 2. 质疑点 / 改进机会
- **重复的核心数值逻辑**：两份 per-group advantage 归一化（`diffusion_nft.py:47-71` vs `grpo/continuous.py:43-79`）实质相同。这不是"跨家族刻意统一形状"那种值得保留的薄重复——它是同一段会随超参（eps clamp、global_std 语义）演化的数值代码被抄了两份，任一处改了归一化定义另一处会悄悄漂移。属于 AGENTS.md 要 flag 的"真实复杂度的共享抽象缺失"。
- **god-method 倾向**：`compute_batch_timestep_loss`（`diffusion_nft.py:93-244`）一个方法里塞了：replay tensor 抽取与校验、timestep 归一化（`t/1000` 启发式，`:156`）、noise 注入构造 `xt`、三路 forward（forward/previous/reference）、NFT positive/negative 目标组装、KL、metrics 装配。150 行单方法，职责可拆。
- `compute_loss`（`:73-91`）从 `inputs.metadata` 掏 `model` / `rollout_batch` / `timestep_index`（`:77-79`）——通过 `metadata: dict[str, Any]` 传强类型依赖，绕过了 `AlgorithmInput` 的 typed 字段，属弱契约。GRPO 走 `inputs.signals`（typed），DiffusionNFT 走 stringly-typed metadata，两条算法在同一协议下契约不一致。可记录，但这是设计权衡（diffusion 自带 model-forward），优先级低于重复逻辑。

## 3. 建议动作
- 把 per-group advantage 归一化抽到一个共享 helper，例如 `vrl/algorithms/advantages.py` 的 `group_relative_advantages(rewards, group_ids, *, eps, adv_clip_max, global_std)`，让 `GRPO` 和 `DiffusionNFT` 都调用它。统一 `global_std` 下 `numel<=1` 的处理（取 GRPO 的防御写法 `rewards.numel() > 1` 守卫）。这是把"会演化的数值定义"收敛到单一出处，符合 fix-root-cause。
- 把 `compute_batch_timestep_loss` 拆成几个私有步骤：`_resolve_replay_tensors(batch)` → `(x0, prompt_embeds, t_raw)`、`_build_noisy_input(x0, t, replay)` → `xt`、`_nft_targets(forward, previous, beta)` → `(positive, negative)`、`_assemble_metrics(...)`。主方法只编排。
- 模块级 helper `_forward_previous_policy_adapter` / `_forward_reference`（`:260-302`）是 justified 的（封装 PEFT adapter 切换的 try/finally 复杂度），保留。

## 4. 不动什么 / 为什么不是过度清理
- 不动 `DiffusionNFTConfig` 与 `DiffusionNFT` 的存在本身——它是被 factory 实际装配的核心算法，不是薄 wrapper。
- 不动 `uses_evaluator = False`（`:40`）这类家族标记——它是协议契约的一部分（GRPO 用 evaluator signals，NFT 不用），跨家族一致性标记，保留。
- 不动 `_forward_previous_policy_adapter` / `_forward_reference` 的薄函数形态——它们封装了真实的 adapter 切换复杂度（set_adapter/disable_adapters 的 try/finally + fallback 链），是移除真复杂度的抽象，符合 AGENTS.md "consistency over cleanup"，不要为省行数拍平。
- advantage helper 抽取要保证 `GRPO` / `DiffusionNFT` 数值行为不变，不要顺手"优化"归一化定义。

## 5. 验证
- 抽 helper 后：单测固定 `rewards`/`group_ids` 输入，断言 `GRPO` 与 `DiffusionNFT` 的 `compute_advantages_from_tensors` 输出与重构前逐元素相等（可临时保留旧实现做 golden 对比）。
- 跑 `pytest tests/algorithms`（含 NFT/GRPO 的 advantage 与 loss 测试）。
- grep `group_relative_advantages` 确认两处算法都改为调用同一 helper，旧的内联循环已删。
- `ruff check vrl/algorithms/diffusion_nft.py vrl/algorithms/grpo/continuous.py`。
