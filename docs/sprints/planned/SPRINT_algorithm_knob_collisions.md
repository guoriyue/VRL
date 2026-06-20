# SPRINT: 修正算法超参旋钮命名碰撞与重复 clamp（eps_clip / advantage-bound / KL 系数）(planned)

状态：planned（2026-06-20）
范围：消除 GRPO/TokenGRPO/DiffusionNFT 三族算法在 `eps` vs `eps_clip`、`adv_clip_max` vs `advantage_high`、`init_kl_coef` vs `kl_beta`、以及 `algorithm.kl_reward` 四处旋钮上的命名碰撞、重复 clamp 与错位归属，并修复 base grpo.yaml 中冻结策略的错值。

## 0. Core Decision（先看这一段）

四处算法旋钮存在真实的命名/语义碰撞，且其中一处已经在线上 config 里造成了"冻结策略"的真实误配，必须修：

1. **`eps` 与 `eps_clip` 前缀碰撞 → base config 写错值（high）**：`GRPOConfig` 同时有 `eps`（advantage 标准差下限，1e-4）与 `eps_clip`（PPO ratio clip 带宽，0.2）。两者都用科学计数法、共享 `eps` 前缀，读起来像同一族。base `configs/base/algorithm/grpo.yaml` 把 `eps_clip` 写成 `1.0e-4`（近乎零信赖域，直接冻结策略），下游实验 config 不得不加注释手动覆盖回 `0.2`。**修复方向**：把 PPO clip 改名为 `clip_ratio`（verl/trl 约定），并把 base 值修回真实信赖域。

2. **`adv_clip_max` 与 `advantage_high` 双重 clamp（medium）**：advantage 先在 `group_relative_advantages` 内被 clamp 到 `[-adv_clip_max, +adv_clip_max]`，再在 `compute_batch_timestep_loss` 里被 clamp 到 `[-advantage_high, +advantage_high]`。默认都是 5.0 时第二次 clamp 是纯 no-op；但 `advantage_high` 还兼任 reward_mix 归一化尺度和 policy_loss 重缩放。**修复方向**：把 clamp 边界统一到 `adv_clip_max`，把 `advantage_high` 的"尺度"职责改名为 `advantage_scale`，删掉重复 clamp。

3. **KL 系数跨族两名 + 误导前缀（medium）**：GRPO/TokenGRPO 叫 `init_kl_coef`，DiffusionNFT 叫 `kl_beta`，指同一个"乘在 kl_loss 上加进 policy_loss 的标量"。且 `init_` 前缀来自 trlx/verl 的自适应 KL 控制器，但本仓没有 `target_kl`/`horizon`/退火（grep 已确认不存在）。**修复方向**：统一为 `kl_coef`，去掉误导性的 `init_`。

4. **`kl_reward` 错位挂在 algorithm.* 下（low）**：`kl_reward` 声明在 `AlgorithmConfig`、写在 algorithm YAML，但没有任何算法 dataclass 消费它——builder 明确把它放进 `ignored_keys`。它实际被 rollout collector 消费（reward shaping）。**修复方向**：迁到 rollout.* 或改名 `kl_reward_coef` 并在 schema 注明是 collector-consumed。

四项都已逐一重开文件核实为真，且无一与既有 hygiene/naming sprint 重叠。`eps_clip`、`init_kl_coef`、`kl_beta`、`advantage_high`、`kl_reward` 都是 user-facing config 边界，改名必须 schema + 消费点 + 全部 YAML 同步进行。

## 1. 现状实锤

### 1.1 eps / eps_clip 前缀碰撞 + base 错值

`vrl/algorithms/grpo/continuous.py:25-28`：

```python
    eps_clip: float = 0.2          # PPO ratio clip 带宽
    init_kl_coef: float = 0.0
    eps: float = 1e-4              # advantage 标准差下限
    adv_clip_max: float = 5.0
```

`eps_clip` 的真实用途是 PPO surrogate clip（`vrl/algorithms/grpo/continuous.py:106`）：

```python
        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip)
```

而 `eps` 只是 advantage 除法的数值下限（传给 `group_relative_advantages`，`continuous.py:70`）。两者尺度差几个数量级却共享前缀。

base config 把 PPO clip 写成了 1e-4，`configs/base/algorithm/grpo.yaml:4-6`：

```yaml
  eps_clip: 1.0e-4
  init_kl_coef: 0.0
  eps: 1.0e-8                   # numerical stability for advantage division
```

后果实锤——下游 `configs/experiment/ar/janus_pro/online_r1_grpo_aesthetic.yaml:25-28` 不得不加注释手动覆盖：

```yaml
  # The base grpo.yaml ships eps_clip=1e-4 — a near-zero trust region that
  # freezes the policy. The proven non-R1 AR baseline (online_grpo_ocr) uses
  # 0.2; match it so updates are real. init_kl_coef mirrors that baseline.
  eps_clip: 0.2
```

`online_grpo_ocr.yaml:10` 同样把它覆盖成 `0.2`。注意 cosmos_predict2 两个 config（`online_grpo_v2w_reference.yaml:17`、`online_grpo_kling_video_reward.yaml:30`）用的是 `1.0e-3`，也是个被科学计数法迷惑的小值——base 错值已经污染了多个实验。

`TokenGRPOConfig` 继承自 `GRPOConfig`（`vrl/algorithms/grpo/token.py:21`），所以 `eps_clip` 的改名会通过继承自动覆盖 token / multisegment 两族。

### 1.2 adv_clip_max vs advantage_high 双重 clamp

`DiffusionNFTConfig` 同时声明两个 [-x,x] clamp 旋钮，默认都是 5.0，`vrl/algorithms/diffusion_nft.py:20-24`：

```python
    adv_clip_max: float = 5.0
    ...
    advantage_high: float = 5.0
```

第一次 clamp 在 advantage 计算内（`vrl/algorithms/advantages.py:35`，由 `diffusion_nft.py:63` 传入）：

```python
        advantages[mask] = torch.clamp(group_adv, -adv_clip_max, adv_clip_max)
```

第二次 clamp 在 loss 里，`vrl/algorithms/diffusion_nft.py:241`：

```python
        adv = torch.clamp(advantages, -advantage_high, advantage_high)
```

默认相等时第二次纯 no-op。但 `advantage_high` 还兼任另外两个职责（不能直接删），`diffusion_nft.py:245,264`：

```python
        reward_mix = ((adv / advantage_high) / 2.0 + 0.5).clamp(0.0, 1.0)   # 归一化尺度
        ...
        policy_loss = original_policy_loss.mean() * advantage_high           # loss 重缩放
```

并且诊断 metric `adv_saturation` 只对 `adv_clip_max` 报告饱和率，`vrl/trainers/online/trainer.py:518-520`：

```python
        _clip_max = getattr(self.algorithm.config, "adv_clip_max", None)
        adv_saturation = (
            float((_adv_abs >= _clip_max - 1e-6).sum().item()) / _total
```

——所以当用户只调 `advantage_high`、让两者发散时，clamp 行为变了但 metric 完全看不见。

### 1.3 KL 系数跨族两名 + init_ 误导前缀

同一概念两个名字。GRPO，`vrl/algorithms/grpo/continuous.py:149`：

```python
            kl_term = cfg.init_kl_coef * kl_loss
```

DiffusionNFT，`vrl/algorithms/diffusion_nft.py:266`：

```python
        kl_term = float(cfg.kl_beta) * kl_loss
```

默认值还不一致（`continuous.py:26` 是 0.0，`diffusion_nft.py:23` 是 1.0）。`init_` 前缀暗示自适应退火，但全仓 grep `target_kl`/`horizon` 无任何 KL 控制器命中（命中的全是 data preprocessing 的 `horizontal_flip`），证明该前缀名不副实。

`init_kl_coef` 被 trainer 多处 getattr 读取以决定是否需要 ref model，`vrl/trainers/online/trainer.py:666-677` 与 `1085-1096`：

```python
                            init_kl_coef = float(
                                getattr(self.algorithm.config, "init_kl_coef", 0.0),
                            )
                            ... need_ref=init_kl_coef > 0,
```

以及 `vrl/scripts/common/online.py:202`。改名需同步这些 getattr。

### 1.4 kl_reward 错位挂在 algorithm.* 下

`kl_reward` 声明在 schema，`vrl/config/schema.py:105`：

```python
    kl_reward: Any = None
```

但没有任何算法 dataclass 消费它——builder 明确忽略，`vrl/config/builders.py:135`：

```python
    ignored_keys = {"kind", "kl_reward"}
```

真正的消费者是 rollout collector：从 `algorithm.kl_reward` 拷出（`vrl/rollouts/collector/config.py:57`），在 reward 上减掉 per-step KL（`vrl/rollouts/collector/batch_builder.py:116-119`）：

```python
        if self.context.kl_reward > 0:
            ...
                - self.context.kl_reward * kl_tensor.sum(dim=1)
```

于是 `algorithm.*` 下出现三个 KL-looking key（`init_kl_coef`/`kl_beta` 是 loss-side，`kl_reward` 是 reward-side 且被算法 builder 静默忽略），读者无法区分。`kl_reward` 这名字也读起来像 metric，而 verl 的对应名是 `use_kl_in_reward`。

## 落地方案

按严重度推进，每步是一个独立可验收的 config-boundary 改名。

### A. eps_clip → clip_ratio + 修 base 错值（high，先做）

1. `vrl/algorithms/grpo/continuous.py`：把 `eps_clip` 字段改名为 `clip_ratio`，更新 `continuous.py:106,159` 两处消费点；`token.py:71,82,93,102` 通过继承自动跟随，逐处把 `cfg.eps_clip` 替换为 `cfg.clip_ratio`。
2. `vrl/config/schema.py:99`：schema key `eps_clip` → `clip_ratio`。
3. 修 base：`configs/base/algorithm/grpo.yaml:4` 改为 `clip_ratio: 0.2`（真实信赖域）。
4. 全部实验 YAML 同步改名：`online_grpo_ocr.yaml:10`、`online_r1_grpo_aesthetic.yaml:28`、`cosmos_predict2/online_grpo_v2w_reference.yaml:17`、`cosmos_predict2/online_grpo_kling_video_reward.yaml:30`（grep `eps_clip` 全量定位）。cosmos 两个 `1.0e-3` 借此 review 是否也是被科学计数法迷惑的错值——若是 PPO clip 真实意图，提升到合理值。
5. base 修值后，`online_r1_grpo_aesthetic.yaml` / `online_grpo_ocr.yaml` 里"覆盖回 0.2"的显式覆盖与注释可删（默认已正确）。

### B. 统一 advantage clamp，拆出 scale 职责（medium）

1. `vrl/algorithms/diffusion_nft.py`：删除 `diffusion_nft.py:241` 的第二次 clamp，clamp 唯一来源为 `group_relative_advantages` 内的 `adv_clip_max`。
2. 把 `advantage_high` 改名为 `advantage_scale`（保留它在 `:245` 归一化与 `:264` 重缩放的尺度职责），更新 `:148-150` 的校验与 `DiffusionNFTConfig:24` 字段名。
3. `vrl/config/schema.py:96`：`advantage_high` → `advantage_scale`。
4. YAML：`configs/base/algorithm/diffusion_nft.yaml:11` 改名。
5. （可选增强）让 `adv_saturation` metric 同时覆盖 NFT 的 scale，或在注释里说明 NFT 的 clamp 唯一来源是 `adv_clip_max`。

### C. KL 系数统一为 kl_coef（medium）

1. `vrl/algorithms/grpo/continuous.py:26`、`vrl/algorithms/diffusion_nft.py:23`：两个字段统一为 `kl_coef`，更新各自消费点（`continuous.py:127,130,149`；`token.py:79,82,93`；`diffusion_nft.py:266`）。
2. `vrl/config/schema.py:102-103`：把 `init_kl_coef` 与 `kl_beta` 合并为单一 `kl_coef` key。
3. trainer getattr 同步：`vrl/trainers/online/trainer.py:666-677`、`1085-1096`；`vrl/scripts/common/online.py:202`；`vrl/scripts/perf/fp8_rollout_drift_probe.py:127`。
4. 全部 YAML：grep `init_kl_coef`/`kl_beta`（anima_preview3、sd3_5、wan_2_1/2_2、cosmos、diffusion_nft 等十余处）统一改 `kl_coef`。
5. DPO 的 `beta` 不动——它是 temperature，不是 loss-side KL 系数（见 finding load-bearing caveat）。

### D. kl_reward 归位（low，可与 C 同批）

1. 把 `kl_reward` 从 `algorithm.*` 迁到 `rollout.*`（它真正被 collector 消费），或保留位置但改名 `kl_reward_coef` 并在 `vrl/config/schema.py:105` 注明 "collector-consumed, not algorithm-consumed"。
2. 同步迁移消费点 `vrl/rollouts/collector/config.py:57`（拷贝来源地址）与 base YAML `grpo.yaml:13` / `diffusion_nft.yaml:16`。
3. 若改名，`vrl/config/builders.py:135` 的 `ignored_keys` 相应更新或（迁出 algorithm.* 后）移除该 key。

## 验证（finishing criteria）

- `pytest` 全绿（尤其 algorithm / config builder / collector 相关测试）。
- 对每个改过的 base + experiment YAML 跑 config-resolve（`build_algorithm_config` / `build_rollout_config_from_cfg`），确认无 "unknown ... config field" 报错、且解析出的值符合预期。
- grep 确认旧名 `eps_clip`、`init_kl_coef`、`kl_beta`、`advantage_high` 在 `vrl/` 与 `configs/` 下零残留（除非有意保留兼容别名）。
- 确认 base `grpo.yaml` 的 PPO clip 默认值不再冻结策略（≥0.1 量级）；确认 `online_r1_grpo_aesthetic.yaml` / `online_grpo_ocr.yaml` 的显式覆盖已可安全删除。
- DiffusionNFT 跑一步 forward，确认删掉第二次 clamp 后 loss 数值与默认 5.0/5.0 配置一致（no-op 删除应零数值变化）。

## 非目标 / Non-Goals

- 不引入自适应 KL 控制器（`target_kl`/`horizon`）——只是去掉误导性的 `init_` 前缀，不新增退火逻辑。
- 不改 DPO 的 `beta`（temperature，与 KL 系数无关）。
- 不改 `adv_clip_max` 本身的语义或默认值——它是统一后的唯一 clamp 来源，保持不变。
- 不做与本主题无关的算法数学重构（reward_mix 公式、NFT 正负分解等保持原样，仅改名与去重 clamp）。

## References

- `vrl/algorithms/grpo/continuous.py:22-30`（GRPOConfig 字段）, `:70,106,127-130,149,159`（消费点）
- `vrl/algorithms/grpo/token.py:21,71,79-82,93,102`（TokenGRPO 继承 + 消费）
- `vrl/algorithms/grpo/multisegment.py:17`（MultiSegmentTokenGRPOConfig 继承）
- `vrl/algorithms/diffusion_nft.py:16-25`（DiffusionNFTConfig 字段）, `:48-65,148-150,241,245,264,266,288`（clamp / scale / kl 消费）
- `vrl/algorithms/advantages.py:13,35`（group_relative_advantages clamp）
- `vrl/config/schema.py:95-105`（AlgorithmConfig 五个 key）
- `vrl/config/builders.py:130-140`（_dataclass_payload + ignored_keys）
- `vrl/rollouts/collector/config.py:57,123-124,165`、`core.py:167`、`batch_builder.py:37,116-119`（kl_reward 真实消费链）
- `vrl/trainers/online/trainer.py:514-523`（adv_saturation metric）, `:666-677,1085-1096`（init_kl_coef getattr 门控）
- `vrl/scripts/common/online.py:194,202-203`（ref model 由 init_kl_coef 决定）
- `vrl/scripts/perf/fp8_rollout_drift_probe.py:127`
- `configs/base/algorithm/grpo.yaml:1-13`、`configs/base/algorithm/diffusion_nft.yaml:1-16`
- `configs/experiment/ar/janus_pro/online_r1_grpo_aesthetic.yaml:24-29`、`.../online_grpo_ocr.yaml:10-11`
- `configs/experiment/diffusion/cosmos_predict2/online_grpo_v2w_reference.yaml:17,21`、`.../online_grpo_kling_video_reward.yaml:30-31`
- 其余 init_kl_coef YAML：anima_preview3、sd3_5、wan_2_1/2_2 系列（grep `init_kl_coef` 全量）

相关 sprint：[[SPRINT_precision_naming_unification]]（同类 config-key 命名统一）、[[SPRINT_config_string_settings]]、[[SPRINT_config_as_signatures]]（config 旋钮类型化/边界）。
