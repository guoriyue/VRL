# SPRINT: Rollout-Correction —— RS reject-sampling + bypass-vs-recompute（在已落地的 TIS/drift 之上扩展）

状态：**done（2026-06-20 落地，66 个 RS/GRPO 单测全绿；250 个 algorithms+config+precision-bridge 测试通过，仅 2 个与本 sprint 无关的 Wan2.2/SD3.5 配置预存失败；ruff clean）**。范围限定：只在 `logprob_mismatch.py` 与 importance-ratio 算法（`grpo/token.py`、`grpo/continuous.py`）这条已有缝里加两个正交机制（RS 拒绝采样 + bypass 复用 rollout log-prob 省掉 replay forward），不重写 trainer 主循环、不动 evaluator 协议。

## 落地记录（2026-06-20）

- `PrecisionCorrectionConfig` 扩了 RS 三字段 + bypass 开关（`logprob_mismatch.py`）：`rs_mode`（off/seq_mean_k1/seq_max_k1，`token_*` 在 `__post_init__` 直接拒绝并指向 seq 模式）、`rs_log_ratio_low/high`（默认 ln0.5/ln2.0 log-ratio 数值，非 string）、`recompute_old_logprob`（off=bypass 唯一实现；on 在构造时 `NotImplementedError`，不做静默 no-op）。schema/builder 全自动从 dataclass 字段派生，无需改 `schema.py`/`builders.py`。
- 新增 `apply_rejection_sample_mask(log_ratio, config, *, mask=None)` 与 `combine_keep_masks(*masks)`（同文件，与 `apply_truncated_importance_weight` 并列）。per-sample（1D，连续扩散）下 seq_mean/seq_max 退化同判据；token（B,L）下按序列轴聚合，返回 (B,1) 广播掩码，`mask` 把聚合限制在有效 token。
- `continuous.py` / `token.py` 各加 RS 消费：`keep = combine_keep_masks(tis_keep, rs_keep)`（token 再乘 token mask），分母真实剔除（非梯度稀释）；新增 `rs_seq_masked_fraction` 标量，挂进 `TrainStepMetrics`（与 `tis_clip_fraction` 同作用域——只在 per-step 算法 metrics，不进 trainer 聚合/CSV）。`MultiSegmentTokenGRPO` 经 `super().compute_loss()` 自动覆盖。
- `fp8_rollout_drift_probe.py` 加 RS 分支（seq_mean_k1 + 默认带），报 `rs_seq_masked_fraction` 并标注 <5% OK / ≥5% 收紧带或退 recompute。
- 测试：`test_logprob_mismatch.py`（RS config 校验 + mask 函数 + combine）、`test_grpo.py::TestGRPORejectSampling`（连续：越界剔除 + RS/TIS 合并分母）、`test_grpo_token.py::TestTokenRejectSampling`（seq_max 单步 outlier 拒整段、seq_mean 抵消保留、RS+TIS 折进 eff_mask、mask 感知聚合）。

关联：
- [[SPRINT_fullparam_and_fp8_precision]] —— TIS（`apply_truncated_importance_weight`）与 drift guard（`precision_guard.py`）的归属 sprint；本 sprint 是它的直接延伸，复用同一个 `PrecisionCorrectionConfig` 槽位与同一套 `compute_logprob_mismatch_stats` 度量。
- [[SPRINT_fp8_rollout_gemm_kernel]] —— fp8 rollout kernel；fp8 rollout 是放大 rollout→replay drift 的主要来源，本 sprint 是它的 RL 侧安全网。
- 当前分支 `fp8-rollout-precision-tis` 正是 TIS/drift 落地分支；本 sprint 的 RS 与 bypass 与该分支同源，应在其上继续而非另起炉灶。

## 0. Core Decision（先看这一段）

vrl 已经有「测量（drift metrics）+ 截断（TIS）」这半套 rollout-correction；缺的是 verl-omni §3.3 里与截断**正交**的另外两件事：(1) **RS 拒绝采样**——把 log-ratio 落在带外的样本整段丢弃（权重置 0），与 TIS 的「保留但夹住」互补；(2) **bypass-vs-recompute**——当 drift 有界时，直接把 rollout 时记录的 `old_log_prob` 当作 PPO 的 old 用，省掉每个训练 timestep 的 replay forward（单卡 32GB 上是实打实的 FLOP 节省，对应 NFT ~62% backward 的关键路径）。关键的 diffusion 洞见是：**SDE window 很短（通常为 2 步），token-level RS 统计功效极低**（单步的 per-token stat 已是上千 latent 维的均值），所以拒绝必须做在 sequence 维——优先 `seq_mean_*` / `seq_max_*`，不要 `token_*`。落地方式是把这两件事塞进 vrl 已有的 `PrecisionCorrectionConfig` + `apply_truncated_importance_weight` 这条缝，复用同一个 trainer 级注入点，不新建并行体系。

## 1. vrl 现状：已有「测量 + 截断」，但 mask 模式是「带内截断」不是「拒绝采样」

vrl 的 TIS 已经支持四种模式，`mask` 模式甚至已经会返回一个 keep-mask：

```python
tis_mode: str = field(default="off")  # "off" | "truncate" | "clip" | "mask"
...
if mode == "mask":
    keep = (ratio <= cap) & (ratio >= low)
    return ratio, keep.to(ratio.dtype)
```
（`vrl/algorithms/logprob_mismatch.py:89`、`:126`）

算法侧也已经消费这个 keep-mask，把被拒样本从 mean 分母里剔掉：

```python
ratio, tis_keep = apply_truncated_importance_weight(raw_ratio, pc)
...
if tis_keep is not None:
    policy_loss = (per_sample_loss * tis_keep).sum() / tis_keep.sum().clamp_min(1.0)
```
（`vrl/algorithms/grpo/continuous.py:98`、`:103`；token 版同构于 `vrl/algorithms/grpo/token.py:60`、`:67`，且把 keep-mask 与 token mask 相乘：`eff_mask = mask if tis_keep is None else mask * tis_keep`，`token.py:61`）

**这就是 DELTA 的精确位置**：vrl 现在的 `mask` 模式判据是「IS 权重 `ratio` 本身落在 `[low, cap]` 带内」——它丢的是「权重越界」的样本，本质还是把 TIS 的硬截断换成硬剔除，**判据仍是 per-element 权重**。verl-omni 的 RS 是另一个判据维度：在 **log-ratio** 上、按**序列聚合后**（`seq_mean_k1` / `seq_max_k1`）设一个带 `"low_high"`，整段序列要么全留要么全弃。两者正交：TIS 控「单点权重大小」，RS 控「整条轨迹的偏移是否离谱」。vrl 目前没有 sequence 聚合判据，也没有 `seq_mean/seq_max` 这个轴。

度量侧 vrl 已经齐全，RS 不需要重造度量：

```python
return LogprobMismatchStats(
    logprob_abs_diff_mean=..., logprob_abs_diff_max=...,
    ratio_abs_dev_mean=..., ratio_abs_dev_max=...,
    mismatch_kl=float((-delta).mean()),
    mismatch_k3_kl=float((ratio - delta - 1.0).mean()),
    finite=finite,
)
```
（`vrl/algorithms/logprob_mismatch.py:62`）——`mismatch_kl` / `mismatch_k3_kl` 正是 verl-omni doc 里 `rollout_corr/kl` 与 `rollout_corr/k3_kl` 的等价物，RS 的 `seq_masked_fraction` 只需新增一个标量，挂进 `TrainStepMetrics` 即可。

trainer 级注入点也已经就位（这是 RS 复用的关键，不要另开 config 块）：

```python
if hasattr(algorithm, "precision_correction"):
    algorithm.precision_correction = config.precision_correction
```
（`vrl/trainers/online/trainer.py:333`；config 字段定义在 `vrl/trainers/core/types.py:256`，YAML key `trainer.precision_correction`；schema 暴露在 `vrl/config/schema.py:337`）

## 2. vrl 现状：`old_log_prob` 已经是「rollout 行为 log-prob」，bypass 几乎是免费的

bypass-vs-recompute 的可行性，取决于 vrl 里 `old_log_prob` 到底从哪来。读 evaluator 实锤了：**vrl 的 `old_log_prob` 就是 rollout 时存进 trajectory segment 的那个张量**，不是另跑一遍 replay 算出来的：

```python
def _old_log_prob_from_trajectory(self, segment, *, log_prob, timestep_idx):
    ...
    value = role_tensor(segment, "old_log_prob").value
    return self._select_loss_value_if_needed(value, log_prob, timestep_idx=timestep_idx)
```
（`vrl/rollouts/evaluators/trajectory.py:182`、`:194`）

而每个训练 timestep 真正花 forward 的，是这一步——它产出 `signals.log_prob`（current/new policy）：

```python
with record_function("trainer.replay"):
    signals = self.evaluator.evaluate(self.model, chunk_batch, j, ref_model=..., signal_request=...)
```
（`vrl/trainers/online/trainer.py:669`）

所以 vrl 当前结构与 verl-omni 的 bypass 语义**已经天然对齐**：vrl 的 `old_log_prob`（rollout 记录）= verl-omni bypass 里 `old_log_probs := rollout_log_probs` 的结果。**DELTA 在于「recompute」那一支 vrl 根本没有**——verl-omni 的标准（decoupled，`bypass_mode=False`）流程是 rollout log-prob 之外**再单独跑一遍 full-precision old recompute** 产出 `old_log_probs`，doc 说这步 ~20% per-step 时间；vrl 现在省掉了这步（直接用 rollout 记录），等价于「永远 bypass」，但**没有把它当成一个可控开关，也没有在 bypass 时补上 IS/RS 这层 off-policy 修正**。

verl-omni 把 bypass 写成一行零成本替换：

```python
def apply_bypass_mode_to_diffusion_batch(batch: DataProto) -> None:
    ...
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
```
（`verl_omni/trainer/diffusion/rollout_correction.py:63`、`:81`）

并明确：bypass 下 PPO ratio `exp(current − rollout)` 已经充当 IS，`rollout_is` 只算诊断、**只有 `rollout_rs` 影响梯度**：

```python
# ppo_clip: PPO ratio handles IS, only RS mask is applied.
assert rc_cfg.loss_type == "ppo_clip", ...
if rc_cfg.rollout_rs:
    rs_mask = modified_mask
    weights = rs_mask if weights is None else weights * rs_mask
```
（`verl_omni/workers/utils/losses.py:50`、`:54`）

这正好对上 vrl 现状：vrl 的 `raw_ratio = torch.exp(signals.log_prob - old_log_probs)`（`grpo/continuous.py:93`）在 bypass 下就是 `exp(current − rollout)`，PPO clip 已经在做 IS，**所以 vrl 要补的就是那条 RS mask**。换句话说，vrl 的 bypass 路径与 verl-omni 的 `bypass_mode=True` + `loss_type=ppo_clip` 是同一个数学对象，只差一个 RS 判据。

**这给本 sprint 一个干净的边界**：vrl 不需要去实现 verl-omni 的 decoupled recompute（那是为了「rollout 与 training stack 评估同一轨迹略有差异」时拿一个更准的 old 当锚点）。vrl 单卡 32GB、reward+rollout 串在关键路径上，**recompute 是奢侈品**；vrl 要做的是把「永远 bypass」显式化为一个 `recompute_old_logprob: off|on` 开关（默认 off=bypass，省 forward），并在 bypass 下用 drift guard 守住「drift 有界」这个前提、用 RS 兜住越界轨迹。

## 3. verl-omni §3.3 的 RS 配方与短 SDE window 的功效问题

verl-omni 的 RS 通过两个 string 配置驱动，`rollout_rs` 选聚合模式、`rollout_rs_threshold` 给一个 `"low_high"` 带：

```
rollout_rs: ${oc.select:algorithm.rollout_correction.rollout_rs,null}   # e.g. "token_k1", "seq_sum_k1", "seq_mean_k1"
rollout_rs_threshold: ${oc.select:...}                                   # K1: "lower_upper" string
```
（`verl_omni/trainer/config/diffusion/actor/diffusion_actor.yaml:81`、`:84`）

推荐预设是 sequence 维 + log-ratio 带 `[ln0.5, ln2.0]≈[−0.69,+0.69]`：

```
algorithm.rollout_correction.rollout_rs=seq_mean_k1
algorithm.rollout_correction.rollout_rs_threshold="0.5_2.0"
```
（`docs/algo/rollout_correction.md:39`）

**短 SDE window 的功效洞见（本 sprint 必须吸收的 diffusion-specific 结论）**，doc 写得很直接：

> With only 2 tokens, token-level statistics have very low power — a single token cannot be rejected in isolation because the per-token stat is averaged from thousands of latent dims. Prefer `seq_mean_*` or `seq_max_*` modes ...
（`docs/algo/rollout_correction.md:105`）

> The SDE window is short (`sde_window_size` is usually 2) ... The LLM default `"0.5_2.0"` means the *mean* log-ratio over only 2 steps must lie in `[−0.69, 0.69]`. A single outlier step can reject the entire sample. If rejected_ratio is high, widen to e.g. `"0.3_3.0"` or `"0.2_5.0"`.
（`docs/algo/rollout_correction.md:98`、`:103`）

还有两个工程细节本 sprint 要照搬：
- **梯度稀释**：RS 把被拒元素的 loss 置 0，但**不**从 `mean()` 分母里移除，所以高拒绝率下有效梯度按 `kept/total` 缩水（`docs/algo/rollout_correction.md:82`）。注意这与 vrl 现有 TIS `mask` 模式的做法**相反**——vrl 是 `.sum()/tis_keep.sum()`（`continuous.py:104`，从分母剔除）。本 sprint 要明确选一个语义（建议沿用 vrl 的「从分母剔除」=真实 off-policy 拒绝，不稀释梯度幅度），并在 metric 里同时报 `rs_seq_masked_fraction` 让人能监控拒绝率。
- **监控阈值**：`rollout_rs_seq_masked_fraction` 持续 > ~5% 说明 rollout drift 过大，应收紧带或退回 recompute（`docs/algo/rollout_correction.md:78`）——这正好接上 vrl 的 drift guard：guard 负责「fail/warn」，RS 负责「在 loss 里兜住」，两者读同一套 `compute_logprob_mismatch_stats`，不能各算各的。

## 4. 落地方案（在已有缝里加，不新建并行体系）

### 4.1 `PrecisionCorrectionConfig` 扩 RS 字段（同一个 dataclass，不新建 config 块）

在 `vrl/algorithms/logprob_mismatch.py` 的 `PrecisionCorrectionConfig` 上加 RS 轴，与现有 `tis_*` 平级（沿用 user 的 explicit `field(...)` 拼写约定，见 MEMORY「Explicit field() spelling」）：
- `rs_mode: str = field(default="off")` —— `"off" | "seq_mean_k1" | "seq_max_k1"`。**不实现 `token_*`**：§3 已论证短 window 下 token-level RS 低功效，提供它只会诱导误用。`__post_init__` 直接拒绝 `token_*` 并在报错里指向 seq 模式。
- `rs_log_ratio_low: float` / `rs_log_ratio_high: float` —— 对应 verl-omni 的 `"low_high"` 带，但**存成 log-ratio 数值**（避免再 parse string）。default 取 `ln(0.5)≈-0.693` / `ln(2.0)≈0.693`，与 verl-omni 预设一致；`__post_init__` 校验 `low < high`，且 `rs_mode != off` 时与 `tis_mode` 可同时开（正交）。
- `recompute_old_logprob: str = field(default="off")` —— `"off" | "on"`。`off`=bypass（沿用 vrl 现状，用 rollout 记录的 `old_log_prob`，省 replay-as-old forward）；`on`=保留位（vrl 当前无 recompute 路径，本 sprint **不实现** recompute，仅把这个开关与校验留好，并在 doc 注明 `on` 为 未验证 未实现，避免 no-op 旋钮）。

> 注意架构卫生（AGENTS.md）：不要为 RS 新建一个 `RolloutRejectionConfig` 单独文件——RS 与 TIS 是同一个「rollout→replay 精度修正」概念的两面，verl-omni 也把它们放在同一个 `rollout_correction` 块。落在已有的 `PrecisionCorrectionConfig`，复用 `trainer.precision_correction` 这一个注入点。

### 4.2 RS 判据函数（与 `apply_truncated_importance_weight` 并列，同一文件）

在 `logprob_mismatch.py` 加 `apply_rejection_sample_mask(log_ratio, config) -> keep_mask`：
- 入参是 **log-ratio**（`fresh_log_prob - old_log_prob`，即 `signals.log_prob - old_log_prob`），不是 IS 权重——这是 RS 与 TIS 的判据差异，§1 已说明。
- `seq_mean_k1`：对序列轴做 mean，落在 `[low, high]` 外则整段 keep=0。
- `seq_max_k1`：取序列轴上 `|log_ratio|` 最大的那步判带（最严格，单步 outlier 即拒整段）。
- 连续 GRPO 的 `log_prob` 是 per-sample（`continuous.py`，已是序列聚合），此时 seq_mean / seq_max 退化为同一判据——与 doc「bypass 模式下 (B,1) 各模式等价」一致（`docs/algo/rollout_correction.md:104`）；token GRPO 的 `(B, L)` 才区分 mean/max。

### 4.3 算法侧消费（token.py / continuous.py 各加 ~6 行，与 TIS 对称）

在两处现有 TIS 调用点之后，把 RS keep-mask 与 TIS keep-mask 相乘后并入 `eff_mask` / 分母：
- `continuous.py:98` 之后：`rs_keep = apply_rejection_sample_mask(signals.log_prob - old_log_probs, pc)`，与 `tis_keep` 合并；policy_loss 分母同时计入。
- `token.py:60`/`:61` 之后：`eff_mask = mask * (tis_keep or 1) * (rs_keep or 1)`。
- 新增 metric：`rs_seq_masked_fraction = (1 - rs_keep.mean())`，挂进 `TrainStepMetrics`（与现有 `tis_clip_fraction` 对称，`continuous.py:159`）。复用 §1 已有的 `compute_logprob_mismatch_stats` 报 `mismatch_kl/k3_kl`，不另算 KL。

### 4.4 bypass 与 drift guard 的联动（不改 guard，只在 doc/config 约束）

bypass（`recompute_old_logprob=off`，默认）下，前提是「drift 有界」。vrl 的 `precision_drift_guard`（`precision_guard.py:75`，`auto` 模式在 rollout≠compute 精度时自动 `fail`）已经守这个前提。本 sprint 只需在 config doc 里写明组合契约：**fp8/bf16 rollout + bypass 时，必须开 drift guard（auto/fail）+ RS（seq_mean_k1）**——guard 在训练第一步前查 parity、RS 在每步 loss 里兜住越界轨迹。两者读同一个 `compute_logprob_mismatch_stats`，判据不漂移。

## 5. 验证（finishing criteria，落地后必须跑）

1. **单测（CPU 可跑，参照 verl-omni `tests/trainer/diffusion/test_rollout_correction_on_cpu.py`）**：
   - `seq_mean_k1` 带 `[ln0.5, ln2.0]`：构造 mean log-ratio=0 的样本全留、mean=ln(3) 的样本全弃；`seq_max_k1` 在单步 outlier 下拒整段。
   - RS 与 TIS 同时开：keep-mask = `tis_keep * rs_keep`，分母正确剔除。
   - `rs_mode="token_k1"` 触发 `__post_init__` ValueError 并指向 seq 模式。
   - `recompute_old_logprob="on"` 报「未实现」而非静默 no-op。
2. **drift probe 复用**：用现成的 `vrl/scripts/perf/fp8_rollout_drift_probe.py`（已 import `PrecisionCorrectionConfig` + `compute_logprob_mismatch_stats`，见 `:40`、`:124`）加 RS 分支，在真实 fp8 rollout 轨迹上确认 `rs_seq_masked_fraction` 在合理带内（< ~5%，否则按 doc 收紧/退 recompute）。
3. **不回归 TIS**：`rs_mode=off` 时数值与当前 `fp8-rollout-precision-tis` 分支逐位一致（RS 完全旁路）。

## Non-Goals

- **不实现 decoupled recompute 路径**。vrl 单卡 32GB、reward+rollout 串在关键路径（NFT ~62% backward / ~35% generation），verl-omni 的 old-recompute 是为多卡/可承受 ~20% per-step 开销的场景设计的；vrl 默认 bypass 是优势，`recompute_old_logprob=on` 仅留接口位（未验证 未实现）。
- **不新建 `RolloutCorrectionConfig`/`rollout_correction.py` 并行体系**。RS 落在已有 `PrecisionCorrectionConfig` + `logprob_mismatch.py`，复用 `trainer.precision_correction` 注入点（架构卫生：不为单概念两面拆两套 config）。
- **不实现 `token_*` RS 模式**。§3 已论证短 SDE window 下 token-level 拒绝低功效，提供它会诱导误用。
- **不动 evaluator 协议 / trainer 主循环 / weight-sync**。RS 是 loss 内的 per-element 权重，diffusion 无 padding，拒绝=权重 0，无需新 mask 通路（与 verl-omni `docs/algo/rollout_correction.md:123` 同结论）。
- **不重做 drift 度量**。`compute_logprob_mismatch_stats` 已覆盖 KL/k3/ratio-dev，RS 只新增一个 `rs_seq_masked_fraction` 标量。
- **不引入 `rollout_is`/IS 权重再乘**。bypass + ppo_clip 下 PPO ratio 已充当 IS（verl-omni `losses.py:50`），vrl 的 `raw_ratio` 同构，RS 是唯一影响梯度的新增项。

## References

阅读文档：
- `/home/mingfeiguo/Desktop/verl-omni/docs/algo/rollout_correction.md` —— IS+RS 配方、bypass 语义、短 SDE window 调参表（`:39` 预设、`:78` 监控阈值、`:82` 梯度稀释、`:98`/`:104`/`:105` window=2 功效、`:111`-`:123` 接入点）

verl-omni 代码（本轮实际读过）：
- `/home/mingfeiguo/Desktop/verl-omni/verl_omni/trainer/diffusion/rollout_correction.py:63`、`:81`（bypass 一行替换）、`:89`-`:161`（decoupled IS/RS 折叠进 `rollout_is_weights`）
- `/home/mingfeiguo/Desktop/verl-omni/verl_omni/workers/utils/losses.py:50`、`:54`（ppo_clip 下只应用 RS mask、PPO ratio 充当 IS）
- `/home/mingfeiguo/Desktop/verl-omni/verl_omni/trainer/config/diffusion/actor/diffusion_actor.yaml:81`、`:84`（`rollout_rs` / `rollout_rs_threshold` config key）

vrl 代码（本轮实际读过，落地目标）：
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/logprob_mismatch.py:89`、`:110`、`:126`（`PrecisionCorrectionConfig` 四模式 + `apply_truncated_importance_weight` keep-mask）、`:40`-`:70`（`compute_logprob_mismatch_stats` 度量源）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/continuous.py:93`、`:98`、`:103`、`:151`（`raw_ratio` / TIS / 分母剔除 / mismatch 度量）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/algorithms/grpo/token.py:60`、`:61`、`:67`（token 版 TIS 折进 `eff_mask`）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/trainer.py:333`（trainer 级注入 `precision_correction`）、`:669`（`trainer.replay` 即 current-policy recompute forward）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/rollouts/evaluators/trajectory.py:182`、`:194`（`old_log_prob` 源自 rollout 记录的 trajectory segment —— vrl 已是 bypass 语义的实锤）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/online/precision_guard.py:75`、`:55`（drift guard `auto`→`fail`，与 RS 联动守 bypass 前提）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/trainers/core/types.py:256`、`/home/mingfeiguo/Desktop/wm-infra/vrl/config/schema.py:337`（`trainer.precision_correction` config 字段与 schema 暴露）
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/perf/fp8_rollout_drift_probe.py:40`、`:124`（已用 `PrecisionCorrectionConfig` 的 drift probe，RS 验证复用入口）
