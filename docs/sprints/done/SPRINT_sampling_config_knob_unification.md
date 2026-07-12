# SPRINT: SamplingConfig 旋钮词汇统一（CFG 强度 / 步数 / noise_level / cfg 布尔 / train_segments / final_image_policy 归属）(planned)

状态：已完成（2026-06-21）。第一档（死分支删除）+ 第二档（CFG 名统一）+ 步数 `num_flow_steps`→`num_steps` 全链路改名 + noise_level/cfg 消歧注释全部落地。

- 已落地（2026-06-21，本轮）：步数统一——核实后 `num_flow_steps` 与 cfg_scale/cfg_weight 同理，**不是上游 `generate()` 关键字**（`vrl/math/ar/flow_matching.py` 是 VRL 自有 Euler 积分器），故走全链路改名而非翻译层：`flow_matching.py` / `nextstep_1/{runner,model,runtime}.py` / schema（删 `num_flow_steps`，`num_steps` 已注册）/ `configs/sampling/ar/continuous_image_256_1024tok.yaml` / 两份 config 注释 / `tests/math/test_ar_flow_matching.py` / `tests/e2e/test_real_checkpoint_rl.py` dotlist override 全部统一为 `num_steps`。noise_level（双归属故意保留，rollout 为 canonical）与 cfg（do-CFG 布尔，cfg_scale/cfg_weight 已改名后 homonym 已基本消解）各补 schema 注释而非改名。验证：config 双 sweep（code + experiment）零 unknown-key；`pytest tests/math/test_ar_flow_matching.py tests/algorithms/test_dpo.py tests/config/ tests/models/`（315 passed）+ `tests/models/ar/ tests/generation/ tests/rollouts/`（261 passed）全绿。
- 已落地（第一档）：删 `rollout.train_segments`（collector first-present 链）、`sampling.r1.final_image_policy`（first-present 链 + SamplingConfig.r1 ConfigBlock + RootConfig 等值交叉校验），final_image_policy 收敛为 rollout 单一真源（保留 legality 校验）。删除随之失效的 `test_token_grpo_multisegment_policy_mismatch_raises`。tests/config + collector + factory 111 passed。
- 已落地（第二档，CFG 强度）：`cfg_scale`（NextStep）/ `cfg_weight`（Janus）→ **`guidance_scale` 全链路改名**，而非原计划的「YAML 表面统一 + adapter 内部保留上游名」。核实后两家的 CFG 都只流经**自有代码**（Janus 自写 `guided = uncond + s*(cond-uncond)`；NextStep 自有 `vrl/math/ar/flow_matching.flow_sample_with_logprob`），`cfg_scale`/`cfg_weight` **不是上游 `generate()` 的关键字**，所谓「上游 API parity」只是命名识别度，无硬绑定 → 故选内部一致的彻底改名（单一词汇、无翻译层）。schema 删 `cfg_scale`/`cfg_weight` 两 registry key（`guidance_scale` 已注册）。anima eval CLI 保留 `--cfg-scale` 作 `--guidance-scale` 的兼容别名。AR 配置解析 `sampling.guidance_scale`（nextstep 4.5 / janus 5.0）零 unknown-key 告警；models/rollouts/config/trainers/algorithms 全量 595 passed。`cfg` 布尔开关与 wan `guidance_scale_2` 未动。
范围：`SamplingConfig` / `RolloutConfig` 里一组「一个概念、多个拼写或多个归属」的采样旋钮 —— 同一个量在不同 model family 下用不同 key 名，或者同一个 key 在两三个 section 里各有一份；其中已确认存在**完全没有 live config 使用者的死分支**（`rollout.train_segments`、`sampling.r1.final_image_policy`）。

## 0. Core Decision（先看这一段）

`SamplingConfig` 当前把同一个采样概念按 family 方言拆成多个 key，再靠 collector 的 first-present 链路和 schema 的 cross-field 校验把它们勉强缝合在一起。本 sprint 不追求把所有 family 方言一次性铲平（CFG 强度的 `cfg_weight`/`cfg_scale` 确有上游 API parity 约束），而是分两档落地：

1. **直接删死分支（零行为风险，先做）**：
   - `rollout.train_segments` 这个 fallback 分支**没有任何 live config 使用**（live 归属是 `algorithm.train_segments` 与 `sampling.r1.train_segments`），删掉。
   - `sampling.r1.final_image_policy` 这个归属**没有任何 live config 使用**（唯一 live 归属是 `rollout.final_image_policy`，`r1` 采样 preset 里还专门写注释说明「set once in rollout」），连带 `_cross_field_validate` 里那段「两边都设了必须相等」的 12 行平等校验一起删。
2. **统一 config 表面拼写（保留上游 API 名作 adapter 内部名）**：CFG 强度统一到 `guidance_scale`，步数统一到 `num_steps`，AR runtime 在 model-build 提取层把 YAML 的统一名映射成上游 `generate()` 需要的 `cfg_scale`/`cfg_weight`/`num_flow_steps`。这一档不能 remove-outright，必须先迁移 AR 的 sampling preset YAML 再删 schema 旧 key。

`cfg` 布尔（do-CFG 开关）与 `cfg_scale`/`cfg_weight` 同前缀但语义不同，顺手在 schema 注释里点明 / 或重命名为 `do_cfg`，属低优先。

关联：[[SPRINT_precision_naming_unification]]（同类「一个概念多个拼写」治理），[[SPRINT_config_string_settings]]（config 旋钮 typing），[[SPRINT_config_unknown_key_warning]]（未知 key 告警机制，迁移期可复用做 deprecation 提示）。

## 1. 现状实锤

### 1.1 CFG 强度：三个 key 指同一个量

`vrl/config/schema.py:270-274` 把三个 key 并排声明，谁都不是 source of truth：

```python
cfg: Any = None
cfg_scale: Any = None
cfg_weight: Any = None
fps: Any = None
guidance_scale: Any = None
```

三个 family 各读一个：

- diffusion（cosmos/wan）读 `guidance_scale` —— `vrl/generation/diffusion/layout.py:96` `guidance_scale=float(sampling["guidance_scale"])`
- nextstep_1 AR 读 `cfg_scale` —— `vrl/models/ar/nextstep_1/runtime.py:245` `cfg_scale = float(sampling["cfg_scale"])`
- janus_pro AR 读 `cfg_weight` —— `vrl/models/ar/janus_pro/runtime.py:299` `cfg_weight = float(sampling.get("cfg_weight", 5.0))`

config 侧也按 family 分裂：`configs/sampling/denoise/35_step_cfg_7.yaml:4` `guidance_scale: 7.0`；`configs/sampling/ar/continuous_image_256_1024tok.yaml:8` `cfg_scale: 4.5`；`configs/sampling/ar/r1_image_384_576tok.yaml:5` `cfg_weight: 5.0`。三者最终都落到同一段 `uncond + scale*(cond-uncond)` / `do_cfg = scale > 1.0` 数学（如 `vrl/models/diffusion/cosmos/predict2_5/model.py:258` `do_cfg = guidance_scale > 1.0`）。

> load-bearing caveat：`cfg_weight` 与上游 Janus-Pro、`cfg_scale` 与上游 NextStep-1 的 `generate()` 签名一致，runtime adapter 下游可能仍需以这两个名字传参 —— 所以统一只动「YAML 表面 key」，runtime 提取层做名字映射，不动上游 model-internal attribute 名。

### 1.2 步数：`num_steps` vs `num_flow_steps`

`vrl/config/schema.py:281-283` 两个 key 并排：

```python
num_flow_steps: Any = None
num_frames: Any = None
num_steps: Any = None
```

diffusion 读 `num_steps`（`vrl/generation/diffusion/layout.py:91` `num_steps = int(sampling["num_steps"])`，config 如 `configs/sampling/denoise/35_step_cfg_7.yaml:3` `num_steps: 35`）；nextstep_1 flow family 读 `num_flow_steps`（`vrl/models/ar/nextstep_1/runtime.py:246` `num_flow_steps = int(sampling["num_flow_steps"])`，config `configs/sampling/ar/continuous_image_256_1024tok.yaml:6` `num_flow_steps: 20`）。同是「积分步数」，两个拼写。

> caveat：`num_flow_steps` 是 nextstep_1 flow-matching 全链路的既有 arg 名，rename 牵动 `math/ar/flow_matching.py` / `runner.py` / `model.py`，非纯 config 层改动。

### 1.3 noise_level：声明在 rollout 和 sampling 两个 section

`vrl/config/schema.py:230` `noise_level: float | None = None`（RolloutConfig），`vrl/config/schema.py:280` `noise_level: Any = None`（SamplingConfig）。两个 section 都有 live config 使用者：

- diffusion 走 rollout 归属：`configs/base/rollout/diffusion.yaml:8`、`configs/base/rollout/flow_matching_sde.yaml:7`、多个 experiment YAML（如 `configs/experiment/diffusion/wan_2_1/online_grpo_physics.yaml:32`）都写在 `rollout:` 下。
- AR continuous 走 sampling 归属：`configs/sampling/ar/continuous_image_256_1024tok.yaml:7` `noise_level: 1.0` 写在 `sampling:` 下。

cross-field 校验只认 rollout 归属（`vrl/config/schema.py:529-534`，token_grpo + nextstep_1 强制 `rollout.noise_level`），而 nextstep_1 runtime 又是从 sampling-merge 出来的 dict 里读（`vrl/models/ar/nextstep_1/runtime.py:247` `noise_level = float(sampling["noise_level"])`）。

> caveat：collector 用 `_merge_flat_section_values(values, cfg, "sampling")`（`vrl/rollouts/collector/config.py:56`）把 `sampling.*` 拍平进 rollout values dict，所以 nextstep_1 确实从 sampling-derived dict 读到它。删任一归属前必须确认 merge 仍然填充该值。

### 1.4 final_image_policy：sampling.r1 归属是死分支

collector 的 first-present 链路 `vrl/rollouts/collector/config.py:58-63`：

```python
_copy_first_present(
    values, cfg, "final_image_policy",
    ("rollout.final_image_policy", "sampling.r1.final_image_policy"),
)
```

schema 还在两处声明并加了平等校验 `vrl/config/schema.py:543-552`（注释自己都写「set final_image_policy in ONE place」却保留两个家 + 12 行 equality cross-check）。

**实锤：`sampling.r1.final_image_policy` 没有任何 live config 使用。** 全仓 grep 只命中 `configs/base/rollout/ar_r1.yaml:10` `final_image_policy: always_generate`（rollout 归属），而 `configs/sampling/ar/r1_image_384_576tok.yaml:10-11` 专门写注释说明「set once in rollout … not duplicated here」。janus_pro runtime 从 merge 后的 sampling dict 读（`vrl/models/ar/janus_pro/runtime.py:1018` 量级），值经 collector flat-merge 从 rollout 归属流过来。→ `sampling.r1.final_image_policy` 这个 fallback 分支 + cross-validator 平等校验是纯维护税，可删。

### 1.5 train_segments：rollout 归属是死分支

`vrl/rollouts/collector/config.py:64-73` 三元 fallback：

```python
_copy_first_present(
    values, cfg, "train_segments",
    ("rollout.train_segments", "algorithm.train_segments", "sampling.r1.train_segments"),
)
```

**实锤：`rollout.train_segments` 没有任何 live config 使用。** 全仓 grep 命中两处，都不在 rollout：`configs/base/algorithm/token_grpo_multisegment.yaml:13`（algorithm 归属，被 `vrl/scripts/common/factory.py:232` `_cfg_select(cfg, "algorithm.train_segments", {})` 直读）与 `configs/sampling/ar/r1_image_384_576tok.yaml:12`（sampling.r1 归属，经 collector + `vrl/trajectory/builders.py` 从 sampling request dict 读）。→ 三元里第一个分支 `rollout.train_segments` 是死的，可删，链路收敛为两元。

### 1.6 `cfg` 布尔与 `cfg_*` 浮点同前缀（HOMONYM，低优先）

`vrl/config/schema.py:270-272` 里 `cfg`（布尔开关）与 `cfg_scale`/`cfg_weight`（浮点强度）三连排，前缀 `cfg` 同时是「开关名」和「强度名词干」。`cfg` 布尔在 config 里确有独立使用：`configs/sampling/denoise/10_step_no_cfg.yaml:5` `cfg: false`、`configs/sampling/denoise/35_step_cfg_7.yaml:5` `cfg: true` 等 6 份 denoise preset。它在 diffusion model 里与 guidance_scale 再做一次 AND：`vrl/models/diffusion/cosmos/predict2_5/model.py:454` `do_cfg=batch_context["cfg"] and batch_context["guidance_scale"] > 1.0`，runtime 内部已用 `do_cfg` 命名。

> caveat：`cfg` 布尔 key 被 cosmos predict2 / predict2_5 / anima 的 model.py 从 batch_context dict 读（`predict2/model.py:465`、`predict2_5/model.py:454` 等），并经 `vrl/trajectory/builders.py:484` `"cfg": bool(payload.get("cfg", False))` 注入 trajectory metadata。rename YAML key 必须与这些 dict 读 lockstep，否则静默 no-op。

## 落地方案

### 第一档：删死分支（零行为风险，优先）

1. `vrl/rollouts/collector/config.py:64-73` —— `train_segments` 的 first-present 链路去掉 `"rollout.train_segments"`，收敛为 `("algorithm.train_segments", "sampling.r1.train_segments")`；保留两元注释说明 algorithm 是 multisegment dataclass 的 owner、sampling.r1 是 r1 recipe preset 归属。
2. `vrl/rollouts/collector/config.py:58-63` —— `final_image_policy` 去掉 `"sampling.r1.final_image_policy"`，收敛为单元 `("rollout.final_image_policy",)`。
3. `vrl/config/schema.py:543-552` —— 删掉 rollout-vs-sampling.r1 的平等 cross-check（`rollout_policy and sampling_policy and rollout_policy != sampling_policy` 那段），只保留 `vrl/config/schema.py:553-556` 的取值合法性检查（`{"always_generate","use_selfcheck"}`，且改为只读 `rollout.final_image_policy`）。
4. `vrl/config/schema.py:260-263` —— `SamplingConfig.r1` 的 `ConfigBlock(("final_image_policy", "train_segments"))` 收缩为 `("train_segments",)`（`final_image_policy` 不再接受 sampling.r1 归属，避免再被当合法 key 注册）。同步更新 `configs/sampling/ar/r1_image_384_576tok.yaml:10-11` 的注释（已说明在 rollout，注释保留即可）。

### 第二档：统一表面拼写（migrate-then-remove）

5. CFG 强度：在 AR model-build 提取层（`vrl/models/ar/nextstep_1/runtime.py:245`、`vrl/models/ar/janus_pro/runtime.py:299`）改为读 `sampling["guidance_scale"]`，内部仍以 `cfg_scale`/`cfg_weight` 传给上游 `generate()`。迁移 `configs/sampling/ar/continuous_image_256_1024tok.yaml:8`、`configs/sampling/ar/r1_image_384_576tok.yaml:5` 改用 `guidance_scale`。两份 YAML 迁移完成后，从 `vrl/config/schema.py:271-272` 删除 `cfg_scale`/`cfg_weight`。
6. 步数：同样把 nextstep_1 提取层（`vrl/models/ar/nextstep_1/runtime.py:246` 及其下游 `flow_matching.py`/`runner.py`/`model.py` 的 `num_flow_steps` 形参）统一读 `num_steps`；迁移 `configs/sampling/ar/continuous_image_256_1024tok.yaml:6`；最后删 `vrl/config/schema.py:281` 的 `num_flow_steps`。此项链路较长，可作为第二档的独立子任务。
7. noise_level：保留 `rollout.noise_level` 作 canonical（cross-validator 认它），在 `vrl/config/schema.py:280` 的 `SamplingConfig.noise_level` 加 `# AR families 在 sampling 段设此值，collector flat-merge 进 rollout values；与 rollout.noise_level 同一旋钮` 一行注释，把「故意双归属」说明白；或把 AR continuous 的 `noise_level` 也迁到 rollout 段后删 sampling 声明（取决于 AR rollout base 是否方便承载）。

### 低优先

8. `cfg` 布尔：在 `vrl/config/schema.py:270` 加注释点明它是 do-CFG 开关、与 `cfg_scale`/`cfg_weight` 无关；如要进一步消歧则重命名为 `do_cfg` 并 lockstep 更新 `vrl/trajectory/builders.py:484`、cosmos predict2/predict2_5/anima 的 `batch_context["cfg"]` dict 读、以及 6 份 `configs/sampling/denoise/*.yaml`。

## 验证（finishing criteria）

- 第一档每删一个分支后，跑 config-resolve（对 `configs/base/algorithm/token_grpo_multisegment.yaml`、`configs/base/rollout/ar_r1.yaml`、`configs/sampling/ar/r1_image_384_576tok.yaml` 三个真实 YAML 组合）确认 `train_segments` / `final_image_policy` 仍解析出原值，无 KeyError、无 unknown-key 告警。
- 全仓 grep 确认被删的 `rollout.train_segments`、`sampling.r1.final_image_policy` 在 `configs/` 与 `vrl/` 里零引用后再删。
- 第二档迁移后：grep `cfg_scale`/`cfg_weight`/`num_flow_steps` 在 `configs/` 内零命中（只允许出现在 runtime adapter 内部传参处），并对 AR sampling preset 跑一次 model-build 提取，确认 `guidance_scale`/`num_steps` 被正确读出。
- `pytest` 全绿（schema 解析测试、collector config 测试、trajectory builder 测试）。

## 非目标 / Non-Goals

- 不动上游 Janus-Pro / NextStep-1 的 model-internal attribute 名（`cfg_weight` / `cfg_scale` / `num_flow_steps` 作为 adapter 下游传参名按 parity 保留）。
- 不重构 collector 的 `_copy_first_present` / `_merge_flat_section_values` 机制本身（属 [[SPRINT_config_as_signatures]] / [[SPRINT_config_string_settings]] 范畴），本 sprint 只收敛它喂进去的 path 列表。
- 不重命名 precision 相关旋钮（见 [[SPRINT_precision_naming_unification]]）。
- 不把 `cfg` 布尔的 rename 列为必做项（低优先，lockstep 成本高于收益时只加注释即可）。

## References

- `vrl/config/schema.py:227-285`（RolloutConfig / SamplingConfig 声明：noise_level 双归属 230/280、final_image_policy 231-233、SamplingConfig.r1 260-263、cfg/cfg_scale/cfg_weight/guidance_scale 270-274、num_flow_steps/num_steps 281-283）
- `vrl/config/schema.py:528-556`（cross-field 校验：noise_level rollout 强制 529-534、final_image_policy 平等校验 543-556）
- `vrl/rollouts/collector/config.py:46-77`（build_rollout_config_from_cfg + first-present 链路：final_image_policy 58-63、train_segments 64-73、sampling flat-merge 56）
- `vrl/generation/diffusion/layout.py:90-96`（num_steps / guidance_scale 读取）
- `vrl/models/ar/nextstep_1/runtime.py:245-247`（cfg_scale / num_flow_steps / noise_level 读取）
- `vrl/models/ar/janus_pro/runtime.py:299`（cfg_weight 读取）
- `vrl/models/diffusion/cosmos/predict2_5/model.py:258,411,454`（do_cfg 推导 + batch_context["cfg"] AND guidance_scale）
- `vrl/models/diffusion/cosmos/predict2/model.py:465`（batch_context["cfg"] 同构）
- `vrl/trajectory/builders.py:484`（"cfg" bool 注入 trajectory metadata）
- `vrl/scripts/common/factory.py:232`（algorithm.train_segments 直读）
- `configs/sampling/denoise/35_step_cfg_7.yaml:3-5`、`configs/sampling/denoise/10_step_no_cfg.yaml:5`（num_steps / guidance_scale / cfg bool）
- `configs/sampling/ar/continuous_image_256_1024tok.yaml:6-8`（num_flow_steps / noise_level / cfg_scale）
- `configs/sampling/ar/r1_image_384_576tok.yaml:5,9-15`（cfg_weight / r1.train_segments / final_image_policy 注释）
- `configs/base/rollout/ar_r1.yaml:10`（final_image_policy 唯一 live 归属）
- `configs/base/algorithm/token_grpo_multisegment.yaml:13`（train_segments algorithm 归属）
- `configs/base/rollout/diffusion.yaml:8`、`configs/base/rollout/flow_matching_sde.yaml:7`、`configs/experiment/diffusion/wan_2_1/online_grpo_physics.yaml:32`（noise_level rollout 归属 live 使用）
