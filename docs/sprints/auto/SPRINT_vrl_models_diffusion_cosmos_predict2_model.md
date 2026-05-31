# SPRINT(auto): vrl/models/diffusion/cosmos/predict2/model.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/diffusion/cosmos/predict2/model.py` (694 LOC)
角色判定: core
结论: improve

## 0. 一句话
文件是真核心（Cosmos Predict2 V2W 的 diffusion model），唯一可改点是 `restore_eval_state` 在 base 与 `CosmosPredict2ReplayModel` 之间几乎逐字重复，只差 scheduler 取值来源，应提到一个 helper 消除复制。

## 1. 现状（读代码得出）
`CosmosPredict2Model(DiffusionModelBase)` 实现完整 RL 路径：`from_spec`（加载 pipeline + patch safety checker）、`apply_lora`、`encode_prompt`、`prepare_sampling`（构造 Video2World 6-tuple + padding mask）、`forward_step`（走共享 `DiffusionBackboneCaller` + `CosmosPredict2DiffusionBackboneRunner`）、`export_*`/`restore_eval_state`/`replay_forward`、`decode_latents`。被 registry、train、多个 parity 测试引用，是核心资产。

`CosmosPredict2ReplayModel(CosmosPredict2Model)` 是只含 transformer+scheduler 的 replay 子类，把 pipeline-only 能力全部改成 `raise RuntimeError`，由 `runtime.build_cosmos_predict2_replay_runtime_bundle` 装配，被 `train.py:78` 与 `tests/models/test_minimal_replay_runtime_wiring.py` 使用——非死代码。

问题：两处 `restore_eval_state` 实质相同。base 版 (model.py:452-497) 与 replay 版 (model.py:622-655) 构造的 `CosmosPredict2SamplingState` 字段完全一致，唯一差异是 timesteps/scheduler 的来源：

base：
```python
timesteps=self.pipeline.scheduler.timesteps,
scheduler=self.pipeline.scheduler,
```
replay：
```python
timesteps=self.scheduler.timesteps,
scheduler=self.scheduler,
```
其余 ~16 行（prompt_embeds / init_latents / 各 mask / indicator / fps / sigma_conditioning 的取法）逐字重复。

## 2. 质疑点 / 改进机会
- **重复构造逻辑**：同一个 dataclass 的还原逻辑写了两遍，差异仅在 scheduler 来源。base 用 `self.pipeline.scheduler` 而非 `self.scheduler` 是多余的——base 的 `self.scheduler` property (model.py:179) 本就返回 `self.pipeline.scheduler`。也就是说两份代码若都改用 `self.scheduler`，body 完全一致，可合并为一份。证据：model.py:452-497 vs 622-655；`scheduler` property at model.py:178-180。
- **腐烂风险**：`CosmosPredict2SamplingState` 之后加/改字段时，必须同时改两处，否则 replay 路径会悄悄漏字段。

## 3. 建议动作
- 让 base `restore_eval_state` 改用 `self.scheduler`（等价，因为 property 已转发 pipeline.scheduler），然后从 `CosmosPredict2ReplayModel` 删除被复制的 `restore_eval_state`，让它继承 base 版。这样两类共用一份还原逻辑，且差异点（scheduler 来源）天然由各自的 `scheduler` property 决定。
- 注意跨家族一致性：sibling `predict2_5/model.py` 在 base(429) 与 ReplayModel(640) 也有同款重复。若改，应两家族一起改以保持一致；若团队更看重两家族“逐字镜像”，则维持现状并改判 keep-justified。该取舍留给 reviewer。

## 4. 不动什么 / 为什么不是过度清理
- 不动 `CosmosPredict2ReplayModel` 的 `raise RuntimeError` 桩方法（encode_prompt/prepare_sampling/decode_latents/pipeline）：它们是刻意的能力边界（replay model 不拥有 pipeline-only 模块），是 fail-fast 契约，保留。
- 不动 `_align_replay_tensor` / `_replay_tensor` / `_shared_replay_tensor`（model.py:662-686）：它们与 predict2_5 同名同形，是跨家族一致的 replay 张量对齐 helper，移除真实复杂度，保留。
- 不动 `CosmosPredict2SamplingState`（dataclass）与 `from_spec` 里的 safety-checker passthrough patch：都是真实业务边界。
- `family = "cosmos-predict2-diffusers-v2w"` 等 ALL_CAPS/字面量是协议/family 名边界，保留。

## 5. 验证
- `pytest tests/models/test_cosmos_predict2_diffusion_backbone_parity.py tests/models/test_diffusion_decode_layout_parity.py -q` 确认 forward/decode 数值不变。
- `pytest tests/models/test_minimal_replay_runtime_wiring.py -q` 确认 replay model 还原路径仍 work。
- `grep -n "restore_eval_state" vrl/models/diffusion/cosmos/predict2/model.py` 改后应只剩 base 一处定义。
