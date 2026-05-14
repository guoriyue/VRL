# SPRINT：Rollout Config Projection Cleanup

## 0. Core Decision

最终设计收敛成三条边界：

```text
YAML owns values/defaults.
Rollout registry owns wiring only.
Schema owns generic YAML projection.
```

也就是说，registry 不再保存每个 family/model 的采样字段白名单。它只说明某个 rollout family 怎么接到 runtime：

- family/task/aliases
- executor/runtime builder/runtime spec extractor
- collector kind/request prefix/return artifacts/metadata wrapper
- gatherer/capability metadata

采样参数来自 YAML 的 `sampling` 和 `rollout` section。传给 engine request 的字段由 `RolloutSettings.request_sampling()` 从已解析 YAML 自动投影。

## 1. What We Removed

删除 family-specific Python 默认值：

```python
num_steps = 35
guidance_scale = 7.0
height = 704
width = 1280
num_frames = 93
fps = 16
sample_batch_size = 8
n_samples_per_prompt = 4
```

这些值只应该存在于：

```text
configs/sampling/*.yaml
configs/base/rollout/*.yaml
configs/base/algorithm/*.yaml
configs/experiment/*.yaml
```

删除 rollout registry 里的字段合同：

- `RolloutContract`
- `sampling_fields`
- `DIFFUSION_COMMON_SAMPLING_FIELDS`
- `DIFFUSION_VIDEO_SAMPLING_FIELDS`
- `_janus_contract()`
- `NEXTSTEP_CONTRACT`

删除 request builder 对 registry 字段白名单的依赖：

```python
RolloutEngineRequestBuilder(..., sampling_fields=(...))
```

现在 request builder 只接收 resolved settings，然后调用：

```python
settings.request_sampling()
```

## 2. What Stays In Code

`vrl/rollouts/family_registry.py` 仍然需要存在，但它不再描述配置字段。它只做 wiring：

```python
RolloutFamilyEntry(
    family="janus_pro",
    task="ar_t2i",
    aliases=("janus", "janus_pro_1b"),
    collector=CollectorMetadata(
        kind="ar_discrete",
        request_prefix="janus_pro",
        return_artifacts=DEFAULT_RETURN_ARTIFACTS,
    ),
    executor_cls="vrl.models.families.janus_pro.runtime:JanusProPipelineExecutor",
    runtime_builder="vrl.models.families.janus_pro.runtime:build_janus_pro_runtime_bundle",
    runtime_spec_extractor="vrl.models.families.janus_pro.runtime:extract_janus_pro_runtime_spec",
    gatherer=GathererMetadata(
        import_path="vrl.models.families.janus_pro.runtime:JanusProChunkGatherer",
    ),
    capability=ar_discrete_family_capability("janus_pro", "ar_t2i"),
)
```

这些不是实验默认值，也不是 field schema。它们是 Python runtime 入口，不能从普通 YAML 自动推出来。

## 3. YAML Projection Rule

`vrl/rollouts/settings.py` 做通用投影：

- merge flat `rollout.*`
- flatten `rollout.sde.type/window_size/window_range` 到 `sde_type/sde_window_size/sde_window_range`
- merge flat `sampling.*`，让 sampling preset 覆盖 rollout 里的同名采样值
- copy `algorithm.kl_reward`
- 如果存在 SDE sampling，则派生 `return_kl = kl_reward > 0`
- copy R1 的 `final_image_policy` 和 `train_segments`

`request_sampling()` 会排除 rollout 控制字段：

```python
{
    "kl_reward",
    "n",
    "n_samples_per_prompt",
    "rollout_batch_size",
}
```

因此 `n_samples_per_prompt` 仍然用于 collector group size，但不会被塞进 engine request sampling。

## 4. Trade-Off

这版故意不再做 per-family field whitelist。

优点：

- registry 不再复制 Janus/NextStep/diffusion 的字段清单
- 新 model 加采样参数时通常只改 YAML 和 executor
- 不会再出现 “YAML 一份字段，registry 又一份字段” 的漂移

代价：

- 拼错字段不会在 registry projection 阶段被白名单拦住
- 缺少 executor 必需字段时，会在 collector/executor 消费该字段时失败

这是更合理的代价，因为现在的目标是让 YAML 成为配置 source of truth，而不是在 registry 里维护第二套 schema。

## 5. Verification

已验证：

```bash
python -m ruff check .
python -m pytest -q
```

并抽样确认这些 recipe 的 request sampling keys 来自 YAML projection：

- `configs/experiment/sd3_5_ocr_grpo.yaml`
- `configs/experiment/cosmos_predict2_2b_grpo.yaml`
- `configs/experiment/cosmos_predict2_5_2b_diffusionnft.yaml`
- `configs/experiment/janus_pro_1b_ocr_grpo.yaml`
- `configs/experiment/janus_pro_1b_r1_ocr_grpo.yaml`
- `configs/experiment/nextstep_1_ocr_grpo.yaml`
