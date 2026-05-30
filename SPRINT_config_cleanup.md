# Sprint: Config directory cleanup

## 背景

`configs/` 已经增长到 90+ 个文件，问题不是单纯“文件多”，而是有几类配置会误导读者：

1. **真正 dead / deprecated 的配置**：没有实验引用、没有文档入口，或者对应实现已经不存在。
2. **看起来可用但实际不可用的 orphan 配置**：例如 reward 实现已经删除，但 config 还留着。
3. **机械 sampling preset 的潜在过度拆分**：部分小文件确实只有数字，但是否 inline 需要逐个判断，不能只按引用次数决定。

这次 sprint 不追求最大化删除数量。目标是删掉会误导人的配置，同时保留清晰的配置边界。

## 目标

- 删除明确 dead/deprecated 的配置。
- 保持所有现有 experiment resolved config 不变。
- 不破坏 `dataset/`、`reward/`、`model/`、`base/` 这些 config group 的架构边界。
- 更新所有文档引用，避免删除后 README 或 docs 里的命令失效。

## Scope In

### 1. 删除真正 dead/deprecated configs

| File | Action | Reason |
|---|---|---|
| `configs/base/distributed/ray_rollout_single_gpu.yaml` | Delete | Deprecated compatibility preset. Current recipes use `ray_rollout_colocated_single_gpu.yaml`. |
| `configs/reward/anime_anatomy_structure.yaml` | Delete | Orphan config. The reward implementation and registry entry are gone; loading this reward would fail. |
| `configs/sampling/denoise/30_step_cfg_4_5.yaml` | Delete if no docs/CLI references remain | No active experiment references it. It is a mechanical denoise preset, not a model/reward/data boundary. |
| `configs/sampling/denoise/50_step_cfg_5.yaml` | Delete if no docs/CLI references remain | Same as above. |

Before deleting each file, run repo-wide reference checks across `configs/`, `README.md`, `docs/`, `scripts/`, `tests/`, and sprint docs. If a public doc still references the file, update the doc or keep the config.

### 2. Update docs for deleted names

If `README.md` or `docs/` mention a deleted config, update the command to the current supported preset.

Examples:

- Replace deprecated single-GPU references with `ray_rollout_colocated_single_gpu`.
- Remove references to `anime_anatomy_structure`.
- Do not delete documented public presets unless the doc is updated in the same change.

### 3. Add resolved-config parity verification

Before deleting anything, snapshot resolved configs for all experiments:

```bash
python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
from vrl.config.loading import load_config

root = Path("configs/experiment")
out = Path("/tmp/wm_config_before")
out.mkdir(parents=True, exist_ok=True)
for path in sorted(root.rglob("*.yaml")):
    name = path.relative_to(root).with_suffix("").as_posix()
    cfg = load_config(f"experiment/{name}")
    target = out / f"{name.replace('/', '__')}.yaml"
    target.write_text(OmegaConf.to_yaml(cfg, resolve=True))
PY
```

After cleanup, produce `/tmp/wm_config_after` the same way and compare:

```bash
diff -ru /tmp/wm_config_before /tmp/wm_config_after
```

For this sprint, the diff should be empty for active experiments.

## Scope Out

### Keep dataset configs separate

Do **not** inline these, even if currently single-use:

| File | Why keep |
|---|---|
| `configs/dataset/anime_anatomy.yaml` | Dataset boundary: manifest paths, metadata schema, and sampler policy belong in `dataset/`. |
| `configs/dataset/anime_safety_stress.yaml` | Same. Keeps anime safety data reusable and grepable. |
| `configs/dataset/geneval.yaml` | GenEval metadata schema is a domain boundary and is covered by config tests. |
| `configs/dataset/pickscore_sfw.yaml` | PickScore SFW prompt source is a dataset boundary and is covered by config tests. |
| `configs/dataset/pickapic_v2.yaml` | DPO dataset/preprocessing preset; plausible future reuse. |
| `configs/dataset/video_world_v2w.yaml` | Video pipeline specifics: media type, conditioning, and source report. |

Current tests intentionally enforce this boundary:

```python
if "data" in raw:
    inline_data.append(path.relative_to(CONFIGS_ROOT).as_posix())
assert inline_data == []
```

### Keep reward configs separate

Do **not** inline these reward building blocks:

| File | Why keep |
|---|---|
| `configs/reward/geneval.yaml` | Reward adapter boundary; actual evaluator import path and artifact dirs belong in reward config. |
| `configs/reward/nsfw_safety.yaml` | Safety classifier defaults are reusable and audited in tests. |
| `configs/reward/pickscore.yaml` | Model/processor defaults are reward implementation defaults, not experiment prose. |
| `configs/reward/claude_anatomy.yaml` | Multi-line CLI command array + external rubric file. |
| `configs/reward/codex_image_qa.yaml` | Multi-line CLI + inline scoring rubric. |
| `configs/reward/videocon_physics.yaml` | Includes distributed reward worker resource config; infrastructure concern. |

Current tests also enforce single-reward config building blocks:

```python
components = raw.get("reward", {}).get("components", {})
if len(components) != 1:
    offenders.append(...)
assert offenders == []
```

### Keep model and infrastructure presets that represent real boundaries

| File | Why keep |
|---|---|
| `configs/model/diffusion/wan_2_1/14b.yaml` | Model-size boundary. Even without an active experiment, 14B is a real Wan variant and useful for CLI/recipe comparison. |
| `configs/model/ar/nextstep_1/1_1.yaml` | Architecture config: LoRA target modules and freeze policy. |
| `configs/base/distributed/ray_rollout.yaml` | Public infrastructure preset referenced by docs/CLI examples. Not dead just because no experiment imports it. |
| `configs/base/distributed/ray_rollout_cross_node.yaml` | Infrastructure preset; expected to grow with cross-node experiments. |
| `configs/base/rollout/orchestration/continuous.yaml` | Orchestration mode base; keep as an explicit boundary. |
| `configs/recipe/offline/diffusion_dpo.yaml` | Algorithmic recipe boundary. |
| `configs/recipe/online/ar_continuous_token_grpo.yaml` | Algorithmic recipe boundary. |
| `configs/recipe/online/ar_discrete_token_grpo.yaml` | Algorithmic recipe boundary. |
| `configs/profile/one_epoch/r1_codex_qa.yaml` | Actively referenced by R1 config and config tests. |

### Keep Wan 480p video sampling preset for now

Do **not** delete `configs/sampling/video/480p_33f.yaml` in this sprint.

Reason: Wan 2.1 1.3B quality discussion and official guidance make 480p a real evaluation boundary, not just an unused number file. It can be deleted later only if we decide the repo intentionally supports only 240p video configs.

## Optional Follow-Up Sprint

If we still want to reduce config count after this cleanup, open a separate sprint for **sampling preset consolidation only**.

Candidate files to evaluate there:

| File | Decision needed |
|---|---|
| `configs/sampling/ar/continuous_image_256_1024tok.yaml` | Keep if AR continuous presets are expected to stay comparable across experiments. |
| `configs/sampling/ar/discrete_image_384_576tok.yaml` | Keep if AR discrete presets are expected to stay comparable across experiments. |
| `configs/sampling/denoise/10_step_no_cfg.yaml` | Possible inline candidate if it remains single-use and no docs reference it. |
| `configs/sampling/image/896x1152.yaml` | Possible inline candidate, but keep if Anima full-body resolution remains a recurring eval boundary. |

Do not combine this optional sampling cleanup with dataset/reward/model cleanup.

## Non-Goals

- Do not inline `data:` into experiment YAMLs.
- Do not inline reward model defaults into experiment YAMLs.
- Do not delete model-size presets only because no current experiment imports them.
- Do not optimize for raw file count over grepability and stable config boundaries.
- Do not change Python code unless a test needs to be adjusted because a documented architecture rule intentionally changed. This sprint should not change those rules.

## Critical Files To Read

- `tests/config/test_load_all_experiments.py`
- `tests/config/test_prompt_dataset_configs.py`
- `README.md`
- `docs/sprints/SPRINT_cross_node_rollout.md`
- `SPRINT_reward_consolidation.md`
- Every config file listed under Scope In before deleting it.

## Verification

Use `rg`, not `grep`, and search docs as well as configs.

```bash
# No dangling references to deleted names.
rg -n "ray_rollout_single_gpu|anime_anatomy_structure|30_step_cfg_4_5|50_step_cfg_5" \
  configs README.md docs scripts tests SPRINT_*.md

# Config loading and schema validation.
python -m pytest tests/config/ -x -q

# Prompt dataset/reward boundaries still hold.
python -m pytest tests/config/test_prompt_dataset_configs.py -q

# Resolved active experiments are unchanged.
diff -ru /tmp/wm_config_before /tmp/wm_config_after
```

Expected result:

- Deleted files have zero live references.
- All config tests pass.
- Resolved active experiment configs are unchanged.
