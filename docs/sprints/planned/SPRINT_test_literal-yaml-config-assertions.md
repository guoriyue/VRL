# SPRINT: 测试硬抄 YAML 声明值，而非测加载/派生行为（planned）

状态：未开始（2026-06-21）。
范围：清理一簇 experiment / data 测试里**逐字复刻 YAML 声明值**的字面相等断言（`num_steps=20`、`guidance_scale=1.0`、batch 几何 `8/32/1`、`memory_fraction=0.55`、`height/width=128`、loader `prompt_manifest`、dataset path `datasets/videophy/train.txt`、LoRA `target_modules` 集合）。按本仓库已记录的 **no exact-config tests** 规则：config 是「声明」，不是「行为契约」；把声明值钉进断言只会让测试在每次调参编辑时无故变红，且不覆盖任何 loader / resolver 逻辑。**保留**这些文件里真正派生/解析的断言（`TrainerConfig` 计算出的 `gradient_accumulation_steps`、resolver 的结构化字段），它们测的是行为，不是声明。

> 优先级：medium。本 sprint 只动断言，不动任何 `vrl/` 源码、不动任何 config YAML —— YAML 永远是 single source of truth，可以自由重调。

## 0. Core Decision（先看这一段）

判定一条断言是「死字面快照」还是「真行为契约」，只有一个标准：**这个值是 config 声明的，还是代码派生/校验出来的？**

- **声明值**（直接写在某个 YAML group 里、loader 原样穿透）→ 钉它的字面量等于把声明复制进测试。调参一改 YAML，测试无故红，却没测出任何 loader/resolver bug。这正是 `registered_rollout_families() == tuple(FAMILY_REGISTRY)` 那个 canonical 反例的同类：手写一份 key 列表 = bug，因为源已经机械派生。→ **删字面断言**，或改成「从 `load_config(...).data` 派生 expected」/「断真实不变量」。
- **派生值**（`TrainerConfig.gradient_accumulation_steps` 由 `rollout_batch_size / microbatch_size / ...` 算出；resolver 的 `lifecycle.rollout.mode` 由拓扑推出；能 raise 的校验）→ 这是行为，**保留**。

三种安全替代，按场景选：

1. **依赖 load-and-validate sweep**（`test_all_experiments_load_and_validate`，`tests/config/test_load_all_experiments.py:202-224`）—— 它已对每个 experiment 跑 `load_config` + typed build，纯「能不能 parse」的覆盖它全包了。单条声明值不值得再钉。
2. **从源派生 expected**：要验某个值确实流到了某处，就 `load_config(...).data.X` 拿来对比 resolver/plan 的输出，让 YAML 留作唯一源。
3. **断真实不变量/关系**（而非 magic number）：`rollout_batch_size % n_samples_per_prompt == 0`；resolver 校验过的 `0 < rollout_gpu_memory_fraction <= 1`。

## 1. 现状实锤

涉及 2 个测试文件，共 4 处字面断言块。逐条已开文件核对行号 + 已 grep 源确认值可派生。

### 1.1 `tests/config/test_load_all_experiments.py:231-246` —— Cosmos NFT denoise/LoRA 字面快照

```python
# test_cosmos_predict25_nft_uses_paper_timestep_budget
assert cfg.sampling.num_steps == 20
assert cfg.sampling.cfg is False
assert cfg.sampling.guidance_scale == 1.0
assert cfg.actor.timestep_fraction == 0.5
assert cfg.model.use_lora is True
assert set(cfg.model.lora.target_modules) == {
    "ff.net.0.proj", "ff.net.2", "to_k", "to_out.0", "to_q", "to_v",
}
```

源全是声明 YAML，已核：
- `num_steps=20` / `guidance_scale=1.0` / `cfg=false` 逐字来自 denoise group `configs/sampling/denoise/20_step_no_cfg.yaml`（该文件三行就是这三个值），experiment 通过 `defaults: - /sampling/denoise/20_step_no_cfg` 引入（`online_nft_kling_video_reward.yaml:8`）。
- `timestep_fraction=0.5` 直接写在 `online_nft_kling_video_reward.yaml:41`。
- `target_modules` 集合复制自 cosmos LoRA model group 的声明。

这些都不是 typed/派生契约，是纯声明。`use_lora is True` 与 `num_steps` 等同属声明，但 `cfg is False` 与 `guidance_scale == 1.0` 的耦合**代码里并未强制**（grep `vrl/` 无「cfg False ⇒ guidance_scale==1.0」校验，CFG 分支判断是运行期 `guidance_scale > 1.0`，见 `vrl/models/diffusion/cosmos/anima/model.py:187`）—— 所以这里没有可断的真不变量，直接删。

### 1.2 `tests/config/test_load_all_experiments.py:253-259` —— rollout batch 几何字面快照

```python
# test_cosmos_predict25_kling_reward_uses_paper_rl_batch
assert cfg.rollout.n_samples_per_prompt == 8
assert cfg.rollout.rollout_batch_size == 32
assert cfg.rollout.sample_batch_size == 1
assert cfg.rollout.microbatch_size == 1
```

四个值逐字来自 `online_nft_kling_video_reward.yaml:51,52,59,55`（`n_samples_per_prompt:8`、`rollout_batch_size:32`、`sample_batch_size:1`、`microbatch_size:1`），纯声明，调参即红。

**同一函数 `:262-272` 的 `gradient_accumulation_steps==32` 断言是合法的，必须留** —— 它构造 `TrainerConfig(...)` 并断其**派生**出的 `gradient_accumulation_steps`，测的是 `rollout_batch_size / microbatch_size` 这条派生逻辑，不是 YAML 字面。

### 1.3 `tests/config/test_load_all_experiments.py:322-327` —— single-GPU async debug recipe 字面快照

```python
# test_sd35_single_gpu_async_debug_uses_persistent_colocated_rollout
resolved = resolve_distributed_resources(cfg)
assert resolved.rollout_gpu_memory_fraction is not None   # ← 留：真不变量
assert resolved.rollout_gpu_memory_fraction == 0.55       # ← 删：声明字面
assert resolved.lifecycle.rollout.mode == "resident"      # ← 留：resolver 派生
assert cfg.rollout.rollout_batch_size == 2                # ← 删
assert cfg.rollout.sample_batch_size == 1                 # ← 删
assert cfg.sampling.height == 128                         # ← 删
assert cfg.sampling.width == 128                          # ← 删
```

源已核：`height/width=128`（`online_grpo_ocr_single_gpu_async_debug.yaml:22-23`）、`memory_fraction=0.55`（`:43`）、`rollout_batch_size:2`/`sample_batch_size:1`（`:57-58`）全是声明。

关键：`memory_fraction` 在 resolver 里是**原样穿透**（`vrl/ray/resources.py:301` `gpu_memory_fraction = config.rollout_gpu_memory_fraction`，`:354` 直接塞进 `Resolved*`），所以 `resolved.rollout_gpu_memory_fraction == 0.55` 仍只是钉声明值，没测到 resolver 逻辑。resolver **真正做的事**是校验区间（`:302` `if gpu_memory_fraction is not None and not 0.0 < gpu_memory_fraction <= 1.0: raise`）—— 这才是该断的不变量。

**本函数其余断言全留**：`schedule_mode=="continuous"`、`require_separate_gpus is False`、`max_stale_policy_versions==1`、`allow_overlap is True`、`rollout.gpu_pool=="trainer"`、`resolved.lifecycle.rollout.mode=="resident"`、`rollout_gpu_memory_fraction is not None` —— 它们测的是结构/resolver 派生（尤其 `lifecycle.rollout.mode` 是从拓扑推出的，见 `resources.py:165-169`）。

### 1.4 `tests/data/test_setup.py:167-176` —— videophy dataset loader/path 字面快照

```python
# test_for_experiment_resolves_real_wan_experiment
setup.main(["for-experiment", "diffusion/wan_2_1/online_grpo_kling_video_reward"])
out = json.loads(capsys.readouterr().out)
assert out["experiment"] == "diffusion/wan_2_1/online_grpo_kling_video_reward"  # ← 留：调用参数回显
assert out["loader"] == "prompt_manifest"                                       # ← 派生自 config，别再字面钉
assert out["ready"] is True                                                     # ← 留：resolver 行为
assert any(step["path"] == "datasets/videophy/train.txt" for step in out["steps"])  # ← 派生自 config
```

`loader=="prompt_manifest"` 与 `path=="datasets/videophy/train.txt"` 逐字来自 `configs/dataset/videophy.yaml`（`data.loader: prompt_manifest`、`data.manifest: datasets/videophy/train.txt`），经 experiment 的 `defaults: - /dataset/videophy`（`online_grpo_kling_video_reward.yaml:10`）引入。dataset path 一旦改名/搬家，测试就红，但 resolver 行为没变。

该测试的**真契约**是：`resolve_experiment_dataset_plan` 能把一个真实 shipped experiment 解析为 `ready=True` 且 steps 齐全。已核 `vrl/scripts/data/bootstrap.py:120-130`：`_cmd_for_experiment` 自己就是 `load_config(...).data.get("loader"/"manifest")` 再喂给 resolver —— 所以 expected 完全可以从同一个 `load_config(...).data` 派生，而不是再抄一遍 YAML 字符串。

## 2. 落地方案

通用模式：**声明值的字面相等断言 → 删，或 expected 从 `load_config(...).data` 派生，或断真实不变量。派生/resolver 断言原样保留。**

### A. `:231-246`（Cosmos denoise/LoRA）—— 删字面块，保 sweep 覆盖

denoise / guidance / target_modules 没有可断的代码不变量，删掉整段字面相等。`num_steps>=window hi` 这类真耦合（若存在）才值得断，但这里不涉及；`test_all_experiments_load_and_validate` 已覆盖 parse。

BEFORE：

```python
assert cfg.sampling.num_steps == 20
assert cfg.sampling.cfg is False
assert cfg.sampling.guidance_scale == 1.0
assert cfg.actor.timestep_fraction == 0.5
assert cfg.model.use_lora is True
assert set(cfg.model.lora.target_modules) == {
    "ff.net.0.proj", "ff.net.2", "to_k", "to_out.0", "to_q", "to_v",
}
```

AFTER（只保留 LoRA 开关这个结构事实 + 真不变量「开了 LoRA 必有非空 target_modules」；删掉所有声明字面）：

```python
# Denoise budget (num_steps / cfg / guidance_scale) and the exact LoRA
# target_modules set are declarative YAML (sampling/denoise/20_step_no_cfg +
# the cosmos LoRA model group). Per the no-exact-config rule they are free to
# be retuned; load+validate coverage lives in test_all_experiments_load_and_validate.
assert cfg.model.use_lora is True
# Real invariant, not a literal: enabling LoRA must declare which modules to wrap.
assert cfg.model.lora.target_modules  # non-empty when use_lora is True
```

> 若团队坚持要一个数值锚，唯一有意义的是「cfg 关 ⇒ 不做 CFG」这类**代码强制**的耦合 —— 但 grep 确认 `vrl/` 无此校验（CFG 由运行期 `guidance_scale > 1.0` 决定），故不引入伪不变量，直接删。

### B. `:253-259`（rollout batch 几何）—— 删四条字面，留派生 `gradient_accumulation_steps`，可选加关系不变量

BEFORE：

```python
assert cfg.rollout.n_samples_per_prompt == 8
assert cfg.rollout.rollout_batch_size == 32
assert cfg.rollout.sample_batch_size == 1
assert cfg.rollout.microbatch_size == 1
```

AFTER（删字面；如要保留「paper 几何」意图，断关系而非 magic number；`gradient_accumulation_steps` 派生断言不动）：

```python
# Batch geometry (n_samples_per_prompt / rollout_batch_size / sample_batch_size /
# microbatch_size) is declarative YAML a tuner is free to change. Assert the real
# coupling instead of pinning the paper's magic numbers.
assert cfg.rollout.rollout_batch_size % cfg.rollout.n_samples_per_prompt == 0
# (unchanged below) gradient_accumulation_steps is a DERIVED TrainerConfig value —
# this stays: it tests the rollout_batch_size / microbatch_size derivation.
derived = TrainerConfig(
    optim=OptimConfig(lr=1e-4),
    n_samples_per_prompt=cfg.rollout.n_samples_per_prompt,
    rollout_batch_size=cfg.rollout.rollout_batch_size,
    microbatch_size=cfg.rollout.microbatch_size,
    timestep_fraction=0.5,
    total_epochs=1,
    output_dir="x",
    drop_zero_advantage=False,
)
assert derived.gradient_accumulation_steps == cfg.rollout.rollout_batch_size // cfg.rollout.microbatch_size
```

> 注意 AFTER 把 `== 32` 也改成 `== rollout_batch_size // microbatch_size`：原 `== 32` 同样是把声明值算出来的字面快照，断派生**公式**比断派生**字面**更稳。若想保守，保留 `== 32` 亦可（它毕竟测了派生路径），但删掉前四条声明字面是本 sprint 的硬要求。

### C. `:322-327`（async debug recipe）—— 删 `0.55/2/1/128` 字面，留 resolver 派生 + 区间不变量

BEFORE：

```python
resolved = resolve_distributed_resources(cfg)
assert resolved.rollout_gpu_memory_fraction is not None
assert resolved.rollout_gpu_memory_fraction == 0.55
assert resolved.lifecycle.rollout.mode == "resident"
assert cfg.rollout.rollout_batch_size == 2
assert cfg.rollout.sample_batch_size == 1
assert cfg.sampling.height == 128
assert cfg.sampling.width == 128
```

AFTER（保留 resolver 结构/派生断言；`memory_fraction` 改断 resolver 真正校验的区间；删 batch/分辨率声明字面）：

```python
resolved = resolve_distributed_resources(cfg)
# memory_fraction passes through resolve_distributed_resources unchanged, so
# pinning == 0.55 only echoes the YAML. Assert the resolver's real invariant
# instead — it validates 0 < fraction <= 1 (vrl/ray/resources.py:302).
assert resolved.rollout_gpu_memory_fraction is not None
assert 0.0 < resolved.rollout_gpu_memory_fraction <= 1.0
# resident lifecycle is DERIVED from gpu_pool=trainer + memory_fraction (not YAML) — keep.
assert resolved.lifecycle.rollout.mode == "resident"
# height/width and the batch sizes are declarative debug-recipe YAML; load+validate
# coverage is in test_all_experiments_load_and_validate. No literal pins.
```

> 上方未列出的同函数断言（`schedule_mode` / `require_separate_gpus` / `max_stale_policy_versions` / `allow_overlap` / `rollout.gpu_pool` / `rollout_gpu_memory_fraction is not None`）原样保留 —— 它们测结构与 resolver 派生。

### D. `tests/data/test_setup.py:167-176`（videophy）—— expected 从 `load_config(...).data` 派生

BEFORE：

```python
setup.main(["for-experiment", "diffusion/wan_2_1/online_grpo_kling_video_reward"])
out = json.loads(capsys.readouterr().out)
assert out["experiment"] == "diffusion/wan_2_1/online_grpo_kling_video_reward"
assert out["loader"] == "prompt_manifest"
assert out["ready"] is True
assert any(step["path"] == "datasets/videophy/train.txt" for step in out["steps"])
```

AFTER（loader/path 从 experiment 加载后的 config 派生，YAML 留作唯一源；resolver 行为不变量原样断）：

```python
from vrl.config.loading import load_config

experiment = "diffusion/wan_2_1/online_grpo_kling_video_reward"
setup.main(["for-experiment", experiment])
out = json.loads(capsys.readouterr().out)

# Derive expected loader/manifest from the same config the resolver loads, so the
# test tracks the dataset group instead of re-typing its YAML strings.
data = load_config(f"experiment/{experiment}").data
assert out["experiment"] == experiment
assert out["loader"] == data.loader
# Resolver behavior contract (the real point of the test), not a config literal:
assert out["ready"] is True
assert all(step["present"] and step["complete"] and step["get"] == "" for step in out["steps"])
assert any(step["path"] == data.manifest for step in out["steps"])
```

> `out["steps"]` 的结构（`present`/`complete`/`get`）已核于 `vrl/scripts/data/bootstrap.py:50-64`：ready 时每个 step `complete=True`、`get==""`。断这个结构不变量比断单条 path 字面更能护住 resolver 行为。

## 3. 验证（finishing criteria）

```bash
cd /home/mingfeiguo/Desktop/wm-infra

# 1) 声明字面已从这两个文件消失（不应再有这些 magic 值的相等断言）
grep -nE 'num_steps == 20|guidance_scale == 1\.0|cfg\.sampling\.(height|width) == 128|rollout_gpu_memory_fraction == 0\.55|rollout_batch_size == (2|32)|n_samples_per_prompt == 8' \
  tests/config/test_load_all_experiments.py    # 期望：无输出
grep -nE '"prompt_manifest"|datasets/videophy/train\.txt' \
  tests/data/test_setup.py                       # 期望：无输出（已改为派生）

# 2) 派生/resolver 断言仍在（不得误删）
grep -n 'gradient_accumulation_steps' tests/config/test_load_all_experiments.py   # 期望：仍命中
grep -n 'lifecycle.rollout.mode == "resident"\|rollout_gpu_memory_fraction is not None' \
  tests/config/test_load_all_experiments.py                                       # 期望：仍命中

# 3) 测试全绿
pytest tests/config/test_load_all_experiments.py -q
pytest tests/data/test_setup.py -q

# 4) 零回归
pytest tests/config tests/data -q
```

finishing 标准：上面 grep #1 零命中、#2 仍命中，pytest 全绿。改 YAML 调参（如把 `num_steps` 调成 16、把 `memory_fraction` 调成 0.6）后，这些测试**不应**再红（手动抽验一条即可）。

## 4. 非目标 / Non-Goals

- **不动任何 `vrl/` 源码、不动任何 config YAML** —— YAML 是 single source of truth，本 sprint 只调整测试断言。
- **不删派生/resolver 断言**：`gradient_accumulation_steps`（派生）、`lifecycle.rollout.mode`、`schedule_mode`、`gpu_pool`、`allow_overlap` 等结构/派生检查全部保留。
- **不引入伪不变量**：`cfg is False ⇒ guidance_scale == 1.0` 在代码里并无强制（CFG 由运行期 `guidance_scale > 1.0` 决定），故 §A 直接删而非编一个假耦合断言。
- **不扩展到本 sprint 之外的 frozen-snapshot 簇**：registry key-list、protocol method tuple、reward score-key map、scheduler-class 字符串等同类问题归属各自的 `SPRINT_test_*`，本 sprint 只收口「字面 YAML 声明值」这一类，限于上述 2 个文件 4 处。

## References

- `tests/config/test_load_all_experiments.py:202-224`（load-and-validate sweep，依赖项）、`:231-246`、`:253-272`、`:305-327`
- `tests/data/test_setup.py:167-176`
- `configs/sampling/denoise/20_step_no_cfg.yaml`（`num_steps/guidance_scale/cfg` 声明源）
- `configs/experiment/diffusion/cosmos_predict2_5/online_nft_kling_video_reward.yaml:8,41,51,52,55,59`
- `configs/experiment/diffusion/sd3_5/online_grpo_ocr_single_gpu_async_debug.yaml:22-23,43,57-58`
- `configs/dataset/videophy.yaml`、`configs/experiment/diffusion/wan_2_1/online_grpo_kling_video_reward.yaml:10`（`defaults: - /dataset/videophy`）
- `vrl/ray/resources.py:301-302,354`（memory_fraction 原样穿透 + `0 < x <= 1` 校验）、`:165-169`（resident lifecycle 派生）
- `vrl/scripts/data/bootstrap.py:30-80,120-130`（`resolve_experiment_dataset_plan` + `_cmd_for_experiment` 从 `load_config(...).data` 取 loader/manifest）
- `vrl/config/loading.py:119-160`（`load_config` 返回 `DictConfig`，`.data` 可派生 expected）
- 关联：[[SPRINT_test_frozen-registry-snapshots]]（registry key-list 同类）、[[SPRINT_test_duplicated-default-constants]]（protocol/dataclass 默认值同类）
