# SPRINT: Unify reward-model identity keys 与 Kling 分数词汇 (planned)

状态：planned（2026-06-20）
范围：统一 reward 模型「装载哪个 checkpoint」的三种拼法（`reward_name` / `reward_model_name` / `model_name`），拆开 `reward_name` 的「请求标签 vs HF repo id」二义，并清理 Kling 分数 dict 的原始键 dual-write、`_DEFAULT_REVISION` 冗余别名、GenEval 单值 `evaluator` 占位旋钮。

## 0. Core Decision（先看这一段）

reward 域里「指向某个 HF checkpoint」这一个值被拼成三种 key（`reward_name` / `reward_model_name` / `model_name`），而真正消费它们的代码靠一条复制了 4 次的 `worker_config.get("reward_model_name") or worker_config.get("model_name") or resolved_reward_name` `or`-链来兜底。更糟的是 `reward_name` 同时是 inference 请求里的**逻辑标签**（`RewardInferenceRequest.reward_name`，required+validated）和 video reward YAML 里的 **HF repo id**（`reward_name: KlingTeam/VideoReward@main`）——一个 key 干两件事。

落地核心：
1. **worker_config 里统一用 `reward_model_name` 作为「装载哪个模型」的唯一 key**（video reward 已经在用它，image reward 用 `model_name`）。在 `base.py._init_disk_artifact_reward` 做**一次** normalize（把 top-level `reward_name` / `model_name` 归一到 `worker_config["reward_model_name"]`），让三个 loader（base/kling/videocon）只读一个 key，删掉各自复制的 `or`-链。
2. **拆开 `reward_name` 的二义**：video reward YAML 改用 `reward_model_name: KlingTeam/VideoReward@main`，`reward_name` 回归为逻辑标签（默认 registry key）。`base.py:212` 的 `or resolved_reward_name` 兜底随之删除。配置必须**同 PR 迁移**。
3. **Kling `_normalize_scores` 只输出 public 别名**（`overall_reward/visual_quality/...`），不再 dual-write 原始 `VQ/MQ/TA/Overall`；如要留作 debug provenance，则显式标注且禁止作为 `score_key`。
4. 顺手删 `_DEFAULT_REVISION` 文件内冗余别名、删 GenEval 单值 `evaluator` 占位旋钮。

非目标：不改 `score_key` 的 typing（[[SPRINT_config_string_settings]] 已决定 LEAVE：per-reward 词汇 + 复合表达式）；不改 image reward 的 `model_name` 拼法（见 §落地方案 取舍）；不动 `parse_hf_repo_revision` 的 `repo@revision` grammar（[[SPRINT_design_smell_audit]] 已抽好）。

## 1. 现状实锤

### 1.1 同一个「装载哪个模型」三种拼法 + 复制 4 次的 or-链

`base.py` 在把 worker_config 喂给 Ray runtime 前解析：

```python
# vrl/rewards/base.py:209-214
reward_model_name = str(
    worker_config.get("reward_model_name")
    or worker_config.get("model_name")
    or resolved_reward_name
    or "",
).strip()
```

同一条 `or`-链（去掉 `resolved_reward_name` 兜底）在两个 disk loader 与两个 RewardModel `__init__` 里逐字复制：

```python
# vrl/rewards/models/kling_video_reward.py:186-190 (KlingVideoRewardModel.__init__)
self.reward_model_name = str(
    self.worker_config.get("reward_model_name")
    or self.worker_config.get("model_name")
    or "",
).strip()
```

```python
# vrl/rewards/models/kling_video_reward.py:622-626 (_resolve_model_root)
reward_model_name = str(
    worker_config.get("reward_model_name")
    or worker_config.get("model_name")
    or "",
).strip()
```

```python
# vrl/rewards/models/videocon_physics.py:250-254 (_resolve_model_root)
reward_model_name = str(
    worker_config.get("reward_model_name")
    or worker_config.get("model_name")
    or "",
).strip()
```

（videocon_physics.py:55-57 的 `VideoConPhysicsModel.__init__` 同样复制一份。）

配置侧 image reward 用 `model_name`，video reward 用 `reward_name`：

```yaml
# configs/reward/aesthetic.yaml:7
      model_name: openai/clip-vit-large-patch14
# configs/reward/pickscore.yaml:8
      model_name: yuvalkirstain/PickScore_v1
# configs/reward/kling_video_reward.yaml:19
      reward_name: KlingTeam/VideoReward@main
# configs/reward/videocon_physics.yaml:21
      reward_name: videophysics/videocon_physics@main
```

一个作者要新写 reward 配置，必须读 `base.py` 才知道这三个 key 哪个生效、`or` 优先级是什么；拼错不会报错，只会 fall through 到下一个分支。

注：[[SPRINT_design_smell_audit]] 只抽过 `parse_hf_repo_revision`（`repo@revision` 文法），**没有**碰这条三名 `or`-链——这是新增项。

### 1.2 `reward_name` 二义：请求标签 vs HF repo id

`reward_name` 在 inference 协议里是必填的逻辑标签：

```python
# vrl/rewards/inference.py:110,119-120
    reward_name: str
    ...
    if not self.reward_name:
        raise ValueError("RewardInferenceRequest.reward_name is required")
```

而 function reward 里它是友好标签（`reward_name="aesthetic"` 之类，见 base.py:230 `reward_name=resolved_reward_name`）。但 kling/videocon 的 YAML 把它写成 HF repo（`KlingTeam/VideoReward@main`），并由 base.py:212 的 `or resolved_reward_name` 把同一字符串塞进 `worker_config["reward_model_name"]` 当 model id：

```python
# vrl/rewards/base.py:217-224
if not has_model_factory and (reward_model_name or model_path):
    if reward_model_name:
        worker_config["reward_model_name"] = reward_model_name
    worker_config["model_factory"] = model_factory
    if not str(worker_config.get("reward_model_version", "")).strip():
        worker_config["reward_model_version"] = (
            reward_model_name or model_path
        )
```

一个 debug log 里看到 `request.reward_name`，无法判断它是友好 tag（`kling_video_reward`）还是可下载的 repo ref（`KlingTeam/VideoReward@main`）——两者都合法、都出现。

### 1.3 Kling 分数 dict dual-write 原始键

`_normalize_scores` 同时返回原始 model 键与 public 别名：

```python
# vrl/rewards/models/kling_video_reward.py:677-683
def _normalize_scores(raw_scores: Mapping[str, Any]) -> dict[str, float]:
    raw = {str(key): float(value) for key, value in raw_scores.items()}
    scores = dict(raw)                       # 保留 VQ/MQ/TA/Overall
    for public_key, model_key in _SCORE_KEY_MAP.items():
        if model_key in raw:
            scores[public_key] = float(raw[model_key])   # 又加 visual_quality/...
    return scores
```

```python
# vrl/rewards/models/kling_video_reward.py:32-37
_SCORE_KEY_MAP = {
    "overall_reward": "Overall",
    "visual_quality": "VQ",
    "motion_quality": "MQ",
    "text_alignment": "TA",
}
```

实锤：configs/tests 里**只**用 public 别名作 `score_key`，原始键从未被任何 `score_key` 选中：

```
configs/reward/kling_video_reward.yaml:20:      score_key: overall_reward
configs/.../online_grpo_physics*.yaml:      score_key: motion_quality
tests/rewards/ray/test_runtime.py:41: score_key: str = "overall_reward"
tests/rewards/ray/test_resource_lifecycle.py:123,199,220: "overall_reward"
```

`grep '"Overall"|"VQ"|"MQ"|"TA"' vrl/scripts/` 无命中——没有 eval/analysis 脚本读原始键。每个分数 dict 因此带两份同值条目（`Overall` 与 `overall_reward` 等），reader 分不清哪个是公开契约。

### 1.4 `_DEFAULT_REVISION` 文件内冗余别名

```python
# vrl/rewards/models/kling_video_reward.py:31
_DEFAULT_REVISION = DEFAULT_HF_REVISION
# vrl/rewards/models/videocon_physics.py:36
_DEFAULT_REVISION = DEFAULT_HF_REVISION
# vrl/rewards/models/hub.py:7
DEFAULT_HF_REVISION = "main"
```

两个文件各自 import `DEFAULT_HF_REVISION` 后立即绑一个仅用一次的同值本地别名（kling:631 / videocon:260 各一处调用 `parse_hf_repo_revision(..., default_revision=_DEFAULT_REVISION)`）。同文件两个名字指一个常量，徒增「它们会不会发散」的疑问。

### 1.5 GenEval `evaluator` 单值占位旋钮

```python
# vrl/rewards/models/geneval.py:49
self.evaluator = str(cfg.get("evaluator", "import_path"))
# vrl/rewards/models/geneval.py:61-62
if self.evaluator != "import_path":
    raise ValueError(f"unsupported GenEval evaluator: {self.evaluator!r}")
```

唯一读 `evaluator` 的分支对非 `"import_path"` 直接 raise——它是死旋钮：真正的后端选择是「是否注入了 `scorer` callable，否则解析 `import_path`」。`functions/geneval.py:32,40` 在 function wrapper 上又存了一份 `self.evaluator`。配置作者看到 `evaluator` + `import_path` 两个旋钮，会误以为有多种 evaluator 模式，实际只有一个合法值。

（区别于 [[SPRINT_design_smell_audit]] §4.2「不统一 factory 的 evaluator wiring」——那讲的是 algorithm factory 的 kind dispatch，与此处 GenEval reward 的单值旋钮无关。）

## 落地方案

### A. worker_config 统一 model-id key（§1.1 + §1.2）

1. 在 `base.py._init_disk_artifact_reward`（`vrl/rewards/base.py:140-226`）里做**一次** normalize：把 top-level `reward_name`（仅当它看起来是 repo，即含 `/`）与 `worker_config.model_name` 归一进 `worker_config["reward_model_name"]`，作为唯一的 model-id key。保留 `reward_name` 作逻辑标签传给 `RewardFunction.__init__`（base.py:230）。
2. 删除 base.py:209-214 里的 `or resolved_reward_name` 兜底；删除 kling_video_reward.py:186-190、:622-626 与 videocon_physics.py:55-57、:250-254 各自的 `or worker_config.get("model_name")` 分支，让 4 处只读 `worker_config["reward_model_name"]`（已由 base 归一）。
3. **同 PR 迁移配置**：`configs/reward/kling_video_reward.yaml:19`、`configs/reward/videocon_physics.yaml:21` 把 `reward_name: <repo>@rev` 改写为
   - `reward_name: kling_video_reward` / `reward_name: videocon_physics`（逻辑标签）
   - `worker_config.reward_model_name: KlingTeam/VideoReward@main` / `videophysics/videocon_physics@main`（model id）
4. 注释里更新 kling/videocon YAML 已有的 “resolve reward_model_name from the local HF cache” 提示（kling:25-26 / videocon:27-28），它们已经在说 `reward_model_name`，迁移后即名实一致。

取舍：**保留 image reward 的 `model_name`**。aesthetic/pickscore/nsfw 的 model 是直接 `self.worker_config.get("model_name")` 读、且 function wrapper（`functions/aesthetic.py:17,24`、`functions/pickscore.py:18,26`）把它当 `__init__` 关键字参数透传，是这些 in-memory reward 的稳定公开参数；它们走的是 `_init_reward_model`（in-memory），不走 disk-artifact 的 `or`-链。强行统一成 `reward_model_name` 收益小、迁移面大。本 sprint 只统一 **disk-artifact / Ray-pool** 路径的 model-id key 为 `reward_model_name`，并在 normalize 里继续容忍 `model_name`（一处别名，留一行 deprecation 注释），不再让三个 loader 各自重复 `or model_name`。

### B. Kling 分数只输出 public 别名（§1.3）

把 `_normalize_scores`（kling_video_reward.py:677-683）改为不再 `dict(raw)` carry-over，只 emit `_SCORE_KEY_MAP` 的 public 键。先用 `grep -rn '"Overall"\|"VQ"\|"MQ"\|"TA"' vrl/ docs/ configs/ tests/` 复核无消费者（当前 §1.3 已确认 vrl/scripts 无命中）；若 debug JSONL 仍想保留原始值，改为在 debug payload 里单独带，而不混进 `scores`，并禁止其作 `score_key`。

附带统一「aggregate」命名一致性（原 finding「Overall vs overall」）：本 sprint **不**单独迁移 videocon 的 `overall`（videocon_physics.py:143），因为 [[SPRINT_config_string_settings]]:84 已决定 `score_key` 是 per-reward 故意词汇、默认 score_key 是 `physical_commonsense` 不是 `overall`，迁移收益不抵破坏面。仅在 README（`configs/reward/README.md`）补一句「aggregate 在 Kling 叫 `overall_reward`、在 VideoCon 叫 `overall`」消歧即可。

### C. 删 `_DEFAULT_REVISION` 别名（§1.4）

删 kling_video_reward.py:31 与 videocon_physics.py:36 的 `_DEFAULT_REVISION`，把 :631 / :260 的调用直接传 `DEFAULT_HF_REVISION`（`parse_hf_repo_revision` 本身已默认它，多数情况可整参省略，见 hub.py 签名）。

### D. 删 GenEval `evaluator` 死旋钮（§1.5）

删 geneval.py:49 的 `self.evaluator`、:61-62 的 guard，以及 functions/geneval.py:32,40 的 `evaluator` 参数与字段；后端完全由「注入了 `scorer` 否则用 `import_path`」决定（geneval.py:64 已是此逻辑）。若仍有 checked-in/operator 配置写 `evaluator: import_path`，在 reward.kwargs 容忍并忽略（reward.kwargs 是故意 OPEN 面，见 [[SPRINT_config_string_settings]]），不报错。

## 验证（finishing criteria）

- `grep -rn 'or worker_config.get("model_name")\|or resolved_reward_name' vrl/rewards/` 只剩 base.py 一处归一逻辑（其余 3 处 loader 的复制链消失）。
- `python -c "import vrl.config..."` + reward 配置 resolve：kling/videocon 配置用新 `worker_config.reward_model_name` 能解析出原 repo id（`reward_model_name or model_path` 非空，base.py:217 分支照常进入）。
- `pytest tests/rewards/` 全绿，特别是 `tests/rewards/ray/test_runtime.py`、`test_resource_lifecycle.py`（`score_key="overall_reward"` 仍命中 public 别名）。
- Kling `_normalize_scores` 单测断言返回 dict 只含 public 键、不含 `Overall/VQ/MQ/TA`（除非显式 debug 通道）。
- GenEval reward 在不传 `evaluator` 时按 `import_path` 工作；传无关 `evaluator` 值不再 raise。
- `grep -rn "_DEFAULT_REVISION" vrl/rewards/models/` 无命中。

## 非目标 / Non-Goals

- 不把 `score_key` 收成 typed Literal（[[SPRINT_config_string_settings]] 已决 LEAVE：per-reward 词汇 + `a+b` 复合表达式）。
- 不迁移 image reward 的 `model_name`（in-memory reward 的稳定公开参数，见 §A 取舍）。
- 不迁移 videocon 的 `overall` 拼法为 `overall_reward`（仅 README 消歧）。
- 不动 `parse_hf_repo_revision` 的 `repo@revision` 文法（[[SPRINT_design_smell_audit]] 已抽好）。
- 不重命名 inference 协议的 `reward_name` 字段（它是 request 标签 protocol boundary，只是不再让它兼任 model id）。

## References

- `vrl/rewards/base.py:140-234`（`_init_disk_artifact_reward`、三名 `or`-链 + reward_name 兜底）
- `vrl/rewards/models/kling_video_reward.py:31-37`（`_DEFAULT_REVISION` + `_SCORE_KEY_MAP`）、`:186-190`、`:617-637`（`_resolve_model_root`）、`:677-683`（`_normalize_scores`）
- `vrl/rewards/models/videocon_physics.py:36`、`:55-57`、`:130-144`（`overall`）、`:243-261`
- `vrl/rewards/models/hub.py:7`（`DEFAULT_HF_REVISION`）
- `vrl/rewards/models/geneval.py:46-62`、`vrl/rewards/functions/geneval.py:32,40`
- `vrl/rewards/models/aesthetic.py:17-18`、`vrl/rewards/models/pickscore.py:27-28`、`vrl/rewards/models/nsfw_safety.py:34`
- `vrl/rewards/functions/aesthetic.py:17-33`、`vrl/rewards/functions/pickscore.py:18-26`
- `vrl/rewards/inference.py:105-126`（`RewardInferenceRequest.reward_name` required）
- `configs/reward/kling_video_reward.yaml:19-26`、`configs/reward/videocon_physics.yaml:21-31`、`configs/reward/aesthetic.yaml:7`、`configs/reward/pickscore.yaml:7-8`、`configs/reward/nsfw_safety.yaml:7`、`configs/reward/README.md:26`
- `tests/rewards/ray/test_runtime.py:41`、`tests/rewards/ray/test_resource_lifecycle.py:123,199,220`
- `docs/sprints/done/SPRINT_design_smell_audit.md:202-204`（已抽 `parse_hf_repo_revision`，未碰三名链）、`docs/sprints/done/SPRINT_config_string_settings.md:84`（`score_key` LEAVE）

相关 sprint：[[SPRINT_design_smell_audit]]、[[SPRINT_config_string_settings]]、[[SPRINT_precision_naming_unification]]