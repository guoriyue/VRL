# SPRINT：`GenerationRequest.sampling` 从 dict 袋子到分层类型（done）

状态：**done（2026-09-05 立项，当日三片落地；执行记录见 §8，与 §3/§4 的偏差也记在那里）**。本文是数据类型清理时对 `GenerationRequest.sampling: dict[str, Any]`
的审计结论：它不是一个"该类型化的闭合 key 集"，而是六个来源合并出来的 wire 载荷，把它改成
typed 是一次跨 config / collector / wire / 三个 binding / 七个 family runtime 的重设计，不是清理。
本文记录证据、分层方案和分三片落地的顺序，供独立 sprint 执行。

## 0. 一句话

`request.sampling` 今天是六个来源的并集（见 §1），读方按家族多态地取 key（见 §2）。正确的
形状不是"一个 typed dataclass 替换 dict"，而是把**引擎级**的三个 key 提到
`GenerationRequest` 字段（第一片，完全机械）、把**去噪级**的 SDE/denoise 参数落成一个从
`SdeConfig` 派生的 typed 对象（第二片）、把**家族级**的 `SamplingSection` pydantic 实例原样
上 wire 并用 `extra="forbid"` 吃掉 `request_overrides`（第三片）。

## 1. 今天 `request.sampling` 的六个来源

`vrl/rollouts/collector/config.py::RolloutCollectorConfig.from_cfg` + `generation_sampling()`
与 `vrl/rollouts/collector/requests.py::GenerationRequestBuilder.build`：

| 来源 | 进入方式 | key 集 |
|---|---|---|
| `rollout.*` 平铺字段 | `generation_request_rollout_fields()`：`RolloutConfig` 里 `runtime_owner == "generation_request"` 的字段 | 由 schema 元数据决定（含 `samples_per_generation_batch`、`denoise_mode`、`return_kl` 等） |
| `rollout.sde.*` | `SdeConfig.model_fields` 逐个映射为 `sde_<name>` | `sde_type`、`sde_window_size`、`sde_window_range` |
| `sampling.*` | 家族的 `SamplingSection` 子类 `model_dump(exclude_unset=True)`（`vrl/models/families/registry.py:348`） | 按家族：diffusion 的 width/height/num_steps/guidance_scale/num_frames/fps/max_sequence_length；AR 的 temperature/top_p/image_size/… |
| `algorithm.train_segments` | 直接塞入 | `train_segments` |
| `trajectory_storage` | 由 typed `TrajectoryStoragePolicy` 派生（非默认时才写） | `trajectory_storage` |
| 每条 prompt 的 `request_overrides` | `PromptExample.request_overrides` 任意 dict，`sampling.update(...)` | **开放集** |
| 运行期改写 | `vrl/generation/ray/runtime.py:251` 把 `samples_per_generation_batch: "auto"` 改成探测出的 int | — |

## 2. 读方清单（按文件）

```
vrl/generation/execution/planner.py            samples_per_generation_batch
vrl/generation/ray/runtime.py                  samples_per_generation_batch（读+改写）
vrl/generation/execution/batch_placement.py    num_steps
vrl/generation/bindings/full_sequence_denoise/executor.py   trajectory_storage
vrl/generation/bindings/full_sequence_denoise/layout.py     num_steps fps seed guidance_scale height width
                                               num_frames|frame_count negative_prompt denoise_mode
                                               sde_window_range sde_window_size noise_level sde_type
                                               return_kl return_prev_sample_mean cache_ref_noise_pred teacache
vrl/generation/bindings/token_autoregressive/layout.py      image_token_num image_size max_text_length seed
                                               ar_scheduler_batch_size
vrl/generation/bindings/token_autoregressive/executor.py    attention_backend ar_engine seed
                                               ar_paged_block_size ar_paged_cache_dtype
vrl/trajectory/builders.py                     train_segments
vrl/models/families/{janus_pro,emu3,glm_image,nextstep_1,llamagen}/runtime.py  家族各自的 key
vrl/models/families/{causvid,magi_1}/model.py  width height num_frames fps num_steps guidance_scale …
```

三个观察：

1. **前四行是引擎概念，不是采样概念。** `samples_per_generation_batch`（planner 批宽）、
   `trajectory_storage`（回放张量放哪）、`train_segments`（哪些 segment 可训练）由 planner /
   executor / builders 这些家族无关的代码读，它们只是搭了 `sampling` 这趟车。
2. **diffusion layout 那 18 个 key 有 typed 源头**：`SdeConfig`（type/window_*）、
   `RolloutConfig` 的 generation_request 字段、家族 `SamplingSection`。layout 里的
   `sampling.get("noise_level", 1.0)` 之类的默认值是 config 默认值的第二份拷贝。
3. **家族 runtime 的 key 集就是各自的 `SamplingSection` 字段**（`JanusProSamplingSection`
   有 guidance_scale/image_size/image_token_num/temperature……），wire 上却是它 dump 出来的
   dict，再在 runtime 里 `sampling.get("temperature", self.model.config.temperature)` 手工回填。

## 3. 目标形状

```python
@dataclass(slots=True, init=False)
class GenerationRequest:
    request_id: str
    family: str
    task: str
    inputs: list[GenerationInput]
    samples_per_prompt: int
    # 第一片：引擎级 knob 成为字段
    samples_per_generation_batch: int | Literal["auto"] | None
    train_segments: dict[str, bool] | None
    trajectory_storage: TrajectoryStoragePolicy | None
    # 第二片：去噪级参数一个 typed 对象（AR 请求为 None）
    denoise: DenoiseRequestOptions | None
    # 第三片：家族采样段原样上 wire（pydantic 可 pickle；extra=forbid 已在类上）
    sampling: SamplingSection
    runtime_debug: bool
    policy_version: int | None
```

- `DenoiseRequestOptions` 从 `SdeConfig` + `RolloutConfig` 的 generation_request 字段派生
  （`denoise_mode`、`sde_type`、`sde_window_size`、`sde_window_range`、`noise_level`、
  `return_kl`、`return_prev_sample_mean`、`cache_ref_noise_pred`、`teacache`、`seed`、
  `negative_prompt`），**默认值只在 config schema 上存在一次**；
  `DiffusionRequestLayout.parse_sampling_params` 变成从两个 typed 对象装配
  `DenoiseRequest` + `DenoiseSDEParams`，不再有 `.get(key, default)`。
- `request_overrides` 的施加点从 `dict.update` 改为
  `sampling.model_copy(update=overrides)` + 重新 validate：未知 key 在 collector 构造请求时
  被 `extra="forbid"` 拒掉，而不是静默进 wire 后被某个 runtime 忽略。
- `samples_per_generation_batch: "auto"` 的运行期改写变成 `replace(request, samples_per_generation_batch=n)`。

## 4. 分三片落地（每片独立可 merge）

### 第一片：三个引擎级 key 升为字段（纯机械）

写方：`requests.py`（从 `RolloutCollectorConfig` 取）；读方：`planner.py`、`ray/runtime.py`、
`execution/worker.py:372`（probe 复制 request）、`full_sequence_denoise/executor.py:342`、
`trajectory/builders.py:620`、`collector/config.py`。

测试面（本次审计实测）：`samples_per_generation_batch` 85 行 / 20 个文件，
`trajectory_storage` 27 行 / 7 个文件，`train_segments` 10 行 / 5 个文件——全部是把
`sampling={"samples_per_generation_batch": n}` 改成 `samples_per_generation_batch=n` 的改写。

### 第二片：`DenoiseRequestOptions`

新增在 `vrl/generation/steps/denoise/config.py`（`DenoiseSDEParams` 已在那里）。
`DiffusionRequestLayout.__init__` 的四个 `default_*` 保留（家族默认值是执行器的，不是 config 的）。
删掉 layout 里所有 `sampling.get(key, literal_default)`；`_parse_sde_window_range` /
`_validate_sde_window_size` 挪到 typed 对象的 `__post_init__`。

### 第三片：`SamplingSection` 上 wire

`registry.py:348` 不再 `model_dump`，直接把 section 实例交给 `RolloutCollectorConfig`；
七个家族 runtime 把 `sampling.get("temperature", self.model.config.temperature)` 改成
`request.sampling.temperature`，缺省回落留在各家族的 `SamplingSection` 字段默认值上
（今天这些字段默认 `None`，回落逻辑在 runtime——第三片要把回落值上移到 section 或者
保留 `None` 语义，二选一，在执行时按每个家族的 `model.config` 决定）。
`ARRequestLayout._sampling_int` 与 emu3 / glm_image 各自手写的
`"max_text_length" not in sampling → raise` 合并为 section 校验。

## 5. 非目标

- 不把 `request_overrides` 关掉：它是 prompt manifest 层的合法扩展点，第三片只是让它
  fail-closed。
- 不改 `GenerationRequest` 的 wire 身份（`request_id` / `sample_rows()` / `policy_version`）。
- 不动 `vrl/scripts/eval/_sampling.py::resolve_eval_sampling`：它的 docstring 明确说这是
  给 eval 脚本用的 runtime dict，不是 config 层对象；第二片落地后可以让它产出
  `DenoiseRequestOptions`，但那是后续。

## 6. 验收

- 每片：`tests/generation tests/rollouts tests/trajectory tests/models` 全绿；
  `grep -rn 'sampling\.get(\|sampling\["' vrl` 的命中数逐片下降，第三片后只剩家族
  runtime 之外的零命中。
- `tests/config/test_unknown_keys.py` 新增一条：`request_overrides={"typo_key": 1}` 在
  collector 构造请求时报错，而不是进 wire。

## 7. 本次审计已顺手改掉的

- `vrl/generation/execution/batch_placement.py`：`request.sampling.get("max_new_tokens")`
  分支——全仓没有任何写方（`max_new_tokens` 只出现在 reward worker_config 里），是死代码
  五形式第 2 形（活调用者、死语义），已删。

## 8. 执行记录（2026-09-05）

三片各一个 commit，每片都过全量 CPU 套件（`CUDA_VISIBLE_DEVICES=""`）、config lint 与
config snapshot gate；未跑任何 GPU / 训练。

### 第一片 — `refactor(generation): lift the engine-level knobs off the sampling dict`

按 §4 原样落地：`samples_per_generation_batch` / `train_segments` /
`trajectory_storage` 成为 `GenerationRequest` 字段；`RolloutCollectorConfig` 携带前两者，
`generation_sampling()` 及其对 storage policy 的 wire 再编码删除；planner、Ray runtime 的
`auto` 改写（`replace(request, samples_per_generation_batch=n)`）、executor 的
`apply_wire_storage_policy`（直接读 typed policy）、multisegment builder 改读字段。
`tests/quality/preview.py` 用 `replace()` 而不是 request_overrides 处理 `auto`。

### 第二片 — `refactor(generation): carry the rollout-owned denoise knobs as typed DenoiseRequestOptions`

`DenoiseRequestOptions`（`vrl/generation/steps/denoise/config.py`）持有
`denoise_mode / noise_level / sde_type / sde_window_size / sde_window_range / return_kl /
return_prev_sample_mean / cache_ref_noise_pred / teacache`，构造时校验词表与窗口形状，
`resolve_sde_window_range(num_steps)` 只做依赖 schedule 的那一步。collector 从
`rollout.*` / `rollout.sde.*` / `sampling.teacache` 投影一次（`return_kl` 仍由 KL reward
系数派生），layout 的四个私有 parser 与全部 `sampling.get(key, literal)` 删除；factory 的
SDE evaluator 与 nextstep runtime 读同一对象。

**与 §3 的两处偏差**：

- `seed` / `negative_prompt` **没有**进 `DenoiseRequestOptions`。它们不是 `rollout.*` 的
  旋钮：AR layout / executor、causvid、magi 都从各自的 sampling 里读 `seed`，`negative_prompt`
  由去噪家族的 sampling section 拥有。它们在第三片成为 section 字段（见下）。
- `denoise` 对**所有**家族的 collector 请求都构造，不是"AR 请求为 None"。证据是
  nextstep_1（AR continuous）消费 `rollout.noise_level`
  （`vrl/models/families/nextstep_1/runtime.py`）。手工构造的请求 `denoise=None` 时读方回落到
  `DenoiseRequestOptions()` 的默认值，默认值仍只此一份。

### 第三片 — `refactor(rollouts): per-prompt request_overrides are validated against the family sampling section`

**与 §4 的偏差**：没有把 pydantic `SamplingSection` 实例放上 wire，`request.sampling`
仍是"该家族 section 已声明键"的 dict。原因：(a) 三十多个测试文件用 `family="test"` 等
不在 registry 里的家族手工构造请求，实例上 wire 会迫使每个都经过 registry；(b) 七个家族
runtime 的 `sampling.get(key, self.model.config.default)` 在 section 字段默认 `None` 的前提下
只会变成 `x if x is not None else default`，类型化收益为零。fail-closed 的价值在边界拿到：

- `vrl/config/schema.py::require_sampling_overrides(family, overrides)`：用家族的
  `SamplingSection`（`extra="forbid"`）校验每条 prompt 的 `request_overrides`，错误信息与
  YAML 同形（`unknown sampling.typo_key`）。`GenerationRequestBuilder.build` 在构造请求时
  调用；去噪旋钮的覆盖走 `replace(DenoiseRequestOptions)`。
- section 词表补齐真实的 override 来源：`SamplingSection.seed`（droid 数据集每行的
  `request_overrides.seed`）、`DenoiseImageSamplingSection.negative_prompt`（preview 与
  anima 的反向提示）、`SharedAttentionARSamplingSection.ar_paged_block_size / ar_paged_cache_dtype`
  （原先只有测试能产生的 vllm_paged 旋钮，现在可从 YAML 到达）。snapshot gate 的差异只有
  这些新字段的 `null`。
- `require_native_ar_engine` 与 `sampling.ar_engine` 删除：边界关闭后没有任何 section 声明
  它，guard 只剩测试能触发（dead code 五形之一、二）。`SPRINT_config_string_settings.md`
  当年"保留 vestigial guard"的裁定建立在 sampling 开放的前提上，前提已不成立。
- `PromptExample.request_overrides` 的注释写明词表来源。

验收测试在 `tests/rollouts/runtime/test_engine_requests.py::test_engine_request_builder_rejects_a_request_override_outside_the_family_vocabulary`
（§6 写的是 `tests/config/test_unknown_keys.py`；它是 collector 边界，放在 collector 的测试里）。

### §6 的 grep 指标

`grep -rn 'sampling\.get(\|sampling\["' vrl`：96 → 74。剩余命中全部是家族级键的读方：

| 位置 | 命中 | 性质 |
|---|---|---|
| 七个家族 runtime / model（janus_pro、emu3、glm_image、llamagen、nextstep_1、causvid、magi_1、flux） | 35 | 家族自己的 section 字段回落到 `model.config` |
| `full_sequence_denoise/layout.py` | 11 | 几何 / fps / max_sequence_length / seed / negative_prompt，回落到 executor 的 `default_*` |
| `token_autoregressive/{layout,executor}.py` + `ar_attention_backends.py` | 7 | AR binding 级键（seed、image_*、attention_backend、ar_paged_*、ar_scheduler_batch_size） |
| `execution/batch_placement.py` | 1 | `num_steps` 作为 diffusion 成本启发 |
| `rollouts/collector/requests.py` | 1 | `fps` 复制进 reward metadata |
| `scripts/eval/*` | 19 | §5 非目标：eval 脚本自己的 runtime dict |

"家族 runtime 之外零命中"未达成：AR binding 的 7 处正是
`SPRINT_config_resolution_consolidation.md` P3 计划并入 `ARSamplingParams` 的那组键，
属于该 sprint；`batch_placement` 与 `requests.py` 各一处读的是家族键，随家族 section
类型化一起走。
