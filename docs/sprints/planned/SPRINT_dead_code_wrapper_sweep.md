# SPRINT: Dead-code & test-only wrapper sweep — 单调用/零调用清除（第二轮）

**日期**: 2026-07-12  **状态**: EXECUTED（2026-07-13，基于 f37e4e93）

**执行修正（与原计划的差异）**:
1. Sprint 2 第 8 项 `bundled_config_resource` **保留**：4 个 config 测试文件是它的
   preset 内省消费方，且与生产使用的 `list_bundled_configs` 同族；删除只会让
   4 处各自手写 importlib.resources。
2. Sprint 2 第 9 项 `launch_contract.to_dict` **保留**：与 rewards `wire.py`
   编解码对同类的对称 wire codec（`from_dict` 生产侧、`to_dict` 测试侧构造
   合法 payload），测试用它对 contract 字段演进保持鲁棒——与本仓对 wire 协议对
   的既有裁决一致。
3. Sprint 2 第 3 项 `RolloutCollector.collect` 删除后，单发便捷逻辑落
   `tests/rollouts/collector/_collect.py`（测试侧 helper，仿 `_collector_control`
   惯例），8 处测试调用点改走它。
4. `prepare_latents` bug 的防复发从"逐家回归测试"升级为根因防护：
   `LatentDecodePlan.__post_init__` 校验三个 callable 字段（构造点即失败），
   附一条单元回归测试。
**来源**: 392 文件逐文件通读审计（8 个按包扫描 agent + 全部 DELETE 判定人工亲验：
定义外 grep 含 `module:function` 字符串分发、registry、getattr 鸭子分发），
标准取自 AGENTS.md（死代码五形式 / 薄函数保留清单 / test-only 调用方 = 死）。
上一轮 grab-bag audit（2026-07-10）之后新积累/漏网的部分。

**已顺手修掉的活 bug（本 sprint 范围外，工作区未提交）**：
sana / hunyuan_image / hunyuan_video 三家 `decode_latents` 的
`prepare_latents=(lambda ...,)` 尾逗号使其成为 1-tuple 而非 callable，
`ChunkedLatentDecoder._decode_chunk`（`common/latent_decode.py:48`）一调用即
`TypeError: 'tuple' object is not callable`。已改为裸 lambda（3 处各删 1 个逗号），
ruff 全过、`tests/models/diffusion` 199 全绿。三家 decode 路径此前无任何测试覆盖 →
变更清单 §3 补一条回归测试。

---

## Sprint 1 — 零调用死代码（纯删除，零行为变化）

全部经"定义外全仓 grep（含字符串分发）= 0 命中"亲验：

1. `vrl/generation/ar/layout.py:72` `expand_prompts` — 全仓零引用（连测试都没有）。
2. `vrl/generation/ray/runtime.py:137` `with_release_after_collect` +
   `:422` `release` — 两个自述 "compatibility facade" 的兼容残桩，零调用。
3. `vrl/rewards/models/{geneval.py:95, nsfw_safety.py:139, ocr.py:164}` 三个
   `*_reward_model` factory — 零引用；对应 reward wrapper 直接构造模型类，
   活的 factory（aesthetic/pickscore/kling 等）都有 `model_factory` 字符串调用方。
   各文件 `__all__` 同步删。
4. `vrl/nn/quantization/fp8.py:57` `vllm_block_fp8_available` — 零引用；
   blockwise 的真实依赖检查在 `_blockwise_gemm` 内联。`quantization/__init__.py`
   再导出同步删。
5. `vrl/utils/profiling.py:413` `torch_profiler_step = capture_torch_trace` 别名 —
   零引用（同行 `record_function` 别名是活的，保留）；模块 docstring 同步改。
6. `vrl/trainers/offline/dpo.py:153` `sd_unet_forward` — 全仓零调用（连测试没有）。
   SD1.5 UNet DPO 不是支持路径（无 recipe/config/测试），删；`offline/__init__.py`
   导出与 dpo.py:192 docstring 提及同步删。

## Sprint 2 — test-only 死代码（生产零调用；删源码 + 改测试）

1. AR 五家 `trainable_param_count`（emu3:274 / janus_pro:264 / glm_image:339 /
   llamagen:219 / nextstep_1:200）+ 四家 `has_lora_adapter`（emu3:327 /
   janus_pro:297 / glm_image:390 / llamagen:222）— 共 9 个方法，全部只有测试断言在读；
   生产 adapter 路径走 `disable_adapter_on`。janus model.py:291 docstring 提及同步改。
   测试改为直接断言 `count_trainable_params(...)` / `disable_adapter` 可调用性，
   或直接删对应断言。
2. `vrl/rollouts/batch/core.py:30` `stack_batches` — 仅测试；生产走
   `split_batch_by_group`/`select_batch`。`batch/__init__.py` 导出同步删。
3. `vrl/rollouts/collector/core.py:153` `RolloutCollector.collect` — 仅测试的
   单发便捷 API（`collect_unscored` + `score_rollouts` 纯组合）；生产路径
   `collect_prompt_batches` 不经过它。~9 处测试改调分步 API。
   **注意**：同类的 `RolloutConfig.get` / `request_sampling` 是 `cfg_get`/getattr
   鸭子分发的**活**方法，勿动。
4. `vrl/rollouts/collector/config.py:26` `RolloutConfig.require` — 仅测试（5 处）。
5. `vrl/generation/execution/chunk_placement.py:128` `to_metrics` /
   `:170` `predict_bytes` — 仅测试；生产写 `metrics["chunk_memory"]` 直接
   `dict(chunk_output.memory)`（worker.py:767），读侧走 `from_metrics`。
   测试改用 `dataclasses.asdict` / 内联仿射公式。
6. `vrl/utils/stats.py:114` `from_phase_dict` — 仅测试；测试改 `cls()` + `add_phases`。
7. `vrl/scripts/data/danbooru.py:1109` `hard_negative_rows` + `:1134`
   `label_queue_rows` — 未接线的 hard-negative 标注管道对，仅测试；
   `__all__` 两条 + `tests/data/test_danbooru.py` 对应用例同步删。
   （若 hard-negative 挖掘计划仍活，接 `setup.py` 子命令代替删除——默认删。）
8. `vrl/config/loading.py:36` `bundled_config_resource` — 仅测试；
   测试改走 `list_bundled_configs`/`load_config`，`vrl/config/__init__` 导出同步删。
9. `vrl/generation/launch_contract.py:94` `GenerationRuntimeLaunchContract.to_dict` —
   仅测试（`from_dict`/`from_value` 是活的）；测试改手拼 dict。

## Sprint 3 — 成片裁决（按"未来有用则留"规则，已定）

1. **KEEP — Ray 物理 stage-pipeline 脚手架**（`ray/pipeline_runner.py`、
   `ray/stage_worker.py`、`pipeline/topology.py` 的 `PipelineStageRuntimePolicy`
   调度字段）：生产零构造、仅测试可达，但它是 parked sprint
   `SPRINT_diffusion_rollout_stage_pipeline`（hybrid 多卡计划 next-step ⭐）的地基。
   进程内 `SerialPipelineRunner`/`PipelineStageWorkerCore` 路径本来就是活的。
   在 topology.py 模块 docstring 补一行指向该 sprint 的保留理由，防下轮审计再翻。
2. **KEEP — `plan`/`forward_plan`**（`generation/ar/executor.py:57,66` +
   `generation/diffusion/executor.py:418,487`）：e2e real-checkpoint 测试的
   全请求入口，防全请求路径与生产 per-chunk 路径漂移的 parity surface；
   删了 e2e 就得自己重排 chunk 编排 = 引入它要防的漂移。
3. **KEEP — `_debug_snapshot` 链**（continuous schedule→owner→runtime snapshot）：
   自述 test/diagnostic seam，14 处测试在用。
4. **DROP — `DistributedKRepeatSampler`**（`vrl/trainers/data/samplers.py` 整文件）：
   flow_grpo 遗产。当前架构里 K 重复发生在生成层（`rollout.n_samples_per_prompt`），
   数据层走 prompt loader（`data.sampler.type`），torch DataLoader 级 K-repeat
   sampler 无未来路径。连删：`trainers/__init__.py` + `trainers/data/__init__.py`
   导出、`tests/trainers/test_data.py` 整文件。
   附带发现（删除后即无关）：其 `set_epoch` 从未被任何 loop 调用，
   即使被用起来 epoch 也恒为 0、每 epoch 洗牌顺序相同——印证无人依赖。

## Sprint 4 — 过时注释/命名（3 行修正）

1. `vrl/rewards/inference.py:351` `score_artifacts_with_model` docstring 仍称
   "Shared by ... the Ray worker"——Ray reward pool 已删（2026-07-01 plan A），改为
   单调用方事实。
2. `vrl/models/diffusion/cosmos/anima/runtime.py:17`
   `resolve_anima_replay_model_build` docstring 声称 "referenced by symbol in
   train.py"——该引用不存在，改为 registry/e2e 字符串边界事实。
3. `vrl/generation/ar/layout.py:140` `require_rows` → `_require_rows`
   （diffusion 侧同名方法是私有，对齐命名）。

## 非目标

- probe/perf 脚本内部的单调用 helper（测量工具惯例，KEEP）。
- `vrl/trajectory/`、`vrl/math/`、`vrl/ray/` 审计全干净，零动作。
- wire 编解码对、FSDP/DCP adapter、registry 字符串分发函数等单调用 KEEP 项
  （见审计记录，均在薄函数保留清单内）。

## 验证

- 全量 pytest（基线 2238 passed / 18 skipped / 1 deselected——deselect 的
  `test_real_cumem_one_shot...` 是本机缺 vLLM CuMemAllocator 的既有环境缺口）。
- ruff check + format --check 仅限触碰文件。
- config resolve 三关（load_config → warn_unknown_keys → parse_config）跑一遍
  动过导出的包所涉 experiment preset。
- 新增：sana / hunyuan_image / hunyuan_video `decode_latents` 的
  `prepare_latents` callable 回归测试（tuple bug 的直接防复发）。
