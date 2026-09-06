# SPRINT: 完成 tiny-real diffusers fixtures 转换（轨道四剩余工作）

状态：**done（2026-09-05 六项落地；2026-09-06 环境同步后第 4 项 cosmos3 也落地，见文末）**。轨道顺序 4 / 6。风险：medium。

审计与已完成子集见
`docs/sprints/done/SPRINT_tiny-real-diffusers-fixtures_audit.md`。commit `300ef8c7`
已经加入真实 VAE fixture，并把 VAE 内存策略与 Wan DPO encoder 测试从方法调用记录改成
真实对象状态断言。本计划只保留仍未落地的工作，不重复执行该子集。

## 剩余实施清单

1. **Pipeline shell：**加入无下载、CPU、config-init 的 tiny pipeline shell builder，
   用它替换 frozen-offload 测试中自声明的 pipeline 替身。
2. **Scheduler 与生产 guard：**用真实 scheduler 替换 `_TinyScheduler`；真实测试覆盖到位后，
   删除 Mochi 与 PixArt Sigma 中只为替身存在、会静默跳过标准化的两个 scheduler guard。
3. **Anima：**先在 parity 测试中加入两个基于真实 transformer 的调用与分支断言，再删除旧的
   `test_forward_step.py`；必须保持“先加后删”。
4. **Cosmos3：**在兼容的 diffusers 版本上加入 tiny transformer / pipeline fixtures，
   完成 packed-static 装配、forward 参数与 CFG/decode 三组真实对象测试。
5. **NextStep：**加入真实 f8 VAE，并把重复 fixture 构造收敛到家族 fixture 模块；保留不可导入
   上游包的边界替身。
6. **SANA：**把重复的 scheduler 构造收敛为共享 builder，同时保留用于验证 hub 参数投影的
   `from_pretrained` recorder。
7. **基础设施标注收尾：**按当前 `real_cover` 契约补齐审计登记的真实对位与诚实缺口；
   命名目标可以位于 default lane，但目标必须存在且 `why` 必须解释覆盖差距。

## 明确非目标

- 保留 `_IdentityDecodeVAE`：它是让 layout 与反归一化算术可观测的 identity 探针，不是模型替身。
- 保留 scheduler wrong-class 测试中的假对象：它隔离“类名不匹配”这一半校验，真实错误类会同时
  引入 config 不匹配。
- 保留 NextStep 的 `sys.modules` 注入与 `UpstreamModel.unpatchify`：对应上游包不是仓库依赖，
  这是必要的包边界适配。
- 保留 SANA `from_pretrained` recorder：它验证 revision/subfolder 参数投影，不试图复现 Hub。
- 保留 Wan 与 SANA 的 fail-loud scheduler guards；它们保护真实可选字段或提供明确错误。
- 不把 diffusers pipeline 下载车道并入本轨；本轨 fixtures 必须 config-init、CPU、无网络。

## 完成判据

- 上述七组剩余工作全部落地，原审计中对应替身和重复构造消失。
- 每个删除动作先有真实对象测试覆盖，且相关家族测试、默认测试全集与 scoped Ruff 全绿。
- `real_cover` 标注通过架构守卫，所有 `tracked_in` 路径在磁盘上存在。

## References

- 审计快照：`docs/sprints/done/SPRINT_tiny-real-diffusers-fixtures_audit.md`
- 已落地子集：commit `300ef8c7`
- Track 1 契约：`docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md`

## 落地记录（2026-09-05）

| # | 项 | commit | 落地形态 |
|---|---|---|---|
| 1 | Pipeline shell | `ae9827a1d` | `build_tiny_pipeline_shell(transformer=, vae=, scheduler=, text_encoder=None)` 是真 `DiffusionPipeline` 子类，`.components` 由 diffusers 自己派生（含 `None` 槽 + 非 module scheduler）；`test_frozen_offload.py` 用 `meta` 设备观测真实搬迁，`_RecordingModule` / `_FakePipeline` 删除 |
| 2 | Scheduler + 生产 guard | `685173d00` | `_TinyScheduler` 删除，六处注入改真类（`getattr(diffusers, scheduler_classname)()` / `FlowMatchEulerDiscreteScheduler()`）；wan 断言由 echo `[1.0]` 改类身份；新增 mochi/pixart 参数化测试证明 `prepare_replay` 真的替换了 loader 交出的实例（类 + `num_steps`，不钉字面 ladder）；两处 `config is not None` 半个 guard 删除，`num_steps is None` 半边保留 |
| 3 | Anima | `8debce307` | parity 文件先加 `test_cond_branch_runs_first_on_the_positive_embeds`（按身份钉 embeds）与 `test_cfg_off_runs_one_forward_and_reports_a_zero_uncond`，再删 `test_forward_step.py` 与 `_ConstantAnimaTransformer` |
| 5 | NextStep | `7f23670a5` | 新 `tests/models/families/nextstep_1/fixtures.py`：`install_stub_nextstep_pipeline`（两份重复的 `gen_pipeline` 注入收敛）、`build_tiny_nextstep_vae`（f8ch16 几何的真 `AutoencoderKL`）、`build_decode_only_nextstep_model(vae=)`；32 由 VAE 算出，`unpatchify` 保持包边界替身 |
| 6 | SANA | `bb9c6bfd8` | `tests/scripts/eval/fixtures.py::build_official_sana_scheduler(**overrides)` 由 `SCHEDULER_PROTOCOL` 派生 kwargs 构造真类；四份 echo 替身删除；revision recorder 改为 patch 真类的 `from_pretrained`，不再替换 `sys.modules["diffusers"]` |
| 7 | real_cover | 随 2 / 5 | nextstep decode → `tests/e2e/test_real_checkpoint_rl.py::test_real_checkpoint_online_rl_updates_trainable_weights`；mochi/pixart ladder → `tests/models/steps/denoise/test_scheduler_logprob_parity.py::test_family_scheduler_sample_replay_parity`；架构守卫 `tests/architecture/test_real_cover_labels.py` 绿 |

### 第 4 项 cosmos3（2026-09-06 补齐，commit `d1b238a8b`）

9 月 5 日时两个解释器都没有 diffusers 0.39 的 `Cosmos3OmniPipeline`；GPU 空出来后
`uv sync --inexact --extra cosmos --extra reward` 把 `.venv` 同步到 lock（`--inexact`
保住了 pytest / ruff 等不在所选 extra 里的已装包；miniconda base 仍是 0.37.1，未动）。

落地形态与审计 §6.2 一致，差异只在数字：`build_tiny_cosmos3_tokenizer`（真
`PreTrainedTokenizerFast` + word-level 词表 + 最小 chat template，`<|vision_start|>` /
`eos_token_id` 都是 pipeline `__init__` 真读的）、`build_tiny_cosmos3_transformer`
（30,256 参数，`patch_latent_dim = 4 * 2²`、`mrope_section` 和为 `head_dim // 2`）、
`build_tiny_cosmos3_pipeline`（`enable_safety_checker=False`，与 `from_build` 同）。
`build_tiny_wan_vae` 加了 `latents_mean` / `latents_std` 参数，让 decode 的反归一化可观测。
四个测试：packed_static 装配（一个样本、`num_steps` 个 timestep、11 个 transformer kwarg
名、`sequence_length = und_len + num_vision_tokens`、noisy token 数由 latent 与 patch 算出）；
flow 域 CFG combine `uncond + g*(cond-uncond)` 与 float32 raw velocity；cfg-off 单次 forward；
`latents_mean/std` 反归一化后走真 VAE decode。实测 4 passed，首个测试 1.22 s（含
diffusers cosmos3 懒加载），其余 < 50 ms。

这些测试**不 skip**低版本 diffusers：lock 钉的是 0.39，装了旧版是环境错误，按 RW-11
的原则让它红而不是无声跳过。

### 有意不做的标注

审计 §9 最后一行（VAE tiling 是否真的降低 decode 峰值显存）没有贴 `real_cover`：
`tests/models/steps/denoise/common/test_vae_decode_memory.py` 里翻转 `use_tiling` 的
测试用的是真 `AutoencoderKL`，没有替身；`real_cover` 的契约是"本测试用了替身"，
给一个没有替身的测试贴标签是滥用 marker。该缺口保留在审计 §9 表中登记。

### 验收

- 触碰的七个测试文件 + `tests/architecture/test_real_cover_labels.py`：150 passed。
- 全仓 CPU gate（`CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 pytest tests -q`）：3447 passed / 63 skipped（改动前 3444；净 +3 = anima +1、mochi/pixart 参数化 +2）。
