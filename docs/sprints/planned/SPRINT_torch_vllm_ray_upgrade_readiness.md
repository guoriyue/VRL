# SPRINT：torch / vLLM / Ray 升级预检与收获清单

状态：**planned（2026-10-20）**。触发事件：决定升级 torch 到 ≥2.13、vLLM 到 ≥0.22，或 Ray 到 ≥2.59。

来源：PyTorch Conference NA 2026 Day 1。005 是 PyTorch 2.10–2.14 的 release blog 和 2.15 计划；007 是 Ray 2.56–2.59 的 release notes；011 是 vLLM 0.22–0.32 的 release notes；093 是 torch.distributed 新 API。

## 0. 结论先行

当前锁定版本：torch 2.11.0、triton 3.6.0、ray 2.55.1、vllm 0.21.0（`ar-vllm` extra，共享 GPU 环境用 `--no-deps` 安装）、diffusers 0.41.0.dev0（git rev）、transformers 5.17.0。

CI 没有 GPU lane：`gpu_tests`、`gpu_vllm_tests` 都是 manual，所以升级时必须在 GPU 主机上手动跑。

升级分两步：
1. **预检（P0–P2）**：把会坏的地方提前找出来。
2. **收获（P3）**：升级后才能用的功能，一项一个 commit，每项都要实测。

## 1. 已知风险点（HEAD a914644c6 核对）

| 风险 | 来源 | VRL 位置 |
|---|---|---|
| torch 2.15 计划默认使用 nccl2/torchcomms，要求 eager init 并指定 device | 005 / 093 | `init_process_group` 只在 `vrl/utils/process_group.py:38` 和 `tools/delivery/node_rank.py:20` 调用，都**没传 `device_id`**。前者被 `TrainerProcessGroup`（`vrl/trainers/distributed.py:244`）和 `EngineProcessGroup`（`vrl/generation/execution/rank_group.py:68`）共用。另有两个 gloo 子组：`distributed.py:254` 和 `:363` |
| FSDP2 私有 API | — | `vrl/models/parking.py:173-195`：`_get_fsdp_state`、`_fsdp_param_groups`、`reset_sharded_param`、DTensor `_local_tensor`；`parking_disk_store.py:50,62`；`strategy.py:719`；`wan_2_1/model.py:1172` |
| `torch._dynamo.eval_frame.OptimizedModule`、`torch._scaled_mm` | — | `weight_utils.py:45`、`fsdp.py:128`、`fp8.py:104`、`fp4.py:215` |
| vLLM CuMemAllocator 内部结构 | 011 | `parking.py:781-787` 读 `allocator_and_pools` 并 `.pop(tag)`，docstring 称之为 "the one required internal seam"。`:730-733` 依赖"每次 `use_memory_pool` 都新建一个 MemPool"这一行为 |
| vLLM 0.32 连带要求 torch 2.13 和 transformers `<5.20` | 011 | `pyproject.toml` 的 `ar-vllm` extra 和 `[tool.uv] conflicts` |
| Ray 2.59 默认对本地集群开启 token 认证 | 007 | `vrl/scripts/common/online.py:156` 用 `ray.init(address="local")`；`tools/delivery/node_probe.py:61`；测试 fixture（`tests/conftest.py:257` 等） |
| 2.13 把 `all_gather_into_tensor` 改名为 `*_single` | 005 | 已核对：VRL 没有用到。用到的是 `all_gather`、`all_to_all_single`、`broadcast` 等常规 API，不受影响 |

## 2. 阶段

### P0. 在现有版本上提前试跑（不升级）

1. 在两个 `init_process_group` 调用点补上 `device_id`（必须在 `torch.cuda.set_device` 之后，`distributed.py:240` 已经这样做了）。语义不变，并为 2.15 做好准备。
2. 用 `TORCH_DISTRIBUTED_USE_TORCHCOMMS=1` 在 GPU 主机上跑 `bazel test --config=gpu //tests:gpu_tests`，以及一次 2 卡 FSDP smoke，记录失败。gloo 子组在 torchcomms 下的行为要单独确认。
3. **验收**：两种后端下 GPU 测试结果相同，或失败都被记录下来并归因。

### P1. 升级分支：torch 2.13 + vLLM 0.32 + transformers <5.20

1. 用 `uv lock` 一次升级这一组，在独立分支上做。按 memory 规则走 `uv sync --frozen`，不手工 `pip install`。
2. 必跑：
   - `bazel test //...`（CPU lanes）；
   - `--config=gpu //tests:gpu_tests //tests:gpu_vllm_tests`，重点是 `tests/generation/execution/test_worker_sleep.py`、`tests/utils/test_cuda_memory.py`、`tests/trainers/test_fsdp_cuda_parking.py`；
   - 一次真实的 GRPO smoke（SANA 或 SD3.5，开 parity gate）。
3. **归因方法**（032 vLLM nightly triage 的做法）：同一个 HEAD、同一组测试，"全部锁定版本"对比"只升一个包"，就能直接判断是哪个包导致的回归，不用二分。
4. **验收**：parking、parity gate、checkpoint resume 都通过；GRPO smoke 的 logprob 差不劣于升级前。

### P2. Ray ≥2.59

1. 确认 `ray.init(address="local")` 在开启 token 认证后的行为：本地自动生成 token，还是需要显式配置。跨节点时 `RAY_ADDRESS` 路径要配 token，写进 launch 文档。
2. **验收**：`ray_slow_tests` 和一次单机 Ray rollout smoke 通过。

### P3. 收获清单（升级后才做，各自要有实测依据）

| 项 | 版本 | 用在哪里 | 何时做 |
|---|---|---|---|
| FSDP2 `set_separate_reduce_scatter_group(True)` | 2.13 | `vrl/trainers/fsdp.py` | 多节点 sprint P3 测出 RS 和 AG 串行时 |
| Inductor 计算-通信重叠默认开启 | 2.14 | 自动生效 | 升级后对比 FSDP 步时即可 |
| 小编译区域开销优化 | 2.14 | `regional_compile.py` | 升级后重测 regional compile 收益 |
| compile-on-one-rank | 2.14 | 多 rank 冷编译 | 与 compile sprint 的 P2 二选一，不要叠加 |
| `torch.while_loop` 可被 CUDA graph 捕获 | 2.14 | 去噪循环 | 吞吐 sprint 的 T5 探针证明 graph 有收益时 |
| `_record_memory_history(record_host=True)` | 2.14 | 观测 parking 和 CPU offload 的 pinned 内存 | 下一次 host 内存问题出现时 |
| `FlightRecorderHook` / `NanCheckHook` | 093（版本以 release notes 为准） | 多节点 sprint P0 | 替换环境变量方案 |
| `torch.distributed._transport`（单边 RDMA） | 093，实验性 | 权重同步 | 归 `SPRINT_weight_sync_transport_seam.md` |
| `torch.cuda.graph_annotations.mark_kernels` | 2.15 nightly | CUDA graph 下保留 `profile_range` 阶段名 | 只在真正上了 CUDA graph 时 |
| `nested_graph_breaks` | nightly | regional compile 深层 break | compile sprint P3 发现难以消除的 break 时 |

## 3. 不做

- 不把 `torch.cuda.*` 改成 `torch.accelerator.*`（042）：VRL 只跑 CUDA，这只是改名，按 CLAUDE.md 的清理规则不做。
- 这一轮不加 GPU CI lane。GPU 主机不在 CI 里；预检清单写在本文件，手动执行。
