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

## 4. 实施记录（2026-10-20）

### P0
- **`device_id`**：`fix(distributed): bind NCCL default groups to the rank device`。trainer 在 `set_device` 之后传 `context.device`；多卡 engine 的 nccl 组传当前设备；`tools/delivery/node_rank.py` 也一样。gloo 路径不变。
  - 风险：绑定设备后 NCCL 会提前建好通信器，之后建子组会走 `ncclCommSplit`。CP 的 gloo 子组只含部分 rank，不在组内的 rank 会做一次 nocolor split；每个 rank 的 split 次数相同，理论上不会死锁，但 cp≥2 且 dp≥2 时必须在多卡上确认。
- **torchcomms**：torch 2.11 确实读取 `TORCH_DISTRIBUTED_USE_TORCHCOMMS`，但只有 `torchcomms` 包能 import 时才生效。这个包不在 venv 里，也不在 `uv.lock` 里，所以现在设这个变量既没效果也不报错。P0 第 2 步要先把这个包加进 lock。

### P1 升级分支（独立分支，不合入 main）
- 分支 `worktree-agent-afc02c2a4afb728e9`，commit `6628b1e97 build(deps): torch 2.13, vLLM 0.32, Ray 2.59, transformers <5.20`。
- vLLM 0.32.0 要求 `torch==2.13.0`、`torchvision==0.28.0`、`transformers>=5.16.1,<5.20`；lock 解析结果是 torch 2.13.0+cu130、triton 3.7.1、ray 2.59.0、transformers 5.17.0（不变）、pillow 13.0.0。
- 独立环境在 scratch 目录的 `venv_upgrade`，`.venv` 没有动。
- CPU 测试在新旧环境下结果**完全一致**：主要几组都是 1822 passed / 8 skipped；其余目录 787 passed / 1 failed；Ray slow 113 passed / 1 failed。两个失败在 main 上同样存在，与升级无关。
- `bazel test //...` 在本机下载依赖时 `Unknown host`，没有跑成。
- **GPU 测试（RTX 5090，2026-10-21）**：
  - 主环境（torch 2.11 + vLLM 0.21 的独立环境）：`-m "gpu and not e2e and not slow_test" --distributed` 32 passed / 26 skipped。跳过的原因是需要 2 或 4 张卡、缺 bitsandbytes 等。
  - 升级环境：先是 1 failed。`test_cpu_offload_parking_releases_segment_pinned_by_blas_workspace` 断言 BLAS workspace 小于 32 MiB，而 torch 2.13 在 5090 上恰好分配 32 MiB（2.11 是 8.5 MiB），workspace 照样钉住 2 GiB 的显存段。在升级分支上修正（`911965ce7`）后，同样的命令 32 passed / 28 skipped。
- **真实 GRPO smoke（升级环境）**：SD3.5 LoRA，PickScore，2 个 epoch 再 resume 1 个 epoch。首步 parity 差为 0，之后两次是 9.0e-3 和 8.4e-3，与主环境同一配置的模式一致（LoRA 瘦 GEMM 的 batch 依赖，见 parity sprint）。从 `checkpoint-final` resume 的 `start_epoch=2`，曲线连续。
- 剩余：`bazel test //...` 的 CPU lanes（本机下载依赖时 `Unknown host`）。

### P2 Ray 2.59 token 认证（实测）
- `ray.init(address="local")` 新建集群时，Ray 会在 driver 里写入 `RAY_AUTH_MODE=token`（`ray.shutdown()` 之后也不清除），在 `~/.ray/auth_token` 生成 token，并打一条 WARNING。运行不会失败。
- 用显式地址连接已有集群时不会自动开启认证。不带 token 去连一个有 token 的集群，报的是普通的 `ConnectionError`，错误信息里不提认证。
- 跨节点时，head、worker 节点和 driver 上都要设 `RAY_AUTH_MODE=token` 并共享同一个 token，或者全局设 `RAY_AUTH_MODE=disabled`。VRL 是否显式设置这个变量，需要用户决定。

### 私有 API 核对（torch 2.13 / vLLM 0.32）
- FSDP2 的 `_get_fsdp_state`、`_fsdp_param_groups`、`reset_sharded_param`，DTensor 的 `_local_tensor`，以及 `torch._scaled_mm` 和 `OptimizedModule` 都还在。
- 2.13 的单数 `_fsdp_param_group` 变成了弃用 property；`parking.py:174` 先取复数形式，所以不受影响。
- vLLM 0.32 的 `allocator_and_pools` 从"每个 tag 一个元组"改成"每个 tag 一个列表"；`wake_up(tags)` 现在还会唤醒所有不在 `DEFERRABLE_TAGS` 里的 tag。目前一个进程只有一个 owner，所以基本等价，但要用 GPU 测试确认。

### 顺带发现
`.venv` 里已经没有 vllm 了（memory 里记录的手动安装被某次 sync 清掉了），所以 colocated 的 CuMem parking 在 `.venv` 下跑不起来。这次 GPU 运行用的是 scratch 目录里按 lock 同步、再单独装了 vllm 0.21 的 `venv_main`。按 CLAUDE.md 不能手动装进 `.venv`，需要用户决定要不要恢复。

## 5. 实施记录（2026-10-21，main 上不升级能做的部分）

升级分支仍未合入 main，本轮没有改 `pyproject.toml`、`uv.lock`、`MODULE.bazel.lock`。

### 逐项状态
| 项 | 状态 | 依据 |
|---|---|---|
| P0.1 `device_id` | 已完成 | `3e80af016`；`vrl/trainers/distributed.py:283`、`vrl/generation/execution/rank_group.py:77`、`tools/delivery/node_rank.py:21` |
| P0 风险：cp≥2 且 dp≥2 的子组 split | 本轮补了 GPU 测试，未在 GPU 上跑 | `91e3ff1b0`，见下 |
| P0.2 torchcomms 对照 | 需用户决定 | `torchcomms` 不在 lock 里，要先改 lock |
| P1 升级分支 | 在升级分支上，合入需用户决定 | `6628b1e97`、`911965ce7`，见下 |
| P1 `bazel test //...` CPU lanes | 未跑 | 本机拉依赖时 `Unknown host`；命令见下 |
| P2 token 认证行为 | 已实测（§4）；launch 文档本轮补上 | `a9a8f8fc8` |
| P2 VRL 是否显式设 `RAY_AUTH_MODE` | 需用户决定 | 见下 |
| 私有 API 核对 | 已完成（§4），本轮复核了源码 | 见下 |
| 弃用警告扫描 | 本轮完成；修了一处 Pillow 弃用 | `78d41642b` |
| P3 收获清单（10 项） | 不做 | 都要升级后实测，按表里"何时做"的门槛触发 |
| §3 不做（`torch.accelerator` 改名、GPU CI lane） | 不做 | 维持原结论 |
| `.venv` 缺 vllm | 需用户决定 | 见 §4"顺带发现" |

### 本轮提交
- `tests(trainers): check cp=2 x dp=2 groups form under a device-bound NCCL group`：`tests/trainers/test_context_parallel.py` 新增 4 卡 gpu 测试。它先 `TrainerProcessGroup.open`（NCCL，绑 `device_id`），再按 `FSDPStrategy.prepare_model` 的顺序建 FSDP mesh、CP gloo 组、CP mesh，每个组各做一次集合通信。读 torch 2.11 源码（`distributed_c10d.py` 约 1979-1997 行）确认：默认组绑了设备后，gloo 子组的非成员 rank 也会对 NCCL 默认通信器做一次 nocolor split，成员 rank 不做。`ContextParallelPeerGroup.create` 在 dp=2 时，每个 rank 恰好做一次。`TORCH_DIST_INIT_BARRIER` 默认关闭，`new_group` 末尾没有全局 store barrier，所以理论上不会死锁。同一测试体换成 4 个 gloo rank 在 CPU 上跑通了（只能验证断言和 mesh，验证不了 NCCL split）。
  - GPU 命令（4 卡主机，两种环境各跑一次）：`python -m pytest -q -p no:cacheprovider -m gpu tests/trainers/test_context_parallel.py::test_four_rank_nccl_cp_and_dp_groups_split_from_the_bound_default_group`
- `a9a8f8fc8 docs(cross_node): pick one Ray auth mode on every node before Ray 2.59`：`docs/training_examples/online_nft_kling_video_reward_cross_node/README.md` 新增 "Ray token authentication" 一节。读 Ray 2.59 源码后补充了 §4 没写到的一点：`ray start --head` 只在本机已有 token 时才开启认证（`ray/scripts/scripts.py:1069`）；`ray start --address` 和带显式地址的 `ray.init` 都不会自动开启（`ray/_private/worker.py:1968-1971`）；而任何一次单机 `vrl-train` 都会生成 `~/.ray/auth_token`（`worker.py:1868`）。所以一台跑过单机任务的 head 再起跨节点集群，会悄悄开启认证，而 worker 和 driver 不会，结果是一个不提认证的 `ConnectionError`。文档要求在所有节点上显式设 `RAY_AUTH_MODE`（`token` 加共享 token，或 `disabled`）。Ray 2.55.1 也认这几个变量（`RAY_AUTH_MODE`、`RAY_AUTH_TOKEN`、`RAY_AUTH_TOKEN_PATH`、`ray get-auth-token`），所以升级前后都适用。
- `78d41642b rewards(local_edit): read the gray plane with Image.get_flattened_data`：`Image.getdata` 在 Pillow 12.2（当前 lock）和 13.0（升级 lock）下都报弃用，Pillow 14 删除；`get_flattened_data` 两个版本都有。

### 弃用警告扫描
在 `.venv`（torch 2.11 / Ray 2.55.1）和 scratch 的 `venv_upgrade`（torch 2.13 / vLLM 0.32 / Ray 2.59）下，对本 worktree 的代码跑同一组 CPU 测试：`-m "not gpu and not e2e and not slow_test" -W default tests`。两边都是 2651 passed / 9 skipped。升级环境的警告是主环境的子集：没有新出现的 torch、vLLM、Ray 弃用。主环境独有的是 Ray 2.55 的 accelerator env var FutureWarning，以及子进程和 `/dev/null` 的 ResourceWarning。VRL 自己代码触发的弃用只有 `local_edit.py` 的 Pillow 一处，已修。其余都来自第三方或 vendored 代码：`torch.nn.utils.weight_norm`、`torch.jit.script_method`、diffusers 的 `torch_dtype`，不在本 sprint 范围。

### 私有 API 复核（不需要改 main）
- `vrl/models/parking.py:174-176`：torch 2.11 的 `FSDPState` 只有单数 `_fsdp_param_group`，2.13 才有复数 `_fsdp_param_groups`，单数成了弃用 property。现在的代码先取复数、取不到再退回单数，两个版本都正确，而且 2.13 下不会碰到弃用 property。
- `parking.py:781-787`：vLLM 0.32 的 `allocator_and_pools` 是 `dict[str, list]`，0.21 是 `dict[str, tuple]`。`pop(tag)` 在两个版本下都对。`building()` 是一次性的，每个 tag 只有一项，所以 `del retained` 的效果相同。

### 需用户决定
- **合入升级**：建议 cherry-pick `6628b1e97`（pyproject/uv.lock/MODULE.bazel.lock，加上 `tests/build/torch_cuda_test.py` 的版本断言）和 `911965ce7`（cuBLAS workspace 上限放宽到 64 MiB），不要直接 merge 分支。原因是分支基于 `cb380f0f6`，上面另有 6 个 commit（`e96e9a5f4`..`dd6577449`），它们在 main 上已经以 rebase 后的哈希存在（`f64e07084`、`e1ccfc288`、`cae124178`、`cd2480e0d`、`c59653968`；`dd6577449` 在 main 上拆成了 `e6dffcb64` 和 `488237944`）。这两个 commit 改的文件在 `dd6577449` 和当前 main 之间没有变化，cherry-pick 能直接应用。
- **合入后的跟进**（合入前都不能做，否则 2.11 会坏或文档不对）：删掉 `parking.py:175-176` 的单数回退；更新 cross_node runbook 里两处 "Ray 2.55.1" 的版本描述（新的 token 一节和 Teardown 一节）；更新本文件 §0 的锁定版本；跑 `bazel test //...` 和上面的 4 卡测试。
- **VRL 是否自己设 `RAY_AUTH_MODE`**：Ray 2.59 下，自有本地集群（`vrl/scripts/common/online.py:127` 的 `address="local"`）会开启认证、打 WARNING，并把 `RAY_AUTH_MODE=token` 留在 driver 环境里。另外按源码推断（未实测）：同一主机上多个 torchrun rank 各自 `ray.init` 时（如 `online_grpo_hpsv3_fsdp_4x_l40s`），如果 `~/.ray/auth_token` 还不存在，几个 rank 会同时生成并写同一个文件，存在一个很小的 token 不一致窗口。在 launcher 里显式设 `disabled` 或预先生成 token 都能消除，但这是安全策略选择，不在这里替用户定。
- **torchcomms（P0.2）**：要先把 `torchcomms` 加进 lock，然后用 `TORCH_DISTRIBUTED_USE_TORCHCOMMS=1 bazel test --config=gpu //tests:gpu_tests` 做对照。
- **`bazel test //...`**：需要能联网的主机，在升级分支上跑。
