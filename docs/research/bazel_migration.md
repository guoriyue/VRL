# Bazel 迁移验收记录

目标：统一依赖、构建、测试与运行；不通过 Bazel 包装旧安装器冒充迁移完成。

## 基线（2026-09-12）

- 仓库起点：614b8ece5，工作区干净；已有历史提交保留，不推送。
- Python 包：pyproject.toml + uv.lock；发布打包使用 setuptools。
- 外部源码：third_party Git submodule + editable 包装；videophy 尚未初始化。
- 依赖隔离：cosmos/reward、ar-vllm、videoeval 明确冲突；MAGI-1 使用独立环境。
- CountGD：自定义源码下载、补丁、Python 包锁、venv、安装验证。
- GPU：RTX 5090，驱动 580.173.02；本机可做真实 CUDA 验证。
- 现有入口：Makefile setup/verify、vrl-train、vrl-reward-service；CI 在 .github/workflows/ci.yml。
- 原仓库无 Bazel；原有 .venv 不作为迁移验收证据。

## 验收清单

状态（2026-09-13）。`[~]` = 部分完成，说明在括号里；未勾选项目见文末"未完成"。

- [x] 固定 Bazel 9.2.0、rules_python 2.3.3、Python 3.12.13；首个 sandbox 导入测试通过。
- [x] 从唯一依赖来源生成 Bazel 所需锁，避免手工双份版本。（uv.lock → 按 profile 导出；
  CountGD 的 `third_party/countgd/requirements.txt` 是它那一栈的唯一表）
- [~] 分离主模型、vLLM、MAGI-1、CountGD 依赖目标。（主模型 `//:vrl`、vLLM `//:vrl_vllm`、
  shared-GPU `//:vrl_shared_gpu`、CountGD `//:vrl_countgd` 已分离；videoeval 与 MAGI-1 未做）
- [~] 固定 CUDA Toolkit、宿主编译器、Torch ABI、GPU 架构；真实扩展编译及执行。
  （Toolkit 13.0.2 = torch cu130、LLVM 19.1.7 + sysroot、sm_120 已固定并在 GPU 执行；
  仓库没有自有 Torch 扩展源码，扩展编译链没有真实用例可验收）
- [x] 显式处理 Triton/JIT 编译依赖与缓存；驱动作为运行平台要求。
- [x] 外部源码版本与补丁进入构建输入。（CountGD 为 http_archive + patch + http_file；
  其他 vendored 上游仍是 git submodule，由 `make setup` 拉取）
- [x] CountGD 权重与依赖进入 Bazel，评分/服务等价性通过后删除旧安装器。
- [x] 真实生成与训练步骤测试通过，非 CPU/mock 替代。（5 个真实权重 case）
- [x] Reward 服务集成测试通过。（CPU lane 内真实子进程 + HTTP；CountGD 服务 smoke）
- [~] Ray、torchrun、跨节点产物交付与解释器选择明确并验证。（单机 torchrun 双 rank、
  本地 Ray 集群、python zip 在空环境运行均已验证；真实多节点未验证——只有一台机器）
- [x] 普通 lint/配置/单元测试不下载所有模型权重。
- [x] 干净 checkout 验证，无原有 venv、隐式 CUDA_HOME 依赖。（见下）
- [x] CI、文档、运行入口迁移；已替代旧流程删除。

## 架构边界

保留评分协议、HTTP 服务、模型适配器和 Python 打包。
替换依赖准备与构建职责，不改训练算法。
版本/hash/schema 是真实输入边界，应保留；不在工作流中复制版本表。
Bazel 规则负责已声明依赖，宿主驱动与 GPU 不能被普通构建替代。
当前基础目标不代表 CUDA 或训练链路验收。

## 当前可运行入口

安装 Bazelisk 后，由 .bazelversion 选择仓库固定版本：

```bash
make setup                                   # submodules + entry points
make verify                                  # uv lock --check && bazel test //...
bazel test --config=gpu //tests:gpu_tests //tests:gpu_vllm_tests
HF_HOME=... WM_REAL_MODEL_RL_CASES=cached bazel test --config=gpu --config=real_weights //tests:e2e_real_checkpoint_tests
bazel test //third_party/countgd:service_smoke_test
bazel run //:vrl_train -- --config <experiment>
bazel run //:vrl_supervise -- --config <experiment>
bazel run //:vrl_reward_service -- --config vrl/config/reward_service/<service>.yaml
bazel run //third_party/countgd:reward_service -- --config vrl/config/reward_service/countgd.yaml
bazel build --build_python_zip //:vrl_train
```

## CUDA 工具链阶段

已通过：
- Bazel 下载 CUDA 12.8.1 redistribution 的 cccl/cudart/nvcc；清单固定 SHA-256。
- LLVM 19.1.7 + 固定 Chromium Debian sysroot + libstdc++。
- 显式 compute_120:sm_120；实际 RTX 5090 执行加法核并验证回传数值。
- 清除 CUDA_HOME、CUDA_PATH、VIRTUAL_ENV、PYTHONPATH 后测试通过。
- aquery 确认编译命令使用 Bazel external 内的 nvcc、LLVM wrapper 和 sysroot。
- ldd 确认 cudart 来自 Bazel 产物；运行时 glibc 和 NVIDIA 驱动仍由 Linux 主机提供。

命令：

```bash
bazel test --config=cuda //tests/toolchains:cuda_execution_test
```

真实测试不是 mock，但它目前只是独立核函数，不是 Torch 扩展、
VRL 生成或训练验收。CUDA 清单的整项验收仍未勾选。
系统平台的动态 loader/glibc 兼容范围尚需在干净执行镜像验证。
CPU 目标无需 --config=cuda；GPU 测试标记 manual，普通 //... 不运行 GPU。

发现并解决：
- LLVM 18 官方 Linux 包要求未声明的 libtinfo.so.5，未在宿主机安装补救；
  改用 CUDA 支持的 LLVM 19 发行包。
- NVCC x86 拒绝 libc++，明确选择 sysroot 的 libstdc++。

## Python 依赖阶段（2026-09-12）

已通过：
- `uv.lock` 仍是唯一权威；`tools/dependencies/uv_exports.bzl` 用 Bazel 下载的固定 uv 0.10.2
  按 profile（main/tests/lint/vllm/videoeval）`uv export --frozen` 生成 requirements，
  仓库里不再保存第二份版本表。
- uv 导出会丢掉跨包 extras（`cuda-toolkit[cublas,...]` → `cuda-toolkit`），导致 torch
  拿不到 NVIDIA 运行库。`tools/dependencies/restore_export_extras.py` 用 uv.lock 已记录的
  依赖边把 extras 补回导出行；只复制锁中的事实。运行它的解释器是 rules_python 管理的
  `@python_3_12_13_host`。
- rules_python 2.3.3 原生 `pip.parse(uv_lock=)` 在本仓库不可用：锁含互斥 extras，
  `extra ==` 标记无法求值（已实测）。
- CUDA Toolkit 从 12.8.1 改为 13.0.2：uv.lock 的 torch 2.11.0 轮子绑定 cu130
  （`nvidia-cuda-runtime 13.0.96`），12.8 的 nvcc 与之 ABI 不匹配。CUDA 13 需要额外
  `crt`、`nvvm`、`culibos` 组件；宿主驱动 580.x 满足 CUDA 13。
- `bazel test //tests/build:torch_cuda_test`：沙箱内 torch `2.11.0+cu130` 导入并在
  RTX 5090 上执行矩阵乘；15 个 nvidia-* 包来自 `@vrl_pypi`，无 `.venv`、`CUDA_HOME`。
- `bazel test --config=cuda //tests/toolchains:cuda_execution_test` 在 13.0.2 上重新通过；
  aquery 确认 `nvcc 13.0.88`，ldd 确认 `libcudart.so.13` 来自 Bazel external。

标签说明：pip hub 里的包是 `@vrl_pypi//<pkg>` 或 `@vrl_pypi//<pkg>:pkg`，不是 `@vrl_pypi//:<pkg>`。

## VRL 库、测试与入口阶段（2026-09-12）

已通过（全部 `CUDA_VISIBLE_DEVICES=""`，清除 `CUDA_HOME`/`VIRTUAL_ENV`/`PYTHONPATH`）：
- `//:vrl`：`vrl/**` 源码 + 预设 YAML/资产；依赖是 `@vrl_pypi` hub 的 `all_requirements`，
  即 uv.lock main profile（core + cosmos + reward + reward-service + data + test + lint）
  的闭包，不在 BUILD 里重复任何版本或包名。
- `//third_party:vendored`：用 `imports` 替代 `pip install -e third_party`，
  条目与 `third_party/pyproject.toml` 的 `where` 一一对应。
- `//tests:*_tests`：每个测试包一个 lane，`-m 'not e2e and not slow_test'`，
  与 `make verify` 相同的选择；`//tests:config_tests` 不加标记过滤。
  15 个 lane 共约 3,700 个测试通过；`datasets/` 里已跟踪的 manifest 作为 runfiles。
- `//:config_lint`、`//:dead_flags`、`//tools/lint:ruff_check`：三个 lint 门。
  ruff 二进制来自锁定的 wheel（`<repo>/bin/ruff`），不是宿主安装。
- `bazel run //:vrl_train -- --help`、`bazel run //:vrl_reward_service -- --help`。

发现并处理：
- `tests/` 各包互相 import fixtures 甚至 test 模块，所以 `//tests:support` 带整棵树，
  每个 lane 只运行传入的文件。
- 测试直接 import `cloudpickle`，原来只是 .venv 里顺带装了（vllm/tilelang 的依赖）；
  已加入 pyproject `test` 组并重新 `uv lock`（只新增 2 行）。
- `tests/models/test_sequence_parallel.py` 调用了已改签名的 `destroy_rank_process_group()`，
  在 .venv 下同样失败；已修。
- ruff 门发现 `vrl/models/precision.py`、`vrl/nn/modules/ar_decoder.py` 未格式化；已修。
- uv 导出 repo rule 需要 `ctx.watch` 锁文件，否则 uv.lock 改动后不重新导出。

未做：GPU 路径（Triton、生成、训练步）、vLLM/videoeval/CountGD 独立环境、
Ray/torchrun 解释器、CI。

## 独立依赖栈与真实 GPU lane（2026-09-12）

已通过：
- `@vrl_pypi_vllm` hub：uv.lock 的 `ar-vllm` extra + test 组（vllm 0.21.0 锁定 torch 2.11.0，
  与主栈同一 torch，隔离的是 flashinfer/依赖闭包与 ABI 约束）。`//:vrl_vllm` 用
  `tools/python/defs.bzl::vrl_library` 把同一份源码接到这个 hub。
- `bazel test --config=gpu //tests:gpu_tests`（主 hub，`-m gpu`）：34 个真实 GPU 测试
  通过，含 Triton JIT 内核 `fused_linear_logprob`、fp4/fp8 量化内核。Triton 缓存与
  Inductor 缓存由 `tools/pytest/main.py` 指到 `TEST_TMPDIR`；ptxas 来自 triton wheel，
  `libcuda.so` 来自宿主驱动（执行平台要求，不是构建输入）。
- `bazel test --config=gpu //tests:gpu_vllm_tests`（vLLM hub）：33 个通过、0 跳过，
  含 Janus/NextStep vLLM paged-attention 对照 HF 单步、真实 vLLM ops、fp8 block kernel、
  CuMem 停车（`test_real_cumem_parking_with_another_process_allocation` 真实启动另一
  进程占显存）。
- `.bazelrc`：`precompile=force_disabled`（同一测试树对多个 hub 构建，.pyc 输出会冲突）；
  `--config=gpu` 只筛 `gpu` 标签并透传设备可见性。

受阻：
- videoeval（VBench 0.1.5 → transformers 4.33.2 → tokenizers 0.13.3）没有 cp312 wheel，
  rules_python 只能走 `pip wheel` 源码构建，需要非受管 Rust 工具链。未建 hub；
  `uv.lock` 里的 profile 保留。可选出路：rules_rust 驱动 maturin，或 vbench 升级。
- MAGI-1：官方 requirements 需要 flash-attn 2.4.2 + flashinfer cu124/torch2.4 源码构建，
  与 uv.lock 无交集（README 已要求用户自建 `third_party/MAGI-1/.venv`）。未建 hub。

## 真实权重生成 + 训练步（2026-09-13）

`bazel test --config=gpu --config=real_weights //tests:e2e_real_checkpoint_tests`
（`HF_HOME` 指向操作者的 HF 缓存，`WM_REAL_MODEL_RL_CASES` 选 case；权重是操作者选择的
checkpoint，不是构建输入；`HF_HUB_OFFLINE=1`）。每个 case 是一次真实 rollout（生成）+
一次优化器更新，断言可训练权重变化、reward std/advantage 非退化、loss/grad 有限、
rollout-vs-replay log-prob 一致。在 RTX 5090 上从 Bazel runfiles 通过：

| case | 模型 | 说明 |
|---|---|---|
| wan_2_1 | Wan2.1-T2V-1.3B | 视频，LoRA |
| sd3_5 | SD3.5-medium | 图像，LoRA |
| sd3_5_dance_grpo | SD3.5-medium | DanceGRPO |
| cosmos_predict2 | Cosmos-Predict2-2B-Video2World | 需要 vLLM CuMem 停车 |
| cosmos_predict2_kling_real_reward | 同上 + KlingTeam/VideoReward | 真实 reward 模型评分 |

依赖栈：`//:vrl_shared_gpu` = 主 hub + vLLM hub。两个 hub 出自同一 uv.lock，100 个共同包
版本全部一致（已核对），所以这等价于 README "shared-GPU" 的 `vllm --no-deps` 覆盖，
只是变成了声明式依赖。

修复的测试漂移（e2e 长期没跑，和运行时脱节）：`_IndexReward` 不是 `RewardFunction`；
`_SyntheticDiffusionReplayCollector` 还在实现旧的 `generate_rollout/evaluate_rollout`
而调度器已改调 `prepare_training_batches`。

仍失败、与 Bazel 无关：
- sd3_5_flow_dppo、sd3_5_grpo_guard：case 覆盖项与新校验冲突（严格 on-policy 且
  `ppo_epochs=1` 时比率恒为 1，算法拒绝启动）。需要改 case 配置，属算法测试维护。
- cosmos_anima、cosmos_anima_safe：训练步之后触发 `PrecisionDriftError`（已知未解）。
- janus_pro：需要 `deepseek-ai/Janus` 源码包（不在 PyPI，也不在 uv.lock，.venv 里同样缺）。
- cosmos_predict2_5：本机无缓存；nextstep_1：需要 64 GiB 显存。

## 解释器与分布式启动（2026-09-13）

rules_python 2.x 为每个 `py_binary`/`py_test` 在 runfiles 里生成一个 venv
（`<target>.runfiles/_main/<pkg>/_<name>.venv/bin/python3` → 受管 CPython 3.12.13），
其 `site-packages/_bazel_site_init.py` 把该 target 的全部依赖路径加入 `sys.path`。
因此 `sys.executable` 就是"正确解释器"，子进程用它启动时自动得到同一闭包：

- torchrun：`//tests/build:torchrun_test` 用 `sys.executable -m torch.distributed.run
  --nproc-per-node=2` 起两个 gloo rank，每个 rank 断言 `vrl` 来自 runfiles、
  `sys.executable` 不在任何 `.venv` 下，并完成 all_reduce。`vrl/scripts/supervise.py`
  就是这样拼命令的，所以 `bazel run //:vrl_supervise` 覆盖单进程与 DDP/FSDP 单机多卡。
- Ray：`//tests:ray_slow_tests` 45 个真实本地 Ray 集群测试通过（raylet + actor 由
  Bazel venv 解释器启动）。
- Reward 服务：`//tests:rewards_tests` 里以子进程启动 `vrl.rewards.service.server`
  并经 HTTP 探活。

跨节点产物：`bazel build --build_python_zip //:vrl_train` 生成自包含
`bazel-bin/vrl_train.zip`（3.3 GB：受管解释器 + torch/cu130 + 全部依赖 + vrl 源码与预设）。
在 `env -i` 只有 `/usr/bin/python3` 的环境里 `python3 vrl_train.zip --help` 正常，
zip 的 `__main__` 解包后执行内嵌的 3.12.13。远端节点两种交付方式：
(a) 同一路径挂载 runfiles 树 / 解包后的 zip（Ray worker 复用 driver 的 `sys.executable`
路径，README 已有此要求）；(b) 容器镜像内置 zip。宿主仍需提供 NVIDIA 驱动与 glibc ≥ 2.28
（torch 轮子 manylinux_2_28）。

未验证：真实多节点（本机只有一台机器）。MAGI-1 的 `model.python_executable` 仍指向
用户自建环境，未迁移。

## CountGD 与 NVIDIA 库来源（2026-09-13）

CountGD 进入 Bazel：
- `@countgd_src`：`http_archive` 固定 revision `b6f362b3` + sha256，qualified 的 torch-2
  兼容改动改为 `third_party/countgd/ms_deform_attn_torch2.patch`（由原字符串替换生成，
  已验证补丁后文件哈希与安装器记录一致）。Bazel 内置 patcher 不接受 hunk 中间的
  `\ No newline at end of file`，所以用 `patch_tool = "patch"`——这是唯一的宿主工具依赖。
- 8 个 Space 资产（cfg_app、checkpoint、BERT 6 文件）：`http_file` 固定 HF revision + sha256。
- `@vrl_pypi_countgd`：`third_party/countgd/requirements.txt`（带 hash，torch/torchvision/triton
  指向 download.pytorch.org cu128）是唯一版本表；原 `countgd_environment_lock.py` 的 Python
  表由它一次性生成后删除。补入 pydantic 及其 3 个依赖：`vrl.rewards.service.server`
  2026-09-05 起 import pydantic，而锁是 09-04 定的——旧安装器的服务 smoke 今天同样失败
  （已实测）。
- `//third_party/countgd:runtime`：组装规则复制源码 + 资产 + 4 个 audited build 目录副本，
  用 reward 模型自己的 `_runtime_tree_digest` 校验：133 个文件，
  `e41c4fd6…` == `COUNTGD_RUNTIME_TREE_SHA256`，并写 `install_manifest.json`。
- `//third_party/countgd:reward_service`、`:anima_exact_count_checkpoint_eval`、
  `:service_smoke_test`（manual；起服务、校验 model_version、HTTP 评分）。

等价性：同一 48 个 image/class 对（`docs/runs/.../eval_epoch_0180/0*.png` × sign/letter/person）
分别在 Bazel 栈和旧安装器（miniconda 3.12.2 + 锁定 wheel）环境里跑 `CountGDModel.detect`：
445 个检测框，bbox 与置信度逐位相同（最大差 0.0）。Bazel 服务经 HTTP 完成 48 个请求。
据此删除 `vrl/scripts/rewards/install_countgd.py`、`countgd_environment_lock.py` 及其测试。

NVIDIA 库来源（重要发现）：rules_python 把每个 wheel 放在各自目录，torch 轮子里
`$ORIGIN/../../nvidia/<lib>/lib` 的 RPATH 失效，动态链接器退回 ldconfig，**宿主的
`/usr/local/cuda-*` 库被按 soname 命中**（主 hub 之前映射了宿主的 `libcudart.so.13.0.96`，
CountGD hub 直接因宿主 cuda-12.1 的 `libcusparse/libnvJitLink` 崩溃）。修法：
`tools/python/sitecustomize.py`（随 `//tools/python:nvidia_preload` 的 `imports` 进入每个
vrl target 的 venv 路径）在解释器启动时按 torch 自己的顺序 `RTLD_GLOBAL` 预加载 hub 内的
运行库；子进程（torchrun rank、reward worker）同样生效。只加载 torch 清单里的库——
`libnvblas` 是 BLAS 拦截器，全局加载会劫持 CPU BLAS（已踩坑）。
`//tests/build:torch_cuda_test` 与 `torchrun_test` 断言进程映射中没有 `/usr/local/cuda`。

Torch 扩展编译链：仓库里没有自有 `.cu/.cpp`；两个 Triton 内核在运行时 JIT（已在 GPU lane
验证）。CountGD 的 `MultiScaleDeformableAttention` CUDA op 在 qualified 的 CPU 服务里不构建
（上游安装同样不构建）。因此"用 Bazel 编 Torch 扩展"没有真实用例可验收；rules_cuda 工具链
（13.0.2，与 torch cu130 同版）与 LLVM 已就绪，独立核函数在 GPU 执行通过。

`.bazelignore`：`.venv`（torch 包内自带 BUILD 文件）、`data`、`outputs`。
`bazel test //...`：21/21 通过。

## CI、文档与干净 checkout（2026-09-13）

- `.github/workflows/ci.yml`：`uv lock --check`；`bazel test //...` + 两个入口 `--help`
  （bazelisk + Bazel 缓存）；打包 job 不变（`uv build` + 轮子外部安装校验——保留的
  Python 打包能力）。原 uv sync / pytest / editable third_party 的 job 删除。
- `Makefile`：`setup` = 子模块 + 构建入口；`verify` = CI 门。`third_party/pyproject.toml`
  删除，`//third_party:vendored` 的 `imports` 是唯一的 vendored 源根清单。
- README Setup/Dependencies 改写为 target/栈/宿主要求表；`vdn_h3/vendor.py` 安装提示同步。
- 主 hub 扩到 pyproject 允许组合的全部 extras（新增 ocr、detection、optim8bit），
  GPU lane 从 34 增至 37（bitsandbytes 三个测试跑起来了）。

干净 checkout：`git clone` 到临时目录（无 `.venv`、无子模块），`env -i`（只有 HOME 和
bazelisk 所在 PATH）+ 全新 `--output_base`：
- `bazel test //...`：21 个 lane 中 20 个通过；`//tests:models_tests` 失败是 CausVid/Echo/
  MAGI-1 测试要读子模块源码。拉取 CausVid、joyai_echo、vdn-minimax-h3 后只剩 MAGI-1
  一个用例（它的子模块没拉）。子模块是源码 checkout 的一部分，`make setup` 负责。
- `bazel test --config=gpu //tests/build:torch_cuda_test //tests/toolchains:cuda_execution_test`
  在同一环境通过：torch 在 GPU 执行且进程内没有宿主 CUDA 库；独立核函数在 sm_120 执行。

## 未完成（明确记录）

1. **MAGI-1 环境**：官方 requirements 需 flash-attn 2.4.2 + flashinfer（cu124/torch2.4）
   源码构建，与 uv.lock 无交集；仍按 README 由用户自建 `third_party/MAGI-1/.venv` 并配置
   `model.python_executable`。可行路线：rules_python `pip.parse` + 预构建 wheel 仓库，或
   接受非受管的 CUDA 源码构建；两者都不是"固定工具链"，未做。
2. **videoeval 环境**：tokenizers 0.13.3 无 cp312 wheel（VBench → transformers 4.33.2），
   需要 Rust 源码构建；未建 hub。
3. **真实多节点**：只有一台机器。交付路径（zip / 同路径挂载）与解释器选择已验证到单机；
   Ray 多节点 runtime_env、NCCL 跨节点未跑。
4. **Torch C++/CUDA 扩展的 Bazel 编译**：没有真实用例（仓库无扩展源码；CountGD 的 CUDA op
   在 qualified CPU 服务里不构建）。rules_cuda + LLVM 工具链就绪但只有独立核函数证据。
5. **e2e 仍失败的 case**（与构建无关，已记录原因）：sd3_5_flow_dppo、sd3_5_grpo_guard、
   cosmos_anima、cosmos_anima_safe、janus_pro；cosmos_predict2_5 无缓存、nextstep_1 需 64 GiB。
6. **宿主工具**：GNU `patch`（CountGD 补丁的换行标记）、`git`（子模块）、NVIDIA 驱动、
   glibc ≥ 2.28。
