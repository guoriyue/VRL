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

下列所有项目完成并验证之前，整体目标保持进行中。

- [x] 固定 Bazel 9.2.0、rules_python 2.3.3、Python 3.12.13；首个 sandbox 导入测试通过。
- [x] 从唯一依赖来源生成 Bazel 所需锁，避免手工双份版本。
- [ ] 分离主模型、vLLM、MAGI-1、CountGD 依赖目标。（主模型、vLLM 已分离；videoeval 受阻；MAGI-1、CountGD 未做）
- [ ] 固定 CUDA Toolkit、宿主编译器、Torch ABI、GPU 架构；真实扩展编译及执行。
- [x] 显式处理 Triton/JIT 编译依赖与缓存；驱动作为运行平台要求。
- [ ] 外部源码版本与补丁进入构建输入。
- [ ] CountGD 权重与依赖进入 Bazel，评分/服务等价性通过后删除旧安装器。
- [ ] 真实生成与训练步骤测试通过，非 CPU/mock 替代。
- [ ] Reward 服务集成测试通过。
- [ ] Ray、torchrun、跨节点产物交付与解释器选择明确并验证。
- [x] 普通 lint/配置/单元测试不下载所有模型权重。
- [ ] 干净 checkout 验证，无原有 venv、隐式 CUDA_HOME 依赖。
- [ ] CI、文档、运行入口迁移；已替代旧流程删除。

## 架构边界

保留评分协议、HTTP 服务、模型适配器和 Python 打包。
替换依赖准备与构建职责，不改训练算法。
版本/hash/schema 是真实输入边界，应保留；不在工作流中复制版本表。
Bazel 规则负责已声明依赖，宿主驱动与 GPU 不能被普通构建替代。
当前基础目标不代表 CUDA 或训练链路验收。

## 当前可运行入口

安装 Bazelisk 后，由 .bazelversion 选择仓库固定版本：

```bash
bazel test //tests/build:python_toolchain_test
```

首个测试仅证明受管解释器和显式 Python 源码依赖可用，不证明完整仓库隔离。
迁移期间 uv.lock 继续作为已有 Python 依赖解析的权威来源；
后续 Bazel 消费格式由导出生成，禁止手动编辑第二份版本表。

## 下一步

优先接入 rules_cuda 的 deliverable Toolkit（NVIDIA redistribution 清单及 hash）
和固定 C++ 编译工具链，执行真实 GPU 核函数测试。
然后接入锁定 Torch 依赖及实际 VRL 内核/生成/训练目标。

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
  RTX 5090 上执行矩阵乘；15 个 nvidia-* 包来自 `@pypi`，无 `.venv`、`CUDA_HOME`。
- `bazel test --config=cuda //tests/toolchains:cuda_execution_test` 在 13.0.2 上重新通过；
  aquery 确认 `nvcc 13.0.88`，ldd 确认 `libcudart.so.13` 来自 Bazel external。

标签说明：pip hub 里的包是 `@pypi//<pkg>` 或 `@pypi//<pkg>:pkg`，不是 `@pypi//:<pkg>`。

## VRL 库、测试与入口阶段（2026-09-12）

已通过（全部 `CUDA_VISIBLE_DEVICES=""`，清除 `CUDA_HOME`/`VIRTUAL_ENV`/`PYTHONPATH`）：
- `//:vrl`：`vrl/**` 源码 + 预设 YAML/资产；依赖是 `@pypi` hub 的 `all_requirements`，
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
- `@pypi_vllm` hub：uv.lock 的 `ar-vllm` extra + test 组（vllm 0.21.0 锁定 torch 2.11.0，
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
