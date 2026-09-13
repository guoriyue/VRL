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
- [ ] 从唯一依赖来源生成 Bazel 所需锁，避免手工双份版本。
- [ ] 分离主模型、vLLM、MAGI-1、CountGD 依赖目标。
- [ ] 固定 CUDA Toolkit、宿主编译器、Torch ABI、GPU 架构；真实扩展编译及执行。
- [ ] 显式处理 Triton/JIT 编译依赖与缓存；驱动作为运行平台要求。
- [ ] 外部源码版本与补丁进入构建输入。
- [ ] CountGD 权重与依赖进入 Bazel，评分/服务等价性通过后删除旧安装器。
- [ ] 真实生成与训练步骤测试通过，非 CPU/mock 替代。
- [ ] Reward 服务集成测试通过。
- [ ] Ray、torchrun、跨节点产物交付与解释器选择明确并验证。
- [ ] 普通 lint/配置/单元测试不下载所有模型权重。
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
