# SPRINT: 可信的干净构建基线

状态：**DONE（2026-07-09）**。性质：验证与开发环境正确性；不增加模型、算法或训练功能。

## 0. 目标

让仓库声明的每一道门都真正执行、可在干净 source checkout 中复现，并且与 README
承诺的 clone + editable-install 使用方式一致。功能测试通过但 CI、锁文件或关键正确性测试
空转，不算绿色基线。

## 1. 起始证据

- `pytest -m "not e2e and not slow_test" -q`：1569 passed、7 skipped、23 deselected。
- `ruff check .`：40 errors；CI 的 quality job 使用相同命令，因此当前 main 必红。
- `uv lock --check` 失败；新增 extra 与锁文件漂移，VBench 的 transformers pin 与主训练栈冲突。
- 当前 package job 上传的 wheel 不含仓库级 `configs/`、`datasets/` 与 reward asset，安装后不能
  解析 README quickstart；README 本身只承诺 source checkout + editable install。
- 干净 `HF_HOME` 下 scheduler parity 为 2 passed / 33 skipped，测试依赖开发机缓存。
- real-checkpoint e2e 手工遍历 Hugging Face `models--*/snapshots` 并按 mtime 选版本，重复实现
  依赖内部协议。

## 2. 交付范围

1. 逐项清零 Ruff；删除问题已回答并记录的一次性 perf probe，保留仍服务活跃验证的 overlap probes。
2. 修复并冻结依赖解析；CI 使用 `uv.lock`，不再浮动 `pip install`。
3. 明确 VBench 的独立环境边界，不宣称与训练 extras 任意组合。
4. 删除不受支持的 wheel artifact job，改为干净 source checkout 的 editable-install/config-resolve smoke。
5. 把 scheduler config 固化为带来源的离线 test fixtures，干净 cache 下完整执行 parity。
6. e2e checkpoint 解析改用 huggingface_hub 官方 API。
7. 增加 raw-YAML portability gate，禁止用户 home 绝对路径进入长期配置。
8. 同步 adding-family、README、NORTH_STAR 与 sprint 状态，使文档反映 generic builder/registry 现实。

## 3. 应保留的边界

- `base -> recipe -> experiment` 三层配置；本 sprint 不 flatten。
- family registry 及其模块级 taxonomy：它是刻意隔离的运行时 source of truth。
- schema key、环境变量名、checkpoint 文件名、模型维度、测试 fixture 等真实 ALL_CAPS 边界。
- 协议 adapter、public facade、lazy-import builder 与从文件树派生集合的薄函数。
- still-active 的 single-GPU overlap probes 和真实 runtime telemetry。

## 4. 非目标

- 不做全仓 `ruff format`。
- 不发布 wheel；若未来需要，单独解决 configs、datasets、assets 与 third-party 的资源定位。
- 不跑新的 GPU 学习曲线；可信 video curve 是本 sprint 后的产品主线。
- 不清理 generation planner 的 metadata-only stage/axis 契约；它是独立架构 sprint。
- 不扩模型族、不重写算法、不改变 rollout/chunk/OOM 行为。

## 5. 验收

```bash
uv lock --check
uv sync --frozen --extra dev --extra cosmos
uv run ruff check .
uv run python -m vrl.config.lint
uv run pytest -m "not e2e and not slow_test" -q -ra
uv run pytest -m slow_test -q -ra
```

额外门：

- 空 `HF_HOME` 的 scheduler parity 不因缺 cache 跳过。
- clean source archive 能 editable-install 并解析 quickstart 配置。
- tracked YAML 不含用户 home 绝对路径。

## 6. 完成记录

- Ruff 从 40 errors 收敛到 0；删除已完成生命周期的 NCU occupancy 与旧 synthetic
  rollout-stage probe，保留仍被活跃 sprint 使用的 overlap/profiling 工具。
- `uv.lock` 可冻结解析 317 packages；主 `cosmos`/`reward` 环境统一在
  `transformers>=5.13,<6`，VBench 0.1.5 的 Transformers 4.33.2 环境由 uv conflict 明确隔离。
- CI 使用 pinned uv + frozen lock，初始化 submodules 并安装单一 `third_party/pyproject.toml`
  wrapper；不再上传不能解析仓库配置的 wheel，改为 source editable-install smoke。
- 15 个 checkpoint-owned scheduler config 作为带 repo/revision 的长期 fixture 入库；空
  `HF_HOME`、强制 offline 下 34 passed / 0 skipped。
- real-checkpoint e2e 改走 `snapshot_download(local_files_only=True)`；不再手走 Hugging Face
  cache 目录或按 mtime 猜 revision。
- raw YAML portability gate 入库，两份 DROID 配方改用 canonical 相对 checkpoint。
- adding-family、README roster/dependency contract、NORTH_STAR 与显著漂移的 sprint 目录已对账。
- CPU-safe 验收（按用户要求隐藏 CUDA、排除共享 Ray 启动路径）：config 32 passed；广泛回归
  1418 passed / 13 skipped / 14 deselected；GLM+Echo clean-env 32 passed；Cosmos+reward
  105 passed / 1 optional skip。
- GPU/CUDA 与 slow Ray lane 未在本轮执行：GPU 正被其他任务占用，且共享 Ray 验证可能干扰
  正在运行的训练。代码与 CI lane 已配置，待资源空闲时由 CI/人工补跑。
