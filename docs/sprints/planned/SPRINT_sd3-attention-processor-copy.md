# SPRINT: 删除 SD3 attention processor 的 upstream 复制品（planned）

状态：**planned / CPU-only verification**。单独提交，不与依赖或 probe 删除混合。

## 目标

VRL 的 `SD3JointAttentionProcessor` 与当前 diffusers
`JointAttnProcessor2_0` 执行相同的 q/k/v projection、joint concat、PyTorch SDPA、split 和 output projection。`TorchSDPAAttentionKernel` 只在中间转发相同参数。更关键的是，`SD3Transformer2DModel` 构造时已经给所有 attention block 安装 upstream processor；VRL 随后用复制品覆盖它，没有增加行为。

正确边界是：使用 upstream processor，并用 dependency contract test 固化假设。不要继续维护 dependency internals 的复制品。

## 改动

### 删除

- `vrl/nn/layers/attention/joint.py`
- `vrl/nn/kernels/attention/sdpa.py`
- `tests/nn/layers/test_sd3_joint_attention_processor.py`
- `vrl/models/families/sd3_5/model.py` 中：
  - `install_sd3_joint_attention_processor`
  - `_candidate_transformers`
  - 只负责 reinstall 的 rollout/replay `__init__` 与 `_set_transformer` override
- 两个 package `__init__.py` 中对应 re-export
- `quantized_sd3_forward_profile.py` 中 install import/guard
- denoise base docstring 中把 SD3 reinstall 当合法 override 的例子

### 替换测试

把 `tests/models/families/sd3_5/test_attention_processor_install.py` 重命名为
`test_attention_processor_contract.py`，并改成 stock dependency contract test：

1. 构造一个 CPU tiny `SD3Transformer2DModel`，不下载权重。
2. 断言 `attn_processors` 非空。
3. 断言所有默认 processor 都是 `JointAttnProcessor2_0`。

这不是保留旧实现测试，而是保护新的 dependency boundary。以后更新 diffusers lock 时，如果默认 processor 变化，CI 会要求显式重新评估 numerics，而不是静默换后端。

### 修正文档

更新两份未来设计稿，移除已删除类型作为“未来积木”的指针：

- `docs/sprints/parked/SPRINT_diffusion_native_transformer_executor.md`
- `docs/sprints/parked/SPRINT_attention_kernel_medium.md`

未来若确实需要可替换 attention backend，应从当时的 upstream API 与真实多消费者抽象重新设计，不能为未实现未来用途保留当前零消费者 wrapper。`done/` 与 `info/` 中有日期的历史测量记录保持原样。

## 能力边界

运行时数学在当前 resolved diffusers 版本上不变；仓库自己的 equivalence test 已证明两者数值一致。变化是 **processor ownership 从 VRL 移回 diffusers**。这不是“零能力损失”：未来 dependency 可能更换默认实现，因此必须保留上面的 contract test。

## 薄类型与常量判决

- 删除 `TorchSDPAAttentionKernel`：单一消费者、只服务复制 processor，不是现存 protocol 或跨 family abstraction。
- 删除 install/traversal helpers：它们只为冗余覆盖服务；包装层遍历不再有决策要做。
- 删除四个退化 override：移除 install 后只剩 `super()` 或基类同体赋值。
- **保留 `SD3_5ReplayModel`**：即使类体很薄，它仍是 registry identity、MRO composition 与 replay protocol 边界。
- 保留 attention package `__init__.py`；目录仍是包边界并含其他实现。
- 保留 `quantized_sd3_forward_profile.py`；它是长期 perf runner，只移除冗余 install。
- 本簇没有应迁移的业务 ALL_CAPS 表。

## Non-Goals

- 不改变 SD3 rollout/replay math、CFG、LoRA、quantization 或 compile 顺序。
- 不引入新的通用 attention manager/handler。
- 不 pin 一个更窄的 diffusers 版本来代替测试。
- 不删除历史 `done/`/`info/` 文档中的测量 provenance。

## 验收

```bash
rg -n \
  'SD3JointAttentionProcessor|TorchSDPAAttentionKernel|install_sd3_joint_attention_processor' \
  vrl tests docs/sprints/planned docs/sprints/parked
# expected: no matches

CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
  tests/models/families/sd3_5 \
  tests/models/steps/denoise \
  tests/nn \
  -q -m 'not gpu and not e2e and not slow_test'
```

对实际修改的 Python 文件执行 Ruff 四步：

```bash
sd3_files=(
  vrl/models/families/sd3_5/model.py
  vrl/models/steps/denoise/base.py
  vrl/nn/layers/attention/__init__.py
  vrl/nn/kernels/attention/__init__.py
  vrl/scripts/perf/quantized_sd3_forward_profile.py
  tests/models/families/sd3_5/test_attention_processor_contract.py
)
.venv/bin/ruff check --fix "${sd3_files[@]}"
.venv/bin/ruff format "${sd3_files[@]}"
.venv/bin/ruff check "${sd3_files[@]}"
.venv/bin/ruff format --check "${sd3_files[@]}"
```

## References

- `vrl/nn/layers/attention/joint.py`
- `vrl/nn/kernels/attention/sdpa.py`
- `vrl/models/families/sd3_5/model.py`
- `tests/nn/layers/test_sd3_joint_attention_processor.py`
- `tests/models/families/sd3_5/test_attention_processor_install.py`
- `vrl/scripts/perf/quantized_sd3_forward_profile.py`
- `pyproject.toml` (`diffusers` range)
