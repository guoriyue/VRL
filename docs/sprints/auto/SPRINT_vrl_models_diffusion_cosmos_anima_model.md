# SPRINT(auto): vrl/models/diffusion/cosmos/anima/model.py

状态: proposed (auto-generated per-file audit)
文件: `vrl/models/diffusion/cosmos/anima/model.py` (859 LOC)
角色判定: core
结论: consolidate

## 0. 一句话
这是一个真核心文件（Anima 的 RL 路径 model wrapper，被 registry 和 train/generate 脚本引用），但它把三个独立关注点塞进了一个 859 行的文件：RL 适配器层（`AnimaModel`/`AnimaReplayModel`）、一整套手写的 Qwen3→Cosmos text adapter 神经网络（`AnimaLLMAdapter`/`TransformerBlock`/`Attention`/`RotaryEmbedding` + rope helpers）、以及若干 checkpoint 装载/dtype 工具，建议把 adapter 网络抽到独立 `adapter.py`。

## 1. 现状（读代码得出）
文件实际承担三层不同抽象：

1. RL 路径 model 类（被引擎/trainer 直接用）：
   - `class AnimaModel(DiffusionModelBase)` (model.py:45) 实现 `from_spec`/`apply_lora`/`forward_step`/`replay_forward`/`decode_latents` 等接口方法。
   - `class AnimaReplayModel(AnimaModel)` (model.py:504) trainer replay-only 变体。
2. 一整套独立的 text-adapter 神经网络定义（与 RL 接口无关，纯 `nn.Module` 架构，复刻 ComfyUI 的 Anima adapter 以装载单文件 checkpoint 权重）：
   - `class AnimaLLMAdapter(nn.Module)` (model.py:637)
   - `class TransformerBlock(nn.Module)` (model.py:680)
   - `class Attention(nn.Module)` (model.py:728)
   - `class RotaryEmbedding(nn.Module)` (model.py:777)
   - `def rotate_half` (model.py:804) / `def apply_rotary_pos_emb` (model.py:810)
3. checkpoint/config 装载工具：`_load_anima_transformer` (model.py:573)、`_load_anima_llm_adapter` (model.py:597)、`_cosmos_t2i_transformer_config` (model.py:552)、`_qwen3_06b_config` (model.py:618)、`_resolve_torch_dtype` (model.py:839)。

第 2 类（adapter 网络 + rope helpers，约 637–818 行，~180 行）是一个自洽的子系统：它不引用 `AnimaModel`，只被 `AnimaLLMAdapter` 内部和 `_load_anima_llm_adapter` 引用。

## 2. 质疑点 / 改进机会
- 职责过载（god-file 倾向）：单文件同时定义"RL 接口适配器"和"被适配的底层网络架构"。`TransformerBlock`/`Attention`/`RotaryEmbedding` 是通用名字的 generic nn 模块，和 RL 接口代码放在一起会污染该文件的可读性与可 grep 性。证据：model.py:680 `class TransformerBlock`、model.py:728 `class Attention` 都是无前缀的泛名类，定义在一个名为 `model.py`（语义=RL model wrapper）的文件里。
- ALL_CAPS 规则：本文件**没有**违规的 ALL_CAPS 手抄结构。`_cosmos_t2i_transformer_config()` (model.py:552) 与 `_qwen3_06b_config()` (model.py:618) 返回的是模型架构维度（`in_channels`/`num_layers`/`hidden_size` 等），且必须与外部单文件 checkpoint 的权重 shape 精确对齐——属于 AGENTS.md 明确允许保留的"模型架构维度"边界，不要 derive、不要动。
- 命名：`AnimaModel.family = "cosmos-predict2-anima-t2i"` (model.py:48) 与 registry 的 `"cosmos-predict2-anima"`（registry.py:201）/`ANIMA_FAMILY`（runtime.py:24）不一致。已确认 `self.family` 只用于 `describe()`（model.py:210）和 `export_batch_context` 的 `model_family`（model.py:370）自描述，**不参与 registry 查找**，因此不是 bug；但 `-t2i` 后缀与注册名不同会让人误以为是查找键。建议在 sprint 范围内顺手加一行注释说明这是自描述 tag、非注册键，或与 task 维度对齐——非阻断项。
- 非死代码（已 grep 核实）：`AnimaReplayModel` 与 `load_anima_transformer_component` 在仓库内仅由同家族 `runtime.py` 引用，而 `runtime.py` 的 `build_anima_replay_runtime_bundle` 经 `scripts/diffusion/cosmos/train.py` 与 `rollouts/families/registry.py` 接入 import graph，故均为 live，**不可删**。

## 3. 建议动作
consolidate（拆分，不删不合并业务逻辑）：
- 新建 `vrl/models/diffusion/cosmos/anima/adapter.py`，迁入第 2 类全部符号：`AnimaLLMAdapter`、`TransformerBlock`、`Attention`、`RotaryEmbedding`、`rotate_half`、`apply_rotary_pos_emb`。
- `model.py` 改为 `from vrl.models.diffusion.cosmos.anima.adapter import AnimaLLMAdapter`（`_load_anima_llm_adapter` 只需要 `AnimaLLMAdapter`）。
- 在 `__init__.py` 的 `__all__` 中保留 `AnimaLLMAdapter` 的导出位置不变（当前由 model.py 的 `__all__` 导出，迁移后从 adapter.py re-export），避免破坏外部引用契约。
- 顺手在 model.py:48 `family` 处加一行 WHY 注释说明它是自描述 tag 而非 registry key。
- 非目标：不动 `_cosmos_t2i_transformer_config`/`_qwen3_06b_config`（架构维度边界）、不动 `AnimaReplayModel`（live、跨家族一致的 replay 形状）、不合并 `forward_step`/`replay_forward`（接口契约）。

## 4. 不动什么 / 为什么不是过度清理
- `AnimaReplayModel`（model.py:504）与 sibling `CosmosPredict2ReplayModel`（predict2/model.py:580）、`AnimaPipelineExecutor` 与 `CosmosPipelineExecutor` 形成跨家族一致形状。AGENTS.md "consistency over cleanup" 要求保留这种统一结构——即使方法体多为 `raise RuntimeError` 守卫，也不可拍平合并进 `AnimaModel`。
- 两个 `*_config()` 函数是架构维度边界，保留。
- 本 sprint 仅做"按关注点切文件"，不改任何运行逻辑、不删任何符号、不改 registry 字符串，属于结构卫生而非功能改写，风险低。

## 5. 验证
- 拆分后跑：`python -c "import vrl.models.diffusion.cosmos.anima as a; print(a.AnimaModel, a.AnimaPipelineExecutor)"` 确认 re-export 不破。
- `grep -rn "AnimaLLMAdapter\|TransformerBlock\|RotaryEmbedding\|apply_rotary_pos_emb" vrl/` 确认所有引用点已指向新模块且无悬空 import。
- `ruff check vrl/models/diffusion/cosmos/anima/` 确认无未用 import / 循环 import。
- 若有 Anima 相关测试：`pytest -k anima`；否则至少 import 冒烟 + `build_anima_runtime_bundle` 路径的现有脚本不报错。
