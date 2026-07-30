> **执行状态（2026-07-30）：审计完成，实施部分完成。** commit `300ef8c7`
> 已落地真实 VAE 状态断言及对应 fixture 子集；其余工作已重写为
> `docs/sprints/planned/SPRINT_tiny-real-diffusers-fixtures.md`。本文以下长文仅保留为
> 审计证据与原施工快照，不再作为待执行清单。

# SPRINT: tiny-real diffusers 对象与依赖它们的家族转换

状态：**planned**。Order 4 of 6，risk **medium**（依赖一次 lock 升级 + 删两处生产 guard + 删一个测试文件）。

> **一句话**：仓库已经把 `tests/models/steps/denoise/fixtures.py` 当成 tiny-real 的黄金标准（12 个
> `build_tiny_*` transformer builder，seeded、config-init、不下载、CPU）。本轨道把同一套机械扩到
> **VAE / scheduler / pipeline shell** 三类 diffusers 对象上，并顺着它们把 8 个测试文件里的
> 自声明替身换成真对象。共同前提只有一句：**真 diffusers 对象便宜到可以随手造**——实测一个真
> `AutoencoderKL` 2.3 ms、一个真 scheduler 0.2 ms、一个真 pipeline shell 0.3 ms、一个真
> `Cosmos3OmniPipeline` 2.9 ms。

## 0. 本 sprint 改变什么

| 维度 | 数字 |
|---|---|
| 转换的测试文件 | 8 个（其中 1 个删除、1 个新建） |
| 转换的测试函数 / 实例 | 21 个函数 / 40 个实例 |
| 删除的手写替身类 | 5 个（`_FakeVAE` ×2、`_FakePipeline`(frozen_offload)、`_TinyScheduler`、`_ConstantAnimaTransformer`）+ 4 处重复的手写 `DPMSolverMultistepScheduler` |
| 新增 fixture builder | 5 个（`build_tiny_autoencoder_kl` / `build_tiny_wan_vae` / `build_tiny_pipeline_shell` / `build_tiny_cosmos3_transformer` / `build_tiny_cosmos3_pipeline`）+ 2 个 per-family fixtures.py（nextstep_1、tests/scripts/eval 共用 builder） |
| 生产代码清理 | 2 处（`mochi/model.py:300`、`pixart_sigma/model.py:320` 的 `getattr(scheduler, "config", None)` 半个 guard） |
| **净新增覆盖** | cosmos3 家族**从字面意义的零覆盖**到 3 个真对象 T2 测试；mochi `prepare_replay` / pixart `prepare_replay` **今天在测试里从不执行**，转换后开始执行；anima `do_cfg=False` 分支全仓今天无人覆盖 |
| **默认车道 wall-clock** | **约 +0.15 s**（实测分项见 §7）。**不是** brief 里写的 +0.6 s——那 0.6 s 是 `standard_mochi_scheduler` 里 `diffusers.pipelines.mochi.pipeline_mochi` 的懒加载，默认车道**今天已经付过**（`tests/models/families/mochi/test_backbone_parity.py::test_standard_mochi_scheduler_descends` 直接调它）。cosmos3 的边际 import 实测 **3.1 ms**，不是 0.6 s。 |

### 翻案：上一轮审计对 `_FakeVAE` 的 keep 裁定判错

`docs/sprints/done/SPRINT_test_suite_tiny_real_and_fake_audit.md:538` 写着：

> `_FakeVAE`（`test_vae_decode_memory.py:17`、`test_decode_layout_parity.py`）——只记录
> `enable_tiling/enable_slicing`（policy 的全部契约）

这句话的前半是对的（policy 的契约确实只有这两个方法），后半的推论是错的。自声明的 fake 能证明的
是「**我们**调了两个方法名」，永远证明不了「**diffusers 真的有**这两个方法名、而且它们真的翻转
了状态」——而后半句才是生产会断的地方。实测（7 个真实 VAE 类全部通过）：

```
AutoencoderKL          tiling True slicing True   ->  enable_tiling() 后 use_tiling True
AutoencoderKLWan       tiling True slicing True
AutoencoderDC          tiling True slicing True
AutoencoderKLCosmos / KLHunyuanVideo / KLMochi / KLQwenImage  同上
```

换成真 `AutoencoderKL` 后断言从 `vae.calls == ["enable_tiling", "enable_slicing"]` 变成
`(vae.use_tiling, vae.use_slicing) == (True, True)`——**严格更强**，代价 2.3 ms。

**注意**：同一句 keep 裁定里的 `_IdentityDecodeVAE`（`test_decode_layout_parity.py:33`）**判对了**，
本轨道不动它（理由见 §8）。翻的是 `_FakeVAE` 那一半，不是整条。

---

## 1. 门禁步骤（唯一的前置，其余条目全部独立）

**只有 cosmos3 一条依赖它。§2–§6 的所有条目都可以在不做这一步的情况下独立落地。**

```bash
uv lock
uv sync
```

复核结论（全部实测，不是推断）：

- `pyproject.toml:43` 现为 `"diffusers>=0.39.0,<0.40"`；不能只依赖 lock 偶然选中
  0.39，因为 clean resolve 必须同样满足 Cosmos3 的生产 import floor。
- `uv.lock:1828` 当前钉 `version = "0.39.0"`，project metadata 也记录相同下限。
- 0.39.0 的 wheel 里确实有 cosmos3：
  ```
  diffusers/models/transformers/transformer_cosmos3.py
  diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py
  __init__.py 导出: Cosmos3AVAEAudioTokenizer, Cosmos3OmniTransformer, Cosmos3OmniPipeline
  ```
  而 0.38.0 里 `[n for n in dir(diffusers) if 'Cosmos' in n]` 没有任何 `Cosmos3*`——这就是
  `vrl/models/families/cosmos/cosmos3/model.py:88` 那句 `from diffusers import Cosmos3OmniPipeline`
  写在函数体内是因为 `cosmos` 是 optional extra，而不是因为 API 仍只存在于 git-main。
- **回归风险实测**：把 0.39.0 解压后用 `PYTHONPATH` 覆盖跑 `pytest tests/models tests/scripts
  tests/generation`（即全部会碰 diffusers 的目录），得到 **1708 passed / 3 skipped**，与 0.38.0
  基线的 **1708 passed / 3 skipped** 逐项一致。
  > 诚实限定：覆盖法跑出来 115.6 s vs 基线 106.3 s，这 9 s 是 scratch 目录冷 page cache + 无
  > 预编译 `.pyc` 的产物，不是版本差异。**真正的时间基线必须在 `uv lock` 之后重测**，不要引用这两个数。

---

## 2. 逐条清单

| 测试路径 | 今天假的是什么 | 变成什么 | 成本（实测） |
|---|---|---|---|
| `tests/models/steps/denoise/common/test_vae_decode_memory.py:22` | `_FakeVAE`：手写 `enable_tiling/enable_slicing`，只 append 字符串 | **T2** 真 `AutoencoderKL`（7 处构造点） | +20 ms |
| `tests/models/steps/denoise/test_frozen_offload.py:33` | `_FakePipeline.components`：手写 dict，塞一个 `object()` 当非 module | **T2** 真 `DiffusionPipeline` shell（`register_modules`），`.components` 天然带 1 个非 module + 1 个 `None` 槽 | +10 ms |
| `tests/models/interfaces/test_minimal_replay_runtime_wiring.py:217` | `_TinyScheduler`：`timesteps = tensor([1.0])`，**没有 `.config`** | **T2** 真 `FlowMatchEulerDiscreteScheduler` / `DDIMScheduler` / `UniPCMultistepScheduler` / `CogVideoXDDIMScheduler` | +8 ms |
| `vrl/models/families/mochi/model.py:300`、`vrl/models/families/pixart_sigma/model.py:320` | 生产代码为 fake 弯腰：`getattr(scheduler, "config", None)` | 删掉 `config is not None` 那半个 guard | −4 行 |
| `tests/models/families/cosmos/anima/test_forward_step.py`（整文件） | `_ConstantAnimaTransformer`：手写 5-kwarg 签名 + 常量输出 | 删除；ordering 主张搬进 `test_backbone_parity.py`，并补 `do_cfg=False` 分支 | +5 ms |
| **新建** `tests/models/families/cosmos/cosmos3/test_backbone_parity.py` | 今天**零覆盖**（全仓 14 处 cosmos3 引用全是注册表/schema，家族自身代码零执行） | **T2** 真 `Cosmos3OmniPipeline`（真 transformer + 真 `AutoencoderKLWan` + 真 `UniPCMultistepScheduler` + 真 tokenizer） | +60 ms |
| `tests/models/families/nextstep_1/test_model_loading.py:94-101` | `class VAE`：`decode()` 返回硬编码尺寸的 zeros | **T2** 真 f8 `AutoencoderKL`（几何由真上采样算出） | +21 ms |
| `tests/scripts/eval/test_sana_aesthetic_checkpoint_eval.py:754/807/844`、`tests/scripts/eval/test_sana_checkpoint_compare.py:36` | 4 份手写 `class DPMSolverMultistepScheduler`，`.config` 直接由 `SCHEDULER_PROTOCOL` 派生（**循环**） | **T2** 一个共享 builder，返回真 `DPMSolverMultistepScheduler` | +10 ms |
| `tests/scripts/test_wan_dpo_encoders.py:20` | `_FakeVAE`：手写 `config.z_dim/latents_mean/latents_std` + 脚本化 `latent_dist.sample` | **T2** 真 `AutoencoderKLWan`（config 字段来自真 diffusers config） | +20 ms |

---

## 3. VAE 系：`build_tiny_autoencoder_kl` / `build_tiny_wan_vae`

### 3.1 新 builder（落在 `tests/models/steps/denoise/fixtures.py`，与 12 个 transformer builder 并列）

```python
def build_tiny_autoencoder_kl(
    *, seed: int = 0, downsamples: int = 1, latent_channels: int = 4,
) -> Any:
    """A real tiny ``AutoencoderKL`` on CPU, random-init from a seed.

    ``downsamples`` is the number of 2x spatial steps, so the decode geometry a
    test asserts on is COMPUTED by diffusers (latent HxW * 2**downsamples), never
    declared by the fixture. f8 (``downsamples=3``) is the NextStep tokenizer
    geometry; the default f2 is enough for flag/state tests.
    """
    from diffusers import AutoencoderKL
    torch.manual_seed(seed)
    blocks = downsamples + 1
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * blocks,
        up_block_types=("UpDecoderBlock2D",) * blocks,
        block_out_channels=(4,) * blocks, layers_per_block=1,
        latent_channels=latent_channels, norm_num_groups=2, sample_size=32,
    )
```

实测：`downsamples=1` → 3,135 参数 / 2.3 ms 构建；`downsamples=3, latent_channels=16` →
9,387 参数 / 3.4 ms 构建 / 6.5 ms decode，`(2,16,4,4) -> (2,3,32,32)`。**seed 决定性已实测**：
同 seed 两次 init 的全部 `state_dict` 张量 `torch.equal` 为 True。

`build_tiny_wan_vae` 同形，返回真 `AutoencoderKLWan(base_dim=4, z_dim=…, dim_mult=[1,1],
num_res_blocks=1, latents_mean=…, latents_std=…)`——10,433 参数 / 2.7 ms，`config.z_dim` /
`config.latents_mean` / `config.latents_std` 是**真 diffusers config 字段**。

### 3.2 `test_vae_decode_memory.py` — 7 处构造点

**今天的断言**（`:34-39`）：

```python
def test_configure_vae_decode_memory_calls_declared_methods() -> None:
    """Checks configure VAE decode calls declared methods."""     # docstring 只复述函数名
    vae = _FakeVAE()
    configure_vae_decode_memory(vae, VaeDecodeMemory(tiling=True, slicing=True), owner="test VAE")
    assert vae.calls == ["enable_tiling", "enable_slicing"]
```

它证明的是：我们按顺序调了两个我们自己定义的方法名。它证明不了 diffusers 还有这两个方法，
更证明不了它们改了状态。生产侧 `vrl/models/steps/denoise/common/vae_decode_memory.py:_call_required`
用 `getattr(target, method_name, None)` + `callable()` 探测——**探测的对象在测试里是我们自己写的**。

**替换**（同名不变，docstring 换成不变量）：

```python
def test_configure_vae_decode_memory_flips_the_real_diffusers_tiling_flags() -> None:
    """The two knobs must land on diffusers' own state, not just be called.

    A hand-written double can only prove we invoked two method NAMES; it cannot
    prove diffusers still exposes them or that they change anything. Both are
    what breaks in production when the dependency moves.
    """
    vae = build_tiny_autoencoder_kl()
    assert (vae.use_tiling, vae.use_slicing) == (False, False)

    configure_vae_decode_memory(vae, VaeDecodeMemory(tiling=True, slicing=True), owner="test VAE")

    assert (vae.use_tiling, vae.use_slicing) == (True, True)
```

**丢掉的那个断言，以及为什么可以丢**：`calls == [...]` 里的**顺序**主张没有生产对应物。实测真
`AutoencoderKL` 两种顺序结果相同：

```
tiling-then-slicing  use_tiling True  use_slicing True
slicing-then-tiling  use_tiling True  use_slicing True
```

顺序是 fake 独有的可观测量，不是不变量。**这不是删覆盖，是删一个假不变量。**

其余 6 处（`:35 / :163 / :178 / :216 / :312 / :582-583`）同样处理，断言从
`bundle.model.vae.calls == ["enable_tiling"]` 变成 `(vae.use_tiling, vae.use_slicing) == (True, False)`
——多测一件事：**slicing 没被误开**。今天 `calls == ["enable_tiling"]` 也能表达这点，所以这里是平手，
真正的增益仍是「方法名与状态是 diffusers 的，不是我们的」。

`test_configure_vae_decode_memory_raises_on_missing_method`（`:43-47`，传 `object()`）**保持不动**：
它是 T1，`object()` 本来就是真的「没有这个方法的对象」。

顺带的 clarity：本文件 5 条 `"""Checks …"""` docstring（`:34 :44 :177 :212 :307`）纯复述函数名，
按 AGENTS.md 一律改成写不变量或删掉。

**verify**：
```bash
.venv/bin/python -m pytest tests/models/steps/denoise/common/test_vae_decode_memory.py -q -p no:randomly --durations=10
```

### 3.3 `tests/scripts/test_wan_dpo_encoders.py` — 真 `AutoencoderKLWan`

**今天**（`:20-32`）手写 `config = SimpleNamespace(z_dim=…, latents_mean=[0.5,-0.25], latents_std=[2.0,4.0])`
和 `encode()` 返回一个固定 `raw`。生产侧 `vrl/scripts/families/wan_2_1/train_dpo.py:39-49` 读的正是
`vae.config.z_dim / latents_mean / latents_std`——**这三个字段名是 diffusers 的 config schema，
今天由测试自己定义**。

**替换**：真 `AutoencoderKLWan`，`raw` 用同 seed 二次抽样对齐（`latent_dist.sample()` 消耗的是
`torch.randn`，同 seed 同 shape 结果一致）：

```python
torch.manual_seed(1234)
z = encode_pixels(pixels)
torch.manual_seed(1234)
raw = vae.encode(x).latent_dist.sample()
torch.testing.assert_close(z, (raw - mean) / std)
torch.testing.assert_close(z * std + mean, raw)          # decode 反归一化的往返
assert not torch.allclose(z, (raw - mean) * std)         # 历史 bug 的假侧
```

**已实测跑通**：1 passed，0.02 s。三条断言（含那条守历史 bug 的假侧断言）全部保留。

**verify**：`.venv/bin/python -m pytest tests/scripts/test_wan_dpo_encoders.py -q -p no:randomly`

---

## 4. Pipeline shell 系：`build_tiny_pipeline_shell`

### 4.1 为什么 `.components` 必须是真的

`vrl/models/steps/denoise/base.py:515-522` 的 `move_frozen_components` 是**纯 diffusers 契约消费者**：

```python
components = getattr(pipeline, "components", None)
if not isinstance(components, Mapping):
    return
transformer = getattr(self, "transformer", None)
for module in components.values():
    if not isinstance(module, nn.Module) or module is transformer:
        continue
```

它依赖三件 diffusers 的事实：(a) `.components` 是 Mapping；(b) 里面会混进**非 module** 项
（scheduler / tokenizer）；(c) 里面可能有 **`None` 槽**（可选组件）。今天的
`_FakePipeline`（`test_frozen_offload.py:33-43`）只手写了 (a) 和 (b)，(c) 完全没有。

实测真 `DiffusionPipeline` 子类（`register_modules`，全部形参无默认值）：

```
components keys: ['scheduler', 'text_encoder', 'transformer', 'vae']
None slots:      ['text_encoder']
non-module:      ['scheduler']
build:           0.0003 s
```

真 `Cosmos3OmniPipeline` 更极端——**两个** `None` 槽（`sound_tokenizer`、`safety_checker`）+
**两个**非 module（`text_tokenizer`、`scheduler`）。

### 4.2 替换

新 builder：

```python
def build_tiny_pipeline_shell(*, transformer, vae, scheduler, text_encoder=None) -> Any:
    """A real ``DiffusionPipeline`` whose ``.components`` diffusers itself derives.

    Every slot is a REQUIRED __init__ parameter, so diffusers' ``_get_signature_keys``
    keeps them all in ``.components`` — including the ``None`` one. That is the shape
    ``move_frozen_components`` has to survive and a hand-written dict cannot promise.
    """
```

`test_frozen_offload.py` 的两个测试改成：真 tiny transformer + 真 tiny `AutoencoderKL` + 真
`FlowMatchEulerDiscreteScheduler` + `text_encoder=None`，断言从「`_RecordingModule.to_devices`
记了什么」改成**真的搬走了**（用 `"meta"` 设备，`next(vae.parameters()).device.type == "meta"`，
`next(transformer.parameters()).device.type == "cpu"`），并新增两条对 diffusers 契约本身的断言：

```python
assert [k for k, v in pipe.components.items() if v is None] == ["text_encoder"]
assert [k for k, v in pipe.components.items()
        if v is not None and not isinstance(v, nn.Module)] == ["scheduler"]
```

**已实测跑通**（含 `move_frozen_components("meta")` 的真设备迁移）：passed，< 5 ms。

`_RecordingModule`（`:22-32`）随之删除——它存在的唯一理由是 fake pipeline 里的模块不是真的、
只能靠记录 `.to` 参数间接验证。

**verify**：`.venv/bin/python -m pytest tests/models/steps/denoise/test_frozen_offload.py -q -p no:randomly`

---

## 5. Scheduler 系：`_TinyScheduler` 退役 + 两处生产 guard 清理

**这两件事不能拆开做**，顺序是「先换测试，后删 guard」。

### 5.1 `_TinyScheduler` 是那两处生产 guard 的唯一存在理由

```python
# tests/models/interfaces/test_minimal_replay_runtime_wiring.py:217-224
class _TinyScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1.0])
        self.sigmas = torch.tensor([1.0])
```

**它没有 `.config`。** 而生产代码为此弯了腰，两处，注释还写得明明白白：

```python
# vrl/models/families/mochi/model.py:296-300
# A scheduler without .config is a hand-injected test double — only
# real diffusers schedulers carry the shipped (inverted) config that
# needs standardizing.
config = getattr(self._scheduler, "config", None)
if num_steps is not None and config is not None:
```

`vrl/models/families/pixart_sigma/model.py:316-320` 逐字同构。

后果不是「多了一行防御」，是**这两条生产路径在测试里从不执行**。实测把 `_TinyScheduler` 换成真
scheduler 后（12 个 descriptor 家族全过）：

```
mochi          FlowMatchEulerDiscreteScheduler  numel 2   [1000.0, 975.0]   <- standard_mochi_scheduler 真的跑了
pixart_sigma   DDIMScheduler                    numel 2   [500, 0]          <- pixart_ddim_scheduler 真的跑了
其余 10 家      (未标准化，numel 1000)
```

今天这两行是 dead branch；换掉替身，它们就活了。

### 5.2 动作

1. 删 `_TinyScheduler`（`:217-224`），6 处注入点（`:334 :341 :375-377 :410 :447 :487`）改成真类：
   - `load_flow_match_scheduler` → `FlowMatchEulerDiscreteScheduler()`
   - `load_diffusers_scheduler` → `getattr(diffusers, class_name)()`（registry 的
     `scheduler_classname` 就是它自己的 source of truth：cogvideox→`CogVideoXDDIMScheduler`、
     pixart_sigma→`DDIMScheduler`、wan_2_1→`UniPCMultistepScheduler`，其余为 `None` 走 flow-match）
2. `:391` 的 `assert bundle.scheduler.timesteps.tolist() == [1.0]` 是**纯 echo**——`[1.0]` 是
   `_TinyScheduler.__init__` 写死的值，wan 的 `prepare_replay` 根本不碰 timesteps。改成测试名字本来
   就承诺的那件事：`assert isinstance(bundle.scheduler, UniPCMultistepScheduler)`。
   （同一测试里的 `assert scheduler_classes == ["UniPCMultistepScheduler"]` 是对**我方**代码的真断言，
   保留。）
3. 删两处生产 guard 的 `config is not None` 那半：
   ```python
   num_steps = build.num_steps
   if num_steps is not None:
       self._scheduler = standard_mochi_scheduler(self._scheduler.config, int(num_steps), build.device)
   ```
   **`num_steps is not None` 那半必须保留**——`vrl/models/steps/denoise/build.py:96-99` 记录了它的
   真实生产者：`# If None, a caller such as the DPO trainer sets scheduler timesteps itself.`

### 5.3 两条断言纪律（实测结论，写进新代码的注释里）

- **不要断言 `scheduler.timesteps.numel() == 0`。** 每个 diffusers scheduler 在 `__init__` 就填满
  训练 ladder，实测 4 个类全是 **1000**。（对照：`TNA-13` 那条的正确用法是先
  `set_timesteps(0)` 再断言 0——那是**调用后**的状态，不是构造后的。）
- **不要断言字面 ladder 数值。** 实测 `set_timesteps(2)` 给出
  `FlowMatchEuler [1000.0, 1.0]` / `DDIM [500, 0]` / `UniPC [999, 500]` / `CogVideoXDDIM [500, 0]`
  ——全部与 diffusers 版本耦合。断**类身份** + `num_inference_steps` + `numel`。

**verify**：
```bash
.venv/bin/python -m pytest tests/models/interfaces tests/models/families/mochi tests/models/families/pixart_sigma tests/models/steps/denoise -q -p no:randomly
```
（实测 prototype：12 个家族全过，2.12 s 含冷启动；warm 边际 8 ms。）

---

## 6. 家族转换

### 6.1 anima：删 `test_forward_step.py`，主张搬家而不是丢弃

**前提必须先落地**，否则就是删覆盖。`test_forward_step.py`（40 行，1 个测试）里有 2 条
`test_backbone_parity.py` **没有**的主张：

1. **CFG 分支顺序**：`_ConstantAnimaTransformer` 让 `value = encoder_hidden_states[:, :1, :1]`，
   prompt=ones / negative=zeros，于是 `noise_pred_cond == ones` / `noise_pred_uncond == zeros`
   钉死了「第 0 次 forward 吃的是 positive embeds」。parity 文件只断言 `cond != uncond`，不区分顺序。
2. **`do_cfg=False` 分支**：两个文件都没有。全仓今天无人覆盖
   （`vrl/models/families/cosmos/anima/model.py:328-329` 的 `noise_pred_uncond = torch.zeros_like(cond)`）。

**动作**（先加后删）：

`record_forward_calls`（`fixtures.py:391`）本来就捕获全部 kwargs，所以 ordering 主张可以在真
transformer 上直接表达，**不需要常量输出的替身**：

```python
def test_cond_branch_runs_first_on_the_positive_embeds() -> None:
    """Branch order is load-bearing: the CFG combine at model.py:342 assumes
    calls[0] is cond. A swap keeps every shape and dtype and only shifts the
    guided result — invisible without pinning WHICH embeds each call received."""
    ...
    assert calls[0]["encoder_hidden_states"] is state.prompt_embeds
    assert calls[1]["encoder_hidden_states"] is state.negative_prompt_embeds

def test_cfg_off_runs_one_forward_and_reports_a_zero_uncond() -> None:
    """CFG-off must not fabricate a second forward; noise_pred is the raw cond."""
    assert len(calls) == 1
    torch.testing.assert_close(out["noise_pred"], out["noise_pred_cond"])
    assert torch.count_nonzero(out["noise_pred_uncond"]) == 0
```

**已实测跑通**：2 passed；第二个测试 < 5 ms（第一个 1.30 s 是孤立进程里
`CosmosTransformer3DModel` 的懒加载，在真实套件里由 `test_backbone_parity.py` 自己付掉，边际约 5 ms）。

之后删除 `tests/models/families/cosmos/anima/test_forward_step.py` 整个文件（连同
`_ConstantAnimaTransformer`）。净测试数 **+1**（1 → 2 新增，−1 删除，parity 文件从 1 个测试变 3 个）。

顺带 clarity：`test_backbone_parity.py:58` 的 `"""Checks Anima forward step runs real unbatched
CFG on a real backbone."""` 也是复述函数名，改写。

**verify**：`.venv/bin/python -m pytest tests/models/families/cosmos -q -p no:randomly --durations=10`

### 6.2 cosmos3：从字面意义的零覆盖到 3 个 T2 测试（依赖 §1）

**今天的覆盖**：`grep -rn cosmos3 tests/` 共 14 处命中，逐条读过——全部是 reward 模块同名、
schema 表项、registry 表项、MRO 名单、以及 `test_vae_decode_memory.py:295-297`（那里
`Cosmos3Model` 被 `monkeypatch.setattr` 整类换掉）。**家族自身 407 行代码零执行**。而
`model.py:1-27` 的模块 docstring 声称 "verified against diffusers@main source"——那是一句注释，
不是一个测试。

**替换后能跑通的东西（已完整实测，非设计稿）**：

```
prepare_sampling  0.0082s   latents (1, 4, 5, 16, 16)  num_noisy_vision_tokens=320  do_cfg=True
forward_step      0.0042s   noise_pred / _cond / _uncond 全部 (1, 4, 5, 16, 16)
CFG combine       uncond + g*(cond - uncond)  一致
decode_latents    (…, 32, 32)
```

这条路径真正走过的**生产逻辑**：`pipe.prepare_latents` 的 **12 字段按位解包**（`model.py:172-177`
的注释明说类型标注是过时的 11 元组）、`pipe._prepare_text_segment` / `_prepare_vision_segment`、
`_assemble_packed_static`、transformer 的 **11 个 kwarg 名**、返回值的 **3 元组解包**、
`pipe._mask_velocity_predictions`。这些今天全靠一段 docstring 担保。

**新 builder**（放 `tests/models/steps/denoise/fixtures.py`，与其余 12 个并列）：

```python
def build_tiny_cosmos3_transformer(*, seed: int = 0) -> Any:
    """Tiny real ``Cosmos3OmniTransformer`` (33K params) on CPU, cache-free.

    ``patch_latent_dim`` must equal ``latent_channel * latent_patch_size**2`` and
    ``rope_scaling['mrope_section']`` must sum to ``head_dim // 2`` — both are
    construction parameters here, so the tiny geometry is honest, not a coincidence.
    """
    from diffusers import Cosmos3OmniTransformer
    torch.manual_seed(seed)
    return Cosmos3OmniTransformer(
        head_dim=16, hidden_size=32, intermediate_size=64,
        latent_channel=4, latent_patch_size=2, patch_latent_dim=16,
        num_attention_heads=2, num_hidden_layers=1, num_key_value_heads=1,
        vocab_size=66, rope_scaling={"mrope_section": [4, 2, 2]},
    )


def build_tiny_cosmos3_pipeline(*, seed: int = 0) -> Any:
    """Tiny real ``Cosmos3OmniPipeline``: real transformer + real AutoencoderKLWan
    + real UniPCMultistepScheduler + a real in-memory tokenizer.

    The tokenizer is a genuine ``PreTrainedTokenizerFast`` built from a
    ``tokenizers.Tokenizer`` object via the library's own public constructor — the
    pipeline's ``__init__`` really calls ``convert_tokens_to_ids('<|vision_start|>')``
    and reads ``eos_token_id`` (pipeline_cosmos3_omni.py:403-406), so a stand-in
    would put those two contract points back out of reach.
    """
```

实测（warm，即套件里已 import 过 diffusers/transformers/tokenizers 的情况）：
**边际 import 3.1 ms，pipeline 构建 2.9 ms**，transformer 33,136 参数。

**新测试文件** `tests/models/families/cosmos/cosmos3/test_backbone_parity.py`，3 个测试：

1. `test_prepare_sampling_packs_one_sample_from_the_real_pipeline_builders` —— 钉
   `latents.shape[0] == 1`（`runtime.py:53` 的 `samples_per_chunk=1` 是硬约束）、
   `timesteps.numel() == request.num_steps`、packed_static 的键集合。
2. `test_forward_step_returns_raw_velocity_and_the_cosmos3_cfg_combine` —— 钉
   `uncond + g*(cond-uncond)`（**不是** predict2 的 `cond + g*(cond-uncond)`）与 shape。
3. `test_decode_latents_denormalizes_with_the_real_vae_stats` —— 走真
   `pipe._vae_latents_mean` / `_vae_latents_inv_std` / `video_processor`。

**已实测跑通**：3 passed。

**verify**：
```bash
uv lock --upgrade-package diffusers && uv sync
.venv/bin/python -m pytest tests/models/families/cosmos/cosmos3 -q -p no:randomly --durations=5
.venv/bin/python -m pytest tests -q -p no:randomly      # 全量零回归门禁
```

### 6.3 nextstep_1：真 f8 VAE 让 decode 几何变成算出来的

**今天**（`test_model_loading.py:94-101`）：

```python
class VAE:
    dtype = torch.float32
    @staticmethod
    def decode(latent):
        return SimpleNamespace(sample=torch.zeros(latent.shape[0], 3, decoded_size, decoded_size))
```

`decoded_size` 是测试自己传进去的 32；然后
`test_nextstep_decode_enforces_requested_geometry` 断言 `decoded.shape == (2, 3, 32, 32)`，并断言
`image_size=64` 会 raise `"requested 64x64, decoded 32x32"`。**32 从头到尾是测试自己声明的**。

生产侧 `vrl/models/families/nextstep_1/model.py:288-303` 真正做的是：从 token 数开方得到 side、
`unpatchify` 成 latent、`vae.decode(...)`、再拿 `pixels.shape[-2:]` 与 `image_size` 比。**中间那步
「真 VAE 的空间上采样倍率」是唯一能让 32 有意义的东西。**

**替换**：`build_tiny_autoencoder_kl(downsamples=3, latent_channels=16)`（f8，NextStep-1-f8ch16
的几何），token 网格取 `(2, 16, D)` → side=4 → latent `(2,16,4,4)` → 真 decode → `(2,3,32,32)`。
实测 9,387 参数 / 3.4 ms 构建 / 6.5 ms decode，输出 shape 确认 `(2, 3, 32, 32)`。

**新 `tests/models/families/nextstep_1/fixtures.py`**（沿用 emu3 / glm_image / llamagen 已有的
per-family fixtures.py 惯例），托管：
- `build_stub_nextstep_pipeline()` —— `gen_pipeline.NextStepPipeline` 的 `sys.modules` 注入替身，
  今天在 `test_model_loading.py:22` 和 `:62` **重复写了两遍**；
- `build_decode_only_nextstep_model(*, vae)` —— 今天的 `_decode_only_model`，改为接受一个真 VAE。

`UpstreamModel.unpatchify` **保持替身**（理由见 §8）。

**verify**：`.venv/bin/python -m pytest tests/models/families/nextstep_1 -q -p no:randomly`

### 6.4 `tests/scripts/eval`：4 份重复的循环 scheduler → 1 个真 builder

**今天最刺眼的一条**（`test_sana_checkpoint_compare.py:34-40` + `:575-578`）：

```python
class DPMSolverMultistepScheduler:
    def __init__(self, **overrides) -> None:
        config = dict(checkpoint_compare.SCHEDULER_PROTOCOL)   # <- 就是被校验的那张表
        config.pop("class_name")
        config.update(overrides)
        self.config = config
...
def test_scheduler_protocol_accepts_official_identity() -> None:
    assert sana_inference.validate_scheduler(DPMSolverMultistepScheduler()) == checkpoint_compare.SCHEDULER_PROTOCOL
```

`validate_scheduler`（`vrl/scripts/eval/sana_inference.py:60-80`）读 `type(scheduler).__name__` +
6 个 config 键，然后和 `SCHEDULER_PROTOCOL` 比。替身的类名是本地写的 `DPMSolverMultistepScheduler`，
config 是 `SCHEDULER_PROTOCOL` 本身。**这个测试在构造上不可能失败**，而它要挡的事故正是
「diffusers 改了 `use_flow_sigmas` / `flow_shift` 的名字或默认值」。

**替换**：一个共享 builder（放 `tests/scripts/eval/fixtures.py`，两个文件 import）：

```python
def build_official_sana_scheduler(**overrides: Any) -> Any:
    """The real ``DPMSolverMultistepScheduler`` at SANA's official protocol.

    Config-init, no download. Passing an override is how the drift tests produce a
    scheduler that must be REJECTED — and because the object is genuine, an upstream
    rename of any protocol key turns the accept-case red instead of staying green.
    """
    from diffusers import DPMSolverMultistepScheduler
    kwargs = {"algorithm_type": "dpmsolver++", "solver_order": 2, "solver_type": "midpoint",
              "use_flow_sigmas": True, "flow_shift": 3.0, "prediction_type": "flow_prediction"}
    kwargs.update(overrides)
    return DPMSolverMultistepScheduler(**kwargs)
```

**实测**：构建 1.0 ms；`validate_scheduler(real)` 返回值与 `SCHEDULER_PROTOCOL` 完全相等；
`test_scheduler_protocol_rejects_wrong_config` 的**全部 6 个参数化 drift 用例**在真类上逐个
`rejected (good)`：

```
algorithm_type='sde-dpmsolver++'  solver_order=3  solver_type='heun'
use_flow_sigmas=False             flow_shift=1.0  prediction_type='epsilon'
```

即测试体零改动，只换构造函数。落点：`test_sana_aesthetic_checkpoint_eval.py:754/807/844` 三处 +
`test_sana_checkpoint_compare.py:36` 一处，共 4 份重复定义收敛成 1 个。

**verify**：`.venv/bin/python -m pytest tests/scripts/eval -q -p no:randomly`

---

## 7. 成本账（全部实测，`-p no:randomly`）

### 单位成本

| 对象 | 冷（进程内首次） | 暖（套件里的真实边际） |
|---|---|---|
| `AutoencoderKL`（f2 / f8） | 2.3 ms / 3.4 ms | 同左 |
| `AutoencoderKLWan` | 2.7 ms | 同左 |
| 任一 diffusers scheduler `__init__` | 0.1–0.7 ms | 同左 |
| `DiffusionPipeline` shell（`register_modules`） | 0.3 ms | 同左 |
| `Cosmos3OmniTransformer` config-init | 1.5 ms | 同左 |
| `Cosmos3OmniPipeline`（含 in-memory tokenizer） | 2.9 ms | 同左 |
| `from diffusers import Cosmos3Omni*` | 0.856 s（进程里 diffusers 全冷） | **3.1 ms** |
| `standard_mochi_scheduler` 首调 | **0.608 s** | **0.1 ms** |

> **对 brief 的两处修正。**
> (1) brief 把 +0.6 s 归给「cosmos3 子模块一次性 import」。实测不成立：在 `WanTransformer3DModel` /
> `CosmosTransformer3DModel` / `AutoencoderKL` 已 import 的进程里，`Cosmos3OmniTransformer` 的边际
> import 是 **0.000 s**、`Cosmos3OmniPipeline` 是 **0.018 s**、两者合计 **3.1 ms**。
> (2) 那 0.6 s 真实存在，但属于 **`standard_mochi_scheduler`**——它里面
> `from diffusers.pipelines.mochi.pipeline_mochi import linear_quadratic_schedule`（`mochi/model.py:66`）
> 会拉起整个 mochi pipeline 模块。而默认车道**今天已经付过**这笔钱：
> `tests/models/families/mochi/test_backbone_parity.py:106-118` 和
> `tests/models/steps/denoise/test_scheduler_logprob_parity.py:116-118` 都直接调它。
> 所以本轨道让 mochi 的 `prepare_replay` 真的执行，边际是 **0.1 ms**，不是 0.6 s。

### 分项 wall-clock 增量

| 条目 | 增量 |
|---|---|
| `test_vae_decode_memory.py`（7 处真 VAE） | +20 ms |
| `test_frozen_offload.py`（真 pipeline shell） | +10 ms |
| `test_minimal_replay_runtime_wiring.py`（16 处真 scheduler + mochi/pixart 标准化真的跑） | +8 ms |
| anima（删 1 测试 / 加 2 测试） | +5 ms |
| **cosmos3（新增 3 测试）** | **+60 ms** |
| nextstep（真 f8 VAE，2 测试） | +21 ms |
| `tests/scripts/eval`（真 scheduler ×10） | +10 ms |
| `test_wan_dpo_encoders.py`（真 Wan VAE） | +20 ms |
| **合计** | **≈ +0.15 s** |

对 189 s 全量基线是 **+0.08 %**，对 104 s fast lane 是 **+0.14 %**。全部留在默认车道，
**没有一条需要 opt-in lane**。

---

## 8. NON-GOALS（本区里刻意保留的替身，逐条给理由）

以下都在本轨道的射程内、都读过源码，**一律不动**。

### (b) 无法按需制造的环境/状态

| 保留项 | 位置 | 为什么不能转 |
|---|---|---|
| `_IdentityDecodeVAE` | `tests/models/steps/denoise/common/test_decode_layout_parity.py:33` | 它是 **identity 探针**，不是模型替身。真 VAE 会做空间上采样，测试要钉的 `image == latents / 2.0 + 0.5`（`:75`）这种逐元素等式当场失效——被测的是 wrapper 的**反归一化与 layout 变换算术**，透明探针是唯一能让这段算术可观测的仪器。上一轮审计（`:539`）对它的裁定是对的，本轨道明确背书。 |
| `test_scheduler_protocol_rejects_wrong_class` 里的假 `FlowMatchEulerDiscreteScheduler` | `tests/scripts/eval/test_sana_checkpoint_compare.py:567-572` | 它要造的状态是「**config 全对、只有类名错**」。真 `FlowMatchEulerDiscreteScheduler` 的 config 里根本没有 `algorithm_type` / `solver_order`，换成真类会让 class 与 config **同时**不匹配，从而无法定位 `validate_scheduler` 里 class-name 那一半是否还在起作用。这是隔离半个校验的正当仪器。 |
| `_ConstantAnimaTransformer` 的**载荷注入**思路 | 被删的 `anima/test_forward_step.py:38` | 常量输出确实能钉分支顺序，但 §6.1 证明同一主张可以用 `record_forward_calls` 在**真** transformer 上表达得更强（直接断言每次 forward 收到的是哪个 embeds 张量）。所以这里不是「保留替身」，是「主张搬家后替身失去存在理由」——**先加后删**，不允许直接删。 |

### (c) 进程/包边界替身（真对应物不在本进程）

| 保留项 | 位置 | 为什么不能转 |
|---|---|---|
| `gen_pipeline.NextStepPipeline` 的 `sys.modules` 注入 | `tests/models/families/nextstep_1/test_model_loading.py:22,62` | 实测 `import nextstep_model` / `import gen_pipeline` 都是 `ModuleNotFoundError`，且未在 `pyproject.toml` 声明。断言的是**我方**行为（传了哪两个 repo id、哪两个 revision、`Pipeline` 收到什么 kwargs），不是替身的脚本化返回值。本轨道只把它**去重**成一个 builder，不改语义。 |
| `UpstreamModel.unpatchify` | `tests/models/families/nextstep_1/test_model_loading.py:88-92` | 同上，`nextstep_model` 不可导入。它只提供 latent 网格形状；VAE 那半（真正决定 decode 几何的那半）已在 §6.3 转成真对象。 |
| `test_official_scheduler_uses_build_revision_projection` 的 `from_pretrained` recorder | `tests/scripts/eval/test_sana_aesthetic_checkpoint_eval.py:806-840` | 断言的是 `calls == [("test/sana", {"subfolder": "scheduler", "revision": …})]`，即 **revision 投影**这件我方逻辑。真 `from_pretrained` 属于 hub-cache 轨道（RW-09）的射程，不在本轨。<br>**但本轨顺手收一个 clarity 缺陷**：它用 `monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(...))` 把**整个 diffusers 模块**换掉（`:822-826`）。生产侧 `load_official_scheduler` 是函数内 `from diffusers import DPMSolverMultistepScheduler`（`sana_inference.py:44`），所以 `monkeypatch.setattr(diffusers, "DPMSolverMultistepScheduler", …)` 足够，且不会让同一测试体内任何其它 diffusers 引用意外落空。 |

### 明确不碰的相邻代码

- **`vrl/models/families/wan_2_1/model.py:1379`** 的 `_config_value(getattr(scheduler, "config", None), …)`
  **不是**同一类缺陷，不删。它 `return None` 之后由调用方抛出明确错误
  （`"Wan dual-stage routing requires scheduler.config.num_train_timesteps"`，`:1365`），
  是对一个真正可选字段的防御；而 mochi/pixart 那两处是**静默跳过整段标准化**。行为不同，裁定不同。
- **`vrl/scripts/eval/sana_inference.py:63`** 的 `config = getattr(scheduler, "config", None); if config is None: raise TypeError(...)`
  同理——它 fail loud，不是为替身弯腰。
- 不重命名、不移动任何文件（新增的 `fixtures.py` 除外，那是既有 per-family 惯例）。
- 不动 `tests/models/steps/denoise/registry.py` 的 T2-PIPE 车道（`hf-internal-testing/tiny-*`
  真下载，5.2 s，需网络）——它解决的是 `model_index.json → 组件图`的装配问题，与本轨道的
  config-init 车道正交。

### 与 in-flight 工作的冲突（**不要重做**）

`git status` 显示 `tests/models/families/flux/test_diffusion_nft_interface.py` 有 owner 的
**未提交改动**，内容正是 brief 里提到的「`to_out.0` / `add_*_proj` 只匹配一半 block 的 FLUX
transformer」：diff 里 `build_tiny_wan_transformer` 已换成 `build_tiny_flux_transformer`，
`_TINY_WAN_LORA_TARGETS` 已换成完整的 8 项 `_FLUX_LORA_TARGETS`，`_build()` 的
`SimpleNamespace` 已换成真 `ModelBuild`。**这一条已经做完，本 sprint 不列入范围。**
落地时先 `git stash list` / `git diff` 确认它已入库，避免与本轨道对 `fixtures.py` 的编辑撞车
（本轨道只**新增** builder，不改动既有的 12 个）。

`tests/generation/execution/test_execute_request_pipelined.py` 也有未提交改动，与本轨道无交集。

---

## 9. HONEST GAPS（本 sprint 明说「进程内测不到」的东西）

> **现状修正（2026-07-30）：** 轨道一已经注册 `real_cover` 并落地 AST 守卫。
> 以下机制说明保留的是审计时基线；剩余施工应按当前契约把对应缺口直接标注，
> `tracked_in` 指向新的剩余计划或本审计快照，以实际 ownership 为准。

> **机制说明**：`pyproject.toml:202` 开着 `--strict-markers`，而 `markers` 列表（`:203-211`）里
> **没有** `real_cover`。在它被 infra 轨注册之前，任何 `@pytest.mark.real_cover` 都会让整个文件
> **collection 硬失败**，不是 warning。因此本轨道**不使用该 marker**，honest gap 先以本节 + 测试
> docstring 记录；等 infra 轨落地注册与 AST meta-test 后，再把下表逐条贴上去。

| 缺口 | 进程内能到哪 | 真实对应物 |
|---|---|---|
| **cosmos3 数值正确性** | T2 已覆盖：packed_static 装配、11 个 transformer kwarg 名、3 元组解包、CFG combine 公式、decode 反归一化。**未覆盖**：33K 随机权重的输出没有任何数值意义。 | **没有 e2e case。** `tests/e2e/test_real_checkpoint_rl.py` 的 `CASES` 里没有 cosmos3 条目（有 janus_pro、nextstep_1 等）。**这是本轨道诚实登记的新缺口**，落点是给 `CASES` 加一个 `RealCheckpointCase(case_id="cosmos3", …)`，属于 e2e 轨的工作。 |
| **cosmos3 真 tokenizer 词表** | in-memory `PreTrainedTokenizerFast` 只保证 `convert_tokens_to_ids('<\|vision_start\|>')` 和 `eos_token_id` 可用；`Cosmos3Model.encode_prompt` 走的 `pipe.tokenize_prompt`（含 duration/resolution 模板拼接）**没有**被覆盖。 | Hub 上没有 `hf-internal-testing` 的 tiny cosmos3 repo（已查 API，`cosmos3` / `cosmos-3` / `Cosmos3Omni` 三个关键词在该 org 下均为空），所以 T2-PIPE 也够不到。真对应物同上，只能在 e2e case 里。 |
| **NextStep 的 `unpatchify` 与 pipeline 装配** | 真 VAE 让 decode 几何变真；`nextstep_model` / `gen_pipeline` 侧仍是 `sys.modules` 注入。 | `tests/e2e/test_real_checkpoint_rl.py` 已有 `case_id="nextstep_1"`（`:383`）。**真对应物已存在**，不是缺口，只是今天没有任何东西指向它。 |
| **mochi / pixart 的 replay 数值 parity** | 转换后 `standard_mochi_scheduler` / `pixart_ddim_scheduler` 真的执行，ladder 的**类身份与长度**被钉住。**未覆盖**：这两条 ladder 与 rollout 侧实际用的 ladder 是否逐元素相等。 | `tests/models/steps/denoise/test_scheduler_logprob_parity.py:116-125` 已经在真 scheduler 上做 log-prob parity。这条不是缺口，是**已有真实车道没被指认**。 |
| **VAE tiling 的真实内存效果** | 真 `AutoencoderKL` 上 `use_tiling` 确实翻转；**未覆盖**：tiling 是否真的降低了 decode 峰值显存。 | 需要 CUDA + `torch.cuda.max_memory_allocated`。属于 `@pytest.mark.gpu` 车道，本轨道不建。**明确登记为未覆盖。** |

---

## 10. 验证与落地顺序

**排序原则**：门禁步骤单独一批（可回滚），生产代码改动排在依赖它的测试改动之后。

| 批次 | 内容 | 门禁 |
|---|---|---|
| **A**（无依赖） | §3 VAE builder + `test_vae_decode_memory.py` + `test_wan_dpo_encoders.py`；§4 pipeline shell + `test_frozen_offload.py` | `pytest tests/models/steps/denoise tests/scripts/test_wan_dpo_encoders.py -q` |
| **B**（无依赖） | §5 `_TinyScheduler` 退役 → **然后**删两处生产 guard | `pytest tests/models/interfaces tests/models/families/mochi tests/models/families/pixart_sigma -q`，随后全量 |
| **C**（无依赖） | §6.1 anima：**先**在 parity 文件加 2 个测试并跑绿，**再**删 `test_forward_step.py` | `pytest tests/models/families/cosmos -q` |
| **D**（无依赖） | §6.3 nextstep fixtures + 真 f8 VAE；§6.4 `tests/scripts/eval` 共享 builder | `pytest tests/models/families/nextstep_1 tests/scripts/eval -q` |
| **E**（**依赖 §1 门禁**） | `uv lock --upgrade-package diffusers` → `uv sync` → §6.2 cosmos3 fixtures + 新测试文件 | `pytest tests -q`（全量零回归）+ `pytest tests/models/families/cosmos/cosmos3 -q --durations=5` |

每批结束后：

```bash
.venv/bin/ruff check --fix <touched .py>
.venv/bin/ruff format <touched .py>
.venv/bin/ruff check <touched .py> && .venv/bin/ruff format --check <touched .py>
```

**最终门禁**（两条都必须过）：

```bash
.venv/bin/python -m pytest tests -q -p no:randomly
.venv/bin/python -m pytest tests -m "not e2e and not slow_test" -q -p no:randomly --durations=20
```

第二条同时用于确认 §7 的 +0.15 s 预算成立：`--durations=20` 里不应出现任何本轨道新增/修改的测试。

---

## 11. 关联

- [[SPRINT_test_suite_tiny_real_and_fake_audit]]（done）——本轨道**翻案**其 `:538` 对 `_FakeVAE`
  的 keep 裁定，**背书**其对 `_IdentityDecodeVAE`（`:539`）和 NextStep 包边界（`:582`）的裁定。
- Track 1（infra）——`real_cover` marker 的注册 + AST meta-test。§9 的登记表在它落地后才能贴到代码里。
- Track 6（e2e）——`tests/e2e/test_real_checkpoint_rl.py` 的 `CASES` 新增 cosmos3 条目，是 §9 第一行
  缺口的唯一真实对应物。
