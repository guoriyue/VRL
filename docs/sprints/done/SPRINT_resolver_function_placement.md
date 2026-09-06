# SPRINT：`require_*` / `build_*` / `resolve_*` 自由函数的归属审计（done）

状态：**done（2026-09-05，第二轮扩审同日）**。基线：main @ `8c7ed939`，
扩审基线 `7b8cb15b`（上游「拆 guard / 删防御性测试」8 连提交之后）。
判据来自 AGENTS.md「Placement — where a shared helper belongs (the four rules)」。
起因：用户观察「仓库里有很多 `require_xx` / `build_xx` / `resolve_xx` 这种孤立函数，
能不能塞进一个干净的 dataclass」。

> 本文所有数字都是本机 grep 实测，附 path:line。凡是判「保留」的，都给了函数体级别的理由，
> 不是按调用点或命名猜的。

---

## 0. 一句话

`vrl/` 里 1362 个模块级函数中，这三个前缀占 123 个。**其中只有 5 个是真的缺了家**——
判据是「函数名和返回类型是同一个概念」（`resolve_X(...) -> X`），它们本质是构造器。
其余 118 个各有正当理由留作自由函数，把它们塞进 dataclass 会造 god object 或打断
registry 的点号字符串分派。5 个已改完，全套 4429 passed / 13 skipped 零失败。

## 1. 计数

```
require_    20      validate_   28      normalize_  11
build_      71      resolve_    32      parse_       3
                                        总模块级 def  1362
```

## 2. 判据

仓库**早就有正确写法**，只是没贯彻：

```
vrl/run.py:77                              def from_root(cls, root: RootConfig) -> OnlineRunConfig
vrl/config/builders.py:61                  def from_cfg(cls, cfg) -> RewardRuntimeConfig
vrl/scripts/eval/denoise_generation.py:65  def from_root(...) -> ImageSampling
vrl/rewards/models/countgd.py:107          def from_mapping(cls, value) -> CountGDConfig
vrl/rewards/service/server.py:108          def from_mapping(cls, value) -> RewardServiceConfig
vrl/generation/ray/config.py               def from_public_section(cls, section) -> RolloutWorkerConfig
```

**判据：`resolve_X(...) -> X` —— 函数名和返回类型指的是同一个概念时，这个自由函数就是
一个没写成 classmethod 的构造器。** 这条判据可机械检验（函数与类同模块、返回类型即该类、
函数体里唯一的构造调用就是该类），不依赖口味。

## 3. 桶 1：保留（118 个，附理由）

### 3.1 `build_parser()` × 20 —— argparse 惯例

每个 CLI 脚本一个，零共享、零重复逻辑。合并它们会把 20 个互不相干的 flag 集合塞进一处。

### 3.2 5 个 `build_*_replay_runtime_bundle(build: ModelBuild) -> RuntimeBundle` —— registry 协议边界

签名逐字相同，看着像重复五遍。**但 registry 用点号字符串分派它们**：

```
vrl/models/families/registry.py:659   "vrl.models.families.causvid.runtime:build_causvid_replay_runtime_bundle"
vrl/models/families/registry.py:678   "vrl.models.families.magi_1.runtime:build_magi_1_runtime_bundle"
vrl/models/families/registry.py:923   "vrl.models.families.cosmos.cosmos3.runtime:build_cosmos3_replay_runtime_bundle"
vrl/models/families/registry.py:938   "vrl.models.families.cosmos.anima.runtime:build_anima_replay_runtime_bundle"
vrl/models/families/registry.py:955   "vrl.models.families.echo.runtime:build_echo_replay_runtime_bundle"
```

这是 lazy-import 边界（registry 解析时不能 import torch/diffusers）。改成方法会打断分派。
**这也是死代码审计第 1 形态的提醒：纯符号 grep 找不到它们的调用者。**

### 3.3 跨 section 校验器 —— 合并进 `RootConfig` 就是 god object

| 函数 | 读到的 section 数 |
|---|---|
| `vrl/config/validation.py:291 compile_conflicts(root)` | **4**：`model.torch_compile` + `actor.gradient_checkpointing` + `distributed.training.strategy` + `distributed.resources.rollout.gpus_per_engine` |
| `vrl/config/validation.py:371 validate_compile_compatible(root)` | 同上（薄封装） |
| `vrl/config/validation.py:383 validate_guarded_rollout_drift(root, precision)` | `sampling` + 外部 `PrecisionPolicy` |
| `vrl/trainers/activation_checkpointing.py:99 validate_compile_checkpointing_compatible(root)` | 2：`actor` + `model` |
| `vrl/config/validation.py:37,55 validate_production_*(root)` | `production` + `reward` + `data` |

`compile_conflicts` 的 docstring 自己写了为什么必须集中：

> ONE home for the compile compatibility matrix. Each of these was discovered
> separately and used to be enforced somewhere different — grad-checkpointing in
> the trainer, FSDP2 in the strategy builder, sequence parallelism nowhere at all
> — so adding the fifth meant first finding the other four.

它们正是 AGENTS.md Rule 1 反例里的「public composition facade whose inputs span
several owners」。挂到 `RootConfig` 上既造 god object，又抹掉这段论证。

### 3.4 配置键路径参数 —— Rule 1 明确的反例

`vrl/utils/config.py:45 require_exact_int(value, *, path="actor.train_batch_size")`（8 个调用点）。
那个字符串命名的是**配置键**，不是调用者身份。同理
`vrl/trajectory/types.py:48 validate_string_tuple(name, values)`、
`vrl/trajectory/validation.py:292 validate_shape_prefix(name, value, expected)` —— 命名的是错误域。

### 3.5 单 section 读取但目标不是自己 —— 不动

`resolve_gradient_checkpointing_mode(root)`（4 个调用点）只读 `root.actor.gradient_checkpointing`，
按 Rule 1 该挂到 actor section 上；但它返回 `str` 而不是一个类型，且 `root.actor` 可为 `None`，
挂上去会把 None 处理推给每个调用点。**收益是负的，判保留。**
`resolve_train_target(root) -> str` 同理。

### 3.6 多输入构造器 —— 不是同一个概念

`vrl/trajectory/builders.py` 的 6 个 `build_*_trajectory(...)` 收十几个张量拼一个
`TrajectoryBatch`，不是「从一个所有者派生」。`build_rollout_collector` / `build_rollout_schedule`
/ `build_strategy` / `build_optimizer` 同类。

## 4. 桶 2 + 桶 3：已执行（5 个）

函数体**原封不动**搬进类里，只改签名首行加 `cls`、把自身构造调用换成 `cls(`。零行为变化。

| 之前 | 之后 | 调用点 |
|---|---|---|
| `resolve_training_resume_config(root) -> TrainingResumeConfig` | `TrainingResumeConfig.from_root(root)` | vrl 1 / tests 3 |
| `resolve_clean_target(source) -> CleanTargetRef` | `CleanTargetRef.resolve(source)` | vrl 2 / tests 0 |
| `resolve_training_context(root, *, device, env) -> DistributedTrainingContext` | `DistributedTrainingContext.from_root(...)` | vrl 1 / tests 7 |
| `resolve_precision_policy(section) -> PrecisionPolicy` | `PrecisionPolicy.from_section(section)` | vrl 18 / tests 64 |
| `resolve_distributed_resources(root, *, reward_inference) -> ResolvedDistributedResources` | `ResolvedDistributedResources.resolve(...)` | vrl 3 / tests 90 |

旧自由函数**整体删除，不留兼容 shim**，5 个 `__all__` 导出名一并去掉。实测残留引用为 0
（历史 sprint 文档里的 6 处提及是归档，按约定不改写）。

### 4.1 施工中踩到的两个坑（记下来，下次同类改动会再遇到）

**坑 1：单行签名的 lift 会静默失败。** 把 `def f(\n` 换成 `def f(\n    cls,\n` 的脚本对
`def resolve_precision_policy(section: ...) -> PrecisionPolicy:` 这种单行签名不匹配，
def 行没被改名，随后的全局改名把它撞成 `def PrecisionPolicy.from_section(...)`。
**教训：lift 之后必须 `ast.parse` + grep 确认新方法名真的出现，不能只看类名还在。**

**坑 2：monkeypatch 目标会静默失去意义。** 全仓 7 处测试 patch 的是**模块属性**：

```python
monkeypatch.setattr(online, "resolve_training_context", lambda _cfg, *, device: ctx)
monkeypatch.setattr(ray_resources, "resolve_distributed_resources", lambda _cfg, **kw: res)
monkeypatch.setattr(validation, "resolve_precision_policy", counted(...))
```

改名后若只做符号替换，它们会变成「用 lambda 替换掉整个类」——测试照样绿，但断言的东西
没了。全部改成 patch classmethod：

```python
monkeypatch.setattr(
    ray_resources.ResolvedDistributedResources,
    "resolve",
    classmethod(lambda _cls, _cfg, **_kwargs: res),
)
```

`tests/config/test_builders.py:83` 本来就有正确写法（`RewardRuntimeConfig.from_cfg.__func__`），
照抄即可。

## 4bis. 第二轮：把同一条判据推到其余前缀

第一轮只审了 `require_/build_/resolve_`（123 个）。第二轮把剩下的前缀全过了一遍：

```
validate_ 28   normalize_ 11   apply_ 12   load_ 32   select_ 9
format_ 7      parse_ 3        compute_ 3  to_ 5      make_/create_/derive_ 各 1
```

### 4bis.1 又找到 4 个构造器形状（已改）

| 之前 | 之后 | 调用点 |
|---|---|---|
| `validate_reward_config(cfg) -> RewardConfig` | `RewardConfig.from_cfg(cfg)` | 6 |
| `parse_hf_repo_revision(str) -> HuggingFaceRepoRevision` | `HuggingFaceRepoRevision.parse(ref)` | 6 |
| `parse_reward_inference_config(value, *, context) -> RewardInferenceConfig` | `RewardInferenceConfig.parse(...)` | 7 |
| `compute_logprob_mismatch_stats(a, b) -> LogprobMismatchStats` | `LogprobMismatchStats.compute(a, b)` | 14 |

前三个类与函数同模块。`validate_reward_config` 是唯一跨模块的：函数在
`vrl/config/validation.py`，`RewardConfig` 和它用的 `_extract_error_message` 都在
`vrl/config/schema.py`，而 schema **不** import validation，所以方法落在 schema 侧无环。

**判据在这一轮的意义**：它把「名字前缀」和「实际形状」解耦了。`validate_`、`parse_`、
`compute_` 三个不同的动词下面藏着同一个东西——返回类型即概念的构造器。反过来，
27 个 `validate_*` 里只有 1 个符合，说明这条判据不是「凡是自由函数都该变方法」的托词。

### 4bis.2 `Any` 参数：51 个里只有 5 处该改（已改）

按 Rule 2「`Any` 需要收据」逐个查。**28 个在 `vrl/trajectory/builders.py`**——全是张量参数，
该模块必须 import-time torch-free（config 解析会走到它），收据成立，不动。
`resolve_torch_dtype` / `require_plain_dtype` / `require_pipeline_offload_mode` /
`require_checkpoint_source_member` 收的是原始 YAML / manifest 值，收据成立。
`to_cpu_snapshot` / `to_builtin_deep` 是泛型递归遍历，收据成立。

真正无收据、且 `getattr` 正在守护一个**已声明字段**的有 5 处：

```python
# vrl/trainers/weight_sync.py —— 错误信息自己就写着 RuntimeBundle
-def require_trainable_modules(bundle: Any) -> Mapping[str, Any]:
-    modules = getattr(bundle, "trainable_modules", None)
-    if not isinstance(modules, Mapping) or not modules:
+def require_trainable_modules(bundle: RuntimeBundle) -> Mapping[str, Any]:
+    modules = bundle.trainable_modules
+    if not modules:

# vrl/models/loader.py —— 三个 getattr 守的都是 ModelBuild 的声明字段
-    quantization = getattr(getattr(build, "precision", None), "quantization", None)
-    if not nvfp4_available(getattr(build, "device", None)):
+    quantization = build.precision.quantization
+    if not nvfp4_available(build.device):
```

另外两处用 `TYPE_CHECKING` 标注，**刻意不新增运行时 import 边**：
`validate_rollout_schedule_topology(config: RolloutOrchestrationConfig,
resources: ResolvedDistributedResources)`，以及
`build_token_family_bundle(entry: ModelFamilyEntry)`——registry 是在方法体里惰性 import
这个模块的，模块级 import 回去会把环闭合。

标注之后**没有**出现新的所有权信号：这 5 个函数的必需参数标注后仍是多个，或所属类型在
另一个包，所以它们留在原地。Rule 2 的收益在这里是「删掉 6 个 `getattr` 防御」，不是搬家。

### 4bis.3 其余前缀：全部保留

- `normalize_wan_model_build` / `normalize_magi_1_model_build`：看着是同型双胞胎，
  但和 §3.2 一样由 registry 点号字符串分派
  （`registry.py:660,823,843`），改成方法会打断分派。
- `parse_config(cfg) -> RootConfig`：形状上符合判据，但它是全仓唯一的未知键关口、被多份
  sprint 文档当作契约引用，作为 public facade 保留。**这是判据的例外，明确记下来。**
- `load_* (32)` / `save_* (3)` 是 IO，`format_* (7)` 是渲染，`to_* (5)` 是纯转换，
  `select_* (9)` / `apply_* (12)` 是无主语的纯函数——判据都不命中。
- `normalize_wan_boundary_ratio(value, *, field_name)` 的字符串参数命名配置键，Rule 1 反例。

## 4ter. 命名规则（第三轮补正）

第一、二轮我给 9 个方法起了 5 种名字（`from_root` / `from_section` / `from_cfg` /
`resolve` / `parse` / `compute`），看着像随手取的。**实测仓库早有惯例**——53 个既有
classmethod 构造器，后缀命名的是**输入是什么**：

```
from_build 15   from_root 9   from_mapping 9   from_cfg 2
from_section / from_resources / from_request / from_trials / from_metrics / from_sampling ... 各 1
```

**规则：`from_<输入>`。** 后缀不同不是随意，是输入类型不同：`from_root(root: RootConfig)`、
`from_cfg(cfg: DictConfig)`、`from_section(section: PrecisionConfig)` 是三种不同的入参。

按这条把三个偏离的改回来了：

| 第二轮 | 改正为 | 理由 |
|---|---|---|
| `ResolvedDistributedResources.resolve` | `.from_root` | 入参是 `RootConfig`，和既有 9 个 `from_root` 同族 |
| `CleanTargetRef.resolve` | `.from_source` | 入参名就是 `source` |
| `RewardInferenceConfig.parse` | `.from_mapping` | 入参是 Mapping，仓库已有 9 个 `from_mapping` |

**唯一保留的例外**：`LogprobMismatchStats.compute(fresh, old)`。它不是把一种表示转成另一种，
而是从两个张量**测量**出统计量；`vrl/algorithms/advantages.py:193` 早有 `compute(` 先例。
`HuggingFaceRepoRevision.parse(str)` 同理保留——入参是待解析的非结构化字符串，不是一个可
"from" 的对象。

## 4quater. 负面结果：bucket 5 的两个函数**不能**变成方法

第二轮的提交信息里我写「标注后没有露出新的所有权信号，因为它们仍有多个必需参数」——
**这句话是错的**。标注之后有两个恰好只剩一个必需参数：

```python
require_trainable_modules(bundle: RuntimeBundle) -> Mapping[str, Any]   # 14 个调用点
validate_rollout_quantization_support(build: ModelBuild) -> None        #  2 个调用点
```

按 Rule 1 这就是所有权信号，所以第三轮真的把它们搬成了
`RuntimeBundle.require_trainable_modules()` / `ModelBuild.validate_rollout_quantization_support()`。
**结果 146 个测试红**：

```
AttributeError: '_Bundle' object has no attribute 'require_trainable_modules'
AttributeError: 'types.SimpleNamespace' object has no attribute 'validate_rollout_quantization_support'
```

**根因**：这两个参数从来就不是名义上的 `RuntimeBundle` / `ModelBuild`，而是**结构类型**。
15 个测试文件传的是只带 `trainable_modules` 一个字段的轻量替身；`RuntimeBundle` 是个字段
很多的 dataclass，真造一个代价很高，替身的存在是有理由的。

**这里有一条一般性教训，值得单独记住**：

> **类型标注是提示，方法是运行时契约。** 把 `x: Any` 改成 `x: SomeType` 不改变任何运行时
> 行为（鸭子类型照跑）；把 `f(x)` 改成 `x.f()` 把结构契约变成了名义契约。前者永远安全，
> 后者只在**所有**调用者都真的持有该类型时才安全。Rule 1 的「单一类型参数 = 所有权信号」
> 必须再过一道：这个类型是名义的还是结构的？

因此两处都已回退为自由函数，**标注保留**（它是准确的意图文档，且零运行时代价），并在
docstring 里写明为什么不是方法。另外 `require_trainable_modules` 的消费者全在
`vrl/trainers/`（checkpointing / strategy / weight_sync），把它挂到
`vrl/models/interfaces/` 的 `RuntimeBundle` 上还会把 trainer 的不变量推进 model 接口层——
即使测试问题不存在，这也是反对搬家的第二个独立理由。

## 4quinquies. 第四轮：`require_` 和 `validate_` 是同义词

### 4q.1 先证伪假设

假设「`require_` 收窄并返回值，`validate_` 是纯守卫」。**实测推翻**：

```
require_*:   返回 None 9,  返回值 11
validate_*:  返回 None 14, 返回值 14
```

48 个函数，两个前缀各占一半，**没有任何区分**。后果不是审美：`validate_timeout(x)` 返回
必须使用的 `float`，`require_string_tuple(n, v)` 什么都不返回——从调用点看不出返回值要不要接。

### 4q.2 定规则并统一（23 处改名）

> `require_X(...) -> X` —— 收窄/归一化并**返回**，返回值必须用
> `validate_X(...) -> None` —— 纯守卫，抛或过

9 个 `require_*` 返回 None → 改为 `validate_*`；14 个 `validate_*` 返回值 → 改为 `require_*`。
共 205 处引用、84 个文件，零行为变化，改名后无撞名。历史 sprint 文档按约定不改写。

### 4q.3 「双职责」核查：6 个丢返回值的调用点，全部正当

改名前有 6 个调用点丢掉了返回值，怀疑是「一个函数做两件事」。逐个读完调用点，**结论是它们
都对**——这些函数校验并归一化，有的调用者要归一化结果，有的只要那道检查：

| 函数 | 用返回值 | 只当守卫 | 判定 |
|---|---|---|---|
| `require_weights_for` | 2（`load_weights_into` 要剥掉前缀的 state） | 1（`validate_trainable_state` 刻意不改权重） | 正当 |
| `require_source_backed_video_world_manifest_pair` | 5（data CLI 要 report） | 3（config 校验只要检查） | 正当 |
| `require_remote_checkpoint_source_pin` | 1（`checkpoint_identity` 要 pin） | 1（wan config 只要检查） | 正当 |
| `require_artifact_manifest_pair` | 4 | 0 | 正当 |
| `require_actor_gpu_ids` | 仅 1 个测试 | 5 | 见下 |
| `require_scheduler` | 仅 1 个测试 | 6 | 见下 |

最后两个的返回值**没有生产消费者**，只有一个测试读它：

```python
result = require_actor_gpu_ids(..., cross_node=True, ...)
assert result == (0, 0)          # 两个 worker 各自的本地 GPU 0 都被接受

assert (sana_inference.require_scheduler(DPMSolverMultistepScheduler())
        == checkpoint_compare.SCHEDULER_PROTOCOL)
```

按「只有测试读 = 死」的规则该删。**这里不删，理由要写清楚**：这两个函数的失败路径靠抛异常观测，
**接受路径没有别的可观测量**；删掉返回值会把这两条测试削弱成「没抛」。返回值在这里就是
accept-path 的观测口。改名后调用点变成 `require_*(...)` 却不接返回值，这个「丢弃」是显式可见的，
比原来叫 `validate_*` 时更诚实。

## 5. 未决项（唯一一条）

`Any` + `owner=`/`what=` 字符串的 4 个收窄器，按 Rule 1 + Rule 2 都可疑，但它们做的是
「把 `Any` 收窄成 Protocol」，需要读完 `vrl/models/interfaces/replay.py` 才能判：

```
vrl/models/interfaces/replay.py:280   require_replay_model(value: Any, *, owner="model") -> ReplayModel
vrl/models/interfaces/replay.py:286   require_runtime_model(value: Any, *, owner="model") -> RuntimeModel
vrl/models/interfaces/replay.py:169   validate_zero_replay_timestep(timestep_idx: int, *, owner: str)
vrl/models/dtypes.py:104              require_plain_dtype(value: Any, *, what: str, detail="")
```

`require_trainable_modules(bundle: Any)`（14 个调用点）和
`build_trainable_state_sync_getter(bundle: Any)` 的 `bundle` 其实是 `RuntimeBundle`，
属于 Rule 2 的「`Any` 需要收据」——标注类型后可能露出所有者。**本 sprint 不承诺。**

## 6. 非目标

- 不为整齐把 `build_parser()` 合并或搬家。
- 不动 registry 的点号字符串契约。
- 不把跨 section 校验器挂到 `RootConfig`。
- 不给被删的 5 个自由函数留兼容别名——一次改干净，避免两个名字并存。
- 不改写历史 sprint 文档里对旧函数名的提及（那是归档，记录当时为真的事）。

## 7. 验收

第一轮（基线 `8c7ed939`）：

```bash
.venv/bin/python -m pytest tests -q -p no:randomly --ignore=tests/e2e
# 4429 passed, 13 skipped
```

第二轮（基线 `7b8cb15b`，上游删防御性测试之后基数变小）：

```bash
.venv/bin/python -m pytest tests -q -p no:randomly --ignore=tests/e2e
# 3445 passed, 13 skipped
.venv/bin/ruff check vrl tests && .venv/bin/ruff format --check vrl tests
# All checks passed!
```

## 8. 总账

四轮合计：**9 个自由函数变成 classmethod（命名统一为 `from_<输入>`），23 处 `require_`/`validate_`
按「是否返回值」统一，6 个防御性 `getattr` 删除，4 个 `Any` 参数标注，2 次搬家尝试被实测否决
并记录了原因**。
审过的函数总数 123 + 约 120 = 约 240 个；判定「该搬家」的比例约 4%。
这个比例本身是结论：仓库里 `require_/build_/resolve_/validate_` 密集**不是**设计问题，
绝大多数有正当理由；值得改的是那一小撮「名字和返回类型指同一个概念」的构造器。
