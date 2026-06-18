# SPRINT: Helper pass-through & docstring hygiene (follow-on)

状态：**done（2026-06-18 归档至 done/）**。Phase 0–3 全部落地并验证，2026-06-17 收口于 VRL 分支
`helper-passthrough-hygiene`，每个 Phase 独立 commit：Phase 1 = f6813f9，
Phase 0 补完 = d7ec895，Phase 3 = 5372ade；Phase 2 判定全 KEEP 无代码改动。
全套 `pytest -q tests/scripts/` → 58 passed）。
本 sprint 是 `done/SPRINT_small_function_consolidation.md`（2026-06-10 implemented，
Phase A/B/D 已落地，Phase C/A5 未做）的**后续增量**，针对前一轮**明确排除在动作之外**
的两类赘肉——前一轮口径是"只有 inline / merge / delete 三种动作，不做 rename / 文档化 /
新抽象"（见该文 §5.5），所以下面 §2 docstring 去重、§3 recipe 收类都是新领域，不与前轮重叠。

## 0. Core Decision

全仓 AST 重扫（280 文件、830 个模块级函数）后的结论：前一轮已经把"死 API 面 / 纯转发壳 /
薄文件合并"清过一遍；剩下让人觉得"helper 散得到处都是"的，是另外**两类前轮未覆盖的赘肉**，
外加**一处结构性收口**：

```text
(a) DI-by-arg —— 把模块级 import 当参数传进函数再原样调用（含 torch=torch）。纯噪音，
    制造"什么都在被传来传去"的错觉。这是头号目标。
(b) docstring 复述自己下面抛的 error/log —— 同一段话写两遍，改一处漏一处就漂移。
(c) run_online_recipe：~300 行过程 + 10 个共享同一批状态的浮动 single-use helper，
    是教科书级的"该收成一个小类"。结构收益最大，但最有观点，单独一轮做。
```

**不做**全量 inline 那 285 个单调用方 helper（前一轮已裁定"单文件私有 helper 是正常组织
方式"，重复那个结论只会制造 churn）。只做下面有逐条证据的子集。

审计方法（可复现）：

```text
AST 扫 vrl/ 全部 280 文件 →
  · 830 个模块级函数
  · 285 个单调用方私有 _helper（→ 按 §5 Non-Goal，不全量动）
  · 24 个 docstring 行数 > 函数体语句数 且 体 <=8 stmt 的小函数（§2 候选池）
  · 8 处"参数名 == 本模块 import 名"（DI-by-arg 候选）
对每个 DI-by-arg 候选再人工核：参数是否真的就是那个 import 的原样转发。
已剔除的假阳性见 §1 末尾。
```

## 1. Phase 0 — online.py 样板（_save_checkpoint + docstring 于 2026-06-16；_prepare_metrics_csv 于 2026-06-17 补完，commit d7ec895）

用户选中的 `vrl/scripts/common/online.py` 作为三类赘肉的样板，已清并验证：

```text
DI-by-arg 删除（3 处，参数甚至与 import 同名造成遮蔽）：
  _save_checkpoint(..., save_training_checkpoint=, capture_rng_state=)  → 删参数，直接调顶层 import（2 处调用点同步改）【2026-06-16 落地】
  _prepare_metrics_csv(..., prepare_metrics_csv=) + 兄弟 _prepare_eval_metrics_csv + 调用点  → 同上【2026-06-17 补完，见下方修正】
docstring 去重（1 处）：
  _require_supported_online_strategy 的 9 行 docstring 几乎逐句复述下面 NotImplementedError
  的 message → 收成 6 行只讲 WHY，细节留在可 grep 的 error message（单一事实来源）【2026-06-16 落地】
```

验证：`pytest -q tests/scripts/test_online_lifecycle.py` → **11 passed**。
测试用 `monkeypatch.setattr(online, "_save_checkpoint", ...)` 整体替换函数，不绑定参数签名，
故删参数零风险（Phase 3 后该 patch 改为指向 `OnlineRecipeRun.save_checkpoint`）。

> **2026-06-17 修正**：复核发现初版 Phase 0 实际只删了 `_save_checkpoint` 的 DI 参数与 docstring；
> `_prepare_metrics_csv` / `_prepare_eval_metrics_csv` 的 `prepare_metrics_csv=`（默认值即顶层
> import 自身，构成 self-shadow）当时**并未删除**，调用点 `prepare_metrics_csv=prepare_metrics_csv`
> 也还在。绿测掩盖了这点：测试把 `_prepare_metrics_csv` 整体 monkeypatch 成 no-op，不绑定签名。
> commit d7ec895 才真正删掉这两处参数与调用点，函数体直接调顶层 import，claim 至此为真。

## 2. Phase 1 — 剩余 DI-by-arg 分类 + 删除（最高确定性，但不全机械）【已落地 2026-06-17，commit f6813f9】

> **结果**：`cosmos_predict25_kling_eval.py` 顶层已 `import torch`，故按规则 A 删掉
> `_resolve_device(..., torch_module)` 与 `_resolve_dtype(..., torch)` 的透传参数，函数体直接用
> 顶层 `torch`，返回/参数类型从 `Any` 恢复为 `torch.device` / `torch.dtype`，调用点 (main) 同步改。
> `anima/generate.py` 三个 helper 的 `torch` 参数**按规则 B 保留**：该文件顶层无 `import torch`，
> torch 在 `main()` 内 `dry_run` 后才 lazy import 再传入，且 `tests/scripts/test_anima_generate.py:85`
> 直接注入 fake `_Torch` 断言 CUDA fail-fast——是真实的 lazy-import / test-fake 边界，非 DI 噪音。
> `placement.py:72 actor_scheduling_strategy(placement_group)` 仍为合法领域参数，未动。
> 验证：`pytest -q tests/scripts/test_cosmos_predict25_kling_eval.py` → 8 passed；grep 无未解释 `torch_module` 残留。

online.py 清完后，全仓 AST 复扫剩若干 torch-module 参数命中。这里必须先分类，不能把
所有 `torch` 参数都当成噪音删掉：

```text
可直接删除（顶层已 import torch，参数只是在同模块内把 import 原样传一遍）：
  vrl/scripts/eval/cosmos_predict25_kling_eval.py:244   _resolve_dtype(..., torch: Any)

需人工打开再判定（扫描器原先按"参数名 == import 名"会漏掉这种重命名参数）：
  vrl/scripts/eval/cosmos_predict25_kling_eval.py:235   _resolve_device(..., torch_module: Any)
    —— 文件顶层已 import torch；如果测试没有依赖 fake torch_module，就应和 _resolve_dtype
       一起改成直接用模块级 torch。

明确保留或单独决策（不是普通 DI 噪音）：
  vrl/scripts/diffusion/cosmos/anima/generate.py:272    _resolve_device(..., torch: Any)
  vrl/scripts/diffusion/cosmos/anima/generate.py:283    _resolve_dtype(..., torch: Any)
  vrl/scripts/diffusion/cosmos/anima/generate.py:301    _generate_images(..., torch: Any)
    —— anima/generate.py 顶部没有 import torch；main() 在 dry-run 之后 lazy import torch，
       再把 torch 传入这些 helper。这里承担 lazy-import boundary；测试还直接传 fake
       _Torch 断言 CUDA unavailable fail-fast（tests/scripts/test_anima_generate.py）。
       不能按"机械删 torch=torch"处理。要改也应作为单独取舍：改成顶层 import torch +
       monkeypatch 测试，或保留现状。
```

做法：

```text
A. 顶层 import 已存在 + 参数只转发 import 对象 -> 删参数与调用点，函数内直接用模块级 torch。
B. lazy import / test fake / protocol boundary -> 保留，并在文档中写清为什么不是噪音。
C. 参数重命名（torch_module）-> 不靠 AST 名字规则下结论，打开调用点和测试后再判定。
```

删除时顺带把因为 `Any` 参数丢掉的返回/参数类型恢复具体类型。保留时不要为了清零 grep
结果破坏 lazy import / fake boundary。

```text
明确剔除（扫描命中但非 DI-by-arg，保留）：
  vrl/ray/placement.py:72  actor_scheduling_strategy(placement_group: Any)
    —— placement_group 是调用方传入的真实 Ray PlacementGroup 对象，不是被转发的 import；
       函数体局部 import 的是 PlacementGroupSchedulingStrategy。是合法领域参数，不动。
```

验收：`pytest -q tests/scripts/ tests/ -k "anima or cosmos_predict25"` 绿；
grep 确认没有**未解释**的 `torch=torch` / `torch_module` 残留。anima lazy-import 边界若保留，
应在 sprint closeout 里显式列为 intentional。

## 3. Phase 2 — docstring 复述 error 去重（逐个判定，不批量）【已完成 2026-06-17：5 个候选全部判定 KEEP，无代码改动】

> **结果**：逐个打开 5 个候选，全部按"默认保留"裁定为 KEEP——每个 docstring 讲的都是领域 WHY /
> contract，而非复述自己的 `raise`/`log`，故无一处改动：
> - `vae_decode_memory.py apply_generation_memory_policy`：讲 target-keyed contract + 为什么 typo 必须 fail loud。
> - `distributed.py resolve_training_context`：讲 single_process vs fsdp 语义 + 与 strategy.py/fsdp.py 的边界。
> - `builders.py _validate_yaml_home`：讲 optional 字段 typo 的静默回退风险 + vocabulary 由 RootConfig 派生防漂移。
> - `prompts.py load_prompt_manifest`：文档化 .jsonl / .txt 两种受支持格式（含 flow_grpo 约定）。
> - `precision_guard.py resolve_guard_mode`：讲 auto/warn/fail 策略取舍；error 只覆盖非法 mode。
> 各文件 `git diff` 为空。结论与文档原"当前倾向保留"一致，Phase 2 实质为 no-op done。

§2 候选池 24 个 doc-heavy 小函数里，**默认保留**，只动 docstring 在复述本函数自己
`raise`/`log` 文案的那一小撮。候选（既 doc-heavy 又含 `raise`，需逐个开看）：

```text
vrl/models/diffusion/common/vae_decode_memory.py:79  apply_generation_memory_policy
vrl/trainers/distributed.py:68                       resolve_training_context
vrl/config/builders.py:109                           _validate_yaml_home
vrl/trainers/data/prompts.py:29                      load_prompt_manifest
vrl/trainers/online/precision_guard.py:34            resolve_guard_mode
```

判定规则：docstring 与下面 error message **讲同一件事** → docstring 收成一句 WHY，
细节留在 error（用户可见 + 可 grep = 单一事实来源）。docstring 讲的是 message 里**没有**
的领域 WHY → 保留。

```text
当前倾向保留（打开后更像 WHY / contract，不像 error 复述）：
  vrl/models/diffusion/common/vae_decode_memory.py:79  apply_generation_memory_policy
    —— docstring 解释 model.memory target-keyed contract，以及 typo 为什么必须 fail loud。
  vrl/config/builders.py:109  _validate_yaml_home
    —— docstring 解释合法 top-level section 从 RootConfig.model_fields 派生，避免手写
       ALL_CAPS allow-list 漂移；这是 architecture hygiene，不是重复。
  vrl/trainers/online/precision_guard.py:34  resolve_guard_mode
    —— docstring 解释 auto/warn/fail 的策略取舍；error 只处理非法 mode。

明确保留（doc-heavy 但 docstring 是合法且非重复的 WHY，严禁动）：
  vrl/trainers/weight_sync.py:150  unwrap_compile_and_ddp
    —— 1 行循环体 + 11 行 docstring，但跨 3 处复用，docstring 讲"为什么剥 compile/DDP
       却不剥 PEFT"这种非显然领域判断。是该保留的 thin-function + WHY 范本。
  vrl/scripts/common/online.py:140  _warn_global_std_streaming_divergence
    —— docstring 讲 GRPO global_std 在 streaming 下的语义陷阱，warning message 讲用户
       remedy，两者不重复。保留。
```

验收：改动文件 `pytest -q` 绿（纯文本改动，行为零变化）；人工确认每处 error message 仍自洽。

## 4. Phase 3 — run_online_recipe 收成 OnlineRecipeRun（结构性，单独一轮）【已落地 2026-06-17，commit 5372ade】

> **结果**：新增 `OnlineRecipeRun`（`@dataclass(slots=True)`，定义在 online.py，紧邻 run_online_recipe，
> 不进 types.py——它不是 family-hook payload）。持有 `stack` + 本 run 的执行状态
> (`csv_path` / `eval_csv_path` / `rng` / `resume`)，把 5 个 IO 副作用收成方法：
> `prepare_metrics_csv` / `write_metric_row` / `prepare_eval_metrics_csv` / `write_eval_metric_row` /
> `save_checkpoint`。`run_online_recipe` 里 6 处调用点从"塞 2–6 个参数"降为"调方法"；stack 构造上移几行
> 以便 controller 先包住它。`OnlineRecipeStack` 仍是唯一的 wired-runtime owner，仍是交给
> before_step/after_step 的 payload，controller 只读 `self.stack.*`，不复制其字段。
>
> **相对 §4 原计划的两处取舍**（实现时决定，已在 commit message 记录）：
> 1. 连 eval-CSV 两个 writer 一并收成方法（不止文档点名的 3 个 training helper）——否则 training CSV 走
>    方法、eval CSV 仍是自由函数，正是本 sprint 要消的那种不对称。
> 2. controller 存 `resume: bool`（两个 prepare 方法实际只需这个布尔），不存整个 `resume_checkpoint`，
>    避免一个 derived-struct dead field。
>
> 测试：`tests/scripts/` 全套 **58 passed**。`test_online_lifecycle` 的 monkeypatch 从 patch 模块级
> `_save_checkpoint`/`_prepare_metrics_csv`/`_write_metric_row` 改为 patch `OnlineRecipeRun` 的同名方法；
> `test_online_precision_bridge` 那个 CSV 列格式测试改用一个最小 SimpleNamespace stack 驱动 controller。

`vrl/scripts/common/online.py` 的 `run_online_recipe` 是 ~300 行过程，周围浮着一批
single-use 私有 helper，反复在传同一批状态。注意：当前代码已经有 `OnlineRecipeStack`，
而且 stack 已经是 family hook payload / wired runtime objects 的 owner；Phase 3 不能再造
第二个重复 owner。

```text
已有 owner：
  OnlineRecipeStack = family hook payload + wired runtime objects
    cfg / definition / bundle / model / reward_fn / collector / algorithm / evaluator /
    trainer / strategy / trainer_config / collector_config / family / output_dir /
    component_names

仍在 run_online_recipe 里飘着的 execution 状态：
  csv_path / rng / resume_checkpoint / gradient_accumulation_steps / run_error

受害 helper（每个都只在 run_online_recipe 里调一次，靠参数把上面状态喂进去）：
  _prepare_metrics_csv  _write_metric_row  _save_checkpoint
```

收口方向：引入内部 execution controller `OnlineRecipeRun`，**持有 `stack`，不复制
`stack` 已有字段**。它可以持有 `csv_path / rng / resume_checkpoint / run_error` 这类 run
execution 状态，把这几个 helper 变成方法：

```text
OnlineRecipeRun.prepare_metrics_csv()
OnlineRecipeRun.write_metric_row(epoch, metrics)
OnlineRecipeRun.save_checkpoint(path, epoch=...)
```

这样调用点从"塞 6 个参数"降为"调方法"，但 `OnlineRecipeStack` 仍保持原职责：给 family
hooks 看完整 wired runtime，不承担 IO 行为。`OnlineRecipeRun` 是 recipe execution
controller，不是新的 stack / manager / scheduler。

注意：动到训练主入口，改动面大、有观点，**单独一轮、配合 `tests/scripts/` 全套验证**，
不与 Phase 1/2 混提交。这也补上了前轮 `done/SPRINT_small_function_consolidation.md` Phase C
（热点文件碎片整理）漏掉的 online.py（前轮 Phase C 列了 trainer.py / launcher.py /
batch_builder.py，未含 online.py）。

## 5. Non-Goals（明确不做）

```text
1. 不重做前轮 done/SPRINT_small_function_consolidation.md 的 Phase A/B/D（死 API 面 /
   转发壳 / 薄文件合并已落地）。本 sprint 只做 DI-by-arg + docstring 去重 + recipe 收类。
2. 不做 285 个单调用方 helper 的全量 inline——单文件私有 helper 是正常组织方式（前轮已裁定）。
3. 不动 registry/约定式抽象、跨家族 sibling 形状（runner/state/runtime 并行文件、
   rewards/functions/*.py 单函数文件）。
4. 不动合法的 thin-function + WHY docstring（如 unwrap_compile_and_ddp）——
   §2 只针对"复述自己 error"的那一小撮。
5. 前轮 Phase C 未做的其它热点文件（danbooru.py / trainer.py / launcher.py /
   batch_builder.py）不在本 sprint 范围；本 sprint 的 Phase 3 只收 online.py 一个。
```

## 6. 验收

```text
每个 Phase 独立 commit（全部已完成，VRL 分支 helper-passthrough-hygiene）：
  Phase 0  ✅ _save_checkpoint+docstring（2026-06-16）；_prepare_metrics_csv 补完 d7ec895；
           tests/scripts/test_online_lifecycle.py 11 passed。
  Phase 1  ✅ commit f6813f9；删 cosmos eval 两个 torch 透传参数、anima lazy 边界保留；
           test_cosmos_predict25_kling_eval.py 8 passed；grep 无未解释残留。
  Phase 2  ✅ 5 个候选全判 KEEP（WHY/contract，非 error 复述），无代码改动；pytest 绿。
  Phase 3  ✅ commit 5372ade；OnlineRecipeRun 作为 execution controller（持有 stack，不取代 stack）；
           tests/scripts/ 全套 58 passed。
LOC 不设指标（指标驱动的 LOC 清理正是上上轮被回滚的模式）；目标是降低"读一段流程要跳几次"。
实际净变化：online.py −15 行（192 insertions / 207 deletions），调用点参数从 2–6 个降为 0–1 个。
```
