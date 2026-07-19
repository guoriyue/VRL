# SPRINT: 死符号清除 + 单调用者合并（纯机械）

状态：done（2026-07-10）。父记录：`SPRINT_fbag_00_overview.md`（同在 `done/`）。

> 4 个零散小问题,分布在 4 个否则内聚的文件里。每条都经对抗性 verify 保留 + 本人独立 grep 复核。
> 全是低风险机械改动,可以一个 commit 收掉。

## 0. 一句话

3 个零调用者的死函数 + 1 个单调用者的概念拆分。其中两个死函数直接删除；共享
`write_png` 接回真实 data-prep 消费者并删除私拷；谓词并回唯一决策点。

## 1. 死函数(form-1,零调用者)

### 1.1 `utils/media.py:write_png` —— 死,且有重复私拷

`grep -rn write_png vrl/ tests/` 全仓只有 def(`media.py:101`)+ `__all__`(`media.py:239`),
**零真实调用者、零测试调用者**。更糟:它本该的消费者 `vrl/scripts/data/video_world.py` 自己
写了个私有 `_write_png`(`video_world.py:816`,在 151/201 调用)——即共享的 `write_png` 既死
又被抄了一份(form-4)。

**已采用去重方案**:`video_world.py` 改用共享 `media.write_png`,删掉本地 `_write_png`——
一处 owner,消灭重复。`media.py` 的 docstring 自述是"rewards/生成脚本/data-prep 的
domain-neutral 共享家",video_world 正是它服务的 data-prep 消费者。

### 1.2 `config/builders.py:section_to_dataclass` —— 死

`grep -rn section_to_dataclass vrl/ tests/` 只有 def(`builders.py:116`);唯一其他命中在
`docs/sprints/done/SPRINT_config_as_signatures.md`(历史文档,非代码)。未进 `__all__`。删。

### 1.3 `rollouts/batch/ops.py:shuffle_and_rebatch_batches` —— 死

零调用者(只有 def `ops.py:95` + `__all__` `ops.py:206`)。`trainer.py:724-727` 的注释记录了
rebatch 路径已被移除——这是它 dead-semantics 的来源。删除 def + `__all__` 条目。

## 2. 单调用者概念拆分(form-3)

### 2.1 `rollouts/collector/config.py:_has_sde_sampling` → 并入 `_add_derived_values`

`_has_sde_sampling`(`config.py:126-130`)零外部调用者,唯一使用点是 `config.py:122`
`_add_derived_values` 内部(唯一调用者)。函数体是个 3 行谓词
`any(name in values for name in (sde_type, sde_window_size, sde_window_range))`,只在那一处用。
把它内联进 `_add_derived_values`——一个决策读在一处,不跨函数跳转。

**辨析(为何这条是拆分而非保留)**:它不是 lazy-import 边界,不是可独立测试的概念(无自己的
测试),不是跨家族一致形状——就是一个决策被拆成两个私有函数、第二个恰好只有第一个调用。
正是 AGENTS.md form-3 的定义。

## 3. 验证

删除前逐一 `grep -rnw <symbol> vrl/ tests/` 复确认零调用者(§1 三个已确认);内联后跑
`tests/rollouts/` + `tests/config/` + `tests/utils/`(若有)确认零回归。这批不改行为,测试应全绿。

执行结果：`section_to_dataclass` / `shuffle_and_rebatch_batches` 已无生产/测试调用者；共享
`write_png` 现有两个真实生产调用者，本地 `_write_png` 私拷已清零；`_has_sde_sampling` 已并回
唯一决策点。config、collector、trainer、video-world 定向测试与 Ruff 均通过。仓库级验证结果
由本轮最终提交统一记录。

## 4. 明确不动

- 4 个文件本身都是 cohesive(media.py/builders.py/batch/ops.py/collector/config.py)——**只删/并
  这几个符号,不重构文件**。
- `collector/config.py` 的其余 `_` build-step helper 是**均匀命名的构建步**,保持一致形状
  (grep/debug 友好),不要因为"也像单调用者"就一起并——它们的一致性价值高于合并。

## 引用

- `vrl/utils/media.py:101,239` + `vrl/scripts/data/video_world.py:151,201,816`
- `vrl/config/builders.py:116`
- `vrl/rollouts/batch/ops.py:95,206` + `vrl/trainers/online/trainer.py:724-727`
- `vrl/rollouts/collector/config.py:122,126-130`
