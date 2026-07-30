# SPRINT: 固化 generation → models import floor

状态：**DONE（2026-07-30，CPU-only）**。来源：[[SPRINT_layering_audit]] 最后一项架构门禁。

## 目标

`vrl/generation` 可以依赖 model 的公共构建与身份边界，但不应直接依赖某个 family 实现或 model 内部 step。当前合法 import 只有四组：

```text
vrl.models.interfaces
vrl.models.loader
vrl.models.dtypes
vrl.models.checkpoint_identity
```

前 3 组是 model runtime floor；`checkpoint_identity` 是只读、family-neutral 的 checkpoint 协议边界。当前约定成立，但没有测试阻止新的 `vrl.models.families.*` 或 `vrl.models.steps.*` import 悄悄进入 generation。

## 落地结果

在 `tests/architecture/test_generation_rollout_boundaries.py` 增加了 AST gate：

1. 遍历 `vrl/generation/**/*.py` 的全部 import，包含函数内 lazy import。
2. 对所有 `vrl.models` 边，只允许上面的四个前缀。
3. 报错列出 `relative/path.py: imports target`，复用现有 `_python_files` 与 `_format_violations`。

允许前缀应写成一个明确的测试 taxonomy，不能从当前 import 自动派生。它是刻意维护的架构边界；自动派生会让任何新越界 import 自动进入白名单，失去 gate 的意义。

执行前复核发现，旧 `_imports` 只返回 `ImportFrom.module`，会漏掉
`from vrl import models`，也无法区分 `from vrl.models import loader, families` 的两个 alias。
因此本 sprint 同时把 scanner 修正为 alias-aware 的 syntactic targets，并解析相对 import：

```text
from vrl import models                       -> vrl.models
from vrl.models import loader, families      -> vrl.models.loader
                                                vrl.models.families
from ...models import steps                  -> vrl.models.steps
```

前缀判断使用 exact-or-dot 语义，`vrl.models.interfaces_bad` 不会误入
`vrl.models.interfaces` 白名单。永久测试同时覆盖合法、非法、mixed aliases、lazy import
与相对 import。

## 保持不变

- 不移动 `checkpoint_identity.py`。它不反向 import generation，当前职责是 model build/checkpoint 的身份协议。
- 不禁止 generation 通过 registry/launch contract 调用 family；本 sprint 只约束静态 import 边。
- 不改 `vrl/families` 的 import-lightness gate；现有门禁已经覆盖该方向。
- 不新增生产 helper、facade 或 package。现有 AST helper 是跨多个架构测试共享的真实抽象，保持不变。
- 不把允许集合做成生产 ALL_CAPS 常量。它只属于架构测试；作为刻意隔离的 boundary taxonomy，测试内常量是合理例外。

## 验收

```bash
.venv/bin/python -m pytest \
  tests/architecture/test_generation_rollout_boundaries.py -q
```

gate 的反例由永久 scanner/classifier 测试覆盖，不在生产文件留下临时 mutation。

本 sprint 不需要 GPU，不运行训练、生成或仓库全量测试。

## References

- `tests/architecture/test_generation_rollout_boundaries.py`
- `vrl/generation/ray/launcher.py`
- `vrl/generation/execution/worker.py`
- `vrl/models/checkpoint_identity.py`
- [[SPRINT_layering_audit]]
