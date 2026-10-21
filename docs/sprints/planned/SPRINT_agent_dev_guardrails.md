# SPRINT：用 Claude 开发 VRL 的护栏——仓库内 review skill 与确定性 hook

状态：**P0–P2 已实施（2026-10-20/21），只剩一项等用户决定**：要不要把 CLAUDE.md 的编码规则搬进 review skill（见 §4 "待用户决定"）。用户拍板后，按决定执行完这一项，就删除本文件。

**2026-10-21 用户决定：`.claude/` 下的东西一律不进 git。** 本文原计划把 skill 和 hook 放进仓库（P2 写的是项目级 `.claude/settings.json`），这一条作废，见 §5。

来源：PyTorch Conference NA 2026 Day 1：046《Fighting Agents with Agents: Claude in PyTorch CI》、016 PyTorch AI coding playbook BoF、061 KernelAgent、032 vLLM nightly triage。

## 0. 结论先行

PyTorch 的做法：

- 模型本身是通用的，项目专属规则放在仓库的 `.claude/skills/` 下（18 个 skill）。`pr-review` skill 由 bot 和人共用。
- 高风险动作由确定性 hook 拦截，而不是靠 prompt 约束。
- 审计结论必须按 JSON schema 输出，并且宁可判 unsure，也不要错误地排除。

VRL 的规则现在散落在 `CLAUDE.md`（31 行）和 memory 中，只能靠模型自觉遵守；没有 hook，也没有 review skill。`.claude/` 下只有 `skills/reward`，它是一个 symlink。`settings.local.json` 里只有 permissions。

## 1. 阶段

### P0. 仓库内的 review skill（`.claude/skills/review/`，参照 reward skill 的形式）

把以下规则收拢成一份 checklist，供主 agent 自审和 review subagent 共用：

- 签名规则：`*` 和关键字参数；
- 局部 helper 规则；
- cleanup 范围规则：只改名不算结构清理；
- 测试规则：不写回显字段的测试，需要 GPU 的测试打上 gpu 标记；
- "不做投机或保护性机制"；
- "单调用方的 helper 要内联"；
- 每个 commit 要给出阅读顺序，并把机械改动和语义改动分开（016）；
- 是否引入 device-to-host sync：`vrl/trainers` 和 `vrl/algorithms` 中目前有 44 处 `.item()`（016 计划中的 AI lint 项）；
- 参数化测试一行都没生成时，应当判失败（061：KernelAgent 遇到过"生成了 0 个 config，CI 照样通过"）。

CLAUDE.md 只保留这份 skill 的入口和少数硬规则，避免两处重复。memory 中可以由代码或配置表达的规则也迁移过来。

**验收**：用这个 skill 回审最近 10 个 commit，能复现以前人工 review 抓到的问题类型。

### P1. 审计输出结构化

审计类 subagent（dead-surface、签名、层次审计）统一输出 `{item, verdict: keep|delete|unsure, evidence_paths, callers_counted}`，prompt 明确要求"拿不准就判 unsure"（046 CI Advisor 的做法）。只有 delete 且附有证据路径的条目才进入执行。

### P2. 确定性 hook（需要用户逐项同意）

1. PreToolUse(Bash) 拦截 `git push`，要求本条消息里有明确的 push 授权。这对应 CLAUDE.md 的"never push unless asked"。
2. PreToolUse(Bash) 直接拒绝针对 `.venv` 的 `pip install` 和 `uv pip install`。这对应 CLAUDE.md 的"Never pip install into .venv by hand"。
3. 可选：长任务的时间预算提醒（046 用 hook 在 job 限时前提醒 Claude 收尾）。

实现前先读 hooks 文档（`update-config` skill），配置放在项目级 `.claude/settings.json`，不放 local。

**验收**：故意触发这两条命令，都会被拦截，并给出可读的原因。

## 2. 不做

- 不在 GitHub 上部署 `@claude` workflow、triage bot 或 CI Advisor：这是单人仓库，没有外部 PR 洪流（046 的场景是每天上百个 PR）。
- 不加 `AGENTS.md` 软链接：目前只用 Claude Code。

## 3. 实施记录（2026-10-20/21）

规则：`.claude/` 下的 skill、hook、settings 都留在本机，不提交，`.gitignore` 整体忽略 `.claude/`。所以下面的 skill 和 hook 只在本机，仓库里只有 CLAUDE.md 指向 skill 的一行入口和空参数化的 pytest 配置。

| 条目 | 状态 | 证据 |
| --- | --- | --- |
| P0 review skill 的检查项 | 已实施（本机） | `.claude/skills/review/SKILL.md` §1：CLAUDE.md 的规则改写成检查项，另加 device-to-host 同步、空参数化、数值、所有权、commit 形状几条 |
| P0 CLAUDE.md 只留入口 | 换了做法，见"待用户决定" | `CLAUDE.md` 最后一条 bullet；skill 开头写明"不重述 CLAUDE.md" |
| P0 memory 中可用代码/配置表达的规则迁入 | 已实施 | skill 第 2、4、9 条（单调用方 helper、不做投机机制、config/state/command）；空参数化改成 pytest 配置 `empty_parameter_set_mark = "fail_at_collect"`（`build(pytest): fail collection on an empty parametrize`） |
| P0 验收 | 已实施 | 用 skill 让 subagent 回审 10 个 commit，找出 5 条应修的问题（静默变 0 的 reward、空操作的设置、测量和语义改变混在一个 commit、注释引用后面才有的行为、死代码），都已修复 |
| P1 审计 JSON 输出 | 已实施（本机） | skill §2：`{item, verdict: keep\|delete\|unsure, evidence_paths, callers_counted, reason}` |
| P2.1 拦 `git push` / P2.2 拦往 `.venv` 装包 | 已实施（本机） | `.claude/hooks/guard_bash.py`，登记在 `.claude/settings.json` 的 PreToolUse(Bash)；只在命令位置匹配，解析出错时放行；测试在 `.claude/hooks/test_bash_guard_hook.py` |
| P2.3 时间预算提醒 | 不做 | 没有对应需求 |

`vrl/trainers` + `vrl/algorithms` 里的 `.item()` 现在是 45 处（立项时 44 处），按计划留在 skill 第 7 条作为 AI lint，不加计数门禁。

**待用户决定**：

1. CLAUDE.md 是否瘦身到"只留 skill 入口 + 少数硬规则"。现在的做法反过来：编码规则仍放在 CLAUDE.md，skill 只引用、不重述，两处也没有重复。CLAUDE.md 每个会话都会加载，skill 要按需加载；把签名、helper、测试规则搬进 skill，写代码时就不一定能看到这些规则。所以没有照计划搬。
2. CLAUDE.md 那一行入口指向的 `.claude/skills/review/` 只在本机，别的 checkout 里不存在。保留这一行，还是删掉。
3. origin 上已有的 `.claude/skills/reward`（指向 `reward/skills/reward` 的软链接）是这条规则之前提交的，还在跟踪。要不要也撤掉。
