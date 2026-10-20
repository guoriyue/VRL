# SPRINT：用 Claude 开发 VRL 的护栏——仓库内 review skill 与确定性 hook

状态：**planned（2026-10-20）**。P2 会修改 `.claude/` 下的配置，需要用户逐项同意后才执行。

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
