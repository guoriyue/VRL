# SPRINT：由证据生产器支撑的推理质量测试

状态：**PLANNED；结构验证器已落地，证据生产器缺失（2026-07-18）**。

## 目标

昂贵训练启动前，一个显式启用的测试必须证明：同一份 resolved experiment 能通过三条相互独立的路径生成有效输出：

1. 官方 native/reference 推理路径；
2. 仓库实际使用的 production rollout 路径；
3. 从持久化 trajectory artifact 独立重建的 replay 路径。

证明仍由 `tests/quality/` 拥有。生产代码不得导入测试包，不得围绕质量阶段暂停或恢复训练，不得缓存 PASS，
也不得根据测试状态做 runtime policy 决策。operator 或 launch workflow 在 `vrl-train` 前立即运行该测试；
skip 表示**没有证明**，不能视为通过。

本 sprint 证明推理正确并检测坍缩，不承诺训练一定提升质量。

## 当前仓库事实

已经落地：

- `tests/quality/protocols/families/` 覆盖从 `FAMILY_REGISTRY` 派生的全部 canonical entry，
  没有第二份手工维护的 family vocabulary。
- 协议通过 `PolicySemantics` 的正交轴分类 trainable policy：temporal organization、step kind、
  action distribution 和 trajectory layout。
- CPU 测试验证 config/model identity、artifact hash、media decode、shape、condition sensitivity、
  replay tolerance、alignment direction、segment order 和必需 corruption 名称。
- `tests/quality/test_sana_real_inference.py` 使用独立 denoise loop 证明正确的 SANA FP16/no-autocast
  production model-build/forward path，并证明已拒绝的 BF16 outer-autocast 路径会发生偏离；
  该测试没有经过完整 `GenerationRuntime`、worker、binding 或 trajectory，因此不能冒充 rollout 证明。
- `tests/quality/test_protocols.py` 阻止 `vrl/` 导入 `tests.quality`，也阻止新增 production
  `vrl.quality` package。

仍然缺失：

- 没有通用 producer 真正为一份 resolved experiment 执行 native、production、replay 和 scorer 路径。
- `tests/quality/evidence.py` 当前接受 `replay_max_abs_error`、matched/shuffled alignment 和
  corruption score 等 manifest scalar。只验证阈值不能证明这些数字确实来自声明的 artifact 或 scorer。
- checked-in profile 是 test oracle，不是该模型已经跑过真实推理的证据。
- CI 是 CPU-only；模型或 GPU 不可用时，显式启用的 real-checkpoint test 可能 skip。

## 必需架构

### 测试拥有的 producer

在 `tests/quality/producers/` 下增加 producer 支持。producer 可以导入 production runtime code 和
upstream/native library，但依赖必须单向：生产代码绝不能导入 producer。

每次 producer 调用接收一份 resolved experiment config，并将 raw record 和 artifact 写入 caller 指定的
输出目录。它必须：

1. resolve 唯一且不可变的 model identity 和 revision；
2. 在全新进程中启动官方 native/reference path；
3. 在另一个全新进程中启动完全一致的 production rollout path；
4. 持久化 replay 所需的 trajectory input；
5. 不复用 rollout memory object，独立重建 replay；
6. 执行 pinned independent scorer 和所有已注册 corruption；
7. 从这些 raw record 计算全部报告指标；
8. 写入 input、output、source、dependency lock、resolved config、scorer、protocol、可选 checkpoint
   和 environment identity 的 hash。

evidence validator 必须从逐样本 raw record 重新计算 summary。当引用的 artifact 已包含足够信息派生某个值时，
不得接受 caller 提供的 scalar 作为证明。

### Family-specific native 边界

upstream protocol 不同时，native adapter 保持 family-specific。这是合理的 thin-file boundary：
独立实现不能调用它要验证的 production executor。共享 artifact、process、hashing、replay 和 scorer 机制放入
通用 producer 支持。

只有当 model identity、sampling protocol、conditioning 和 trainable policy semantics 确实相同时，
一个 adapter 才能覆盖多个 registry entry。alias 不产生新 coverage；不同 checkpoint/task variant 需要独立覆盖。

### Artifact 生命周期

producer output 是一次性验证 artifact。把它放在 import graph 之外，使用 `*_preflight` 这类可识别名称；
只保留解释 launch 决策所需的 report、contact sheet 和最小 provenance。不要把大型生成媒体或 scratch trajectory
提交到仓库。

## 正反证明

每个真实 producer 测试至少包含一个有效正例和一个故意破坏的反例，证明断言确实会失败。

通用反例：

- source/config/model/checkpoint hash 不匹配；
- replay tensor 或 scheduler step 被扰动；
- prompt 或 reference 被置换；
- solid color、patch block、noise、blur、saturation 或 confetti；
- 不同 seed 产生完全相同输出。

joint-denoise policy 还必须覆盖错误 scheduler、timestep mapping、dtype、autocast、frame count/order、freeze、
repeat、flicker 和 reverse。

causal-token policy 还必须覆盖非法 token range、premature EOS、token repetition、prefix/schedule mutation、
错误 decoder path 和 segment reorder。continuous-token policy 还必须运行完整 flow decoder；缩短的 smoke config
不是 native-quality proof。

reference-conditioned image/video/world policy 必须使用真实 condition asset。零 tensor 和 synthetic placeholder
不能证明 conditioning 正确。

## 人工检查边界

pixel statistics 单独不能认证视觉质量。report 必须列出每个 native 和 production artifact 的 prompt/condition、
path 和 hash，并生成供人工打开的 contact sheet 或 video index。人工检查补充 machine assertion，但不能替代 replay、
identity 或 corruption check。

## 实施范围

只新增或扩展 test-owned asset：

- `tests/quality/producers/`：subprocess、artifact、replay、scorer 和 family-native producer 支持；
- `tests/quality/test_real_inference_preflight.py`：为一份 resolved config 运行或验证 produced evidence；
- `tests/quality/evidence.py`：从 raw record 重新计算指标；
- `tests/quality/protocols/`：protocol/version fixture；
- `tests/quality/README.md`：精确的 pre-launch command 和 skip semantics；
- family-specific independent path 的 focused real-checkpoint test。

不要新增：

- `vrl/quality/`；
- 复制进 `ModelFamilyEntry` 的 quality field；
- `vrl/scripts/train_worker.py` 或 training phase orchestrator；
- 为质量测试增加的 trainer/Ray pause-resume state；
- `quality.enabled` 或 `skip_preflight` 一类 YAML switch；
- 平行的 `SUPPORTED_FAMILIES` constant。

## 完成门槛

CPU-only gate：

1. registry entry 新增或重命名时，registry-derived profile coverage 仍完整；
2. production-to-test import 保持不可能；
3. evidence summary 从 raw record 重算，篡改 scalar/artifact/hash 时会失败；
4. 每种 modality corruption 都有 true/false regression test；
5. 未知 protocol field 和过期 model/config revision fail closed。

对每个 canonical entry，只有满足以下 real-checkpoint gate 才能声称已验证：

1. native/reference、production 和 independent replay 都实际执行；
2. 正确输出通过，注册的错误 dtype/scheduler/condition/segment case 失败；
3. pinned scorer 按注册 margin 将 matched/clean artifact 排在 shuffled/corrupted artifact 之前；
4. artifact 已被打开检查，report 记录人工检查但不谎称自动完成；
5. report 绑定后续 training launch 使用的精确 source 和 experiment identity。

第一条 vertical slice 是 SANA，因为它已经有独立证明的正确和错误 precision path。随后增加一个 video
joint-denoise entry 和一个 causal-token image entry，再扩展到整个 registry。producer 未实现的 family 始终是
**unproven**，绝不能隐式通过。

## 架构卫生

保留 `FAMILY_REGISTRY` 作为唯一、刻意隔离的 taxonomy/config table，并从中派生 profile coverage。
schema/version、protocol name、checkpoint name 和 test fixture constant 都是真实 ALL_CAPS 边界；不要把 typed field
复制成手工维护的 validation set。

只有在提供独立 protocol、process ownership 或 versioned wire boundary 时，才保留 thin native adapter、subprocess
entrypoint 和 artifact codec。不要为了减少行数而压平这些边界，也不要为没有真实 family 差异的 policy profile
创建对称空 adapter。

## 参考

- `tests/quality/README.md`
- `tests/quality/evidence.py`
- `tests/quality/test_protocols.py`
- `tests/quality/test_real_inference_preflight.py`
- `tests/quality/test_sana_real_inference.py`
- `vrl/families/registry.py`
- `vrl/families/semantics.py`
- `docs/MODEL_TAXONOMY.md`
