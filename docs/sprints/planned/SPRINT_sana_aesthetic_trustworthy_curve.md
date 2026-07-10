# SPRINT: SANA 美学 GRPO 可信曲线 —— 小模型上验证"这套系统能把 reward 推上去"

状态:**planned(GPU 空闲即启动;当前被 cosmos 长跑占卡)**。

## 背景

Phase 1 的 10 个新家族全部只做到 rollout 级验证(出图合理、replay parity 0)。
**还没有任何新家族被证明训练曲线可信**——reward 上升、importance ratio 健康、
KL 不爆。架构瘦身、fp8、编排修复都是为训练服务的;这个 sprint 是它们的
第一次端到端兑现。选 SANA 因为它是信任基础最好的家族(与 diffusers
SanaPipeline 同 seed 视觉一致 + parity 0 误差,见 thin-seam sprint)且 1.6B
单卡毫无压力。选美学 reward 因为它零外部依赖(CLIP ViT-L + 内置 LAION MLP 头,
`vrl/rewards/assets/sac+logos+ava1-l14-linearMSE.pth`)且是经典可复现基线
(DDPO/flow-GRPO 都用它证明过管线)。

配置已就绪并通过 `load_config` 解析:
`vrl/config/presets/experiment/diffusion/sana/online_grpo_aesthetic.yaml`
(drawbench 192 prompts、LoRA r16、10 步去噪 CFG 4.5、每 prompt 16 样本、
`debug.first_step` 开启)。

## 范围

### 1. 主跑:美学 GRPO 曲线

```
python -m vrl.scripts.train --config experiment/diffusion/sana/online_grpo_aesthetic
```

- 首步 parity 守卫必须通过(`debug.first_step=true`,rollout logprob == replay)。
- 跑到 reward 曲线形态可判定(参考 flow-GRPO 的美学任务,数百 step 内应见
  明显上升;确切步数以首跑观察为准,不预设)。
- 记录并归档:reward 均值/分布曲线、importance ratio 直方图、KL、梯度范数、
  每 N step 的定性样本图(同 seed 前后对比)。

### 2. 判定标准(先于跑之前定死,防事后解释)

- **PASS**:美学分对 baseline(step 0 采样)统计显著上升,且样本图肉眼可辨
  变化方向与美学分一致(更干净的构图/光照,而不是崩坏出高分噪声)。
- **FAIL-学不动**:曲线平坦 → 按"诊断先于治疗"排查:advantage 是否全零
  (group 内 reward 方差)、LoRA 是否真的在更新(参数范数变化)、lr。
- **FAIL-reward hacking**:分数上升但样本图崩坏 → 记录现象,降 lr/加 KL 重跑
  一次;仍崩则归档为美学 reward 的已知形态,不追加防御工程(经典结果,预期内)。

### 3. 附带验证:LoRA + fp8 rollout 训练冒烟(第二优先)

master-free fp8 的 adapter 同步路径只有单测覆盖,没进过真实训练循环。主跑
PASS 后,同配置加 `model.rollout_quantization=fp8` 短跑(~50 step):

- 权重同步(adapter → 已丢 master 的 fp8 rollout)不报错;
- reward 趋势与 bf16 主跑同向;
- 显存峰值记录进 sprint(fp8 rollout 应显著低于 bf16)。

这条通了,"32GB 单卡 LoRA 训 17B"才算解锁。

## 验收标准

- 主跑达到 PASS 判定,曲线与样本归档到 `docs/runs/`(resolved_config 一起)。
- fp8 冒烟完成且结论明确(通过/失败+原因)。
- 过程中发现的任何管线 bug 修在根因层,并有回归测试。
- 无新增 fake:本 sprint 全部是真实端到端验证。

## 非目标

- 调出 SOTA 美学分——目标是"曲线可信",不是刷分。
- 扩到第二个家族(Lumina2/CogVideoX 的曲线验证是后续 sprint,等本 sprint
  的方法论定型)。
- PickScore/OCR 等其他 reward 的验证。
- 训练吞吐优化。

## 参考

- 实验配置:`vrl/config/presets/experiment/diffusion/sana/online_grpo_aesthetic.yaml`
- 美学 reward:`vrl/rewards/functions/aesthetic.py`、`vrl/rewards/models/aesthetic.py`
- SANA rollout 验证记录:
  `docs/sprints/done/SPRINT_thin_model_seam_and_ten_model_expansion.md`(条目 1)
- fp8 master-free 实现:`vrl/nn/quantization/fp8.py`、commit `3e054d7b`
