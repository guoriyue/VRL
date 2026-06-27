# SPRINT: DeepSeek-VL2-Tiny 作为 grounding judge / reward 模型

状态：**planned（2026-06-27）**。目标：把 `DeepSeek-VL2-Tiny`(VLM,带 visual grounding)接成一个新的 model-backed reward,用"生成图里物体的空间正确性 / VLM 判别"给 RL 打分。**先解依赖硬冲突 + 验信号有效性两道门,再谈接入。**

## 0. 结论先行（最投机的一项,两道硬门）

- VL2-Tiny **能塞进 32GB**(1.0B active / 3.37B MoE,~10–15GB 权重),接入 seam 与现有 model-backed reward(kling/pickscore)一致,工程上可行。
- 但有两道**硬门**,过不了就不做:
  1. **依赖硬冲突**：VL2 钉 `transformers==4.38.2`,reward extras 现 `>=4.49`。这是**版本区间不相交**,不是放宽就能解,可能要独立 runtime 进程/独立 extra。
  2. **信号有效性**：grounding-as-reward 是**噪声 reward**(解析脆、judge 不稳)。接之前必须验证它与既有 reward / 人评**正相关**,否则就是给 RL 喂噪声。
- 这两道都属"先证明值得做"的 KILL-RISK,不是落地任务。

## 1. 证据

### 1.1 新 reward 要实现的接口（已有清晰模板）

- `vrl/rewards/base.py:88,94`：`async def score(rollout) -> float` / `score_batch(...)`;`_init_reward_model()`(111)把 model-backed reward 接上 runtime + artifact builder。
- `vrl/rewards/ray/model.py:21`：RewardModel 协议 `__call__(artifact, request) -> Mapping[str, float]`,框架按 `score_key` 选分。
- 模板:`vrl/rewards/models/kling_video_reward.py`、`vrl/rewards/models/pickscore.py`(懒加载 + `score_media(media, prompt, request)` 返回含 `score_key` 的 dict)。
- 注册:`vrl/rewards/functions/registry.py` 的 `_REWARD_REGISTRY`(model_factory 用 import-path 字符串,如 `"vrl.rewards.models.deepseek_vl2_judge:deepseek_vl2_judge_model"`)。

→ 新增 2 文件(function wrapper + model)+ 改 registry/`__init__`/pyproject + 一个 yaml,**与 kling reward 的接入完全同形**。

### 1.2 VL2-Tiny 推理 + grounding（`~/Desktop/deep-research/DeepSeek-VL2`）

```python
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
proc = DeepseekVLV2Processor.from_pretrained("deepseek-ai/deepseek-vl2-tiny")
model = AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True, torch_dtype=torch.bfloat16)
# grounding prompt: "<image>\n<|ref|>The cat.<|/ref|>." → 输出 "<|ref|>...<|/ref|><|det|>[[x0,y0,x1,y1]]<|/det|>"  (坐标 0–999)
```

- HF `deepseek-ai/deepseek-vl2-tiny`;1.0B active / 3.37B total;~10–15GB 权重,**fits 32GB**(官方 <40GB,可选 `incremental_prefilling` chunk_size=512)。
- grounding 解析:正则取 `<|ref|>/<|det|>`,坐标 `/999*W` 反归一化(`deepseek_vl2/serve/app_modules/utils.py:270-313`)。

### 1.3 依赖冲突（硬）

VL2 `pyproject.toml`：`transformers==4.38.2`(PIN);本仓 reward extra 现 `transformers>=4.49`。**区间不相交** → 不能同进程简单共存。

## 2. 应该做什么

### Phase 0 — 两道 KILL-RISK 门（先做,过门才继续）

1. **依赖可行性 probe**：实测 VL2-Tiny 在本仓 `transformers` 版本下能否加载(很可能不能)。结论三选一:(a) 真实兼容区间存在→放宽约束;(b) 不存在→VL2 进**独立 reward runtime 进程/独立 extra**(`execution=pool`,与主训练环境隔离);(c) 都不可行→停。
2. **信号有效性 probe**：对一组生成图(含"物体在/不在指定位置"的正负样本),跑 VL2 grounding 打分,验证分数与**预期空间正确性 / 既有 reward** 的相关性。**不相关 → 停**(不给 RL 喂噪声 reward)。

### Phase 1 — 接入（两门都过后）

- 新增 `vrl/rewards/models/deepseek_vl2_judge.py`(`__call__` + factory)与 `vrl/rewards/functions/deepseek_vl2_judge.py`(RewardFunction wrapper),照 kling 模板。
- 打分方案(由 Phase-0 选最稳的)：(A) 二值——prompt 中物体被 grounding 命中=1 否则 0;(B) IoU——若 metadata 有 GT box;(C) VLM yes/no judge。默认从 (A)/(C) 起,(B) 仅在有 GT 时。
- 注册进 `_REWARD_REGISTRY`,加 `configs/reward/deepseek_vl2_judge.yaml`,deps 进 reward extra(或独立 extra)。

## 3. 风险

| 风险 | 处理 |
|---|---|
| **transformers 区间不相交**(4.38.2 vs ≥4.49) | Phase-0 门;大概率走独立 runtime 进程隔离 |
| **grounding-as-reward 是噪声** | Phase-0 信号相关性门;不相关不接 |
| **32GB 与 rollout 共卡 OOM** | probe 实测峰值;`execution=pool` 隔离或分时 |
| **MoE 路由延迟 / 解析脆** | 批量打分;正则多策略兜底 + 解析失败记日志 |

## 4. 非目标

- 不把 VL2 当生成模型(它只做理解/grounding,不出图)。
- 不在信号有效性未验证前接进任何真实 RL recipe。
- 不为接 VL2 而降级本仓 `transformers`(会波及整个训练栈);冲突走隔离,不走全局降版。

## 5. 验收

- [ ] Phase-0 两份 probe 结论落本文件:依赖可行性(同进程/隔离/不可行)+ grounding 分与空间正确性的相关性。**两门都过才进 Phase 1。**
- [ ] 接入后:新 reward 走 `RewardModel.__call__ -> {score_key: float}` 契约,`registry` 注册,clean-install 不破既有 reward 栈。
- [ ] 一句话决策:接 / 隔离接 / 不接 + 依据。

**参考**：`vrl/rewards/base.py:88-138`、`vrl/rewards/ray/model.py:21`、`vrl/rewards/models/{kling_video_reward.py,pickscore.py}`、`vrl/rewards/functions/registry.py`;`~/Desktop/deep-research/DeepSeek-VL2`(`inference.py:63-103`、`deepseek_vl2/serve/app_modules/utils.py:270-313`、`pyproject.toml`);HF `deepseek-ai/deepseek-vl2-tiny`。
