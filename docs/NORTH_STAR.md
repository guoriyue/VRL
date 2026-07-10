# NORTH STAR — visual-rl 的赌注与路线图

> 这份文档回答一个问题:**visual-rl 要成为什么?** 以及**凭什么别人会用它,而不是去 prompt Claude 改 slime/verl 来跑自己的 image/video RL 实验。**

## 0. 一句话定位（the bet）

**visual-rl = 视觉生成式 RL 的专用引擎**:diffusion + 自回归(AR)+ video/world-model 的 RL 训练,在这里**一份 config 就能正确跑通、跨模型族、且结果可复现**。

类比(这是用户要的框架):

| 框架 | 赌注(他们押对的那一件事) |
|---|---|
| DeepSpeed | "大模型训练是显存受限" → ZeRO / FSDP |
| Ray | "ML 需要通用分布式编排" → actor/task |
| vLLM / SGLang | "推理是 KV 显存受限" → PagedAttention + continuous batching |
| **visual-rl** | **"视觉生成式 RL 是一个独立的系统+正确性问题(多步去噪轨迹 + 连续隐空间 log-prob + 像素奖励 + rollout↔train 共卡),LLM-RL 框架结构上变不成它"** |

**关键的独立判断**:不要试图"成为 vLLM"。vLLM 是**推理服务**,而我们的品类是**训练**。给 diffusion checkpoint 做 HTTP serving 这件事 diffusers/ComfyUI 已经占了,去抄它只会稀释赌注。我们赢的方式不是"比谁快"——自家 probe 已证明 rollout DiT ~94% MFU,**性能天花板已近**;我们赢在**正确性 + 可复现 + 覆盖广度**,这三样是 LLM-RL 框架抄不过来的。

## 1. 为什么这是一个真品类,而不是一个 feature

一个想 RL-tune 自己 diffusion/video 模型的研究者,今天会去 hack slime/verl——然后被框架的核心抽象在每一步顶住,因为**阻抗失配是结构性的,不是改个 config 能解的**:

| LLM-RL 框架假设 | 视觉生成式 RL 的现实 |
|---|---|
| rollout = 逐 token 单次前向 + KV-cache | rollout = **多步去噪轨迹**(20–50 步),每步一次全 DiT 前向 |
| log-prob = 类别分布的 cross-entropy | log-prob = **连续隐空间的 Gaussian 密度**(flow-matching / SDE 转移) |
| 奖励在 text 上 | 奖励在**解码后的像素/视频**上(VAE decode → image/video reward model) |
| 显存瓶颈 = KV-cache | 显存瓶颈 = **去噪轨迹存储 + DiT 激活 + VAE**;PagedAttention 无关 |
| 单一解码范式 | **两套范式**:diffusion(连续) + AR-image(VQ token、CFG、image-token logits) |
| 无条件 / 前缀条件 | **world-model 条件**:reference image/video、I2V、V2W |

→ 这就是用户的洞察:"别人没法 prompt Claude 改 slime 来跑他们的 RL 实验"——因为要改的不是配置,是整套"sample→log-prob→reward"管线。**谁先把这套做对、做顺、做全,谁就拥有这个品类。**

## 2. 我们已经有的护城河(grounded,top-3)

三块"text-LLM 框架最难复制"的资产(已读代码确认):

1. **去噪轨迹 rollout 编排** — `vrl/rollouts/orchestration/continuous/`(producer/queue/consumer + staleness)+ `vrl/generation/diffusion/executor.py`(逐步去噪、每步记 `log_prob/prev_sample_mean/std_dev_t`)+ `vrl/trainers/weight_sync.py`(version-stamped 权重同步)。多步轨迹捕获 + 每步 replay 重算 old_log_prob,是 AR-token rollout 根本没有的形状。
2. **统一模型族契约** — `vrl/models/interfaces/runtime.py`(`RuntimeModel.replay_forward`)+ `vrl/rollouts/families/registry.py`。当前 23 个注册项（17 diffusion + 6 AR）藏在同一个 replay 契约后面；descriptor-driven diffusion 族只需模型目录 + registry entry，trainer/algorithm 零改动。
3. **diffusion 专属 RL 数学** — `vrl/algorithms/grpo/continuous.py` + `diffusion_nft.py` + `vrl/math/diffusion/flow_matching.py`。Flow-DPPO 的隐空间非对称 KL 信赖域、DiffusionNFT 的 likelihood-free 严格 on-policy、连续 log-prob 上的 TIS/RS 精度漂移校正——这些和类别 RL **数学上不兼容**,不是换 config 能搬的。

护城河的本质是 **"正确性"**:`replay` / `old_log_prob` 契约(verl 规则:别污染 old_log_prob)是我们的 run 可信、而 slime-hack 不可信的根本原因。

## 3. 距离 category-defining 还差什么(诚实,已重排优先级)

gap audit 给的清单我**重排过**(独立判断,见 §0):去掉了"建 vLLM 式 serving API = P0"(那不是这个品类的赌注)。真正的 P0 是**信任**和**上手**,因为它们决定"别人会不会用",而不是"能不能跑":

| 级别 | gap | 为什么是这个级别 |
|---|---|---|
| **P0 信任** | **没有可复现的 reference 曲线** | 没人会采用一个无法证明"reward 真在涨"的 RL 框架。这是采用的第一阻塞,也是赌注的证明。现状:只有 SD3.5-OCR 一条像样曲线,且明确声明"不是 metric report"。 |
| **P0 广度** | **23 个注册项只有 1 个 validated** | 每个 validated 族 = 一批采用。但广度依赖 P0-信任(先有可信曲线方法论)。**注意:不要陷入"验证全部族"的跑步机**——先钉 3 条(每范式一条:image-diffusion / video-diffusion / AR-image)作为信任锚。 |
| **P1 正确性** | 持续硬化 replay/log-prob 契约 + E2E 不在 CI | 这是护城河本身。convergence 回归现在 CI 看不见(GPU 训练不在 PR CI)。 |
| **P1 上手** | quickstart/依赖矩阵已落地，但 clean-source 基线未守住 | README 已有第一条 job 与 extra 映射；当前缺口是 frozen lock、干净 source 安装验证和冲突 extra 的明确隔离。 |
| **P1 广度** | adding-family 指南已落地，需持续跟 registry 现实同步 | 指南现按 descriptor/custom-adapter/AR 三条路径写；后续用行为测试而非目录 roster 防漂移。 |
| **P2 性能** | torch.compile / FA-3 / overlap 覆盖 | MFU 北极星。但 rollout 已 ~94% MFU——**这是优化不是赌注**,别当卖点。 |
| **P2 规模** | 多卡/多机 online FSDP 仍 gated | 大模型(Wan 14B、video)需要;但**单卡正确性优先**,多机是放大器不是地基。 |

## 4. 路线图(把现有散落的 sprint 归到赌注下)

按"先让人信、再让人用、再让人扩、最后让人快/大"排序:

### 阶段 A —— 信任层(P0,最高优先)
- **3 条 reference 曲线**(image-diffusion=SD3.5/Flux、video-diffusion=Cosmos/Wan、AR-image=Janus),每条:固定 prompt 集 + BLOCK 测试判显著性 + 一键复现命令 + reward-vs-epoch 的 csv。归并现有 `SPRINT_cosmos_predict2_2b_trustworthy_curve` / flux algo validation。
- **E2E 收敛回归进 CI**(哪怕 nightly + tiny model):防止"能跑但不学"的回归隐形。

### 阶段 B —— 上手层(P0)
- ✅ **`README` quickstart**:已有一条命令启动 SD3.5-OCR，并写明预期的 optimizer/reward 信号。
- ✅ **依赖矩阵**:已映射训练、reward、data、eval extras；冲突环境必须显式隔离。
- ✅ **"Why visual-rl, not slime/verl"**:README 与本文已说明结构性失配。
- **可信 clean-source 基线**:frozen lock、source editable-install/config-resolve smoke、离线关键正确性测试（`SPRINT_trustworthy_clean_build`）。

### 阶段 C —— 广度层(P0/P1)
- ✅ **"Adding a Model Family" 指南**:按 descriptor-driven diffusion / custom adapter / AR adapter 三条真实路径说明。
- 按信任层方法论,逐族补可信曲线(归并现有 30+ 个 validation sprint)。

### 阶段 D —— 性能层(P2,持续)
- torch.compile / FA-3 / rollout↔train overlap。**定位为"保持 MFU-bound",不是卖点。**(已有 compile/MFU sprint 归此。)

### 阶段 E —— 规模层(P2)
- 解 online 多卡 FSDP 的 gate(rank-split + Ray 协调),让 Wan-14B / 长视频可训。

## 5. 北极星指标(怎么算我们赢了)

1. **复现性**:一个外部研究者,`git clone` + 一条命令,能在自己机器上复现一条我们发布的 reward-上升曲线。(今天:否)
2. **上手时间**:从 clone 到第一个 RL job 在跑 < 10 分钟。(今天:路径已给出，clean-source gate 落地中)
3. **加族成本**:加一个新 diffusion/AR 族到能跑 RL < 1 天,只动注册表 + 族目录,不动 trainer/algorithm。(今天:descriptor 路径与指南已落地，仍缺外部贡献验证)
4. **正确性可证伪**:每条护城河契约(replay old_log_prob、flow-matching log-prob、staleness)都有"改坏源码→测试红"的守护。(进行中)
5. **采用信号**:出现"我用 visual-rl 跑了我的 image-RL 实验",而不是"我 hack 了 slime"。

---

**TL;DR**:别做 vLLM 的推理服务克隆。做**视觉生成式 RL 的 DeepSpeed**——押"这是个独立的正确性+系统问题"。护城河(去噪 rollout / 统一族契约 / diffusion RL 数学)、quickstart 与加族路径已经有了；当前第一缺口是**让人信(可复现曲线 + 真正执行的正确性门)**。性能别当卖点(已近 MFU 顶),正确性和覆盖才是。
