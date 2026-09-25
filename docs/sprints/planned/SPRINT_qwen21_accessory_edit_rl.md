# SPRINT: Qwen-Image-2.1 小物件精细编辑 RL（动漫 + 真人饰品）

状态：**planned**（2026-09-23）。先做 Phase 0 探测，不训练。

## 0. 一句话

用 RL（不做监督微调）让 Qwen-Image-2.1 学会**精细的小物件编辑**：在动漫角色和真人照片上加 / 换 / 去掉
耳环、发夹、蝴蝶结、项链、choker、眼镜、帽饰、指甲颜色、小纹身这类饰品，其他地方（脸、身份、背景、别的饰品）不变。

## 1. 从上一个 sprint 带过来的四条教训（SPRINT_qwen21_object_move_edit_rl）

1. **训练前先测 base 在"训练采样设置"下的成功率**（512 px / 10 步 / SDE）。要落在 20–60%：太低 RL 没东西可放大
   （物体移动 run1 首轮 83% 的组全零），太高说明不需要练。这是 Phase 0 的硬关卡。
2. **奖励必须给"忠实但没做到"的输出中间分**，否则 GRPO 会把所有忠于原图的输出一起压低，模型学会重画整张图
   （run1：held-out 0.107 → 0.001）。排序要求：改对了 > 没改（忠实） > 改错饰品 > 重画。
3. **评测不能只看自己的奖励**：要有 held-out 的独立判断（另一个模型或人工目测）和公开基准的回归检查。
4. **指令本身必须合理、可执行，并经过验证。** 物体移动的指令是"把笔记本电脑移到画面左边"——左边没有桌子，
   电脑只能悬空，这条指令本身就不成立；base 的"复制 / 不动"有一部分是在回应一个不合理的要求。当时的可行性规则只查了
   "左边有空地"，没查"那里有没有能放东西的表面"。合理的写法是锚定到场景里真实存在的东西："把电脑放到椅子上"
   "放到床头柜上面"。本 sprint 的对应要求：
   - 每条指令要有**场景依据**：add 耳环要求耳朵可见（不是被头发完全挡住），指甲颜色要求手可见，项链要求脖子可见；
     change / remove 要求那件饰品确实在图里（标签 + 裁剪复核）。
   - 指令生成后用 VLM 做一次**可执行性检查**（"这张图里能不能给她加上耳环 / 能看到她的耳朵吗"），不通过的丢弃；
     抽查 30 条人工确认。
   - 指令写法要具体到位置和对象（"给她左耳加一个小的银色星形耳环"），而不是泛泛的"加点饰品"。

## 2. 数据（只需要源图 + 指令 + 元数据，RL 不需要目标图）

- **动漫源图**：`deepghs/danbooru2024-sfw`（已确认可下载；按 tar 索引单张取图，不整包下载）。用 WD 标签模型
  重新打标签，只留单人（`solo`）、头部清晰、分辨率够的图；饰品清单来自标签。许可：Danbooru 图是同人作品，
  **仅限研究 / 内部使用**。
- **真人源图**：Open Images V7（图片 CC BY 2.0，标注 CC BY 4.0），带 Necklace / Glasses / Sunglasses / Hat /
  Tiara / Scarf 等框；没有耳环 / 戒指标注，这些由奖励模型自己判断。可选补充 Fashionpedia（衣服上的蝴蝶结、珠饰、刺绣）。
  CelebAMask-HQ 有耳环 / 项链 mask 但非商用，只作评测参考。
- **指令**：由标签程序化生成三类——add（图里没有的饰品）、remove（图里有的）、change（换颜色 / 换形状 / 换种类）；
  元数据 `{domain, op, target, source_item?}`，用的是标签词表里的词，不从自然语言里反解析。
- **held-out**：按图片 / 角色划开的自建 split（不与训练共享图片或角色）；回归检查用 GEdit-Bench 人像 / 增删子集、
  DLEBench（小目标编辑，目标占画面 1–10%）。

## 3. 奖励（全部包装现成模型）

`R = P^γ · (0.5 + 0.45·Δ − 0.4·W)`，γ≈2：

- **Δ 目标变化**：编辑前后目标饰品"有没有"的概率差（add 为正向，remove 反向，change = 新饰品出现 × 旧饰品消失）。
  动漫：`SmilingWolf/wd-eva02-large-tagger-v3`（Apache-2.0）在**头部裁剪放大图**和全图上的标签概率；
  真人：Qwen3-VL-2B/4B 回答"是否戴着 X"的 yes 概率（VQAScore 做法），同样先裁剪放大——小物件不裁剪，VLM 基本看不见。
- **W 改错**：其他饰品标签 / 问题的最大变化（加了错的东西、把别的饰品弄没了）。
- **P 保持（门槛）**：编辑区域外 EfficientLoFTR 留在原位的比例（沿用 object_move 的实现）× 身份相似度
  （动漫：`deepghs/ccip` 角色相似度；真人：DINOv2 人脸裁剪，AdaFace 仅研究用）；编辑区域面积超过约 8% 视为失败。
- 不用 EditReward 当主信号：它只看整图，看不到小耳环，而且偏好过度编辑的图。

**关卡（训练前）**：每个领域约 30 例人工构造（正确编辑 / 不动 / 改错饰品 / 重画 / 整图风格变化），排序必须正确；
每个用到的标签 / 问题在约 100 张裁剪图上抽查可靠性。

## 4. 执行顺序与关卡

- **Phase 0 探测（GPU，只生成不训练）**：动漫 40 张 + 真人 40 张，每张 2 条指令，base 在 1024/40 步和训练采样设置
  两种下各生成，人工目测 + 初版奖励打分。**关卡**：确认 base 在这类编辑上确实有明显失败（哪几类），且训练设置下成功率 20–60%。
- Phase 1 奖励 + 排序关卡；Phase 2 数据构建 + 奖励与人工判断的一致性；Phase 3 dry run + 训练（沿用 OCR 单卡配方）；
  Phase 4 held-out 判定（学会 / 持平 / 练坏）+ 对比页。

## 5. 非目标 / 注意

- 不做监督微调。不写手工规则（颜色阈值、区域规则）。
- 许可：动漫源图仅研究用；真人源图优先 Open Images；人脸身份模型的商用许可问题见 §3。

## 6. Phase 0 准备（2026-09-23，CPU / 网络，未用 GPU）

- 动漫：从 `deepghs/danbooru2024-sfw` 前 3 个 tar 按索引单张取 450 张（不整包下载）；WD SwinV2 v3 打标签 → 单人且短边 ≥768 的 207 张；
  deepghs 头部检测（YOLO onnx，CPU）→ 201 张有头，头高 ≥15% 画面的 188 张；去掉 monochrome / no_humans 等。目测检测框准确；
  标签对小饰品有漏标（头饰 + 蝴蝶结没标出），印证"要裁剪放大再判"。
- 真人：Open Images V7 validation（CC BY 2.0）里"恰好一张非群体、非描绘的人脸且脸高 ≥12%"的 1473 张，
  优先有饰品框或耳朵框的，下载 160 张；按元数据旋转。目测仍混有倒置 / 侧躺 / 解剖图 / 拼贴 / 画作 → 需要 VLM 可执行性检查。
- 指令候选 203 条（动漫 120，真人 83），每条具体到饰品、样式和位置；操作以 add 为主（源图里现有饰品不多）。
- 可执行性检查：Qwen2.5-VL-7B 回答"图是否正、主体是否一个人、要改的部位是否清楚可见、要换 / 去掉的饰品是否真的在"，
  取 yes 概率 ≥0.5，每个领域最多 40 条。
- 探测：base 在 1024/40 步（每条 1 张，目测用）和 512/10 步（每条 4 张，近似训练设置下的成功率；注意这是 ODE，
  训练用 SDE）两种设置下生成。脚本在会话 scratchpad `accessory/`（`build_candidates.py`、`vlm_check.py`、`run_probe.sh`），
  排在物体移动 run2 的评测之后自动运行。

## 7. 扩展探索：动漫 / 真人肖像还能练什么（2026-09-24，两个 Opus 子代理，CPU/网络，未生成图）

报告与 150+150 条逐张看图核实的指令在会话 scratchpad `explore/{anime,portrait}/`（REPORT.md、prompts.jsonl）。
所有"base 失败率"都是从公开基准推断，**训练前必须在训练采样设置下实测**（每类约 40 例）。

| | 动漫（danbooru2024-sfw，仅研究用） | 真人肖像（Open Images V7，CC BY 2.0） |
|---|---|---|
| 推荐先跑 | **改表情**：Qwen 编辑最弱项是"没让改的地方保持不变"（GEditBench v2 保持分 <1000），改表情逼它动脸又保住角色；CCIP 不受表情影响，WD 头部裁剪读表情稳 | **换衣服款式 / 加一层**：替换类在基准只拿中等分（Kontext GEdit material 5.5），最可能落在 20–60%；衣服面积大，VLM 不裁剪就能判 |
| 备选 | 改衣服颜色 | 表情（能改但身份漂移，FED-Bench 身份相似度 0.58；ArcFace 仅研究用） |
| 不推荐 | 瞳色（目标太小，估计 >80% 失败）、发色（可能太容易） | 换颜色、换背景（本身基本做对，失败只在顺带改脸） |
| 判"改到了" | WD v3 头部裁剪标签概率差 | Qwen3-VL "是否穿着 X" yes 概率 |
| 判"其他没变" | CCIP 角色相似度 + LoFTR 头部以外原位 | 人脸身份（研究用 ArcFace / 商用退回 DINOv2）+ 人像外原位 + 改动面积 |

数据：
- 动漫初取 450 张（前 3 个 tar 顺序取）→ 单人有脸非漫画 185 张；代理额外加的"去暗示性"过滤会砍到 79 张，**按用户要求不加**（数据集本身已是 Danbooru SFW 分级；不取 explicit 分级）。
  按 Danbooru 评分 ≥15、宽高 ≥1024、单人重取（元数据 metadata.parquet 里符合的 68.9 万张），40 分片各取评分最高 25 张 → 1000 张，
  可用 978（评分中位数 166，短边中位数 1447 px）：改发色 707、改表情 774、改衣色 476、瞳色 581、换背景 402、加眼镜 822。
  脚本 scratchpad `explore/anime_v2/refetch.py`；源图页 <https://claude.ai/artifact/9L95chb9U9ouqGFb1dMng1>。
- 真人 160 张核对 141 张，可用 58（41%）；不可用主因：画质 18、脸不可见 15、公众人物 13、未成年或疑似 13、多人 8。全量估计衣服类约 350、发型类约 500。
  Fashionpedia 不值得下图（宽松许可仅 251/1158，上半身衣服 >8% 的只有 62 张）。
- 内容边界：真人只做普通换装 / 发型 / 饰品 / 表情 / 背景，不做暴露化、不改体型、不显年轻；未成年、公众人物排除。

共同的坑：基准没有动漫子集，Qwen-Image-2.1 没有按类别的编辑分数；社区反馈输出会整体错位几像素、轻微放大（Qwen-Image #229），
奖励里的原位检查要保留——但 object_move 的 LoFTR 背景项在单调纹理上会冤枉正确编辑（见 SPRINT_qwen21_object_move_edit_rl §16），先修再复用。

## 8. 扩成"动漫角色任意属性编辑"（2026-09-24，用户要求：不只是表情，脸、发色、衣服颜色、饰品都要能改）

**任务定义**：一条指令改角色的一个属性，其余不变。七个家族共用一套元数据 `metadata.attribute_edit = {family, target_tags, removed_tags, head_box, ...}`：

| 家族 | 指令例子 | 应出现的标签 | 应消失的标签 |
|---|---|---|---|
| expression | Change her facial expression from a smile to an angry frown… | frown, angry | — |
| hair_color | Change her long blonde hair to mint green… | green_hair | blonde_hair |
| eye_color | Change the color of her blue eyes (the irises) to red… | red_eyes | blue_eyes |
| outfit_color | Recolor her pink dress to navy blue, keeping its shape… | blue_dress | pink_dress |
| add_accessory | Add round thin-framed glasses to her face / Put a red beret on her head | glasses / hat | — |
| remove_accessory | Remove her hair ornament completely and fill in what was behind it | — | hair_ornament |
| background_swap | Replace the plain background with a snowy forest… | snow, forest | simple_background |

**数据**（已就位，`data/external/anime_attribute/`，manifest 提交在 `manifests/anime_attribute/candidates.jsonl`）：
按 Danbooru 评分 ≥15、宽高 ≥1024 重取的 1,000 张 + 30 张旧核实图；WD-v3 标签、头部框；由标签程序化生成 **7,194 条**候选
（发色 1,414、加饰品 1,383、表情 1,334、瞳色 1,162、衣色 950、去饰品 549、换背景 402），其中 30 条表情指令逐张核实过，其余靠
Qwen2.5-VL 可执行性检查过滤（`probe/vlm_check.py`）。要更多图：`probe/refetch_more.py`（按索引单张取，不整包下载）。

**Phase 0 探测包**（`data/external/anime_attribute/probe/run_probe.sh`，单张 48 GB 卡约 1.5 h）：可执行性过滤 → base 在 1024/40 与 512/10×4 出图 →
BEFORE|AFTER 盲判图。关卡：每个家族的 base 成功率（盲判）落在 20–60% 才进 RL；基本都成的家族（预计换背景、换发色）不练，只做回归检查。

**奖励设计（对齐"要 solid、不要启发式"的要求）**：主信号用 VLM 判官——对每张编辑图问"目标属性是否变成了 X""原属性是否不见了""角色是否还是同一个""其余是否没变"，
取 yes 概率的几何平均，不经过框和阈值（物体移动那边同一套判官的对照实验正在验证，一致率达标就沿用）。
WD-v3 标签差（头部裁剪放大）、CCIP 角色相似度、DINOv2 分块背景一致率作为**诊断量**并入日志，用来核对判官、不进训练键。
无论哪种，都先在 Phase 0 的盲判输出上验证排序（改对 > 忠实没改 > 改错 > 重画），再开训。

**4×48 GB 训练**：沿用 OCR/物体移动的单卡配方做数据并行（FSDP2 路径），每条指令样本数可以开到 16–32；VLM 判官单独占一张卡走 reward service，
不与训练争显存。多家族混训时按家族分层采样，held-out 按图片划分、每个家族 ≥40 条。
