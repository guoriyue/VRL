# SPRINT: Anima Limb and Hand RL Experiment

## 0. Core Decision

本 sprint 只解决 Anima 的肢体和手部问题：

- 肢体连接更稳定。
- 关节方向更合理。
- 动作姿态看起来能站住、能发力、不违背常识。
- 手和手指更稳定，减少 missing hand、bad hands、extra fingers、melted fingers。

本 sprint 不讨论与 limb/hand 无关的目标。

核心判断：

- 主要瓶颈不是写 reward class，而是构建可追溯的 anime limb/hand 数据闭环。
- prompt 不能靠手写 20k，也不能靠 LLM 凭空编；prompt 要从 Danbooru metadata 的 tag 分布和受控模板派生。
- 训练 reward 第一阶段只保留两个信号：`anime_limb_plausibility` 和 `anime_hand_finger`。
- real human limb/hand 数据不能和 anime limb/hand 直接混成同一个训练分布；第一阶段 reward、校准集和 hard negative 都应该 anime-specific。

## 1. Current Code Reality

Anima 已经有 online GRPO 训练入口：

```text
vrl.scripts.diffusion.cosmos.train:train_anima_grpo
```

reward registry 已支持多 reward 加权组合：

```text
vrl/rewards/functions/registry.py
```

所以本 sprint 应该沿用现有 online GRPO 和 reward registry，只新增 limb/hand 数据集、reward 和实验配置。

## 2. Target Outcome

本 sprint 要让 Anima 在以下维度上可测量地变好：

- full-body prompt 下手、脚、手臂、腿更稳定。
- action pose prompt 下动作姿态更合理。
- 肩、肘、腕、髋、膝、踝的连接关系更少出错。
- missing hand、missing feet、extra limbs、disconnected arm、impossible leg bend 明显减少。
- bad hands、extra fingers、missing fingers、melted fingers 明显减少。

第一阶段只优化单人主体。多人互动、复杂遮挡、漫画分镜和连续动作不纳入第一阶段。

## 3. Anime vs Real Limb/Hand Decision

需要区分 anime limb/hand 和 real limb/hand。

可以共用的是 defect taxonomy：

```text
missing_hand
missing_feet
extra_limbs
disconnected_arm
impossible_leg_bend
bad_hands
extra_fingers
missing_fingers
melted_fingers
implausible_action_pose
```

不能直接共用的是 reward model 和 calibration data。原因：

- anime 人体比例、线条、遮挡方式、衣服结构和真实照片不同。
- real pose/hand detector 在 anime 图上经常漏检或误检，直接做训练 reward 容易 reward hacking。
- real hand 的纹理和 anime hand 的线稿/上色差异很大，直接混正样本会让 hand reward 学偏。
- 动作合理性可以借用现实世界常识，但最终判断要在 anime 图上校准。

第一阶段策略：

- `AnimeLimbPlausibilityReward` 用 anime positives 和 Anima hard negatives 训练。
- `AnimeHandFingerReward` 用 anime hand crops 和 Anima bad hand crops 训练。
- real human limb/hand 数据最多作为预训练参考或离线诊断，不进入第一阶段 reward calibration。
- 如果后续引入 real data，manifest 必须显式写 `domain: real`，并和 `domain: anime` 分开报告。

## 4. Dataset Strategy

数据不是来自一个单一现成数据集，而是四类数据合成一个训练闭环。

### 4.1 Danbooru Metadata and Images

用途：

- 构建 anime limb/hand prompt 分布。
- 构建 anime positive image manifest。
- 为 limb plausibility classifier 提供正样本。
- 为 hand/finger classifier 提供 hand-visible 正样本来源。

第一阶段过滤规则只围绕肢体和手部可见性：

```text
include:
  solo
  1girl | 1boy
  full_body
  standing | walking | running | sitting | kneeling | dancing | fighting_stance
  hands | hand_up | outstretched_arm | arms_up | arms_behind_back
  feet | boots | shoes
  highres
  score >= threshold

exclude:
  cropped
  portrait
  upper_body
  cowboy_shot
  multiple_girls
  multiple_boys
  text
  watermark
  lowres
```

prompt 不是直接复制 tag。tag 要转成训练 bucket 和 slot：

```text
standing_front
standing_side
walking
running
sitting_full_body
kneeling
action_pose
arms_visible
hands_visible
feet_visible
hand_focus
```

### 4.2 Prompt Provenance

`train_prompts: 20000` 和 `eval_prompts: 1000` 的来源是 Danbooru metadata 过滤后的 tag set，再经过受控模板和 bucket 规则派生。

prompt 来源链路：

```text
Danbooru metadata
  -> solo/full_body/limb-visible post filter
  -> tag normalization
  -> bucket assignment
  -> slot extraction
  -> controlled prompt templates
  -> dedupe and train/eval split
  -> baseline Anima failure mining
  -> hard prompt promotion
```

一个 Danbooru post 的 tag set 先被拆成结构化 slot：

```text
subject:
  1girl -> anime woman
  1boy -> anime man

framing:
  full_body -> full body

pose/action:
  standing | walking | running | sitting | kneeling | dancing | fighting_stance

limb constraints:
  hands_visible
  both_hands_visible
  feet_visible
  arms_visible

clothing/accessory anchors:
  boots | shoes | dress | coat | armor | school_uniform | sportswear

scene:
  simple_background | city_street | forest | studio_background | stage
```

模板示例：

```text
{subject}, full body, {pose}, {limb_constraints}, {clothing}, {scene}, detailed anime illustration
```

示例：

```text
source tags:
  1girl, solo, full_body, standing, boots, long_hair, simple_background

generated prompt:
  anime woman, full body, standing, both hands visible, boots, simple background, detailed anime illustration
```

`both hands visible`、`feet visible` 这种短语不是从单个 Danbooru tag 直接复制，而是实验目标注入的 constraint。它们必须写入 metadata，后续 reward 才知道这个 prompt 需要检查什么。

prompt manifest 必须记录 provenance：

```json
{"prompt": "anime woman, full body, standing, both hands visible, boots, simple background, detailed anime illustration", "metadata": {"bucket": "standing_front", "source": "danbooru2023_tags", "template_id": "limb_hand_v1", "source_post_ids": [1234567, 2345678, 3456789], "source_tags": ["1girl", "solo", "full_body", "standing", "boots", "simple_background"], "constraints": ["both_hands_visible", "feet_visible"], "domain": "anime"}}
```

v0 prompt 组成：

```text
13000 tag-derived limb/hand prompts:
  从 Danbooru solo/full_body/limb-visible metadata 的高频 tag 共现分布生成。

4000 balanced bucket prompts:
  人为补齐稀有 bucket，例如 feet_visible、walking、running、kneeling、action_pose、hand_focus。

2000 hard prompts:
  base Anima baseline generation 中 limb/hand 失败率高的 prompt family。

1000 neutral prompts:
  不强制 full-body，但仍然带手或动作约束，用来避免模型只学 full-body 模板。

1000 eval prompts:
  从同一生成器产出，但按 source_post_id、tag signature 和 template_id 去重后固定，不参与训练。
```

LLM 可以作为可选 paraphraser，但不能作为 prompt 的事实来源。允许做的事情只有：

- 把 tag template 改写成更自然的 prompt。
- 保留 metadata 中的 bucket、source tags、constraints 和 domain。
- 不新增没有来源的角色、多人互动或复杂姿态。

第一阶段最稳的做法是不用 LLM，直接用受控模板生成 prompt。这样 reward 失败时能追溯是哪个 tag bucket 或 constraint 出问题。

### 4.3 Anime Hand Crops

用途：

- 训练或校准 hand/finger classifier。
- 给 `AnimeHandFingerReward` 提供正样本和 hard negative。

来源：

- Gwern/Danbooru hand-visible crops 作为 positive hand crops。
- base Anima 生成图裁出的 bad hand crops 作为 negative hand crops。

hand crop 不直接作为 GRPO prompt dataset，而是作为 reward model/classifier 的训练数据和验证数据。

### 4.4 Base Anima Hard Negatives

用途：

- 用当前 base Anima 在固定 prompt suite 上生成大量候选图。
- 把失败样本挖出来作为 limb/hand classifier 的负样本。
- 挖出 reward hacking 样本，反复更新 reward。

第一阶段 failure labels：

```text
bad_hands
missing_hand
missing_feet
extra_fingers
missing_fingers
melted_fingers
extra_limbs
disconnected_arm
impossible_leg_bend
implausible_action_pose
cropped_full_body
wrong_person_count
```

推荐生成策略：

- 每个 prompt 生成 4 到 8 张。
- 固定 seed grid，保证 baseline 和训练后模型可对比。
- 每轮保留高 reward 但人工/VLM 判断失败的样本，加入 hard negative queue。

### 4.5 Human/VLM Label Queue

用途：

- 对 hard negative 做小规模高质量标注。
- 校准自动 reward 和真实视觉质量之间的相关性。
- 评估 RL 训练是否只是在骗 classifier。

标注问题只问 limb/hand：

```text
Are both arms anatomically connected?
Are the legs plausible for the requested action?
Are hands visible when requested?
Are fingers plausible enough for the image scale?
Which failure labels apply?
```

## 5. Dataset Artifacts

新增数据配置：

```text
configs/dataset/anime_limb_hand.yaml
```

新增 manifest 目录：

```text
datasets/anime_limb_hand/train_prompts.jsonl
datasets/anime_limb_hand/eval_prompts.jsonl
datasets/anime_limb_hand/positive_images.jsonl
datasets/anime_limb_hand/hand_crops.jsonl
datasets/anime_limb_hand/hard_negatives.jsonl
datasets/anime_limb_hand/preference_pairs.jsonl
```

`train_prompts.jsonl` 示例：

```json
{"prompt": "anime woman, full body, standing, both hands visible, boots, simple background, detailed anime illustration", "metadata": {"bucket": "standing_front", "source": "danbooru2023_tags", "template_id": "limb_hand_v1", "source_post_ids": [1234567, 2345678, 3456789], "source_tags": ["1girl", "solo", "full_body", "standing", "boots", "simple_background"], "constraints": ["both_hands_visible", "feet_visible"], "domain": "anime"}}
```

`positive_images.jsonl` 示例：

```json
{"image_path": "/data/danbooru2023/images/0001/1234567.jpg", "source": "danbooru2023", "post_id": 1234567, "score": 42, "tags": ["1girl", "solo", "full_body", "standing", "boots"], "domain": "anime"}
```

`hand_crops.jsonl` 示例：

```json
{"image_path": "/data/anime_limb_hand/hand_crops/000001.png", "source": "danbooru2023", "parent_image_path": "/data/danbooru2023/images/0001/1234567.jpg", "labels": ["hand_ok"], "domain": "anime"}
```

`hard_negatives.jsonl` 示例：

```json
{"image_path": "/data/anime_limb_hand/hard_negatives/000001.png", "source": "anima_base", "prompt": "anime woman, full body, standing, both hands visible, boots, simple background", "labels": ["bad_hands", "missing_feet"], "severity": 2, "domain": "anime"}
```

`preference_pairs.jsonl` 示例：

```json
{"prompt": "anime man, full body, walking, both hands visible, city street", "chosen": "/data/anima_limb_hand/pairs/0001_chosen.png", "rejected": "/data/anima_limb_hand/pairs/0001_rejected.png", "labels": ["limb_plausibility", "hand_finger_quality"], "domain": "anime"}
```

第一阶段规模目标：

```text
train_prompts: 20000
eval_prompts: 1000
positive_anime_images: 10000
base_anima_generated_images: 40000
human_or_vlm_labeled_hard_negatives: 5000-10000
positive_hand_crops: 10000
generated_bad_hand_crops: from base_anima_generated_images
```

## 6. Data Build Scripts

新增脚本：

```text
vrl/scripts/data/build_anime_limb_hand_prompts.py
vrl/scripts/data/build_anime_limb_hand_positives.py
vrl/scripts/data/build_anime_hand_crops.py
vrl/scripts/data/mine_anima_limb_hand_hard_negatives.py
vrl/scripts/data/export_anime_limb_hand_label_queue.py
```

`build_anime_limb_hand_prompts.py`：

- 输入 Danbooru metadata 本地路径。
- 过滤 solo、full_body、limb-visible、高质量图片。
- 统计 tag frequency 和 tag co-occurrence。
- 把 tag set 映射成 subject、framing、pose/action、limb constraints、clothing、scene slots。
- 使用固定模板生成 prompt，不依赖 LLM 生成事实内容。
- 按 bucket 平衡采样，保证 standing、walking、running、kneeling、feet_visible、hands_visible、hand_focus 都有覆盖。
- 按 `source_post_ids`、tag signature 和 `template_id` 去重并切分 train/eval。
- 输出 `train_prompts.jsonl`、`eval_prompts.jsonl` 和 bucket distribution report。

`build_anime_limb_hand_positives.py`：

- 输入 Danbooru image root 和 metadata root。
- 输出 positive image manifest。
- 不复制图片，manifest 只记录路径和元数据。

`build_anime_hand_crops.py`：

- 输入 positive image manifest 和 generated image manifest。
- 裁出 anime hand crops。
- 输出 positive hand crops 和 generated bad hand crops manifest。

`mine_anima_limb_hand_hard_negatives.py`：

- 输入固定 eval prompt suite。
- 调用当前 Anima checkpoint 生成候选图。
- 跑 limb/hand classifier 和诊断脚本。
- 输出待标注 hard negative manifest。

`export_anime_limb_hand_label_queue.py`：

- 把 hard negative manifest 导出标注队列。
- 每张图只问 limb/hand 相关问题。

## 7. Reward Design

第一阶段训练 reward 只有两个：

```text
anime_limb_plausibility
anime_hand_finger
```

### 7.1 AnimeLimbPlausibilityReward

用途：

- 作为第一阶段主 reward。
- 判断身体完整性、肢体连接、关节方向和动作姿态是否合理。
- 惩罚 missing feet、extra limbs、disconnected arms、impossible leg bend、cropped full body。
- 对 action pose prompt 特别关注动作是否能站得住、重心是否合理、手臂和腿是否和动作一致。

训练方式：

- 用 anime positive images 作为正样本。
- 用 base Anima hard negatives 作为负样本。
- 做多标签 classifier，而不是只做一个二分类分数。
- 输出总分和 failure mode 分数，方便看模型到底改善了什么。

训练标签：

```text
positive:
  complete_body
  plausible_limbs
  plausible_action_pose
  hands_ok
  feet_ok

negative:
  bad_hands
  missing_hand
  missing_feet
  extra_limbs
  disconnected_arm
  impossible_leg_bend
  cropped_full_body
  implausible_action_pose
```

### 7.2 AnimeHandFingerReward

用途：

- 作为手和手指的专项 reward。
- 检查 hand visibility、finger count stability、melted fingers、extra fingers、missing fingers。
- 单独存在是因为手部是小区域问题，直接混进全身 classifier 容易被身体完整性分数淹没。

训练方式：

- 用 anime hand-positive crops 作为正样本。
- 用 base Anima 生成图裁出的 bad hand crops 作为负样本。
- 先做 crop-level classifier，再把每张图的 hand crop scores 聚合成 image-level reward。

### 7.3 Diagnostics

诊断脚本可以报告以下信息，但第一阶段不进入 reward composition：

```text
keypoint_coverage:
  measure obvious limb/crop failures only.

tag_bucket_coverage:
  check whether generated image still matches the requested pose/action bucket.
```

这些诊断信号不能替代 anime-specific reward。它们只用于解释失败样本和筛选 label queue。

### 7.4 Reward Composition

第一阶段目标配置：

```yaml
reward:
  components:
    anime_limb_plausibility: 1.0
    anime_hand_finger: 0.4
```

解释：

- `anime_limb_plausibility` 是主优化目标。
- `anime_hand_finger` 是专项补偿，避免模型肢体变好但手仍然坏。

权重不是最终值。必须先用 baseline generated candidates 做 reward calibration，再开始 RL。校准时重点看两个主 reward 是否能把 anime positive images 和 Anima hard negatives 分开；如果分不开，不进入训练。

## 8. Training Experiment

新增实验配置：

```text
configs/experiment/diffusion/anima_preview3/online_grpo_limb_hand.yaml
```

目标 defaults：

```yaml
defaults:
  - /recipe/online/flow_matching_grpo
  - /model/diffusion/cosmos/anima_preview3
  - /sampling/image/512
  - /sampling/denoise/10_step_cfg_4_5
  - /dataset/anime_limb_hand
```

训练策略：

- 使用现有 Anima GRPO 入口。
- LoRA rank 从 16 或 32 开始。
- learning rate 保守设置，避免 reward 过拟合到少数 failure mode。
- `rollout.n` 从 4 开始，保证同一 prompt 下能比较多个候选。
- 保持 KL/ref regularization，避免模型只迎合 limb/hand classifier。
- 训练集以 limb/hand prompt 为主，但保留少量 neutral prompt，防止模型只会 full-body 模板。

## 9. Experiment Phases

### Phase 1: Build Dataset Manifests

交付：

- `configs/dataset/anime_limb_hand.yaml`
- `datasets/anime_limb_hand/train_prompts.jsonl`
- `datasets/anime_limb_hand/eval_prompts.jsonl`
- `datasets/anime_limb_hand/positive_images.jsonl`
- `datasets/anime_limb_hand/hand_crops.jsonl`

验收：

- prompt bucket 分布可打印。
- eval prompt 固定 seed 后可复现。
- 每条 prompt 都有 provenance metadata。
- 所有 image path 都是本地显式路径。

### Phase 2: Baseline Generation

交付：

- base Anima 在 eval prompts 上的固定 seed generation。
- candidate manifest。
- 初版 hard negative manifest。

验收：

- 每个 bucket 至少有足够候选样本。
- failure label schema 覆盖主要 limb/hand 问题。

### Phase 3: Reward Calibration

交付：

- anime positives、hard negatives、baseline generations 的 reward 分布报告。
- 两个 reward component 的 histogram 和 AUC。
- 高 reward 失败样本清单。

验收：

- anime positive images 的 limb/hand reward 分布明显高于 hard negatives。
- high reward but bad limb/hand 的样本被加入 hard negative queue。
- 如果 reward 不能区分正负样本，本阶段不进入 RL 训练。

### Phase 4: Train Limb and Hand Classifiers

交付：

- limb plausibility classifier checkpoint。
- hand/finger classifier checkpoint。
- reward wrapper。
- reward registry entry。

验收：

- classifier 在留出 hard negative set 上稳定区分主要 failure modes。
- inference batch path 和 VRL reward API 兼容。
- reward 输出记录 component scores，便于观察 reward hacking。

### Phase 5: First GRPO Run

交付：

- `online_grpo_limb_hand.yaml`
- first RL checkpoint。
- baseline vs RL 对比报告。

验收：

- `limb_plausibility_score` 上升。
- `hand_finger_score` 上升。
- `finger_defect_rate` 下降。
- `implausible_action_pose_rate` 下降。
- `missing_hand_rate` 下降。
- `missing_feet_rate` 下降。

### Phase 6: Hard-Negative Loop

交付：

- 训练后模型生成的新 hard negatives。
- 更新后的 classifier training set。
- 第二轮 reward calibration。

验收：

- reward hacking 样本减少。
- 人工 A/B 对比中，RL 模型在 limb/hand quality 上优于 base Anima。

## 10. Metrics

自动指标：

```text
limb_plausibility_score
hand_finger_score
both_hands_visible_rate
feet_visible_rate
finger_defect_rate
missing_hand_rate
missing_feet_rate
extra_limb_rate
disconnected_limb_rate
implausible_action_pose_rate
cropped_full_body_rate
```

人工评估：

```text
base_vs_rl_limb_preference_rate
base_vs_rl_hand_preference_rate
base_vs_rl_action_pose_preference_rate
```

评估规则：

- 所有模型使用同一 eval prompt suite。
- 所有模型使用同一 seed grid。
- limb、hand、action pose 分开打分，避免一个维度掩盖另一个维度。

## 11. Acceptance Criteria

本 sprint 完成时必须满足：

- 有固定的 anime limb/hand prompt eval suite。
- 每条 prompt 都能追溯到 source tags、template_id、bucket、constraints 和 domain。
- 有 anime positive images、anime hand crops 和 Anima hard negatives 的 manifest。
- reward calibration 证明 limb/hand reward 能区分正负样本。
- 有可运行的 Anima limb/hand GRPO config。
- 首轮训练相对 base Anima 在 limb/hand metrics 上改善。
- 所有数据来源和本地路径都可追溯。

## 12. Non-Goals

本 sprint 不做：

- 重新设计 Cosmos/Anima 模型结构。
- 构建通用真实人体姿态数据集。
- 把 real human limb/hand 数据直接混进 anime reward calibration。
- 训练多人互动、复杂遮挡或连续动作。
- 把 pose detector 或 anime tagger 做成第一阶段主训练 reward。
- 把 Danbooru 图片重新发布到 repo。
- 用单一 VLM prompt 代替可校准 reward。

## 13. Risks and Mitigations

reward hacking 风险：

- 每轮收集 high reward bad samples。
- limb/hand classifier 必须持续吸收新 hard negatives。
- 自动指标之外保留人工 A/B。

domain mismatch 风险：

- 不把 real human limb/hand data 当主训练数据。
- real detector 第一阶段只作为诊断参考，主 reward 来自 anime positives 和 Anima hard negatives。
- manifest 必须保留 `domain` 字段，防止 anime 和 real 数据被无意混在一起。

prompt overfitting 风险：

- eval prompt 和 train prompt 分开生成。
- 每个 bucket 固定 train/eval split。
- 保留 neutral prompt mix，防止模型只会 full-body 模板。

## 14. Implementation Checklist

- [ ] 新增 `configs/dataset/anime_limb_hand.yaml`。
- [ ] 新增 `datasets/anime_limb_hand/train_prompts.jsonl` seed set。
- [ ] 新增 `datasets/anime_limb_hand/eval_prompts.jsonl` seed set。
- [ ] 实现 Danbooru metadata 到 limb/hand prompt manifest 的构建脚本。
- [ ] 实现 tag normalization、slot extraction、template rendering 和 provenance metadata。
- [ ] 输出 prompt bucket distribution 和 train/eval de-dup report。
- [ ] 实现 positive image manifest 构建脚本。
- [ ] 实现 anime hand crop manifest 构建脚本。
- [ ] 实现 base Anima hard negative mining 脚本。
- [ ] 实现 label queue export。
- [ ] 实现 `AnimeLimbPlausibilityReward`。
- [ ] 实现 `AnimeHandFingerReward`。
- [ ] 实现 limb/hand diagnostic report，不进入第一阶段 reward composition。
- [ ] 注册 reward 到 `vrl/rewards/functions/registry.py`。
- [ ] 新增 `configs/experiment/diffusion/anima_preview3/online_grpo_limb_hand.yaml`。
- [ ] 新增 reward calibration report 命令。
- [ ] 跑 base Anima eval baseline。
- [ ] 跑首轮 GRPO。
- [ ] 生成 baseline vs RL report。

## 15. References

External:

- Danbooru2023 dataset: https://huggingface.co/datasets/nyanko7/danbooru2023
- Danbooru2021 notes: https://gwern.net/danbooru2021
- Gwern anime crops: https://gwern.net/crop
- Anime hand detection model: https://huggingface.co/deepghs/anime_hand_detection
- DeepGHS hand detection docs: https://dghs-imgutils.deepghs.org/main/_modules/imgutils/detect/hand.html

Local:

- `/home/mingfeiguo/Desktop/wm-infra/vrl/rewards/functions/registry.py`
- `/home/mingfeiguo/Desktop/wm-infra/vrl/scripts/diffusion/cosmos/train.py`
