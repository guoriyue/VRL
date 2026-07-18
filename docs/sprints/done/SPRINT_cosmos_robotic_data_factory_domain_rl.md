# SPRINT: Cosmos Robotic Data Factory — 只接真实公开数据源

Status: **DONE — public-source substrate only (2026-07-18)**. The DROID/Bridge/
JRDB provenance and configuration substrate landed. This does not prove that a
Cosmos policy learned; learning evidence is owned by the separate trustworthy-
curve sprint.

> ⚠️ **更新（2026-06-28，SPRINT_future_reward）**：本文下面引用的 `target_video_similarity`
> reward 及其 `target_video_similarity_probe` 脚本**已删除** —— 实测证伪(pixel-L1,最优解=糊均值,
> 判别 gap 仅 ~4%)。改用 `target_dino_similarity`(DINOv2 感知锚)+ `motion_dynamics`(RAFT 运动 guard),
> 判别探针换成 `vrl.scripts.eval.future_reward_discrimination_probe`。下文凡提到 `target_video_similarity`
> / 旧 probe 的命令与权重均已过时,按 SPRINT_future_reward 替换。

核心边界：训练数据必须来自可下载、可解析、有 provenance 的真实公开数据集。仓库不再保留 `sidewalk_world` / `home_world` 这类本地手工 manifest bridge，也不把目标域名字伪装成数据集。

## 0. 核心结论

先不要问“调哪个 reward”。先问：

1. 生成数据要喂给哪个下游系统：perception、prediction、planning，还是 policy？
2. 目标域真实 heldout 里 Cosmos 现在失败在哪个维度？
3. 失败维度是数据覆盖/SFT 问题，还是可排序、可验证、适合 RL 的问题？
4. 训练样本是否能追溯到真实公开数据源，而不是手工拼出来的本地占位 manifest？

第一条可跑路线现在收窄为：

| 路线 | 真实来源 | 作用 |
|---|---|---|
| **DROID / LeRobot target V2W** | `lerobot/droid_100` on Hugging Face | 从真实机器人 manipulation demo 解析首帧和 target clip，给 Cosmos V2W 做 target-aware RL |

Sidewalk delivery 和 household chores 仍然是重要目标域，但必须等具体公开源 importer 接入后再进入训练配置。

## 1. 已落地 substrate

保留：

- `PromptExample` artifact schema 新增 `target_image` / `target_video`。
- target artifact 从 prompt example 传到 collector metadata，再进入 reward artifact metadata。
- `target_video_similarity` reward：读取 manifest 里的 `target_video` / `target_image`，比较生成视频和真实 target clip。
- `target_video_similarity_probe`：离线检查 target reward 是否能正常读真实 target media。
- `video-world-targets` importer：从公开 LeRobot/Hugging Face 数据集下载/解析真实 robot videos，生成 `reference_image` + `target_video` manifest。
- `droid_target_v2w` dataset config。
- `online_grpo_droid_target_240p` Cosmos config。

删除：

- `vrl/scripts/data/robot_world.py`
- `sidewalk-world-local` / `home-world-local`
- `sidewalk_delivery_v2w` / `home_manipulation_v2w`
- `online_grpo_sidewalk_delivery_240p` / `online_grpo_home_manipulation_240p`

原因：这些是“本地已有素材转 manifest”的桥，不是公开数据集下载/解析器。它们容易让人误以为仓库已经支持某个真实 dataset。

## 2. 当前真实数据路线：DROID / LeRobot target clips

准备数据：

```bash
python -m vrl.scripts.data.setup video-world-targets \
  --repo-id lerobot/droid_100 \
  --name droid_targets \
  --limit 50 \
  --eval-limit 8 \
  --max-target-frames 33
```

输出：

```text
data/external/video_world/manifests/droid_targets_train.jsonl
data/external/video_world/manifests/droid_targets_eval.jsonl
data/external/video_world/droid_targets_report.json
data/external/video_world/references/*.png
data/external/video_world/targets/*.mp4
```

manifest row 形状：

```json
{
  "prompt": "Put the marker in the pot",
  "reference_image": "video_world/references/droid_000001_first.png",
  "target_video": "video_world/targets/droid_000001_target.mp4",
  "task_type": "video2world",
  "metadata": {
    "source": "droid",
    "source_repo": "lerobot/droid_100",
    "source_split": "main",
    "source_episode": "000001",
    "source_video": "videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4",
    "source_frame_index": 0,
    "decode_method": "pyav_http_target_clip",
    "conditioning": "first_frame"
  }
}
```

训练配置：

```bash
python -m vrl.scripts.data.setup for-experiment diffusion/cosmos_predict2/online_grpo_droid_target_240p
```

实际训练仍然需要先跑 discrimination probe：

```bash
python -m vrl.scripts.eval.target_video_similarity_probe \
  --manifest data/external/video_world/manifests/droid_targets_eval.jsonl \
  --out outputs/droid_target_similarity_probe.jsonl
```

## 3. Reward 策略

`target_video_similarity` 是 baseline 学习信号，不是最终 robot quality verifier。

它能学到：

- 真实 demo continuation 的粗视觉/时序相似性。
- 首帧 conditioned V2W 是否朝真实 target clip 的方向发展。
- 比纯 Kling visual/motion reward 更贴近任务数据。

它不能单独保证：

- 接触物理一定正确。
- 物体状态谓词一定满足。
- 机器人动作一定可执行。
- 生成数据一定能提升真实 policy。

当前 recipe：

```yaml
reward:
  components:
    target_video_similarity: 0.80
    kling_video_reward: 0.20
```

原因：

- `target_video_similarity` 是从真实 demo 解析出的任务信号。
- Kling 只做视觉/运动质量 guard，避免生成结果退化成低质量视频。

## 4. 后续公开数据源，不许用本地占位桥

> **落地（2026-07-08）：两个目标域的 importer 均已接通。**
>
> - **Home chores / household manipulation —— 零新代码，路线已验证**：现有通用 LeRobot
>   解析器（`video-world-targets`，v2.0/v2.1 布局自适配）直接吃 BridgeData V2 的 LeRobot 公开
>   移植 `IPEC-COMMUNITY/bridge_orig_lerobot`（53k 家用厨房操作 episode、完全公开无 gate）。
>   3-episode 端到端实测通过（下载→PyAV 解码→首帧 PNG+target mp4→manifest+report→validation
>   PASS，真实指令 prompt 如 "put small spoon from basket to tray"）。正式导入命令：
>   `python -m vrl.scripts.data.setup video-world-targets --repo-id IPEC-COMMUNITY/bridge_orig_lerobot
>   --name bridge_home_targets --source bridge --camera observation.images.image_0 --limit N --eval-limit M`
> - **Sidewalk delivery —— 新 importer `jrdb-targets` 已落地**（`vrl/scripts/data/jrdb.py`）：
>   直接解析 JRDB 官方解压布局（`images/<camera>/<sequence>/<frame>.jpg`），按 stride 切
>   非重叠 clip，导出首帧+target mp4+全套 provenance+report+train/eval split+validation。
>   **无匿名下载器是 JRDB 的注册墙所致**（per-user license）——命令在布局缺失时响亮报错并给出
>   下载指引；prompt 由 sequence 名的地点 token 模板生成并以 `prompt_source` 字段如实标注
>   （JRDB 无语言标注）。CPU 测试 `tests/data/test_jrdb_import.py`（合成布局，4 测，含全命令路径）。
> - 训练 config 仍按原则**不预建**：等各目标域 audit（§0 的四问）过了再落。

原要求存档：

Sidewalk delivery 目标域下一步应该接：

- JRDB / JackRabbot：真实 mobile robot 视角，人群、室内外校园、social navigation。
- 接入要求：下载器、序列解析器、首帧/clip 导出、source report、train/eval split、provenance metadata。
- 在这些代码落地前，不保留 sidewalk delivery 训练 config。

Home chores / household manipulation 目标域下一步应该接：

- RoboCasa：公开模拟家庭/厨房任务 demo。
- DROID / BridgeData / Open X-Embodiment：真实 robot manipulation demos。
- 接入要求：不要要求用户先手工写本地 manifest；importer 必须直接解析公开 dataset layout。

## 5. 工程边界

保持不变：

- `RewardInferenceArtifact` / `RewardInferenceRequest` 协议层保留。
- `RewardFunction._init_disk_artifact_reward` 保留。
- trainer / GRPO / Cosmos replay/logprob runtime 不按 target domain 分叉。
- thin reward function 和 model 文件保留：`functions/` 是训练 runtime adapter，`models/` 是可离线调用的 scoring implementation。

新增真实数据约束：

- source-backed V2W validation 对普通 V2W 仍要求 `reference_image`。
- 当 experiment 使用 `target_video_similarity` 时，production validation 额外要求 `target_video` artifact 存在。
- source report 必须记录 `repo_id`、`source_split`、`decode_method`、train/eval rows、manifest paths 和 validation summary。

非目标：

- 不用 `Kling overall_reward` 作为 Robotic Data Factory 主 reward。
- 不在没有 target domain audit 前直接跑长 RL。
- 不把 rare-event frequency 当主 reward；rare event 用采样和过滤控制。
- 不加没有 detector/label 支撑的空壳 pedestrian/contact reward。
- 不再接受“本地路径 JSONL bridge”作为公开数据集接入。

## 5.5 引擎/性能确认 + 240p_33f 生成质量陷阱（实测 2026-06-27, RTX 5090 32GB）

在动 audit/reward 之前，先确认**引擎层**：Cosmos Predict2 2B 能在机器人参考数据上端到端跑 RL 管线，且 GPU 跑在健康 compute-bound 区间。**结论分两半：引擎/性能/利用率真实有效；但具体的 240p_33f 配置生成是垃圾、不能用于真训练（见 point 6）——别把“引擎能跑”读成“模型能学”。**

**新增配置**：`configs/experiment/cosmos_predict2/online_grpo_v2w_reference_fullparam_240p.yaml`
= `video_world_v2w`（per_sample 机器人首帧）+ 240p_33f + full-param + 8bit Adam + ppo4。补上了 trustworthy_curve sprint §3.5 标记缺失的“真 reference + full-param”配置。

**工具**（`vrl/scripts/perf/`）：`gpu_preflight`（定标 MFU 分母）+ `video_dit_mfu_probe`（隔离 DiT MFU）+ `gpm_sampler`（NVML GPM SM 级计数器，非 `nvidia-smi` 的“有 kernel 驻留”伪占用）。

### 实测结果

1. **MFU 分母（定标）**：这台 5090 真实 **bf16 dense 峰值 = 231.5 TFLOPS**（不是 vendor 419 fp8/sparse headline）；torch 已 build sm_120；最快 SDPA = flash@187（cuDNN@183，约 5% 内打平）。FA-3 在 Blackwell sm_120 不存在。

2. **隔离 DiT forward MFU**（合成权重，编译后稳态）：cosmos-predict2 2B，frames=8 代表长度下 **eager 76% → torch.compile 93% MFU（216 TFLOPS）**，1.24x 融合收益。attn% 随帧数上升（11%→47%@16f），视频侧唯一剩的无损杠杆是 FA-3（不存在）/ fused AdaLN。`torch_compile` 在模型配置里默认开 → 生产即拿 93%。

3. **端到端训练 SM 占用**（GPM，活跃计算窗口，已剔除 compile 空窗）：rollout denoise 与 training backward 签名高度一致。

   | 阶段 | 采样窗口 | sm_util | sm_occupancy | tensor 占空比 | DRAM 带宽 |
   |---|---|---|---|---|---|
   | rollout denoise（8 样本） | 8×约 9s | 约 76% | 约 11% | 约 31% | 约 14% |
   | training backward | 488s 连续 | **78.0%**（p90 78.9） | 约 12% | 约 31% | 约 17% |

   读法：**SM 约 77% 忙、DRAM 仅约 15%（远非带宽 bound）、非 launch-bound、非空转**。backward 那 488s 窗口 sm_util 纹丝不动（p50 78.1 / p90 78.9，零方差）= 真实持续 GEMM 计算，不是 compile autotuning 抖动。tensor 占空比约 30% 是 video DiT 的 norm/AdaLN/elementwise + grad-ckpt recompute 在 CUDA core 上与 tensor GEMM 串行的已知特征；其唯一无损解（FA-3）在 Blackwell sm_120 不存在 → 这已是该模型在这张卡上的实际性能天花板，与项目 MFU-bound north star 的结论一致。

   **吞吐 vs 效率要分开看**：单步 wall-clock 偏大（rollout 约 130s + backward 约 488s，ppo_epochs=4 → 32 个 full-param microbatch backward，约 15s/个，与 rollout 约 15s/样本同量级）。这个“慢”是**单卡 full-param 2B × ppo4 的规模属性**（要提速靠多卡 FSDP，见 SPRINT_multi_gpu_training），**不是利用率/效率问题**——效率（SM 占用）实测就是好的。首步还额外吃一次性 rollout+backward 的 torch.compile 编译。

4. **训推一致 + 机制健康（正确性）**：一个完整 epoch 跑通并写出 `checkpoint-final/checkpoint.pt`（11.8GB，`uses_lora=false` → full-param 2B 权重+8bit 优化器态）。`metrics.csv` 实测：
   - `first-step log-prob diff: mean=0.000000`、`logprob_abs_diff_mean=0.0028`、`approx_kl=1.1e-5` → full-param + 8bit Adam + compile **不污染 old_log_prob**，训推一致。
   - **`clip_fraction=0.52`** → ppo_epochs=4 让 GRPO trust-region clip 真正咬合（ppo=1 会恒 0，flux 验证证实）；`grad_norm=0.035`（full-param 非零梯度）、`advantage` 非塌缩、`group_size=8`、`reward_mean=-4.585`（Kling overall, in-distribution 负尺度）。
   - 机制层面坐实了 trustworthy_curve P0 的两个判据（full-param 240p_33f 单卡 fit + `clip_fraction>0` 机制活）。**但注意 point 6**：240p_33f 生成是垃圾，所以这个“fit”是个不能用的 shape——P0 的“显存可行”达到了，“可信曲线”反而被 shape 问题挡死，不是数据扩量就能解的。

5. **显存**：rollout 峰值约 17GB；training backward 峰值约 31GB（含一外部进程占 532MiB）。**首跑在编译后的 backward 图处 OOM**（差 32MiB，但有 1.19GB “reserved-but-unallocated” 碎片）——根因是 inductor 编译的 backward 图峰值 > eager，叠加碎片顶破 32GB。**修复 = `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，已实测证实**：带此 env 的复跑在 run1 OOM 的同一 backward 处稳定顶在 31.2GB 跑过去（不再 OOM）。

   运行处方（单卡 32GB full-param）：

   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vrl-train \
     --config experiment/cosmos_predict2/online_grpo_v2w_reference_fullparam_240p
   ```

6. **生成质量在 240p_33f 崩掉（2026-06-27 复核）**：肉眼检查生成的 `reward_artifacts/sample-*.mp4`——**只有 frame 0（被 reference 的 init_latents/cond_indicator 钳住）是连贯的真实机器人场景，frame 4 起全部退化成彩虹噪声**。逐帧 neighbor-diff 统计骗人（彩虹是大色块、空间平滑），必须肉眼看。reward=-4.58 / visual_quality≈-1.5 正是在反映这个垃圾输出。
   - **根因（已实测坐实，非假设）**：同一套管线在原生 **704p_93f 生成连贯**——单样本检查 `online_grpo_v2w_reference`（704p_93f）肉眼确认是连贯的机器人厨房操作视频（机械臂+物品，93 帧稳定一致），生成耗时约 733s/样本。所以管线/条件/VAE/denoise 都对，**问题纯粹是 240p_33f 对 Cosmos Predict2 2B V2W 严重 OOD**（原生约 704p_93f）。
   - **含义**：RL 在拿垃圾 rollout 做优化 → 这个配置**不能**用来出可信曲线。**单卡 32GB 的根本矛盾**：能 fit full-param 的 shape（240p_33f）生成是坏的；能正常生成的 shape（704p_93f）full-param 激活又 OOM。要同时“真生成 + full-param”必须上**多卡 FSDP**（见 SPRINT_multi_gpu_training），或退而用 LoRA + 704p_93f（梯度小，project_first_trustworthy_curve 已证推不动）。

7. **降 rollout 时间的杠杆（实测 2026-06-27）**：rollout 是绝对瓶颈（704p 单样本 733s × 8 = 约 98min，远超 backward 488s）。实测各档：

   | 档位 | 1 样本耗时 | token vs 704p | 720p ckpt 生成 |
   |---|---|---|---|
   | 704p_93f（原生） | 733s | 1x | 连贯 |
   | 480p_33f | 约 77s（约 10x 快） | 5.4x 少 | 垃圾 |
   | 240p_33f | 约 15s | 22x 少 | 垃圾 |

   - **关键耦合**：分辨率提速 = 真实的（480p 约 10x 快），但**降分辨率对 720p checkpoint 是 OOD → 生成垃圾**。要拿这个提速，必须下**官方 480P 2B V2W checkpoint**（832×480 原生，模型卡列为支持变体），不是改 config 就行——shape 和 checkpoint 是绑定的。
   - **不改分辨率、保 704p 连贯的提速杠杆**（可叠加）：去噪步数 35→20（`20_step_cfg_5_0`，约 1.75x）；关 CFG（`20_step_no_cfg`，约 2x，且对 RL log-prob 更干净）；720P+NATTEN 稀疏注意力变体（砍 attention，Blackwell 无 FA-3 时的替代）；fp8 rollout（`vrl/config/precision.py`，约 1.5-2x，但改 old_log_prob → 必须配 TIS-RS 修正）。
   - **组合**：480P checkpoint（约 10x）× 20步no-cfg（约 3.5x）≈ 35x → 733s/样本 → 约 20s/样本，单卡 RL 才真正可行。
   - **采坑记录**：CLI `sampling.video.height=480` 不生效（生成读扁平 `sampling.height`，见 `layout.py:104`）；改 shape 要走 config 的 defaults bucket，并肉眼验生成（统计的 neighbor-diff 对彩虹色块无效）。

**结论（修正后）**：**引擎/性能层就绪，但 240p_33f 这条具体配置不可用于真训练**。成立的：端到端管线打通（rollout→reward→backward→ckpt）、训推一致精确、SM 占用健康（约 77%、DiT 93% MFU、DRAM 非瓶颈，这些与输出好坏无关、测量有效）、expandable_segments 修 OOM。**不成立的**：把这当“cosmos 能学”——240p_33f 生成垃圾。下一步真正的门是**shape**：要么多卡跑 704p_93f full-param，要么找一个该模型生成不崩的最小 shape 再谈 RL。

## 6. 验证

- `python -m py_compile vrl/scripts/data/video_world.py vrl/scripts/eval/target_video_similarity_probe.py`
- `pytest tests/data/test_setup.py tests/data/test_video_world_manifests.py tests/data/test_artifact_manifest_validation.py tests/config/test_load_all_experiments.py::test_cosmos_target_v2w_production_validation_requires_target_clip -q`
- `pytest tests/data tests/rewards/functions tests/rewards/inference tests/config/test_load_all_experiments.py tests/config/test_schema.py -q`

## 7. 外部参考

- DROID: https://droid-dataset.github.io/
- LeRobot DROID sample: https://huggingface.co/datasets/lerobot/droid_100
- JRDB: https://jrdb.erc.monash.edu/
- RoboCasa: https://robocasa.ai/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- Cosmos Predict2: https://github.com/nvidia-cosmos/cosmos-predict2
- Cosmos Predict2 Video2World model card: https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World
