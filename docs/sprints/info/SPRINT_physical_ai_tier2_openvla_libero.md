# Physical-AI Tier 2 — OpenVLA-OFT × LIBERO-10 real eval (run record)

状态：**done（2026-06-25）**。性质：**第一次真实 model+simulator 验证 P1 contract**。
承接 `docs/sprints/planned/SPRINT_physical_ai_model_support.md` §Tier-2 / §P2，以及
`SPRINT_physical_ai_p0p1_landing.md`（P0/P1 contract）。

## 一句话

把真实的 OpenVLA-OFT 7B token/continuous-action 策略在真实 LIBERO-10 MuJoCo
闭环里跑通了，全程走 P1 的 `Env` / `ActionPolicy` / `ActionTrajectoryBatch`
契约，产出真实 success rate + 视频 + 每集 `ActionTrajectoryBatch`。这证明 P1
seam 不是纸面设计，能承载真实机器人策略闭环。

## 落地的长期资产（VRL repo）

- `vrl/rollouts/envs/libero.py` — `LiberoEnv`：把 LIBERO `OffScreenRenderEnv`
  包成 `Env` 协议（reset/step/render，chunk 开环执行，重依赖延迟导入）。
- `vrl/models/vla/openvla_oft.py` — `OpenVLAOFTPolicy`：把已加载的 OpenVLA-OFT
  模型包成 `ActionPolicy`，复用官方 `get_vla_action`（借 run-shape，不重写解码）。
- `vrl/scripts/eval/openvla_oft_libero_eval.py` — 真实 eval 入口，驱动闭环 →
  `ActionTrajectoryBatch` + success/video/episode JSON。
- `tests/rollouts/envs/test_vla_adapters.py` — 适配器协议一致性测试（CPU，无 GPU）。

## 关键发现（real findings）

1. **OpenVLA-OFT 是 continuous L1-regression，不是 token policy。** 官方 LIBERO
   OFT checkpoint 用确定性 L1 回归 action head，`predict_action` 无 token 分布 →
   **没有可复算 logprob**。所以 `can_replay_logprob=False`、`ActionChunk.log_prob=None`、
   `ActionTrajectoryBatch.is_trainable=False`。这更新了 sprint 里「OpenVLA-OFT =
   token policy，能记 token logprob」的旧假设：**OFT 路线天然只能 eval/SFT，要
   on-policy RL logprob 得回到 discrete-token base OpenVLA 或 PI0.5 flow logprob。**
   （契约如实表达：eval-only batch，trainer 必须拒绝，不伪造 logprob。）

2. **cosmos-rl 的 `Haozhan72/...` checkpoint 与 moojink repo chunk 不匹配**
   （16 vs 8）。按 sprint「只借 run-shape」原则改用 moojink 官方匹配 checkpoint
   `moojink/openvla-7b-oft-finetuned-libero-10`（chunk=8），保证解码正确。

## 结果

```text
模型: moojink/openvla-7b-oft-finetuned-libero-10  (7B, L1-regression OFT, chunk=8)
环境: LIBERO-10 (libero_10), MuJoCo EGL headless, max_steps=520
硬件: RTX 5090 (sm_120), torch 2.9.1+cu128

success_rate = 6/6 = 1.00   (3 tasks × 2 trials, 小样本)
  task0 (alphabet soup + tomato sauce → basket): 2/2   chunks 48,38
  task1 (cream cheese + butter):                 2/2   chunks 30,32
  task2 (turn on stove + moka pot):              2/2   chunks 30,29
moojink 官方全 500-trial libero_10 ≈ 94%；本次小样本 6/6 与之一致量级。
```

每集 `ActionTrajectoryBatch.actions` 形状 `[1, n_chunks, 8, 7]`（batch, chunks,
horizon=8, action_dim=7），`is_trainable=False`（eval-only，符合 OFT logprob 结论）。

## 复现 runbook（这次踩的坑全在这）

被迫在 Blackwell (5090/sm_120) 上跑 2 年前为 A100/torch2.2 写的 recipe，逐个解决：

1. **隔离 env**（不污染其它项目）：`conda create -n vla_eval python=3.10`。
2. **torch 必须 cu128**（sm_120 无 2.2.0 内核）：从已知可用的 `faithc` env 复制
   torch 2.9.1+cu128 stack（torch/torchvision/nvidia/torchgen + dist-info），比重下
   3GB 快；`pip install "numpy<2"`（robosuite/LIBERO 要 numpy 1.x）。
3. **openvla-oft stack**：`pip install git+https://github.com/moojink/transformers-openvla-oft.git`
   （双向 attn fork，transformers 4.40.1）；`pip install --no-deps -e openvla-oft`
   （拿 `prismatic`，跳过 torch==2.2 pin）；`pip install draccus==0.8.0 timm==0.9.10
   peft==0.11.1 sentencepiece==0.1.99 accelerate einops diffusers==0.30.3 absl-py
   matplotlib wandb ...`。
4. **LIBERO sim**：`pip install --no-deps -e LIBERO`；`pip install --no-deps
   robosuite==1.4.1`；`pip install "mujoco==2.3.7" numba==0.59.1 scipy
   "opencv-python==4.10.0.84" termcolor h5py bddl easydict cloudpickle "gym==0.25.2"`。
   跳过 `pynput`（teleop，native `evdev` 在 conda 工具链下编译失败：缺 `BUS_CEC` 等
   新内核常量）。
5. **真实 tensorflow-cpu==2.15.1**：`resize_image_for_policy` 在**推理时**用
   `tf.image.resize(lanczos3)` + JPEG round-trip 做训练分布匹配，不能 stub（会污染
   输入分布）。protobuf 自动降到 4.25.9，transformers/torch 仍正常。
6. **LIBERO 配置**：`~/.libero/config.yaml` 预写（否则首次 import 交互式 input → EOF）。

eval 脚本里固化的 compat shim（都带注释）：
- meta-path stub finder：stub 训练专用的 `tensorflow_datasets/tensorflow_graphics/
  dlimp` + teleop `pynput`（真 `tensorflow` 不 stub）。
- `torch.load` weights_only=False shim：torch≥2.6 默认翻转，LIBERO 的 numpy
  pickled init-states 解不开。
- LIBERO namespace 包：直接加 `--libero-root` 到 sys.path（PEP-660 editable 注册为空）。

## 复跑

```bash
MUJOCO_GL=egl PYTHONPATH=~/Desktop/vrl2/VRL ~/miniconda3/envs/vla_eval/bin/python \
  -m vrl.scripts.eval.openvla_oft_libero_eval \
  --checkpoint ~/Desktop/vla_models/openvla-7b-oft-finetuned-libero-10 \
  --task-suite libero_10 --tasks 3 --trials 2 --max-steps 520 \
  --out outputs/openvla_oft_libero/eval.json
```

## 还没做（仍 gate）

- on-policy RL（GRPO/DAPO）：OFT 无 logprob → 不接；要 RL 先上 discrete OpenVLA
  或 PI0.5 flow logprob（P3 probe）。
- 没把 VLA family 注册进 rollout family registry（还没有可训 runtime builder）。
- 没跑全 500-trial（10 task × 50），只做了小样本验证 seam 跑通。
