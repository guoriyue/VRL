<!-- Auto-generated 2026-07-20 by a 44-agent web-research workflow (6-angle sweep -> 24-source deep-read -> adversarial claim verification -> synthesis). arXiv IDs of 2026 preprints are as-cited by the agents; verify before citing externally. -->

# Video-Model RL: Methods, Rewards, Datasets — with a Robotics Focus

## 1. TL;DR landscape

The field has converged on **GRPO over flow-matching video diffusion** (Flow-GRPO / DanceGRPO lineage) as the dominant post-training recipe, with a smaller but growing **differentiable reward-backprop** branch (Diffusion-DRF, NewtonRewards) and a few **DPO / PPO** outliers (VisionReward-MPO, ReWorld). The real differentiation is in the **reward**, not the optimizer: three camps have emerged — (a) *verifiable* rewards (programmatic physics/action checks with no trained judge: PhysRVG, NewtonRewards, RLVR-World, RLIR), (b) *wrapped-frozen-foundation-model* rewards (V-JEPA2, Depth Anything 3, Any4D as physics/geometry referees: MIND-V, World-R1, VGGRPO), and (c) *purpose-trained* multi-dimensional judges (RoboAlign-R1, ReWorld, VisionReward). For the reader's specific robotics thesis, the strongest end-to-end evidence that video-model RL lifts *real* robotics-relevant features comes from **EVA** (real-robot OOD success 42%→60%) and **RLVR-World** (kills degenerate static rollouts 48.6%→9.9%) — both crucially validated against ground truth, not aesthetics. A recurring, load-bearing caveat: **physics understanding is uncorrelated with visual realism (Physics-IQ, r=-0.46, p=.247, not significant)**, so aesthetic rewards (PickScore/LAION) provably will not move the robotics axis.

---

## 2. Section A — Video-model RL methods

`New-to-reader?` uses the reader's stack (Flow-GRPO/DanceGRPO/GRPO-Guard/Flow-DPPO/DiffusionNFT/token-GRPO; Cosmos-2.5/Wan/SD3.5/Flux/SANA bases): **new** = novel mechanism, **adjacent** = variant of something owned, **already-have** = essentially owned.

### A.1 — GRPO-style (policy-gradient, scalar reward)

| Method | RL objective | Reward | Dataset | Base model | Code | New-to-reader? |
|---|---|---|---|---|---|---|
| **EVA** ([2603.17808](https://arxiv.org/abs/2603.17808)) | Flow-GRPO, clipped + KL (β=0.004), LoRA r32, backbone frozen; 8×A800 | **Trained** Inverse Dynamics Model → kinematic executability penalties (accel/jerk/limits) | RoboTwin (1,050 traj / 21 tasks SFT, 210 eval); real bimanual 250 teleop traj | Wan2.1-14B DiT (diffusion forcing, from "LVP" ckpt) | [github.com/RobbinW/EVA](https://github.com/RobbinW/EVA) | adjacent (algo+base owned; **reward is the trained-IDM path reader rejected**) |
| **TempFlow-GRPO** ([2508.04324](https://arxiv.org/abs/2508.04324)) | Flow-GRPO refinement: trajectory branching (branch=4) + noise-aware per-timestep weighting; "temporal"=denoising timestep, **image-only** | Frozen public: PickScore, HPSv2/v3, GenEval | Reuses Flow-GRPO prompt sets (GenEval + HP) | SD3.5-M, FLUX.1-dev, QwenImage | [github.com/Shredded-Pork/TempFlow-GRPO](https://github.com/Shredded-Pork/TempFlow-GRPO) | new (credit-assignment machinery) |
| **PhysRVG** ([2601.11087](https://arxiv.org/abs/2601.11087)) | GRPO wrapped in **MDcycle** (conditionally blends a Flow-Matching supervised loss when a group's avg trajectory-offset > threshold); V2V (obs 5 frames) | **Verifiable** "Physics-Grounded Metric": SAM2 tracker + Newtonian collision detection → collision-weighted trajectory-offset (needs paired GT future) | ~10M (Panda-70M+InternVid+WebVid-10M + proprietary); ~700 curated physics clips; PhysRVGBench (~50 held-out) | Wan2.2-TI2V-5B | [github.com/ant-research/PhysRVG](https://github.com/ant-research/PhysRVG) — *see caveat* | new (verifiable reward + MDcycle = validated diffusion-loss regularizer) |
| **VGGRPO** ([2603.26599](https://arxiv.org/abs/2603.26599)) | GRPO **in latent space** (skips VAE decode) | Frozen **Any4D/VGGT** 4D geometry model → camera-smoothness + geometry-reprojection-consistency | DL3DV, RealEstate10K, MiraData (190 static + 200 dynamic eval captions) | Wan2.1-1B, Wan2.2-5B | Project page only, **no repo** | new (4D-geometry reward) — *metrics medium-confidence* |
| **TaRoS** ([2511.19356](https://arxiv.org/abs/2511.19356)) | DanceGRPO; **reward-aggregation layer** only (Hoyer-sparsity + mean-threshold adaptive component weights, anti-Goodhart) | Frozen: VideoAlign, Qwen2.5-VL-72B/7B, LAION (eval) | DanceGRPO 50k prompts; iStock 1.4k eval | Wan2.1-T2V-1.3B/14B, HunyuanVideo | **no code** | adjacent (upgrades static composite-reward weights) |
| **Flash-GRPO** ([2605.15980](https://arxiv.org/abs/2605.15980)) | **Single denoise-step** Flow-GRPO + iso-temporal grouping + temporal grad-rectification; ~6× cheaper | Frozen: HPSv3 (top-30% frames), VideoAlign motion | DanceGRPO prompts (300 eval) | Wan2.1-T2V-1.3B/14B | [github.com/Shredded-Pork/Flash-GRPO](https://github.com/Shredded-Pork/Flash-GRPO) | new (single-step + iso-temporal grouping) |
| **TAGRPO** ([2601.05729](https://arxiv.org/abs/2601.05729)) | GRPO + contrastive **trajectory-alignment** on intermediate latents from shared initial noise + FIFO rollout bank; **I2V** | Frozen public: HPSv3 + Q-Save (VQ/DQ/IA) | Internal ~10K image-text pairs; TAGRPO-Bench 200 | Wan2.2, HunyuanVideo-1.5 (I2V) | promised, no live URL | new (contrastive latent trajectory align for I2V) |
| **RLVR-World** ([2505.13934](https://arxiv.org/abs/2505.13934)) | GRPO over AR discrete-token sequence (paper labels it "GRPO"; "token-level" is a fair characterization, not verbatim); no value fn | **Verifiable** = eval metric as reward: video R = −Σ(L1+LPIPS) vs GT future | RT-1 (87,212 robot traj) + text-game/web sets | iVideoGPT (AR VQ-token) — **not diffusion** | [github.com/thuml/RLVR-World](https://github.com/thuml/RLVR-World) | adjacent (owns token-GRPO; reward flavor new) |
| **RLIR** ([2509.23958](https://arxiv.org/abs/2509.23958)) | GRPO (both AR & diffusion-forcing WMs), terminal-only reward | **Trained** VPT-IDM inverse-action reward (recovered action == commanded action) | Minecraft VPT ~2,000h; ~1,000 post-train samples | MineWorld (AR 300M-1.2B), NFD (diffusion 310/774M) | **no code** | new reward (IDM action-recovery) — *Minecraft-only* |
| **MIND-V** ([2512.06628](https://arxiv.org/abs/2512.06628)) | GRPO on **flow_grpo** codebase; hierarchical planner + staged test-time rollouts | **PFC reward = frozen V-JEPA2** latent-prediction cosine + Qwen-VL 1-5 aesthetic | Bridge v2 (108 eval) | CogVideoX-5B | [github.com/Richard-Zhang-AI/MIND-V](https://github.com/Richard-Zhang-AI/MIND-V) | new reward (V-JEPA2 PFC); **RL adds only +1.2pp; peak VRAM ~70GB @K=3 (~122GB @K=5)** |
| **Cosmos-Predict2.5 / DDRL** ([2511.00062](https://arxiv.org/abs/2511.00062)) | **DDRL** (Data-regularized RL for Diffusion Models) = GRPO + diffusion-loss regularizer **replacing KL** (kl_beta=0, data_beta=0.01, clip_ratio=1e-4) | VideoAlign *(medium/low confidence — from secondary sources, not paper body)* | 200M clips; robot variants on Libero/RoboCasa | Cosmos-Predict2.5 2B/14B | [cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) + [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) | adjacent (base owned; **DDRL = at-scale version of reader's DESIGN-ONLY diffusion-loss regularizer**) |
| **RoboAlign-R1** ([2605.03821](https://arxiv.org/abs/2605.03821)) | GRPO + KL to frozen ref; distilled reward in-loop; **SWR** inference drift fix | **Trained** distilled 6-dim robotics judge (Qwen3-VL-8B teacher → 98M student ~50fps) | RobotWorldBench 10K (RT-1/Bridge V2/CALVIN/LIBERO) | Custom 138M AR FSQ world model | repo announced, **empty** | new (SWR trick portable) — *trained judge conflicts with reader policy* |
| **World-R1** ([2604.24764](https://arxiv.org/abs/2604.24764)) | **Flow-GRPO** (48 groups × 8); periodic decoupled schedule (dynamic-only regularizer every 100 steps); 48-96 H200 | Frozen **3D stack**: Qwen3-VL meta-view + **Depth Anything 3 → 3DGS** recon (1-LPIPS) + camera-trajectory alignment; HPSv3 as separate general reward | ~3,000 Gemini text prompts (~500 dynamic) | Wan2.1-T2V-1.3B/14B | [github.com/microsoft/World-R1](https://github.com/microsoft/World-R1) (MIT) | new reward (3D/geometry + camera controllability) |
| **Reward as An Agent / DynDiff-GRPO** ([2606.19990](https://arxiv.org/abs/2606.19990)) | GRPO variant: **spatial noise reallocation** toward dynamic regions (σ⊙M_t); buffer-anchored (no frozen ref) | Agentic VLM judge (Qwen3.5, planning/vote/reflect) — **proprietary, not released** | AgiBotWorld-Beta, PAI-Bench robotics, multi-embodiment set | Cosmos-Predict2.5-2B, Kairos-3.0-Robot-4B | **no code** | new lever (dynamic-aware SDE noise) — *gains modest ~1-3pt; medium confidence* |

> **PhysRVG release caveat:** the paper states code and checkpoints "will be released publicly soon"; a HF weights URL (`HappyP4nda/PhysRVG`) is referenced. Treat release status as **announced, not confirmed-live**.

### A.2 — Reward-backprop (differentiable, NOT policy-gradient)

| Method | RL objective | Reward | Dataset | Base model | Code | New-to-reader? |
|---|---|---|---|---|---|---|
| **Diffusion-DRF** ([2601.04153](https://arxiv.org/abs/2601.04153)) | Truncated backprop through last-K denoise + VAE decoder (VADER/DRaFT family) | **Frozen VLM as differentiable VQA critic** (Qwen2.5-VL-7B): TA/Physical-Fidelity/VQ next-token logprob; reference-guided. VBench-2.0 Physics 48.40→56.85 (+8.44) | 5K captioned from OpenVid-1M; 32×A100 | Wan2.1-1.3B-T2V | Project page, **no repo** | new (whole optimization family; VLM-as-differentiable-reward) |
| **NewtonRewards** ([2512.00425](https://arxiv.org/abs/2512.00425)) | Differentiable physics loss backprop through denoiser (DRaFT/ReFL-style) | **Verifiable, frozen**: RAFT 2nd-order flow residual (Newton's 2nd law) + V-JEPA2 mass-conservation | NewtonBench-60K (Kubric+PyBullet+Blender; 5 primitives); 8×H100 | Open-Sora v1.2 | [github.com/cvlab-stonybrook/NewtonRewards](https://github.com/cvlab-stonybrook/NewtonRewards) | adjacent reward (RAFT owned); new optimization axis |

> **Trainer caveat (reader-specific):** reward-backprop mutates the forward path — this conflicts with the reader's verl-style "don't corrupt old_log_prob" rule for GRPO. These are a *separate trainer*, not drop-in rewards.

### A.3 — Preference-DPO

| Method | RL objective | Reward | Dataset | Base model | Code | New-to-reader? |
|---|---|---|---|---|---|---|
| **VisionReward** ([2412.21059](https://arxiv.org/abs/2412.21059)) | **MPO** = offline Diffusion-DPO with **Pareto-dominant** pair selection (multi-objective) | **Trained** VisionReward (CogVLM2-Video-19B) — 64-item checklist incl. named Physics/Stability/Dynamics/Shape-preservation dims + logistic head | VisionRewardDB-Video 33K×64 (~2M Q); ~1,795 video pairs for weights | Reward: CogVLM2-Video-19B; Gen: SDXL / CogVideoX-2b | [github.com/zai-org/VisionReward](https://github.com/zai-org/VisionReward) + HF ckpts | new (reward model + MPO); reader has DPPO but not Pareto-pair DPO |

### A.4 — World-model / PPO

| Method | RL objective | Reward | Dataset | Base model | Code | New-to-reader? |
|---|---|---|---|---|---|---|
| **ReWorld** ([2601.12428](https://arxiv.org/abs/2601.12428)) | **PPO-style** with **CFM-likelihood proxy** (uses conditional-flow-matching loss as logprob stand-in; O(d) vs O(d²·T)) | **Trained** hierarchical 4-head reward (InternVideo2-1B): physical realism / embodiment / task completion / VQ (~85.3% acc, 0.901 AUC) | ~235K preference pairs from RH20T (GPT-4o auto-labeled); Bridge V2 SFT | Cosmos-2B/14B named; **only 2B actually RL-aligned in experiments** | **no code** | new (CFM-proxy PPO + trained embodied reward) |

> **ReWorld correction (verified):** the ~−24pt catastrophic ablation drop is from *naively plugging L_CFM into a standard PPO ratio (wrong sign)* — NOT from removing the CFM proxy. Removing the CFM-likelihood surrogate entirely (replacing with a reward-weighted L2 loss) costs only −6.8 (61.9→55.1). Experiments post-train Cosmos-2B; 14B is named as the policy family but reports no RL-aligned results.

### A.5 — Non-RL but robotics-adjacent (for completeness, not RL)

- **PhysisForcing** ([2606.28128](https://arxiv.org/abs/2606.28128)) — despite "Reinforced" in the title, this is a **supervised auxiliary-loss regularizer** (trajectory-alignment + relational-contact loss from a frozen video encoder), NOT RL. Closed-loop manipulation success **16%→24%**. Base: Wan2.2-I2V-A14B, Cosmos3-Nano. Code: [github.com/DAGroup-PKU/PhysisForcing](https://github.com/DAGroup-PKU/PhysisForcing) (inference+weights out; training "coming soon"). *Confidence medium.* Transferable: its frozen-encoder relational-contact target could become a new GRPO reward.

---

## 3. Section B — Reward models & functions to wrap

| Name | What it scores | How obtained | Open-weight? | URL |
|---|---|---|---|---|
| **VisionReward (video)** | 64-item checklist incl. **Physics compliance, Stability, Dynamic/movement, Shape preservation** + scalar | Trained: CogVLM2-Video-19B SFT + logistic-regression weights | **Yes** (HF + ModelScope) | [github.com/zai-org/VisionReward](https://github.com/zai-org/VisionReward) |
| **VideoPhy-2-AutoEval** | Semantic Adherence + Physical Commonsense (1-5) + physical-rule class | Fine-tune of VideoCon-Physics (mPLUG-Owl-Video backbone; "7B" inferred not quoted) on ~50K human annots | **Yes** (`videophysics/videophy_2_auto`, MIT) | [github.com/Hritikbansal/videophy](https://github.com/Hritikbansal/videophy) |
| **V-JEPA2 PFC-style reward** | Physical-plausibility as latent-future-prediction cosine sim (world-model self-consistency) | Wrapped frozen V-JEPA2 (no training) | **Yes** (V-JEPA2 public) | via MIND-V [github.com/Richard-Zhang-AI/MIND-V](https://github.com/Richard-Zhang-AI/MIND-V) |
| **Newtonian kinematic residual** | Constant-image-plane-acceleration (Newton's 2nd law) via RAFT optical flow, 2nd-order finite diff | Rule-based on frozen RAFT | **Yes** (RAFT public; RAFT already in reader stack) | [github.com/cvlab-stonybrook/NewtonRewards](https://github.com/cvlab-stonybrook/NewtonRewards) |
| **V-JEPA2 mass-conservation** | Appearance/mass-embedding stability vs reference | Rule-based on frozen V-JEPA2 | **Yes** | same |
| **SAM2 + Newtonian collision offset** | Object trajectory-offset vs paired GT future, collision-reweighted | Rule-based; SAM2 tracker only | **Yes** (SAM2 public) | [github.com/ant-research/PhysRVG](https://github.com/ant-research/PhysRVG) |
| **Depth Anything 3 → 3DGS recon** | 3D/geometric self-consistency (1-LPIPS re-render); camera-trajectory alignment | Wrapped frozen Depth Anything 3 + 3DGS fit | **Yes** | [github.com/microsoft/World-R1](https://github.com/microsoft/World-R1) |
| **Any4D / VGGT 4D geometry** | Camera-motion-smoothness + geometry-reprojection consistency (dynamic scenes) | Wrapped frozen 4D foundation model | Any4D/VGGT public; **VGGRPO wrapper not released** | [project page](https://zhaochongan.github.io/projects/VGGRPO) |
| **HPSv3** | Human-preference image/frame reward | Public checkpoint | **Yes** (MizzenAI/HPSv3) | github.com/MizzenAI/HPSv3 |
| **RoboAlign 6-dim / ReWorld 4-dim judges** | Instruction-following, manipulation success, contact realism, physics adherence, temporal consistency | **Trained** (Qwen3-VL-8B→98M student / InternVideo2-1B) | **No** (unreleased) — *conflicts with reader's "wrap-only" rule* | [2605.03821](https://arxiv.org/abs/2605.03821) / [2601.12428](https://arxiv.org/abs/2601.12428) |
| **Inverse Dynamics Model (action-recovery)** | "Did the video execute the commanded action?" | **Trained** (VPT-IDM for Minecraft; EVA's IDM for RoboTwin) | VPT-IDM public (Minecraft only); robot IDMs must be trained | [2509.23958](https://arxiv.org/abs/2509.23958) / [2603.17808](https://arxiv.org/abs/2603.17808) |

**Reader-policy filter:** VisionReward, VideoPhy-2-AutoEval, V-JEPA2 PFC, RAFT/V-JEPA2 Newtonian rewards, SAM2 collision offset, Depth-Anything-3 3DGS all satisfy "wrap ready public models." RoboAlign/ReWorld judges and robot-domain IDMs require training a judge — the exact path the reader's memory records rejecting.

---

## 4. Section C — Datasets

### Prompt / preference sets
- **DanceGRPO 50k prompt set** — reused by TaRoS, Flash-GRPO, TempFlow-GRPO (via Flow-GRPO). ([2511.19356](https://arxiv.org/abs/2511.19356))
- **VisionRewardDB-Video** — 33K samples × 64 dims (~2M annotation questions); + 1,795 video preference pairs. Released. ([2412.21059](https://arxiv.org/abs/2412.21059))
- **ReWorld ~235K preference pairs** — from RH20T, GPT-4o auto-labeled, dimension-isolated. Not released. ([2601.12428](https://arxiv.org/abs/2601.12428))
- **World-R1** — ~3,000 Gemini-synthesized text prompts (~500 dynamic), released in repo. ([2604.24764](https://arxiv.org/abs/2604.24764))
- **OpenVid-1M subset** — 5K captioned videos (Diffusion-DRF training). ([2601.04153](https://arxiv.org/abs/2601.04153))

### Physics datasets / benchmarks (eval — keep held-out)
- **VideoPhy-2** — 197 actions, 3,940 prompts, 6,800 human-annotated videos (~102K annots); Hard subset 60 actions/1,200 prompts. **MIT**, on HF (`videophysics/videophy2_train/_test`). Best model **~22% joint on Hard** *at the paper's publication* — later leaderboard entries (e.g. Wan2.2-27B-A14B ~47.7%) score higher, so 22% is the paper figure, not the current live ceiling. ([2503.06800](https://arxiv.org/abs/2503.06800))
- **Physics-IQ** — 396 real 4K videos, 66 scenarios, 5 physics families; prediction-vs-GT-future protocol. **Apache-2.0** code / CC-BY-4.0 data. **Best Physics-IQ = 24.1% (VideoPoet multiframe)** vs the 100% real-physical-variance ceiling. Physics-vs-visual-realism correlation **r=-0.46, p=.247 (not significant)**. Reader can benchmark their Cosmos/Wan checkpoints (now on the post-paper leaderboard). ([2501.09038](https://arxiv.org/abs/2501.09038))
- **VBench-2.0** — 5 categories × 18 dims (Physics, Commonsense, Controllability, Human Fidelity, Creativity); `pip install vbench`. Evaluation only. ([arXiv:2503.21755](https://arxiv.org/abs/2503.21755), [github.com/Vchitect/VBench](https://github.com/Vchitect/VBench))
- **NewtonBench-60K** — synthetic (Kubric+PyBullet+Blender), 50K train + 10K held-out, 5 Newtonian primitives, RGB+masks+depth+metadata. To be released. ([2512.00425](https://arxiv.org/abs/2512.00425))
- **PhysRVGBench** — ~50 held-out physics videos (from a ~700-video curated collection), IoU + trajectory-offset metrics. ([2601.11087](https://arxiv.org/abs/2601.11087))

### Robotics-video datasets
- **RoboTwin** — 1,050 sim trajectories / 21 tasks (EVA SFT) + real bimanual 250 teleop traj. ([2603.17808](https://arxiv.org/abs/2603.17808))
- **RT-1 (Google Robot)** — 87,212 trajectories @256×320 (RLVR-World; also in RoboAlign RobotWorldBench). ([2505.13934](https://arxiv.org/abs/2505.13934))
- **RH20T** — source for ReWorld's 235K pairs. ([2601.12428](https://arxiv.org/abs/2601.12428))
- **Bridge V2** — MIND-V (RoboMaster protocol) and ReWorld SFT. ([2512.06628](https://arxiv.org/abs/2512.06628))
- **RobotWorldBench** — 10K video-instruction pairs from RT-1/BridgeData V2/CALVIN/LIBERO. ([2605.03821](https://arxiv.org/abs/2605.03821))
- **AgiBotWorld-Beta / PAI-Bench robotics** — multi-embodiment eval (Reward-as-Agent). ([2606.19990](https://arxiv.org/abs/2606.19990))
- **VPT (Minecraft)** — ~2,000h gameplay w/ action labels, for IDM training (RLIR). ([2509.23958](https://arxiv.org/abs/2509.23958))
- **Libero / RoboCasa** — Cosmos-Predict2.5 robot action-conditioned post-training variants. ([2511.00062](https://arxiv.org/abs/2511.00062))

> Reader already has **VideoPhy** and **DROID** on disk — VideoPhy-2 is the successor (add its Hard prompt set + AutoEval); DROID is the natural substrate if they ever build a robot-domain IDM.

---

## 5. Section D — ROBOTICS FOCUS: concrete (method → reward → dataset) triples

Organized by the reader's three feature axes, plus a measurement axis. Each triple is judged for fit with the single-5090/32GB, wrap-only-rewards, GRPO-diffusion stack. This is the section the reader asked to be richest, so each entry carries the concrete numbers that make the case.

### D.1 — Physical plausibility (world dynamics obey physics)

- **PhysRVG triple** → GRPO+MDcycle → **SAM2-tracked collision-weighted Newtonian trajectory-offset (verifiable, needs paired GT future)** → PhysRVGBench/curated rigid-body clips. Base **Wan2.2-TI2V-5B**. **Best single-5090 fit in the physics camp**: base is owned, reward wraps public SAM2 (no trained judge → satisfies reader policy), 5B full-param fits with the reader's 8-bit-Adam enabler or LoRA. Evidence it works: **IoU 0.64 vs next-best Magi-1 0.27**; **Trajectory Offset 15.03 vs Kling2.5 103.22 (~7× reduction)**; VideoPhy-2 SA 0.76 / PC 0.64 (both best). Ablation shows the value chain: baseline IoU 0.15 → +full-FT 0.38 → +FT+RL 0.61 → **+FT+MDcycle 0.64** (MDcycle is the diffusion-loss regularizer that stabilizes GRPO's small-batch convergence). **Ported lesson: raw trajectory-matching reward-hacks toward static/linear motion — you MUST up-weight collision frames** (best collision weights (w, w_adj, w_col) = (1, 2, 3), threshold=8; Fig.6 ablation). Two genuinely new pieces to port: (1) the verifiable physics reward category (closer to RLVR than the reader's judge-wrapping set); (2) MDcycle as the concrete, adaptive instantiation of the reader's DESIGN-ONLY "diffusion-loss regularizer" — landed *with* a gating threshold, not as a bare knob. ([2601.11087](https://arxiv.org/abs/2601.11087))

- **NewtonRewards triple** → reward-backprop → **RAFT 2nd-order optical-flow residual (Newton's 2nd law: ‖φ_{t+1}−2φ_t+φ_{t−1}‖²) + V-JEPA2 mass-conservation** → NewtonBench-60K. RAFT already in reader stack; but it's a *different trainer* (mutates forward path, conflicts with old_log_prob rule) — **adopt the reward as a GRPO scalar**, not the backprop. Measured: ID benchmark avg **+9.75%** (accel RMSE +8.46%, IoU +14.78%), OOD avg **+8.60%**. Ablation lesson mirrors PhysRVG: kinematic-only reward drives all velocities to 0 so **objects vanish** — the mass term is required (dropping mass = reward hacking; dropping kinematic = only +0.25%). Replication needs 8×H100, but the frozen-RAFT+V-JEPA2 reward itself is single-GPU friendly. ([2512.00425](https://arxiv.org/abs/2512.00425))

- **MIND-V PFC reward** → wrap **frozen V-JEPA2 latent-future-prediction cosine** as a GRPO scalar → Bridge v2. Fits reader policy perfectly (wraps a public world model); a physics-reward flavor they lack (they have VideoCon-Physics/PhyMotion but no world-model-latent-prediction reward). Honest caveat from the ablation: in MIND-V the RL step itself only lifted sub-task success **60.1%→61.3% (+1.2pp)** and PFC score 0.419→0.445 (+6.2%); the bulk of the manipulation gain came from hierarchical planning + staged test-time rollouts (w/o staged rollouts = 32.7%), *not* the reward. So **take the reward, skip the 70GB pipeline** (peak VRAM ~70GB @K=3, ~122GB @K=5 — does not fit the reader's card as a whole system). ([2512.06628](https://arxiv.org/abs/2512.06628))

- **Diffusion-DRF physics reward (reference-guided VLM VQA)** → reward-backprop → **frozen Qwen2.5-VL-7B, Physical-Fidelity axis, contrastive against a caption-matched real clip** → OpenVid-1M subset. VBench-2.0 **Physics 48.40→56.85 (+8.44)**, Motion Rationality +2.87, Dynamic Attribute +5.86; and it is the *only* method in the paper that stays above base across the whole training horizon (PickScore/VideoAlign/Vanilla-DRF all peak then collapse below base = reward hacking). Same trainer caveat as NewtonRewards (differentiable, not GRPO). The transferable idea for the reader: their VLM judges (Cosmos-Reason, VideoScore2, VideoCon-Physics) are currently consumed as scalars — this shows the *reference-anchored contrastive* framing squeezes more physics signal from the same frozen VLM. ([2601.04153](https://arxiv.org/abs/2601.04153))

### D.2 — Action-following / controllability

- **RLIR / EVA triple** → GRPO → **Inverse Dynamics Model action-recovery reward** → VPT (Minecraft, RLIR) or RoboTwin (EVA). This is the strongest published evidence an IDM-recovery reward lifts controllability *and* co-improves visual fidelity:
  - **EVA** (real robot): kinematic plausibility **70.5%→91.4%**, "perfect execution" 68.1%→83.8%, RoboTwin-sim avg 46.2%→52.6%, and — the load-bearing end-to-end result — **real-robot seen tasks 52%→64%, OOD 42%→60%**. Its IDM itself scores 89.52% avg execution success on ground-truth demos. (Note: the 42→60 / 52→64 figures are the real bimanual-robot Table 3 results; RoboTwin is the paired simulation benchmark.)
  - **RLIR** (Minecraft world models): MineWorld-700M action-following **F1 0.70→0.81** (+15.7%), FVD 231→207; NFD-774M (diffusion) F1 0.77→0.83, closing much of the gap to the IDM-on-real-video upper bound F1 0.87 — and visual quality improves even though *only* the inverse-action reward is optimized.
  - **Friction (explicit):** both require *training* an IDM. VPT-IDM is public but Minecraft-only; there is **no ready public robot-domain IDM**, so applying this to Cosmos/Wan/DROID means training one — exactly the zero-training path the reader's memory records building-then-deleting. This is the concrete evidence to weigh that decision against, not a drop-in given the reader's wrap-only rule. ([2509.23958](https://arxiv.org/abs/2509.23958), [2603.17808](https://arxiv.org/abs/2603.17808))

- **Cosmos-Predict2.5 robot line (VLA+RL) / DDRL** → DDRL (GRPO + diffusion-loss regularizer, kl_beta=0/data_beta=0.01/clip_ratio=1e-4) → action-conditioned Libero/RoboCasa variants. Base already owned; this is the closest published VLA-adjacent RL recipe on the reader's exact backbone. Caveat: the concrete knobs live in the `cosmos-rl` repo (`examples/ddrl.md`) and the dedicated DDRL paper ([2512.04332](https://arxiv.org/abs/2512.04332)), *not* in the body of the Cosmos-Predict2.5 paper (2511.00062), which describes RL post-training only at a high level. ([2511.00062](https://arxiv.org/abs/2511.00062))

- **World-R1 camera-trajectory reward** → Flow-GRPO → **camera-trajectory-alignment reward (controllability analog)** + Depth-Anything-3 3DGS → text prompts. Makes generated motion obey a specified camera trajectory — an action-following proxy for viewpoint control. Caveat: the camera-trajectory reward needs camera-conditioned/controllable generation the reader's T2V stack may not yet expose. ([2604.24764](https://arxiv.org/abs/2604.24764))

- **DynDiff-GRPO controllability lever** → **spatial SDE-noise reallocation toward motion-salient regions** (Reward-as-Agent) → any of the above. The reader's Flow-GRPO applies SDE noise uniformly; their existing **RAFT motion guard could supply the dynamic-saliency mask** (σ_noise = σ ⊙ M_t) to implement this, expanding action-space exploration while keeping static scene regions stable. Measured gains are modest (Cosmos-2.5-2B PAI-Bench overall 79.42→80.75; Kairos-4B 80.35→81.88, +3.08 on the domain/physical-dynamics score) — *medium confidence, no code*. ([2606.19990](https://arxiv.org/abs/2606.19990))

### D.3 — World-model fidelity / dynamics consistency (video-prediction-as-reward)

- **RLVR-World triple** → token-GRPO → **L1+LPIPS vs held-out GT future (verifiable reconstruction)** → RT-1. The cleanest verifiable world-model-fidelity result: directly kills degenerate static rollouts (**repetition 48.6%→9.9%**, +79.6% rel) and improves LPIPS 14.8→13.4, MSE 0.659→0.486, PSNR 23.1→24.1, SSIM 80.9→82.4; also improves real2sim policy-eval faithfulness vs a handcrafted simulator. Reader has the data pattern (DROID) to add an **L1+LPIPS-vs-GT world-model-fidelity reward** to their diffusion Flow-GRPO loop — new to their catalog, satisfies verifiable-not-judge. Caveat: RLVR-World's backbone is an AR discrete-token model (iVideoGPT) that is an action-conditioned next-state predictor; the reader's diffusion rollouts aren't action-conditioned the same way, so the mapping isn't 1:1 and action-following improves only implicitly. ([2505.13934](https://arxiv.org/abs/2505.13934))

- **World-R1 / VGGRPO geometry triple** → Flow-GRPO (latent, VGGRPO) → **3DGS reconstruction-consistency (Depth Anything 3) / 4D geometry-reprojection (Any4D)** → DL3DV/RealEstate10K/MiraData. A stronger, model-derived proxy for "is the generated world geometrically self-consistent" than the reader's 2D DINOv2/RAFT proxies. World-R1 is open (MIT) and quantified: 3D reconstruction consistency **PSNR +10.23 dB / +7.91 dB** over the two Wan baselines (World-R1-Small 27.63/0.858/0.201 PSNR/SSIM/LPIPS vs Wan2.1-1.3B 17.40/0.550/0.467), while VBench general quality is preserved/improved. VGGRPO's latent-space GRPO (skip VAE decode) is a useful single-GPU throughput idea, but its metrics are *medium-confidence and there is no code*. ([2604.24764](https://arxiv.org/abs/2604.24764), [2603.26599](https://arxiv.org/abs/2603.26599))

- **ReWorld triple** → PPO + CFM-likelihood proxy → **trained InternVideo2 4-head embodied reward** → RH20T. On the reader's exact Cosmos-2B base (note: only 2B is RL-aligned in the experiments; 14B is named but unrun). Results: overall 61.9 vs 54.4 SFT, FVD 190 vs 240, physical realism 5.9 vs 5.1; reward model 85.3% acc / 0.901 AUC. The **CFM-likelihood proxy** (use CFM denoising loss as logprob stand-in, O(d) vs O(d²·T)) is an interesting cheaper alternative to exact SDE-logprob worth an A/B — but note two things: the reward is a *trained* judge (policy conflict), and the paper's own ablation shows the proxy is safe only in ReWorld's FPO formulation (naively dropping L_CFM into a standard PPO ratio with the wrong sign collapses performance by −24pt; merely removing the surrogate costs only −6.8). ([2601.12428](https://arxiv.org/abs/2601.12428))

### D.4 — Measure it (held-out eval, not training)

Wrap **VideoPhy-2-AutoEval** as a reward *and* eval; report GRPO runs on **Physics-IQ (Verified)** and **VBench-2.0 Physics/Commonsense/Controllability**. The reader's own Cosmos-Predict2/2.5 and Wan2.1/2.2 are already on the post-paper Physics-IQ leaderboard, so they can benchmark GRPO-finetuned checkpoints directly. Physics-IQ's **r=-0.46 (physics vs visual realism, p=.247, not significant)** is the external justification that physics-targeted rewards are *necessary* — aesthetic rewards won't move the axis. ([2503.06800](https://arxiv.org/abs/2503.06800), [2501.09038](https://arxiv.org/abs/2501.09038), [2503.21755](https://arxiv.org/abs/2503.21755))

---

## 6. Section E — Gap analysis vs the reader's stack + top things to try next

**What the reader already owns and needs nothing new for:** the GRPO-on-flow-matching substrate (EVA, PhysRVG, World-R1, MIND-V, Reward-as-Agent all build on exactly Flow-GRPO/DanceGRPO), Wan/Cosmos bases, and VideoAlign/VideoCon-Physics/PhyMotion/RAFT/DINOv2 rewards. Almost every robotics paper here is *the reader's own recipe with a different reward*.

**The genuine gaps, ranked — each mapped to what the reader would add or change:**

1. **No verifiable (RLVR-style) reward.** Every reader reward is a learned judge or a self-supervised guard; none verifies against ground truth. **→ Add a verifiable physics reward** as a new block in `rewards/`. Best fit: **PhysRVG's SAM2 + collision-weighted Newtonian trajectory-offset** (wraps public SAM2, no trained judge, Wan2.2 base owned) or **NewtonRewards' RAFT 2nd-order residual + V-JEPA2 mass** (RAFT already owned) consumed *as a GRPO scalar* (not the reward-backprop trainer). Why: it's the one reward category proven to survive reward-hacking (Physics-IQ shows aesthetic rewards can't touch physics) and it slots cleanly into the wrap-only policy. *Watch the reward-hacking traps both papers document (static-motion / vanishing-object collapse) — collision-frame up-weighting and the mass term are the mitigations.* ([2601.11087](https://arxiv.org/abs/2601.11087), [2512.00425](https://arxiv.org/abs/2512.00425))

2. **No world-model-latent-prediction reward.** **→ Wrap frozen V-JEPA2 (MIND-V PFC)** as a GRPO scalar alongside VideoCon-Physics/PhyMotion. Cheapest new-flavor add, fits policy, directly targets world-model fidelity; take only the reward, not the 70GB pipeline. Set expectations from MIND-V's own ablation: the RL step alone contributed +1.2pp there — this is a fidelity-signal add, not a large-gain lever on its own. ([2512.06628](https://arxiv.org/abs/2512.06628))

3. **Diffusion-loss regularizer is DESIGN-ONLY in the reader's memory.** **→ Port DDRL** (cosmos-rl): replace KL with the diffusion-loss regularizer, concrete knobs `kl_beta=0, data_beta=0.01, clip_ratio=1e-4`, on the reader's own Cosmos-2.5 base. It's the at-scale-validated instantiation of exactly the idea the reader flagged — and PhysRVG's MDcycle is a second, independent confirmation that a conditionally-gated flow-matching loss stabilizes GRPO. Carry the reader's own caveat: `clip_ratio=1e-4` matches the tight-clip value their Flux/Cosmos sprints found interacts with compile noise (clip_fraction spikes). Attribution note: knobs are in `cosmos-rl/examples/ddrl.md` and the DDRL paper ([2512.04332](https://arxiv.org/abs/2512.04332)), not the Cosmos-Predict2.5 paper body. ([2511.00062](https://arxiv.org/abs/2511.00062))

4. **Rollout is the reader's profiled bottleneck; full-trajectory video GRPO barely fits 32GB.** **→ Adopt a sample-efficiency upgrade to the existing SDE-logprob path**: **Flash-GRPO** (single-step, ~6× cheaper — but the ablation shows naive single-step is a no-op at 4.64 vs vanilla 4.67; the gain only appears with iso-temporal grouping 5.31 → +rectification 5.42, so those two knobs are load-bearing) or **TempFlow-GRPO** (trajectory branching + noise-aware per-timestep weighting, 2–4.5× fewer steps to convergence). Both drop onto the reader's SDE-logprob collector/loss; both are single-GPU-motivated. Trade-off: TempFlow's branch=4×6 rollouts multiply *per-group* rollout cost, trading throughput for sample efficiency — relevant since rollout is already the bottleneck. ([2605.15980](https://arxiv.org/abs/2605.15980), [2508.04324](https://arxiv.org/abs/2508.04324))

5. **No held-out physics eval, and (if compositing many rewards) no saturation-aware weighting.** **→ Add VideoPhy-2-AutoEval + Physics-IQ + VBench-2.0** as held-out eval to prove RLHF moved physics not just aesthetics; and if the reward count grows, **TaRoS's Hoyer-sparsity adaptive component weights** keep the hard physics signal alive when easy aesthetic components saturate — a drop-in on the reader's static-weight `rewards/composite` registry (recompute per-group component weights from mean-threshold + Hoyer separation before advantage standardization; adds only cheap group statistics, no extra model). ([2503.06800](https://arxiv.org/abs/2503.06800), [2501.09038](https://arxiv.org/abs/2501.09038), [2511.19356](https://arxiv.org/abs/2511.19356))

**Two more portable, lower-priority pieces:**
- **VisionReward (video)** — a released public reward with named Physics/Stability/Dynamics/Shape sub-scores; drops into the reader's "wrap ready public models" pattern and gives per-dimension control signals the scalar-only rewards lack. Its *optimization* method (MPO Pareto-DPO) is not the reader's GRPO, but the *reward* is pluggable. Reward model is CogVLM2-Video-19B — tight on 32GB, likely needs the reader's `sleep_offload` path. ([2412.21059](https://arxiv.org/abs/2412.21059))
- **RoboAlign-R1's SWR** (Sliding-Window Re-encoding: decode last frame → re-encode as fresh context → reset KV-cache) is a directly portable *inference* trick for long autoregressive/world rollouts (+2.8% SSIM, −9.8% LPIPS at ~1.1% latency), independent of its trained-judge reward. ([2605.03821](https://arxiv.org/abs/2605.03821))

**Explicitly-flagged tension (not a recommendation):** EVA + RLIR are the strongest evidence that a **trained IDM action-recovery reward** materially improves controllability *and* co-improves visual fidelity (EVA real-robot OOD +18pt; RLIR +15.7% F1, FVD −10.4%). Both require training an IDM — the path the reader's memory records rejecting on the wrap-only principle, and no ready public robot-domain IDM exists (VPT-IDM is Minecraft-only). Recorded here as decision evidence, not as a proposed add.

**Feasibility summary (single 5090 / 32GB):** plausible as-is or reward-only — PhysRVG (5B, 8-bit Adam/LoRA), World-R1-Small (Wan2.1-1.3B, reward only), Diffusion-DRF (1.3B + memory tricks, if adopting reward-backprop), Flash/TempFlow-GRPO (fewer steps), V-JEPA2/RAFT/SAM2 rewards, VBench-2.0/Physics-IQ/VideoPhy-2 eval. **Not** single-card — MIND-V full pipeline (~70–122GB), NewtonRewards replication (8×H100), EVA (8×A800), World-R1-Large (96×H200), VisionReward's 19B reward without offload.

---

## 7. References (deduped)

- EVA — https://arxiv.org/abs/2603.17808 · code https://github.com/RobbinW/EVA · https://eva-project-page.github.io/
- TempFlow-GRPO — https://arxiv.org/abs/2508.04324 · code https://github.com/Shredded-Pork/TempFlow-GRPO
- PhysRVG — https://arxiv.org/abs/2601.11087 · code (announced, not confirmed live) https://github.com/ant-research/PhysRVG · weights (announced) https://huggingface.co/HappyP4nda/PhysRVG · https://lucaria-academy.github.io/PhysRVG/
- VGGRPO — https://arxiv.org/abs/2603.26599 · https://zhaochongan.github.io/projects/VGGRPO
- TaRoS — https://arxiv.org/abs/2511.19356
- Flash-GRPO — https://arxiv.org/abs/2605.15980 · code https://github.com/Shredded-Pork/Flash-GRPO · https://shredded-pork.github.io/Flash-GRPO.github.io/
- TAGRPO — https://arxiv.org/abs/2601.05729
- RLVR-World — https://arxiv.org/abs/2505.13934 · code https://github.com/thuml/RLVR-World
- RLIR — https://arxiv.org/abs/2509.23958 (HTML: https://arxiv.org/html/2509.23958v1)
- MIND-V — https://arxiv.org/html/2512.06628v1 · code https://github.com/Richard-Zhang-AI/MIND-V
- Diffusion-DRF — https://arxiv.org/abs/2601.04153 · https://snap-research.github.io/diffusion-drf/
- Cosmos-Predict 2.5 / DDRL — https://arxiv.org/abs/2511.00062 · code https://github.com/nvidia-cosmos/cosmos-predict2.5 · https://github.com/nvidia-cosmos/cosmos-rl · DDRL paper https://arxiv.org/abs/2512.04332
- VisionReward — https://arxiv.org/abs/2412.21059 · code https://github.com/zai-org/VisionReward
- RoboAlign-R1 — https://arxiv.org/abs/2605.03821 · repo (empty) https://github.com/Alexander-wu/RoboAlign_R1 · https://roboalign-r1.netlify.app/
- NewtonRewards — https://arxiv.org/abs/2512.00425 · code https://github.com/cvlab-stonybrook/NewtonRewards · https://cvlab-stonybrook.github.io/NewtonRewards
- VBench-2.0 — https://arxiv.org/abs/2503.21755 · code https://github.com/Vchitect/VBench · https://vchitect.github.io/VBench-2.0-project/
- VideoPhy-2 — https://arxiv.org/abs/2503.06800 · code https://github.com/Hritikbansal/videophy · https://videophy2.github.io/
- Physics-IQ — https://arxiv.org/abs/2501.09038 · code https://github.com/google-deepmind/physics-IQ-benchmark
- ReWorld — https://arxiv.org/abs/2601.12428
- Reward as An Agent — https://arxiv.org/abs/2606.19990 (HTML: https://arxiv.org/html/2606.19990v1)
- World-R1 — https://arxiv.org/abs/2604.24764 · code https://github.com/microsoft/World-R1 (MIT) · https://aka.ms/world-r1
- PhysisForcing — https://arxiv.org/abs/2606.28128 · code https://github.com/DAGroup-PKU/PhysisForcing

*Confidence flags carried in-text: VGGRPO metrics (medium, no code), Cosmos-2.5 reward=VideoAlign (medium/low, secondary source), Reward-as-Agent/DynDiff-GRPO gains (medium, no code), PhysisForcing (medium). Corrected per verification: PhysRVG code/weights are announced-not-confirmed (paper says "will be released soon"); Physics-IQ best score is 24.1% (not 29.5%), p=.247; ReWorld's −24pt drop is naive-PPO-wrong-sign not proxy-removal, and only Cosmos-2B is RL-aligned; VideoPhy-2 "22%" is the paper figure, live leaderboard now higher; VideoPhy-2-AutoEval "7B" is inferred from the mPLUG-Owl backbone, not quoted; DDRL knobs are documented in cosmos-rl/the DDRL paper, not the Cosmos-Predict2.5 paper body. All other cited facts are high-confidence per the source extraction.*