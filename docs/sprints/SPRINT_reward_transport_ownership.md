# SPRINT：reward 层的传输归属——按 miles 的分工收拢

状态：plan（2026-09-15），在 `review/all-gpu-main-20260914` 上写；执行等这条分支 review 收尾后单独提交。

## 问题

`ebee861d` 之后每个模型型 reward 都是 `DiskArtifactRewardFunction` 子类，22 个文件里 10 个把同样 7 个传输参数
（`device`、`scorer`、`inference`、`artifact_format`、`media_type`、`artifact_dir`、`retain_artifacts`）
在自己的 `__init__` 里重新声明一遍再原样转发给基类（`pickscore.py:31-60`、`nsfw_safety.py:31-61`、
`motion_dynamics.py:33-57`……）。而 registry（`registry.py:225-305`）在构造之前已经把 `inference` 解析好、
HTTP 的 `scorer` 注入好、`sleep_offload` 决定好，子类没有增加任何信息，只是搬运。

对照 miles_diffusion（`miles/rollout/rm_hub/`）：传输只在 `AsyncRewardActorPool` 一处；每个 reward 文件是
Scorer（模型）/ Actor（薄壳）/ Pool 子类（7 行把 args 映射给池）/ `xxx_rm` 函数（3 行），模型代码与 reward
函数本体不碰传输。VRL 要保留的是类型化配置和 CuMem 时分，不是这份签名重复。

## 目标形状

1. **传输是一个值，registry 构造一次。** `vrl/rewards/base.py` 增加
   `RewardTransport(scorer, artifact_store, retain_artifacts, device)`（frozen dataclass），以及
   `resolve_reward_transport(reward_cls, *, name, inference, device, kwargs, memory_parking_required)`：
   把 `registry.py:228-303` 的 kind 分支和 `DiskArtifactRewardFunction.__init__` 里
   `base.py:585-668` 的 scorer / store / worker_cfg 逻辑合到这一个函数。registry 收尾变成
   `reward_cls(transport=transport, **model_kwargs)`。
2. **子类只声明"算什么分"。** `DiskArtifactRewardFunction.__init__(self, device="cuda", *, transport=None,
   score_key=None, debug_dir="", **worker_config)`；`transport is None` 时用 `device` 构造 in-process 传输，
   这样测试里 `OCRReward(device="cpu")` 的直接构造照常。类级声明块（`model_factory`、`request_prefix`、
   `default_*`、`in_process_media`、`eager_model`）不变，它已经是 reward 的身份。
3. **删掉能删的构造函数。** kwargs 与 `worker_config` 一一对应的子类不再有 `__init__`：
   pickscore、aesthetic、geneval_owl、image_sharpness、wd_tagger、motion_dynamics、target_dino_similarity、
   codex_image_qa。保留几行 `__init__` 的只有需要自己校验的：OCR（`score_key` 白名单）、
   nsfw_safety（模型自己的 `scorer` 可调用对象与传输 `scorer` 消歧）。
4. 非磁盘型的 `geneval` 不动；registry 对它拒绝 http/service 的规则不变。

## 不做的

- 不引入 CLI flag 或 miles 的 `args` 对象；配置仍走 `reward.kwargs.<name>` 与 `reward.inference.<name>`。
- 不改 in-process / service / http 三条传输的行为，不改 `worker_config` 送到服务的内容。
- 不合并 `InferenceRewardFunction` / `CumemRewardFunction` / `DiskArtifactRewardFunction` 三层；先收签名，层级留给下一轮看是否还需要三层。

## 验收（改前改后都要跑）

- `tests/rewards`：全部通过；`test_every_former_in_process_reward_can_run_as_a_managed_service` 的
  `scorer.worker_config` 断言逐 reward 相同（这是"送到服务的东西没变"的证据）。
- `tests/scripts/test_online_entrypoint.py`、`tests/scripts/test_online_lifecycle.py`（reward runtime 构造路径）。
- 全套 CPU；只允许当时已知的上游红灯。
- 行数：22 个 reward 文件合计从 1493 行下降，每个迁移文件不再出现 `retain_artifacts: bool = False`
  （`grep` 计数 10 → 2）。

## 顺序

1. base：`RewardTransport` + `resolve_reward_transport`，`DiskArtifactRewardFunction.__init__` 改签名。
2. registry：`from_dict` 改调 resolver。
3. 22 个子类逐个迁移，一次提交。
4. 测试按上面的验收跑；账本写进 `SPRINT_test_credibility.md`。

## 同一条分支上还挂着的三件事（review 时一起定）

- `cb92c573` 让 `vrl.generation` 与 `vrl.rewards` 互相 import（`RewardArtifactSpec` /
  `MaterializedArtifact`），两个架构分层测试红；两个 dataclass 应搬到 `vrl/utils/artifacts.py`。
- continuous 队列里 `attempt`、`backpressure_seconds/entries` 只进日志，按派生字段规则删或标 display-only。
- `787d703c` 用生成 batch=1 过 replay-parity 门，是绕过不是修根因（4 对 4 仍 0.0111）。
