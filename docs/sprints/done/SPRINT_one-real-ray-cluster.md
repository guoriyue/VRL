> **执行状态（2026-07-30）：DONE。** 共享真 Ray 集群及本轨转换已由 commit
> `337b8d31` 落地。本文以下内容保留为执行前审计、原型数据与施工计划快照。

# SPRINT: 轨道二 — 一个真 Ray 集群，以及它买得起的真 Ray 转换（净省 ~17s）

原计划状态：**planned（已由文件首部 DONE 执行状态取代）**。轨道顺序 2 / 6
（tier-policy 地基之后，三条更贵的转换轨道之前）。风险：**medium**。

> **这是六条轨道里唯一净省时间的一条，也是唯一一条“先省钱、再花钱”的。**
> 今天真 Ray 集群有六个 owner，每个各自手搓一遍同样的 `ray.shutdown()` / 关 uv hook /
> `ray.init()`；其中 `test_rollout_launcher.py` 那份**从不还原**
> `RAY_ENABLE_UV_RUN_RUNTIME_ENV`，把它泄漏成进程级全局状态。本轨道把这套机械收成
> **一个 contextmanager + 两个薄壳**（function-scoped `local_ray`、package-scoped
> `local_ray_pkg`），然后把省下来的集群花在**单独做都不划算**的真 Ray 转换上。
>
> tier 判据见 `docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md`（轨道一）。
> 本文不复述判据，只在需要时引用。

---

## 0. 头条：本轨道让套件**变快**

先给结论，再给账：

| 车道 | 今天 | 本轨道之后 | 差额 |
|---|---|---|---|
| **fast PR lane**（`-m "not e2e and not slow_test"`） | 基线 | 基线 + **0.3 ms** | **≈ 0**（CRD-05 的真 resolver，两次 ×0.13 ms） |
| **nightly slow_test lane** | — | — | **−17.4 s** |
| ↳ 其中 `tests/ray` 的 4 个独立集群测试 | **17.53 s** | **5.50 s**（含 CRD-02 搭车） | **−12.0 s**（已原型，3 次复测） |
| ↳ 其中 `tests/generation/ray` 的 `local_ray` 起停 | **15.09 s** 集群 churn | 一次 ~4.5 s | **−7.9 s**（GR-01 原型实测的下限） |
| ↳ 新增真覆盖的执行成本 | 0 | **+2.54 s** | 6 个新真 Ray 测试 |

**PR 车道增量必须单独说：本轨道对 PR 车道的净增量是 0.3 毫秒。** `ci.yml:131` 的
`uv run --no-sync pytest -m "not e2e and not slow_test"` 把 `slow_test` 全部 deselect，
所以除 CRD-05 的 resolver 半边与 GR-03 Part A（0 s）以外，本轨道**一行都不进 PR 车道**。

**为什么 CRD-02 必须搭 CRD-01 的车（这条决定了排序）：** CRD-02 的转换在共享集群上实测
**0.20 s**；单独起一个集群做同一件事要 **+4.5 s**。同一个转换，价格差 22 倍。本轨道里
GR-02 / GR-03 / GR-06 / CRD-02 五个新真 Ray 测试，**没有一个在“每测试起停集群”下划算**，
在共享集群下全部是 0.2–1.3 s 的零头。所以“共享机械”和“靠它才划算的转换”必须是同一个
sprint，不能拆。

---

## 0.1 复核时发现的一件比审计更严重的事：泄漏是 CI 绿灯的**必要条件**

审计说 launcher 的 `RAY_ENABLE_UV_RUN_RUNTIME_ENV` 泄漏“是 `tests/ray` 那段
`import pytest` 防御看起来死掉的真实原因”。复核时我把这条推到底，结论比审计更硬：

**那 4 处裸 `ray.init()` 在 `uv run` 下根本起不来 worker。而 `uv run` 正是 CI 调用 pytest
的方式（`ci.yml:131` / `:166`）。**

隔离实验（除了那一个 flag，其它完全相同）：

```
$ uv run --no-sync python probe_uv_hook.py            # 复刻 test_ray_actor_pool.py:41 的裸 ray.init()
driver: RAY_ENABLE_UV_RUN_RUNTIME_ENV = True
ray.exceptions.RaySystemError: System error: Failed to startup worker after retrying 5 times.

$ RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run --no-sync python probe_uv_hook.py
driver: RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
worker: {'has_pytest': True, 'executable': '.../.venv/bin/python3',
         'can_import_test_module': 'ok', 'cwd_on_path': True}
```

（另有一次 `uv run --no-sync pytest tests/ray/test_ray_actor_pool.py -m slow_test` 跑到
**600 s 超时被杀**、一行输出都没有。我不把这次当主证据，因为超时的成因没有隔离干净；
上面那对直连探针才是干净的对照。）

根因在 Ray 自己的门里，`_skip_env_hook=True` **管不到它**：

```python
# .venv/lib/python3.12/site-packages/ray/_private/worker.py:1385-1393
if ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV:
    from ray._private.runtime_env.uv_runtime_env_hook import _get_uv_run_cmdline, hook
    cmdline = _get_uv_run_cmdline()
    if cmdline:
        return hook(runtime_env)          # <- 在 _skip_env_hook 之前就 return 了
if ray_constants.RAY_RUNTIME_ENV_HOOK in os.environ and not _skip_env_hook:
```

所以 `tests/conftest.py:106-107` 那两行 flag 翻转是**唯一**能关掉 uv 路径的东西，
`_skip_env_hook=True` 只挡另一条 hook。conftest 的注释是对的，但它没说清楚这一点。

**为什么 nightly 今天是绿的：** pytest 的目录收集顺序是字典序，`tests/generation/` 在
`tests/ray/` 之前（已实测 `--collect-only`：`algorithms, architecture, config, data, e2e,
generation, math, models, nn, quality, ray, rewards, ...`）。于是
`tests/generation/ray/test_rollout_launcher.py:128` 先把 flag 设成 `False` 且**永不还原**，
后面 `tests/ray` 的 4 处裸 `ray.init()`、以及更后面 `tests/scripts` 的两个 Ray fixture，
全都白捡这个泄漏。

**这条直接定死了一个硬排序约束：**

> **单独修泄漏（给 launcher 补上还原）会当场把 `tests/ray` 和 `tests/scripts` 的 nightly
> 打红。** 还原只能和“每个 `ray.init()` 站点自己拥有 flag 翻转”一起落地。本轨道把
> `tests/ray`（4 处）收进来；`tests/scripts` 的两处（`test_online_lifecycle.py:38` 的
> `preinitialized_ray`、`test_online_ray_cluster.py:130`）不在本轨道的猎取范围里，但它们
> 是同一个 blocker，所以本轨道**把它们也改成消费同一个 contextmanager**——纯机械改动、
> 0 成本，见 §3.3。不这样做，泄漏就修不了。

---

## 1. 实测基线（本机，HEAD `812cc3cf`，`.venv/bin/python -m pytest ... -q -p no:randomly`）

| 口径 | 结果 |
|---|---|
| `pytest tests/ray` | **134 passed / 18.06 s**（第二次 18.01 s） |
| ↳ 4 个独立集群测试的 call 时间 | 5.23 + 4.54 + 4.29 + 3.53 = **17.59 s**（第二次 5.24 + 4.52 + 4.24 + 3.53 = **17.53 s**） |
| `pytest tests/generation/ray -m slow_test` | **8 passed / 39.48 s** |
| `pytest tests/ray tests/generation/ray` | **253 passed / 61.86 s** |
| `pytest --collect-only` | **3817 collected**（tier-policy 基线 3816 + 未追踪的 GR-06 探针文件） |
| `pip list \| grep -i random` | 空 —— **pytest-randomly 未安装**，模块不会交错执行 |
| Ray 版本 | 2.55.1；`ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV` 默认 **True** |

### 1.1 共享集群原型（scratchpad，module-scoped 一个集群跑 5 个测试，3 次复跑）

```
5.86 s / 5.94 s / 5.90 s   (5 passed)

2.15 s  setup      <- 集群启动，整包付一次
2.31 s  teardown   <- 集群关闭，整包付一次
0.07 s  call  test_ray_actor_group_launch_lifecycle                     (今天 5.24 s)
0.07 s  call  test_run_actor_jobs_awaits_real_object_refs               (今天 4.24 s)
0.26 s  call  test_owner_reserves_trainer_gpu_and_binds_roles_...       (今天 4.52 s)
0.40 s  call  test_owner_shares_one_bundle_for_rollout_and_reward_...   (今天 3.53 s)
0.20 s  call  test_probe_actor_kill_failure_is_a_create_failure  <- CRD-02 转换版，搭车
```

**4 个原测试 + 共享集群 = 4.50 + 0.80 = 5.30 s（今天 17.53 s，−12.2 s）；
CRD-02 搭车后 5.50 s（−12.0 s，同时多一个真测试）。**
`tests/ray` 整目录从 **18.03 s → ~6.0 s**。

原型用的初始化参数是 `address="local", num_cpus=8, num_gpus=4, include_dashboard=False,
log_to_driver=False, _skip_env_hook=True` + flag 翻转。`num_gpus=4` 是**逻辑记账**，
`test_global_placement.py:604-609` 的注释已经写明这一点，探针 actor 只调
`ray.get_gpu_ids()`，从不碰 CUDA——原型在这台单卡机上全绿证实了这条。

### 1.2 `tests/generation/ray` 的 churn 明细（`--durations`，实测）

```
2.12 s setup + 2.31 s teardown   test_lifecycle_fsm::test_shutdown_kills_only_owned_actor      (call 0.11 s)
2.11 s setup + 2.07 s teardown   test_runtime_config::test_runtime_capability_is_and_...        (call 1.57 s)
2.10 s setup + 1.20 s teardown   test_runtime_config::test_runtime_capability_false_without_... (call  hidden)
2.10 s setup + 1.08 s teardown   test_runtime_config::test_runtime_capability_false_when_...    (call 0.70 s)
5.48 / 5.46 / 5.42 / 5.38 s call test_rollout_launcher × 4   <- 集群启停在 call 里（_init_ray 在测试体内）
```

**`local_ray` 的 4 次起停 = 8.43 s setup + 6.66 s teardown = 15.09 s，买到的断言只有
0.11 + 1.57 + ~0.6 + 0.70 ≈ 3.0 s。** launcher 的 4 个测试各含一次完整启停
（`_init_ray` 2.1 s + finally 的 `ray.shutdown()` 2.3 s ≈ 4.4 s），真 launcher 工作
只占每个 ~1.0 s。

---

## 2. 总表

| 测试路径 | 今天替身/浪费的是什么 | 变成 | tier | 实测成本 |
|---|---|---|---|---|
| `tests/conftest.py:88-119` | 无替身。function-scoped `local_ray`，每个真 Ray 测试付一次全启停 | 提成 `_real_local_ray()` contextmanager + 两个薄壳 | — | **−7.9 s**（GR-01 原型） |
| `tests/ray/conftest.py`（新） | `tests/ray` 4 处手搓 `ray.init` 各自一套 | package-scoped `local_ray_pkg` 消费者 | — | **−12.0 s**（本文 §1.1） |
| `tests/ray/test_ray_actor_pool.py:5-8, :17` | `try: import pytest / except` + `pytestmark = ... if pytest is not None else ()` | 删死代码，恢复 `pytestmark = pytest.mark.slow_test` | — | 0 |
| `tests/ray/test_ray_actor_pool.py:38-43, :68` / `:88-92, :136` | 裸 `ray.init(ignore_reinit_error=True, ...)`，不关 uv hook | 消费 `local_ray_pkg` | T1（本来就是） | 5.24→0.07 s / 4.24→0.07 s |
| `tests/ray/test_global_placement.py:617-619, :645` / `:651-653, :678` | 同上，`num_cpus=8, num_gpus=4` | 消费 `local_ray_pkg` | T1（本来就是） | 4.52→0.26 s / 3.53→0.40 s |
| `tests/ray/test_global_placement.py:611` | `pytestmark_slow = pytest.mark.slow_test`（命名错误、零引用、完全无效） | **轨道一已认领删除** | — | 0 |
| `tests/ray/test_global_placement.py:528-601` | 5 个嵌套 fake 类（`_Method`/`_Actor`/`_PlacementGroup`/`_RemoteProbe`/`_Ray`）+ 4 个 monkeypatch，60 行 setup / 4 行断言 | **T1**：真 PG、真 `_ProbeActor`、真 `pg.ready()`、真 `remove_placement_group`；只留 `kill_actors` 一个注入 | T1 | **+0.20 s** |
| `tests/ray/test_chunk_dispatch.py:254-385` | `_FakeActor` 内嵌 `_ExecuteChunk.remote` → `_FakeRef`，`RayGenerationExecutor.execute` 全仓**零真 Ray 覆盖** | **保留全部 3 个 fake-ref 测试** + 新增 2 个真 Ray 执行器孪生 + 改掉文件头 over-claim | T1（新增） | **+0.18 s** |
| `tests/ray/test_cross_node_preflight.py:17-42` | `_resources(...) = SimpleNamespace(rollout_num_gpus=…)` 顶替 `ResolvedDistributedResources` | **T1**：真 `resolve_distributed_resources(...)` | T1 | **+0.26 ms** |
| `tests/ray/test_cross_node_preflight.py:17-22` ＋ `test_dependencies.py:10-15` | `_ray` / `_node` 逐字重复两份 | 去重进 `tests/ray/_helpers.py` | — | 0（纯移动） |
| `tests/ray/test_cross_node_preflight.py:36-42` | 通篇无 assert，靠“不抛即通过” | 加 `assert placement.cross_node_preflight(...) is None`（实测返回 `None`） | T1 | 0 |
| `tests/generation/ray/conftest.py`（新） | — | package-scoped `local_ray_pkg`（带 launcher 的 `worker_process_setup_hook`） | — | 见 §3.2 |
| `tests/generation/ray/test_rollout_launcher.py:119-136` | `_init_ray` 把 `RAY_ENABLE_UV_RUN_RUNTIME_ENV` 设 `False` **从不还原**；4 处调用 + 4 处 `ray.shutdown()` | 删掉 `_init_ray`，消费 `local_ray_pkg` | — | 见 §3.2（上限 −13 s，未原型） |
| `tests/generation/ray/test_weight_sync.py:51-57, :100-120` | 整个 Ray object store 与 actor wire：`_FakeRay.put` 返回 `("state", value)` | **保留全部 5 个** + 新增 2 个真 Ray 测试（真 `ray.put` 一次共享、真跨进程 auto-deref、真 ack 归属） | T1（新增） | **+1.31 s** |
| `tests/generation/ray/test_health_monitor.py:76-84` | `_install_ray` 把生产 `kill_actors` monkeypatch 掉 | **Part A**：删掉那个 monkeypatch，让真 `kill_actors` 跑（`_FakeRay.kill` 签名已兼容） | T1 | **0 s，默认车道** |
| `tests/generation/ray/test_health_monitor.py:118-142` | `assert ray.killed == [healthy, wedged]` 断言测试自己的 lambda 往自己的 list 里 append | **保留全部 8 个** + 新增 1 个真卡死 actor / 真 `ray.get(timeout=)` 到期 / 真 `RayActorError` 连坐 | T1（新增） | **+0.6 s** |
| `tests/generation/execution/test_chunk_memory_shadow.py:285-321` | `_probe_worker` 的 `probe` 是普通函数 → 生产只走 local 分支；`.remote()` + `ray.get(timeout=600)` 分支全仓零覆盖 | **保留现有 local-branch 测试原样**（它覆盖真生产路径）+ 采纳已存在的真 Ray 探针 | T1（新增） | **+0.25 s** |
| `tests/scripts/test_online_lifecycle.py:32-42`、`tests/scripts/test_online_ray_cluster.py:26-30` | 两个 Ray fixture 都不翻 uv flag，靠 launcher 的泄漏活着 | 改成消费同一个 contextmanager（**解除泄漏修复的 blocker**） | — | 0 |

**合计：机械 1 处（conftest）+ 2 个新 conftest + 1 个新 `_helpers.py`；
真转换 3 处（CRD-02 / CRD-05 / GR-03 Part A）；新增真 Ray 测试 6 个；
删死代码 2 处；标注 4 处（用轨道一的 `real_cover`）。
nightly slow lane 净 −17.4 s；PR lane +0.3 ms。**

---

## 3. 机制：一个 contextmanager + 两个薄壳

### 3.1 落点与形状

`tests/conftest.py` 里把现有 `local_ray`（`:88-119`）的**函数体**提成一个
contextmanager，fixture 退化成薄壳。现有 docstring 的契约**逐字保留**——那段“uv hook
会打包整个 CWD 并用新解析出的环境起 worker”的解释是载荷性的（§0.1 已给出它比注释里
说的更严重），只补一句“`_skip_env_hook=True` 挡不住 uv 路径，flag 翻转才是唯一开关”。

```python
@contextlib.contextmanager
def real_local_ray(**init_kwargs):
    """Real local Ray cluster (small, CPU-only) for real-Ray tests.

    ``address="local"`` always starts a fresh cluster, so an operator cluster
    already running on the host is never hijacked; teardown disconnects and stops
    only the processes this driver spawned (never ``ray stop``).

    Ray's uv hook packages the entire CWD and launches workers through a newly
    resolved project environment, which strips the driver's dev dependencies.
    ``_skip_env_hook=True`` does NOT cover that path -- worker.py returns from
    the uv branch before it is consulted -- so flipping
    ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` is the only switch, and restoring it is
    mandatory: leaving it False leaks into every later ray.init() in the run.
    """
    ray = pytest.importorskip("ray")
    from ray._private import ray_constants

    ray.shutdown()
    previous_uv_hook = ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    try:
        ray.init(
            address="local",
            num_cpus=8,
            num_gpus=4,
            include_dashboard=False,
            log_to_driver=False,
            _skip_env_hook=True,
            **init_kwargs,
        )
        yield ray
    finally:
        ray.shutdown()
        ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = previous_uv_hook


@pytest.fixture()
def local_ray():
    with real_local_ray() as ray:
        yield ray
```

`tests/ray/conftest.py` 与 `tests/generation/ray/conftest.py` 各放一个同名 package-scoped
覆盖（pytest 的就近 conftest 覆盖是标准做法，不需要新名字——**这就是“两个薄壳”里的第二个**）：

```python
_PKG_INIT_KWARGS: dict[str, Any] = {}          # tests/ray：无额外参数


@pytest.fixture(scope="package")
def local_ray():
    with real_local_ray(**_PKG_INIT_KWARGS) as ray:
        yield ray
```

**参数统一成 `num_cpus=8, num_gpus=4`（今天 `local_ray` 是 `num_cpus=2` 且不给 GPU）。**
理由：`tests/ray` 的 placement 测试需要 8 CPU / 4 逻辑 GPU；`tests/generation/ray` 的
actor 全是 `num_cpus=0`，多给 CPU/逻辑 GPU 对它们是惰性的（§1.1 原型在同一个 8/4 集群上
把两类测试一起跑绿）。**统一参数是让两个薄壳可互换的前提**，否则 launcher 一次
`ray.shutdown()` 之后 package fixture 拿到的就是错配的集群。

### 3.2 launcher 为什么必须进同一个 package 集群（本节的省时**未原型验证**）

`tests/generation/ray` 里文件的字典序是
`test_health_monitor` → `test_lifecycle_fsm` → `test_oom_split` → `test_ray_resident_session`
→ **`test_rollout_launcher`** → `test_runtime_config` → …

也就是说 launcher **夹在两个 `local_ray` 消费者中间**。只要 launcher 还拥有自己的集群，
它的 `ray.shutdown()` 就会在整包中途把 package 集群拆掉，后面 `test_runtime_config` 三条
拿到的是死 handle。所以只有两种自洽方案：

- **(a) 一个 package 集群，创建时就带上 launcher 的 `runtime_env`**（`_PKG_INIT_KWARGS =
  {"runtime_env": {"worker_process_setup_hook": _worker_setup_hook(repo_root)}}`）。
  hook 只在 **worker 进程**里改 `registry.FAMILY_REGISTRY["janus_pro"]` /
  `ModelFamilyEntry.build_rollout` / `resolve_checkpoint_model_identity`；本包其余 actor
  （`_ReleaseWorker` / `_SlotWorker` / 新增的 `_InstallWorker` / `_Wedged` / `_ProbeActor`）
  一个都不碰 registry，所以是惰性的。`test_oom_split` 的 `_CapacityWorker` 不在 slow lane、
  不起真 Ray（本包 8 个 `slow_test` 就是 1 个 lifecycle_fsm + 3 个 runtime_config +
  4 个 rollout_launcher，已实测枚举），也不受影响。
- **(b) launcher 保留自己的 module-scoped 集群**，package 薄壳必须做成可重入
  （`if not ray.is_initialized(): re-init`）。

**取 (a)。** 理由不是省时间而是**语义**：hook 是测试脚手架（把测试用 executor 发布到一个
可 import 的生产模块上），不是被测行为；让一个集群带着它，比让 fixture 学会“被别人拆掉后
自己爬起来”要简单得多，也少一条隐式时序依赖。

**成本上限（算术，非原型）：** launcher 4 个测试今天各含 ~4.4 s 集群启停 = ~17.6 s；
进 package 集群后这 17.6 s 归零（集群那 4.5 s 已经计在 `local_ray` 那一侧）。
`tests/generation/ray` 的 slow lane 有望从 **39.48 s → ~12 s**。
**本文的合计里我只记 GR-01 原型实测的 −7.9 s，不记这 −13 s 的上限**——它需要执行者自己
原型一次再写进 PR 描述。如果 (a) 出现意外，退回 (b)，−7.9 s 仍然成立。

### 3.3 顺手解除泄漏修复的 blocker：`tests/scripts` 两处

`tests/scripts/test_online_lifecycle.py:32-42` 的 `preinitialized_ray` 与
`tests/scripts/test_online_ray_cluster.py:26-31` 的 `isolated_ray` **都不翻 uv flag**，
今天靠 launcher 的泄漏活着（`tests/scripts` 在 `tests/ray` 之后收集，也白捡）。

- `preinitialized_ray`：改成走 `real_local_ray()`。它已经是“共享 + module-scoped teardown”
  的形态（`:25-30` 的 `_module_ray_teardown`），本轨道只换初始化机械，不动它的 scope，
  也不动它“必须让集群活着”的断言（`:692` / `:923` 的 `is_initialized()`）。
- `isolated_ray`：**不能**换成 `real_local_ray()`。它的契约是**测试前后都没有 driver 连接**
  （`ray.shutdown(); yield ray; ray.shutdown()`），被测代码 `online._RayClusterSession.connect`
  自己去 `ray.init`。这是一个**故意相反**的契约，折进共享 fixture 会直接把它测的东西抹掉。
  改法：只加 flag 翻转 + 还原（3 行），保留 shutdown/shutdown 形态。

**Verify：**
```bash
.venv/bin/python -m pytest tests/scripts/test_online_lifecycle.py tests/scripts/test_online_ray_cluster.py -q -p no:randomly
```

### 3.4 medium 风险的来源，必须写进 conftest 的 docstring

**共享集群没有 per-test actor 命名空间，也没有 per-test 的 `cluster_resources()` 快照。**
两条硬规则：

1. **每个测试必须自己回收它建的 actor**（`ray.kill(actor, no_restart=True)` 或
   `group.shutdown()` / `owner.shutdown()`）。§1.1 的原型里
   `test_run_actor_jobs_awaits_real_object_refs` 就必须显式 kill 两个 `_PayloadWorker`；
   不清理会在 nightly 日志里留下不确定条数的 `SchedulingCancelled` ERROR。
2. **不要写依赖“干净 actor 命名空间”或 `cluster_resources()` 精确值的新测试。**
   这条要写在 fixture docstring 里而不是只写在本文里——将来任何这类测试都会**静默**出错，
   而不是红。

---

## 4. 逐条

### 4.1 `CRD-09` — `tests/ray` 4 处手搓 `ray.init` + 一段**看起来**死掉的防御

**位置：** `tests/ray/test_ray_actor_pool.py:5-8` / `:17` / `:38-43,68` / `:88-92,136`；
`tests/ray/test_global_placement.py:617-619,645` / `:651-653,678`；`tests/conftest.py:88-119`。

**今天的样子（逐字，已复核）：**

```python
# tests/ray/test_ray_actor_pool.py:5-8
try:
    import pytest
except ModuleNotFoundError:  # Ray workers import this module for test actors.
    pytest = None
...
# :17
pytestmark = pytest.mark.slow_test if pytest is not None else ()
```

```python
# tests/ray/test_ray_actor_pool.py:38-43（另一处 :88-92 同形）
ray.shutdown()
group = None
try:
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2, log_to_driver=False)
```

**它证明了什么：** 这 4 个测试**本来就是 T1**，断言的都是真东西。缺陷不在断言强度，在
**重复手搓一份 `tests/conftest.py:88` 已经存在、而且带着一条重要正确性修复的机械**——
关掉 uv hook。`local_ray` 今天只有 4 个消费者，全在 `tests/generation/ray`
（`test_lifecycle_fsm.py:347`、`test_runtime_config.py:302/326/348`）；`tests/ray` 一份都没复用。

**关于那段 `try: import pytest` 防御，审计的裁定要修正。** 审计说它“今天是死的”。
我复核后的结论是：**它不是死的，是瞄错了。**

- Ray 用 **module path 按引用**序列化 actor class，worker 必须真的
  `import tests.ray.test_ray_actor_pool`——所以“worker 会 import 本模块”这个前提是**真的**。
  （反证：我把同样的 actor class 放进一个不在 `sys.path` 上的 scratchpad 模块，worker 直接
  `ModuleNotFoundError: No module named 'probe_shared_ray'`。）
- 但**它防的那个具体异常永远不会到达**。实测（`.venv/bin/python -m pytest`，uv hook 名义上
  开着但驱动不是 `uv run`）：worker 里 `find_spec('pytest')` 为 `True`、
  `import tests.ray.test_ray_actor_pool` 成功。而在真会触发 hook 的 `uv run` 下，失败发生在
  **worker 启动阶段**（`Failed to startup worker after retrying 5 times`），
  根本走不到模块 import，`ModuleNotFoundError` 不会出现。

所以**删除仍然是对的裁定，但理由要改写**：删的不是“已经没人会踩的防御”，是“防不住真实
失败模式、还顺带把 `pytestmark` 变成一个三元表达式（轨道一的 AST 车道解析必须为它专门穿透
`IfExp`，见轨道一 §3.3）的死壳”。真正的修复是让这个文件消费带 flag 翻转的 fixture。

**改法：** 新建 `tests/ray/conftest.py`（§3.1 的 package 薄壳）；4 处手搓全部改为消费它；
删 `:5-8` 的 try/except 与 `:17` 的三元，恢复成 `pytestmark = pytest.mark.slow_test`。
`test_global_placement.py:611` 那行零引用的 `pytestmark_slow` **由轨道一删**（轨道一 §8.1
已认领），本轨道不重复动。

**成本：** 见 §1.1，**−12.0 s**，全在 `slow_test` lane。删死代码 0。

**Verify：**
```bash
.venv/bin/python -m pytest tests/ray -q -p no:randomly --durations=10
# 期望：134 passed，目录总时长 18.0 s -> ~6.0 s，durations 里不再出现 4 个 4-5 s 的 call
```

---

### 4.2 `CRD-01` — `RayGenerationExecutor.execute` 全仓零真 Ray 覆盖

**位置：** `tests/ray/test_chunk_dispatch.py:254-385`（`_FakeActor` / `_FakeRef` /
3 个 `execute` 测试）；`tests/ray/test_ray_actor_pool.py:79-136`。

**今天的断言，以及它**确实**证明了的东西：**

```python
# tests/ray/test_chunk_dispatch.py:325-326
assert actors[0].executed == ["prompt:0:samples:0:2", "prompt:0:samples:4:6"]
assert actors[1].executed == ["prompt:0:samples:2:4", "prompt:0:samples:6:8"]
```

这是生产 planner 算出来的绑定，不是脚本返回值。**这三个测试有牙，一个都不删。**

**问题在文件头的 over-claim（逐字，已复核）：**

```python
# tests/ray/test_chunk_dispatch.py:8-12
The fakes here control event-loop completion ORDER, which real Ray cannot make
deterministic — they are a controlled clock, not a Ray protocol fake. The
protocol assumption they encode (real ObjectRefs await directly and resolve to
the task result) is pinned against a live cluster by the real-Ray twin
tests/ray/test_ray_actor_pool.py::test_run_actor_jobs_awaits_real_object_refs.
```

前半句是对的。后半句**不成立**：那个孪生钉的是 `run_actor_jobs`，**不是**
`RayGenerationExecutor.execute`。实测 `grep -rn RayGenerationExecutor`（全仓，排除
`__pycache__`）：消费者只有 `vrl/generation/ray/launcher.py:128`（生产）、
`vrl/generation/ray/runtime.py:68`（类型标注）、`tests/ray/test_chunk_dispatch.py`（fake actor）、
`tests/generation/ray/test_oom_split.py`（fake actor）。**`execute` 的真 Ray 覆盖是 0。**

后果是具体的：fake actor 是进程内直调，`ChunkExecutionEnvelope(request, chunk)` /
`ChunkExecutionResult` / `SampleChunk` / `GenerationRequest` **从来没被 pickle 过**。任何一个
字段变成不可序列化（lambda、开着的文件句柄、torch device 引用）在测试里 100% 通过，
在生产第一个 chunk 就炸。

**改法：**

1. 在 `tests/ray/` 新增两个真 Ray 执行器孪生（`round_robin` 与 `dynamic` 各一），
   消费 `local_ray_pkg`：一个 `@ray.remote(num_cpus=0)` 的 `_ChunkWorker.execute_chunk(envelope)
   -> ChunkExecutionResult`，断言 gather 顺序与 `runtime_debug.chunk_schedule`。
   **必须在测试末尾 `ray.kill` 掉这两个 actor**（§3.4 规则 1）。
2. **三个 fake-ref 测试原样保留**，并按轨道一的机制打 `real_cover` 指向新孪生。
3. 文件头 `:8-12` 后半句改成准确表述：fake ref 钉的是**完成顺序**，`run_actor_jobs` 的
   ObjectRef 直接 await 由 `test_ray_actor_pool.py::test_run_actor_jobs_awaits_real_object_refs`
   钉，**执行器整条 envelope→result 的过线**由新孪生钉。

> **与轨道一的分工（不要重复动）：** 轨道一 §2 与 HONEST GAPS 表里已认领
> “`_FakeRef`/`_FakeWorker` → 指向 `test_ray_actor_pool.py::test_run_actor_jobs_awaits_real_object_refs`”
> 这一条标注，以及 §8.4 的 docstring 压缩。本轨道只负责：**新建执行器孪生 + 为它写自己的
> 标注 + 改掉那句 over-claim**。轨道一自己写了规则：“创建孪生的 sprint 自己写自己的标注”，
> 守卫对悬空目标会红，先标后建必然中断。

**成本：** 审计实测新孪生 **0.09 s/个**，两个 **+0.18 s**，`slow_test` lane。PR lane 0。

**Verify：**
```bash
.venv/bin/python -m pytest tests/ray/test_chunk_dispatch.py tests/ray/test_ray_actor_pool.py -q -p no:randomly
```

---

### 4.3 `CRD-02` — 60 行 setup / 4 行断言，断言路径上没有一个真实协作者

**位置：** `tests/ray/test_global_placement.py:528-601`。

**今天的断言（有牙的那部分）：**

```python
# :598-601
assert caught.value.__cause__ is cleanup_error
assert remove_calls == [placement_group]
assert owner._placement_group is None
assert owner._placement_ready is False
```

**它证明了什么：** 断言本身是真主张（cleanup 失败必须冒泡成 create 失败、PG 必须被移除、
ownership 必须释放）。**但断言路径上一个真实协作者都没有**，而且 fake 里还藏着一句自证断言：

```python
# :571-576
@staticmethod
def get(refs, **_kwargs):
    if refs == "ready-ref":
        return None
    assert refs == ["probe-ref"]      # <- 在断言 fake 自己被怎么调用
    return [("node", (0,))]
```

那是把 Ray 的调用约定手抄了一遍再断言自己抄对了。5 个嵌套 fake 类
（`_Method` `:537` / `_Actor` `:544` / `_PlacementGroup` `:550` / `_RemoteProbe` `:557` /
`_Ray` `:565`）+ 4 个 monkeypatch，**60 行 setup / 4 行断言**——正是“setup 长到断言看不见”。

**改法（挂 CRD-01 的共享集群，只留一个注入）：**

```python
def test_probe_actor_kill_failure_is_a_create_failure(local_ray, monkeypatch) -> None:
    """Real PG, real _ProbeActor, real remove_placement_group; only the kill
    OUTCOME is injected -- a Ray actor kill cannot be made to fail on demand."""
    owner = GlobalRayPlacementOwner(_resolve({...}), _worker())
    cleanup_error = RuntimeError("probe kill failed")

    def failing_kill(ray, actors):
        resource_cleanup.kill_actors(ray, actors)      # really kill them, then report failure
        return [(actors[0], cleanup_error)]

    monkeypatch.setattr("vrl.ray.placement.kill_actors", failing_kill)

    with pytest.raises(RuntimeError, match="probe actor cleanup incomplete") as caught:
        owner.create()

    assert caught.value.__cause__ is cleanup_error
    assert owner._placement_group is None
    assert owner._placement_ready is False
```

**我已实测跑通并通过（§1.1 第 5 行，0.20 s）。** 5 个 fake 类和 3 个 monkeypatch 全删
（约 −55 行）。断言不变，还额外证明了**真 PG 确实被 `remove_placement_group` 移除、
ownership 确实释放**——这在今天是 fake 的 `remove_calls.append` 记的账。
`failing_kill` 里先调真 `kill_actors` 是必需的：否则探针 actor 会留在共享集群上（§3.4 规则 1）。

**`remove_calls == [placement_group]` 这一条断言会丢掉**（真 `remove_placement_group` 不记账），
换成上面那条 `owner._placement_group is None`——它是同一个不变量的**下游可观测量**，
且在真 PG 上比记账更强。**这不是删覆盖**：同一个主张，观测点从替身的 list 换成生产状态。

**成本（对审计的修正）：** 审计写 **0.05 s**。我复测 **0.20 s**（3 次一致）。差额来自真 PG
create + 真 GPU 探针 actor，审计那个数字大概没包含。结论不变——独立起集群要 **+4.5 s**，
所以必须搭车。落 `slow_test` lane，PR lane 0。

**`:384-434`（`_create_raw_placement_group` 失败）与 `:437-484`（`ready()` transport 失败）
不转。** 阻塞点具体：`vrl/ray/placement.py:48` 的 `_PLACEMENT_READY_TIMEOUT_S = 600.0`，
在 `:267` 被 `ray.get(pg.ready(), timeout=...)` 使用；真等一次 ready 超时要 10 分钟。
按 tier 政策这是 **T3-ENV**，保留并标注（见 HONEST GAPS）。

**Verify：**
```bash
.venv/bin/python -m pytest tests/ray/test_global_placement.py -q -p no:randomly --durations=6
```

---

### 4.4 `GR-01` — `local_ray` 是 function-scoped，每个真 Ray 测试付一次全启停

**位置：** `tests/conftest.py:88-119`；消费者 `tests/generation/ray/test_lifecycle_fsm.py:347`、
`tests/generation/ray/test_runtime_config.py:302/326/348`。

**没有替身。这是本轨道的预算发动机，成本为负。** 数据见 §1.2：4 次起停 15.09 s，
买到 ~3.0 s 的真断言。

**改法：** §3.1 的 contextmanager + 两个薄壳；`tests/generation/ray/conftest.py` 的 package
薄壳带上 launcher 的 `runtime_env`（§3.2 方案 (a)）。**现有 docstring 的契约逐字保留**——
uv hook 那段是载荷性的，并按 §0.1 补一句 `_skip_env_hook` 管不到 uv 路径。

**成本：** 审计原型实测 `pytest tests/generation/ray -m slow_test` **40.04 s → 32.13 s
（−7.9 s，8 个全绿）**；我今天复测基线 **39.48 s**，与审计一致。方案 (a) 的上限见 §3.2。
PR lane 0。

**Verify：**
```bash
.venv/bin/python -m pytest tests/generation/ray -m slow_test -q -p no:randomly --durations=20
# 期望：8（+3 个新增 = 11）passed；durations 里 setup/teardown 各只出现一次
```

---

### 4.5 `GR-02` — RL 系统里风险最高的那条路，100% 假 Ray

**位置：** `tests/generation/ray/test_weight_sync.py:51-57`（`_FakeRay`）、
`:100-120`、`:123-143`。

**今天的断言（逐字）：**

```python
# :117-120
assert ray.put_calls == [{"w": 1}]
shared_state = ("state", {"w": 1})
assert first.update_weights.calls == [(shared_state, 4)]
assert second.update_weights.calls == [(shared_state, 4)]
```

而 `shared_state` 那个值就是 `_FakeRay.put` 被写成要返回的东西：

```python
# :55-57
def put(self, value: Any) -> tuple[str, Any]:
    self.put_calls.append(value)
    return ("state", value)
```

**这就是 owner 点名的失败模式的教科书形态**：测试手工构造出替身被告知要返回的值，
再断言替身记下了它。没有 ObjectRef、没有序列化、没有进程边界。

而生产在这里做了一条**具体的、结构上无法用这个替身验证的**主张：

```python
# vrl/generation/ray/weight_sync.py:50-56
# Serialize the (potentially large) state dict once into the object store and
# hand every worker the same ObjectRef. ... Ray auto-dereferences the ref into
# the real dict before the worker method runs.
shared_state = ray.put(state_ref)
```

`put()` 返回一个 tuple 的替身，**永远测不到“worker 方法运行前 ref 被自动解引用成真 dict”**。
`:73-97` 的 local（非 remote）分支是真的，原样保留。

**改法：新增 2 个 `@pytest.mark.slow_test` 测试，现有 5 个全部保留。**

- **(a) `test_real_ray_weight_sync_derefs_one_shared_put`**：真 Ray actor
  `_InstallWorker.update_weights(state_ref, policy_version)`，**在 actor 进程内**断言
  `isinstance(state_ref, dict)`（auto-deref 这条主张只有真 Ray 能证），存下来、返回
  `policy_version`；驱动真 `RayGenerationWeightSync.push_to_rollout_workers(
  {"w": torch.arange(6)}, policy_version=4)` 跨两个 actor；断言
  `ray.get([a.installed_sum.remote() for a in actors]) == [15.0, 15.0]`——真张量真的过了
  两次进程边界。固定张量、无 RNG。
- **(b) `test_real_ray_weight_sync_rejects_wrong_ack`**：第二个 actor 返回
  `policy_version - 1`；断言真 `RuntimeError` 匹配 `rollout-1.*version 4.*expected 5`，
  经由真 `asyncio.gather` over 真 awaitable ObjectRef 触发。
  **顺便钉住一条今天完全没覆盖的东西：ack 是按提交序归属的，不是按完成序。**
  生产靠 `zip(remote_workers, installed_versions, strict=True)`
  （`weight_sync.py:65-69`）把 ack 配回 worker，而 `asyncio.gather` 保序（`:64`）。要让这条可观测，
  给 `rollout-0` 塞一次 `time.sleep(0.5)` 让它**最后**完成——如果哪天有人换成
  `asyncio.as_completed`，错误消息里的 worker id 会变成 `rollout-0`，测试红。

审计实测：2 passed，call 0.65 s + 0.66 s。

**成本：+1.31 s** 的 call 时间，`slow_test` lane；GR-01 摊掉集群后就只有这 1.31 s。
PR lane 0。复用 package 薄壳，无新 fixture。

**保留并标注的那一条：** `assert ray.put_calls == [{"w": 1}]` 是**全仓唯一**钉住
“一次 `put` 共享给 N 个 worker”（而不是 N 次 put）的断言。真实端没有可观测量能替代它——
真 `ray.put` 不记账，object store 的引用计数也不是稳定可断言的接口。
**保留 `:100-120` 整条测试，打 `real_cover` 指向 (a)。**

**Verify：**
```bash
.venv/bin/python -m pytest tests/generation/ray/test_weight_sync.py -q -p no:randomly
```

---

### 4.6 `GR-03` — 生产的 kill 被 monkeypatch 掉，断言的是测试自己的 lambda

**位置：** `tests/generation/ray/test_health_monitor.py:76-84`（`_install_ray`）、
`:118-142`、`:145-160`。

**今天的断言（逐字）：**

```python
# :142
assert ray.killed == [healthy, wedged]
```

而 `ray.killed` 是被这个 monkeypatch 填进去的：

```python
# :81-84
monkeypatch.setattr(
    "vrl.generation.ray.health_monitor.kill_actors",
    lambda _ray, actors: [ray.kill(actor) for actor in actors] and [],
)
```

**生产的 `kill_actors` 从不执行。** 这条断言在说“测试自己的 lambda 往测试自己的 list 里
append 了两次”。真 `ray.get` 超时也从来没被行使过，只有一个手抛的
`TimeoutError("ray.get timed out")`（`:128`）。

生产 module docstring 写着真正的不变量（`vrl/generation/ray/health_monitor.py:5-8`）：
“kills the fleet so the blocked driver call fails and the attempt unwinds”，
测试注释 `:123-124` 也重复了一遍（“its pending Ray calls raise RayActorError”），
但**进程内没有任何东西会产出 `RayActorError`**。

**改法分两半，可分别落地：**

**Part A（0 s，默认车道，先做）：删掉 `:81-84` 那个 monkeypatch，让真 `kill_actors` 跑。**
可行性是签名级的、已复核：

```python
# vrl/ray/resource_cleanup.py:19-26 —— 真 kill_actors
for actor in actors:
    try:
        ray.kill(actor, no_restart=True)
    except Exception as error: ...
return failures                       # 成功时返回 []
```

```python
# tests/generation/ray/test_health_monitor.py:56-57 —— _FakeRay.kill 的签名已经兼容
def kill(self, actor: Any, *, no_restart: bool = False) -> None:
    self.killed.append(actor)
```

真 `kill_actors(_FakeRay(), actors)` 会调 `_FakeRay.kill(actor, no_restart=True)`、
记进 `ray.killed`、返回 `[]`，生产 `:172` 的调用点继续走。**`ray.killed` 这条断言从
“测试的 lambda 记的账”升级成“生产的 `kill_actors` 遍历顺序 + `no_restart=True` 的传参”**。
`_FakeRay` 本身保留——它替的是 Ray 那条线，不是被测行为。0 成本，纯收益，PR 车道。

**Part B（+0.6 s，`slow_test`）：新增 1 个真卡死 actor 的测试。**

```python
@pytest.mark.slow_test
def test_real_wedged_worker_times_out_and_the_fleet_really_dies(local_ray) -> None:
```

两个真 actor：`_Probe.health()` 返回 `"ok"`，`_Wedged.health()` 做 `time.sleep(300)`。
用真 `DistributedWorkerHandle` 建真
`RolloutWorkerHealthMonitor(runtime, interval_s=0.01, timeout_s=0.5, first_wait_s=0.0)`，
调 `monitor._run_probes()`。三条断言：

1. **真 `ray.get(timeout=0.5)` 真的到期**——必须断言 `elapsed >= 0.5`。
   没有这一条，“超时真的发生了”就没有任何机制可查（一个立刻抛别的异常的 actor 会让测试
   同样绿）。
2. `runtime.lifecycle.phase is RuntimePhase.SHUTTING_DOWN`，`failure.worker_id == "rollout-1"`。
3. **头条主张**：`with pytest.raises(local_ray.exceptions.RayActorError):
   local_ray.get(healthy.health.remote())`——真 `kill_actors` 把**健康**的那个 actor 也杀了，
   这正是解开 driver 阻塞的机制。

**现有 8 个测试全部保留**（pause/resume/skip-non-remote/thread-join 都便宜且合法可假）。

> **对审计的修正：** 审计的 proposal 写“Keep all seven existing tests”。实际是 **8 个**
> （`test_interval_zero_disables_the_monitor` / `..._probed_within_the_configured_timeout` /
> `..._fails_the_runtime_and_kills_the_fleet` / `..._stops_after_the_first_unreachable_worker` /
> `..._does_not_probe_parked_workers` / `test_resume_rearms_the_grace_period` /
> `test_stop_joins_the_thread` / `..._without_a_health_method_are_skipped`）。

审计实测 Part B：1 passed，call 0.60 s（其中 0.5 s 是故意的探针超时）。

**Verify：**
```bash
.venv/bin/python -m pytest tests/generation/ray/test_health_monitor.py -q -p no:randomly    # Part A
.venv/bin/python -m pytest tests/generation/ray -m slow_test -q -p no:randomly              # Part B
```

---

### 4.7 `GR-06` — `.remote()` 分支全仓零覆盖（本轨道的中心诚实缺口）

**位置：** `tests/generation/execution/test_chunk_memory_shadow.py:285-296`（`_probe_worker`）、
`:298-321`。

**今天的替身与断言：**

```python
# :285-296
def _probe_worker(worker_id: str, answer: int, calls: list[str]) -> Any:
    def probe(request: Any, *, max_samples: int) -> dict[str, Any]:
        calls.append(worker_id)
        return {"samples_per_chunk": answer, "budget_bytes": 32 * GB, "trials": []}
    return SimpleNamespace(worker_id=worker_id, actor=SimpleNamespace(probe_chunk_size=probe))
...
# :319-320
assert calls == ["w0", "w1"]
assert [req.sampling["samples_per_chunk"] for req in executed] == [4, 4]
```

**它诚实地证明了 local 分支。** 这**不是**假路径——生产**故意**支持普通 callable：

```python
# vrl/generation/ray/runtime.py:255-264
remote = getattr(probe, "remote", None)
if callable(remote):
    refs.append(remote(request, max_samples=max_samples))
else:
    local_results.append(probe(request, max_samples=max_samples))
if refs:
    ray = require_ray()
    local_results.extend(await asyncio.to_thread(ray.get, refs, timeout=600))
```

所以这个替身**合法**，`:298-321` 原样保留。**缺口是 `.remote()` + `ray.get(timeout=600)`
这条生产真正走的分支，全仓零覆盖**：这里没有，`test_rollout_launcher.py` 里也没有
（它的 `_worker_config` 设 `sync_trainable_state=False`，而且没有一个 launcher 测试调
`generate()`）。

**改法：采纳工作树里已经存在的那个探针，把它从一次性验证物升级成长期资产。**

`tests/generation/execution/test_zzscratch_probe_real.py`（未追踪，docstring 自称
“Scratch: real-Ray verification of the GR-06 proposal”）**就是这个测试，而且已经写对了**。
我今天跑过：**1 passed，call 0.25 s**（setup 2.06 s + teardown 2.07 s 是集群，摊销后归零）。
它做对了三件关键的事：

1. **带到达 barrier**（`_Arrivals` 跨 actor 计数 + 20 s deadline）。没有它，测试名里的
   “concurrently” 不成立——一个把 `ray.get` 挪进 dispatch 循环的实现会顺序探测而测试照样绿。
   有了它，那个实现会在 barrier 上 `TimeoutError("probes were dispatched sequentially")`。
2. **在 actor 进程内断言 wire 契约**：`isinstance(request, GenerationRequest)`、
   `request.sampling["samples_per_chunk"] == "auto"`、`request.inputs[0].prompt == "p"`。
   这顺带证明了 `GenerationRequest` 活过了 Ray 序列化——今天没有任何测试查这一点。
3. **fleet-min + probe-once**：`== [4, 4]` 与 `probe_calls == [1, 1]`。

**动作：**
- 把它移进 `tests/generation/ray/test_runtime_config.py`（那里已经拥有 `RayGenerationRuntime`
  的真 Ray 测试，也就自然落在 §3.2 的 package 薄壳下），改用 package 薄壳，去掉
  `try: import pytest` 那段（它是从 `test_ray_actor_pool.py` 抄来的，同样瞄错，见 §4.1）。
- 给 `:298-321` 的 local-branch 测试打 `real_cover` 指向它。
- **删掉 scratch 文件本身**——按 AGENTS.md 的一次性/长期资产规则，它的答案一旦落进
  canonical 路径与本文，scratch 副本就该消失。

> **诚实说明：** 审计给 GR-06 的 confidence 是 **medium**，成本是“~0.6 s（类比推算）”。
> 我实测 **0.25 s**，比推算便宜；而且原型已经在工作树里跑绿，所以本轨道把它按 **high**
> 对待。这是唯一一条 confidence 被上调的 finding。

**Verify：**
```bash
.venv/bin/python -m pytest tests/generation/ray tests/generation/execution/test_chunk_memory_shadow.py -q -p no:randomly
```

---

### 4.8 `CRD-05` — 一半是合法的 T3，一半把整段真链路跳过了

**位置：** `tests/ray/test_cross_node_preflight.py:17-42`、`tests/ray/test_dependencies.py:10-48`、
（可行性证据）`tests/ray/test_resources.py:559-647`。

两个替身，**裁定相反**：

**(1) `_ray(nodes)` / `_node(ip, gpu)` —— 合法的 T3-SCALE，保留。**

```python
# tests/ray/test_cross_node_preflight.py:17-22（与 test_dependencies.py:10-15 逐字重复）
def _ray(nodes):
    return SimpleNamespace(nodes=lambda: nodes)

def _node(ip, gpu, *, alive=True):
    return {"Alive": alive, "NodeManagerAddress": ip, "Resources": {"GPU": gpu}}
```

`inspect_cluster(ray, ...)` 把 ray 当参数注入，断言的是生产代码算出来的
`topo.driver_gpus` / `non_driver_gpus`（对 alive 的非 driver 节点求和），不是脚本返回值。
3 节点 2 GPU 的活集群在单元测试里造不出来。**保留 + 标注。**

**(2) `_resources(...)` —— 不合法，转真。**

```python
# tests/ray/test_cross_node_preflight.py:25-26
def _resources(*, rollout_num_gpus):
    return SimpleNamespace(rollout_num_gpus=rollout_num_gpus)
```

`cross_node_preflight(ray, resources: ResolvedDistributedResources)` 的第二个参数在生产里
永远是 `resolve_distributed_resources(cfg)` 的输出。用一个只有一个属性的 namespace 顶替，
**把「cross_node 配置 → 资源解析 → preflight」这条真链路整段跳过了**。

**改法（实测已跑通）：**

```python
def _resources():
    return resolve_distributed_resources(OmegaConf.create({"distributed": {"resources": {
        "visible_devices": "auto", "cross_node": True,
        "trainer": {"num_gpus": 1},
        "rollout": {"num_gpus": 1, "gpus_per_worker": 1, "num_workers": 1},
    }}}))
```

我今天实测：

```
first resolve  ms: 0.259
steady resolve ms: 0.128
rollout_num_gpus = 1 | type = ResolvedDistributedResources
driver-GPU case raises: cross_node rollout: the driver/head node exposes 1 Ray GPU(s), so roll…
accept case returns: None
```

在无 CUDA 依赖的机器上跑得通（`_auto_visible_cuda_devices` 在没 `torch.cuda` 时返回 `()`，
cross_node 走显式计数）。`tests/ray/test_resources.py:559`
`test_cross_node_rollout_satisfies_budget_from_explicit_counts` 已经在默认车道用同一条路径
断言 `rollout_devices == (1,)`——先例现成。

**同时修两处小账：**

- **去重。** `_ray` / `_node` 逐字重复两份，收进 `tests/ray/_helpers.py`
  （先例：`tests/trainers/online/_helpers.py`，已复核存在）。纯移动。
- **`:36-42` 通篇没有 assert**，靠不抛异常，读起来像空测试：
  ```python
  def test_preflight_non_hybrid_accepts_head_with_zero_gpus(monkeypatch):
      """Plain cross-node: head with --num-gpus=0 + enough remote GPUs passes."""
      monkeypatch.setattr(dependencies, "current_node_ip", lambda: "10.0.0.1")
      ray = _ray([_node("10.0.0.1", 0.0), _node("10.0.0.2", 1.0)])
      placement.cross_node_preflight(ray, _resources(rollout_num_gpus=1))
  ```
  实测 `cross_node_preflight` 返回 `None`，所以改成
  `assert placement.cross_node_preflight(ray, _resources()) is None`——显式，不需要注释约定。

**成本（对审计的修正）：** 审计写“真 resolve 0.9 ms，两条共 +1.8 ms”。我实测
**首次 0.26 ms、稳态 0.13 ms**，两条 **+0.3 ms**。**这是本轨道唯一进默认车道的成本。**

**Verify：**
```bash
.venv/bin/python -m pytest tests/ray/test_cross_node_preflight.py tests/ray/test_dependencies.py tests/ray/test_resources.py -q -p no:randomly
```

---

## NON-GOALS — 本轨道明确不做，逐条给理由

**环境性保留（`(b)` 类，永久合法）。**
CRD-06 那一类的判决在本区同样适用：**你没法“不拥有”一张 GPU**，也没法按需制造一次
600 秒的 PG ready 超时。以下替身**留着**，但按 tier 政策**必须被标注**——
**标注机制（marker 注册 + AST 守卫 + `--real-cover-report`）属于轨道一，本轨道只使用，
不重新定义**（见 `docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md` §3）。

- **`tests/ray/test_global_placement.py:384-434` / `:437-484`｜保留 + `real_cover`。**
  阻塞行具体：`vrl/ray/placement.py:48` 的 `_PLACEMENT_READY_TIMEOUT_S = 600.0`，
  在 `:267` 被 `ray.get(pg.ready(), timeout=...)` 使用。真等一次到期 10 分钟。**T3-ENV。**
- **`tests/ray/test_global_placement.py:487-525`（`_probe_partial_actor_construction`）｜保留。**
  它注入的是“第 2 个探针 actor 构造失败”，真集群上无法按需制造。**T3-ENV。**
- **`tests/ray/test_resource_cleanup.py:12-27` / `:30-45`｜保留。**
  按需失败的 `ray.kill` / `remove_placement_group`，同上。**轨道一已认领这两条的标注**，
  本轨道不重复。
- **`tests/ray/test_cross_node_preflight.py` / `test_dependencies.py` 的 `_ray`/`_node`｜保留 + `real_cover(None, ...)`。**
  活的多节点 Ray 拓扑在进程内造不出来（§4.8 (1)）。
- **`tests/ray/test_chunk_dispatch.py` 的 `_FakeRef` / `_FakeWorker`｜保留。**
  它们控制的是 **asyncio 完成顺序**，真 Ray 给不了确定性（审计实测 dynamic 分布 8 次跑出
  两种结果；这一条我未复测，按审计记录）。**这是受控时钟，不是 Ray 协议替身。**
- **`tests/generation/ray/test_weight_sync.py:100-120` 的 `ray.put_calls == [{"w": 1}]`｜保留 + `real_cover`。**
  全仓唯一钉住“一次 put 共享给 N 个 worker”的断言，真实端没有等价可观测量（§4.5）。
- **`tests/generation/ray/test_health_monitor.py` 的 `_FakeRay`｜保留（Part A 之后）。**
  它替的是 Ray 那条线；Part A 只是停止 monkeypatch 掉生产的 `kill_actors`。
- **`tests/generation/execution/test_chunk_memory_shadow.py:298-321` 的 local-branch 测试｜原样保留。**
  它覆盖的是**真生产路径**（普通 callable 分支），不是假路径。

**被复核推翻、明确不做的提案。**

- **不把 `_FakeRef` / `_FakeActor` 的三个 `execute` 测试换成真 Ray。** 它们钉的是确定性的
  完成顺序与绑定，真 Ray 给不了。本轨道**加**孪生，**不换**（§4.2）。
- **不重写 `tests/architecture/test_generation_rollout_boundaries.py` 已有的子串检查。**
  复核实测：`_forbidden_imports` 的 `_python_files` 走 `root.rglob("*.py")`，传**文件**路径
  返回 `[]`，AST 换法在两处会变成**永久绿的空断言**；而且它抓不到
  `from vrl.generation.execution import (...)` 这个真逃逸路径（`launcher.py:11` 正是这个形态，
  而 `vrl/generation/execution/__init__.py` re-export 了 `build_engine_plan`）。**与 Ray 无关，
  且提案是净负。不做。**
- **不给 `tests/config/test_load_all_experiments.py:893` 之类的无断言测试加“不抛即通过”注释。**
  复核实测全仓 38 个无断言测试里 37 个已经把保证写进名字（accepts / allows / is_noop / …），
  再加一层手写注释是同一事实的第三次复述、无机器校验、必然腐烂。本轨道对
  `test_cross_node_preflight.py:36` 的处理是**加真断言**（`is None`），不是加注释（§4.8）。
- **不动 `tests/rewards/**`、`tests/trainers/**`、`tests/models/**`、`tests/nn/**`。**
  RW-05 / RW-09 / RW-10 / TNA-14 的复核都各自有结论（有的推翻、有的改判），它们归 rewards
  轨与 tiny-real 轨。本轨道**只碰 Ray 机械与 Ray 断言路径**。
- **不改 `pyproject.toml`**（marker 注册是轨道一的动作）、**不改 `ci.yml`**、
  **不改 `vrl/` 一行**。本轨道是纯测试侧改动。
- **不删任何覆盖。** 两处删除都是死代码：`test_ray_actor_pool.py:5-8` + `:17`（§4.1 给出
  “防不住真实失败模式”的证据），以及 scratch 探针文件（其内容被移进 canonical 路径）。
  CRD-02 丢掉的 `remove_calls == [placement_group]` 换成了同一不变量的更强下游断言（§4.3）。
- **不把 `test_online_ray_cluster.py` 的 `isolated_ray` 折进共享 fixture。** 它的契约是
  “测试前后都没有 driver 连接”，与共享集群**方向相反**（§3.3）。

---

## HONEST GAPS — 本轨道标为“进程内未覆盖”的东西

**中心那一条：`GR-06` 的 `.remote()` 分支——本轨道把它从缺口变成覆盖。**

| 事实 | 今天 | 本轨道之后 | 车道 |
|---|---|---|---|
| `probe_chunk_size` 的 `.remote()` + `ray.get(timeout=600)` 扇出 | **全仓零覆盖**（`test_chunk_memory_shadow.py` 只走 local；`test_rollout_launcher.py` 从不调 `generate()`） | 新测试（采纳工作树探针，0.25 s） | `slow_test` |
| `GenerationRequest` 活过 Ray 序列化 | 无人检查 | 同上，actor 进程内断言 | `slow_test` |
| `RayGenerationExecutor.execute` 的 envelope→result 过线 | **全仓零真 Ray 覆盖**（3 个消费者全是 fake actor 或生产） | CRD-01 的 2 个执行器孪生 | `slow_test` |
| `ray.put` 一次共享后 worker 内的 auto-deref | 结构上不可测（`put()` 返回 tuple） | GR-02 (a)，actor 进程内 `isinstance(state_ref, dict)` | `slow_test` |
| 真 `ray.get(timeout=)` 到期 + 真 `RayActorError` 连坐整个 fleet | 只有手抛的 `TimeoutError` | GR-03 Part B，断言 `elapsed >= 0.5` | `slow_test` |
| 真 `kill_actors` 的遍历与 `no_restart=True` 传参 | 被 monkeypatch 掉 | GR-03 Part A | **默认车道** |
| 真 PG 被 `remove_placement_group` 移除、ownership 释放 | fake 的 `remove_calls` 记账 | CRD-02 | `slow_test` |
| cross_node 配置 → 资源解析 → preflight 整条链 | 被单属性 namespace 跳过 | CRD-05 | **默认车道** |

**保留为缺口、`real_cover(None, ...)`（没有真对位，本轨道也不建）：**

| 替身 | 阻塞点（具体行） | 有 e2e 对位吗 |
|---|---|---|
| `_ray` / `_node` 的多节点拓扑 | 活的 3 节点 / 2 GPU 集群在进程内造不出来；`inspect_cluster` 只接受被注入的 `ray` | **没有。** `tests/e2e/test_real_checkpoint_rl.py` 的 `CASES` 全是单节点 checkpoint 用例，没有一条跨节点。**这就是缺口本身，本文如实登记，不假装有覆盖。** |
| `_PLACEMENT_READY_TIMEOUT_S = 600.0` 到期（`test_global_placement.py:437-484`） | `vrl/ray/placement.py:48` → `:267` | **没有。** 真等一次 10 分钟，任何车道都不该付。 |
| `_create_raw_placement_group` 失败（`:384-434`）／探针 actor 部分构造失败（`:487-525`） | 真集群上无法按需制造这两种失败 | **没有。** |
| 真集群上一次会失败的 `ray.kill`（CRD-02 唯一保留的注入） | `ray.kill` 在健康集群上不会失败 | **没有**，但被它包住的其它协作者本轨道全部转真了。 |

**多节点 NCCL / 真 GPU 那一类不在本轨道范围内**，它们的登记在轨道一的 HONEST GAPS 表里。

---

## 顺序、依赖与合计

### 依赖轨道一的部分（只有标注，不是转换）

本轨道的 **4 处 `real_cover` 标注**（`test_chunk_dispatch.py` 新孪生指针、
`test_weight_sync.py:100-120`、`test_chunk_memory_shadow.py:298-321`、
`test_cross_node_preflight.py` / `test_dependencies.py` 的 `_ray`/`_node`）依赖轨道一的
marker 注册。**`pyproject.toml:203` 有 `--strict-markers`，未注册就写标注 = 整个文件
collection ERROR**（不是 warning）。

**但转换部分完全不依赖轨道一。** §4.1–§4.8 的机械、删除、真转换、新测试可以在轨道一之前
落地；标注作为一个后续小 PR 补上。**不要在轨道一之前的 PR 里写 `real_cover`。**

反向依赖：轨道一 §8.1 认领了 `test_global_placement.py:611` 的 `pytestmark_slow` 删除，
HONEST GAPS 表也认领了 `test_chunk_dispatch.py` 与 Ray 清理注入的标注。
**本轨道不重复这三处。**

### 落地顺序（按依赖，不按日历）

**必须串行的只有第 1 → 2，其余可并行。**

1. **落 `tests/conftest.py` 的 `real_local_ray()` contextmanager + function-scoped 薄壳。**
   零行为变化（现有 4 个消费者拿到同样的集群，参数从 `num_cpus=2` 变 `num_cpus=8, num_gpus=4`）。
   ```bash
   .venv/bin/python -m pytest tests/generation/ray -m slow_test -q -p no:randomly
   # 期望：8 passed，时长与 39.48 s 基线同量级
   ```
2. **`tests/ray/conftest.py` + `tests/generation/ray/conftest.py` 的 package 薄壳；
   4 处手搓 `ray.init` 与 launcher 的 `_init_ray` 全部改为消费它；删
   `test_ray_actor_pool.py:5-8` / `:17`。**
   **这一步和「给 launcher 补上 flag 还原」是同一次提交**——§0.1 已证明拆开会打红 nightly。
   同一次提交里还要带上 §3.3 的 `tests/scripts` 两处。
   ```bash
   .venv/bin/python -m pytest tests/ray -q -p no:randomly --durations=10          # 期望 134 passed / ~6.0 s
   .venv/bin/python -m pytest tests/generation/ray -m slow_test -q -p no:randomly # 期望 8 passed / ~12 s（方案 a）
   .venv/bin/python -m pytest tests/scripts/test_online_lifecycle.py tests/scripts/test_online_ray_cluster.py -q -p no:randomly
   # 关键回归：在 uv run 下也要绿（这是 CI 的调用形态）
   uv run --no-sync pytest -m "not e2e" tests/ray tests/generation/ray tests/scripts -q
   ```
3. **2 之后，以下四组可并行：**
   - **CRD-02** 转换（`test_global_placement.py`）
   - **CRD-01** 两个执行器孪生 + 文件头 over-claim 改写
   - **GR-02**（2 个新测试）、**GR-03 Part A + Part B**、**GR-06**（采纳探针 + 删 scratch）
   - **CRD-05**（真 resolver + `_helpers.py` 去重 + `is None` 断言）——**唯一进默认车道的一组，
     可以完全独立于 1/2 先做**
4. **标注 PR（等轨道一）：** 4 处 `real_cover`。
   ```bash
   .venv/bin/python -m pytest tests/architecture -q -p no:randomly      # 守卫必须绿
   .venv/bin/python -m pytest tests -q --real-cover-report              # 登记册里出现这 4 条，各带车道
   ```
5. **收尾门禁：**
   ```bash
   .venv/bin/python -m pytest tests -q -p no:randomly
   .venv/bin/python -m pytest -m "not e2e and not slow_test" -q
   .venv/bin/ruff check <touched> && .venv/bin/ruff format --check <touched>
   ```

### 合计

| 项 | 车道 | 实测/来源 |
|---|---|---|
| CRD-09 + CRD-01 的机械（`tests/ray` 共享集群） | `slow_test` | **−12.0 s**（本文 §1.1，3 次复测） |
| GR-01 的机械（`tests/generation/ray` 共享集群） | `slow_test` | **−7.9 s**（审计原型；上限 −13 s 见 §3.2，未原型） |
| CRD-02 转换 | `slow_test` | **+0.20 s** |
| CRD-01 执行器孪生 ×2 | `slow_test` | **+0.18 s**（审计 0.09 s/个） |
| GR-02 新测试 ×2 | `slow_test` | **+1.31 s**（审计 0.65 + 0.66 s） |
| GR-03 Part A | 默认 | **0 s** |
| GR-03 Part B | `slow_test` | **+0.60 s**（审计） |
| GR-06 采纳探针 | `slow_test` | **+0.25 s**（本文实测） |
| CRD-05 真 resolver ×2 | 默认 | **+0.3 ms**（本文实测） |
| **nightly slow lane 净** | | **−17.4 s** |
| **fast PR lane 净** | | **+0.3 ms** |
| **新增真 Ray 测试** | | **6 个**（CRD-02 1 + CRD-01 2 + GR-02 2 + GR-03 1 + GR-06 1，其中 GR-06 移入而非新写） |
| **删除** | | 2 处死代码（`test_ray_actor_pool.py:5-8/:17`；scratch 探针文件） |

**Definition of done：**
1. `pytest tests/ray -q` 的 `--durations` 里**不再出现任何 4–5 s 的 call**，且目录总时长
   ≤ 7 s（今天 18.0 s）。
2. `grep -rn "ray.init(" tests/ | grep -v conftest` 只剩 `test_online_ray_cluster.py`
   那条**故意**的（被测代码自己 init 的那个契约）。
3. **`grep -rn "RAY_ENABLE_UV_RUN_RUNTIME_ENV" tests/` 的每一处翻转都有配对的还原。**
4. `uv run --no-sync pytest -m "not e2e" tests/ray tests/generation/ray tests/scripts -q` 绿
   ——这是本轨道最重要的一条门禁，它今天**只在有泄漏时**才绿。

---

## 与现场的偏差（执行前必须先核对）

**HEAD 已从审计时移动到 `812cc3cf`。我逐条复核了 8 条 finding 引用的每一个路径与替身。**

| finding | 路径/替身 | 复核结果 |
|---|---|---|
| GR-01 | `tests/conftest.py:88-115` → **`:88-119`** | ✅ 存在，行号 +4 |
| GR-01 | `test_lifecycle_fsm.py:346` → **`:347`**；`test_runtime_config.py:301,325,347` → **`:302,326,348`** | ✅ 全部 +1 |
| CRD-09 | `test_ray_actor_pool.py:5-8, :17, :38/41-43/68, :88/90-92/136` | ✅ 行号**逐字命中** |
| CRD-09 | `test_global_placement.py:617-619,645 / 651-653,678` | ✅ 逐字命中 |
| CRD-01 | `test_chunk_dispatch.py:254-385`（`_FakeActor` :254、`_FakeRef` :37、测试 :316/:337/:364） | ✅ 逐字命中；文件头 over-claim 在 `:8-12`，逐字命中 |
| CRD-02 | `test_global_placement.py:528-601` 的 5 个 fake 类 + 4 个 monkeypatch + 自证断言 `:575` | ✅ 逐字命中 |
| CRD-05 | `test_cross_node_preflight.py:17-42`、`test_dependencies.py:10-48`、`test_resources.py:559` | ✅ 逐字命中，`_ray`/`_node` 确实重复两份 |
| GR-02 | `test_weight_sync.py:52-58` → **`:51-57`**；`:100-121` → **`:100-120`**；`:123-143` | ✅ 存在，行号 −1 |
| GR-03 | `test_health_monitor.py:76-85` → **`:76-84`**；`:118-144` → **`:118-142`**；`:146-160` → **`:145-160`** | ✅ 存在，行号 −1～−2 |
| GR-06 | `test_chunk_memory_shadow.py:283-296` → **`:285-296`**；`:298-321` | ✅ 存在，行号 +2 |

**没有一条 finding 的替身消失或被业主改掉。** 但有 **6 处数字/裁定要改**：

1. **CRD-01/CRD-09 的「占该目录全部时间的 52%」不成立了。** 今天 4 个测试的 call
   17.53 s / 目录 18.01 s = **97%**。审计的分母（~33 s）在 HEAD 上已经缩了。
   **省时结论不变（甚至更干净），份额说法要改。**
2. **CRD-02 的「共享集群上 0.05 s」偏低 4 倍。** 实测 **0.20 s**（3 次一致）。结论不变。
3. **GR-03 的「keep all seven existing tests」是 8 个。**
4. **CRD-09 的「这段 try/except 今天是死的」要改写成「瞄错了」。** 见 §4.1：worker 确实会
   import 本模块（Ray 按 module path 序列化 actor class，已用反证验证），但它防的
   `ModuleNotFoundError` 在真会触发的 `uv run` 下永远到不了——失败发生在 worker 启动阶段。
   **删除仍然正确，理由必须换。**
5. **GR-06 的成本从「~0.6 s（类比）」降到实测 0.25 s，confidence 从 medium 上调到 high**
   （原型已在工作树里跑绿）。
6. **CRD-05 的「真 resolve 0.9 ms」偏高 7 倍。** 实测首次 0.26 ms / 稳态 0.13 ms。

**新增一条审计没有的、比审计更严重的事实：** §0.1 的 uv-hook 泄漏是 CI 绿灯的必要条件。
它给本轨道加了一条硬排序约束，也把 `tests/scripts` 两处拉进了同一次提交的范围。

**工作树状态（`git status --porcelain`，复核时）：**

```
 M tests/generation/execution/test_execute_request_pipelined.py     <- 业主的，轨道三已认领，本轨道不碰
 M tests/models/families/flux/test_diffusion_nft_interface.py       <- 同上
?? tests/generation/execution/test_zzscratch_probe_real.py          <- GR-06 的探针；本轨道 §4.7 采纳并删除
?? docs/sprints/planned/SPRINT_*.md                                 <- 本系列 6 份
```

> 简报说工作树里有 `vrl/rewards/`、`vrl/families/registry.py`、`pyproject.toml` 的在飞编辑。
> **已过期**——那些改动已经进了 `812cc3cf` 之前的 12+ 个提交（`1b40b646` / `d4940ca8` /
> `89da5b52` / `02adae55` 等）。开工前重跑 `git status` 确认。

---

## References

- 判据与机制：`docs/sprints/done/SPRINT_tier-policy-and-real-cover-labels.md`（轨道一：
  marker 注册、AST 守卫、`--real-cover-report`、车道地图）
- 同系列：`SPRINT_zero-cost-real-object-swaps.md`（轨道三）、
  `SPRINT_tiny-real-diffusers-fixtures.md`、`SPRINT_reward-tiny-real-and-optional-lanes.md`、
  `SPRINT_docstring-truth-and-double-dedup.md`
- 标准与生命周期：`AGENTS.md`（Evidence-First Work / Dead Code Audit 五形态 /
  Long-term Assets vs One-shot Validation）、`docs/sprints/README.md`
- 车道定义：`tests/conftest.py:1-18`（docstring）、`:57-85`（`pytest_collection_modifyitems`）；
  marker 表 `pyproject.toml:200-211`；CI 车道 `.github/workflows/ci.yml:128-131`（PR）、
  `:163-166`（nightly）
- 被改的机械：`tests/conftest.py:88-119`（`local_ray`）、
  `tests/generation/ray/test_rollout_launcher.py:119-136`（`_init_ray`，flag 泄漏点在 `:128`）、
  `tests/scripts/test_online_lifecycle.py:32-42`、`tests/scripts/test_online_ray_cluster.py:26-30`
- 生产源：`vrl/ray/placement.py:48`（`_PLACEMENT_READY_TIMEOUT_S`）、`:240-292`（`create`）、
  `:435-474`（`_probe_gpu_bundles`）、`:108-114`（preflight 的 `num-gpus=0` 报错）；
  `vrl/ray/resource_cleanup.py:12-26`（`kill_actors`）；
  `vrl/generation/ray/weight_sync.py:50-69`（一次 `put` + `zip(strict=True)` 的 ack 归属）；
  `vrl/generation/ray/health_monitor.py:1-12`（不变量）、`:125-172`（`_run_probes` + kill）；
  `vrl/generation/ray/runtime.py:248-264`（probe 的 local / `.remote()` 双分支）
- 依赖内部（本文引用的唯一一处）：
  `.venv/lib/python3.12/site-packages/ray/_private/worker.py:1375-1397`
  （uv hook 在 `_skip_env_hook` **之前** return）、`ray/_private/ray_constants.py:546`
  （`RAY_ENABLE_UV_RUN_RUNTIME_ENV` 默认 `True`）
- 去重先例：`tests/trainers/online/_helpers.py`
- 复核用探针（一次性，答案已落进本文，用完即删）：
  `/tmp/claude-1000/-home-mingfeiguo-Desktop-VRL/3d48dcbf-816b-47e2-9eb3-237c62e9083f/scratchpad/probe_shared_ray.py`（共享集群原型，5 passed / 5.86–5.94 s）、
  `.../scratchpad/probe_uv_hook.py`（uv-hook 隔离实验）
