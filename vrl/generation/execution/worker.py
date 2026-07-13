"""Generation worker core independent of the Ray actor wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import GenerationChunkExecutor
from vrl.models.interfaces import require_runtime_model
from vrl.utils.config import import_from_path
from vrl.utils.cuda_memory import (
    CumemPool,
    gpu_used_bytes,
    release_cuda_memory,
    release_cuda_memory_for_parking,
)
from vrl.utils.logging import init_logger
from vrl.utils.profiling import TorchProfilerConfig

logger = init_logger(__name__)


class GenerationWorkerCore:
    """Own one generation executor and execute plan-aware chunks."""

    def __init__(
        self,
        worker_id: str,
        launch_contract: GenerationRuntimeLaunchContract | Mapping[str, Any],
        *,
        metadata_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.launch_contract = GenerationRuntimeLaunchContract.from_value(launch_contract)
        self.family = self.launch_contract.family
        self.executor: GenerationChunkExecutor | None = None
        self._policy_version: int | None = self.launch_contract.policy_version
        # Flipped on once a model that supports versioned trainable-state slots
        # receives its first weight install; from then on execute_chunk activates
        # the slot for each request's stamped version instead of comparing against
        # one global version (which is what makes a non-draining sync safe).
        self._uses_versioned_slots = False
        profiler_config = self.launch_contract.extra.get("torch_profiler", {})
        self._profiler_config = (
            TorchProfilerConfig(**dict(profiler_config))
            if isinstance(profiler_config, Mapping)
            else TorchProfilerConfig()
        )
        self._profiler_output_dir = str(
            self.launch_contract.extra.get("profiler_output_dir", "outputs/"),
        )
        self._profiler_step = 0
        self._metadata_provider = metadata_provider or self._fallback_metadata
        # Set to a CumemPool when load_policy pools the model for sleep-offload
        # (extra["sleep_offload"]); sleep()/wake() then release/restore GPU physical
        # pages through it instead of the naive to(cpu)/to(gpu) round trip. vLLM's
        # own "weights" tag = the offloaded (not discarded) slice of a sleeping
        # engine, keeping semantics identical to verl-omni's level-1.
        self._cumem: CumemPool | None = None
        # The pre-load physical footprint is the only valid residual baseline;
        # CUDA contexts and shared libraries make zero incorrect in general.
        self._preload_gpu_used_bytes: int | None = None
        # CPU offload needs one local state value: the original device is both the
        # wake target and proof that the two-part model move committed.
        self._cpu_parked_device: Any | None = None

    def load_policy(self) -> None:
        """Build the family executor from the serialized launch contract."""

        if self.executor is not None:
            if self._parking_requested():
                self._require_complete_parking_backend()
            return
        from vrl.utils.memory import log_host_memory

        if self._parking_requested():
            self._preload_gpu_used_bytes = self._gpu_used_bytes()
        self._cpu_parked_device = None
        log_host_memory(f"generation_worker:{self.worker_id}:before_load_policy", log=logger)
        try:
            self.executor = self._build_executor_maybe_pooled()
            if (
                self.executor.family != self.launch_contract.family
                or self.executor.task != self.launch_contract.task
            ):
                raise ValueError(
                    "executor identity does not match launch contract: "
                    f"{self.executor.family}/{self.executor.task} != "
                    f"{self.launch_contract.family}/{self.launch_contract.task}",
                )
            if self._parking_requested():
                self._require_complete_parking_backend()
        except BaseException as load_error:
            try:
                self.release_policy()
            except BaseException as release_error:
                raise RuntimeError(
                    "generation policy load and cleanup both failed: "
                    f"load={load_error!r}; cleanup={release_error!r}",
                ) from release_error
            raise
        log_host_memory(f"generation_worker:{self.worker_id}:after_load_policy", log=logger)

    def _build_executor_maybe_pooled(self) -> GenerationChunkExecutor:
        """Build the executor, pooling the model in CuMemAllocator when requested.

        A sleep-offload diffusion worker (``extra["sleep_offload"]``, set by the
        colocated-trainer lease) allocates the whole model inside vLLM's
        CuMemAllocator pool, so ``sleep``/``wake`` release and restore the GPU
        *physical* pages via CUDA virtual memory. AR remains on the verified CPU
        fallback because GLM-Image's temporary decode path calls ``empty_cache``;
        doing that inside vLLM's pluggable allocator scope is unsupported. The
        worker rejects families or executor shapes that have not declared and
        implemented complete fallback parking instead of accepting a no-op.
        """

        if not self._parking_requested() or self.launch_contract.generation_kind != "diffusion":
            return self._build_executor()
        pool = CumemPool.try_create(tag="weights")
        if pool is None:
            return self._build_executor()
        try:
            with pool.building():
                executor = self._build_executor()
        except BaseException as build_error:
            release_cuda_memory(gc_collect=True, ipc_collect=True)
            try:
                pool.close()
            except BaseException as close_error:
                raise RuntimeError(
                    "generation pooled build and cleanup both failed: "
                    f"build={build_error!r}; cleanup={close_error!r}",
                ) from close_error
            raise
        self._cumem = pool
        return executor

    def release_policy(self) -> None:
        """Drop loaded model state so the worker releases CUDA memory before exit."""

        pool = self._cumem
        if pool is not None and pool.asleep:
            # Tagged tensors must be mapped before they are destroyed; otherwise
            # the allocator keeps their CPU backups and retained MemPool alive.
            pool.wake()
        self.executor = None
        release_cuda_memory(gc_collect=True, ipc_collect=True)
        if pool is not None:
            pool.close()
            self._cumem = None
            release_cuda_memory(gc_collect=True, ipc_collect=True)
        self._preload_gpu_used_bytes = None
        self._cpu_parked_device = None

    def sleep(self) -> WorkerMemoryParkingSnapshot:
        """Offload the loaded model to host RAM, freeing the GPU without discarding it.

        Level-1 offload-and-restore (cf. vLLM ``sleep_level=1``): unlike
        ``release_policy`` (level-2 discard → ``wake``/``load_policy`` cold-reloads
        the frozen VAE / text-encoders from disk), this keeps the executor and its
        loaded model alive, parking *all* weights on CPU so a colocated trainer
        reclaims the GPU for its step. ``wake`` then restores them from host RAM
        without re-reading the checkpoint. ``nn.Module.to`` moves only registered
        submodules (the transformer), so the diffusers pipeline's unregistered
        frozen components ride the same ``move_frozen_components`` hook the
        in-process driver-offload path uses. No-op when nothing is loaded.
        """

        if self.executor is None:
            if self._parking_requested():
                raise RuntimeError(
                    f"generation worker {self.worker_id!r} cannot park before policy load",
                )
            used_bytes = self._gpu_used_bytes()
            snapshot = WorkerMemoryParkingSnapshot(
                worker_id=self.worker_id,
                backend="cpu_only",
                baseline_gpu_used_bytes=used_bytes,
                loaded_gpu_used_bytes=used_bytes,
                residual_gpu_used_bytes=used_bytes,
                residual_bytes_limit=0,
            )
            snapshot.validate()
            return snapshot
        if self._parking_requested():
            self._require_complete_parking_backend()
        # This is provenance for the operation being attempted, not durable
        # worker state. Sampling here includes generation-time caches allocated
        # after load_policy() returned.
        loaded_bytes = self._gpu_used_bytes()
        pool = self._cumem
        if pool is not None:
            # cumem owns the whole pooled model (transformer + frozen components);
            # one call offloads every tagged allocation to host RAM and releases the
            # physical pages, no per-module .to() round trip needed.
            if not pool.asleep:
                pool.sleep()
            self._release_cuda_memory_for_parking()
            backend = "cumem"
        else:
            model = self.executor.model
            restore_device = self._cpu_parked_device
            if restore_device is None:
                # Capture before the CPU move: afterwards model.device can no
                # longer tell wake() which CUDA device owned the model.
                restore_device = self._executor_device(self.executor)
                move_frozen = getattr(model, "move_frozen_components", None)
                try:
                    model.to("cpu")
                    if callable(move_frozen):
                        move_frozen("cpu")
                except BaseException as move_error:
                    # Do not commit a parked sentinel until BOTH registered and
                    # unregistered modules moved. Best-effort rollback keeps a
                    # retry from reporting a false successful handoff.
                    rollback_errors: list[BaseException] = []
                    try:
                        model.to(restore_device)
                    except BaseException as rollback_error:
                        rollback_errors.append(rollback_error)
                    if callable(move_frozen):
                        try:
                            move_frozen(restore_device)
                        except BaseException as rollback_error:
                            rollback_errors.append(rollback_error)
                    if rollback_errors:
                        raise RuntimeError(
                            "generation CPU parking and rollback both failed: "
                            f"move={move_error!r}; rollback={rollback_errors!r}",
                        ) from rollback_errors[0]
                    raise
                self._cpu_parked_device = restore_device
            self._release_cuda_memory_for_parking()
            backend = "cpu_offload" if str(restore_device).startswith("cuda") else "cpu_only"

        residual_bytes = self._gpu_used_bytes()
        baseline_bytes = self._preload_gpu_used_bytes
        if baseline_bytes is None:
            if self._parking_requested() or residual_bytes:
                raise RuntimeError(
                    f"generation worker {self.worker_id!r} has no pre-load GPU "
                    "parking baseline; load it with sleep_offload enabled",
                )
            baseline_bytes = 0
        snapshot = WorkerMemoryParkingSnapshot(
            worker_id=self.worker_id,
            backend=backend,
            baseline_gpu_used_bytes=baseline_bytes,
            loaded_gpu_used_bytes=loaded_bytes,
            residual_gpu_used_bytes=residual_bytes,
        )
        snapshot.validate()
        return snapshot

    def wake(self) -> None:
        """Restore a slept model from host RAM onto its GPU (no disk reload).

        The counterpart to ``sleep``. If the executor was evicted instead of slept
        (e.g. a hard release happened in between), rebuild it via ``load_policy`` so
        a wake is always safe to call.
        """

        model = getattr(self.executor, "model", None)
        if model is None:
            self.load_policy()
            return
        pool = self._cumem
        if pool is not None:
            # Restore the pooled model's physical pages from host RAM at the same
            # virtual addresses — the tensors stay valid cuda tensors throughout.
            pool.wake()
            return
        device = self._cpu_parked_device
        if device is None:
            return
        move = getattr(model, "to", None)
        if callable(move):
            move(device)
        move_frozen = getattr(model, "move_frozen_components", None)
        if callable(move_frozen):
            move_frozen(device)
        self._cpu_parked_device = None

    def _parking_requested(self) -> bool:
        return bool(self.launch_contract.extra.get("sleep_offload"))

    def _require_complete_parking_backend(self) -> None:
        """Validate the concrete executor path before a shared-GPU run proceeds."""

        if self._cumem is not None:
            return
        executor = self.executor
        model = getattr(executor, "model", None)
        if model is None or not callable(getattr(model, "to", None)):
            raise RuntimeError(
                f"generation worker {self.worker_id!r} cannot completely park "
                f"{type(executor).__name__}: CPU fallback requires executor.model.to(...)"
            )
        if self.launch_contract.generation_kind == "diffusion" and not callable(
            getattr(model, "move_frozen_components", None),
        ):
            raise RuntimeError(
                f"generation worker {self.worker_id!r} cannot completely park "
                f"{type(model).__name__}: diffusion CPU fallback requires "
                "move_frozen_components(...)"
            )

    # Instance-assignable test seams over shared parking bookkeeping.
    @classmethod
    def _gpu_used_bytes(cls) -> int:
        return gpu_used_bytes()

    @classmethod
    def _release_cuda_memory_for_parking(cls) -> None:
        return release_cuda_memory_for_parking()

    def update_weights(self, state_ref: Any, policy_version: int) -> int:
        """Install weights and return the policy version as the commit ACK.

        When the model supports versioned trainable-state slots, install the new
        version as a retained slot WITHOUT overwriting the slots older in-flight
        requests still depend on (the non-draining-sync path); ``execute_chunk``
        then activates the right slot per request. Otherwise keep the single
        in-place overwrite (the draining-barrier path).
        """

        self.load_policy()
        policy_obj = getattr(self.executor, "model", None)
        if bool(getattr(policy_obj, "supports_versioned_trainable_state", False)):
            model = require_runtime_model(
                policy_obj,
                owner=f"{type(self.executor).__name__}.model",
            )
            model.install_trainable_state(int(policy_version), state_ref)
            self._uses_versioned_slots = True
        elif state_ref is not None:
            model = require_runtime_model(
                policy_obj,
                owner=f"{type(self.executor).__name__}.model",
            )
            model.load_trainable_state(state_ref)
        # The single scalar now means "current submit version": current_policy_version()
        # is what the producer reads to stamp NEW requests, so it must track the
        # latest installed version even in slot mode. Per-chunk results take their
        # version from request.policy_version, not this field.
        self._policy_version = int(policy_version)
        return self._policy_version

    def current_policy_version(self) -> int | None:
        return self._policy_version

    def supports_versioned_trainable_state(self) -> bool:
        """Whether the loaded model can retain versioned trainable-state slots.

        Drives the runtime's non-draining-weight-sync capability. Requires the
        model to be built, so callers query it after at least one weight sync.
        """

        self.load_policy()
        model = getattr(self.executor, "model", None)
        return bool(getattr(model, "supports_versioned_trainable_state", False))

    def worker_metadata(self, *, runtime_debug: bool = False) -> dict[str, Any]:
        try:
            metadata = dict(self._metadata_provider())
        except Exception:
            metadata = self._fallback_metadata()
        metadata.setdefault("worker_id", self.worker_id)
        metadata.setdefault("node_ip", "unknown")
        metadata.setdefault("gpu_ids", [])
        metadata["policy_version"] = self._policy_version
        if runtime_debug:
            metadata["runtime_debug"] = True
            metadata["executor_type"] = type(self.executor).__name__ if self.executor else None
            policy = getattr(self.executor, "model", None) if self.executor is not None else None
            if policy is not None:
                from vrl.utils.model_diagnostics import (
                    parameter_state_summary,
                    trainable_state_digest,
                )

                metadata["trainable_state"] = trainable_state_digest(policy)
                metadata["parameter_state"] = parameter_state_summary(policy)
                metadata["policy_type"] = type(policy).__name__
        return metadata

    def execute_chunk(self, envelope: ChunkExecutionEnvelope) -> ChunkExecutionResult:
        self.load_policy()
        request = envelope.request
        chunk = envelope.chunk
        runtime_debug = bool(request.metadata.get("_runtime_debug"))
        expected_version = request.policy_version
        model = getattr(self.executor, "model", None)
        if self._uses_versioned_slots and expected_version is not None:
            # Slot mode: serve the request from the slot for ITS stamped version,
            # not the worker's latest. A missing slot means the request outlived
            # the retention window — a typed stale-slot result, not a mismatch.
            if not model.has_trainable_state(expected_version):
                return ChunkExecutionResult(
                    request_id=request.request_id,
                    worker_id=self.worker_id,
                    chunk=chunk,
                    output=None,
                    metrics=self._chunk_metrics(envelope, runtime_debug=runtime_debug),
                    policy_version=expected_version,
                    error=(f"trainable-state slot evicted for policy_version={expected_version}"),
                    stale_slot=True,
                )
        elif expected_version is not None and self._policy_version != expected_version:
            return ChunkExecutionResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                metrics=self._chunk_metrics(envelope, runtime_debug=runtime_debug),
                policy_version=self._policy_version,
                error=(
                    "policy_version mismatch: "
                    f"expected={expected_version}, actual={self._policy_version}"
                ),
            )
        # In slot mode the result must carry the REQUEST's version (the executor
        # asserts result.policy_version == request.policy_version), which is how an
        # old in-flight request keeps passing after the worker installs a newer slot.
        result_version = expected_version if expected_version is not None else self._policy_version
        try:
            assert self.executor is not None
            if self._uses_versioned_slots and expected_version is not None:
                model.activate_trainable_state(expected_version)
            output = self._profile_forward_chunk(envelope)
            return ChunkExecutionResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=self._to_cpu(output),
                metrics=self._chunk_metrics(
                    envelope,
                    runtime_debug=runtime_debug,
                    chunk_output=output,
                ),
                policy_version=result_version,
            )
        except Exception as exc:
            return ChunkExecutionResult(
                request_id=request.request_id,
                worker_id=self.worker_id,
                chunk=chunk,
                output=None,
                metrics=self._chunk_metrics(envelope, runtime_debug=runtime_debug),
                policy_version=result_version,
                error=str(exc),
            )

    def probe_chunk_size(
        self,
        request: Any,
        *,
        max_samples: int,
        execute_steps: int = 2,
        margin: float = 0.05,
        knee_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Startup chunk-size probe (SPRINT_chunk_size_probe): pick the largest
        safe ``samples_per_chunk`` for this worker by running truncated real
        chunks — vLLM's profile-run shape, adapted to a chunked rollout.

        Runs BEFORE the first real request (caller contract). Trials at n=1 and
        n=min(4, max) give a two-point affine fit of peak bytes (demand is
        affine in n); the fitted candidate is then CONFIRMED with one real trial
        because the allocator layer (segment rounding, fragmentation) is not.
        The rollout owns its GPU for this phase, so the probe budgets against the
        device total rather than instantaneous free memory. Trial timing feeds a
        knee rule: growth that no longer improves ms/sample is refused (no memory
        risk for a flat throughput return).
        Probe outputs are discarded; trainable state / policy_version untouched.
        """

        import time
        from dataclasses import replace as dataclass_replace

        import torch

        from vrl.generation.execution.chunk_placement import (
            AffinePeakFit,
            ChunkMemoryReading,
        )
        from vrl.generation.execution.chunks import SampleChunk
        from vrl.utils.profiling import record_function

        if not torch.cuda.is_available():
            raise RuntimeError("chunk-size probe requires CUDA")
        if max_samples < 1:
            raise ValueError(f"probe max_samples must be >= 1, got {max_samples}")
        self.load_policy()
        executor = self.executor
        for method in (
            "build_prompt_stage_input",
            "run_prompt_encode_stage",
            "run_prepare_stage",
            "run_denoise_stage",
            "run_decode_stage",
        ):
            if not callable(getattr(executor, method, None)):
                raise TypeError(
                    f"{type(executor).__name__} does not expose the diffusion "
                    "chunk stages; samples_per_chunk: auto is diffusion-only",
                )

        _, total_bytes = torch.cuda.mem_get_info()
        budget_bytes = int(total_bytes)

        def run_trial(n: int, *, timed_label: str) -> dict[str, Any]:
            probe_sampling = {**dict(request.sampling), "samples_per_chunk": n}
            probe_request = dataclass_replace(
                request,
                request_id=f"chunk-probe-{self.worker_id}-n{n}",
                inputs=[request.inputs[0]],
                samples_per_prompt=n,
                sampling=probe_sampling,
            )
            chunk = SampleChunk(
                prompt_index=0,
                prompt=probe_request.prompts[0],
                sample_start=0,
                sample_count=n,
            )
            stage_durations: dict[str, float] = {}
            started = time.perf_counter()
            try:
                # The four stage methods ARE forward_chunk_plan's body; only the
                # wire-storage step is skipped because the output is discarded.
                prompt_input = executor.build_prompt_stage_input(probe_request, chunk)
                prompt_output = executor.run_prompt_encode_stage(
                    prompt_input,
                    stage_durations=stage_durations,
                    record_function=record_function,
                )
                prepared = executor.run_prepare_stage(
                    prompt_output,
                    stage_durations=stage_durations,
                )
                prepared.config = dataclass_replace(
                    prepared.config,
                    execute_steps=execute_steps,
                )
                denoised = executor.run_denoise_stage(
                    prepared,
                    stage_durations=stage_durations,
                )
                chunk_result = executor.run_decode_stage(denoised)
                # CUDA work is async-launched; without a sync here the wall
                # time of one trial leaks into the next and the knee rule
                # compares garbage (observed: n=4 charged 47s, n=16 1.5s).
                torch.cuda.synchronize()
            except Exception as exc:  # OOM is an expected trial verdict
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return {"n": n, "oom": True, "label": timed_label}
            wall_s = time.perf_counter() - started
            memory = chunk_result.memory
            reading = ChunkMemoryReading.from_metrics(memory) if memory is not None else None
            del chunk_result, denoised, prepared, prompt_output, prompt_input
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if reading is None:
                raise RuntimeError(
                    "chunk-size probe trial produced no memory reading "
                    f"(n={n}); cannot size chunks without it",
                )
            return {
                "n": n,
                "oom": False,
                "label": timed_label,
                "peak_bytes": reading.peak_bytes,
                "non_torch_bytes": reading.non_torch_bytes,
                "wall_s": wall_s,
                "per_sample_s": wall_s / n,
            }

        trials: list[dict[str, Any]] = []
        # Warmup at n=1 (cudnn autotune, lazy init) so trial timings compare
        # warm-vs-warm; its memory verdict still counts: OOM at n=1 is terminal.
        warmup = run_trial(1, timed_label="warmup")
        if warmup["oom"]:
            raise RuntimeError(
                "chunk-size probe: a single sample does not fit on this worker "
                f"(phase budget {budget_bytes / 2**30:.1f} GiB); the recipe "
                "shape is too large for this GPU",
            )
        trials.append(warmup)
        low = run_trial(1, timed_label="fit-low")
        trials.append(low)
        final = 1

        if max_samples > 1:
            n_high = min(4, max_samples)
            high = run_trial(n_high, timed_label="fit-high")
            trials.append(high)
            if high["oom"]:
                # The fit anchor itself OOMed: bisect between the known-good 1
                # and n_high for the largest fitting n.
                final = self._bisect_chunk_probe(run_trial, trials, 1, n_high)
            else:
                usable_bytes = int(
                    budget_bytes * (1.0 - margin) - high["non_torch_bytes"],
                )
                fit = AffinePeakFit.from_trials(
                    1,
                    low["peak_bytes"],
                    n_high,
                    high["peak_bytes"],
                )
                candidate = max(1, min(fit.max_samples_within(usable_bytes), max_samples))
                final = n_high if candidate >= n_high else candidate
                if candidate > n_high:
                    confirm = run_trial(candidate, timed_label="confirm")
                    trials.append(confirm)
                    if confirm["oom"]:
                        final = self._bisect_chunk_probe(
                            run_trial,
                            trials,
                            n_high,
                            candidate,
                        )
                    else:
                        final = candidate
                        # Knee rule: growing past n_high must still buy
                        # throughput, otherwise the extra memory risk is free.
                        improvement = 1.0 - (confirm["per_sample_s"] / high["per_sample_s"])
                        if improvement < knee_threshold:
                            final = n_high
        return {
            "samples_per_chunk": int(final),
            "budget_bytes": budget_bytes,
            "trials": trials,
        }

    @staticmethod
    def _bisect_chunk_probe(
        run_trial: Any,
        trials: list[dict[str, Any]],
        low_good: int,
        high_bad: int,
    ) -> int:
        """Largest fitting n in (low_good, high_bad): each trial is seconds."""

        while high_bad - low_good > 1:
            mid = (low_good + high_bad) // 2
            trial = run_trial(mid, timed_label="bisect")
            trials.append(trial)
            if trial["oom"]:
                high_bad = mid
            else:
                low_good = mid
        return low_good

    def execute_request_pipelined(
        self,
        request: Any,
        engine_plan: Any,
        sample_rows: Any,
    ) -> Any:
        """Run ALL of a request's chunks through the executor's software pipeline
        (``forward_plan_pipelined``) on THIS worker, so chunk N+1's denoise overlaps
        chunk N's GPU->CPU copy + host packing — hiding the per-chunk worker
        boundary that per-chunk dispatch leaves serial. Single-worker only (all the
        request's chunks must be here). Returns the gathered ``GenerationOutput``.

        Version safety mirrors ``execute_chunk`` but at the REQUEST level (every
        chunk shares ``request.policy_version``): slot mode serves the request from
        its stamped version's slot (stale slot -> ``StaleSlotDiscard`` so the
        producer counts a graceful discard, never an off-policy train); non-slot
        mode rejects a version mismatch. The version is checked + activated ONCE
        here, not per chunk.
        """

        from vrl.generation.execution.types import StaleSlotDiscard

        self.load_policy()
        expected_version = request.policy_version
        model = getattr(self.executor, "model", None)
        if self._uses_versioned_slots and expected_version is not None:
            if not model.has_trainable_state(expected_version):
                raise StaleSlotDiscard(
                    f"trainable-state slot evicted for policy_version={expected_version}",
                )
            model.activate_trainable_state(expected_version)
        elif expected_version is not None and self._policy_version != expected_version:
            raise RuntimeError(
                "policy_version mismatch: "
                f"expected={expected_version}, actual={self._policy_version}",
            )
        assert self.executor is not None
        forward_plan_pipelined = getattr(self.executor, "forward_plan_pipelined", None)
        if not callable(forward_plan_pipelined):
            raise TypeError(
                f"{type(self.executor).__name__} must implement forward_plan_pipelined(...) "
                "for per-request pipelined execution",
            )
        return forward_plan_pipelined(request, sample_rows, engine_plan)

    def _profile_forward_chunk(
        self,
        envelope: ChunkExecutionEnvelope,
    ) -> Any:
        from vrl.utils.profiling import capture_torch_trace, profile_range

        assert self.executor is not None
        request = envelope.request
        chunk = envelope.chunk
        step = self._profiler_step
        self._profiler_step += 1
        worker_name = (
            f"{self.worker_id}_{self.family}_{self.launch_contract.task}_"
            f"policy{self._policy_version}_chunk{chunk.prompt_index}_{chunk.sample_start}"
        )
        try:
            device = self._executor_device(self.executor)
            forward_chunk_plan = getattr(self.executor, "forward_chunk_plan", None)
            if not callable(forward_chunk_plan):
                raise TypeError(
                    f"{type(self.executor).__name__} must implement "
                    "forward_chunk_plan(...) for distributed chunk execution",
                )
            with (
                capture_torch_trace(
                    self._profiler_config,
                    output_dir=self._profiler_output_dir,
                    step=step,
                    device=device,
                    worker_name=worker_name,
                    trace_subdir=f"generation/{self.worker_id}",
                ),
                profile_range("engine.forward_chunk"),
            ):
                return forward_chunk_plan(request, chunk)
        except Exception:
            logger.exception("generation chunk execution failed")
            raise

    def _chunk_metrics(
        self,
        envelope: ChunkExecutionEnvelope,
        *,
        runtime_debug: bool,
        chunk_output: Any | None = None,
    ) -> dict[str, Any]:
        metrics = self.worker_metadata(runtime_debug=runtime_debug)
        metrics.update(
            {
                "chunk_key": envelope.chunk_key,
            },
        )
        # Byte-admission shadow reading crosses the wire unconditionally (a dozen
        # ints per chunk): calibration data must accrue from every real run, not
        # only runtime_debug ones. The heavyweight debug payload stays gated.
        chunk_memory = getattr(chunk_output, "memory", None)
        if isinstance(chunk_memory, Mapping):
            metrics["chunk_memory"] = dict(chunk_memory)
        if runtime_debug and chunk_output is not None:
            metrics.update(_chunk_output_debug_metrics(chunk_output))
        return metrics

    def _build_executor(self) -> GenerationChunkExecutor:
        launch_contract = self.launch_contract
        builder_path = launch_contract.runtime_builder
        executor_path = launch_contract.executor_cls
        if builder_path is None or executor_path is None:
            raise ValueError(
                "GenerationRuntimeLaunchContract requires runtime_builder and "
                "executor_cls import paths",
            )

        from vrl.models.interfaces.runtime import ModelBuild

        build_runtime_bundle = import_from_path(str(builder_path))
        executor_cls = import_from_path(str(executor_path))
        build_payload = launch_contract.model_build_payload()
        device = build_payload.get("device")
        if isinstance(device, str):
            import torch

            build_payload["device"] = torch.device(device)
        # ModelBuild and RolloutBuildOptions normalize parameter and nested rollout
        # dtypes while reconstructing the primitive Ray payload.
        build = ModelBuild(**build_payload)
        bundle = build_runtime_bundle(build)
        model = require_runtime_model(bundle.model, owner="RuntimeBundle.model")
        # Family- and scheme-agnostic backstop: if rollout quantization asks for a
        # quantized rollout (FP8/NVFP4/...) but this family's builder forgot to swap,
        # the model would silently run at its base dtype — fail loudly instead.
        from vrl.models.loader import assert_rollout_quantization_applied

        assert_rollout_quantization_applied(model, build)
        built = executor_cls(model, **dict(launch_contract.executor_kwargs))
        return _require_chunked_executor(built)

    @staticmethod
    def _executor_device(executor: Any) -> Any:
        policy = getattr(executor, "model", None)
        device = getattr(policy, "device", None)
        if device is not None:
            return device
        try:
            import torch

            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            return "cpu"

    @classmethod
    def _to_cpu(cls, value: Any) -> Any:
        """Move a chunk output to CPU with pinned, queued copies.

        Per-tensor ``.cpu()`` synchronizes the stream once per tensor (~376ms
        of cudaStreamSynchronize per chunk measured on the wire-diet profile);
        pinned-buffer ``non_blocking`` copies queue every transfer and the
        device synchronizes once before the payload is handed to Ray.
        """

        # Lazy import: this module stays torch-free at import time, and the
        # trajectory package (the walker's home) pulls torch transitively.
        from vrl.trajectory.device import map_tensor_tree

        pending = {"cuda_copies": False}

        def _pinned_copy(leaf: Any) -> Any:
            tensor = leaf.detach()
            if getattr(tensor, "is_cuda", False):
                import torch

                host = torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host.copy_(tensor, non_blocking=True)
                pending["cuda_copies"] = True
                return host
            return tensor.cpu()

        copied = map_tensor_tree(value, _pinned_copy, is_leaf=cls._is_tensor)
        if pending["cuda_copies"]:
            import torch

            torch.cuda.synchronize()
        return copied

    @staticmethod
    def _is_tensor(value: Any) -> bool:
        return hasattr(value, "detach") and hasattr(value, "cpu")

    def _fallback_metadata(self) -> dict[str, Any]:
        return {"worker_id": self.worker_id, "node_ip": "unknown", "gpu_ids": []}


def _require_chunked_executor(executor: Any) -> GenerationChunkExecutor:
    forward_chunk_plan = getattr(executor, "forward_chunk_plan", None)
    gather_chunks = getattr(executor, "gather_chunks", None)
    if not callable(forward_chunk_plan) or not callable(gather_chunks):
        raise TypeError(
            f"{type(executor).__name__} does not implement "
            "forward_chunk_plan(...) and gather_chunks(...)",
        )
    return executor


def _chunk_output_debug_metrics(output: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    stage_durations = getattr(output, "stage_durations", None)
    if isinstance(stage_durations, Mapping):
        metrics["stage_durations_s"] = {
            str(key): float(value) for key, value in stage_durations.items()
        }

    engine_counters = getattr(output, "engine_counters", None)
    if isinstance(engine_counters, Mapping):
        metrics["engine_counters"] = _debug_metric_value(dict(engine_counters))

    peak_memory_mb = getattr(output, "peak_memory_mb", None)
    if peak_memory_mb is not None:
        metrics["peak_memory_mb"] = float(peak_memory_mb)

    return metrics


def _debug_metric_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _debug_metric_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_metric_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return repr(value)


__all__ = ["GenerationWorkerCore"]
