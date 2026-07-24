"""Generation worker core independent of the Ray actor wrapper."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from typing import Any

from vrl.generation.execution.memory_parking import WorkerMemoryParking
from vrl.generation.execution.types import (
    ChunkExecutionEnvelope,
    ChunkExecutionResult,
    PipelinedRequestOutOfMemory,
    WorkerMemoryParkingSnapshot,
)
from vrl.generation.launch_contract import GenerationRuntimeLaunchContract
from vrl.generation.protocols import DiffusionStagedChunkExecutor, GenerationChunkExecutor
from vrl.generation.types import GenerationOutput
from vrl.models.interfaces import require_runtime_model
from vrl.utils.config import import_from_path
from vrl.utils.cuda_memory import is_cuda_out_of_memory, release_cuda_memory
from vrl.utils.logging import init_logger
from vrl.utils.profiling import TorchProfilerConfig

logger = init_logger(__name__)

# The chunk-size probe truncates each trial to a fixed handful of denoise steps:
# it only needs the peak-memory shape (affine in samples), not a full sampling.
_PROBE_EXECUTE_STEPS = 2


class GenerationWorkerCore:
    """Own one generation executor and execute plan-aware chunks."""

    def __init__(
        self,
        worker_id: str,
        launch_contract: GenerationRuntimeLaunchContract,
        *,
        metadata_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.worker_id = worker_id
        if not isinstance(launch_contract, GenerationRuntimeLaunchContract):
            raise TypeError(
                "launch_contract must be a GenerationRuntimeLaunchContract, "
                f"got {type(launch_contract).__name__}",
            )
        self.launch_contract = launch_contract
        from vrl.families.registry import get_model_family_entry

        self.family_entry = get_model_family_entry(launch_contract.family)
        self.executor: GenerationChunkExecutor | None = None
        self._memory_parking = WorkerMemoryParking(
            worker_id,
            launch_contract,
        )
        self._policy_version: int | None = self.launch_contract.policy_version
        # Flipped on once a model that supports versioned trainable-state slots
        # receives its first weight install; from then on execute_chunk activates
        # the slot for each request's stamped version instead of comparing against
        # one global version (which is what makes a non-draining sync safe).
        self._uses_versioned_slots = False
        self._profiler_config = TorchProfilerConfig(
            **dict(self.launch_contract.torch_profiler),
        )
        self._profiler_output_dir = self._profiler_config.output_dir or "outputs/"
        self._profiler_step = 0
        self._metadata_provider = metadata_provider or self._fallback_metadata

    def load_policy(self) -> None:
        """Build the family executor from the serialized launch contract."""

        if self.executor is not None:
            self._memory_parking.validate_loaded(self.executor)
            return
        from vrl.utils.memory import log_host_memory

        log_host_memory(f"generation_worker:{self.worker_id}:before_load_policy", log=logger)
        try:
            self.executor = self._memory_parking.build(self._build_executor)
            if (
                self.executor.family != self.family_entry.family
                or self.executor.task != self.family_entry.task
            ):
                raise ValueError(
                    "executor identity does not match launch contract: "
                    f"{self.executor.family}/{self.executor.task} != "
                    f"{self.family_entry.family}/{self.family_entry.task}",
                )
            self._memory_parking.validate_loaded(self.executor)
        except BaseException as load_error:
            # Validation/identity frames can retain the just-built executor after
            # self.executor is cleared. Drop their locals before pool teardown so
            # model tensors die while the retained CuMem registry still exists.
            if load_error.__traceback__ is not None:
                traceback.clear_frames(load_error.__traceback__)
            try:
                self.release_policy()
            except BaseException as release_error:
                raise RuntimeError(
                    "generation policy load and cleanup both failed: "
                    f"load={load_error!r}; cleanup={release_error!r}",
                ) from release_error
            raise
        log_host_memory(f"generation_worker:{self.worker_id}:after_load_policy", log=logger)

    def release_policy(self) -> None:
        """Drop loaded model state so the worker releases CUDA memory before exit."""

        with self._memory_parking.release_scope():
            self.executor = None

    def sleep(self) -> WorkerMemoryParkingSnapshot:
        """Offload the loaded model to host RAM, freeing the GPU without discarding it.

        This keeps the executor alive while its selected backend yields physical
        GPU memory. CuMem restores mappings from a CPU backup, ordinary modules
        move back to their captured device, and Accelerate hooks remain installed
        so their next forward streams weights on demand. ``release_policy`` is a
        separate cold-eviction path that drops the executor entirely.
        """

        restore_device = (
            self._executor_device(self.executor) if self.executor is not None else "cpu"
        )
        return self._memory_parking.sleep(
            self.executor,
            restore_device=restore_device,
        )

    def wake(self) -> None:
        """Restore a slept model from host RAM onto its GPU (no disk reload).

        The counterpart to ``sleep``. If the executor was evicted instead of slept
        (e.g. a hard release happened in between), rebuild it via ``load_policy`` so
        a wake is always safe to call.
        """

        self._memory_parking.require_healthy("wake", executor=self.executor)
        model = getattr(self.executor, "model", None)
        if model is None:
            self.load_policy()
            return
        assert self.executor is not None
        self._memory_parking.wake(self.executor)

    def update_weights(self, state_ref: Any, policy_version: int) -> int:
        """Install weights and return the policy version as the commit ACK.

        When the model supports versioned trainable-state slots, install the new
        version as a retained slot WITHOUT overwriting the slots older in-flight
        requests still depend on (the non-draining-sync path); ``execute_chunk``
        then activates the right slot per request. Otherwise keep the single
        in-place overwrite (the draining-barrier path).
        """

        self._memory_parking.require_active(
            "update_weights",
            executor=self.executor,
        )
        self.load_policy()
        policy_obj = getattr(self.executor, "model", None)
        try:
            if self.launch_contract.versioned_weight_sync and bool(
                getattr(policy_obj, "supports_versioned_trainable_state", False),
            ):
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
        except BaseException as error:
            self._memory_parking.record_model_failure(policy_obj, error)
            raise
        # The single scalar now means "current submit version": it must track the
        # latest installed version even in slot mode. It reaches the producer
        # through this method's ACK (returned below) and worker_metadata()'s
        # "policy_version" field — the producer stamps NEW requests from
        # RolloutLifecycle.current_policy_version() -> runtime.current_policy_version
        # (attribute), not by reading this worker directly. Per-chunk results take
        # their version from request.policy_version, not this field.
        self._policy_version = int(policy_version)
        return self._policy_version

    def supports_versioned_trainable_state(self) -> bool:
        """Whether the loaded model can retain versioned trainable-state slots.

        Drives the runtime's non-draining-weight-sync capability. Requires the
        model to be built, so callers query it after at least one weight sync.
        """

        self.load_policy()
        model = getattr(self.executor, "model", None)
        return bool(
            self.launch_contract.versioned_weight_sync
            and getattr(model, "supports_versioned_trainable_state", False)
        )

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
        self._memory_parking.require_active(
            "execute_chunk",
            executor=self.executor,
        )
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
            self._memory_parking.recover_after_execution_error(model, exc)
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

        self._memory_parking.require_active(
            "probe_chunk_size",
            executor=self.executor,
        )

        import time
        from dataclasses import replace as dataclass_replace

        import torch

        from vrl.generation.execution.chunk_placement import (
            AffinePeakFit,
            ChunkMemoryReading,
        )
        from vrl.generation.execution.chunks import SampleChunk
        from vrl.utils.profiling import profile_range

        if not torch.cuda.is_available():
            raise RuntimeError("chunk-size probe requires CUDA")
        if max_samples < 1:
            raise ValueError(f"probe max_samples must be >= 1, got {max_samples}")
        self.load_policy()
        executor = self.executor
        model = getattr(executor, "model", None)
        if not isinstance(executor, DiffusionStagedChunkExecutor):
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
                    record_function=profile_range,
                )
                prepared = executor.run_prepare_stage(
                    prompt_output,
                    stage_durations=stage_durations,
                )
                prepared.config = dataclass_replace(
                    prepared.config,
                    execute_steps=_PROBE_EXECUTE_STEPS,
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
                self._memory_parking.recover_after_execution_error(model, exc)
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
    ) -> GenerationOutput | PipelinedRequestOutOfMemory:
        """Run ALL of a request's chunks through the executor's software pipeline
        (``forward_plan_pipelined``) on THIS worker, so chunk N+1's denoise overlaps
        chunk N's GPU->CPU copy + host packing — hiding the per-chunk worker
        boundary that per-chunk dispatch leaves serial. Single-worker only (all the
        request's chunks must be here). Returns the gathered ``GenerationOutput``;
        after a CUDA OOM it first joins pending copy work, clears partial request
        state, and returns ``PipelinedRequestOutOfMemory`` for driver-side retry.

        Version safety mirrors ``execute_chunk`` but at the REQUEST level (every
        chunk shares ``request.policy_version``): slot mode serves the request from
        its stamped version's slot (stale slot -> ``StaleSlotDiscard`` so the
        producer counts a graceful discard, never an off-policy train); non-slot
        mode rejects a version mismatch. The version is checked + activated ONCE
        here, not per chunk.
        """

        from vrl.generation.execution.types import StaleSlotDiscard

        self._memory_parking.require_active(
            "execute_request_pipelined",
            executor=self.executor,
        )
        self.load_policy()
        expected_version = request.policy_version
        model = getattr(self.executor, "model", None)
        if self._uses_versioned_slots and expected_version is not None:
            if not model.has_trainable_state(expected_version):
                raise StaleSlotDiscard(
                    f"trainable-state slot evicted for policy_version={expected_version}",
                )
            try:
                model.activate_trainable_state(expected_version)
            except Exception as error:
                self._memory_parking.recover_after_execution_error(model, error)
                raise
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
        try:
            return forward_plan_pipelined(request, sample_rows, engine_plan)
        except RuntimeError as error:
            self._memory_parking.recover_after_execution_error(model, error)
            if not is_cuda_out_of_memory(error):
                raise
            error_text = str(error)
            # The exception traceback retains every partially produced chunk and
            # copy-stream object. Clear those frames before emptying the allocator
            # cache; otherwise the driver's safe per-chunk retry can immediately
            # OOM on tensors held by the failed whole-request pipeline.
            error_traceback = error.__traceback__
            if error_traceback is not None:
                traceback.clear_frames(error_traceback)
                error.__traceback__ = None
            release_cuda_memory()
            return PipelinedRequestOutOfMemory(
                request_id=request.request_id,
                worker_id=self.worker_id,
                error=error_text,
            )
        except Exception as error:
            self._memory_parking.recover_after_execution_error(model, error)
            raise

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
            f"{self.worker_id}_{self.family_entry.family}_{self.family_entry.task}_"
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
        from vrl.models.interfaces.runtime import ModelBuild

        build_payload = dict(launch_contract.model_build)
        build_payload["family"] = self.family_entry.family
        device = build_payload.get("device")
        if isinstance(device, str):
            import torch

            build_payload["device"] = torch.device(device)
        # ModelBuild and RolloutBuildOptions normalize parameter and nested rollout
        # dtypes while reconstructing the primitive Ray payload.
        build = ModelBuild(**build_payload)
        from vrl.models.checkpoint_identity import resolve_checkpoint_model_identity

        worker_model_identity = resolve_checkpoint_model_identity(build)
        if worker_model_identity != launch_contract.expected_model_identity:
            raise ValueError(
                f"rollout worker {self.worker_id} model identity mismatch before "
                "model construction: "
                f"driver={launch_contract.expected_model_identity!r}, "
                f"worker={worker_model_identity!r}",
            )
        bundle = self.family_entry.build_rollout(build)
        loaded_model_identity = resolve_checkpoint_model_identity(build)
        if loaded_model_identity != launch_contract.expected_model_identity:
            raise RuntimeError(
                f"rollout worker {self.worker_id} model checkpoint source changed "
                "during bundle construction: "
                f"before={launch_contract.expected_model_identity!r}, "
                f"after={loaded_model_identity!r}",
            )
        model = require_runtime_model(bundle.model, owner="RuntimeBundle.model")
        # Family- and scheme-agnostic backstop: if rollout quantization asks for a
        # quantized rollout (FP8/NVFP4/...) but this family's builder forgot to swap,
        # the model would silently run at its base dtype — fail loudly instead.
        from vrl.models.loader import assert_rollout_quantization_applied

        assert_rollout_quantization_applied(model, build)
        executor_kwargs = dict(launch_contract.executor_kwargs)
        from vrl.families.registry import GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR

        executor_cls = import_from_path(self.family_entry.executor_cls)
        if self.family_entry.executor_cls == GENERIC_FULL_SEQUENCE_DENOISE_EXECUTOR:
            executor_kwargs.update(
                family=self.family_entry.family,
                task=self.family_entry.task,
            )
        built = executor_cls(model, **executor_kwargs)
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

        copied = map_tensor_tree(
            value,
            _pinned_copy,
            is_leaf=lambda candidate: hasattr(candidate, "detach") and hasattr(candidate, "cpu"),
        )
        if pending["cuda_copies"]:
            import torch

            torch.cuda.synchronize()
        return copied

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
