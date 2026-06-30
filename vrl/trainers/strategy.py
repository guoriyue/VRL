"""Training strategy seam: the boundary between the trainer and how it runs.

The trainer drives the GRPO loop; *how* a step executes on the hardware —
backward, grad clipping, and trainable-state export/load — goes through a
``Strategy`` so the trainer never hard-codes single-process vs FSDP2.

This readiness sprint ships only ``SingleProcessStrategy`` (current behavior
moved behind the protocol, byte-for-byte). The FSDP2 strategy — DTensor-aware
clip and full-state export, a real ``barrier`` — lands in
``SPRINT_multi_gpu_training.md`` and slots in here without touching the trainer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

import torch
from torch import nn

from vrl.trainers.distributed import DistributedTrainingContext
from vrl.utils.config import cfg_path


class Strategy(Protocol):
    """How one training step executes; the only seam the trainer depends on."""

    context: DistributedTrainingContext

    def prepare_model(self, model: Any) -> Any:
        """Return the model the trainer should train (wrapped if the backend needs it).

        Identity for single process; FSDP2 shards the trainable handle here. The
        trainer routes its model through this once at construction so it never
        hard-codes the wrapping.
        """
        ...

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        """Run the backward pass (scaled when an fp16 GradScaler is active)."""
        ...

    def clip_grad_norm(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> float:
        """Clip gradients in place and return the pre-clip total norm."""
        ...

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        """Checkpoint-facing trainable state (nested by module name, CPU tensors)."""
        ...

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        """Rollout-facing flat trainable state (unwrapped, policy-facing keys)."""
        ...

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        """Load a checkpoint-facing trainable state back into the bundle."""
        ...

    def barrier(self) -> None:
        """Synchronize all training ranks (no-op for single process)."""
        ...


class SingleProcessStrategy(Strategy):
    """The current single-GPU behavior, moved behind the strategy protocol.

    Every method here is the existing trainer / checkpoint / weight-sync logic
    verbatim; this installs the seam without changing what a single-process run
    does. ``context`` defaults to a rank0/world1 identity.
    """

    def __init__(self, context: DistributedTrainingContext | None = None) -> None:
        self.context = context or _single_process_context()

    def prepare_model(self, model: Any) -> Any:
        # Single process trains the model as-is; the seam exists so FSDP2 can wrap
        # without the trainer changing.
        return model

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(
        self,
        parameters: Iterable[nn.Parameter],
        max_norm: float,
    ) -> float:
        return float(nn.utils.clip_grad_norm_(parameters, max_norm))

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        from vrl.trainers.checkpointing import export_trainable_state

        return export_trainable_state(bundle)

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.trainers.weight_sync import build_trainable_state_sync_getter

        return build_trainable_state_sync_getter(bundle)()

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        from vrl.trainers.checkpointing import load_trainable_state

        load_trainable_state(bundle, state)

    def barrier(self) -> None:
        return None


def _single_transformer_handle(model: Any) -> tuple[Any, Any]:
    """The trainable ``transformer`` handle + its writer, shared by FSDP2 and DDP.

    Diffusion policies expose the trainable root as a ``transformer`` handle plus
    ``_set_transformer`` to write the wrapped module back (so the pipeline and
    attention processors keep pointing at it). AR families do not yet expose
    explicit trainable roots, and dual-stage Wan exposes a second trainable root
    (whose DEFAULT trainable set is ``("transformer_2",)``), so both fail-fast here
    — blindly wrapping ``transformer`` could wrap a frozen module and leave the real
    trainable one unmanaged (SPRINT_multi_gpu_training.md §5).
    """

    handle = getattr(model, "transformer", None)
    set_transformer = getattr(model, "_set_transformer", None)
    if handle is None or not callable(set_transformer):
        raise NotImplementedError(
            "multi-GPU model wrapping needs a diffusion policy exposing a `transformer` "
            f"handle and `_set_transformer`; {type(model).__name__} exposes neither. "
            "AR families (janus_pro / nextstep_1) need explicit trainable roots first "
            "(SPRINT_multi_gpu_training.md §5).",
        )
    trainable = getattr(model, "trainable_modules", None)
    if isinstance(trainable, Mapping) and set(trainable) != {"transformer"}:
        raise NotImplementedError(
            f"multi-GPU wrapping wraps only the single 'transformer' handle, but "
            f"{type(model).__name__}.trainable_modules = {sorted(trainable)}. "
            "Multi-transformer wrapping (e.g. dual-stage Wan transformer_2) needs "
            "per-handle writers and is not wired yet (SPRINT_multi_gpu_training.md §5).",
        )
    return handle, set_transformer


class FSDPStrategy(Strategy):
    """FSDP2 (``fully_shard`` + DTensor) training behind the same seam.

    The model wraps once in ``prepare_model``; thereafter params/grads/optimizer
    state live as DTensor shards over the mesh. Checkpoint and rollout export both
    gather a full, unwrapped, policy-facing state on rank0 — the trainer and the
    Ray rollout workers never see a shard or a wrapper key. The collective work
    lives in ``vrl/trainers/fsdp.py``; this class is the trainer-facing adapter.

    This is the strategy *layer* of ``SPRINT_multi_gpu_training.md``. The online
    recipe drives it through the same per-rank-local symmetric-colocated path as
    DDP: every torchrun rank owns a local rollout/training device, while FSDP
    handles DTensor sharding and collectives behind this adapter.
    """

    def __init__(
        self,
        context: DistributedTrainingContext,
        *,
        mesh_dims: list[str] | None = None,
        precision_policy: str = "actor",
        reshard_after_forward: bool = True,
    ) -> None:
        self.context = context
        self._mesh_dims = mesh_dims or ["dp_shard"]
        self._precision_policy = precision_policy
        self._reshard_after_forward = reshard_after_forward
        self._mesh: Any | None = None  # built on first prepare_model (needs a live PG)

    def _ensure_mesh(self) -> Any:
        if self._mesh is None:
            from vrl.trainers.fsdp import build_fsdp_mesh

            self._mesh = build_fsdp_mesh(self.context, self._mesh_dims)
        return self._mesh

    def prepare_model(self, model: Any) -> Any:
        """Shard the policy's trainable transformer in place and return the policy."""
        from vrl.trainers.fsdp import (
            apply_fsdp,
            init_training_process_group,
            mixed_precision_policy,
        )

        # Validate the trainable handle BEFORE touching the process group so a bad
        # model fails fast (mirrors DDPStrategy; the guard tests need no live PG).
        handle, set_transformer = _single_transformer_handle(model)
        # Create the process group + bind this rank's cuda device up front, exactly
        # like DDPStrategy. init_device_mesh would lazily auto-init a default group,
        # but it would NOT call torch.cuda.set_device(local_rank) first, so the NCCL
        # group and the per-block fully_shard could bind the wrong card on a
        # single-node multi-GPU box. Doing it here keeps the two strategies
        # symmetric and the device choice explicit. No-op for single_process and
        # when a group already exists (the CPU gloo test fixture pre-inits one).
        backend = "gloo" if self.context.device.type == "cpu" else "nccl"
        init_training_process_group(self.context, backend=backend)
        wrapped = apply_fsdp(
            handle,
            mesh=self._ensure_mesh(),
            mp_policy=mixed_precision_policy(self._precision_policy),
            reshard_after_forward=self._reshard_after_forward,
        )
        set_transformer(wrapped)
        return model

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        # FSDP2 reduce-scatters gradients inside the backward hooks; the bf16 actor
        # recipe runs without a GradScaler, but keep the seam identical to single
        # process so the trainer loop is backend-agnostic.
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(self, parameters: Iterable[nn.Parameter], max_norm: float) -> float:
        # torch.nn.utils.clip_grad_norm_ is DTensor-aware: it reduces the global
        # norm across the mesh and clips the local shards. float() collapses the
        # replicated norm scalar to a Python float for logging.
        return float(nn.utils.clip_grad_norm_(parameters, max_norm))

    def _gather_unwrapped(self, module: Any) -> tuple[Any, dict[str, Any]]:
        """Peel compile/DDP, then gather the sharded module to a full state.

        Mirrors single-process: ``get_model_state_dict`` strips the
        ``_orig_mod.`` compile prefix while ``named_parameters()`` keeps it, so a
        sharded gather + trainable-key select must run on the SAME uncompiled
        module or the two key sets disagree (the select would drop everything).
        Returns the unwrapped module (for trainable-name selection) and its full
        state. PEFT is kept so LoRA keys stay policy-facing.
        """
        from vrl.trainers.fsdp import gather_full_state_dict
        from vrl.trainers.weight_sync import unwrap_compile_and_ddp

        inner = unwrap_compile_and_ddp(module)
        return inner, gather_full_state_dict(inner)

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        from vrl.trainers.weight_sync import require_trainable_modules, to_cpu

        modules = require_trainable_modules(bundle)
        return {name: to_cpu(self._gather_unwrapped(module)[1]) for name, module in modules.items()}

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.trainers.weight_sync import require_trainable_modules, select_trainable_state

        modules = require_trainable_modules(bundle)
        state: dict[str, Any] = {}
        # Gather each sharded module to a full state, then pick the trainable keys
        # the same way single process does — so the rollout-facing key space is
        # identical whether or not the trainer was sharded.
        for module_name, module in modules.items():
            inner, full = self._gather_unwrapped(module)
            state.update(select_trainable_state(inner, str(module_name), full))
        if not state:
            raise ValueError("trainable module state is empty")
        return state

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        from vrl.trainers.fsdp import load_full_state_dict
        from vrl.trainers.weight_sync import require_trainable_modules, unwrap_compile_and_ddp

        modules = require_trainable_modules(bundle)
        # `state` is the checkpoint-facing shape: nested by trainable module name
        # (what export_trainable_state produced). Load into the same uncompiled
        # namespace the export gathered from.
        for name, module in modules.items():
            if name in state:
                load_full_state_dict(unwrap_compile_and_ddp(module), state[name])

    def barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()


class DDPStrategy(Strategy):
    """DistributedDataParallel training behind the same seam.

    For a model that fits on one card (a 2B diffusion transformer + LoRA does), DDP
    replicates the full module on every rank and all-reduces gradients in the
    backward hooks — simpler and cheaper than FSDP2's shard/all-gather, which only
    earns its keep when the model does NOT fit. Because every rank keeps FULL
    params, the checkpoint/rollout export is the same full-state path FSDP uses on
    the unwrapped module (``unwrap_compile_and_ddp`` peels DDP's ``.module``; the
    "gather" is a no-op at the plain-tensor level). Only ``prepare_model`` (wrap)
    diverges from FSDPStrategy.

    Symmetric half of ``SPRINT_symmetric_colocated_ddp.md``. The online multi-rank
    loop that drives N ranks is a separate, not-yet-wired phase, so the online
    recipe still gates non-single_process strategies (see ``run_online_recipe``);
    the strategy itself is exercised on a single CPU rank in
    ``tests/trainers/test_ddp.py``.
    """

    def __init__(
        self,
        context: DistributedTrainingContext,
        *,
        find_unused_parameters: bool = False,
    ) -> None:
        self.context = context
        self._find_unused_parameters = find_unused_parameters

    def prepare_model(self, model: Any) -> Any:
        """Replicate the policy's trainable transformer with DDP and return the policy.

        Wrap only the trainable ``transformer`` handle, not the whole policy: the
        frozen base inside still has ``requires_grad=False`` so it stays out of
        DDP's reducer buckets, whereas wrapping the top-level model would drag the
        frozen VAE/text-encoder into DDP and force ``find_unused_parameters=True``.
        """
        from torch.nn.parallel import DistributedDataParallel

        from vrl.trainers.fsdp import init_training_process_group

        # Validate the trainable handle BEFORE touching the process group so a bad
        # model fails fast (and the guard tests need no live PG).
        handle, set_transformer = _single_transformer_handle(model)
        backend = "gloo" if self.context.device.type == "cpu" else "nccl"
        init_training_process_group(self.context, backend=backend)
        device_ids = [self.context.local_rank] if self.context.device.type == "cuda" else None
        wrapped = DistributedDataParallel(
            handle,
            device_ids=device_ids,
            find_unused_parameters=self._find_unused_parameters,
        )
        set_transformer(wrapped)
        return model

    def backward(self, loss: torch.Tensor, *, grad_scaler: Any | None = None) -> None:
        # DDP all-reduces gradients inside the backward hooks (this IS the
        # synchronized step); the seam stays identical to single-process/FSDP.
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(self, parameters: Iterable[nn.Parameter], max_norm: float) -> float:
        # Grads are already all-reduced (identical on every rank), so a local clip
        # is globally correct.
        return float(nn.utils.clip_grad_norm_(parameters, max_norm))

    def _unwrapped_full_state(self, module: Any) -> tuple[Any, dict[str, Any]]:
        from vrl.trainers.weight_sync import unwrap_compile_and_ddp

        inner = unwrap_compile_and_ddp(module)
        # DDP replicates the full module on every rank (no sharding), so the plain
        # unwrapped state_dict() IS the full policy-facing state — same key space as
        # inner.named_parameters(), which select_trainable_state() checks against.
        #
        # Do NOT route this through the FSDP gather_full_state_dict (DCP
        # get_model_state_dict full_state_dict=True): at world_size>1 its distributed
        # all-gather path drops the PEFT LoRA keys for a *replicated* (non-sharded)
        # module, so select_trainable_state() then reports every lora_A/lora_B param
        # "missing" and the first weight sync raises. The ws=1 CPU test never hit that
        # path (gather is a no-op at ws=1); the real 2x1 NCCL run did. cpu-offload to
        # match the rollout payload contract (the FSDP path uses cpu_offload=True).
        full = {key: value.detach().to("cpu") for key, value in inner.state_dict().items()}
        return inner, full

    def export_trainable_state(self, bundle: Any) -> dict[str, dict[str, Any]]:
        from vrl.trainers.weight_sync import require_trainable_modules, to_cpu

        modules = require_trainable_modules(bundle)
        return {
            name: to_cpu(self._unwrapped_full_state(module)[1])
            for name, module in modules.items()
        }

    def export_rollout_state(self, bundle: Any) -> dict[str, Any]:
        from vrl.trainers.weight_sync import require_trainable_modules, select_trainable_state

        modules = require_trainable_modules(bundle)
        state: dict[str, Any] = {}
        for module_name, module in modules.items():
            inner, full = self._unwrapped_full_state(module)
            state.update(select_trainable_state(inner, str(module_name), full))
        if not state:
            raise ValueError("trainable module state is empty")
        return state

    def load_trainable_state(self, bundle: Any, state: dict[str, Any]) -> None:
        from vrl.trainers.fsdp import load_full_state_dict
        from vrl.trainers.weight_sync import require_trainable_modules, unwrap_compile_and_ddp

        modules = require_trainable_modules(bundle)
        for name, module in modules.items():
            if name in state:
                load_full_state_dict(unwrap_compile_and_ddp(module), state[name])

    def barrier(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()


def build_strategy(cfg: Any, context: DistributedTrainingContext) -> Strategy:
    """Construct the training strategy named by the resolved context.

    The single dispatch point from config/context to a concrete strategy.
    ``fsdp`` reads its FSDP2 knobs from ``distributed.training.fsdp`` and runs the
    §10 readiness gates (combinations that need DTensor-aware state handling not
    yet built) before constructing ``FSDPStrategy``.
    """

    if context.strategy == "single_process":
        return SingleProcessStrategy(context)
    if context.strategy == "fsdp":
        _assert_fsdp_config_supported(cfg)
        return FSDPStrategy(
            context,
            mesh_dims=list(cfg_path(cfg, "distributed.training.fsdp.mesh", ["dp_shard"]) or ["dp_shard"]),
            precision_policy=str(cfg_path(cfg, "distributed.training.fsdp.precision_policy", "actor")),
            reshard_after_forward=bool(
                cfg_path(cfg, "distributed.training.fsdp.reshard_after_forward", True),
            ),
        )
    if context.strategy == "ddp":
        return DDPStrategy(
            context,
            find_unused_parameters=bool(
                cfg_path(cfg, "distributed.training.ddp.find_unused_parameters", False),
            ),
        )
    # resolve_training_context / the schema Literal reject other values upstream;
    # this guards direct callers.
    raise ValueError(
        f"unknown distributed.training.strategy={context.strategy!r}; "
        "expected 'single_process', 'fsdp', or 'ddp'",
    )


def _assert_fsdp_config_supported(cfg: Any) -> None:
    """Fail-fast on fsdp + a feature whose DTensor handling is not implemented yet.

    These are the ``SPRINT_multi_gpu_training.md`` §10 gates: EMA and optimizer
    resume both touch state that, under FSDP2, is sharded as DTensor and needs an
    explicit full gather/scatter the first version does not provide. Better a clear
    error here than silent partial support on real hardware.
    """

    if bool(cfg_path(cfg, "actor.ema.enable", False)):
        raise NotImplementedError(
            "distributed.training.strategy=fsdp with actor.ema.enable=true is not "
            "supported until DTensor-aware EMA is implemented "
            "(SPRINT_multi_gpu_training.md §8/§10). Disable EMA to run FSDP2.",
        )
    resume_from = str(cfg_path(cfg, "trainer.resume_from", "") or "").strip()
    if resume_from:
        raise NotImplementedError(
            "distributed.training.strategy=fsdp with trainer.resume_from is not "
            "supported until FSDP2 optimizer state export/load is implemented "
            "(SPRINT_multi_gpu_training.md §8/§10).",
        )
    if bool(cfg_path(cfg, "model.torch_compile.enable", False)):
        raise NotImplementedError(
            "distributed.training.strategy=fsdp with model.torch_compile.enable=true "
            "is not supported: torch.compile (inductor graph capture) is unsound with "
            "FSDP2 fully_shard's reshard-after-forward all-gathers. Set "
            "model.torch_compile.enable=false to run FSDP2.",
        )


def _single_process_context() -> DistributedTrainingContext:
    return DistributedTrainingContext(
        strategy="single_process",
        distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_primary=True,
        device=torch.device("cpu"),
    )


__all__ = ["DDPStrategy", "FSDPStrategy", "SingleProcessStrategy", "Strategy", "build_strategy"]
