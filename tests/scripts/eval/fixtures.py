"""Real objects shared by the SANA eval-script tests."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from diffusers.image_processor import VaeImageProcessor

from vrl.scripts.eval.sana_inference import SANA_EVAL_SCHEDULER_CONFIG


def build_official_sana_scheduler(**overrides: Any) -> Any:
    """The real ``DPMSolverMultistepScheduler`` at SANA's official protocol.

    Config-init, no download. An override is how the drift tests produce a
    scheduler that must be REJECTED, and because the object is genuine an
    upstream rename of any protocol key turns the accept case red instead of
    letting a double echo the table back.
    """

    from diffusers import DPMSolverMultistepScheduler

    kwargs = {
        key: value for key, value in SANA_EVAL_SCHEDULER_CONFIG.items() if key != "class_name"
    }
    kwargs.update(overrides)
    return DPMSolverMultistepScheduler(**kwargs)


class _TinyTextEncoder(torch.nn.Module):
    """A frozen encoder that, like Gemma-2, maps token ids to hidden states and
    reports its ``dtype``. Returns a one-tuple so ``encoder(...)[0]`` reads like HF."""

    def __init__(self) -> None:
        super().__init__()
        from tests.models.steps.denoise.fixtures import TINY_SANA_CAPTION_DIM

        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(256, TINY_SANA_CAPTION_DIM)

    @property
    def dtype(self) -> torch.dtype:
        return self.embed.weight.dtype

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor]:
        del attention_mask
        return (self.embed(input_ids),)


class TinySanaPipeline:
    """A ``SanaPipeline`` stand-in over REAL tiny modules.

    Everything the shared loader and the eval scripts touch is genuine -- a
    ``SanaTransformer2DModel``, an ``AutoencoderKL``, a frozen text-encoder
    module, the official DPM-Solver++ scheduler -- so ``SanaModel.from_build``,
    the rollout bundle, checkpoint save/restore and the local identity hash all
    run for real. Only ``__call__`` (diffusers' denoising loop, which needs the
    Gemma encoder) is a double: it records the call and paints a solid image
    whose colour says which weights the transformer held at that moment, so
    "base before restore, current after" is witnessed by the weights themselves.
    """

    BASE_COLOR = (10, 20, 200)
    RESTORED_COLOR = (200, 10, 20)

    def __init__(self) -> None:

        from tests.models.steps.denoise.fixtures import (
            build_tiny_autoencoder_kl,
            build_tiny_transformer,
        )

        self.transformer = build_tiny_transformer("sana")
        # f4 so a 32x32 image is the transformer's 8x8 latent (TINY_SANA_LATENT_SHAPE).
        self.vae = build_tiny_autoencoder_kl(downsamples=2)
        self.vae_scale_factor = 4
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.text_encoder = _TinyTextEncoder()
        self.scheduler = build_official_sana_scheduler()
        self.calls: list[dict[str, Any]] = []
        self.loads = 0
        self._labels: dict[float, str] = {self.fingerprint(): "base"}

    @property
    def components(self) -> dict[str, Any]:
        return {
            "transformer": self.transformer,
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "scheduler": self.scheduler,
        }

    def fingerprint(self) -> float:
        return float(sum(p.detach().double().sum().item() for p in self.transformer.parameters()))

    def label_weights(self, label: str) -> None:
        """Name the transformer's current weights so a later call can report them."""

        self._labels[self.fingerprint()] = label

    def holds(self) -> str:
        return self._labels.get(self.fingerprint(), "restored")

    def __call__(self, **kwargs: Any) -> Any:
        import torch
        from PIL import Image

        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        assert "complex_human_instruction" not in kwargs
        held = self.holds()
        self.calls.append(
            {
                **kwargs,
                "scheduler": self.scheduler,
                "generator_seed": kwargs["generator"].initial_seed(),
                "weights": held,
            },
        )
        color = self.BASE_COLOR if held == "base" else self.RESTORED_COLOR
        count = int(kwargs.get("num_images_per_prompt", 1))
        images = [
            Image.new("RGB", (kwargs["width"], kwargs["height"]), color=color)
            for _ in range(count)
        ]
        return SimpleNamespace(images=images)

    # -- the two pipeline methods the SANA family calls on the denoise path ----

    def _tokenize(self, prompts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic word hashing stands in for the Gemma tokenizer."""

        import zlib

        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        mask = torch.zeros(len(prompts), max_length, dtype=torch.long)
        for row, prompt in enumerate(prompts):
            tokens = [zlib.crc32(word.encode()) % 255 + 1 for word in prompt.split()][:max_length]
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[row, : len(tokens)] = 1
        return ids, mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] = "",
        num_images_per_prompt: int = 1,
        device: Any = None,
        max_sequence_length: int = 300,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        ids, mask = self._tokenize(prompts, max_sequence_length)
        embeds = self.text_encoder(ids, attention_mask=mask)[0]
        embeds = embeds.repeat_interleave(num_images_per_prompt, dim=0)
        mask = mask.repeat_interleave(num_images_per_prompt, dim=0)
        if not do_classifier_free_guidance:
            return embeds, mask, None, None
        negatives = (
            [negative_prompt] * len(prompts)
            if isinstance(negative_prompt, str)
            else list(negative_prompt)
        )
        neg_ids, neg_mask = self._tokenize(negatives, max_sequence_length)
        neg_embeds = self.text_encoder(neg_ids, attention_mask=neg_mask)[0]
        return (
            embeds,
            mask,
            neg_embeds.repeat_interleave(num_images_per_prompt, dim=0),
            neg_mask.repeat_interleave(num_images_per_prompt, dim=0),
        )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Any,
        generator: Any,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def install(self, monkeypatch: Any, snapshot: Path, *, on_load: Any = None) -> None:
        """Serve this pipeline for ``DiffusionPipeline.from_pretrained(snapshot)``.

        The HF load is the one external boundary; ``on_load`` lets a test act
        while the model is being constructed (e.g. mutate the snapshot tree).
        """

        from diffusers import DiffusionPipeline

        def from_pretrained(path: Any, **kwargs: Any) -> TinySanaPipeline:
            assert Path(str(path)).resolve() == snapshot.resolve(), path
            self.loads += 1
            # diffusers loads each component at its mapped dtype; the loader
            # maps the trainable transformer to the rollout precision.
            dtypes = kwargs.get("torch_dtype")
            dtype = (
                dtypes.get("transformer", dtypes.get("default"))
                if isinstance(dtypes, dict)
                else dtypes
            )
            if isinstance(dtype, torch.dtype):
                # The cast weights are the same named checkpoint at a new dtype.
                label = self._labels.get(self.fingerprint())
                self.transformer.to(dtype)
                if label is not None:
                    self._labels[self.fingerprint()] = label
            if on_load is not None:
                on_load()
            return self

        monkeypatch.setattr(DiffusionPipeline, "from_pretrained", staticmethod(from_pretrained))


def write_tiny_sana_snapshot(path: Path) -> Path:
    """A local model directory: the official scheduler, a real tiny transformer, and
    the ``model_index.json`` marker.

    ``load_official_scheduler`` reads ``scheduler/`` from it for real
    (``local_files_only``); the generic replay recipe loads ``transformer/``
    through ``SanaTransformer2DModel.from_pretrained`` unpatched, so a replay
    bundle built from this directory has no double at all. The checkpoint
    identity is the hash of this tree.
    """

    from tests.models.steps.denoise.fixtures import build_tiny_transformer

    path.mkdir(parents=True, exist_ok=True)
    build_official_sana_scheduler().save_pretrained(path / "scheduler")
    build_tiny_transformer("sana").save_pretrained(path / "transformer")
    (path / "model_index.json").write_text('{"_class_name": "SanaPipeline"}\n', encoding="utf-8")
    return path


# The SANA online-GRPO recipe as the aesthetic experiment composes it (same
# recipe, model, sampling and dataset presets, same actor/trainer body), with
# the one reward a CPU-only run can execute: model-free image sharpness. The
# aesthetic preset's GPU reward models cannot run where these tests do.
_TINY_SANA_EXPERIMENT = """\
defaults:
  - /recipe/online/flow_matching_grpo
  - /model/sana/1600m
  - /sampling/image/512
  - /sampling/denoise/10_step_cfg_4_5
  - /dataset/drawbench_train_192
  - /reward/image_sharpness
sampling:
  max_sequence_length: 300
actor:
  optim:
    lr: 3.0e-4
  training_microbatch_size: 8
  gradient_checkpointing: true
  ppo_epochs: 4
trainer:
  debug:
    first_step: true
"""


def tiny_sana_online_config(
    tmp_path: Path,
    *,
    prompts: tuple[str, ...] = ("a cat",),
    snapshot: Path | None = None,
    overrides: tuple[str, ...] = (),
) -> Any:
    """The SANA online-GRPO recipe resolved onto a tiny local snapshot.

    Writes the experiment, the snapshot (unless ``snapshot`` names an existing
    one) and a one-row-per-prompt manifest under ``tmp_path`` and returns the
    merged config: fp32, a GPU-less rollout fleet
    (``distributed.resources.rollout.num_gpus=0``), 32x32 two-step sampling
    with the SDE window inside it, a two-sample rollout batch and a CPU
    sharpness reward, so the config runs end to end on a CPU host. Pair it with
    ``TinySanaPipeline.install`` so the real family loader serves the tiny
    pipeline for ``model.path``.
    """

    import json

    from vrl.config.loading import load_config

    tmp_path.mkdir(parents=True, exist_ok=True)
    if snapshot is None:
        snapshot = write_tiny_sana_snapshot(tmp_path / "sana-snapshot")
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    experiment = tmp_path / "tiny_sana_online.yaml"
    experiment.write_text(_TINY_SANA_EXPERIMENT, encoding="utf-8")
    return load_config(
        str(experiment),
        overrides=[
            f"model.path={snapshot}",
            "model.revision=null",
            "model.use_lora=false",
            "model.torch_compile.enable=false",
            f"data.manifest={manifest}",
            f"trainer.output_dir={tmp_path / 'run'}",
            "trainer.total_epochs=1",
            "precision.training.dtype=fp32",
            "precision.rollout.dtype=fp32",
            "sampling.width=32",
            "sampling.height=32",
            "sampling.num_steps=2",
            "rollout.sde.window_range=[0,2]",
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "rollout.samples_per_generation_batch=2",
            "actor.training_microbatch_size=2",
            "distributed.resources.rollout.num_gpus=0",
            *overrides,
        ],
    )


def install_tiny_sana_in_ray_worker() -> None:
    """Ray ``worker_process_setup_hook``: serve the tiny pipeline in this worker.

    A Ray worker is a separate process, so the test's own
    ``TinySanaPipeline.install`` never reaches it; the hook installs the same
    loader double at worker start for the snapshot the job's runtime env names.
    The patch lives as long as the worker process.
    """

    import os

    import pytest

    TinySanaPipeline().install(pytest.MonkeyPatch(), Path(os.environ["TINY_SANA_SNAPSHOT"]))


def tiny_sana_ray_runtime_env(snapshot: Path) -> dict[str, Any]:
    """The Ray runtime env whose workers serve ``snapshot`` through the tiny pipeline."""

    import os

    repo = str(Path(__file__).resolve().parents[3])
    python_path = os.pathsep.join(
        part for part in (repo, os.environ.get("PYTHONPATH", "")) if part
    )
    return {
        "env_vars": {
            "PYTHONPATH": python_path,
            "TINY_SANA_SNAPSHOT": str(snapshot),
            # Workers of a CPU test cluster must not see a card either.
            "CUDA_VISIBLE_DEVICES": "",
        },
        "worker_process_setup_hook": (
            "tests.scripts.eval.fixtures.install_tiny_sana_in_ray_worker"
        ),
    }


@contextlib.contextmanager
def tiny_sana_ray_cluster(snapshot: Path) -> Iterator[Any]:
    """A real local Ray cluster whose workers serve ``snapshot`` through the tiny pipeline."""

    from tests.conftest import real_local_ray

    with real_local_ray(runtime_env=tiny_sana_ray_runtime_env(snapshot)) as ray:
        yield ray


@dataclass(frozen=True, slots=True)
class TinySanaStack:
    """A resolved tiny SANA online run and an in-process generation runtime over it."""

    resolved: Any
    replay: Any
    runtime: Any

    @property
    def family(self) -> Any:
        return self.resolved.family

    def collector_config(self) -> Any:
        from vrl.rollouts.collector import RolloutCollectorConfig

        return RolloutCollectorConfig.from_root(self.resolved.built.root)

    def trainer_bundle(self) -> Any:
        """The trainer-side policy: the run's replay bundle, built for real on CPU."""

        return self.replay.materialize(context="tiny sana stack")


def tiny_sana_stack(
    monkeypatch: Any, tmp_path: Path, *, overrides: tuple[str, ...] = ()
) -> TinySanaStack:
    """Resolve ``tiny_sana_online_config`` and bind the real worker body to this process.

    The runtime is ``InProcessGenerationRuntime`` over the run's own launch
    contract, so request planning, the SANA denoise executor and batch merging
    are the production code paths; only the weights are tiny.
    """

    from tests.generation._in_process_runtime import InProcessGenerationRuntime
    from vrl import run

    cfg = tiny_sana_online_config(tmp_path, overrides=overrides)
    TinySanaPipeline().install(monkeypatch, tmp_path / "sana-snapshot")
    resolved = run.resolve_online_run(cfg)
    replay = run.resolve_model(
        resolved.family,
        resolved.built.root,
        resolved.device,
        precision=resolved.built.precision,
        for_rollout=False,
    )
    return TinySanaStack(
        resolved=resolved,
        replay=replay,
        runtime=InProcessGenerationRuntime(resolved.ray_launch_inputs(replay)),
    )


COSMOS25_TINY_LORA = {"rank": 2, "alpha": 2, "target_modules": ["to_q", "to_k", "to_v"]}
COSMOS25_TINY_SAMPLING = {
    "width": 32,
    "height": 32,
    "num_frames": 1,
    "num_steps": 2,
    "fps": 4,
    "max_sequence_length": 8,
    "guidance_scale": 1.0,
    "negative_prompt": "",
}


def write_tiny_cosmos25_snapshot(path: Path) -> Path:
    """A real on-disk Cosmos-Predict2.5 model directory the family loads unpatched.

    ``transformer/``, ``vae/`` and ``scheduler/`` are genuine diffusers
    ``save_pretrained`` trees of the tiny CosmosTransformer3DModel, the tiny
    AutoencoderKLWan and a flow UniPC scheduler. With
    ``model.skip_text_encoder=true`` the family loads exactly these three
    components and synthesizes prompt embeddings, so resolve -> build_rollout
    -> generate_one_video -> VAE decode runs for real on CPU in well under a
    second; nothing about the model path is a double.
    """

    from diffusers import UniPCMultistepScheduler

    from tests.models.steps.denoise.fixtures import (
        build_tiny_transformer,
        build_tiny_wan_vae,
    )

    build_tiny_transformer("cosmos").save_pretrained(path / "transformer")
    build_tiny_wan_vae().save_pretrained(path / "vae")
    UniPCMultistepScheduler(
        num_train_timesteps=1000,
        use_flow_sigmas=True,
        prediction_type="flow_prediction",
    ).save_pretrained(path / "scheduler")
    return path


def cosmos25_eval_config(snapshot: Path, **sampling: Any) -> Any:
    """The resolved config a Cosmos-2.5 eval reads, pointed at a tiny snapshot."""

    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model": {
                "family": "cosmos-predict2.5",
                "path": str(snapshot),
                "revision": None,
                "use_lora": True,
                "lora": dict(COSMOS25_TINY_LORA),
                "skip_text_encoder": True,
            },
            "precision": {
                "float32_precision": "ieee",
                "training": {"dtype": "fp32"},
                "rollout": {"dtype": "fp32"},
            },
            "sampling": {**COSMOS25_TINY_SAMPLING, **sampling},
            "rollout": {"denoise_mode": "native", "noise_level": 0.0, "sde": {"type": "cps"}},
        },
    )


WAN_TINY_SAMPLING: dict[str, Any] = {
    "width": 32,
    "height": 32,
    "num_frames": 1,
    "num_steps": 2,
    "fps": 4,
    "max_sequence_length": 8,
    "guidance_scale": 1.0,
    "negative_prompt": "",
}


def write_tiny_wan_snapshot(path: Path) -> Path:
    """A real on-disk Wan-2.1 T2V model directory ``WanPipeline.from_pretrained`` loads.

    Every component is the genuine class the family reads: a word-level
    ``PreTrainedTokenizerFast``, a one-layer ``UMT5EncoderModel`` emitting the
    tiny transformer's ``text_dim``, the tiny ``WanTransformer3DModel`` and
    ``AutoencoderKLWan``, and a flow ``UniPCMultistepScheduler``. Saved through
    ``WanPipeline.save_pretrained`` so ``model_index.json`` and the topology
    ``normalize_wan_model_build`` reads (``boundary_ratio`` / ``expand_timesteps``)
    come from diffusers itself, not from a hand-written file.
    """

    from diffusers import UniPCMultistepScheduler, WanPipeline
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

    from tests.models.steps.denoise.fixtures import (
        TINY_WAN_TEXT_DIM,
        build_tiny_transformer,
        build_tiny_wan_vae,
    )

    specials = ["<pad>", "</s>", "<unk>"]
    words = ["a", "the", "cat", "cup", "bowl", "move", "pick", "up", "robot", "arm"]
    vocab = {token: index for index, token in enumerate([*specials, *words])}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    torch.manual_seed(0)
    text_encoder = UMT5EncoderModel(
        UMT5Config(
            vocab_size=len(vocab),
            d_model=TINY_WAN_TEXT_DIM,
            d_kv=4,
            d_ff=16,
            num_layers=1,
            num_heads=2,
            relative_attention_num_buckets=4,
            dropout_rate=0.0,
        ),
    )
    WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=build_tiny_transformer("wan"),
        vae=build_tiny_wan_vae(),
        scheduler=UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=3.0,
        ),
    ).save_pretrained(path)
    return path


def write_prompt_manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    """One JSONL prompt manifest in the shape ``load_prompt_dataset_index`` reads."""

    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


__all__ = [
    "COSMOS25_TINY_LORA",
    "COSMOS25_TINY_SAMPLING",
    "WAN_TINY_SAMPLING",
    "TinySanaPipeline",
    "build_official_sana_scheduler",
    "cosmos25_eval_config",
    "tiny_sana_online_config",
    "write_prompt_manifest",
    "write_tiny_cosmos25_snapshot",
    "write_tiny_sana_snapshot",
    "write_tiny_wan_snapshot",
]
