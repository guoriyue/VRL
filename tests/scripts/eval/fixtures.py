"""Real objects shared by the SANA eval-script tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

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
    """A frozen encoder module that, like an HF model, reports its ``dtype``."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    @property
    def dtype(self) -> torch.dtype:
        return self.proj.weight.dtype


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
            build_tiny_sana_transformer,
        )

        self.transformer = build_tiny_sana_transformer()
        self.vae = build_tiny_autoencoder_kl()
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

    def install(self, monkeypatch: Any, snapshot: Path, *, on_load: Any = None) -> None:
        """Serve this pipeline for ``DiffusionPipeline.from_pretrained(snapshot)``.

        The HF load is the one external boundary; ``on_load`` lets a test act
        while the model is being constructed (e.g. mutate the snapshot tree).
        """

        from diffusers import DiffusionPipeline

        def from_pretrained(path: Any, **_kwargs: Any) -> TinySanaPipeline:
            assert Path(str(path)).resolve() == snapshot.resolve(), path
            self.loads += 1
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

    from tests.models.steps.denoise.fixtures import build_tiny_sana_transformer

    path.mkdir(parents=True, exist_ok=True)
    build_official_sana_scheduler().save_pretrained(path / "scheduler")
    build_tiny_sana_transformer().save_pretrained(path / "transformer")
    (path / "model_index.json").write_text('{"_class_name": "SanaPipeline"}\n', encoding="utf-8")
    return path


def tiny_sana_online_config(
    tmp_path: Path, *, prompts: tuple[str, ...] = ("a cat",), overrides: tuple[str, ...] = ()
) -> Any:
    """The SANA aesthetic online-GRPO preset resolved onto a tiny local snapshot.

    Writes the snapshot and a one-row-per-prompt manifest under ``tmp_path`` and
    returns the merged config: fp32 on CPU, a GPU-less rollout fleet
    (``distributed.resources.rollout.num_gpus=0``), 32x32 two-step sampling and
    a two-sample rollout batch. Pair it with ``TinySanaPipeline.install`` so the
    real family loader serves the tiny pipeline for ``model.path``.
    """

    import json

    snapshot = write_tiny_sana_snapshot(tmp_path / "sana-snapshot")
    manifest = tmp_path / "prompts.jsonl"
    manifest.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    from vrl.config.loading import load_config

    return load_config(
        "experiment/sana/online_grpo_aesthetic",
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
            "rollout.n_samples_per_prompt=2",
            "rollout.prompts_per_batch=1",
            "rollout.samples_per_generation_batch=2",
            "actor.training_microbatch_size=2",
            "distributed.resources.rollout.num_gpus=0",
            *overrides,
        ],
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
        build_tiny_cosmos_transformer,
        build_tiny_wan_vae,
    )

    build_tiny_cosmos_transformer().save_pretrained(path / "transformer")
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
        build_tiny_wan_transformer,
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
        transformer=build_tiny_wan_transformer(),
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
