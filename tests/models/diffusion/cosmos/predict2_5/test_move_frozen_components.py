"""Driver offload must also move the hidden frozen pipeline components.

Cosmos stores its VAE/text-encoder on ``_pipeline`` via object.__setattr__, so
they are not registered submodules and a plain ``model.to(...)`` leaves them on
the GPU. ``move_frozen_components`` moves them too, so the colocated rollout
offload actually frees the fp32 VAE (~0.5-1 GB) instead of leaving it resident.
"""

from __future__ import annotations

from vrl.models.diffusion.cosmos.predict2_5.model import CosmosPredict25Model


class _Component:
    def __init__(self) -> None:
        self.device = None

    def to(self, device: object) -> "_Component":
        self.device = device
        return self


class _Pipeline:
    def __init__(self, *, with_text_encoder: bool) -> None:
        self.vae = _Component()
        self.text_encoder = _Component() if with_text_encoder else None


class _FakeModel:
    """Duck-types just the `_pipeline` attribute the method touches."""

    def __init__(self, *, with_text_encoder: bool) -> None:
        self._pipeline = _Pipeline(with_text_encoder=with_text_encoder)


def test_moves_vae_and_text_encoder_both_directions() -> None:
    m = _FakeModel(with_text_encoder=True)
    CosmosPredict25Model.move_frozen_components(m, "cpu")
    assert m._pipeline.vae.device == "cpu"
    assert m._pipeline.text_encoder.device == "cpu"
    # restore direction
    CosmosPredict25Model.move_frozen_components(m, "cuda:0")
    assert m._pipeline.vae.device == "cuda:0"
    assert m._pipeline.text_encoder.device == "cuda:0"


def test_skips_absent_text_encoder() -> None:
    # synthetic-prompt-embeds path: text_encoder is None — must not raise.
    m = _FakeModel(with_text_encoder=False)
    CosmosPredict25Model.move_frozen_components(m, "cpu")
    assert m._pipeline.vae.device == "cpu"
    assert m._pipeline.text_encoder is None
