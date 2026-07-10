"""Shared fixtures for the interface-contract tests.

Both ``test_runtime_model_contract`` and ``test_replay_model_contract`` need to
run the RuntimeModel / ReplayModel protocol contract over every *registered*
family so a newly registered family cannot silently skip the contract. The
rollout registry (``vrl.rollouts.families``) is the single source of truth for
*which* families exist; it does not, however, expose the concrete model class a
family builds (it stores builder/extractor/executor import paths only — see
``vrl/rollouts/families/registry.py``). The runtime-model class is what
``build_<family>_runtime_bundle`` constructs and the replay-model class is what
``build_<family>_replay_runtime_bundle`` constructs, both living in the family's
``model`` module.

``FAMILY_MODEL_CLASSES`` is therefore the test-side binding from family →
(runtime-model class, replay-model class). ``registered_family_model_classes``
asserts the binding's keys are exactly the registry's families, so registering a
new family without adding it here fails loudly instead of leaving the new family
unprotected by the contract.
"""

from __future__ import annotations

from vrl.rollouts.families import registered_rollout_families
from vrl.utils.config import import_from_path

# family -> (runtime-model import path, replay-model import path).
# Sourced from the family builders (``build_<family>_runtime_bundle`` /
# ``build_<family>_replay_runtime_bundle``). Importing these classes is
# CPU-only and weight-free — instantiating them is NOT (it loads
# checkpoints/GPU), so the contract tests check the *class*, never an instance.
FAMILY_MODEL_CLASSES: dict[str, tuple[str, str]] = {
    "sd3_5": (
        "vrl.models.diffusion.sd3_5.model:SD3_5Model",
        "vrl.models.diffusion.sd3_5.model:SD3_5ReplayModel",
    ),
    "hunyuan_video": (
        "vrl.models.diffusion.hunyuan_video.model:HunyuanVideoModel",
        "vrl.models.diffusion.hunyuan_video.model:HunyuanVideoReplayModel",
    ),
    "hunyuan_image": (
        "vrl.models.diffusion.hunyuan_image.model:HunyuanImageModel",
        "vrl.models.diffusion.hunyuan_image.model:HunyuanImageReplayModel",
    ),
    "glm_image": (
        "vrl.models.ar.glm_image.model:GlmImageModel",
        "vrl.models.ar.glm_image.model:GlmImageReplayModel",
    ),
    "emu3": (
        "vrl.models.ar.emu3.model:Emu3Model",
        "vrl.models.ar.emu3.model:Emu3ReplayModel",
    ),
    "llamagen": (
        "vrl.models.ar.llamagen.model:LlamaGenModel",
        "vrl.models.ar.llamagen.model:LlamaGenReplayModel",
    ),
    "pixart_sigma": (
        "vrl.models.diffusion.pixart_sigma.model:PixArtSigmaModel",
        "vrl.models.diffusion.pixart_sigma.model:PixArtSigmaReplayModel",
    ),
    "cogvideox": (
        "vrl.models.diffusion.cogvideox.model:CogVideoXModel",
        "vrl.models.diffusion.cogvideox.model:CogVideoXReplayModel",
    ),
    "mochi": (
        "vrl.models.diffusion.mochi.model:MochiModel",
        "vrl.models.diffusion.mochi.model:MochiReplayModel",
    ),
    "lumina2": (
        "vrl.models.diffusion.lumina2.model:Lumina2Model",
        "vrl.models.diffusion.lumina2.model:Lumina2ReplayModel",
    ),
    "sana": (
        "vrl.models.diffusion.sana.model:SanaModel",
        "vrl.models.diffusion.sana.model:SanaReplayModel",
    ),
    "flux": (
        "vrl.models.diffusion.flux.model:FluxModel",
        "vrl.models.diffusion.flux.model:FluxReplayModel",
    ),
    "qwen_image": (
        "vrl.models.diffusion.qwen_image.model:QwenImageModel",
        "vrl.models.diffusion.qwen_image.model:QwenImageReplayModel",
    ),
    "wan_2_1": (
        "vrl.models.diffusion.wan_2_1.model:WanT2VDiffusersModel",
        "vrl.models.diffusion.wan_2_1.model:WanT2VReplayModel",
    ),
    "wan_2_1_i2v": (
        "vrl.models.diffusion.wan_2_1.model:WanI2VDiffusersModel",
        "vrl.models.diffusion.wan_2_1.model:WanI2VReplayModel",
    ),
    "cosmos-predict2": (
        "vrl.models.diffusion.cosmos.predict2.model:CosmosPredict2Model",
        "vrl.models.diffusion.cosmos.predict2.model:CosmosPredict2ReplayModel",
    ),
    "cosmos-predict2.5": (
        "vrl.models.diffusion.cosmos.predict2_5.model:CosmosPredict25Model",
        "vrl.models.diffusion.cosmos.predict2_5.model:CosmosPredict25ReplayModel",
    ),
    "cosmos-predict2-anima": (
        "vrl.models.diffusion.cosmos.anima.model:AnimaModel",
        "vrl.models.diffusion.cosmos.anima.model:AnimaReplayModel",
    ),
    "cosmos3": (
        "vrl.models.diffusion.cosmos.cosmos3.model:Cosmos3Model",
        "vrl.models.diffusion.cosmos.cosmos3.model:Cosmos3ReplayModel",
    ),
    "echo": (
        "vrl.models.diffusion.echo.model:EchoModel",
        "vrl.models.diffusion.echo.model:EchoReplayModel",
    ),
    "janus_pro": (
        "vrl.models.ar.janus_pro.model:JanusProModel",
        "vrl.models.ar.janus_pro.model:JanusProReplayModel",
    ),
    "janus_pro_r1": (
        "vrl.models.ar.janus_pro.model:JanusProModel",
        "vrl.models.ar.janus_pro.model:JanusProReplayModel",
    ),
    "nextstep_1": (
        "vrl.models.ar.nextstep_1.model:NextStep1Model",
        "vrl.models.ar.nextstep_1.model:NextStep1ReplayModel",
    ),
}


def registered_family_model_classes() -> dict[str, tuple[type, type]]:
    """Resolve every registered family's (runtime-model, replay-model) classes.

    The family keys come from the registry, not a hand-written list, so a
    newly registered family that is missing from ``FAMILY_MODEL_CLASSES`` raises
    here rather than silently escaping the protocol contract.
    """

    families = registered_rollout_families()
    missing = [f for f in families if f not in FAMILY_MODEL_CLASSES]
    if missing:
        raise AssertionError(
            "FAMILY_MODEL_CLASSES is missing registered families "
            f"{missing}; add their runtime/replay model classes so the "
            "RuntimeModel/ReplayModel contract covers them.",
        )
    stale = [f for f in FAMILY_MODEL_CLASSES if f not in families]
    if stale:
        raise AssertionError(
            f"FAMILY_MODEL_CLASSES has unregistered families {stale}; "
            "remove them so the contract matrix tracks the registry.",
        )
    return {
        family: (
            import_from_path(runtime_path),
            import_from_path(replay_path),
        )
        for family, (runtime_path, replay_path) in FAMILY_MODEL_CLASSES.items()
    }
