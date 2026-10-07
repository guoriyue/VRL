"""Export a layered RL checkpoint as a standard PEFT adapter without rounding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def export_checkpoint(root, *, checkpoint, output):
    from accelerate import init_empty_weights
    from diffusers import QwenImageTransformer2DModel
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file

    plan = json.loads((root / "plan.json").read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    exported = {}
    for name, value in state.items():
        parts = name.split(".")
        assert parts[:1] == ["transformer_blocks"] and parts[1].isdigit()
        assert parts[2] == "block" and parts[-2:] == ["default", "weight"]
        assert parts[-3] in {"lora_A", "lora_B"}
        assert torch.isfinite(value).all()
        key = "base_model.model." + ".".join(parts[:2] + parts[3:-2] + ["weight"])
        assert key not in exported
        exported[key] = value.contiguous()
    output.mkdir(parents=True, exist_ok=False)
    config = LoraConfig(**plan["lora"], inference_mode=True)
    config.save_pretrained(output)
    weights = output / "adapter_model.safetensors"
    save_file(exported, weights)
    restored = load_file(weights)
    assert restored.keys() == exported.keys()
    assert all(torch.equal(restored[name], value) for name, value in exported.items())

    # Check PEFT names and shapes against the full architecture without
    # allocating a second copy of the frozen base during ongoing evaluation.
    architecture = QwenImageTransformer2DModel.load_config(
        plan["base_path"], subfolder="transformer", local_files_only=True
    )
    with init_empty_weights():
        base = QwenImageTransformer2DModel.from_config(architecture)
        model = get_peft_model(base, config)
    expected = {
        name.replace(".default.weight", ".weight"): value
        for name, value in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    assert expected.keys() == exported.keys()
    assert all(expected[name].shape == value.shape for name, value in exported.items())
    record = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "base_revision": plan["base_revision"],
        "tensor_count": len(exported),
        "parameter_count": sum(value.numel() for value in exported.values()),
        "serialization_exact": True,
        "peft_keys_and_shapes_match": True,
        "scope": "Lossless format export and PEFT keys/shapes verified on the full meta architecture. Runtime loading, forward parity and image quality are separate checks.",
    }
    (output / "export_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_checkpoint(args.root, checkpoint=args.checkpoint, output=args.output)


if __name__ == "__main__":
    main()
