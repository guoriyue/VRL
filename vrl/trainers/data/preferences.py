"""Preference-pair datasets for offline DPO training.

Currently supports the Pick-a-Pic v2 schema (HuggingFace
``yuvalkirstain/pickapic_v2``):
  * ``jpg_0``, ``jpg_1`` — image bytes
  * ``label_0`` ∈ {0, 0.5, 1} — 1 ⇔ jpg_0 is the winner
  * ``caption`` — text prompt

The collate function returns batches with the convention used by the
reference Diffusion-DPO repo: ``pixel_values`` has shape ``[B, 6, H, W]``
where channels 0:3 are the *winner* image and channels 3:6 are the
*loser*. Trainers split this into ``[2B, 3, H, W]`` before VAE-encoding.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class PreferenceBatch:
    """A collated batch of preference pairs.

    ``pixel_values`` channel layout (along dim=1): winner-then-loser.
    Splitting helper: ``batch.split_winner_loser()`` returns two
    ``[B, 3, H, W]`` tensors.
    """

    pixel_values: torch.Tensor  # [B, 6, H, W]
    captions: list[str]

    @classmethod
    def collate(cls, examples: Iterable[dict[str, Any]]) -> PreferenceBatch:
        """Collate a list of __getitem__ outputs into a ``PreferenceBatch``."""
        items = list(examples)
        pixel_values = torch.stack([e["pixel_values"] for e in items])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        captions = [e["caption"] for e in items]
        return cls(pixel_values=pixel_values, captions=captions)

    def split_winner_loser(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pixel_values.ndim != 4 or self.pixel_values.shape[1] != 6:
            raise ValueError(
                "preference pixel_values must have RGB pair layout [B, 6, H, W]; "
                f"got {tuple(self.pixel_values.shape)}"
            )
        return self.pixel_values[:, :3], self.pixel_values[:, 3:]

    def stacked_winner_then_loser(self) -> torch.Tensor:
        """Return ``[2B, 3, H, W]`` — winner block first, loser second.

        This is the layout consumed by ``diffusion_dpo_loss``.
        """
        winner, loser = self.split_winner_loser()
        return torch.cat([winner, loser], dim=0)


class PickAPicPreferenceDataset(Dataset):
    """Wraps a HuggingFace ``datasets.Dataset`` of Pick-a-Pic samples.

    Keeps decisive labels (0 or 1) on construction. Resizes and applies the
    configured center/random crop and horizontal flip at item access.
    """

    def __init__(
        self,
        hf_dataset: Any,
        *,
        resolution: int = 512,
        random_crop: bool = False,
        no_hflip: bool = False,
    ) -> None:
        from torchvision import transforms

        # Strip indecisive labels (0.5 means tie)
        keep_idx = [i for i, lbl in enumerate(hf_dataset["label_0"]) if lbl in (0, 1)]
        if len(keep_idx) < len(hf_dataset):
            hf_dataset = hf_dataset.select(keep_idx)
        self._ds = hf_dataset

        ops: list[Any] = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        ops.append(
            transforms.RandomCrop(resolution) if random_crop else transforms.CenterCrop(resolution)
        )
        if not no_hflip:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
        ]
        self._tx = transforms.Compose(ops)

    @classmethod
    def from_hub(
        cls,
        split: str = "train",
        *,
        cache_dir: str | None = None,
        max_samples: int | None = None,
        resolution: int = 512,
        random_crop: bool = False,
        no_hflip: bool = False,
        dataset_name: str = "yuvalkirstain/pickapic_v2",
    ) -> PickAPicPreferenceDataset:
        """Load an indexable preference dataset for a shuffled DataLoader.

        Requires ``datasets`` and ``torchvision``. With ``max_samples``, stream
        a bounded prefix into an indexable Dataset; the limit applies before
        filtering indecisive labels. Without it, load the split directly as an
        indexable Dataset. Both paths support the same shuffled DataLoader.
        The streaming path needs a plain split name such as ``"train"`` rather
        than slice syntax such as ``"train[:N]"``.
        """
        import itertools

        from datasets import Dataset, load_dataset

        if max_samples is not None:
            # Stream the first `max_samples` rows, then materialise a map-style
            # Dataset. A plain `load_dataset(split="train[:N]")` (or a post-hoc
            # `.select`) downloads EVERY shard first just to keep N rows — useless on
            # a 190 GB dataset. Streaming reads only the shards actually consumed.
            stream = load_dataset(dataset_name, split=split, cache_dir=cache_dir, streaming=True)
            rows = list(itertools.islice(stream, max_samples))
            ds = Dataset.from_list(rows)
        else:
            ds = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                streaming=False,
            )
        return cls(
            ds,
            resolution=resolution,
            random_crop=random_crop,
            no_hflip=no_hflip,
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        row = self._ds[idx]
        # Winner is whichever of jpg_0 / jpg_1 has label==1
        if row["label_0"] == 1:
            w_bytes, l_bytes = row["jpg_0"], row["jpg_1"]
        else:
            w_bytes, l_bytes = row["jpg_1"], row["jpg_0"]
        winner = self._tx(Image.open(io.BytesIO(w_bytes)).convert("RGB"))
        loser = self._tx(Image.open(io.BytesIO(l_bytes)).convert("RGB"))
        # Stack on channel dim — winner-then-loser convention
        pix = torch.cat([winner, loser], dim=0)  # [6, H, W]
        return {"pixel_values": pix, "caption": row["caption"]}
