"""Preference dataset loading preserves indexing and caller cache placement."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from vrl.trainers.data.preferences import PickAPicPreferenceDataset, PreferenceBatch


@pytest.mark.parametrize("shape", [(2, 4, 3, 3), (2, 8, 3, 3), (2, 6, 3), (2, 6, 1, 3, 3)])
def test_preference_split_rejects_non_rgb_pair_layout(shape):
    batch = PreferenceBatch(pixel_values=torch.zeros(shape), captions=["a", "b"])
    with pytest.raises(ValueError, match=r"\[B, 6, H, W\]"):
        batch.stacked_winner_then_loser()


def test_preference_split_preserves_winner_then_loser_blocks():
    pixels = torch.arange(12.0).reshape(2, 6, 1, 1)
    batch = PreferenceBatch(pixel_values=pixels, captions=["a", "b"])
    winner, loser = batch.split_winner_loser()
    torch.testing.assert_close(winner, pixels[:, :3])
    torch.testing.assert_close(loser, pixels[:, 3:])
    torch.testing.assert_close(
        batch.stacked_winner_then_loser(), torch.cat([pixels[:, :3], pixels[:, 3:]], dim=0)
    )


@pytest.mark.parametrize("max_samples", [None, 2])
def test_hub_loader_materializes_bounded_stream_and_forwards_cache(monkeypatch, max_samples):
    rows = [
        {"caption": "first", "label_0": 1},
        {"caption": "second", "label_0": 0},
        {"caption": "third", "label_0": 1},
    ]
    calls = []

    class Rows(list):
        def __getitem__(self, key):
            if isinstance(key, str):
                return [row[key] for row in self]
            return super().__getitem__(key)

    def load_dataset(name, *, split, cache_dir, streaming):
        calls.append((name, split, cache_dir, streaming))
        return iter(rows) if streaming else Rows(rows)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=load_dataset, Dataset=SimpleNamespace(from_list=Rows)),
    )
    dataset = PickAPicPreferenceDataset.from_hub(
        dataset_name="test/preferences", cache_dir="chosen-cache", max_samples=max_samples
    )
    assert calls == [("test/preferences", "train", "chosen-cache", max_samples is not None)]
    assert len(dataset) == (3 if max_samples is None else max_samples)
    assert dataset._ds[1]["caption"] == "second"
