"""Preference dataset loading preserves indexing and caller cache placement."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from vrl.trainers.data.preferences import PickAPicPreferenceDataset


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
