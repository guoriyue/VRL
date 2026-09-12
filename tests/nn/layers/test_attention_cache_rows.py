"""Tests for AR per-row cache helpers."""

from __future__ import annotations

import builtins
import importlib.util
import sys

import pytest
import torch
from transformers.cache_utils import DynamicCache

from vrl.nn.layers.attention.cache_rows import ARCacheRows


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("missing transformers", name="transformers"),
        ModuleNotFoundError("missing internal dependency", name="transformers.cache_utils"),
        ModuleNotFoundError("missing dependency", name="some_dependency"),
        ImportError("cannot import name DynamicCache"),
    ],
)
def test_optional_cache_import_only_tolerates_absent_transformers(monkeypatch, error):
    from vrl.nn.layers.attention import cache_rows

    module_name = "_vrl_test_cache_rows_import"
    spec = importlib.util.spec_from_file_location(module_name, cache_rows.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    original_import = builtins.__import__

    def import_with_failure(name, *args, **kwargs):
        if name == "transformers.cache_utils":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_failure)
    if isinstance(error, ModuleNotFoundError) and error.name == "transformers":
        spec.loader.exec_module(module)
        assert module.Cache is None
        values = torch.arange(4).reshape(2, 2)
        rows = module.ARCacheRows.from_batched(values, 2)
        torch.testing.assert_close(rows.gather([1, 0]), values.flip(0))
    else:
        with pytest.raises(type(error)) as caught:
            spec.loader.exec_module(module)
        assert caught.value is error


def test_ar_split_and_concat_rows_preserve_nested_kv_order() -> None:
    """Splitting a batched cache into rows and concatenating rows back preserves the nested
    past_key_values structure and lets rows be reordered.
    """
    key = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    value = key + 100
    cache = {
        "past_key_values": ((key, value),),
        "last_hidden": torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5),
    }

    rows = ARCacheRows.split_batched(cache, 3)
    assert len(rows) == 3
    assert torch.equal(rows[1]["past_key_values"][0][0], key[1:2])
    assert torch.equal(rows[2]["last_hidden"], cache["last_hidden"][2:3])

    merged = ARCacheRows.concatenate([rows[2], rows[0]])
    assert torch.equal(merged["past_key_values"][0][0], torch.cat([key[2:3], key[0:1]]))
    assert torch.equal(
        merged["last_hidden"],
        torch.cat([cache["last_hidden"][2:3], cache["last_hidden"][0:1]]),
    )


@pytest.mark.parametrize("nested", [False, True])
def test_ar_split_rejects_scalar_tensor_without_batch_dimension(nested) -> None:
    value = torch.tensor(1.0)
    if nested:
        value = {"cache": value}
    with pytest.raises(ValueError, match="leading batch dimension"):
        ARCacheRows.split_batched(value, 1)


def test_ar_scatter_scalar_error_preserves_existing_row() -> None:
    original = torch.ones(1, 2)
    rows = ARCacheRows([original])
    with pytest.raises(ValueError, match="leading batch dimension"):
        rows.scatter([0], torch.tensor(1.0))
    assert rows[0] is original


def test_ar_split_rows_rejects_wrong_batch_size() -> None:
    with pytest.raises(ValueError, match="cannot split tensor"):
        ARCacheRows.split_batched(torch.zeros(2, 4), 3)


def test_ar_cache_rows_gather_and_scatter_nested_values() -> None:
    """``ARCacheRows`` gathers rows in the requested order, scatters replacements into nested
    values, and copies one row over another through ``select_rows`` / ``scatter_rows``.
    """
    key = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    value = key + 100
    cache = {
        "past_key_values": ((key, value),),
        "last_hidden": torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5),
    }
    rows = ARCacheRows.from_batched(cache, 3, owner="test.cache")

    gathered = rows.gather([2, 0])
    assert torch.equal(
        gathered["past_key_values"][0][0],
        torch.cat([key[2:3], key[:1]]),
    )

    replacement = {
        "past_key_values": ((torch.full((2, 2, 4), -1.0), torch.full((2, 2, 4), 7.0)),),
        "last_hidden": torch.full((2, 5), 3.0),
    }
    rows.scatter([0, 2], replacement)

    updated = rows.gather([0, 2])
    assert torch.equal(
        updated["past_key_values"][0][0],
        replacement["past_key_values"][0][0],
    )
    assert torch.equal(updated["last_hidden"], replacement["last_hidden"])
    assert torch.equal(rows.gather([1])["past_key_values"][0][0], key[1:2])

    rows.scatter_rows([1], rows.select_rows([0]))
    assert torch.equal(
        rows.gather([1])["last_hidden"],
        replacement["last_hidden"][:1],
    )


def test_ar_cache_rows_rejects_invalid_indices() -> None:
    rows = ARCacheRows.from_batched(torch.zeros(2, 4), 2, owner="test.cache")

    with pytest.raises(IndexError, match=r"test\.cache"):
        rows.gather([2])

    with pytest.raises(ValueError, match="received 1 rows"):
        rows.scatter_rows([0, 1], rows.select_rows([0]))


def test_ar_cache_helpers_preserve_transformers_dynamic_cache_objects() -> None:
    """Row split / concat keep ``DynamicCache`` objects as ``DynamicCache`` (transformers 5 has no
    legacy-tuple conversion) with the K/V content preserved.
    """
    key = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    value = key + 100
    cache = DynamicCache()
    cache.update(key, value, layer_idx=0)

    rows = ARCacheRows.split_batched(cache, 2)
    assert len(rows) == 2
    assert all(isinstance(row, DynamicCache) for row in rows)
    # Read K/V back through the module's own 4/5-tolerant reader:
    # transformers 5 removed DynamicCache.to_legacy_cache.
    assert torch.equal(ARCacheRows._cache_kv_pairs(rows[0])[0][0], key[:1])
    assert torch.equal(ARCacheRows._cache_kv_pairs(rows[1])[0][1], value[1:2])

    merged = ARCacheRows.concatenate([rows[1], rows[0]])
    assert isinstance(merged, DynamicCache)
    merged_key, merged_value = ARCacheRows._cache_kv_pairs(merged)[0]
    assert torch.equal(merged_key, torch.cat([key[1:2], key[:1]], dim=0))
    assert torch.equal(
        merged_value,
        torch.cat([value[1:2], value[:1]], dim=0),
    )


@pytest.mark.parametrize("index", [0.9, True, "0"])
@pytest.mark.parametrize("operation", ["gather", "scatter", "scatter_rows"])
def test_cache_rows_reject_non_integer_indices_without_mutation(index, operation) -> None:
    original = torch.tensor([[10.0], [20.0]])
    rows = ARCacheRows.from_batched(original, 2)
    with pytest.raises(ValueError, match="row indices must be integers"):
        if operation == "gather":
            rows.gather([index])
        elif operation == "scatter":
            rows.scatter([0, index], torch.zeros(2, 1))
        else:
            rows.scatter_rows([0, index], [torch.zeros(1, 1), torch.zeros(1, 1)])
    assert torch.equal(rows.gather([0, 1]), original)


@pytest.mark.parametrize(
    "rows",
    [
        [{"a": 1}, {"a": 1, "b": 2}],
        [{"a": 1, "b": 2}, {"a": 1}],
        [[], [1]],
        [[1], []],
        [(), (1,)],
        [(1,), ()],
        [[1], (1,)],
        [(1,), [1]],
        [{}, []],
    ],
)
def test_concat_rejects_different_row_structures(rows) -> None:
    with pytest.raises(ValueError, match="cannot concatenate AR"):
        ARCacheRows.concatenate(rows)


def test_concat_mapping_order_does_not_change_key_alignment() -> None:
    rows = [
        {"key": torch.tensor([[1]]), "value": torch.tensor([[2]])},
        {"value": torch.tensor([[4]]), "key": torch.tensor([[3]])},
    ]
    merged = ARCacheRows.concatenate(rows)
    assert list(merged) == ["key", "value"]
    torch.testing.assert_close(merged["key"], torch.tensor([[1], [3]]))
    torch.testing.assert_close(merged["value"], torch.tensor([[2], [4]]))


@pytest.mark.parametrize("requested_rows", [1, 3])
def test_dynamic_cache_split_requires_exact_batch_size(requested_rows) -> None:
    cache = DynamicCache()
    key = torch.zeros(2, 1, 3, 4)
    cache.update(key, key.clone(), layer_idx=0)

    with pytest.raises(ValueError, match="cannot split tensor with batch=2"):
        ARCacheRows.split_batched(cache, requested_rows)


@pytest.mark.parametrize("invalid_part", ["key", "value"])
def test_dynamic_cache_split_validates_every_layer_tensor(invalid_part) -> None:
    cache = DynamicCache()
    cache.update(torch.zeros(2, 1, 3, 4), torch.zeros(2, 1, 3, 4), layer_idx=0)
    cache.update(
        torch.zeros(1 if invalid_part == "key" else 2, 1, 3, 4),
        torch.zeros(1 if invalid_part == "value" else 2, 1, 3, 4),
        layer_idx=1,
    )

    with pytest.raises(ValueError, match="cannot split tensor with batch=1 into 2 rows"):
        ARCacheRows.split_batched(cache, 2)


@pytest.mark.parametrize("container", ["tensor", "mapping", "hf_cache"])
def test_concat_rejects_implicit_cache_dtype_promotion(container) -> None:
    rows = [torch.ones(1, 2, 3, 4, dtype=dtype) for dtype in (torch.float16, torch.float32)]
    if container == "mapping":
        rows = [{"past": ((row, row),)} for row in rows]
    elif container == "hf_cache":
        caches = []
        for row in rows:
            cache = DynamicCache()
            cache.update(row, row, 0)
            caches.append(cache)
        rows = caches
    with pytest.raises(ValueError, match="dtype"):
        ARCacheRows.concatenate(rows)
