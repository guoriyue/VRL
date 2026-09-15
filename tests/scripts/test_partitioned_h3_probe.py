import pytest

from vrl.scripts.generation.partitioned_h3_probe import layer_owners


def test_layer_owners_assigns_contiguous_balanced_partitions():
    assert layer_owners(5, [2, 3]) == (2, 2, 2, 3, 3)
    assert layer_owners(1, [1]) == (1,)
    assert layer_owners(64, [2, 3]) == (2,) * 32 + (3,) * 32


@pytest.mark.parametrize("count,devices", [(0, [0]), (1, [0, 1]), (4, []), (4, [1, 1]), (4, [-1])])
def test_layer_owners_rejects_invalid_partition(count, devices):
    with pytest.raises(ValueError):
        layer_owners(count, devices)
