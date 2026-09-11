import pytest
import torch

from vrl.generation.weight_transfer import (
    StagedWeightTransfer,
    iter_weight_chunks,
    weight_manifest,
)


def test_chunks_are_independent_bounded_storage_and_reassemble_exactly():
    state = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
        "bias": torch.tensor([float("nan"), -0.0], dtype=torch.bfloat16),
        "empty": torch.empty(0),
    }
    staged = StagedWeightTransfer("one", 2, weight_manifest(state))
    chunks = list(iter_weight_chunks(state, 9))
    for name, offset, value in chunks:
        assert value.untyped_storage().nbytes() <= 9
        staged.receive("one", (name, offset, value))
        value.fill_(42)  # receiver must own a copy, not a Ray/object-store view
    result = staged.finish("one")
    for key in state:
        expected = state[key].contiguous().reshape(-1).view(torch.uint8)
        actual = result[key].reshape(-1).view(torch.uint8)
        assert torch.equal(actual, expected)


def test_partial_duplicate_and_foreign_chunks_do_not_finish():
    state = {"weight": torch.ones(4)}
    staged = StagedWeightTransfer("one", 2, weight_manifest(state))
    first, second = list(iter_weight_chunks(state, 8))
    with pytest.raises(ValueError, match="another transfer"):
        staged.receive("other", first)
    with pytest.raises(ValueError, match="out-of-order"):
        staged.receive("one", second)
    staged.receive("one", first)
    with pytest.raises(ValueError, match="out-of-order"):
        staged.receive("one", first)
    with pytest.raises(ValueError, match="incomplete"):
        staged.finish("one")
    staged.receive("one", second)
    assert torch.equal(staged.finish("one")["weight"], state["weight"])


def test_worker_staging_does_not_change_live_state_or_ack_a_partial_transfer():
    from tests.generation.execution.test_worker_versioned_slots import _core, _ReadbackModel

    core = _core(_ReadbackModel(), versioned_weight_sync=False)
    state = {"transformer.weight": torch.ones(2, 2)}
    assert core.begin_weight_transfer(weight_manifest(state), "one", 2) == 2
    core.receive_weight_chunk(next(iter_weight_chunks(state, 8)), "one")
    with pytest.raises(ValueError, match="incomplete"):
        core.commit_weight_transfer("one")
    assert core._policy_version == 1
    assert torch.all(core.executor.model.module.weight == -99)
    core.abort_weight_transfer("one")
    assert core._weight_transfer is None
    core.begin_weight_transfer(weight_manifest(state), "two", 2)
    for chunk in iter_weight_chunks(state, 8):
        core.receive_weight_chunk(chunk, "two")
    assert core.commit_weight_transfer("two", verify_content=True) == 2
    assert core._weight_transfer is None
    assert torch.equal(core.executor.model.module.weight, state["transformer.weight"])


def test_bucket_setting_survives_public_runtime_projection():
    from vrl.config.schema import RolloutRuntimeSection
    from vrl.generation.ray.config import RolloutWorkerConfig

    public = RolloutRuntimeSection(update_weight_buffer_size=1024)
    assert RolloutWorkerConfig.from_public_section(public).update_weight_buffer_size == 1024
    assert RolloutWorkerConfig.from_public_section({}).update_weight_buffer_size is None
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            RolloutRuntimeSection(update_weight_buffer_size=value)


def test_receiver_staging_stays_on_cpu_under_a_different_default_device():
    state = {"weight": torch.ones(2), "empty": torch.empty(0)}
    staged = StagedWeightTransfer("one", 1, weight_manifest(state))
    chunks = list(iter_weight_chunks(state, 8))
    with torch.device("meta"):
        for chunk in chunks:
            staged.receive("one", chunk)
        result = staged.finish("one")
    assert all(tensor.device.type == "cpu" for tensor in result.values())
    assert torch.equal(result["weight"], state["weight"])


def test_buckets_pack_small_tensors_without_exceeding_tensor_byte_limit():
    from vrl.generation.weight_transfer import iter_weight_buckets

    state = {f"weight-{i}": torch.ones(2) for i in range(5)}
    buckets = list(iter_weight_buckets(state, 16))
    assert [len(bucket) for bucket in buckets] == [2, 2, 1]
    assert all(
        sum(value.numel() * value.element_size() for _, _, value in bucket) <= 16
        for bucket in buckets
    )


def test_old_weight_buffer_option_is_rejected():
    from vrl.config.schema import RolloutRuntimeSection

    with pytest.raises(ValueError):
        RolloutRuntimeSection(weight_sync_bucket_bytes=1024)


@pytest.mark.parametrize("version", [2.9, "2", True, -1])
def test_invalid_transfer_version_cannot_modify_worker(version):
    from tests.generation.execution.test_worker_versioned_slots import _core, _ReadbackModel

    core = _core(_ReadbackModel(), versioned_weight_sync=False)
    state = {"transformer.weight": torch.ones(2, 2)}
    with pytest.raises(ValueError, match="policy_version"):
        core.begin_weight_transfer(weight_manifest(state), "invalid", version)
    assert core._weight_transfer is None
    with pytest.raises(ValueError, match="policy_version"):
        core.update_weights(state, version)
    assert core._policy_version == 1
    assert torch.all(core.executor.model.module.weight == -99)
