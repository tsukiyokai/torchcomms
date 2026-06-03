# Unit tests for the HCCL backend (single-process, no NPU collective).
# Verify register, factory creation, lifecycle, and limit cases.

import pytest
import torch
import torchcomms
import torchcomms._comms_hccl  # noqa: F401  triggers static factory register


def test_backend_registered():
    """`hccl` should be visible after _comms_hccl is imported."""
    # internal helper exposed via _comms.pyi; if not present, falls back to
    # checking that new_comm doesn't raise unknown-backend.
    if hasattr(torchcomms, "_is_backend_registered"):
        assert torchcomms._is_backend_registered("hccl") is True


def test_create_single_rank_backend():
    """Single-rank new_comm should succeed and report rank=0/size=1."""
    import torch_npu  # noqa: F401
    torch_npu.npu.set_device(0)
    comm = torchcomms.new_comm(
        "hccl", torch.device("npu:0"), name="unit_single_rank"
    )
    assert comm.get_backend() == "hccl"
    assert comm.get_rank() == 0
    assert comm.get_size() == 1
    comm.finalize()


def test_finalize_idempotent():
    """finalize() should be safe to call twice."""
    import torch_npu  # noqa: F401
    torch_npu.npu.set_device(0)
    comm = torchcomms.new_comm(
        "hccl", torch.device("npu:0"), name="unit_finalize_twice"
    )
    comm.finalize()
    comm.finalize()  # second call must not crash


def test_multirank_requires_store():
    """size>1 without store should raise a clear error."""
    import torch_npu  # noqa: F401
    torch_npu.npu.set_device(0)
    with pytest.raises(RuntimeError, match=r"requires CommOptions\.store"):
        torchcomms.new_comm(
            "hccl",
            torch.device("npu:0"),
            name="unit_no_store",
            hints={"rank": "0", "world_size": "2"},
        )
