# PyHcclBackend v2: ctypes-direct HCCL backend for torchcomms.
#
# v1 (git history) forwarded all 22 virtual methods to torch_npu's
# ProcessGroupHCCL. v2 dispatches collectives directly to libhccl.so via
# hccl_ffi.py — collective path is 100% torch_npu-free.
#
# torch_npu is still imported because:
#   1) PrivateUse1 device registration (npu:N) is unavoidable for torch
#      tensors to live on NPU
#   2) we reuse torch_npu.npu.current_stream() so HCCL ops serialize with
#      prior torch ops on the same tensor with no extra sync
#
# Bootstrap uses c10d Store (TCPStore from dist.init_process_group) — rank
# 0 produces HcclRootInfo (4108 bytes) and broadcasts via store.set/get.
#
# Usage (rank R of N):
#     import torch, torch.distributed as dist, torch_npu, torchcomms
#     from hccl_pyend import PyHcclBackend, set_default_pg
#
#     dist.init_process_group(backend="gloo", rank=R, world_size=N)
#     set_default_pg(dist.distributed_c10d._get_default_group())
#     torchcomms.register_backend("py_hccl", PyHcclBackend)
#     comm = torchcomms.new_comm("py_hccl", torch.device(f"npu:{R}"), name="demo")
#     t = torch.ones(8, device=f"npu:{R}")
#     comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False)

from __future__ import annotations

import ctypes
import os
import sys

import torch
import torch.distributed as dist
import torchcomms
from torchcomms._comms import TorchCommBackend

# torch_npu must be imported by the caller before us; re-import here is
# idempotent and gives us .npu.current_stream() access.
import torch_npu  # noqa: F401

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hccl_ffi as ffi
else:
    from . import hccl_ffi as ffi

# ====
# Module state
# ====
_DEFAULT_PG = None


def set_default_pg(pg):
    global _DEFAULT_PG
    _DEFAULT_PG = pg


# ====
# Async work handle
# ====
class _StreamSyncWork:
    """Minimal Work object: wait() drains the stream, is_completed() reports.

    Capture-aware: if the stream is in an active aclmdlRICapture, wait()
    is a no-op — calling aclrtSynchronizeStream during capture would
    invalidate the recording (it's a host-side wait, not a graph node).
    """

    def __init__(self, stream):
        self._stream = stream
        self._done = False

    def wait(self):
        if self._done:
            return
        if ffi.stream_is_capturing(self._stream):
            self._done = True
            return
        ffi.acl_check(ffi.aclrtSynchronizeStream(self._stream),
                      "aclrtSynchronizeStream")
        self._done = True

    def is_completed(self):
        return self._done


def _maybe_async(stream, async_op):
    """Return a work handle (async) or sync the stream now (sync).

    Capture-aware: if the stream is being captured into an aclmdlRI,
    sync mode degrades to async — aclrtSynchronizeStream would break the
    capture. Caller is responsible for ordering when running under capture.
    """
    if async_op or ffi.stream_is_capturing(stream):
        return _StreamSyncWork(stream)
    ffi.acl_check(ffi.aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")
    return None


def _reject_in_capture(stream, op_name: str, alternative: str) -> None:
    """Refuse to run an emulation path that does host-side memcpy inside an
    active aclmdlRICapture — the memcpy cannot be recorded as a graph node
    and the captured graph would silently produce stale data on replay."""
    if ffi.stream_is_capturing(stream):
        raise RuntimeError(
            f"PyHcclBackend.{op_name} is emulated via host-side copy and "
            f"cannot be safely captured into an NPUGraph. Use "
            f"{alternative} (or skip this op in the captured region)."
        )


# ====
# Bootstrap helper
# ====
def _bootstrap_comm(rank: int, world_size: int, store, name: str):
    """Rank 0 generates root info, broadcasts via c10d store, all init comm."""
    # Store key MUST be identical across ranks — `name` is the comm-group
    # identifier (caller picks one shared label per group).
    key = f"py_hccl_root_info_{name}"
    if rank == 0:
        root_info = ctypes.create_string_buffer(ffi.ROOT_INFO_BYTES)
        ffi.check(
            ffi.HcclGetRootInfo(ctypes.cast(root_info, ctypes.c_void_p)),
            "HcclGetRootInfo",
        )
        store.set(key, bytes(root_info.raw))
        root_bytes = bytes(root_info.raw)
    else:
        # store.get blocks until rank 0 sets the key
        root_bytes = bytes(store.get(key))
    assert len(root_bytes) == ffi.ROOT_INFO_BYTES, (
        f"root info size mismatch: got {len(root_bytes)}, "
        f"expected {ffi.ROOT_INFO_BYTES}"
    )
    root_info = ctypes.create_string_buffer(root_bytes, ffi.ROOT_INFO_BYTES)
    comm = ffi.HcclComm()
    ffi.check(
        ffi.HcclCommInitRootInfo(
            world_size,
            ctypes.cast(root_info, ctypes.c_void_p),
            rank,
            ctypes.byref(comm),
        ),
        "HcclCommInitRootInfo",
    )
    return comm


# ====
# Backend
# ====
class PyHcclBackend(TorchCommBackend):
    """torchcomms backend dispatching directly to libhccl.so via ctypes.

    register_backend wants a no-arg class, so we read _DEFAULT_PG from module
    state. Caller must set_default_pg() between init_process_group() and
    register_backend(). The PG is used for rank/size inquiry and store
    rendezvous — its collective methods are never called.
    """

    def __init__(self):
        super().__init__()
        if _DEFAULT_PG is None:
            raise RuntimeError(
                "PyHcclBackend needs a default ProcessGroup. "
                "Call set_default_pg() after init_process_group()."
            )
        self._pg = _DEFAULT_PG
        self._rank = self._pg.rank()
        self._world_size = self._pg.size()
        self._store = dist.distributed_c10d._get_default_store()
        self._comm = None
        self._device = None
        self._name = ""

    # ---- lifecycle ----
    def init(self, device, name, options):
        self._device = device
        # name MUST be identical across ranks — it's the bootstrap key.
        # If caller passes per-rank names the rendezvous deadlocks.
        self._name = name or "py_hccl_default"
        # Mirror native TorchCommHCCL.cpp:82-84 — HCCL needs an active acl
        # device on this thread before HcclCommInitRootInfo. torch_npu's
        # set_device is per-thread state and doesn't carry into HCCL's
        # internal context across the trampoline boundary.
        if device.index is not None:
            ffi.acl_check(
                ffi.aclrtSetDevice(int(device.index)), "aclrtSetDevice"
            )
        self._comm = _bootstrap_comm(
            self._rank, self._world_size, self._store, self._name
        )

    def finalize(self):
        if self._comm is not None:
            ffi.HcclCommDestroy(self._comm)
            self._comm = None

    def get_rank(self):
        return self._rank

    def get_size(self):
        return self._world_size

    def get_backend_name(self):
        return "py_hccl"

    def get_comm_name(self):
        return self._name or "py_hccl_default"

    # ---- internal helpers ----
    def _stream(self):
        return ffi.current_npu_stream()

    def _hccl_dtype(self, t):
        return ffi.torch_dtype_to_hccl(t.dtype)

    # ---- reductions ----
    def all_reduce(self, tensor, op, async_op):
        s = self._stream()
        ffi.check(ffi.HcclAllReduce(
            ffi.tensor_void_p(tensor), ffi.tensor_void_p(tensor),
            tensor.numel(), self._hccl_dtype(tensor),
            ffi.reduce_op_to_hccl(op), self._comm, s,
        ), "HcclAllReduce")
        return _maybe_async(s, async_op)

    def reduce(self, tensor, root, op, async_op):
        s = self._stream()
        ffi.check(ffi.HcclReduce(
            ffi.tensor_void_p(tensor), ffi.tensor_void_p(tensor),
            tensor.numel(), self._hccl_dtype(tensor),
            ffi.reduce_op_to_hccl(op), root, self._comm, s,
        ), "HcclReduce")
        return _maybe_async(s, async_op)

    # ---- broadcast & barrier ----
    def broadcast(self, tensor, root, async_op):
        s = self._stream()
        ffi.check(ffi.HcclBroadcast(
            ffi.tensor_void_p(tensor), tensor.numel(),
            self._hccl_dtype(tensor), root, self._comm, s,
        ), "HcclBroadcast")
        return _maybe_async(s, async_op)

    def barrier(self, async_op):
        # HCCL has no native barrier; emulate with allreduce on dummy.
        dev = self._device or torch.device(f"npu:{self._rank}")
        dummy = torch.zeros(1, dtype=torch.float32, device=dev)
        return self.all_reduce(dummy, torchcomms.ReduceOp.SUM, async_op)

    # ---- p2p ----
    def send(self, tensor, dst, async_op):
        s = self._stream()
        ffi.check(ffi.HcclSend(
            ffi.tensor_void_p(tensor), tensor.numel(),
            self._hccl_dtype(tensor), dst, self._comm, s,
        ), "HcclSend")
        return _maybe_async(s, async_op)

    def recv(self, tensor, src, async_op):
        s = self._stream()
        ffi.check(ffi.HcclRecv(
            ffi.tensor_void_p(tensor), tensor.numel(),
            self._hccl_dtype(tensor), src, self._comm, s,
        ), "HcclRecv")
        return _maybe_async(s, async_op)

    # ---- all_gather family ----
    def all_gather_single(self, output, input, async_op):
        s = self._stream()
        ffi.check(ffi.HcclAllGather(
            ffi.tensor_void_p(input), ffi.tensor_void_p(output),
            input.numel(), self._hccl_dtype(input), self._comm, s,
        ), "HcclAllGather")
        return _maybe_async(s, async_op)

    def all_gather(self, tensor_list, tensor, async_op):
        # HCCL writes to a contiguous N*K buffer; copy out into the user's
        # tensor_list after stream sync. List-form is rare — the contig
        # path (all_gather_single) is preferred for hot paths.
        s = self._stream()
        _reject_in_capture(s, "all_gather (list form)", "all_gather_single")
        N = self._world_size
        K = tensor.numel()
        assert len(tensor_list) == N, (
            f"all_gather: tensor_list len {len(tensor_list)} != world {N}"
        )
        contig = torch.empty(N * K, dtype=tensor.dtype, device=tensor.device)
        ffi.check(ffi.HcclAllGather(
            ffi.tensor_void_p(tensor), ffi.tensor_void_p(contig),
            K, self._hccl_dtype(tensor), self._comm, s,
        ), "HcclAllGather")
        ffi.acl_check(ffi.aclrtSynchronizeStream(s), "aclrtSynchronizeStream")
        for i in range(N):
            tensor_list[i].copy_(contig[i * K:(i + 1) * K].view_as(tensor_list[i]))
        if async_op:
            w = _StreamSyncWork(s)
            w._done = True
            return w
        return None

    def all_gather_v(self, tensor_list, tensor, async_op):
        # Uniform v-variant degenerates to plain all_gather. True uneven
        # support requires HcclAllGatherV with displs — out of scope for
        # this demo backend.
        return self.all_gather(tensor_list, tensor, async_op)

    # ---- reduce_scatter family ----
    def reduce_scatter_single(self, output, input, op, async_op):
        s = self._stream()
        ffi.check(ffi.HcclReduceScatter(
            ffi.tensor_void_p(input), ffi.tensor_void_p(output),
            output.numel(), self._hccl_dtype(input),
            ffi.reduce_op_to_hccl(op), self._comm, s,
        ), "HcclReduceScatter")
        return _maybe_async(s, async_op)

    def reduce_scatter(self, output, input_list, op, async_op):
        s = self._stream()
        _reject_in_capture(s, "reduce_scatter (list form)",
                           "reduce_scatter_single")
        N = self._world_size
        K = output.numel()
        assert len(input_list) == N, (
            f"reduce_scatter: input_list len {len(input_list)} != world {N}"
        )
        contig = torch.empty(N * K, dtype=output.dtype, device=output.device)
        for i in range(N):
            contig[i * K:(i + 1) * K].copy_(input_list[i].reshape(-1))
        ffi.check(ffi.HcclReduceScatter(
            ffi.tensor_void_p(contig), ffi.tensor_void_p(output),
            K, self._hccl_dtype(output),
            ffi.reduce_op_to_hccl(op), self._comm, s,
        ), "HcclReduceScatter")
        return _maybe_async(s, async_op)

    def reduce_scatter_v(self, output, input_list, op, async_op):
        return self.reduce_scatter(output, input_list, op, async_op)

    # ---- all_to_all family ----
    def all_to_all_single(self, output, input, async_op):
        N = self._world_size
        assert input.numel() % N == 0, (
            f"all_to_all_single requires uniform split; "
            f"input numel {input.numel()} not divisible by world {N}"
        )
        per_rank = input.numel() // N
        s = self._stream()
        ffi.check(ffi.HcclAlltoAll(
            ffi.tensor_void_p(input), per_rank, self._hccl_dtype(input),
            ffi.tensor_void_p(output), per_rank, self._hccl_dtype(output),
            self._comm, s,
        ), "HcclAlltoAll")
        return _maybe_async(s, async_op)

    def all_to_all_v_single(self, output, input, output_splits, input_splits,
                             async_op):
        N = self._world_size
        assert len(input_splits) == N and len(output_splits) == N
        SendCountsArr = ctypes.c_uint64 * N
        send_counts = SendCountsArr(*[int(x) for x in input_splits])
        recv_counts = SendCountsArr(*[int(x) for x in output_splits])
        sdispls = SendCountsArr(*[
            sum(int(x) for x in input_splits[:i]) for i in range(N)
        ])
        rdispls = SendCountsArr(*[
            sum(int(x) for x in output_splits[:i]) for i in range(N)
        ])
        s = self._stream()
        ffi.check(ffi.HcclAlltoAllV(
            ffi.tensor_void_p(input),
            ctypes.cast(send_counts, ctypes.c_void_p),
            ctypes.cast(sdispls, ctypes.c_void_p),
            self._hccl_dtype(input),
            ffi.tensor_void_p(output),
            ctypes.cast(recv_counts, ctypes.c_void_p),
            ctypes.cast(rdispls, ctypes.c_void_p),
            self._hccl_dtype(output),
            self._comm, s,
        ), "HcclAlltoAllV")
        return _maybe_async(s, async_op)

    def all_to_all(self, output_tensor_list, input_tensor_list, async_op):
        s = self._stream()
        _reject_in_capture(s, "all_to_all (list form)", "all_to_all_single")
        N = self._world_size
        assert len(output_tensor_list) == N and len(input_tensor_list) == N
        # Concatenate into contig, do HcclAlltoAll, split back out
        per_rank = input_tensor_list[0].numel()
        for t in input_tensor_list:
            assert t.numel() == per_rank, (
                "all_to_all (list) requires uniform per-rank counts; "
                "use all_to_all_v_single for uneven splits"
            )
        dt = input_tensor_list[0].dtype
        dev = input_tensor_list[0].device
        in_contig = torch.empty(N * per_rank, dtype=dt, device=dev)
        out_contig = torch.empty(N * per_rank, dtype=dt, device=dev)
        for i in range(N):
            in_contig[i * per_rank:(i + 1) * per_rank].copy_(
                input_tensor_list[i].reshape(-1))
        s = self._stream()
        ffi.check(ffi.HcclAlltoAll(
            ffi.tensor_void_p(in_contig), per_rank, ffi.torch_dtype_to_hccl(dt),
            ffi.tensor_void_p(out_contig), per_rank, ffi.torch_dtype_to_hccl(dt),
            self._comm, s,
        ), "HcclAlltoAll")
        ffi.acl_check(ffi.aclrtSynchronizeStream(s), "aclrtSynchronizeStream")
        for i in range(N):
            output_tensor_list[i].copy_(
                out_contig[i * per_rank:(i + 1) * per_rank].view_as(
                    output_tensor_list[i]))
        if async_op:
            w = _StreamSyncWork(s)
            w._done = True
            return w
        return None

    # ---- scatter / gather ----
    def scatter(self, output, input_list, root, async_op):
        # HCCL Scatter: root provides a contiguous send buffer of N*K, every
        # rank receives K. Non-root passes anything (HCCL ignores).
        K = output.numel()
        s = self._stream()
        if self._rank == root:
            assert len(input_list) == self._world_size
            send = torch.empty(self._world_size * K, dtype=output.dtype,
                                device=output.device)
            for i in range(self._world_size):
                send[i * K:(i + 1) * K].copy_(input_list[i].reshape(-1))
            send_ptr = ffi.tensor_void_p(send)
        else:
            send_ptr = ctypes.c_void_p(0)
        ffi.check(ffi.HcclScatter(
            send_ptr, ffi.tensor_void_p(output), K,
            self._hccl_dtype(output), root, self._comm, s,
        ), "HcclScatter")
        return _maybe_async(s, async_op)

    def gather(self, output_list, input, root, async_op):
        # HCCL has no native gather; emulate via allgather + select root.
        s = self._stream()
        _reject_in_capture(s, "gather", "all_gather_single + manual root slice")
        N = self._world_size
        K = input.numel()
        contig = torch.empty(N * K, dtype=input.dtype, device=input.device)
        ffi.check(ffi.HcclAllGather(
            ffi.tensor_void_p(input), ffi.tensor_void_p(contig),
            K, self._hccl_dtype(input), self._comm, s,
        ), "HcclAllGather")
        ffi.acl_check(ffi.aclrtSynchronizeStream(s), "aclrtSynchronizeStream")
        if self._rank == root and output_list:
            for i in range(N):
                output_list[i].copy_(
                    contig[i * K:(i + 1) * K].view_as(output_list[i]))
        if async_op:
            w = _StreamSyncWork(s)
            w._done = True
            return w
        return None

    # ---- split ----
    def split(self, ranks, name, options):
        # Mirror native TorchCommHCCL::split. HcclCreateSubCommConfig is
        # HCOMM_WEAK_SYMBOL on cann 9.0 but the ctypes path verified it
        # callable in ~10ms (see _probe_split.py). The native backend's
        # v0.12 "hang" was likely a config-fill issue that this Python
        # path avoids by reproducing HcclCommConfigInit byte-for-byte.
        if self._rank not in ranks:
            raise RuntimeError(
                f"split: current rank {self._rank} not in ranks {ranks}"
            )
        my_idx = ranks.index(self._rank)
        rank_ids = (ctypes.c_uint32 * len(ranks))(*[int(r) for r in ranks])
        # subCommId must be the same across all participating ranks.
        sub_comm_id = ctypes.c_uint64(hash(name) & 0xFFFFFFFFFFFFFFFF).value

        cfg = ffi.HcclCommConfig()
        ffi.hccl_comm_config_init(cfg)

        sub = ffi.HcclComm()
        ffi.check(
            ffi.HcclCreateSubCommConfig(
                ctypes.byref(self._comm),
                len(ranks),
                rank_ids,
                sub_comm_id,
                my_idx,
                ctypes.byref(cfg),
                ctypes.byref(sub),
            ),
            "HcclCreateSubCommConfig",
        )

        # Build a PyHcclBackend wrapper around the new sub-comm.
        child = PyHcclBackend.__new__(PyHcclBackend)
        TorchCommBackend.__init__(child)
        child._pg = self._pg
        child._rank = my_idx
        child._world_size = len(ranks)
        child._store = self._store
        child._device = self._device
        child._name = name
        child._comm = sub
        return child
