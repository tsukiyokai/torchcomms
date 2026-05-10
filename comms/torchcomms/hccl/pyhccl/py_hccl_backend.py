# PyHcclBackend: sanity-check for the torchcomms.register_backend Python path.
# Forwards torchcomms collectives to torch_npu's ProcessGroupHCCL.
# Not the production HCCL backend — this is the perforation/穿刺 demo.
#
# Usage (rank R of N):
#     import torch, torch.distributed as dist, torch_npu, torchcomms
#     from py_hccl_backend import PyHcclBackend, set_default_pg
#
#     dist.init_process_group(backend="hccl", rank=R, world_size=N)
#     set_default_pg(dist.distributed_c10d._get_default_group())
#     torchcomms.register_backend("py_hccl", PyHcclBackend)
#     comm = torchcomms.new_comm("py_hccl", torch.device(f"npu:{R}"), name="demo")
#     t = torch.ones(8, device=f"npu:{R}")
#     comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False)

import torch
import torch.distributed as dist
import torchcomms
from torchcomms._comms import TorchCommBackend

# ====
# Module state
# ====
_DEFAULT_PG = None


def set_default_pg(pg):
    global _DEFAULT_PG
    _DEFAULT_PG = pg


# ====
# Helpers
# ====
def _to_c10d_op(op):
    name = getattr(op, "name", None) or str(op).split(".")[-1].upper()
    return {
        "SUM": dist.ReduceOp.SUM,
        "PRODUCT": dist.ReduceOp.PRODUCT,
        "MIN": dist.ReduceOp.MIN,
        "MAX": dist.ReduceOp.MAX,
    }.get(name, dist.ReduceOp.SUM)


class _PgWork:
    """Wrap c10d.Work to expose wait()/is_completed() expected by trampoline."""

    def __init__(self, work):
        self._w = work

    def wait(self):
        if self._w is not None:
            self._w.wait()

    def is_completed(self):
        return self._w is None or self._w.is_completed()


def _sync_or_async(work, async_op):
    if async_op:
        return _PgWork(work)
    if work is not None:
        work.wait()
    return None


# ====
# Backend
# ====
class PyHcclBackend(TorchCommBackend):
    """torchcomms backend forwarding to torch_npu's ProcessGroupHCCL.

    register_backend wants a no-arg class, so we read _DEFAULT_PG from module
    state. Caller must set_default_pg() between init_process_group() and
    register_backend().
    """

    def __init__(self):
        super().__init__()
        if _DEFAULT_PG is None:
            raise RuntimeError(
                "PyHcclBackend needs a default ProcessGroup. "
                "Call set_default_pg() after init_process_group()."
            )
        self._pg = _DEFAULT_PG
        self._device = None
        self._name = ""

    # ---- lifecycle ----
    def init(self, device, name, options):
        self._device = device
        self._name = name

    def finalize(self):
        pass

    def get_rank(self):
        return self._pg.rank()

    def get_size(self):
        return self._pg.size()

    def get_backend_name(self):
        return "py_hccl"

    def get_comm_name(self):
        return self._name or "py_hccl_default"

    # ---- reductions ----
    def all_reduce(self, tensor, op, async_op):
        opts = dist.AllreduceOptions()
        opts.reduceOp = _to_c10d_op(op)
        return _sync_or_async(self._pg.allreduce([tensor], opts), async_op)

    def reduce(self, tensor, root, op, async_op):
        opts = dist.ReduceOptions()
        opts.reduceOp = _to_c10d_op(op)
        opts.rootRank = root
        opts.rootTensor = 0
        return _sync_or_async(self._pg.reduce([tensor], opts), async_op)

    # ---- broadcast & barrier ----
    def broadcast(self, tensor, root, async_op):
        opts = dist.BroadcastOptions()
        opts.rootRank = root
        opts.rootTensor = 0
        return _sync_or_async(self._pg.broadcast([tensor], opts), async_op)

    def barrier(self, async_op):
        # Use allreduce-on-dummy as barrier on NPU. ProcessGroupHCCL.barrier()
        # plain call hits "No backend type associated with device type cpu"
        # because BarrierOptions.device_ids alone doesn't reroute device
        # discovery through the trampoline path.
        dev = self._device if (self._device is not None and self._device.type != "cpu") else None
        if dev is not None:
            dummy = torch.zeros(1, device=dev)
            opts = dist.AllreduceOptions()
            opts.reduceOp = dist.ReduceOp.SUM
            return _sync_or_async(self._pg.allreduce([dummy], opts), async_op)
        opts = dist.BarrierOptions()
        return _sync_or_async(self._pg.barrier(opts), async_op)

    # ---- p2p ----
    def send(self, tensor, dst, async_op):
        return _sync_or_async(self._pg.send([tensor], dst, 0), async_op)

    def recv(self, tensor, src, async_op):
        return _sync_or_async(self._pg.recv([tensor], src, 0), async_op)

    # ---- all_gather family ----
    def all_gather(self, tensor_list, tensor, async_op):
        return _sync_or_async(self._pg.allgather([tensor_list], [tensor]), async_op)

    def all_gather_v(self, tensor_list, tensor, async_op):
        return self.all_gather(tensor_list, tensor, async_op)

    def all_gather_single(self, output, input, async_op):
        N = self._pg.size()
        K = input.numel()
        chunks = [output.narrow(0, i * K, K) for i in range(N)]
        return _sync_or_async(self._pg.allgather([chunks], [input]), async_op)

    # ---- reduce_scatter family ----
    def reduce_scatter(self, output, input_list, op, async_op):
        opts = dist.ReduceScatterOptions()
        opts.reduceOp = _to_c10d_op(op)
        return _sync_or_async(self._pg.reduce_scatter([output], [input_list], opts), async_op)

    def reduce_scatter_v(self, output, input_list, op, async_op):
        return self.reduce_scatter(output, input_list, op, async_op)

    def reduce_scatter_single(self, output, input, op, async_op):
        N = self._pg.size()
        K = output.numel()
        chunks = [input.narrow(0, i * K, K) for i in range(N)]
        opts = dist.ReduceScatterOptions()
        opts.reduceOp = _to_c10d_op(op)
        return _sync_or_async(self._pg.reduce_scatter([output], [chunks], opts), async_op)

    # ---- all_to_all family ----
    def all_to_all_single(self, output, input, async_op):
        return _sync_or_async(self._pg.alltoall_base(output, input, [], []), async_op)

    def all_to_all_v_single(self, output, input, output_splits, input_splits, async_op):
        return _sync_or_async(
            self._pg.alltoall_base(output, input, list(output_splits), list(input_splits)),
            async_op,
        )

    def all_to_all(self, output_tensor_list, input_tensor_list, async_op):
        return _sync_or_async(self._pg.alltoall(output_tensor_list, input_tensor_list), async_op)

    # ---- scatter / gather ----
    def scatter(self, output, input_list, root, async_op):
        opts = dist.ScatterOptions()
        opts.rootRank = root
        ins = [input_list] if self._pg.rank() == root else []
        return _sync_or_async(self._pg.scatter([output], ins, opts), async_op)

    def gather(self, output_list, input, root, async_op):
        # ProcessGroupHCCL does not support gather (DIST feature not supported,
        # ERR02007). Emulate via allgather then keep the root copy. The
        # native C++ backend uses a direct HcclXxx call instead.
        N = self._pg.size()
        full = [torch.empty_like(input) for _ in range(N)]
        work = self._pg.allgather([full], [input])
        work.wait()
        if self._pg.rank() == root and output_list:
            for i, t in enumerate(full):
                output_list[i].copy_(t)
        return None if not async_op else _PgWork(None)

    # ---- split ----
    def split(self, ranks, name, options):
        new_pg = dist.new_group(ranks, backend="hccl")
        global _DEFAULT_PG
        prev = _DEFAULT_PG
        _DEFAULT_PG = new_pg
        try:
            sub = PyHcclBackend()
            sub.init(self._device, name, options)
            return sub
        finally:
            _DEFAULT_PG = prev
