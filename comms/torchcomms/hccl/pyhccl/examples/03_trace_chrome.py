# trace_dispatch: capture torch.profiler chrome trace for both paths.
# Output goes to /tmp/torchcomms_trace/ on the remote and is rsync'd to evidence/.
# Run: torchrun --nproc_per_node=2 trace_dispatch.py

import ctypes
import os
import sys
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import torchcomms
from torch.profiler import profile, ProfilerActivity, record_function

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hccl_pyend import PyHcclBackend, set_default_pg
import hccl_ffi as ffi


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch_npu.npu.set_device(local)
    dev = torch.device(f"npu:{local}")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    pg = dist.distributed_c10d._get_default_group()
    set_default_pg(pg)

    torchcomms.register_backend("py_hccl", PyHcclBackend)
    comm = torchcomms.new_comm("py_hccl", dev, name="trace")
    backend = comm.get_backend_impl()

    t = torch.ones(1024, device=dev)
    hccl_dtype = ffi.torch_dtype_to_hccl(t.dtype)

    def raw_ctypes_allreduce():
        s = ffi.current_npu_stream()
        ffi.check(ffi.HcclAllReduce(
            ctypes.c_void_p(t.data_ptr()), ctypes.c_void_p(t.data_ptr()),
            t.numel(), hccl_dtype, ffi.HCCL_REDUCE_SUM, backend._comm, s,
        ), "HcclAllReduce")
        ffi.acl_check(ffi.aclrtSynchronizeStream(s), "aclrtSynchronizeStream")

    # warmup
    for _ in range(20):
        comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False)
        raw_ctypes_allreduce()

    out_dir = "/tmp/torchcomms_trace"
    os.makedirs(out_dir, exist_ok=True)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
        for i in range(10):
            with record_function("torchcomms_path"):
                comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False)
        for i in range(10):
            with record_function("raw_ctypes_path"):
                raw_ctypes_allreduce()

    if rank == 0:
        prof.export_chrome_trace(f"{out_dir}/dispatch_trace.json")
        print(f"[rank 0] chrome trace -> {out_dir}/dispatch_trace.json")
        print()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))

    comm.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
