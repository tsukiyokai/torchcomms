# aclgraph end-to-end: torch_npu.npu.NPUGraph capture-replay around
# torchcomms PyHcclBackend.all_reduce on NPU.
#
# Run: torchrun --nproc_per_node=2 04_aclgraph_2rank.py
#
# Mechanism: NPUGraph wraps cann 9.0 aclmdlRICaptureBegin/End. Inside the
# capture context, every device-side op (kernel launch, HCCL collective,
# memcpy) is recorded into an aclmdlRI graph node. Replay reissues all
# nodes onto the bound stream without re-doing host work.
#
# What this validates:
#   1. torch_npu NPUGraph reaches PyHcclBackend's collective path
#   2. PyHcclBackend's _maybe_async / _StreamSyncWork detect the active
#      capture and skip aclrtSynchronizeStream (which would break it)
#   3. HCCL itself records HcclAllReduce as a graph node (HCCL self-support
#      verified separately by _probe_capture.py)

import os
import sys

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import torchcomms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hccl_pyend import PyHcclBackend, set_default_pg


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


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
    comm = torchcomms.new_comm("py_hccl", dev, name="aclgraph_demo")

    # Dedicated capture stream. NPUGraph requires the capture happens on
    # a non-default stream (same constraint as cuda graphs).
    capture_stream = torch_npu.npu.Stream(device=dev)

    expected = float(sum(range(1, world + 1)))  # rank-r contributes r+1; sum = world*(world+1)/2 = 3 for w=2

    # ---- warmup outside capture (HCCL lazy init + caching alloc) ----
    with torch_npu.npu.stream(capture_stream):
        t_warm = torch.full((128,), float(rank + 1), device=dev, dtype=torch.float32)
        comm.all_reduce(t_warm, torchcomms.ReduceOp.SUM, async_op=False).wait() if False else \
            comm.all_reduce(t_warm, torchcomms.ReduceOp.SUM, async_op=False)
        torch_npu.npu.synchronize()
    _check(abs(t_warm.cpu()[0].item() - expected) < 1e-5,
           f"warmup: got {t_warm.cpu()[0].item()}, expected {expected}")
    print(f"[rank {rank}] warmup all_reduce OK -> {t_warm.cpu()[0].item()}",
          flush=True)

    # ---- capture ----
    g = torch_npu.npu.NPUGraph()
    t = torch.full((128,), float(rank + 1), device=dev, dtype=torch.float32)
    with torch_npu.npu.graph(g, stream=capture_stream):
        # PyHcclBackend.all_reduce is sync (async_op=False) but
        # _maybe_async detects the active capture and degrades to async,
        # so the HcclAllReduce kernel is recorded as a graph node instead
        # of being followed by aclrtSynchronizeStream (which would invalidate
        # the capture).
        comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False)
    print(f"[rank {rank}] NPUGraph captured", flush=True)

    # ---- replay 5 times, verify in-place sum each iteration ----
    for i in range(5):
        with torch_npu.npu.stream(capture_stream):
            t.fill_(float(rank + 1))
        g.replay()
        torch_npu.npu.synchronize()
        actual = t.cpu()[0].item()
        _check(abs(actual - expected) < 1e-5,
               f"replay#{i}: got {actual}, expected {expected}")
        print(f"[rank {rank}] replay#{i} all_reduce OK -> {actual}",
              flush=True)

    comm.finalize()
    dist.destroy_process_group()
    print(f"[rank {rank}] ALL PASS", flush=True)


if __name__ == "__main__":
    main()
