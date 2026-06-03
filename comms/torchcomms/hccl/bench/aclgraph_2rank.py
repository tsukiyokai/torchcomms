# aclgraph end-to-end with the *native* C++ TorchCommHCCL backend.
# Mirror of pyhccl/examples/04_aclgraph_2rank.py but uses backend="hccl".
#
# Run: torchrun --nproc_per_node=2 aclgraph_2rank.py
#
# Native backend is already capture-friendly because TorchCommHCCL::makeWork
# (TorchCommHCCL.cpp:40-46) never records an event nor calls
# aclrtSynchronizeStream — it returns a no-op TorchWork and relies on NPU
# stream-order to serialize downstream consumers. That is exactly what an
# active aclmdlRICapture needs.

import os

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import torchcomms
import torchcomms._comms_hccl  # noqa: F401  (triggers static factory register)


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
    store = dist.distributed_c10d._get_default_store()

    comm = torchcomms.new_comm(
        "hccl",
        dev,
        name="aclgraph_native",
        store=store,
        hints={"rank": str(rank), "world_size": str(world)},
    )

    capture_stream = torch_npu.npu.Stream(device=dev)
    expected = float(sum(range(1, world + 1)))  # rank-r contributes r+1; w=2 → 3.0

    # ---- warmup outside capture (HCCL lazy init + caching alloc) ----
    with torch_npu.npu.stream(capture_stream):
        t_warm = torch.full((128,), float(rank + 1), device=dev, dtype=torch.float32)
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
