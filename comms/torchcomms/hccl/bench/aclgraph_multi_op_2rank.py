# aclgraph multi-op with the *native* C++ TorchCommHCCL backend.
# Mirror of pyhccl/examples/05_aclgraph_multi_op_2rank.py with backend="hccl".
#
# Run: torchrun --nproc_per_node=2 aclgraph_multi_op_2rank.py

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
        "hccl", dev, name="aclgraph_multi_native",
        store=store, hints={"rank": str(rank), "world_size": str(world)},
    )

    capture_stream = torch_npu.npu.Stream(device=dev)
    K = 64
    fp32 = torch.float32

    t_ar = torch.empty(K, device=dev, dtype=fp32)
    expected_ar = float(sum(range(1, world + 1)))

    t_bcast = torch.empty(K, device=dev, dtype=fp32)
    expected_bcast = 42.0

    t_ag_in = torch.empty(K, device=dev, dtype=fp32)
    t_ag_out = torch.empty(world * K, device=dev, dtype=fp32)
    expected_ag = [float(r + 100) for r in range(world)]

    t_rs_in = torch.empty(world * K, device=dev, dtype=fp32)
    t_rs_out = torch.empty(K, device=dev, dtype=fp32)
    expected_rs = float(sum(range(1, world + 1)))

    with torch_npu.npu.stream(capture_stream):
        t_ar.fill_(float(rank + 1))
        comm.all_reduce(t_ar, torchcomms.ReduceOp.SUM, async_op=False)
        t_bcast.fill_(42.0 if rank == 0 else 0.0)
        comm.broadcast(t_bcast, root=0, async_op=False)
        t_ag_in.fill_(float(rank + 100))
        comm.all_gather_single(t_ag_out, t_ag_in, async_op=False)
        t_rs_in.fill_(float(rank + 1))
        comm.reduce_scatter_single(t_rs_out, t_rs_in,
                                    torchcomms.ReduceOp.SUM, async_op=False)
        torch_npu.npu.synchronize()
    print(f"[rank {rank}] warmup 4 collectives OK", flush=True)

    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g, stream=capture_stream):
        comm.all_reduce(t_ar, torchcomms.ReduceOp.SUM, async_op=False)
        comm.broadcast(t_bcast, root=0, async_op=False)
        comm.all_gather_single(t_ag_out, t_ag_in, async_op=False)
        comm.reduce_scatter_single(t_rs_out, t_rs_in,
                                    torchcomms.ReduceOp.SUM, async_op=False)
    print(f"[rank {rank}] NPUGraph captured 4 collectives", flush=True)

    for i in range(3):
        with torch_npu.npu.stream(capture_stream):
            t_ar.fill_(float(rank + 1))
            t_bcast.fill_(42.0 if rank == 0 else 0.0)
            t_ag_in.fill_(float(rank + 100))
            t_rs_in.fill_(float(rank + 1))
        g.replay()
        torch_npu.npu.synchronize()

        ar_val = t_ar.cpu()[0].item()
        _check(abs(ar_val - expected_ar) < 1e-5,
               f"replay#{i} all_reduce: got {ar_val}, expected {expected_ar}")
        bcast_val = t_bcast.cpu()[0].item()
        _check(abs(bcast_val - expected_bcast) < 1e-5,
               f"replay#{i} broadcast: got {bcast_val}, expected {expected_bcast}")
        ag_cpu = t_ag_out.cpu()
        for r in range(world):
            chunk_val = ag_cpu[r * K].item()
            _check(abs(chunk_val - expected_ag[r]) < 1e-5,
                   f"replay#{i} all_gather rank{r}: got {chunk_val}, "
                   f"expected {expected_ag[r]}")
        rs_val = t_rs_out.cpu()[0].item()
        _check(abs(rs_val - expected_rs) < 1e-5,
               f"replay#{i} reduce_scatter: got {rs_val}, expected {expected_rs}")
        print(f"[rank {rank}] replay#{i}: ar={ar_val} bcast={bcast_val} "
              f"ag[0]={ag_cpu[0].item()} rs={rs_val} OK", flush=True)

    comm.finalize()
    dist.destroy_process_group()
    print(f"[rank {rank}] ALL PASS", flush=True)


if __name__ == "__main__":
    main()
