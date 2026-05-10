# bench_native_hccl: dispatch overhead of native C++ HCCL backend.
# Pairs with pyhccl/examples/02_bench_dispatch.py (which measured the
# PyHcclBackend forward path). Compare p50 numbers across the two outputs
# to confirm the C++ backend is fast enough vs the Python forward path.
#
# Run: torchrun --nproc_per_node=2 bench_native_hccl.py

import os
import sys
import time
import torch
import torch_npu  # noqa
import torch.distributed as dist
import torchcomms
import torchcomms._comms_hccl  # noqa: F401

WARMUP = 30
ITER = 300


def percentiles_ns(samples, ps=(50, 90, 99)):
    s = sorted(samples)
    n = len(s)
    return tuple(s[min(int(n * p / 100), n - 1)] for p in ps)


def measure(call):
    for _ in range(WARMUP):
        call()
    samples = []
    for _ in range(ITER):
        t0 = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - t0)
    return samples


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch_npu.npu.set_device(local)
    dev = torch.device(f"npu:{local}")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    store = dist.distributed_c10d._get_default_store()

    comm = torchcomms.new_comm(
        "hccl", dev, name="bench_native", store=store,
        hints={"rank": str(rank), "world_size": str(world)},
    )

    shapes = [(4,), (1024,), (1024 * 1024,)]
    if rank == 0:
        header = f"{'shape':>15} {'p50_us':>10} {'p90_us':>10} {'p99_us':>10}"
        print(f"\n{'=' * len(header)}\nnative HCCL backend bench (sync)\n{'=' * len(header)}")
        print(header)

    for shape in shapes:
        t = torch.ones(shape, device=dev)
        ns = measure(lambda: comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False).wait())
        if rank == 0:
            p50, p90, p99 = percentiles_ns(ns)
            print(f"{str(shape):>15} {p50/1000:>10.2f} {p90/1000:>10.2f} {p99/1000:>10.2f}")

    comm.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
