# bench_dispatch: PyHcclBackend dispatch overhead measurement.
#
# Compares per-call host latency between two paths that ultimately hit
# the same ProcessGroupHCCL underneath:
#   path A: torchcomms PyHcclBackend (trampoline + PG.allreduce)
#   path B: dist.all_reduce direct call (PG.allreduce only)
# Difference ≈ Python trampoline overhead.
#
# Run: torchrun --nproc_per_node=2 bench_dispatch.py

import os
import sys
import time
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import torchcomms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_hccl_backend import PyHcclBackend, set_default_pg

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

    dist.init_process_group(backend="hccl", rank=rank, world_size=world)
    pg = dist.distributed_c10d._get_default_group()
    set_default_pg(pg)

    torchcomms.register_backend("py_hccl", PyHcclBackend)
    comm = torchcomms.new_comm("py_hccl", dev, name="bench")

    shapes = [(4,), (1024,), (1024 * 1024,)]

    if rank == 0:
        header = f"{'shape':>15} {'path':>14} {'p50_us':>10} {'p90_us':>10} {'p99_us':>10}"
        bar = "=" * len(header)
        print(f"\n{bar}\nDispatch overhead bench (sync collective, host-side timing)\n{bar}")
        print(header)

    for shape in shapes:
        t = torch.ones(shape, device=dev)

        # Path A: torchcomms PyHcclBackend  (trampoline + PG.allreduce + sync)
        def call_a():
            comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False)

        # Path B: direct c10d  (PG.allreduce + sync)
        def call_b():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

        ns_a = measure(call_a)
        ns_b = measure(call_b)

        if rank == 0:
            pa = percentiles_ns(ns_a)
            pb = percentiles_ns(ns_b)
            print(f"{str(shape):>15} {'torchcomms':>14} "
                  f"{pa[0] / 1000:>10.2f} {pa[1] / 1000:>10.2f} {pa[2] / 1000:>10.2f}")
            print(f"{str(shape):>15} {'c10d_direct':>14} "
                  f"{pb[0] / 1000:>10.2f} {pb[1] / 1000:>10.2f} {pb[2] / 1000:>10.2f}")
            delta_p50 = (pa[0] - pb[0]) / 1000.0
            delta_p99 = (pa[2] - pb[2]) / 1000.0
            print(f"{str(shape):>15} {'overhead':>14} "
                  f"{delta_p50:>10.2f} {'':>10} {delta_p99:>10.2f}")

    comm.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
