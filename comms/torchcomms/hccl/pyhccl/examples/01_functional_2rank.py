# Multi-rank smoke test for PyHcclBackend.
# Run: torchrun --nproc_per_node=2 01_functional_2rank.py

import os
import sys
import torch
import torch_npu  # noqa: F401  (registers PrivateUse1 + populates torch_npu._C)
import torch.distributed as dist
import torchcomms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_hccl_backend import PyHcclBackend, set_default_pg


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch_npu.npu.set_device(local_rank)
    dev = torch.device(f"npu:{local_rank}")

    # ---- bootstrap rendezvous (gloo for store only; PyHcclBackend
    # creates its own HCCL communicator via ctypes) ----
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    pg = dist.distributed_c10d._get_default_group()
    set_default_pg(pg)

    # ---- register torchcomms backend ----
    torchcomms.register_backend("py_hccl", PyHcclBackend)
    # Name is the comm-group identifier — must be identical across ranks
    # (it's used as the c10d-store rendezvous key in PyHcclBackend).
    comm = torchcomms.new_comm("py_hccl", dev, name="demo")
    _check(comm.get_rank() == rank, f"rank mismatch: {comm.get_rank()} != {rank}")
    _check(comm.get_size() == world_size, f"size mismatch")
    _check(comm.get_backend() == "py_hccl", "backend name")

    # ---- all_reduce SUM ----
    # rank r contributes (r+1) on every element; sum = 1+2+..+N
    t = torch.full((8,), float(rank + 1), device=dev)
    work = comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False)
    work.wait()
    expected = float(sum(range(1, world_size + 1)))
    _check(
        torch.allclose(t.cpu(), torch.full((8,), expected)),
        f"all_reduce: got {t.cpu().tolist()}, expected {[expected] * 8}",
    )
    print(f"[rank {rank}] all_reduce OK -> {t.cpu()[0].item()}")

    # ---- broadcast (root=0 sends 42, others get whatever -> 42) ----
    src = torch.full((4,), float(rank * 10), device=dev)
    if rank == 0:
        src.fill_(42.0)
    work = comm.broadcast(src, root=0, async_op=False)
    work.wait()
    _check(
        torch.allclose(src.cpu(), torch.full((4,), 42.0)),
        f"broadcast: got {src.cpu().tolist()}",
    )
    print(f"[rank {rank}] broadcast OK")

    # ---- barrier ----
    work = comm.barrier(async_op=False)
    work.wait()
    print(f"[rank {rank}] barrier OK")

    # ---- all_gather ----
    src = torch.full((4,), float(rank), device=dev)
    out = [torch.empty(4, device=dev) for _ in range(world_size)]
    work = comm.all_gather(out, src, async_op=False)
    work.wait()
    for i, t_i in enumerate(out):
        _check(
            torch.allclose(t_i.cpu(), torch.full((4,), float(i))),
            f"all_gather[{i}]: got {t_i.cpu().tolist()}",
        )
    print(f"[rank {rank}] all_gather OK")

    # ---- async path: all_reduce returns a Work that wait()s ----
    t2 = torch.full((4,), float(rank + 1), device=dev)
    handle = comm.all_reduce(t2, torchcomms.ReduceOp.SUM, async_op=True)
    handle.wait()
    _check(
        torch.allclose(t2.cpu(), torch.full((4,), float(sum(range(1, world_size + 1))))),
        f"async all_reduce result wrong",
    )
    print(f"[rank {rank}] async all_reduce OK")

    # ---- reduce (root collects sum) ----
    t = torch.full((4,), float(rank + 1), device=dev)
    work = comm.reduce(t, root=0, op=torchcomms.ReduceOp.SUM, async_op=False)
    work.wait()
    if rank == 0:
        _check(
            torch.allclose(t.cpu(), torch.full((4,), float(sum(range(1, world_size + 1))))),
            f"reduce root: got {t.cpu()}",
        )
    print(f"[rank {rank}] reduce OK")

    # ---- reduce_scatter_single ----
    # input: each rank holds [N*K] with all (rank+1); output: per-rank gets [K]
    # filled with sum(1..N) since every chunk on every rank is rank+1
    K = 4
    inp = torch.full((world_size * K,), float(rank + 1), device=dev)
    out = torch.empty(K, device=dev)
    work = comm.reduce_scatter_single(out, inp, torchcomms.ReduceOp.SUM, async_op=False)
    work.wait()
    expected = float(sum(range(1, world_size + 1)))
    _check(
        torch.allclose(out.cpu(), torch.full((K,), expected)),
        f"rs_single: got {out.cpu()}, expected {expected}",
    )
    print(f"[rank {rank}] reduce_scatter_single OK")

    # ---- scatter (root=0 sends [r=>r*100] tensors to each rank) ----
    out = torch.empty(4, device=dev)
    if rank == 0:
        ins = [torch.full((4,), float(r * 100), device=dev) for r in range(world_size)]
    else:
        ins = []  # non-root passes empty list
    work = comm.scatter(out, ins, root=0, async_op=False)
    work.wait()
    _check(
        torch.allclose(out.cpu(), torch.full((4,), float(rank * 100))),
        f"scatter: rank {rank} got {out.cpu()}",
    )
    print(f"[rank {rank}] scatter OK")

    # ---- gather (root collects rank-tagged tensors) ----
    src = torch.full((4,), float(rank * 7), device=dev)
    if rank == 0:
        outs = [torch.empty(4, device=dev) for _ in range(world_size)]
    else:
        outs = []
    work = comm.gather(outs, src, root=0, async_op=False)
    work.wait()
    if rank == 0:
        for i, o in enumerate(outs):
            _check(
                torch.allclose(o.cpu(), torch.full((4,), float(i * 7))),
                f"gather[{i}]: {o.cpu()}",
            )
    print(f"[rank {rank}] gather OK")

    # ---- all_to_all_single (uniform; rank r sends rank-tagged chunk to each peer) ----
    # in[i*K:(i+1)*K] is what rank r will send to rank i
    K = 4
    inp = torch.cat([torch.full((K,), float(rank * 100 + i), device=dev) for i in range(world_size)])
    out = torch.empty_like(inp)
    work = comm.all_to_all_single(out, inp, async_op=False)
    work.wait()
    # rank r receives in chunk i: data from rank i with i->r, so value = i*100 + r
    expected = torch.cat([torch.full((K,), float(i * 100 + rank)) for i in range(world_size)])
    _check(torch.allclose(out.cpu(), expected), f"a2a_single: got {out.cpu()}")
    print(f"[rank {rank}] all_to_all_single OK")

    # ---- send / recv (rank0 -> rank1) ----
    if rank == 0:
        t = torch.full((4,), 99.0, device=dev)
        comm.send(t, dst=1, async_op=False).wait()
    else:
        t = torch.zeros(4, device=dev)
        comm.recv(t, src=0, async_op=False).wait()
        _check(torch.allclose(t.cpu(), torch.full((4,), 99.0)), f"recv: {t.cpu()}")
    print(f"[rank {rank}] send_recv OK")

    # ---- split (re-form a sub-comm with all current ranks; same semantics) ----
    # TorchComm.split user signature: (ranks, name, hints=None, timeout=None).
    # PyHcclBackend (ctypes variant) skips split because cann 9.0 ships
    # HcclCreateSubCommConfig as a weak symbol — same status as the native
    # backend (hccl/README.md split 🟡). Treat NotImplemented as expected.
    try:
        sub = comm.split(list(range(world_size)), "sub")
        _check(sub.get_size() == world_size, f"sub size {sub.get_size()}")
        print(f"[rank {rank}] split OK")
    except NotImplementedError as e:
        print(f"[rank {rank}] split SKIPPED ({e.__class__.__name__})")

    comm.finalize()
    dist.destroy_process_group()
    print(f"[rank {rank}] ALL PASS")


if __name__ == "__main__":
    main()
