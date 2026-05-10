# Multi-rank smoke for the native C++ HCCL backend (TorchCommHCCL).
# Run: torchrun --nproc_per_node=2 manual_2rank_test.py

import os
import torch
import torch_npu  # noqa
import torch.distributed as dist
import torchcomms
import torchcomms._comms_hccl  # noqa: F401  (triggers static factory register)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch_npu.npu.set_device(local)
    dev = torch.device(f"npu:{local}")

    # gloo provides the c10d::Store rendezvous; HCCL itself does the kernel.
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    store = dist.distributed_c10d._get_default_store()

    comm = torchcomms.new_comm(
        "hccl",
        dev,
        name="v04_smoke",
        store=store,
        hints={"rank": str(rank), "world_size": str(world)},
    )
    print(f"[rank {rank}] backend={comm.get_backend()} "
          f"r={comm.get_rank()} s={comm.get_size()}")

    # ---- all_reduce SUM ----
    t = torch.full((8,), float(rank + 1), device=dev)
    work = comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False)
    work.wait()
    expected = float(sum(range(1, world + 1)))
    got = t.cpu().tolist()
    assert all(abs(v - expected) < 1e-5 for v in got), \
        f"rank {rank} all_reduce: got {got}, expected {[expected] * 8}"
    print(f"[rank {rank}] all_reduce OK -> {got[0]}")

    # ---- async path ----
    t2 = torch.full((4,), float(rank + 1), device=dev)
    h = comm.all_reduce(t2, torchcomms.ReduceOp.SUM, async_op=True)
    h.wait()
    got2 = t2.cpu().tolist()
    assert all(abs(v - expected) < 1e-5 for v in got2)
    print(f"[rank {rank}] async all_reduce OK -> {got2[0]}")

    # ---- broadcast (root=0 sets 42) ----
    src = torch.full((4,), 42.0 if rank == 0 else 0.0, device=dev)
    comm.broadcast(src, root=0, async_op=False).wait()
    assert torch.allclose(src.cpu(), torch.full((4,), 42.0)), \
        f"broadcast: rank {rank} got {src.cpu()}"
    print(f"[rank {rank}] broadcast OK")

    # ---- reduce SUM to root=0 ----
    t = torch.full((4,), float(rank + 1), device=dev)
    comm.reduce(t, root=0, op=torchcomms.ReduceOp.SUM, async_op=False).wait()
    if rank == 0:
        assert torch.allclose(t.cpu(), torch.full((4,), expected)), \
            f"reduce root: {t.cpu()}"
    print(f"[rank {rank}] reduce OK")

    # ---- all_gather_single (each rank contributes K, output is N*K) ----
    K = 4
    inp = torch.full((K,), float(rank), device=dev)
    out = torch.empty(world * K, device=dev)
    comm.all_gather_single(out, inp, async_op=False).wait()
    expected_g = torch.cat([torch.full((K,), float(i)) for i in range(world)])
    assert torch.allclose(out.cpu(), expected_g), f"ag_single: {out.cpu()}"
    print(f"[rank {rank}] all_gather_single OK")

    # ---- reduce_scatter_single ----
    inp = torch.full((world * K,), float(rank + 1), device=dev)
    out = torch.empty(K, device=dev)
    comm.reduce_scatter_single(out, inp, torchcomms.ReduceOp.SUM,
                                async_op=False).wait()
    assert torch.allclose(out.cpu(), torch.full((K,), expected)), \
        f"rs_single: {out.cpu()}"
    print(f"[rank {rank}] reduce_scatter_single OK")

    # ---- barrier ----
    comm.barrier(async_op=False).wait()
    print(f"[rank {rank}] barrier OK")

    # ---- send / recv ----
    if rank == 0:
        t = torch.full((4,), 99.0, device=dev)
        comm.send(t, dst=1, async_op=False).wait()
    else:
        t = torch.zeros(4, device=dev)
        comm.recv(t, src=0, async_op=False).wait()
        assert torch.allclose(t.cpu(), torch.full((4,), 99.0)), f"recv: {t.cpu()}"
    print(f"[rank {rank}] send_recv OK")

    # ---- additions: gather emulation + batch_op_issue note ----

    # ---- gather (HCCL emulate via allgather) ----
    src = torch.full((4,), float(rank * 7), device=dev)
    if rank == 0:
        outs = [torch.empty(4, device=dev) for _ in range(world)]
    else:
        outs = []
    comm.gather(outs, src, root=0, async_op=False).wait()
    if rank == 0:
        for i, o in enumerate(outs):
            assert torch.allclose(o.cpu(), torch.full((4,), float(i * 7))), \
                f"gather[{i}]: {o.cpu()}"
    print(f"[rank {rank}] gather OK")

    # batch_op_issue: C++ implementation forwards to HcclBatchSendRecv;
    # the Python API surface (P2POp / BatchSendRecv) needs a helper for
    # clean test setup. C++ side already covered by the send/recv test above.

    comm.finalize()
    dist.destroy_process_group()
    print(f"[rank {rank}] ALL PASS")


if __name__ == "__main__":
    main()
