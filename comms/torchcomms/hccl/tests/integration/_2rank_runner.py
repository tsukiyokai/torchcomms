# 2-rank runner script. Launched by tests/integration/test_collectives.py
# via torchrun subprocess. Each function name maps to one collective check;
# selected by argv[1].

import os
import sys
import torch
import torch_npu  # noqa
import torch.distributed as dist
import torchcomms
import torchcomms._comms_hccl  # noqa


def setup():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch_npu.npu.set_device(local)
    dev = torch.device(f"npu:{local}")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    store = dist.distributed_c10d._get_default_store()
    name = sys.argv[1] if len(sys.argv) > 1 else "default"
    comm = torchcomms.new_comm(
        "hccl", dev, name=f"int_{name}", store=store,
        hints={"rank": str(rank), "world_size": str(world)},
    )
    return rank, world, dev, comm


def teardown(comm):
    comm.finalize()
    dist.destroy_process_group()


def cmd_all_reduce():
    rank, world, dev, comm = setup()
    t = torch.full((8,), float(rank + 1), device=dev)
    comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False).wait()
    expected = float(sum(range(1, world + 1)))
    assert torch.allclose(t.cpu(), torch.full((8,), expected)), \
        f"rank {rank}: got {t.cpu()}"
    teardown(comm)


def cmd_broadcast():
    rank, world, dev, comm = setup()
    t = torch.full((4,), 42.0 if rank == 0 else 0.0, device=dev)
    comm.broadcast(t, root=0, async_op=False).wait()
    assert torch.allclose(t.cpu(), torch.full((4,), 42.0)), \
        f"rank {rank}: got {t.cpu()}"
    teardown(comm)


def cmd_all_gather_single():
    rank, world, dev, comm = setup()
    K = 4
    inp = torch.full((K,), float(rank), device=dev)
    out = torch.empty(world * K, device=dev)
    comm.all_gather_single(out, inp, async_op=False).wait()
    expected = torch.cat([torch.full((K,), float(i)) for i in range(world)])
    assert torch.allclose(out.cpu(), expected), f"rank {rank}: got {out.cpu()}"
    teardown(comm)


def cmd_reduce_scatter_single():
    rank, world, dev, comm = setup()
    K = 4
    inp = torch.full((world * K,), float(rank + 1), device=dev)
    out = torch.empty(K, device=dev)
    comm.reduce_scatter_single(
        out, inp, torchcomms.ReduceOp.SUM, async_op=False
    ).wait()
    expected = float(sum(range(1, world + 1)))
    assert torch.allclose(out.cpu(), torch.full((K,), expected)), \
        f"rank {rank}: got {out.cpu()}"
    teardown(comm)


def cmd_send_recv():
    rank, world, dev, comm = setup()
    if rank == 0:
        t = torch.full((4,), 99.0, device=dev)
        comm.send(t, dst=1, async_op=False).wait()
    else:
        t = torch.zeros(4, device=dev)
        comm.recv(t, src=0, async_op=False).wait()
        assert torch.allclose(t.cpu(), torch.full((4,), 99.0))
    teardown(comm)


def cmd_barrier():
    _, _, _, comm = setup()
    comm.barrier(async_op=False).wait()
    teardown(comm)


def cmd_gather():
    rank, world, dev, comm = setup()
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
    teardown(comm)


def cmd_split():
    rank, world, dev, comm = setup()
    # Sub-comm with all current ranks (sub_size == parent_size).
    sub = comm.split(list(range(world)), f"sub_r{rank}")
    assert sub.get_size() == world, f"sub size {sub.get_size()}"
    # all_reduce on the sub-comm to verify it's actually wired.
    t = torch.full((4,), float(rank + 1), device=dev)
    sub.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False).wait()
    expected = float(sum(range(1, world + 1)))
    assert torch.allclose(t.cpu(), torch.full((4,), expected)), \
        f"sub all_reduce: {t.cpu()}"
    sub.finalize()
    teardown(comm)


def cmd_all_to_all_v_single():
    rank, world, dev, comm = setup()
    # Each rank sends a fixed-size chunk to each peer (uniform — exercises
    # the V API path with non-trivial split arrays). Uneven splits are not
    # covered yet.
    K = 4
    in_splits = [K] * world
    out_splits = [K] * world
    inp = torch.cat(
        [torch.full((K,), float(rank * 100 + i), device=dev) for i in range(world)]
    )
    out = torch.empty_like(inp)
    comm.all_to_all_v_single(out, inp, out_splits, in_splits,
                              async_op=False).wait()
    expected = torch.cat([torch.full((K,), float(i * 100 + rank))
                           for i in range(world)])
    assert torch.allclose(out.cpu(), expected), \
        f"a2a_v: rank {rank}: got {out.cpu()}"
    teardown(comm)


def cmd_aclgraph_single():
    rank, world, dev, comm = setup()
    capture_stream = torch_npu.npu.Stream(device=dev)
    expected = float(sum(range(1, world + 1)))
    # warmup outside capture (HCCL lazy init + caching alloc)
    with torch_npu.npu.stream(capture_stream):
        t_warm = torch.full((128,), float(rank + 1), device=dev,
                             dtype=torch.float32)
        comm.all_reduce(t_warm, torchcomms.ReduceOp.SUM, async_op=False)
        torch_npu.npu.synchronize()
    assert abs(t_warm.cpu()[0].item() - expected) < 1e-5
    # capture all_reduce
    g = torch_npu.npu.NPUGraph()
    t = torch.full((128,), float(rank + 1), device=dev, dtype=torch.float32)
    with torch_npu.npu.graph(g, stream=capture_stream):
        comm.all_reduce(t, torchcomms.ReduceOp.SUM, async_op=False)
    # replay × 3
    for _ in range(3):
        with torch_npu.npu.stream(capture_stream):
            t.fill_(float(rank + 1))
        g.replay()
        torch_npu.npu.synchronize()
        assert abs(t.cpu()[0].item() - expected) < 1e-5, \
            f"aclgraph replay: rank {rank} got {t.cpu()[0].item()}"
    teardown(comm)


def cmd_aclgraph_multi():
    rank, world, dev, comm = setup()
    capture_stream = torch_npu.npu.Stream(device=dev)
    K = 64
    fp32 = torch.float32
    t_ar = torch.empty(K, device=dev, dtype=fp32)
    expected_ar = float(sum(range(1, world + 1)))
    t_bcast = torch.empty(K, device=dev, dtype=fp32)
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
    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g, stream=capture_stream):
        comm.all_reduce(t_ar, torchcomms.ReduceOp.SUM, async_op=False)
        comm.broadcast(t_bcast, root=0, async_op=False)
        comm.all_gather_single(t_ag_out, t_ag_in, async_op=False)
        comm.reduce_scatter_single(t_rs_out, t_rs_in,
                                    torchcomms.ReduceOp.SUM, async_op=False)
    for _ in range(3):
        with torch_npu.npu.stream(capture_stream):
            t_ar.fill_(float(rank + 1))
            t_bcast.fill_(42.0 if rank == 0 else 0.0)
            t_ag_in.fill_(float(rank + 100))
            t_rs_in.fill_(float(rank + 1))
        g.replay()
        torch_npu.npu.synchronize()
        assert abs(t_ar.cpu()[0].item() - expected_ar) < 1e-5
        assert abs(t_bcast.cpu()[0].item() - 42.0) < 1e-5
        ag_cpu = t_ag_out.cpu()
        for r in range(world):
            assert abs(ag_cpu[r * K].item() - expected_ag[r]) < 1e-5, \
                f"ag rank{r}: {ag_cpu[r*K].item()}"
        assert abs(t_rs_out.cpu()[0].item() - expected_rs) < 1e-5
    teardown(comm)


COMMANDS = {
    "all_reduce": cmd_all_reduce,
    "broadcast": cmd_broadcast,
    "all_gather_single": cmd_all_gather_single,
    "reduce_scatter_single": cmd_reduce_scatter_single,
    "send_recv": cmd_send_recv,
    "barrier": cmd_barrier,
    "gather": cmd_gather,
    "all_to_all_v_single": cmd_all_to_all_v_single,
    "split": cmd_split,
    "aclgraph_single": cmd_aclgraph_single,
    "aclgraph_multi": cmd_aclgraph_multi,
}


if __name__ == "__main__":
    cmd_name = sys.argv[1]
    if cmd_name not in COMMANDS:
        print(f"unknown command: {cmd_name}; available: {list(COMMANDS)}")
        sys.exit(2)
    COMMANDS[cmd_name]()
    print(f"[rank {os.environ['RANK']}] {cmd_name} OK")
