# Minimal 2-rank ctypes probe — strips torchcomms / PyHcclBackend trampoline.
# Run: torchrun --nproc_per_node=2 _probe_2rank.py

import ctypes
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hccl_ffi as ffi  # noqa: E402

import torch_npu  # noqa: E402,F401  (PrivateUse1 register)
import torch.distributed as dist  # noqa: E402


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch_npu.npu.set_device(local)
    ffi.acl_check(ffi.aclrtSetDevice(local), "aclrtSetDevice")

    print(f"[r{rank}] loaded libs: {ffi.loaded_paths()}", flush=True)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    store = dist.distributed_c10d._get_default_store()

    key = "probe2rank_root_info"
    if rank == 0:
        buf = ctypes.create_string_buffer(ffi.ROOT_INFO_BYTES)
        ffi.check(
            ffi.HcclGetRootInfo(ctypes.cast(buf, ctypes.c_void_p)),
            "HcclGetRootInfo",
        )
        print(f"[r0] root info first 32B: {buf.raw[:32].hex()}", flush=True)
        store.set(key, bytes(buf.raw))
        root_bytes = bytes(buf.raw)
    else:
        store.wait([key], timedelta(seconds=30))
        root_bytes = bytes(store.get(key))
        print(f"[r{rank}] root info first 32B: {root_bytes[:32].hex()}",
              flush=True)

    assert len(root_bytes) == ffi.ROOT_INFO_BYTES, len(root_bytes)
    root_info = ctypes.create_string_buffer(root_bytes, ffi.ROOT_INFO_BYTES)

    print(f"[r{rank}] entering HcclCommInitRootInfo nRanks={world} rank={rank}",
          flush=True)
    comm = ctypes.c_void_p()
    ret = ffi.HcclCommInitRootInfo(
        world, ctypes.cast(root_info, ctypes.c_void_p),
        rank, ctypes.byref(comm),
    )
    print(f"[r{rank}] HcclCommInitRootInfo ret={ret} comm={comm.value}",
          flush=True)
    if ret != 0:
        raise SystemExit(1)

    ffi.HcclCommDestroy(comm)
    dist.destroy_process_group()
    print(f"[r{rank}] DONE", flush=True)


if __name__ == "__main__":
    main()
