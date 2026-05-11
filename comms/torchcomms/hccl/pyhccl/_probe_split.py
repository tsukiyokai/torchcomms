# Probe HcclCreateSubCommConfig on cann 9.0 — does the weak symbol actually
# do something, or hang/PARA as native v0.12 reported?
#
# Strategy: bootstrap a parent HCCL comm (proven path), then try
# HcclCreateSubCommConfig with all ranks (degenerate full-overlap sub-comm).
# Print the return code + elapsed time; on success, destroy the sub comm.
#
# Run: torchrun --nproc_per_node=2 _probe_split.py

import ctypes
import os
import struct
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hccl_ffi as ffi  # noqa: E402

import torch_npu  # noqa: E402,F401
import torch.distributed as dist  # noqa: E402


# Mirror cann 9.0 hccl_types.h HcclCommConfigDef byte-for-byte.
class HcclCommConfig(ctypes.Structure):
    _fields_ = [
        ("reserved", ctypes.c_char * 24),               # configInfo_t overlay
        ("hcclBufferSize", ctypes.c_uint32),
        ("hcclDeterministic", ctypes.c_uint32),
        ("hcclCommName", ctypes.c_char * 128),
        ("hcclUdi", ctypes.c_char * 128),
        ("hcclOpExpansionMode", ctypes.c_uint32),
        ("hcclRdmaTrafficClass", ctypes.c_uint32),
        ("hcclRdmaServiceLevel", ctypes.c_uint32),
        ("hcclWorldRankID", ctypes.c_uint32),
        ("hcclJobID", ctypes.c_uint64),
        ("aclGraphZeroCopyEnable", ctypes.c_uint8),
        # 3 bytes implicit padding for int32 alignment
        ("hcclExecTimeOut", ctypes.c_int32),
        ("hcclAlgo", ctypes.c_char * 1600),
        ("hcclRetryEnable", ctypes.c_char * 50),
        ("hcclRetryParams", ctypes.c_char * 128),
        ("hcclBufferName", ctypes.c_char * 128),
        ("hcclQos", ctypes.c_uint32),
        # 4 bytes implicit padding for uint64 alignment
        ("hcclSymWinMaxMemSizePerRank", ctypes.c_uint64),
    ]


# Constants from hccl_types.h
HCCL_COMM_CONFIG_MAGIC_WORD = 0xF0F0F0F0
HCCL_COMM_CONFIG_VERSION = 10
HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_DEFAULT_OP_EXPANSION_MODE = 0
HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET = -1  # 0xffffffff cast to int32
HCCL_COMM_QOS_CONFIG_NOT_SET = 0xFFFFFFFF
HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE = 16


def hccl_comm_config_init(cfg: HcclCommConfig) -> None:
    # Reproduce cann 9.0 hccl_comm.h:197 static inline HcclCommConfigInit.
    info_blob = struct.pack(
        "=QIIQ",
        ctypes.sizeof(HcclCommConfig),  # size_t size
        HCCL_COMM_CONFIG_MAGIC_WORD,    # uint32 magicWord
        HCCL_COMM_CONFIG_VERSION,       # uint32 version
        0,                               # uint64 reserved
    )
    ctypes.memmove(ctypes.addressof(cfg), info_blob, 24)
    cfg.hcclBufferSize = HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET
    cfg.hcclDeterministic = HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET
    cfg.hcclCommName = b""
    cfg.hcclUdi = b""
    cfg.hcclOpExpansionMode = HCCL_COMM_DEFAULT_OP_EXPANSION_MODE
    cfg.hcclRdmaTrafficClass = HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET
    cfg.hcclRdmaServiceLevel = HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET
    cfg.hcclWorldRankID = 0
    cfg.hcclJobID = 0
    cfg.aclGraphZeroCopyEnable = 0
    cfg.hcclExecTimeOut = HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET
    cfg.hcclAlgo = b""
    cfg.hcclRetryEnable = b""
    cfg.hcclRetryParams = b""
    cfg.hcclBufferName = b""
    cfg.hcclQos = HCCL_COMM_QOS_CONFIG_NOT_SET
    cfg.hcclSymWinMaxMemSizePerRank = HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE


def bind_create_sub():
    fn = ffi._libhccl.HcclCreateSubCommConfig
    fn.argtypes = [
        ctypes.POINTER(ffi.HcclComm),       # parent comm*
        ctypes.c_uint32,                    # rankNum
        ctypes.POINTER(ctypes.c_uint32),    # rankIds*
        ctypes.c_uint64,                    # subCommId
        ctypes.c_uint32,                    # subCommRankId
        ctypes.POINTER(HcclCommConfig),     # config*
        ctypes.POINTER(ffi.HcclComm),       # subComm*
    ]
    fn.restype = ctypes.c_uint
    return fn


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch_npu.npu.set_device(local)
    ffi.acl_check(ffi.aclrtSetDevice(local), "aclrtSetDevice")

    print(f"[r{rank}] sizeof(HcclCommConfig) = {ctypes.sizeof(HcclCommConfig)}",
          flush=True)
    print(f"[r{rank}] loaded libs: {ffi.loaded_paths()}", flush=True)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    store = dist.distributed_c10d._get_default_store()

    # ---- bootstrap parent comm (proven path) ----
    key = "probe_split_root"
    if rank == 0:
        buf = ctypes.create_string_buffer(ffi.ROOT_INFO_BYTES)
        ffi.check(
            ffi.HcclGetRootInfo(ctypes.cast(buf, ctypes.c_void_p)),
            "HcclGetRootInfo",
        )
        store.set(key, bytes(buf.raw))
        root_bytes = bytes(buf.raw)
    else:
        store.wait([key], timedelta(seconds=30))
        root_bytes = bytes(store.get(key))

    root_info = ctypes.create_string_buffer(root_bytes, ffi.ROOT_INFO_BYTES)
    parent_comm = ffi.HcclComm()
    print(f"[r{rank}] HcclCommInitRootInfo nRanks={world}", flush=True)
    ffi.check(
        ffi.HcclCommInitRootInfo(
            world, ctypes.cast(root_info, ctypes.c_void_p),
            rank, ctypes.byref(parent_comm),
        ),
        "HcclCommInitRootInfo",
    )
    print(f"[r{rank}] parent comm = {parent_comm.value}", flush=True)

    # ---- the actual probe: HcclCreateSubCommConfig ----
    HcclCreateSubCommConfig = bind_create_sub()

    rank_ids = (ctypes.c_uint32 * world)(*range(world))
    sub_comm_id = 0xCAFEBABE  # arbitrary but stable across ranks
    cfg = HcclCommConfig()
    hccl_comm_config_init(cfg)

    # Verify config blob looks right (bytes(c_char*24) NUL-truncates, use string_at)
    info = struct.unpack_from(
        "=QIIQ", ctypes.string_at(ctypes.addressof(cfg), 24)
    )
    print(f"[r{rank}] config configInfo: size={info[0]} magic=0x{info[1]:x} "
          f"version={info[2]} reserved={info[3]}", flush=True)

    sub_comm = ffi.HcclComm()
    print(f"[r{rank}] entering HcclCreateSubCommConfig "
          f"rankNum={world} subCommId=0x{sub_comm_id:x} subCommRankId={rank}",
          flush=True)
    t0 = time.time()
    ret = HcclCreateSubCommConfig(
        ctypes.byref(parent_comm),
        world,
        rank_ids,
        sub_comm_id,
        rank,
        ctypes.byref(cfg),
        ctypes.byref(sub_comm),
    )
    elapsed = time.time() - t0
    print(f"[r{rank}] HcclCreateSubCommConfig ret={ret} "
          f"sub_comm={sub_comm.value} elapsed={elapsed:.2f}s",
          flush=True)

    if ret == 0 and sub_comm.value:
        ffi.HcclCommDestroy(sub_comm)
        print(f"[r{rank}] sub_comm destroyed", flush=True)

    ffi.HcclCommDestroy(parent_comm)
    dist.destroy_process_group()
    print(f"[r{rank}] DONE", flush=True)


if __name__ == "__main__":
    main()
