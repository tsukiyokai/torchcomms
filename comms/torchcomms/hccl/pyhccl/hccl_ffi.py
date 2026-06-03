# ctypes wrappers for libhccl.so + minimal libascendcl.so symbols.
#
# Used by hccl_pyend.py to dispatch collectives directly to HCCL C API,
# bypassing torch_npu's ProcessGroupHCCL entirely.
#
# Verified against cann 9.0 headers:
#   include/hccl/hccl_types.h    HCCL_ROOT_INFO_BYTES=4108, enum values
#   include/hccl/hccl.h          collective signatures
#   include/hccl/hccl_comm.h     HcclGetRootInfo / HcclCommInit*
#   include/acl/acl_rt.h         aclrtSynchronizeStream / aclrtSetDevice
#
# IMPORTANT: this module never calls aclInit / aclFinalize. The acl context
# is owned by torch_npu (whichever module was imported first); a duplicate
# aclFinalize on shutdown causes "corrupted size vs. prev_size in fastbins"
# heap aborts (observed during ctypes probe phase).

from __future__ import annotations

import ctypes
import os
import struct
from typing import Any

import torch

# ---- ctypes type aliases ----

HcclResult = ctypes.c_uint
HcclDataType = ctypes.c_uint
HcclReduceOp = ctypes.c_uint
HcclComm = ctypes.c_void_p
aclrtStream = ctypes.c_void_p
aclError = ctypes.c_int32

# ---- constants from hccl_types.h ----

ROOT_INFO_BYTES = 4108

HCCL_SUCCESS = 0
HCCL_E_PARA = 1
HCCL_E_PTR = 2
HCCL_E_MEMORY = 3
HCCL_E_INTERNAL = 4
HCCL_E_NOT_SUPPORT = 5
HCCL_E_RUNTIME = 15

# Reduce ops
HCCL_REDUCE_SUM = 0
HCCL_REDUCE_PROD = 1
HCCL_REDUCE_MAX = 2
HCCL_REDUCE_MIN = 3

# Data types (subset that maps to torch dtypes in 2.10)
HCCL_DATA_TYPE_INT8 = 0
HCCL_DATA_TYPE_INT16 = 1
HCCL_DATA_TYPE_INT32 = 2
HCCL_DATA_TYPE_FP16 = 3
HCCL_DATA_TYPE_FP32 = 4
HCCL_DATA_TYPE_INT64 = 5
HCCL_DATA_TYPE_UINT64 = 6
HCCL_DATA_TYPE_UINT8 = 7
HCCL_DATA_TYPE_UINT16 = 8
HCCL_DATA_TYPE_UINT32 = 9
HCCL_DATA_TYPE_FP64 = 10
HCCL_DATA_TYPE_BFP16 = 11

# Send/recv types for batch_op_issue
HCCL_SEND = 0
HCCL_RECV = 1


class HcclSendRecvItem(ctypes.Structure):
    _fields_ = [
        ("sendRecvType", ctypes.c_uint),
        ("buf", ctypes.c_void_p),
        ("count", ctypes.c_uint64),
        ("dataType", ctypes.c_uint),
        ("remoteRank", ctypes.c_uint32),
    ]


# ---- HcclCommConfig (cann 9.0 hccl_types.h HcclCommConfigDef) ----
# sizeof = 2240 bytes on aarch64. The first 24 bytes are the configInfo_t
# overlay {size_t size, uint32 magicWord, uint32 version, uint64 reserved}.

_HCCL_COMM_CONFIG_MAGIC_WORD = 0xF0F0F0F0
_HCCL_COMM_CONFIG_VERSION = 10
_HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET = -1  # 0xffffffff cast to int32
_HCCL_COMM_QOS_CONFIG_NOT_SET = 0xFFFFFFFF
_HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE = 16


class HcclCommConfig(ctypes.Structure):
    _fields_ = [
        ("reserved", ctypes.c_char * 24),                # configInfo_t overlay
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


def hccl_comm_config_init(cfg: HcclCommConfig) -> None:
    """Reproduce cann 9.0 hccl_comm.h:197 static inline HcclCommConfigInit."""
    info_blob = struct.pack(
        "=QIIQ",
        ctypes.sizeof(HcclCommConfig),
        _HCCL_COMM_CONFIG_MAGIC_WORD,
        _HCCL_COMM_CONFIG_VERSION,
        0,
    )
    ctypes.memmove(ctypes.addressof(cfg), info_blob, 24)
    cfg.hcclBufferSize = _HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET
    cfg.hcclDeterministic = _HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET
    cfg.hcclCommName = b""
    cfg.hcclUdi = b""
    cfg.hcclOpExpansionMode = 0
    cfg.hcclRdmaTrafficClass = _HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET
    cfg.hcclRdmaServiceLevel = _HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET
    cfg.hcclWorldRankID = 0
    cfg.hcclJobID = 0
    cfg.aclGraphZeroCopyEnable = 0
    cfg.hcclExecTimeOut = _HCCL_COMM_EXECTIMEOUT_CONFIG_NOT_SET
    cfg.hcclAlgo = b""
    cfg.hcclRetryEnable = b""
    cfg.hcclRetryParams = b""
    cfg.hcclBufferName = b""
    cfg.hcclQos = _HCCL_COMM_QOS_CONFIG_NOT_SET
    cfg.hcclSymWinMaxMemSizePerRank = _HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE


# ---- library handles ----
#
# Default to bare soname so LD_LIBRARY_PATH decides — on this box,
# `source /home/shanshan/Ascend-9.0/set_env.sh` puts cann 9.0
# (/home/shanshan/Ascend-9.0/cann-9.0.0/...) ahead of the system cann 8.5
# (/usr/local/Ascend/cann-8.5.0/...). The project contract is cann 9.0;
# torch_npu 2.10rc2's cann 8.5 ABI is its own internal compatibility issue.
# PYHCCL_LIBHCCL / PYHCCL_LIBACL escape hatches stay for diagnostics.

_HCCL_PATH = os.environ.get("PYHCCL_LIBHCCL", "libhccl.so")
_ACL_PATH = os.environ.get("PYHCCL_LIBACL", "libascendcl.so")

_libhccl = ctypes.CDLL(_HCCL_PATH)
_libacl = ctypes.CDLL(_ACL_PATH)


def _resolve_loaded_lib_path(soname: str) -> str:
    """Walk /proc/self/maps to discover which libhccl.so is actually
    mapped into this process (dlopen merges by soname, so the first
    loader wins — torch_npu may have pinned cann 8.5 before we got here)."""
    try:
        with open(f"/proc/{os.getpid()}/maps") as f:
            for line in f:
                if soname in line:
                    parts = line.rstrip().split()
                    return parts[-1] if parts else "?"
    except OSError:
        pass
    return "?"


def loaded_paths() -> dict:
    """Diagnostic helper — call from backend init to verify which lib is live."""
    return {
        "libhccl.so": _resolve_loaded_lib_path("libhccl.so"),
        "libascendcl.so": _resolve_loaded_lib_path("libascendcl.so"),
        "libhcomm.so": _resolve_loaded_lib_path("libhcomm.so"),
    }


def _bind(lib, name, argtypes, restype):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


# ---- communicator management (6) ----

HcclGetRootInfo = _bind(
    _libhccl, "HcclGetRootInfo", [ctypes.c_void_p], HcclResult
)
HcclCommInitRootInfo = _bind(
    _libhccl, "HcclCommInitRootInfo",
    [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(HcclComm)],
    HcclResult,
)
HcclCommDestroy = _bind(
    _libhccl, "HcclCommDestroy", [HcclComm], HcclResult
)
HcclGetRankId = _bind(
    _libhccl, "HcclGetRankId",
    [HcclComm, ctypes.POINTER(ctypes.c_uint32)], HcclResult,
)
HcclGetRankSize = _bind(
    _libhccl, "HcclGetRankSize",
    [HcclComm, ctypes.POINTER(ctypes.c_uint32)], HcclResult,
)
HcclGetCommAsyncError = _bind(
    _libhccl, "HcclGetCommAsyncError",
    [HcclComm, ctypes.POINTER(HcclResult)], HcclResult,
)

# HcclCreateSubCommConfig is HCOMM_WEAK_SYMBOL on cann 9.0; verified callable
# from ctypes (see _probe_split.py — sub-comm built in ~10ms with ret=0).
HcclCreateSubCommConfig = _bind(
    _libhccl, "HcclCreateSubCommConfig",
    [
        ctypes.POINTER(HcclComm),         # parent comm*
        ctypes.c_uint32,                   # rankNum
        ctypes.POINTER(ctypes.c_uint32),   # rankIds*
        ctypes.c_uint64,                   # subCommId
        ctypes.c_uint32,                   # subCommRankId
        ctypes.POINTER(HcclCommConfig),    # config*
        ctypes.POINTER(HcclComm),          # subComm*
    ],
    HcclResult,
)

# ---- collectives (14) ----

HcclAllReduce = _bind(
    _libhccl, "HcclAllReduce",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
     HcclDataType, HcclReduceOp, HcclComm, aclrtStream],
    HcclResult,
)
HcclBroadcast = _bind(
    _libhccl, "HcclBroadcast",
    [ctypes.c_void_p, ctypes.c_uint64, HcclDataType, ctypes.c_uint32,
     HcclComm, aclrtStream],
    HcclResult,
)
HcclReduce = _bind(
    _libhccl, "HcclReduce",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
     HcclDataType, HcclReduceOp, ctypes.c_uint32, HcclComm, aclrtStream],
    HcclResult,
)
HcclAllGather = _bind(
    _libhccl, "HcclAllGather",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
     HcclDataType, HcclComm, aclrtStream],
    HcclResult,
)
HcclAllGatherV = _bind(
    _libhccl, "HcclAllGatherV",
    [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
     ctypes.c_void_p, ctypes.c_void_p,
     HcclDataType, HcclComm, aclrtStream],
    HcclResult,
)
HcclReduceScatter = _bind(
    _libhccl, "HcclReduceScatter",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
     HcclDataType, HcclReduceOp, HcclComm, aclrtStream],
    HcclResult,
)
HcclReduceScatterV = _bind(
    _libhccl, "HcclReduceScatterV",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
     ctypes.c_void_p, ctypes.c_uint64,
     HcclDataType, HcclReduceOp, HcclComm, aclrtStream],
    HcclResult,
)
HcclScatter = _bind(
    _libhccl, "HcclScatter",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
     HcclDataType, ctypes.c_uint32, HcclComm, aclrtStream],
    HcclResult,
)
HcclSend = _bind(
    _libhccl, "HcclSend",
    [ctypes.c_void_p, ctypes.c_uint64, HcclDataType, ctypes.c_uint32,
     HcclComm, aclrtStream],
    HcclResult,
)
HcclRecv = _bind(
    _libhccl, "HcclRecv",
    [ctypes.c_void_p, ctypes.c_uint64, HcclDataType, ctypes.c_uint32,
     HcclComm, aclrtStream],
    HcclResult,
)
HcclAlltoAll = _bind(
    _libhccl, "HcclAlltoAll",
    [ctypes.c_void_p, ctypes.c_uint64, HcclDataType,
     ctypes.c_void_p, ctypes.c_uint64, HcclDataType,
     HcclComm, aclrtStream],
    HcclResult,
)
HcclAlltoAllV = _bind(
    _libhccl, "HcclAlltoAllV",
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, HcclDataType,
     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, HcclDataType,
     HcclComm, aclrtStream],
    HcclResult,
)
HcclBatchSendRecv = _bind(
    _libhccl, "HcclBatchSendRecv",
    [ctypes.POINTER(HcclSendRecvItem), ctypes.c_uint32, HcclComm, aclrtStream],
    HcclResult,
)

# ---- acl runtime (just sync; device set is done by torch_npu) ----

aclrtSynchronizeStream = _bind(
    _libacl, "aclrtSynchronizeStream", [aclrtStream], aclError
)
aclrtSetDevice = _bind(
    _libacl, "aclrtSetDevice", [ctypes.c_int32], aclError
)
aclrtGetDevice = _bind(
    _libacl, "aclrtGetDevice", [ctypes.POINTER(ctypes.c_int32)], aclError
)

# ---- aclmdlRICapture (cann 9.0 acl_rt.h:4023+) — for aclgraph support ----
# HCCL self-supports stream capture (probe-verified, see _probe_capture.py).
# Backend uses aclmdlRICaptureGetInfo to skip host-side sync inside an
# active capture so it doesn't break the recording.

ACLMDL_RI_CAPTURE_STATUS_NONE = 0
ACLMDL_RI_CAPTURE_STATUS_ACTIVE = 1
ACLMDL_RI_CAPTURE_STATUS_INVALIDATED = 2

aclmdlRI = ctypes.c_void_p

aclmdlRICaptureGetInfo = _bind(
    _libacl, "aclmdlRICaptureGetInfo",
    [aclrtStream, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(aclmdlRI)],
    aclError,
)


def stream_is_capturing(stream: ctypes.c_void_p) -> bool:
    """True iff `stream` is in aclmdlRICapture ACTIVE state."""
    status = ctypes.c_uint(0)
    ri = aclmdlRI()
    ret = aclmdlRICaptureGetInfo(stream, ctypes.byref(status), ctypes.byref(ri))
    return ret == 0 and status.value == ACLMDL_RI_CAPTURE_STATUS_ACTIVE


# ---- error helpers ----

class HcclError(RuntimeError):
    pass


def check(ret: int, op: str) -> None:
    if ret != HCCL_SUCCESS:
        raise HcclError(f"{op} failed: HcclResult={ret}")


def acl_check(ret: int, op: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{op} failed: aclError={ret}")


# ---- torch.dtype -> HcclDataType ----

_TORCH_TO_HCCL = {
    torch.int8:    HCCL_DATA_TYPE_INT8,
    torch.int16:   HCCL_DATA_TYPE_INT16,
    torch.int32:   HCCL_DATA_TYPE_INT32,
    torch.float16: HCCL_DATA_TYPE_FP16,
    torch.float32: HCCL_DATA_TYPE_FP32,
    torch.int64:   HCCL_DATA_TYPE_INT64,
    torch.uint8:   HCCL_DATA_TYPE_UINT8,
    torch.float64: HCCL_DATA_TYPE_FP64,
    torch.bfloat16: HCCL_DATA_TYPE_BFP16,
}


def torch_dtype_to_hccl(dt: torch.dtype) -> int:
    try:
        return _TORCH_TO_HCCL[dt]
    except KeyError:
        raise RuntimeError(f"unsupported dtype for HCCL: {dt}") from None


# ---- ReduceOp -> HcclReduceOp ----

_REDUCE_OP_NAMES = {
    "SUM": HCCL_REDUCE_SUM,
    "PRODUCT": HCCL_REDUCE_PROD,
    "PROD": HCCL_REDUCE_PROD,
    "MIN": HCCL_REDUCE_MIN,
    "MAX": HCCL_REDUCE_MAX,
}


def reduce_op_to_hccl(op: Any) -> int:
    # torchcomms.ReduceOp is a pybind class wrapping a RedOpType enum;
    # .type yields the underlying enum which exposes .name.
    name = None
    type_attr = getattr(op, "type", None)
    if type_attr is not None and hasattr(type_attr, "name"):
        name = type_attr.name
    elif hasattr(op, "name"):
        name = op.name
    else:
        name = str(op).split(".")[-1]
    name = str(name).upper()
    if name not in _REDUCE_OP_NAMES:
        raise RuntimeError(
            f"HCCL only supports SUM/PROD/MIN/MAX; got {name!r}"
        )
    return _REDUCE_OP_NAMES[name]


# ---- helpers for buffer ptr extraction ----

def tensor_void_p(t: torch.Tensor) -> ctypes.c_void_p:
    """Get device pointer of an NPU tensor as ctypes void*."""
    return ctypes.c_void_p(t.data_ptr())


def current_npu_stream() -> ctypes.c_void_p:
    """Pull the raw aclrtStream pointer from torch_npu's current stream.

    This is the only place we touch torch_npu in collective dispatch:
    we reuse its caching-allocator-managed stream so HCCL ops serialize
    with prior torch ops on the same tensor without any extra sync.
    """
    import torch_npu
    return ctypes.c_void_p(torch_npu.npu.current_stream().npu_stream)
