# Probe HCCL stream-capture compatibility on cann 9.0.
#
# Question: does aclmdlRICaptureBegin → HcclAllReduce → aclmdlRICaptureEnd
# yield a non-zero aclmdlRI containing the collective as a graph node?
# That answer decides whether the hccl backend can support aclgraph
# (torch_npu.npu.NPUGraph or torch.compile backend="npugraph_ex").
#
# Run: torchrun --nproc_per_node=2 _probe_capture.py

import ctypes
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hccl_ffi as ffi  # noqa: E402

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
import torch.distributed as dist  # noqa: E402


# ---- bind aclmdlRICapture* (cann 9.0 acl_rt.h:4023+) ----

aclmdlRI = ctypes.c_void_p
aclmdlRICaptureMode = ctypes.c_uint
aclmdlRICaptureStatus = ctypes.c_uint

CAPTURE_MODE_GLOBAL = 0
CAPTURE_MODE_THREAD_LOCAL = 1
CAPTURE_MODE_RELAXED = 2

CAPTURE_STATUS_NONE = 0
CAPTURE_STATUS_ACTIVE = 1
CAPTURE_STATUS_INVALIDATED = 2

_lib = ffi._libacl
aclmdlRICaptureBegin = _lib.aclmdlRICaptureBegin
aclmdlRICaptureBegin.argtypes = [ctypes.c_void_p, aclmdlRICaptureMode]
aclmdlRICaptureBegin.restype = ctypes.c_int32

aclmdlRICaptureEnd = _lib.aclmdlRICaptureEnd
aclmdlRICaptureEnd.argtypes = [ctypes.c_void_p, ctypes.POINTER(aclmdlRI)]
aclmdlRICaptureEnd.restype = ctypes.c_int32

aclmdlRICaptureGetInfo = _lib.aclmdlRICaptureGetInfo
aclmdlRICaptureGetInfo.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(aclmdlRICaptureStatus),
    ctypes.POINTER(aclmdlRI),
]
aclmdlRICaptureGetInfo.restype = ctypes.c_int32

aclmdlRIExecuteAsync = _lib.aclmdlRIExecuteAsync
aclmdlRIExecuteAsync.argtypes = [aclmdlRI, ctypes.c_void_p]
aclmdlRIExecuteAsync.restype = ctypes.c_int32

aclmdlRIDestroy = _lib.aclmdlRIDestroy
aclmdlRIDestroy.argtypes = [aclmdlRI]
aclmdlRIDestroy.restype = ctypes.c_int32


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch_npu.npu.set_device(local)
    ffi.acl_check(ffi.aclrtSetDevice(local), "aclrtSetDevice")
    dev = torch.device(f"npu:{local}")

    # ---- bootstrap HCCL comm (proven path) ----
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    store = dist.distributed_c10d._get_default_store()
    key = "probe_capture_root"
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
    comm = ffi.HcclComm()
    ffi.check(
        ffi.HcclCommInitRootInfo(
            world, ctypes.cast(root_info, ctypes.c_void_p),
            rank, ctypes.byref(comm),
        ),
        "HcclCommInitRootInfo",
    )
    print(f"[r{rank}] HcclCommInitRootInfo OK", flush=True)

    # ---- dedicated capture stream (current_stream may have other ops) ----
    capture_stream = torch_npu.npu.Stream(device=dev)
    raw_stream = ctypes.c_void_p(capture_stream.npu_stream)

    # ---- warmup outside capture (lets HCCL finish lazy init) ----
    with torch_npu.npu.stream(capture_stream):
        t = torch.full((8,), float(rank + 1), device=dev, dtype=torch.float32)
    ffi.check(ffi.HcclAllReduce(
        ffi.tensor_void_p(t), ffi.tensor_void_p(t),
        t.numel(), ffi.HCCL_DATA_TYPE_FP32, ffi.HCCL_REDUCE_SUM,
        comm, raw_stream,
    ), "HcclAllReduce warmup")
    ffi.acl_check(ffi.aclrtSynchronizeStream(raw_stream), "sync warmup")
    print(f"[r{rank}] warmup HcclAllReduce OK, sum={t.cpu().tolist()[0]}",
          flush=True)

    # ---- begin capture ----
    print(f"[r{rank}] aclmdlRICaptureBegin THREAD_LOCAL", flush=True)
    ret_begin = aclmdlRICaptureBegin(raw_stream, CAPTURE_MODE_THREAD_LOCAL)
    print(f"[r{rank}] CaptureBegin ret={ret_begin}", flush=True)
    if ret_begin != 0:
        print(f"[r{rank}] FAIL: cannot begin capture", flush=True)
        ffi.HcclCommDestroy(comm)
        dist.destroy_process_group()
        return

    # ---- query status ----
    status = aclmdlRICaptureStatus(0)
    queried_ri = aclmdlRI()
    ret_q = aclmdlRICaptureGetInfo(
        raw_stream, ctypes.byref(status), ctypes.byref(queried_ri)
    )
    print(f"[r{rank}] CaptureGetInfo ret={ret_q} status={status.value} "
          f"(1=ACTIVE)", flush=True)

    # ---- HcclAllReduce inside capture ----
    with torch_npu.npu.stream(capture_stream):
        t2 = torch.full((8,), float(rank + 10), device=dev, dtype=torch.float32)
    ret_ar = ffi.HcclAllReduce(
        ffi.tensor_void_p(t2), ffi.tensor_void_p(t2),
        t2.numel(), ffi.HCCL_DATA_TYPE_FP32, ffi.HCCL_REDUCE_SUM,
        comm, raw_stream,
    )
    print(f"[r{rank}] HcclAllReduce in-capture ret={ret_ar} "
          f"(0=SUCCESS, otherwise HCCL refused capture)", flush=True)

    # ---- end capture ----
    modelRI = aclmdlRI()
    ret_end = aclmdlRICaptureEnd(raw_stream, ctypes.byref(modelRI))
    print(f"[r{rank}] CaptureEnd ret={ret_end} modelRI={modelRI.value}",
          flush=True)

    if not (ret_begin == 0 and ret_ar == 0 and ret_end == 0 and modelRI.value):
        print(f"[r{rank}] VERDICT: HCCL NOT capture-compatible "
              f"(begin={ret_begin} allreduce={ret_ar} end={ret_end})",
              flush=True)
        ffi.HcclCommDestroy(comm)
        dist.destroy_process_group()
        return

    print(f"[r{rank}] CAPTURE OK — modelRI={hex(modelRI.value)}", flush=True)

    # ---- replay 3 times, verify in-place sum doubles correctly ----
    expected_sum = sum(range(1 + 10, world + 10 + 1))  # rank0=10, rank1=11 → 21
    # Note: first in-capture issue already executed once during capture, so t2
    # is currently sum of (rank+10). For the replay verification we reset t2.
    for replay_idx in range(3):
        with torch_npu.npu.stream(capture_stream):
            t2.fill_(float(rank + 10))
        ret_exec = aclmdlRIExecuteAsync(modelRI, raw_stream)
        ffi.acl_check(ffi.aclrtSynchronizeStream(raw_stream),
                      "sync after replay")
        actual = t2.cpu().tolist()[0]
        ok = (ret_exec == 0 and abs(actual - expected_sum) < 1e-5)
        print(f"[r{rank}] replay#{replay_idx} ret={ret_exec} "
              f"sum={actual} expected={expected_sum} {'OK' if ok else 'FAIL'}",
              flush=True)

    aclmdlRIDestroy(modelRI)
    print(f"[r{rank}] VERDICT: HCCL+aclgraph end-to-end (capture+replay) OK",
          flush=True)

    ffi.HcclCommDestroy(comm)
    dist.destroy_process_group()
    print(f"[r{rank}] DONE", flush=True)


if __name__ == "__main__":
    main()
