// Copyright (c) 2026 Shan Shan.
//
// Isolated wrapper for c10_npu::getCurrentNPUStream.
// torch_npu/csrc/core/npu/NPUStream.h transitively includes a stale copy of
// acl headers (cann 8.x bundle) that conflict with cann 9.0 system acl_rt.h
// on aclCANNPackageVersion. Keeping this single torch_npu touchpoint in its
// own .cpp prevents the conflict from surfacing in NpuApiImpl.cpp.
//
// This file + NpuGuard.cpp are the only translation units that #include
// <torch_npu/...>. CI lint enforces this with grep.

#include <torch_npu/csrc/core/npu/NPUStream.h>

namespace torch::comms::detail {

// Returns the current NPU stream as an opaque pointer.
// aclrtStream is a typedef for void*, so callers reinterpret_cast back.
void* getCurrentNpuStreamRaw(int device_index) {
  return c10_npu::getCurrentNPUStream(device_index).stream();
}

}  // namespace torch::comms::detail
