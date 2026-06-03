// Copyright (c) 2026 Shan Shan.
//
// NpuApi: Ascend ACL runtime + torch_npu c10 binding abstraction.
// All access to libascendcl + torch_npu funnels through this interface so
// that pta upgrades only break NpuApiImpl.cpp, not the rest of hccl backend.
//
// Mirrors comms/torchcomms/device/cuda/CudaApi.hpp; after collapsing
// CUDA-Graph-only methods (no NPU equivalent) and trimming stream-priority
// APIs, NpuApi has 22 methods total (5 device + 5 stream + 6 memory +
// 5 event + 1 error).

#pragma once

#include <acl/acl_rt.h>
#include <glog/logging.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace torch::comms {

#define NPU_CHECK(npu_api, call, err_str)                                  \
  do {                                                                     \
    aclError _status = (call);                                             \
    if (_status != ACL_ERROR_NONE) {                                       \
      std::stringstream _ss;                                               \
      _ss << err_str << ": " << (npu_api)->getErrorString(_status)         \
          << " at " << __FILE__ << ":" << __LINE__;                        \
      throw std::runtime_error(_ss.str());                                 \
    }                                                                      \
  } while (0)

#define NPU_CHECK_IGNORE(npu_api, call, err_str)                           \
  do {                                                                     \
    aclError _status = (call);                                             \
    if (_status != ACL_ERROR_NONE) {                                       \
      LOG(ERROR) << "[TC] " << err_str << ": "                             \
                 << (npu_api)->getErrorString(_status) << " at "           \
                 << __FILE__ << ":" << __LINE__;                           \
    }                                                                      \
  } while (0)

class NpuApi {
 public:
  virtual ~NpuApi() = default;

  // ---- Device management (5) ----
  [[nodiscard]] virtual aclError setDevice(int device) = 0;
  [[nodiscard]] virtual aclError getDevice(int* device) = 0;
  [[nodiscard]] virtual aclError getDeviceCount(uint32_t* count) = 0;
  [[nodiscard]] virtual aclError memGetInfo(size_t* free, size_t* total) = 0;
  // getDeviceProperties: cann has no exact CUDA equivalent. A future
  // revision can fold this into a SocName + capability struct. For now opaque.
  virtual const char* getSocName(int device) = 0;

  // ---- Stream management (5) ----
  [[nodiscard]] virtual aclError streamCreate(aclrtStream* pStream) = 0;
  [[nodiscard]] virtual aclError streamDestroy(aclrtStream stream) = 0;
  [[nodiscard]] virtual aclError streamWaitEvent(aclrtStream stream,
                                                  aclrtEvent event) = 0;
  [[nodiscard]] virtual aclError streamSynchronize(aclrtStream stream) = 0;
  // Only entry point that touches c10_npu::getCurrentNPUStream.
  virtual aclrtStream getCurrentNPUStream(int device_index) = 0;

  // ---- Memory management (6) ----
  [[nodiscard]] virtual aclError malloc(void** devPtr, size_t size) = 0;
  [[nodiscard]] virtual aclError free(void* devPtr) = 0;
  [[nodiscard]] virtual aclError memcpy(void* dst, const void* src,
                                         size_t count,
                                         aclrtMemcpyKind kind) = 0;
  [[nodiscard]] virtual aclError memcpyAsync(void* dst, const void* src,
                                              size_t count,
                                              aclrtMemcpyKind kind,
                                              aclrtStream stream) = 0;
  [[nodiscard]] virtual aclError hostAlloc(void** pHost, size_t size,
                                            uint32_t flags) = 0;
  [[nodiscard]] virtual aclError hostFree(void* ptr) = 0;

  // ---- Event management (5) ----
  [[nodiscard]] virtual aclError eventCreate(aclrtEvent* event) = 0;
  [[nodiscard]] virtual aclError eventCreateWithFlags(aclrtEvent* event,
                                                       uint32_t flags) = 0;
  [[nodiscard]] virtual aclError eventDestroy(aclrtEvent event) = 0;
  [[nodiscard]] virtual aclError eventRecord(aclrtEvent event,
                                              aclrtStream stream) = 0;
  [[nodiscard]] virtual aclError eventQuery(aclrtEvent event) = 0;

  // ---- Error handling (1) ----
  // Note: cann has no aclGetErrorString. Implementation maps errno -> string
  // via static table + falls back to aclGetRecentErrMsg (thread-local buffer).
  virtual const char* getErrorString(aclError error) = 0;
};

// Default impl: forwards to acl_rt + torch_npu. Lives in NpuApiImpl.cpp.
// This is the single allowed file that #include's <torch_npu/...>.
class DefaultNpuApi : public NpuApi {
 public:
  ~DefaultNpuApi() override = default;

  [[nodiscard]] aclError setDevice(int device) override;
  [[nodiscard]] aclError getDevice(int* device) override;
  [[nodiscard]] aclError getDeviceCount(uint32_t* count) override;
  [[nodiscard]] aclError memGetInfo(size_t* free, size_t* total) override;
  const char* getSocName(int device) override;

  [[nodiscard]] aclError streamCreate(aclrtStream* pStream) override;
  [[nodiscard]] aclError streamDestroy(aclrtStream stream) override;
  [[nodiscard]] aclError streamWaitEvent(aclrtStream stream,
                                          aclrtEvent event) override;
  [[nodiscard]] aclError streamSynchronize(aclrtStream stream) override;
  aclrtStream getCurrentNPUStream(int device_index) override;

  [[nodiscard]] aclError malloc(void** devPtr, size_t size) override;
  [[nodiscard]] aclError free(void* devPtr) override;
  [[nodiscard]] aclError memcpy(void* dst, const void* src, size_t count,
                                 aclrtMemcpyKind kind) override;
  [[nodiscard]] aclError memcpyAsync(void* dst, const void* src, size_t count,
                                      aclrtMemcpyKind kind,
                                      aclrtStream stream) override;
  [[nodiscard]] aclError hostAlloc(void** pHost, size_t size,
                                    uint32_t flags) override;
  [[nodiscard]] aclError hostFree(void* ptr) override;

  [[nodiscard]] aclError eventCreate(aclrtEvent* event) override;
  [[nodiscard]] aclError eventCreateWithFlags(aclrtEvent* event,
                                               uint32_t flags) override;
  [[nodiscard]] aclError eventDestroy(aclrtEvent event) override;
  [[nodiscard]] aclError eventRecord(aclrtEvent event,
                                      aclrtStream stream) override;
  [[nodiscard]] aclError eventQuery(aclrtEvent event) override;

  const char* getErrorString(aclError error) override;
};

}  // namespace torch::comms
