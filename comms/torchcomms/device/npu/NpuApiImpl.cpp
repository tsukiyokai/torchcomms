// Copyright (c) 2026 Shan Shan.
//
// NpuApi default implementation. Uses cann 9.0 system acl headers ONLY;
// the c10_npu::getCurrentNPUStream call is delegated to NpuStreamImpl.cpp
// to keep torch_npu's bundled (older) acl headers out of this translation
// unit — they conflict with cann 9.0 acl_rt.h on aclCANNPackageVersion.

#include "comms/torchcomms/device/npu/NpuApi.hpp"

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <unordered_map>

namespace torch::comms {

namespace detail {
// Implemented in NpuStreamImpl.cpp.
void* getCurrentNpuStreamRaw(int device_index);
}  // namespace detail


// ---- Device ----

aclError DefaultNpuApi::setDevice(int device) {
  return aclrtSetDevice(device);
}

aclError DefaultNpuApi::getDevice(int* device) {
  return aclrtGetDevice(device);
}

aclError DefaultNpuApi::getDeviceCount(uint32_t* count) {
  return aclrtGetDeviceCount(count);
}

aclError DefaultNpuApi::memGetInfo(size_t* free_size, size_t* total_size) {
  return aclrtGetMemInfo(ACL_HBM_MEM, free_size, total_size);
}

const char* DefaultNpuApi::getSocName(int /*device*/) {
  return aclrtGetSocName();
}

// ---- Stream ----

aclError DefaultNpuApi::streamCreate(aclrtStream* pStream) {
  return aclrtCreateStream(pStream);
}

aclError DefaultNpuApi::streamDestroy(aclrtStream stream) {
  return aclrtDestroyStream(stream);
}

aclError DefaultNpuApi::streamWaitEvent(aclrtStream stream, aclrtEvent event) {
  return aclrtStreamWaitEvent(stream, event);
}

aclError DefaultNpuApi::streamSynchronize(aclrtStream stream) {
  return aclrtSynchronizeStream(stream);
}

aclrtStream DefaultNpuApi::getCurrentNPUStream(int device_index) {
  // c10_npu::getCurrentNPUStream is wrapped in NpuStreamImpl.cpp so that
  // torch_npu's bundled acl headers (older) don't reach this TU.
  return reinterpret_cast<aclrtStream>(
      detail::getCurrentNpuStreamRaw(device_index));
}

// ---- Memory ----

aclError DefaultNpuApi::malloc(void** devPtr, size_t size) {
  return aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);
}

aclError DefaultNpuApi::free(void* devPtr) {
  return aclrtFree(devPtr);
}

aclError DefaultNpuApi::memcpy(void* dst, const void* src, size_t count,
                                aclrtMemcpyKind kind) {
  return aclrtMemcpy(dst, count, src, count, kind);
}

aclError DefaultNpuApi::memcpyAsync(void* dst, const void* src, size_t count,
                                     aclrtMemcpyKind kind, aclrtStream stream) {
  return aclrtMemcpyAsync(dst, count, src, count, kind, stream);
}

aclError DefaultNpuApi::hostAlloc(void** pHost, size_t size,
                                   uint32_t /*flags*/) {
  return aclrtMallocHost(pHost, size);
}

aclError DefaultNpuApi::hostFree(void* ptr) {
  return aclrtFreeHost(ptr);
}

// ---- Event ----

aclError DefaultNpuApi::eventCreate(aclrtEvent* event) {
  return aclrtCreateEvent(event);
}

aclError DefaultNpuApi::eventCreateWithFlags(aclrtEvent* event,
                                              uint32_t flags) {
  return aclrtCreateEventExWithFlag(event, flags);
}

aclError DefaultNpuApi::eventDestroy(aclrtEvent event) {
  return aclrtDestroyEvent(event);
}

aclError DefaultNpuApi::eventRecord(aclrtEvent event, aclrtStream stream) {
  return aclrtRecordEvent(event, stream);
}

aclError DefaultNpuApi::eventQuery(aclrtEvent event) {
  // aclrtQueryEvent is deprecated in cann 9.0; use aclrtQueryEventStatus
  // (1201 README L120). Returns 0 if completed, else 1.
  aclrtEventRecordedStatus status;
  return aclrtQueryEventStatus(event, &status);
}

// ---- Error ----

const char* DefaultNpuApi::getErrorString(aclError error) {
  // cann has no aclGetErrorString. Maintain a static errno -> string table
  // for the most common codes; fall back to aclGetRecentErrMsg (thread-local).
  static const std::unordered_map<aclError, const char*> kErrnoTable = {
      {ACL_ERROR_NONE, "ACL_ERROR_NONE"},
      {ACL_ERROR_INVALID_PARAM, "ACL_ERROR_INVALID_PARAM"},
      {ACL_ERROR_UNINITIALIZE, "ACL_ERROR_UNINITIALIZE"},
      {ACL_ERROR_REPEAT_INITIALIZE, "ACL_ERROR_REPEAT_INITIALIZE"},
      {ACL_ERROR_INVALID_FILE, "ACL_ERROR_INVALID_FILE"},
      {ACL_ERROR_WRITE_FILE, "ACL_ERROR_WRITE_FILE"},
      {ACL_ERROR_INVALID_FILE_SIZE, "ACL_ERROR_INVALID_FILE_SIZE"},
      {ACL_ERROR_PARSE_FILE, "ACL_ERROR_PARSE_FILE"},
      {ACL_ERROR_FILE_MISSING_ATTR, "ACL_ERROR_FILE_MISSING_ATTR"},
      {ACL_ERROR_FILE_ATTR_INVALID, "ACL_ERROR_FILE_ATTR_INVALID"},
      {ACL_ERROR_INVALID_DUMP_CONFIG, "ACL_ERROR_INVALID_DUMP_CONFIG"},
      {ACL_ERROR_BAD_ALLOC, "ACL_ERROR_BAD_ALLOC"},
      {ACL_ERROR_API_NOT_SUPPORT, "ACL_ERROR_API_NOT_SUPPORT"},
      {ACL_ERROR_INVALID_DEVICE, "ACL_ERROR_INVALID_DEVICE"},
      {ACL_ERROR_MEMORY_ADDRESS_UNALIGNED,
       "ACL_ERROR_MEMORY_ADDRESS_UNALIGNED"},
      {ACL_ERROR_RESOURCE_NOT_MATCH, "ACL_ERROR_RESOURCE_NOT_MATCH"},
      {ACL_ERROR_INVALID_RESOURCE_HANDLE, "ACL_ERROR_INVALID_RESOURCE_HANDLE"},
      {ACL_ERROR_FEATURE_UNSUPPORTED, "ACL_ERROR_FEATURE_UNSUPPORTED"},
      {ACL_ERROR_PROFILING_FAILURE, "ACL_ERROR_PROFILING_FAILURE"},
  };
  auto it = kErrnoTable.find(error);
  if (it != kErrnoTable.end()) return it->second;
  // Fall back to thread-local recent error message (cann standard).
  return aclGetRecentErrMsg();
}

}  // namespace torch::comms
