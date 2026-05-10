// Copyright (c) 2026 Shan Shan.
//
// NpuGuard: RAII device-switch wrapper around c10_npu::NPUGuard.
// Uses pImpl so the header does not pull in any torch_npu headers; only
// NpuGuard.cpp #include's <torch_npu/...>.

#pragma once

#include <c10/core/Device.h>
#include <memory>

namespace torch::comms {

class NpuGuard {
 public:
  explicit NpuGuard(c10::DeviceIndex device_index);
  ~NpuGuard();

  NpuGuard(const NpuGuard&) = delete;
  NpuGuard& operator=(const NpuGuard&) = delete;
  NpuGuard(NpuGuard&&) = delete;
  NpuGuard& operator=(NpuGuard&&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace torch::comms
