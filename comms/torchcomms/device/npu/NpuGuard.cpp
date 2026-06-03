// Copyright (c) 2026 Shan Shan.
//
// NpuGuard implementation. One of two translation units allowed to
// #include <torch_npu/...> (the other is NpuStreamImpl.cpp).

#include "comms/torchcomms/device/npu/NpuGuard.hpp"

#include <torch_npu/csrc/core/npu/NPUGuard.h>

namespace torch::comms {

struct NpuGuard::Impl {
  c10_npu::NPUGuard guard;
  explicit Impl(c10::DeviceIndex idx) : guard(idx) {}
};

NpuGuard::NpuGuard(c10::DeviceIndex device_index)
    : impl_(std::make_unique<Impl>(device_index)) {}

NpuGuard::~NpuGuard() = default;

}  // namespace torch::comms
