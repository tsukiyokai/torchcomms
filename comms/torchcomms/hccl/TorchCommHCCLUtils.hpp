// Copyright (c) 2026 Shan Shan.
//
// Type/op conversion helpers between torchcomms (PyTorch) and HCCL.

#pragma once

#include <ATen/ATen.h>
#include <hccl/hccl_types.h>

#include "comms/torchcomms/TorchCommTypes.hpp"

namespace torch::comms {

HcclDataType torchDtypeToHccl(at::ScalarType dtype);
HcclReduceOp torchReduceOpToHccl(const ReduceOp& op);

}  // namespace torch::comms
