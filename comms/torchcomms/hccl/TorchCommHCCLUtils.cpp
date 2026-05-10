// Copyright (c) 2026 Shan Shan.

#include "comms/torchcomms/hccl/TorchCommHCCLUtils.hpp"

#include <stdexcept>

namespace torch::comms {

HcclDataType torchDtypeToHccl(at::ScalarType dtype) {
  switch (dtype) {
    case at::kByte:    return HCCL_DATA_TYPE_UINT8;
    case at::kChar:    return HCCL_DATA_TYPE_INT8;
    case at::kShort:   return HCCL_DATA_TYPE_INT16;
    case at::kInt:     return HCCL_DATA_TYPE_INT32;
    case at::kLong:    return HCCL_DATA_TYPE_INT64;
    case at::kHalf:    return HCCL_DATA_TYPE_FP16;
    case at::kFloat:   return HCCL_DATA_TYPE_FP32;
    case at::kDouble:  return HCCL_DATA_TYPE_FP64;
    case at::kBFloat16: return HCCL_DATA_TYPE_BFP16;
    default:
      throw std::runtime_error(
          "torchDtypeToHccl: unsupported dtype " +
          std::string(c10::toString(dtype)));
  }
}

HcclReduceOp torchReduceOpToHccl(const ReduceOp& op) {
  switch (op.type()) {
    case ReduceOp::RedOpType::SUM:     return HCCL_REDUCE_SUM;
    case ReduceOp::RedOpType::PRODUCT: return HCCL_REDUCE_PROD;
    case ReduceOp::RedOpType::MAX:     return HCCL_REDUCE_MAX;
    case ReduceOp::RedOpType::MIN:     return HCCL_REDUCE_MIN;
    default:
      throw std::runtime_error(
          "torchReduceOpToHccl: unsupported ReduceOp type=" +
          std::to_string(static_cast<int>(op.type())));
  }
}

}  // namespace torch::comms
