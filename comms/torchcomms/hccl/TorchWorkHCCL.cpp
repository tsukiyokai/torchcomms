// Copyright (c) 2026 Shan Shan.

#include "comms/torchcomms/hccl/TorchWorkHCCL.hpp"

#include <glog/logging.h>

namespace torch::comms {

TorchWorkHCCL::TorchWorkHCCL(std::shared_ptr<NpuApi> npu_api,
                              aclrtEvent event)
    : npu_api_(std::move(npu_api)), event_(event) {}

TorchWorkHCCL::~TorchWorkHCCL() {
  if (event_ != nullptr) {
    NPU_CHECK_IGNORE(npu_api_, npu_api_->eventDestroy(event_),
                     "TorchWorkHCCL::~TorchWorkHCCL: eventDestroy failed");
    event_ = nullptr;
  }
}

void TorchWorkHCCL::wait() {
  if (waited_.exchange(true)) return;
  if (event_ == nullptr) {
    setStatus(WorkStatus::COMPLETED);
    return;
  }
  // Synchronize host until the recorded event completes.
  // aclrtSynchronizeEvent would be cleanest but cann 9.0 spells it
  // differently per release; query loop is portable.
  while (true) {
    aclError rc = npu_api_->eventQuery(event_);
    if (rc == ACL_ERROR_NONE) break;
    sched_yield();
  }
  setStatus(WorkStatus::COMPLETED);
}

}  // namespace torch::comms
