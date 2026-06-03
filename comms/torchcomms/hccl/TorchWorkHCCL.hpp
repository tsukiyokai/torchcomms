// Copyright (c) 2026 Shan Shan.
//
// TorchWorkHCCL: async work handle for HCCL collectives.
// Holds an aclrtEvent recorded after the collective is enqueued;
// wait() blocks the host until the event completes.

#pragma once

#include <atomic>
#include <memory>

#include <acl/acl_rt.h>

#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/device/npu/NpuApi.hpp"

namespace torch::comms {

class TorchWorkHCCL : public TorchWork {
 public:
  TorchWorkHCCL(std::shared_ptr<NpuApi> npu_api, aclrtEvent event);
  ~TorchWorkHCCL() override;

  TorchWorkHCCL(const TorchWorkHCCL&) = delete;
  TorchWorkHCCL& operator=(const TorchWorkHCCL&) = delete;

  void wait() override;

 private:
  std::shared_ptr<NpuApi> npu_api_;
  aclrtEvent event_{nullptr};
  std::atomic<bool> waited_{false};
};

}  // namespace torch::comms
