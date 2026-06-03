// Copyright (c) 2026 Shan Shan.
//
// TorchCommHCCL: torchcomms HCCL backend for Ascend NPU.
// Backend main class: implements TorchCommBackend by forwarding all
// collectives through HcclApi to libhccl, with NpuApi/NpuGuard handling
// device/stream interaction with torch_npu.

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <ATen/ATen.h>

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchCommBatch.hpp"
#include "comms/torchcomms/TorchCommWindow.hpp"
#include "comms/torchcomms/device/npu/NpuApi.hpp"
#include "comms/torchcomms/hccl/HcclApi.hpp"
#include "comms/torchcomms/hccl/HixlApi.hpp"

namespace torch::comms {

class TorchCommHCCL : public TorchCommBackend,
                      public std::enable_shared_from_this<TorchCommHCCL> {
 public:
  static constexpr std::string_view kBackendName = "hccl";

  TorchCommHCCL();
  ~TorchCommHCCL() override;

  TorchCommHCCL(const TorchCommHCCL&) = delete;
  TorchCommHCCL& operator=(const TorchCommHCCL&) = delete;

  // ---- lifecycle ----
  void init(at::Device device, const std::string& name,
            const CommOptions& options) override;
  void finalize() override;

  int getRank() const override;
  int getSize() const override;
  std::string_view getBackendName() const override { return kBackendName; }
  std::string_view getCommName() const override;
  const CommOptions& getOptions() const override { return options_; }
  const at::Device& getDevice() const override { return device_; }

  // ---- collectives ----
  c10::intrusive_ptr<TorchWork> send(const at::Tensor&, int, bool,
                                      const SendOptions&) override;
  c10::intrusive_ptr<TorchWork> recv(at::Tensor&, int, bool,
                                      const RecvOptions&) override;
  c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>&, bool,
      const BatchP2POptions&) override;
  c10::intrusive_ptr<TorchWork> broadcast(at::Tensor&, int, bool,
                                           const BroadcastOptions&) override;
  c10::intrusive_ptr<TorchWork> all_reduce(at::Tensor&, const ReduceOp&, bool,
                                            const AllReduceOptions&) override;
  c10::intrusive_ptr<TorchWork> reduce(const at::Tensor&, int, const ReduceOp&,
                                        bool, const ReduceOptions&) override;
  c10::intrusive_ptr<TorchWork> all_gather(const std::vector<at::Tensor>&,
                                            const at::Tensor&, bool,
                                            const AllGatherOptions&) override;
  c10::intrusive_ptr<TorchWork> all_gather_v(const std::vector<at::Tensor>&,
                                              const at::Tensor&, bool,
                                              const AllGatherOptions&) override;
  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor&, const at::Tensor&, bool,
      const AllGatherSingleOptions&) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor&, const std::vector<at::Tensor>&, const ReduceOp&, bool,
      const ReduceScatterOptions&) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor&, const std::vector<at::Tensor>&, const ReduceOp&, bool,
      const ReduceScatterOptions&) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor&, const at::Tensor&, const ReduceOp&, bool,
      const ReduceScatterSingleOptions&) override;
  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor&, const at::Tensor&, bool,
      const AllToAllSingleOptions&) override;
  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor&, const at::Tensor&, const std::vector<uint64_t>&,
      const std::vector<uint64_t>&, bool,
      const AllToAllvSingleOptions&) override;
  c10::intrusive_ptr<TorchWork> all_to_all(const std::vector<at::Tensor>&,
                                            const std::vector<at::Tensor>&,
                                            bool,
                                            const AllToAllOptions&) override;
  c10::intrusive_ptr<TorchWork> barrier(bool, const BarrierOptions&) override;
  c10::intrusive_ptr<TorchWork> scatter(at::Tensor&,
                                         const std::vector<at::Tensor>&, int,
                                         bool, const ScatterOptions&) override;
  c10::intrusive_ptr<TorchWork> gather(const std::vector<at::Tensor>&,
                                        const at::Tensor&, int, bool,
                                        const GatherOptions&) override;

  std::shared_ptr<TorchCommBackend> split(const std::vector<int>& ranks,
                                           const std::string& name,
                                           const CommOptions& options) override;

  // ---- one-sided window (HIXL-backed) ----
  std::shared_ptr<TorchCommWindow> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override;

 private:
  c10::intrusive_ptr<TorchWork> makeWork(aclrtStream stream, bool async_op);
  void ensureInitialized(const char* op) const;
  // Used by split() to install a sub-comm into a freshly-constructed sibling.
  void setSubComm(HcclComm sub_comm, int rank, int size, at::Device device,
                   const std::string& name);
  // Lazy: instantiate HixlApi + Initialize + rendezvous engine strings via
  // c10d store on the first new_window() call.
  void initHixlOnce();

  std::unique_ptr<NpuApi> npu_api_;
  std::unique_ptr<HcclApi> hccl_api_;
  HcclComm hccl_comm_{nullptr};
  at::Device device_{at::kCPU};
  std::string comm_name_;
  CommOptions options_;
  int rank_{0};
  int size_{0};
  std::atomic<bool> initialized_{false};

  // HIXL one-sided state (populated on first new_window).
  std::shared_ptr<HixlApi> hixl_api_;
  std::vector<std::string> peer_engines_;
  std::once_flag hixl_init_flag_;
  int next_window_id_{0};
};

}  // namespace torch::comms
