// Copyright (c) 2026 Shan Shan.
//
// TorchCommWindowHCCL: TorchCommWindow implementation backed by HIXL
// (Huawei Xfer Library). Provides one-sided put + signal/wait_signal
// over RDMA / HCCS via libcann_hixl.so.

#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/TorchCommWindow.hpp"
#include "comms/torchcomms/hccl/HixlApi.hpp"

namespace torch::comms {

class TorchCommWindowHCCL : public TorchCommWindow {
 public:
  TorchCommWindowHCCL(std::shared_ptr<HixlApi> hixl_api, int rank,
                      int world_size,
                      std::vector<std::string> peer_engines,
                      c10::intrusive_ptr<c10d::Store> store,
                      std::string name);
  ~TorchCommWindowHCCL() override;

  // ---- collective lifecycle (rendezvous via c10d::Store) ----
  void tensor_register(const at::Tensor& tensor, bool owning = true) override;
  void tensor_deregister() override;
  std::shared_ptr<TorchCommWindow> clone() override;

  // ---- one-sided ops ----
  c10::intrusive_ptr<TorchWork> put(const at::Tensor& tensor, int dstRank,
                                     size_t targetOffsetNelems, bool asyncOp,
                                     const PutOptions& options = {}) override;

  // HIXL is message-passing; no in-process view of remote device memory.
  at::Tensor map_remote_tensor(int rank) override;

  c10::intrusive_ptr<TorchWork> signal(
      int peerRank, bool asyncOp,
      const SignalOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> wait_signal(
      int peerRank, bool asyncOp,
      const WaitSignalOptions& options = {}) override;

  std::shared_ptr<TorchCommWindowAttr> get_attr(int peerRank) override;

 private:
  std::shared_ptr<HixlApi> hixl_api_;
  int rank_;
  int world_size_;
  // Index = rank, value = HIXL engine string ("ip:port"). peer_engines_[rank_]
  // is this rank's own engine string (used as Initialize argument upstream).
  std::vector<std::string> peer_engines_;
  c10::intrusive_ptr<c10d::Store> store_;
  std::string name_;

  hixl::MemHandle local_mem_handle_{nullptr};
  uintptr_t local_addr_{0};
  std::vector<uintptr_t> peer_addrs_;
  std::vector<bool> connected_;

  // GetNotifies is destructive (drains all pending). Single drainer +
  // per-peer queue, with both protected by notify_mutex_. wait_signal
  // calls drain lazily on each iteration.
  std::mutex notify_mutex_;
  std::map<int, std::queue<std::string>> per_peer_inbox_;

  // Drain HIXL notify buffer into per_peer_inbox_; caller holds mutex.
  void drainNotifiesLocked();
};

}  // namespace torch::comms
