// Copyright (c) 2026 Shan Shan.
//
// TorchCommWindowHCCL implementation. See header for design notes.

#include "comms/torchcomms/hccl/TorchCommWindowHCCL.hpp"

#include <chrono>
#include <cstring>
#include <sched.h>
#include <thread>

#include "comms/torchcomms/TorchWork.hpp"

namespace torch::comms {

namespace {

inline std::string addrKey(const std::string& name, int rank) {
  return "hixl_win_" + name + "_addr_" + std::to_string(rank);
}

inline std::vector<uint8_t> uintptrToBytes(uintptr_t v) {
  std::vector<uint8_t> b(sizeof(v));
  std::memcpy(b.data(), &v, sizeof(v));
  return b;
}

inline uintptr_t bytesToUintptr(const std::vector<uint8_t>& b) {
  uintptr_t v = 0;
  std::memcpy(&v, b.data(), std::min(b.size(), sizeof(v)));
  return v;
}

// Parse sender rank from a NotifyDesc.name written by signal() as
// "rank_<N>". Returns -1 if the name doesn't match.
int parseSenderRank(const std::string& name) {
  static const std::string kPrefix = "rank_";
  if (name.size() <= kPrefix.size() ||
      name.compare(0, kPrefix.size(), kPrefix) != 0) {
    return -1;
  }
  try {
    return std::stoi(name.substr(kPrefix.size()));
  } catch (...) {
    return -1;
  }
}

// Polled async-transfer work — wait() loops getTransferStatus until COMPLETED.
class TorchWorkHixlTransfer : public TorchWork {
 public:
  TorchWorkHixlTransfer(std::shared_ptr<HixlApi> api, hixl::TransferReq req)
      : api_(std::move(api)), req_(req) {}
  void wait() override {
    if (waited_.exchange(true)) return;
    while (true) {
      hixl::TransferStatus s;
      auto ret = api_->getTransferStatus(req_, &s);
      if (ret != hixl::SUCCESS) {
        throw std::runtime_error(
            "TorchWorkHixlTransfer::wait: getTransferStatus failed, ret=" +
            std::to_string(ret));
      }
      if (s == hixl::TransferStatus::COMPLETED) break;
      if (s == hixl::TransferStatus::FAILED ||
          s == hixl::TransferStatus::TIMEOUT) {
        throw std::runtime_error(
            "TorchWorkHixlTransfer::wait: transfer ended in status " +
            std::to_string(static_cast<int>(s)));
      }
      sched_yield();
    }
    setStatus(WorkStatus::COMPLETED);
  }

 private:
  std::shared_ptr<HixlApi> api_;
  hixl::TransferReq req_;
  std::atomic<bool> waited_{false};
};

// Already-completed work (used for sync transferSync + signal).
class TorchWorkHixlCompleted : public TorchWork {
 public:
  TorchWorkHixlCompleted() { setStatus(WorkStatus::COMPLETED); }
  void wait() override {}
};

}  // namespace

TorchCommWindowHCCL::TorchCommWindowHCCL(
    std::shared_ptr<HixlApi> hixl_api, int rank, int world_size,
    std::vector<std::string> peer_engines,
    c10::intrusive_ptr<c10d::Store> store, std::string name)
    : hixl_api_(std::move(hixl_api)),
      rank_(rank),
      world_size_(world_size),
      peer_engines_(std::move(peer_engines)),
      store_(std::move(store)),
      name_(std::move(name)),
      peer_addrs_(world_size, 0),
      connected_(world_size, false) {}

TorchCommWindowHCCL::~TorchCommWindowHCCL() {
  if (local_mem_handle_) {
    try {
      tensor_deregister();
    } catch (...) {
      // Best-effort in destructor.
    }
  }
}

void TorchCommWindowHCCL::tensor_register(const at::Tensor& tensor,
                                            bool owning) {
  if (owning) buf_tensor_ = tensor;
  buf_dtype_ = tensor.scalar_type();
  buf_device_ = tensor.device();
  buf_shape_ = tensor.sizes().vec();
  win_size_ = tensor.numel() * tensor.element_size();

  local_addr_ = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  hixl::MemType mt = tensor.device().is_cpu() ? hixl::MEM_HOST
                                                : hixl::MEM_DEVICE;
  HIXL_CHECK(hixl_api_,
             hixl_api_->registerMem(local_addr_, win_size_, mt,
                                      &local_mem_handle_),
             "TorchCommWindowHCCL::tensor_register: registerMem");

  // Broadcast local addr, gather all peer addrs via c10d::Store.
  store_->set(addrKey(name_, rank_), uintptrToBytes(local_addr_));
  std::vector<std::string> peer_keys;
  peer_keys.reserve(world_size_ - 1);
  for (int i = 0; i < world_size_; ++i) {
    if (i != rank_) peer_keys.push_back(addrKey(name_, i));
  }
  if (!peer_keys.empty()) {
    store_->wait(peer_keys, std::chrono::milliseconds(60000));
  }
  for (int i = 0; i < world_size_; ++i) {
    if (i == rank_) {
      peer_addrs_[i] = local_addr_;
      continue;
    }
    auto bytes = store_->get(addrKey(name_, i));
    peer_addrs_[i] = bytesToUintptr(bytes);
  }

  // Connect to every peer (HIXL Connect is idempotent across reuse).
  // 30s timeout default — first-call setup over RDMA can be much slower
  // than the post-handshake transfer cadence.
  constexpr int32_t kConnectTimeoutMs = 30000;
  for (int i = 0; i < world_size_; ++i) {
    if (i == rank_) continue;
    HIXL_CHECK(hixl_api_,
               hixl_api_->connect(peer_engines_[i], kConnectTimeoutMs),
               "TorchCommWindowHCCL::tensor_register: connect");
    connected_[i] = true;
  }
}

void TorchCommWindowHCCL::tensor_deregister() {
  for (int i = 0; i < world_size_; ++i) {
    if (connected_[i]) {
      (void)hixl_api_->disconnect(peer_engines_[i]);
      connected_[i] = false;
    }
  }
  if (local_mem_handle_) {
    (void)hixl_api_->deregisterMem(local_mem_handle_);
    local_mem_handle_ = nullptr;
  }
  buf_tensor_.reset();
  win_size_ = 0;
}

std::shared_ptr<TorchCommWindow> TorchCommWindowHCCL::clone() {
  // 第一版不支持 clone — HIXL Initialize 是 per-process 唯一 server,
  // 多 window 复用 HixlApi 单例没问题，但 store key + addr rendezvous
  // 需要重新设计 (各 window 必须用不同 name)，notify dispatch 路径
  // 需要 disambiguate target window。留到后续迭代实现。
  throw std::runtime_error(
      "TorchCommWindowHCCL::clone not yet implemented");
}

c10::intrusive_ptr<TorchWork> TorchCommWindowHCCL::put(
    const at::Tensor& tensor, int dstRank, size_t targetOffsetNelems,
    bool asyncOp, const PutOptions& options) {
  if (dstRank == rank_) {
    throw std::runtime_error("TorchCommWindowHCCL::put: target=self");
  }
  if (dstRank < 0 || dstRank >= world_size_ || peer_addrs_[dstRank] == 0) {
    throw std::runtime_error("TorchCommWindowHCCL::put: peer not connected");
  }
  size_t elem_size = tensor.element_size();
  size_t bytes = tensor.numel() * elem_size;
  uintptr_t local_addr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  uintptr_t remote_addr = peer_addrs_[dstRank] + targetOffsetNelems * elem_size;
  std::vector<hixl::TransferOpDesc> descs;
  descs.push_back({local_addr, remote_addr, bytes});

  if (asyncOp) {
    hixl::TransferReq req = nullptr;
    HIXL_CHECK(hixl_api_,
               hixl_api_->transferAsync(peer_engines_[dstRank], hixl::WRITE,
                                          descs, &req),
               "TorchCommWindowHCCL::put: transferAsync");
    return c10::make_intrusive<TorchWorkHixlTransfer>(hixl_api_, req);
  }
  int32_t timeout_ms = options.timeout.count() > 0
                            ? static_cast<int32_t>(options.timeout.count())
                            : 30000;
  HIXL_CHECK(hixl_api_,
             hixl_api_->transferSync(peer_engines_[dstRank], hixl::WRITE,
                                       descs, timeout_ms),
             "TorchCommWindowHCCL::put: transferSync");
  return c10::make_intrusive<TorchWorkHixlCompleted>();
}

at::Tensor TorchCommWindowHCCL::map_remote_tensor(int /*rank*/) {
  throw std::runtime_error(
      "TorchCommWindowHCCL::map_remote_tensor: not supported by HIXL "
      "backend — use put() / signal() / wait_signal() for one-sided data "
      "movement instead.");
}

c10::intrusive_ptr<TorchWork> TorchCommWindowHCCL::signal(
    int peerRank, bool /*asyncOp*/, const SignalOptions& options) {
  if (peerRank < 0 || peerRank >= world_size_ || peerRank == rank_) {
    throw std::runtime_error("TorchCommWindowHCCL::signal: bad peerRank");
  }
  int32_t timeout_ms = options.timeout.count() > 0
                            ? static_cast<int32_t>(options.timeout.count())
                            : 1000;
  // name carries sender rank for the receiver's queue lookup; msg carries
  // an optional user-supplied tag (hints["msg"]).
  std::string name = "rank_" + std::to_string(rank_);
  std::string msg = name_;
  auto it = options.hints.find("msg");
  if (it != options.hints.end()) msg = it->second;
  HIXL_CHECK(hixl_api_,
             hixl_api_->sendNotify(peer_engines_[peerRank], name, msg,
                                     timeout_ms),
             "TorchCommWindowHCCL::signal: sendNotify");
  return c10::make_intrusive<TorchWorkHixlCompleted>();
}

void TorchCommWindowHCCL::drainNotifiesLocked() {
  std::vector<std::pair<std::string, std::string>> raw;
  (void)hixl_api_->getNotifies(&raw);
  // hixl::NotifyDesc only carries (name, notify_msg) — no implicit sender
  // engine identity. signal() encodes sender rank as name="rank_<N>"; we
  // parse it back here.
  for (auto& [name, msg] : raw) {
    int sender = parseSenderRank(name);
    if (sender < 0) continue;  // not a backend-issued signal, drop
    per_peer_inbox_[sender].push(msg);
  }
}

c10::intrusive_ptr<TorchWork> TorchCommWindowHCCL::wait_signal(
    int peerRank, bool /*asyncOp*/, const WaitSignalOptions& options) {
  if (peerRank < 0 || peerRank >= world_size_ || peerRank == rank_) {
    throw std::runtime_error("TorchCommWindowHCCL::wait_signal: bad peerRank");
  }
  auto timeout = options.timeout.count() > 0
                      ? options.timeout
                      : std::chrono::milliseconds(60000);
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(notify_mutex_);
      drainNotifiesLocked();
      auto& q = per_peer_inbox_[peerRank];
      if (!q.empty()) {
        q.pop();
        return c10::make_intrusive<TorchWorkHixlCompleted>();
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  throw std::runtime_error(
      "TorchCommWindowHCCL::wait_signal: timeout waiting for peer " +
      std::to_string(peerRank));
}

std::shared_ptr<TorchCommWindowAttr> TorchCommWindowHCCL::get_attr(
    int /*peerRank*/) {
  auto attr = std::make_shared<TorchCommWindowAttr>();
  // HIXL is message-passing; each rank has its own private address space.
  attr->accessType = TorchCommWinAccessType::WIN_ACCESS_TYPE_SEPARATE;
  return attr;
}

}  // namespace torch::comms
