// Copyright (c) 2026 Shan Shan.
//
// TorchCommHCCL implementation.
// Lifecycle, collectives, p2p, and split forward through HcclApi to
// libhccl; stream / device interaction goes through NpuApi.

#include "comms/torchcomms/hccl/TorchCommHCCL.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include <torch/csrc/distributed/c10d/Store.hpp>

#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/hccl/TorchCommHCCLUtils.hpp"
#include "comms/torchcomms/hccl/TorchCommWindowHCCL.hpp"
#include "comms/torchcomms/hccl/TorchWorkHCCL.hpp"

namespace torch::comms {

namespace {

[[noreturn]] void throwNotImplemented(const char* op) {
  throw std::runtime_error(
      std::string("TorchCommHCCL::") + op + " not implemented yet");
}

}  // namespace

// PyTorch + c10d convention: collectives are stream-async by default.
// async_op=False ≠ host blocking — it means the caller gets a Work whose
// wait() is a no-op because stream order alone guarantees downstream ops
// in the same stream observe the collective's effect. Host blocking would
// only happen if the user explicitly calls a host-sync (cpu(), .item()).
//
// async_op=True: same return value but the user is communicating intent
// to overlap compute with collective; we still return a stream-bound Work.
// Future revisions may differentiate (e.g. cross-stream wait).
c10::intrusive_ptr<TorchWork> TorchCommHCCL::makeWork(aclrtStream /*stream*/,
                                                       bool /*async_op*/) {
  // event=nullptr → TorchWorkHCCL::wait() is a no-op; downstream tensor
  // consumers on the same NPU stream auto-wait via stream order.
  return c10::make_intrusive<TorchWorkHCCL>(
      std::shared_ptr<NpuApi>(npu_api_.get(), [](NpuApi*) {}), nullptr);
}

void TorchCommHCCL::ensureInitialized(const char* op) const {
  if (!initialized_.load() || hccl_comm_ == nullptr) {
    throw std::runtime_error(std::string("TorchCommHCCL::") + op +
                              ": backend not initialized");
  }
}

TorchCommHCCL::TorchCommHCCL()
    : npu_api_(std::make_unique<DefaultNpuApi>()),
      hccl_api_(std::make_unique<DefaultHcclApi>()) {}

TorchCommHCCL::~TorchCommHCCL() {
  if (initialized_.exchange(false)) {
    if (hccl_comm_ != nullptr) {
      // Best-effort destroy; ignore errors in destructor.
      (void)hccl_api_->commDestroy(hccl_comm_);
      hccl_comm_ = nullptr;
    }
  }
}

void TorchCommHCCL::init(at::Device device, const std::string& name,
                          const CommOptions& options) {
  device_ = device;
  comm_name_ = name;
  options_ = options;

  rank_ = 0;
  size_ = 1;
  auto rankIt = options.hints.find("rank");
  if (rankIt != options.hints.end()) rank_ = std::stoi(rankIt->second);
  auto sizeIt = options.hints.find("world_size");
  if (sizeIt != options.hints.end()) size_ = std::stoi(sizeIt->second);

  // Set ACL device before any HCCL init.
  NPU_CHECK(npu_api_, npu_api_->setDevice(device.index()),
            "TorchCommHCCL::init: setDevice failed");

  // Store-based rendezvous covers both single- and multi-rank.
  HcclRootInfo root_info;
  if (size_ == 1) {
    HCCL_CHECK(hccl_api_, hccl_api_->getRootInfo(&root_info),
               "TorchCommHCCL::init: HcclGetRootInfo failed");
  } else {
    if (!options.store) {
      throw std::runtime_error(
          "TorchCommHCCL::init: multi-rank requires CommOptions.store. "
          "Pass store=... to torchcomms.new_comm or set up via "
          "torch.distributed.init_process_group first.");
    }
    const std::string key = "torchcomms_hccl_rootinfo_" + name;
    if (rank_ == 0) {
      HCCL_CHECK(hccl_api_, hccl_api_->getRootInfo(&root_info),
                 "TorchCommHCCL::init: HcclGetRootInfo failed");
      std::vector<uint8_t> bytes(sizeof(HcclRootInfo));
      std::memcpy(bytes.data(), &root_info, sizeof(HcclRootInfo));
      options.store->set(key, bytes);
    } else {
      auto timeout = options.timeout.count() > 0
                         ? options.timeout
                         : std::chrono::milliseconds(60000);
      options.store->wait({key}, timeout);
      auto bytes = options.store->get(key);
      if (bytes.size() != sizeof(HcclRootInfo)) {
        throw std::runtime_error(
            "TorchCommHCCL::init: store rootinfo size mismatch, got " +
            std::to_string(bytes.size()) + " expected " +
            std::to_string(sizeof(HcclRootInfo)));
      }
      std::memcpy(&root_info, bytes.data(), sizeof(HcclRootInfo));
    }
  }
  HCCL_CHECK(hccl_api_,
             hccl_api_->commInitRootInfo(static_cast<uint32_t>(size_),
                                          &root_info,
                                          static_cast<uint32_t>(rank_),
                                          &hccl_comm_),
             "TorchCommHCCL::init: HcclCommInitRootInfo failed");
  initialized_.store(true);
}

void TorchCommHCCL::finalize() {
  if (initialized_.exchange(false)) {
    if (hccl_comm_ != nullptr) {
      (void)hccl_api_->commDestroy(hccl_comm_);
      hccl_comm_ = nullptr;
    }
  }
}

int TorchCommHCCL::getRank() const { return rank_; }
int TorchCommHCCL::getSize() const { return size_; }
std::string_view TorchCommHCCL::getCommName() const { return comm_name_; }

// ---- collectives ----

c10::intrusive_ptr<TorchWork> TorchCommHCCL::send(const at::Tensor& tensor,
                                                   int dst, bool async_op,
                                                   const SendOptions&) {
  ensureInitialized("send");
  HcclDataType dtype = torchDtypeToHccl(tensor.scalar_type());
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->send(const_cast<void*>(tensor.data_ptr()),
                              tensor.numel(), dtype,
                              static_cast<uint32_t>(dst), hccl_comm_, stream),
             "send");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::recv(at::Tensor& tensor, int src,
                                                   bool async_op,
                                                   const RecvOptions&) {
  ensureInitialized("recv");
  HcclDataType dtype = torchDtypeToHccl(tensor.scalar_type());
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->recv(tensor.data_ptr(), tensor.numel(), dtype,
                              static_cast<uint32_t>(src), hccl_comm_, stream),
             "recv");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops, bool async_op,
    const BatchP2POptions&) {
  ensureInitialized("batch_op_issue");
  std::vector<HcclSendRecvItem> items;
  items.reserve(ops.size());
  for (const auto& op : ops) {
    HcclSendRecvItem it{};
    it.sendRecvType =
        op.type == BatchSendRecv::P2POp::OpType::SEND ? HCCL_SEND : HCCL_RECV;
    it.buf = const_cast<void*>(op.tensor.data_ptr());
    it.count = op.tensor.numel();
    it.dataType = torchDtypeToHccl(op.tensor.scalar_type());
    it.remoteRank = static_cast<uint32_t>(op.peer);
    items.push_back(it);
  }
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->batchSendRecv(items.data(),
                                       static_cast<uint32_t>(items.size()),
                                       hccl_comm_, stream),
             "batch_op_issue");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::broadcast(
    at::Tensor& tensor, int root, bool async_op,
    const BroadcastOptions&) {
  ensureInitialized("broadcast");
  HcclDataType dtype = torchDtypeToHccl(tensor.scalar_type());
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->broadcast(tensor.data_ptr(), tensor.numel(), dtype,
                                   static_cast<uint32_t>(root), hccl_comm_,
                                   stream),
             "broadcast");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_reduce(
    at::Tensor& tensor, const ReduceOp& op, bool async_op,
    const AllReduceOptions& /*options*/) {
  ensureInitialized("all_reduce");
  HcclDataType dtype = torchDtypeToHccl(tensor.scalar_type());
  HcclReduceOp reduce_op = torchReduceOpToHccl(op);
  void* buf = tensor.data_ptr();
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->allReduce(buf, buf, tensor.numel(), dtype, reduce_op,
                                   hccl_comm_, stream),
             "all_reduce");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::reduce(const at::Tensor& tensor,
                                                     int root,
                                                     const ReduceOp& op,
                                                     bool async_op,
                                                     const ReduceOptions&) {
  ensureInitialized("reduce");
  HcclDataType dtype = torchDtypeToHccl(tensor.scalar_type());
  HcclReduceOp reduce_op = torchReduceOpToHccl(op);
  void* buf = const_cast<void*>(tensor.data_ptr());
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->reduce(buf, buf, tensor.numel(), dtype, reduce_op,
                                static_cast<uint32_t>(root), hccl_comm_,
                                stream),
             "reduce");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list, const at::Tensor& tensor,
    bool async_op, const AllGatherOptions&) {
  ensureInitialized("all_gather");
  if (tensor_list.empty()) {
    throw std::runtime_error("all_gather: empty tensor_list");
  }
  // c10d convention: tensor_list[i] are contiguous slices of one buffer.
  // We use tensor_list[0].data_ptr() as the base. Caller must ensure layout.
  HcclDataType dtype = torchDtypeToHccl(tensor.scalar_type());
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->allGather(const_cast<void*>(tensor.data_ptr()),
                                   tensor_list[0].data_ptr(), tensor.numel(),
                                   dtype, hccl_comm_, stream),
             "all_gather");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_gather_v(
    const std::vector<at::Tensor>& tensor_list, const at::Tensor& tensor,
    bool async_op, const AllGatherOptions& options) {
  // V variant currently routes through uniform allGather. True
  // uneven-size support via HcclAllGatherV is not implemented yet —
  // would require deriving recvCounts / displs from per-rank tensor shapes.
  return all_gather(tensor_list, tensor, async_op, options);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_gather_single(
    at::Tensor& output, const at::Tensor& input, bool async_op,
    const AllGatherSingleOptions&) {
  ensureInitialized("all_gather_single");
  HcclDataType dtype = torchDtypeToHccl(input.scalar_type());
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->allGather(const_cast<void*>(input.data_ptr()),
                                   output.data_ptr(), input.numel(), dtype,
                                   hccl_comm_, stream),
             "all_gather_single");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::reduce_scatter(
    at::Tensor& output, const std::vector<at::Tensor>& input_list,
    const ReduceOp& op, bool async_op, const ReduceScatterOptions&) {
  ensureInitialized("reduce_scatter");
  if (input_list.empty()) {
    throw std::runtime_error("reduce_scatter: empty input_list");
  }
  HcclDataType dtype = torchDtypeToHccl(output.scalar_type());
  HcclReduceOp reduce_op = torchReduceOpToHccl(op);
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->reduceScatter(input_list[0].data_ptr(),
                                       output.data_ptr(), output.numel(),
                                       dtype, reduce_op, hccl_comm_, stream),
             "reduce_scatter");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::reduce_scatter_v(
    at::Tensor& output, const std::vector<at::Tensor>& input_list,
    const ReduceOp& op, bool async_op,
    const ReduceScatterOptions& options) {
  // Route to uniform variant; true uneven HcclReduceScatterV not wired yet.
  return reduce_scatter(output, input_list, op, async_op, options);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::reduce_scatter_single(
    at::Tensor& output, const at::Tensor& input, const ReduceOp& op,
    bool async_op, const ReduceScatterSingleOptions&) {
  ensureInitialized("reduce_scatter_single");
  HcclDataType dtype = torchDtypeToHccl(input.scalar_type());
  HcclReduceOp reduce_op = torchReduceOpToHccl(op);
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->reduceScatter(const_cast<void*>(input.data_ptr()),
                                       output.data_ptr(), output.numel(),
                                       dtype, reduce_op, hccl_comm_, stream),
             "reduce_scatter_single");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_to_all_single(
    at::Tensor& output, const at::Tensor& input, bool async_op,
    const AllToAllSingleOptions&) {
  ensureInitialized("all_to_all_single");
  HcclDataType sendType = torchDtypeToHccl(input.scalar_type());
  HcclDataType recvType = torchDtypeToHccl(output.scalar_type());
  uint64_t per_rank = input.numel() / static_cast<uint64_t>(size_);
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->alltoAll(input.data_ptr(), per_rank, sendType,
                                  output.data_ptr(), per_rank, recvType,
                                  hccl_comm_, stream),
             "all_to_all_single");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_to_all_v_single(
    at::Tensor& output, const at::Tensor& input,
    const std::vector<uint64_t>& output_splits,
    const std::vector<uint64_t>& input_splits, bool async_op,
    const AllToAllvSingleOptions&) {
  ensureInitialized("all_to_all_v_single");
  HcclDataType sendType = torchDtypeToHccl(input.scalar_type());
  HcclDataType recvType = torchDtypeToHccl(output.scalar_type());

  // HcclAlltoAllV expects displacement arrays (cumulative offsets in
  // elements). Compute them from per-rank counts.
  std::vector<uint64_t> sdispls(input_splits.size());
  std::vector<uint64_t> rdispls(output_splits.size());
  for (size_t i = 1; i < input_splits.size(); ++i) {
    sdispls[i] = sdispls[i - 1] + input_splits[i - 1];
  }
  for (size_t i = 1; i < output_splits.size(); ++i) {
    rdispls[i] = rdispls[i - 1] + output_splits[i - 1];
  }

  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->alltoAllV(input.data_ptr(), input_splits.data(),
                                   sdispls.data(), sendType,
                                   output.data_ptr(), output_splits.data(),
                                   rdispls.data(), recvType, hccl_comm_,
                                   stream),
             "all_to_all_v_single");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_to_all(
    const std::vector<at::Tensor>& output_list,
    const std::vector<at::Tensor>& input_list, bool async_op,
    const AllToAllOptions&) {
  ensureInitialized("all_to_all");
  if (input_list.empty() || output_list.empty()) {
    throw std::runtime_error("all_to_all: empty list");
  }
  // c10d convention: list elements are contiguous slices.
  HcclDataType sendType = torchDtypeToHccl(input_list[0].scalar_type());
  HcclDataType recvType = torchDtypeToHccl(output_list[0].scalar_type());
  uint64_t per_rank = input_list[0].numel();
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  HCCL_CHECK(hccl_api_,
             hccl_api_->alltoAll(input_list[0].data_ptr(), per_rank, sendType,
                                  output_list[0].data_ptr(), per_rank,
                                  recvType, hccl_comm_, stream),
             "all_to_all");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::barrier(
    bool async_op, const BarrierOptions&) {
  ensureInitialized("barrier");
  // HCCL has no native HcclBarrier; use HcclAllReduce on a 1-element dummy
  // tensor (the same trick PyHcclBackend uses, and what NCCL
  // does internally for barrier).
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  void* dummy = nullptr;
  NPU_CHECK(npu_api_, npu_api_->malloc(&dummy, sizeof(float)),
            "barrier: dummy malloc");
  // Zero out so reduce result is deterministic.
  // (cann's aclrtMemset is fine; allReduce overwrites anyway.)
  HCCL_CHECK(hccl_api_,
             hccl_api_->allReduce(dummy, dummy, 1, HCCL_DATA_TYPE_FP32,
                                   HCCL_REDUCE_SUM, hccl_comm_, stream),
             "barrier(allreduce-emulate)");
  // Synchronize the stream then free; a per-comm scratch buffer would
  // remove the per-call alloc/free.
  NPU_CHECK(npu_api_, npu_api_->streamSynchronize(stream),
            "barrier: streamSynchronize");
  NPU_CHECK_IGNORE(npu_api_, npu_api_->free(dummy), "barrier: dummy free");
  // Return a no-op already-completed work (event = nullptr → wait returns).
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::scatter(
    at::Tensor& output, const std::vector<at::Tensor>& input_list, int root,
    bool async_op, const ScatterOptions&) {
  ensureInitialized("scatter");
  HcclDataType dtype = torchDtypeToHccl(output.scalar_type());
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  // Root: input_list[0] must be a single contiguous buffer of size N*K (the
  // user pre-arranged it); HcclScatter slices it. A future revision can
  // accept a true list of independent tensors by concatenating through a
  // scratch buffer.
  void* send_buf = (rank_ == root && !input_list.empty())
                        ? const_cast<void*>(input_list[0].data_ptr())
                        : nullptr;
  HCCL_CHECK(hccl_api_,
             hccl_api_->scatter(send_buf, output.data_ptr(), output.numel(),
                                 dtype, static_cast<uint32_t>(root),
                                 hccl_comm_, stream),
             "scatter");
  return makeWork(stream, async_op);
}
c10::intrusive_ptr<TorchWork> TorchCommHCCL::gather(
    const std::vector<at::Tensor>& output_list, const at::Tensor& input,
    int root, bool async_op, const GatherOptions&) {
  ensureInitialized("gather");
  // HCCL has no native gather. Emulate via allgather + select root chunks.
  // PyHcclBackend uses the same trick. Allocate a temporary
  // contiguous buffer of size N*K for the allgather output, copy back into
  // output_list slices on root.
  HcclDataType dtype = torchDtypeToHccl(input.scalar_type());
  aclrtStream stream = npu_api_->getCurrentNPUStream(device_.index());
  uint64_t per_rank = input.numel();
  size_t elem_bytes = input.element_size();
  size_t total_bytes = per_rank * elem_bytes * static_cast<uint64_t>(size_);
  void* scratch = nullptr;
  NPU_CHECK(npu_api_, npu_api_->malloc(&scratch, total_bytes),
            "gather: scratch malloc");
  HCCL_CHECK(hccl_api_,
             hccl_api_->allGather(const_cast<void*>(input.data_ptr()),
                                   scratch, per_rank, dtype, hccl_comm_,
                                   stream),
             "gather(allgather-emulate)");
  // Sync to fully populate scratch before host-side slicing.
  NPU_CHECK(npu_api_, npu_api_->streamSynchronize(stream),
            "gather: streamSynchronize");
  if (rank_ == root) {
    for (size_t i = 0; i < output_list.size() && i < static_cast<size_t>(size_);
         ++i) {
      auto* dst = output_list[i].data_ptr();
      auto* src = static_cast<uint8_t*>(scratch) + i * per_rank * elem_bytes;
      NPU_CHECK(npu_api_,
                npu_api_->memcpyAsync(dst, src, per_rank * elem_bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                "gather: chunk memcpy");
    }
  }
  NPU_CHECK_IGNORE(npu_api_, npu_api_->free(scratch), "gather: scratch free");
  return makeWork(stream, async_op);
}

std::shared_ptr<TorchCommBackend> TorchCommHCCL::split(
    const std::vector<int>& ranks, const std::string& name,
    const CommOptions& /*options*/) {
  ensureInitialized("split");
  // Find current rank's index in `ranks` (sub-comm rank id).
  int my_idx = -1;
  for (size_t i = 0; i < ranks.size(); ++i) {
    if (ranks[i] == rank_) {
      my_idx = static_cast<int>(i);
      break;
    }
  }
  if (my_idx < 0) {
    throw std::runtime_error(
        "TorchCommHCCL::split: current rank " + std::to_string(rank_) +
        " not in ranks list");
  }
  std::vector<uint32_t> rankIds(ranks.begin(), ranks.end());
  uint64_t subCommId = static_cast<uint64_t>(std::hash<std::string>{}(name));

  HcclCommConfig config;
  HcclCommConfigInit(&config);  // cann 9.0 static inline default-fill helper

  HcclComm subComm = nullptr;
  HCCL_CHECK(hccl_api_,
             hccl_api_->createSubCommConfig(
                 &hccl_comm_, static_cast<uint32_t>(ranks.size()),
                 rankIds.data(), subCommId,
                 static_cast<uint32_t>(my_idx), &config, &subComm),
             "split: HcclCreateSubCommConfig");

  auto sub = std::make_shared<TorchCommHCCL>();
  sub->setSubComm(subComm, my_idx, static_cast<int>(ranks.size()), device_,
                   name);
  return sub;
}

void TorchCommHCCL::setSubComm(HcclComm sub_comm, int rank, int size,
                                at::Device device,
                                const std::string& name) {
  hccl_comm_ = sub_comm;
  rank_ = rank;
  size_ = size;
  device_ = device;
  comm_name_ = name;
  initialized_.store(true);
}

// ---- one-sided window (HIXL) ----

void TorchCommHCCL::initHixlOnce() {
  if (!options_.store) {
    throw std::runtime_error(
        "TorchCommHCCL::new_window: HIXL needs CommOptions.store. Pass "
        "store=... to torchcomms.new_comm or set up via "
        "torch.distributed.init_process_group first.");
  }

  hixl_api_ = std::make_shared<DefaultHixlApi>();

  std::string host_ip = "127.0.0.1";
  int base_port = 50000;
  auto it = options_.hints.find("hixl_host_ip");
  if (it != options_.hints.end()) host_ip = it->second;
  it = options_.hints.find("hixl_base_port");
  if (it != options_.hints.end()) base_port = std::stoi(it->second);

  std::string self_engine =
      host_ip + ":" + std::to_string(base_port + rank_);

  // hixl::Initialize switches to its own aclrtContext (verified via debug
  // log: "Switch new aclrt ctx:0x..."). The torch_npu acl context that
  // our caller's tensors live in must be restored afterwards — otherwise
  // register/transfer of torch tensor data_ptr fails the cross-context
  // visibility check (TransferSync hangs until TIMEOUT).
  aclrtContext saved_ctx = nullptr;
  (void)aclrtGetCurrentContext(&saved_ctx);
  HIXL_CHECK(hixl_api_, hixl_api_->initialize(self_engine, {}),
             "TorchCommHCCL::initHixlOnce: hixl::Initialize");
  if (saved_ctx != nullptr) {
    (void)aclrtSetCurrentContext(saved_ctx);
  }

  // Rendezvous engine strings via c10d::Store. Each rank publishes its own
  // engine string under a per-rank key, then waits for & reads the others.
  const std::string key_prefix =
      "hccl_hixl_engine_" + comm_name_ + "_";
  auto own_key = key_prefix + std::to_string(rank_);
  std::vector<uint8_t> own_bytes(self_engine.begin(), self_engine.end());
  options_.store->set(own_key, own_bytes);

  std::vector<std::string> peer_keys;
  peer_keys.reserve(size_ - 1);
  for (int i = 0; i < size_; ++i) {
    if (i != rank_) peer_keys.push_back(key_prefix + std::to_string(i));
  }
  if (!peer_keys.empty()) {
    options_.store->wait(peer_keys, std::chrono::milliseconds(60000));
  }

  peer_engines_.resize(size_);
  for (int i = 0; i < size_; ++i) {
    if (i == rank_) {
      peer_engines_[i] = self_engine;
      continue;
    }
    auto bytes = options_.store->get(key_prefix + std::to_string(i));
    peer_engines_[i] = std::string(bytes.begin(), bytes.end());
  }

  // Listener-ready barrier: hixl::Initialize on each rank starts the
  // RaSocket server (port 16666 on phyId NPU-IP) asynchronously; Connect
  // before the peer's RaSocketListenStart completes returns TIMEOUT.
  // Demo (server_server_d2d.cpp:193) sleeps 5s and works; 2s wasn't
  // enough in the backend path here. Match demo and post a "ready" key
  // so all ranks observe each other's listeners before any Connect.
  std::this_thread::sleep_for(std::chrono::seconds(5));
  const std::string ready_prefix =
      "hccl_hixl_ready_" + comm_name_ + "_";
  options_.store->set(ready_prefix + std::to_string(rank_),
                       std::vector<uint8_t>{1});
  std::vector<std::string> ready_keys;
  ready_keys.reserve(size_);
  for (int i = 0; i < size_; ++i) {
    ready_keys.push_back(ready_prefix + std::to_string(i));
  }
  options_.store->wait(ready_keys, std::chrono::milliseconds(60000));
}

std::shared_ptr<TorchCommWindow> TorchCommHCCL::new_window(
    const std::optional<at::Tensor>& tensor) {
  ensureInitialized("new_window");
  std::call_once(hixl_init_flag_, [this]() { initHixlOnce(); });

  // Each window gets a unique name so its store-rendezvous keys (used in
  // TorchCommWindowHCCL::tensor_register for addr exchange) don't collide
  // across multiple windows on the same comm.
  std::string win_name =
      comm_name_ + "_win" + std::to_string(next_window_id_++);

  auto window = std::make_shared<TorchCommWindowHCCL>(
      hixl_api_, rank_, size_, peer_engines_, options_.store, win_name);
  if (tensor.has_value()) {
    window->tensor_register(tensor.value());
  }
  return window;
}

// ---- Backend factory registration ----
//
// Mirrors how nccl/TorchCommNCCL.cpp registers itself with TorchCommFactory
// at static init time. The name "hccl" must match BACKEND_FLAGS in setup.py.

namespace {

const bool kHcclRegistered = []() {
  TorchCommFactory::get().register_backend(
      std::string(TorchCommHCCL::kBackendName),
      []() { return std::make_shared<TorchCommHCCL>(); });
  return true;
}();

}  // namespace

}  // namespace torch::comms
