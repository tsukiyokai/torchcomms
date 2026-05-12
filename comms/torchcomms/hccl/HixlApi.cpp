// Copyright (c) 2026 Shan Shan.
//
// DefaultHixlApi: thin forwards to hixl::Hixl, with std::string ↔
// ge::AscendString conversion at the boundary so backend code never
// touches AscendString directly.

#include "comms/torchcomms/hccl/HixlApi.hpp"

#include "external/ge_common/ge_api_error_codes.h"

namespace torch::comms {

namespace {
// std::string → ge::AscendString is constructible from c_str.
inline hixl::AscendString toAS(const std::string& s) {
  return hixl::AscendString(s.c_str());
}
inline std::string fromAS(const hixl::AscendString& a) {
  // ge::AscendString::GetString may return nullptr for empty.
  const char* p = a.GetString();
  return p ? std::string(p) : std::string{};
}
}  // namespace

DefaultHixlApi::DefaultHixlApi()
    : hixl_(std::make_unique<hixl::Hixl>()) {}

DefaultHixlApi::~DefaultHixlApi() {
  if (initialized_ && hixl_) {
    hixl_->Finalize();
  }
}

hixl::Status DefaultHixlApi::initialize(
    const std::string& local_engine,
    const std::map<std::string, std::string>& options) {
  std::map<hixl::AscendString, hixl::AscendString> as_options;
  for (const auto& [k, v] : options) {
    as_options.emplace(toAS(k), toAS(v));
  }
  auto ret = hixl_->Initialize(toAS(local_engine), as_options);
  if (ret == hixl::SUCCESS) initialized_ = true;
  return ret;
}

void DefaultHixlApi::finalize() {
  if (initialized_ && hixl_) {
    hixl_->Finalize();
    initialized_ = false;
  }
}

hixl::Status DefaultHixlApi::registerMem(uintptr_t addr, size_t len,
                                         hixl::MemType type,
                                         hixl::MemHandle* handle) {
  hixl::MemDesc desc{};
  desc.addr = addr;
  desc.len = len;
  return hixl_->RegisterMem(desc, type, *handle);
}

hixl::Status DefaultHixlApi::deregisterMem(hixl::MemHandle handle) {
  return hixl_->DeregisterMem(handle);
}

hixl::Status DefaultHixlApi::connect(const std::string& remote_engine,
                                     int32_t timeout_ms) {
  return hixl_->Connect(toAS(remote_engine), timeout_ms);
}

hixl::Status DefaultHixlApi::disconnect(const std::string& remote_engine,
                                        int32_t timeout_ms) {
  return hixl_->Disconnect(toAS(remote_engine), timeout_ms);
}

hixl::Status DefaultHixlApi::transferSync(
    const std::string& remote_engine, hixl::TransferOp op,
    const std::vector<hixl::TransferOpDesc>& descs, int32_t timeout_ms) {
  return hixl_->TransferSync(toAS(remote_engine), op, descs, timeout_ms);
}

hixl::Status DefaultHixlApi::transferAsync(
    const std::string& remote_engine, hixl::TransferOp op,
    const std::vector<hixl::TransferOpDesc>& descs, hixl::TransferReq* req) {
  hixl::TransferArgs args{};
  return hixl_->TransferAsync(toAS(remote_engine), op, descs, args, *req);
}

hixl::Status DefaultHixlApi::getTransferStatus(const hixl::TransferReq& req,
                                                hixl::TransferStatus* status) {
  return hixl_->GetTransferStatus(req, *status);
}

hixl::Status DefaultHixlApi::sendNotify(const std::string& remote_engine,
                                         const std::string& name,
                                         const std::string& msg,
                                         int32_t timeout_ms) {
  hixl::NotifyDesc nd;
  nd.name = toAS(name);
  nd.notify_msg = toAS(msg);
  return hixl_->SendNotify(toAS(remote_engine), nd, timeout_ms);
}

hixl::Status DefaultHixlApi::getNotifies(
    std::vector<std::pair<std::string, std::string>>* notifies) {
  std::vector<hixl::NotifyDesc> raw;
  auto ret = hixl_->GetNotifies(raw);
  if (ret != hixl::SUCCESS) return ret;
  notifies->clear();
  notifies->reserve(raw.size());
  for (const auto& n : raw) {
    notifies->emplace_back(fromAS(n.name), fromAS(n.notify_msg));
  }
  return hixl::SUCCESS;
}

}  // namespace torch::comms
