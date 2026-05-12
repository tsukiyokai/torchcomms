// Copyright (c) 2026 Shan Shan.
//
// HixlApi: HIXL (Huawei Xfer Library) C++ API abstraction.
// Mirrors the HcclApi.hpp dependency-injection pattern — backend code
// goes through this interface so HIXL can be mocked in tests and so
// the std::string ↔ ge::AscendString boundary is centralised here.
//
// Verified against cann 9.0 hixl/hixl.h.

#pragma once

#include <hixl/hixl.h>
#include <hixl/hixl_types.h>

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch::comms {

#define HIXL_CHECK(hixl_api, call, err_str)                                  \
  do {                                                                       \
    hixl::Status _status = (call);                                           \
    if (_status != hixl::SUCCESS) {                                          \
      std::stringstream _ss;                                                 \
      _ss << err_str << ": hixl::Status=" << _status << " at " << __FILE__   \
          << ":" << __LINE__;                                                \
      throw std::runtime_error(_ss.str());                                   \
    }                                                                        \
  } while (0)

class HixlApi {
 public:
  virtual ~HixlApi() = default;

  // ---- lifecycle ----
  [[nodiscard]] virtual hixl::Status initialize(
      const std::string& local_engine,
      const std::map<std::string, std::string>& options = {}) = 0;
  virtual void finalize() = 0;

  // ---- memory registration ----
  [[nodiscard]] virtual hixl::Status registerMem(
      uintptr_t addr, size_t len, hixl::MemType type,
      hixl::MemHandle* handle) = 0;
  [[nodiscard]] virtual hixl::Status deregisterMem(hixl::MemHandle handle) = 0;

  // ---- connection ----
  [[nodiscard]] virtual hixl::Status connect(
      const std::string& remote_engine, int32_t timeout_ms = 1000) = 0;
  [[nodiscard]] virtual hixl::Status disconnect(
      const std::string& remote_engine, int32_t timeout_ms = 1000) = 0;

  // ---- transfer ----
  // Sync transfer: blocks until completion.
  [[nodiscard]] virtual hixl::Status transferSync(
      const std::string& remote_engine, hixl::TransferOp op,
      const std::vector<hixl::TransferOpDesc>& descs,
      int32_t timeout_ms = 1000) = 0;

  // Async transfer: returns a request handle queryable via getTransferStatus.
  [[nodiscard]] virtual hixl::Status transferAsync(
      const std::string& remote_engine, hixl::TransferOp op,
      const std::vector<hixl::TransferOpDesc>& descs,
      hixl::TransferReq* req) = 0;

  [[nodiscard]] virtual hixl::Status getTransferStatus(
      const hixl::TransferReq& req, hixl::TransferStatus* status) = 0;

  // ---- notify ----
  [[nodiscard]] virtual hixl::Status sendNotify(
      const std::string& remote_engine, const std::string& name,
      const std::string& msg, int32_t timeout_ms = 1000) = 0;

  // Drains all pending notifies and returns (engine_name, message) pairs.
  [[nodiscard]] virtual hixl::Status getNotifies(
      std::vector<std::pair<std::string, std::string>>* notifies) = 0;
};

class DefaultHixlApi : public HixlApi {
 public:
  DefaultHixlApi();
  ~DefaultHixlApi() override;

  DefaultHixlApi(const DefaultHixlApi&) = delete;
  DefaultHixlApi& operator=(const DefaultHixlApi&) = delete;

  [[nodiscard]] hixl::Status initialize(
      const std::string& local_engine,
      const std::map<std::string, std::string>& options = {}) override;
  void finalize() override;

  [[nodiscard]] hixl::Status registerMem(
      uintptr_t addr, size_t len, hixl::MemType type,
      hixl::MemHandle* handle) override;
  [[nodiscard]] hixl::Status deregisterMem(hixl::MemHandle handle) override;

  [[nodiscard]] hixl::Status connect(
      const std::string& remote_engine, int32_t timeout_ms = 1000) override;
  [[nodiscard]] hixl::Status disconnect(
      const std::string& remote_engine, int32_t timeout_ms = 1000) override;

  [[nodiscard]] hixl::Status transferSync(
      const std::string& remote_engine, hixl::TransferOp op,
      const std::vector<hixl::TransferOpDesc>& descs,
      int32_t timeout_ms = 1000) override;
  [[nodiscard]] hixl::Status transferAsync(
      const std::string& remote_engine, hixl::TransferOp op,
      const std::vector<hixl::TransferOpDesc>& descs,
      hixl::TransferReq* req) override;
  [[nodiscard]] hixl::Status getTransferStatus(
      const hixl::TransferReq& req, hixl::TransferStatus* status) override;

  [[nodiscard]] hixl::Status sendNotify(
      const std::string& remote_engine, const std::string& name,
      const std::string& msg, int32_t timeout_ms = 1000) override;
  [[nodiscard]] hixl::Status getNotifies(
      std::vector<std::pair<std::string, std::string>>* notifies) override;

 private:
  std::unique_ptr<hixl::Hixl> hixl_;
  bool initialized_{false};
};

}  // namespace torch::comms
