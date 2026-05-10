// Copyright (c) 2026 Shan Shan.
//
// HcclApi: HCCL C API abstraction layer.
// Mirrors the dependency-injection pattern of NcclApi.hpp, allowing
// mockable testing and a single override point for pta upgrades.
//
// API names verified against cann 9.0 headers.
// Total: 14 collectives + 6 communicator-management ops = 20 virtual methods.

#pragma once

#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_types.h>
#include <acl/acl_rt.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace torch::comms {

#define HCCL_CHECK(hccl_api, call, err_str)                                \
  do {                                                                     \
    HcclResult _status = (call);                                           \
    if (_status != HCCL_SUCCESS) {                                         \
      std::stringstream _ss;                                               \
      _ss << err_str << ": HcclResult=" << _status << " at " << __FILE__   \
          << ":" << __LINE__;                                              \
      throw std::runtime_error(_ss.str());                                 \
    }                                                                      \
  } while (0)

class HcclApi {
 public:
  virtual ~HcclApi() = default;

  // ---- Communicator (6) ----
  [[nodiscard]] virtual HcclResult getRootInfo(HcclRootInfo* rootInfo) = 0;
  [[nodiscard]] virtual HcclResult commInitRootInfo(
      uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank,
      HcclComm* comm) = 0;
  [[nodiscard]] virtual HcclResult commDestroy(HcclComm comm) = 0;
  [[nodiscard]] virtual HcclResult getRankId(HcclComm comm, uint32_t* rank) = 0;
  [[nodiscard]] virtual HcclResult getRankSize(HcclComm comm,
                                                uint32_t* rankSize) = 0;
  [[nodiscard]] virtual HcclResult getCommAsyncError(
      HcclComm comm, HcclResult* asyncError) = 0;
  // Sub-communicator (split) — cann 9.0 weak symbol. Backend must check
  // result for ACL_ERROR_NONE-or-equivalent and surface a clear error if
  // the cann build doesn't ship the symbol.
  [[nodiscard]] virtual HcclResult createSubCommConfig(
      HcclComm* parent, uint32_t rankNum, uint32_t* rankIds,
      uint64_t subCommId, uint32_t subCommRankId,
      HcclCommConfig* config, HcclComm* subComm) = 0;

  // ---- Collectives (14) ----
  [[nodiscard]] virtual HcclResult allReduce(void* sendBuf, void* recvBuf,
                                              uint64_t count,
                                              HcclDataType dataType,
                                              HcclReduceOp op, HcclComm comm,
                                              aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult broadcast(void* buf, uint64_t count,
                                              HcclDataType dataType,
                                              uint32_t root, HcclComm comm,
                                              aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult reduce(void* sendBuf, void* recvBuf,
                                           uint64_t count,
                                           HcclDataType dataType,
                                           HcclReduceOp op, uint32_t root,
                                           HcclComm comm,
                                           aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult allGather(void* sendBuf, void* recvBuf,
                                              uint64_t sendCount,
                                              HcclDataType dataType,
                                              HcclComm comm,
                                              aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult allGatherV(void* sendBuf,
                                               uint64_t sendCount,
                                               void* recvBuf,
                                               const void* recvCounts,
                                               const void* rdispls,
                                               HcclDataType dataType,
                                               HcclComm comm,
                                               aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult reduceScatter(void* sendBuf, void* recvBuf,
                                                  uint64_t recvCount,
                                                  HcclDataType dataType,
                                                  HcclReduceOp op,
                                                  HcclComm comm,
                                                  aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult reduceScatterV(void* sendBuf,
                                                   const void* sendCounts,
                                                   const void* sendDispls,
                                                   void* recvBuf,
                                                   uint64_t recvCount,
                                                   HcclDataType dataType,
                                                   HcclReduceOp op,
                                                   HcclComm comm,
                                                   aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult scatter(void* sendBuf, void* recvBuf,
                                            uint64_t recvCount,
                                            HcclDataType dataType,
                                            uint32_t root, HcclComm comm,
                                            aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult send(void* sendBuf, uint64_t count,
                                         HcclDataType dataType,
                                         uint32_t destRank, HcclComm comm,
                                         aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult recv(void* recvBuf, uint64_t count,
                                         HcclDataType dataType,
                                         uint32_t srcRank, HcclComm comm,
                                         aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult alltoAll(const void* sendBuf,
                                             uint64_t sendCount,
                                             HcclDataType sendType,
                                             void* recvBuf,
                                             uint64_t recvCount,
                                             HcclDataType recvType,
                                             HcclComm comm,
                                             aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult alltoAllV(const void* sendBuf,
                                              const void* sendCounts,
                                              const void* sdispls,
                                              HcclDataType sendType,
                                              void* recvBuf,
                                              const void* recvCounts,
                                              const void* rdispls,
                                              HcclDataType recvType,
                                              HcclComm comm,
                                              aclrtStream stream) = 0;
  [[nodiscard]] virtual HcclResult batchSendRecv(HcclSendRecvItem* items,
                                                  uint32_t itemNum,
                                                  HcclComm comm,
                                                  aclrtStream stream) = 0;

  // ---- Special: gather emulation ----
  // HCCL has no native HcclGather; emulate via allGather + select root.
  // Default impl maps this to allGather + memcpy.
  [[nodiscard]] virtual HcclResult gather(void* sendBuf, void* recvBuf,
                                           uint64_t sendCount,
                                           HcclDataType dataType,
                                           uint32_t root, HcclComm comm,
                                           aclrtStream stream) = 0;
};

class DefaultHcclApi : public HcclApi {
 public:
  ~DefaultHcclApi() override = default;

  [[nodiscard]] HcclResult getRootInfo(HcclRootInfo* rootInfo) override;
  [[nodiscard]] HcclResult commInitRootInfo(uint32_t nRanks,
                                             const HcclRootInfo* rootInfo,
                                             uint32_t rank,
                                             HcclComm* comm) override;
  [[nodiscard]] HcclResult commDestroy(HcclComm comm) override;
  [[nodiscard]] HcclResult getRankId(HcclComm comm, uint32_t* rank) override;
  [[nodiscard]] HcclResult getRankSize(HcclComm comm,
                                        uint32_t* rankSize) override;
  [[nodiscard]] HcclResult getCommAsyncError(HcclComm comm,
                                              HcclResult* asyncError) override;
  [[nodiscard]] HcclResult createSubCommConfig(
      HcclComm* parent, uint32_t rankNum, uint32_t* rankIds,
      uint64_t subCommId, uint32_t subCommRankId,
      HcclCommConfig* config, HcclComm* subComm) override;

  [[nodiscard]] HcclResult allReduce(void*, void*, uint64_t, HcclDataType,
                                      HcclReduceOp, HcclComm,
                                      aclrtStream) override;
  [[nodiscard]] HcclResult broadcast(void*, uint64_t, HcclDataType, uint32_t,
                                      HcclComm, aclrtStream) override;
  [[nodiscard]] HcclResult reduce(void*, void*, uint64_t, HcclDataType,
                                   HcclReduceOp, uint32_t, HcclComm,
                                   aclrtStream) override;
  [[nodiscard]] HcclResult allGather(void*, void*, uint64_t, HcclDataType,
                                      HcclComm, aclrtStream) override;
  [[nodiscard]] HcclResult allGatherV(void*, uint64_t, void*, const void*,
                                       const void*, HcclDataType, HcclComm,
                                       aclrtStream) override;
  [[nodiscard]] HcclResult reduceScatter(void*, void*, uint64_t, HcclDataType,
                                          HcclReduceOp, HcclComm,
                                          aclrtStream) override;
  [[nodiscard]] HcclResult reduceScatterV(void*, const void*, const void*,
                                           void*, uint64_t, HcclDataType,
                                           HcclReduceOp, HcclComm,
                                           aclrtStream) override;
  [[nodiscard]] HcclResult scatter(void*, void*, uint64_t, HcclDataType,
                                    uint32_t, HcclComm, aclrtStream) override;
  [[nodiscard]] HcclResult send(void*, uint64_t, HcclDataType, uint32_t,
                                 HcclComm, aclrtStream) override;
  [[nodiscard]] HcclResult recv(void*, uint64_t, HcclDataType, uint32_t,
                                 HcclComm, aclrtStream) override;
  [[nodiscard]] HcclResult alltoAll(const void*, uint64_t, HcclDataType, void*,
                                     uint64_t, HcclDataType, HcclComm,
                                     aclrtStream) override;
  [[nodiscard]] HcclResult alltoAllV(const void*, const void*, const void*,
                                      HcclDataType, void*, const void*,
                                      const void*, HcclDataType, HcclComm,
                                      aclrtStream) override;
  [[nodiscard]] HcclResult batchSendRecv(HcclSendRecvItem*, uint32_t, HcclComm,
                                          aclrtStream) override;
  [[nodiscard]] HcclResult gather(void*, void*, uint64_t, HcclDataType,
                                   uint32_t, HcclComm, aclrtStream) override;
};

}  // namespace torch::comms
