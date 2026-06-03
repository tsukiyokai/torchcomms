// Copyright (c) 2026 Shan Shan.
//
// HcclApi default implementation: thin forwards to HCCL C API.
// API names verified against cann 9.0 hccl.h / hccl_comm.h
// (see evidence/cann90_api_listing.md).

#include "comms/torchcomms/hccl/HcclApi.hpp"

#include <cstring>

namespace torch::comms {

// ---- Communicator ----

HcclResult DefaultHcclApi::getRootInfo(HcclRootInfo* rootInfo) {
  return HcclGetRootInfo(rootInfo);
}

HcclResult DefaultHcclApi::commInitRootInfo(uint32_t nRanks,
                                             const HcclRootInfo* rootInfo,
                                             uint32_t rank, HcclComm* comm) {
  return HcclCommInitRootInfo(nRanks, rootInfo, rank, comm);
}

HcclResult DefaultHcclApi::commDestroy(HcclComm comm) {
  return HcclCommDestroy(comm);
}

HcclResult DefaultHcclApi::getRankId(HcclComm comm, uint32_t* rank) {
  return HcclGetRankId(comm, rank);
}

HcclResult DefaultHcclApi::getRankSize(HcclComm comm, uint32_t* rankSize) {
  return HcclGetRankSize(comm, rankSize);
}

HcclResult DefaultHcclApi::getCommAsyncError(HcclComm comm,
                                              HcclResult* asyncError) {
  return HcclGetCommAsyncError(comm, asyncError);
}

HcclResult DefaultHcclApi::createSubCommConfig(
    HcclComm* parent, uint32_t rankNum, uint32_t* rankIds,
    uint64_t subCommId, uint32_t subCommRankId,
    HcclCommConfig* config, HcclComm* subComm) {
  return HcclCreateSubCommConfig(parent, rankNum, rankIds, subCommId,
                                  subCommRankId, config, subComm);
}

// ---- Collectives ----

HcclResult DefaultHcclApi::allReduce(void* sendBuf, void* recvBuf,
                                      uint64_t count, HcclDataType dataType,
                                      HcclReduceOp op, HcclComm comm,
                                      aclrtStream stream) {
  return HcclAllReduce(sendBuf, recvBuf, count, dataType, op, comm, stream);
}

HcclResult DefaultHcclApi::broadcast(void* buf, uint64_t count,
                                      HcclDataType dataType, uint32_t root,
                                      HcclComm comm, aclrtStream stream) {
  return HcclBroadcast(buf, count, dataType, root, comm, stream);
}

HcclResult DefaultHcclApi::reduce(void* sendBuf, void* recvBuf, uint64_t count,
                                   HcclDataType dataType, HcclReduceOp op,
                                   uint32_t root, HcclComm comm,
                                   aclrtStream stream) {
  return HcclReduce(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
}

HcclResult DefaultHcclApi::allGather(void* sendBuf, void* recvBuf,
                                      uint64_t sendCount,
                                      HcclDataType dataType, HcclComm comm,
                                      aclrtStream stream) {
  return HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

HcclResult DefaultHcclApi::allGatherV(void* sendBuf, uint64_t sendCount,
                                       void* recvBuf, const void* recvCounts,
                                       const void* rdispls,
                                       HcclDataType dataType, HcclComm comm,
                                       aclrtStream stream) {
  return HcclAllGatherV(sendBuf, sendCount, recvBuf, recvCounts, rdispls,
                        dataType, comm, stream);
}

HcclResult DefaultHcclApi::reduceScatter(void* sendBuf, void* recvBuf,
                                          uint64_t recvCount,
                                          HcclDataType dataType,
                                          HcclReduceOp op, HcclComm comm,
                                          aclrtStream stream) {
  return HcclReduceScatter(sendBuf, recvBuf, recvCount, dataType, op, comm,
                           stream);
}

HcclResult DefaultHcclApi::reduceScatterV(void* sendBuf, const void* sendCounts,
                                           const void* sendDispls,
                                           void* recvBuf, uint64_t recvCount,
                                           HcclDataType dataType,
                                           HcclReduceOp op, HcclComm comm,
                                           aclrtStream stream) {
  return HcclReduceScatterV(sendBuf, sendCounts, sendDispls, recvBuf,
                            recvCount, dataType, op, comm, stream);
}

HcclResult DefaultHcclApi::scatter(void* sendBuf, void* recvBuf,
                                    uint64_t recvCount, HcclDataType dataType,
                                    uint32_t root, HcclComm comm,
                                    aclrtStream stream) {
  return HcclScatter(sendBuf, recvBuf, recvCount, dataType, root, comm,
                     stream);
}

HcclResult DefaultHcclApi::send(void* sendBuf, uint64_t count,
                                 HcclDataType dataType, uint32_t destRank,
                                 HcclComm comm, aclrtStream stream) {
  return HcclSend(sendBuf, count, dataType, destRank, comm, stream);
}

HcclResult DefaultHcclApi::recv(void* recvBuf, uint64_t count,
                                 HcclDataType dataType, uint32_t srcRank,
                                 HcclComm comm, aclrtStream stream) {
  return HcclRecv(recvBuf, count, dataType, srcRank, comm, stream);
}

HcclResult DefaultHcclApi::alltoAll(const void* sendBuf, uint64_t sendCount,
                                     HcclDataType sendType, void* recvBuf,
                                     uint64_t recvCount,
                                     HcclDataType recvType, HcclComm comm,
                                     aclrtStream stream) {
  return HcclAlltoAll(sendBuf, sendCount, sendType, recvBuf, recvCount,
                      recvType, comm, stream);
}

HcclResult DefaultHcclApi::alltoAllV(const void* sendBuf,
                                      const void* sendCounts,
                                      const void* sdispls,
                                      HcclDataType sendType, void* recvBuf,
                                      const void* recvCounts,
                                      const void* rdispls,
                                      HcclDataType recvType, HcclComm comm,
                                      aclrtStream stream) {
  return HcclAlltoAllV(sendBuf, sendCounts, sdispls, sendType, recvBuf,
                       recvCounts, rdispls, recvType, comm, stream);
}

HcclResult DefaultHcclApi::batchSendRecv(HcclSendRecvItem* items,
                                          uint32_t itemNum, HcclComm comm,
                                          aclrtStream stream) {
  return HcclBatchSendRecv(items, itemNum, comm, stream);
}

// ---- gather emulation (HCCL has no native HcclGather) ----
//
// Strategy: every rank does an allGather; the root then memcpy's its full
// recv buffer's slices into output. Non-root ranks discard the data.
// Currently allocates a temporary allGather output device buffer per call;
// a per-comm scratch buffer would avoid the alloc churn.

HcclResult DefaultHcclApi::gather(void* sendBuf, void* recvBuf,
                                   uint64_t sendCount, HcclDataType dataType,
                                   uint32_t root, HcclComm comm,
                                   aclrtStream stream) {
  // For now route to allGather and let the caller select root chunk.
  // The TorchCommHCCL::gather wrapper handles non-root buffer alloc.
  return HcclAllGather(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

}  // namespace torch::comms
