#include "Emitter.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace mlir::maps {

static std::string getPaddedPhysicalTileId(const MagiaApplicationDescription &application,
                                           int64_t hartid) {
  std::string id = std::to_string(hartid);
  uint64_t physicalTileCount =
      static_cast<uint64_t>(application.meshWidth) * application.meshHeight;
  size_t width = std::max<size_t>(
      2, std::to_string(physicalTileCount - 1).size());
  if (id.size() < width)
    id.insert(0, width - id.size(), '0');
  return id;
}

std::string getApplicationTileFilename(const MagiaApplicationDescription &application,
                                       int64_t hartid) {
  return "tile_" + getPaddedPhysicalTileId(application, hartid) + ".c";
}

static void emitGeneratedPrelude(const MagiaApplicationDescription &application,
                                 llvm::raw_ostream &os) {
  os << "#include <stdint.h>\n\n";
  os << "#include \"tile.h\"\n";
  os << "#include \"utils/maps_operations.h\"\n";
  os << "#include \"utils/maps_utils_v2.h\"\n\n";
  os << "#define MAPS_APPLICATION_NUM_TOKENS "
     << application.numTokens << "u\n";
  os << "#define MAPS_APPLICATION_NUM_TOKEN_SLOTS "
     << application.numTokenSlots << "u\n";
  os << "#define MAPS_APPLICATION_L1_DATA_OFFSET "
     << kGeneratedL1DataBaseOffset << "u\n";
  os << "#define MAPS_APPLICATION_READY_FLAGS_OFFSET "
     << kGeneratedReadyFlagsOffset << "u\n";
  os << "#define MAPS_APPLICATION_READY_FLAGS_WORDS "
     << kGeneratedReadyFlagsWords << "u\n\n";
  if (application.target == "magia-v3") {
    os << "#define MAPS_V3_TASK_SCRATCH_OFFSET 0xC0000u\n";
    os << "#define MAPS_V3_TASK_SCRATCH_BYTES 0x10000u\n";
    os << "#define MAPS_V3_STACK_OFFSET 0xE0000u\n";
    os << "#define MAPS_V3_CONTROL_OFFSET 0xF0000u\n";
    os << "#define MAPS_V3_L1_BYTES 0x100000u\n";
    os << "_Static_assert(MAPS_V3_TASK_SCRATCH_OFFSET + "
          "MAPS_V3_TASK_SCRATCH_BYTES <= "
          "MAPS_APPLICATION_READY_FLAGS_OFFSET, \"task scratch overlaps "
          "ready state\");\n";
    os << "_Static_assert(MAPS_APPLICATION_READY_FLAGS_OFFSET + "
          "MAPS_APPLICATION_READY_FLAGS_WORDS * sizeof(uint32_t) <= "
          "MAPS_V3_STACK_OFFSET, \"ready state overlaps stack\");\n";
    os << "_Static_assert(MAPS_V3_STACK_OFFSET < MAPS_V3_CONTROL_OFFSET, "
          "\"stack overlaps control state\");\n";
    os << "_Static_assert(MAPS_V3_CONTROL_OFFSET < MAPS_V3_L1_BYTES, "
          "\"control state exceeds L1\");\n\n";
  }
  os << "#ifndef MAPS_NUM_TOKENS\n";
  os << "#define MAPS_NUM_TOKENS MAPS_APPLICATION_NUM_TOKENS\n";
  os << "#endif\n\n";
  os << "#ifdef MAPS_NUM_TOKEN_SLOTS\n";
  os << "#if MAPS_NUM_TOKEN_SLOTS != MAPS_APPLICATION_NUM_TOKEN_SLOTS\n";
  os << "#error \"MAPS_NUM_TOKEN_SLOTS disagrees with the planned execution contract\"\n";
  os << "#endif\n";
  os << "#else\n";
  os << "#define MAPS_NUM_TOKEN_SLOTS MAPS_APPLICATION_NUM_TOKEN_SLOTS\n";
  os << "#endif\n\n";
}

static std::string getL2GlobalSymbol(const GeneratedL2Global &global) {
  return (Twine("maps_application_l2_global_") + Twine(global.descriptionIndex)).str();
}

static std::string getWeightsSymbol(const MagiaApplicationDescription &application,
                                    StringRef suffix) {
  return (Twine(application.outputStem) + "_weights_" + suffix).str();
}

static std::string getRuntimeInputSymbol(const MagiaApplicationDescription &application,
                                         const GeneratedL2Global &global,
                                         StringRef suffix) {
  return (Twine(application.outputStem) + "_input_" +
          normalizeSnakeCase(global.globalName) + "_" + suffix)
      .str();
}

static void emitL2GlobalAddress(const MagiaApplicationDescription &application,
                                const GeneratedL2Global &global,
                                uint64_t offset, llvm::raw_ostream &os) {
  if (global.isConstant) {
    os << getWeightsSymbol(application, "start");
    uint64_t packedOffset = global.packedImageOffset + offset;
    if (packedOffset != 0)
      os << " + " << packedOffset << "u";
    return;
  }
  os << getL2GlobalSymbol(global);
  if (offset != 0)
    os << " + " << offset << "u";
}

static uint64_t getL2GlobalAllocationBytes(const GeneratedL2Global &global) {
  return global.tokenStrideBytes * global.numTokens;
}

static void emitL2GlobalDeclarations(const MagiaApplicationDescription &application,
                                     llvm::raw_ostream &os) {
  if (application.hasPackedWeights) {
    os << "extern const uint8_t " << getWeightsSymbol(application, "start")
       << "[];\n";
    os << "extern const uint8_t " << getWeightsSymbol(application, "end")
       << "[];\n";
    os << "extern const uint8_t " << getWeightsSymbol(application, "size")
       << "[];\n";
  }
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.isConstant)
      continue;
    os << "extern uint8_t " << getL2GlobalSymbol(global) << "["
       << getL2GlobalAllocationBytes(global) << "];\n";
  }
  os << "void maps_application_init_l2_data(uint32_t seed);\n\n";
  os << "static inline void maps_application_init_tensors(uint32_t hartid) {\n";
  os << "  if (hartid == 0u) {\n";
  if (application.hasPackedWeights) {
    os << "    if ((((uintptr_t)" << getWeightsSymbol(application, "start")
       << ") & " << (application.weightsAlignment - 1) << "u) != 0u ||\n";
    os << "        (uintptr_t)" << getWeightsSymbol(application, "end")
       << " - (uintptr_t)" << getWeightsSymbol(application, "start") << " != "
       << application.weightsSize << "u)\n";
    os << "      maps_trap();\n";
  }
  os << "    maps_application_init_l2_data(0x1234u);\n";
  os << "  }\n";
  os << "}\n\n";
}

static StringRef getGlobalKindName(GeneratedGlobalKind kind) {
  switch (kind) {
  case GeneratedGlobalKind::Input:
    return "GLOBAL_INPUT";
  case GeneratedGlobalKind::Output:
    return "GLOBAL_OUTPUT";
  case GeneratedGlobalKind::Initializer:
    return "GLOBAL_INITIALIZER";
  case GeneratedGlobalKind::Intermediate:
    return "GLOBAL_INTERMEDIATE";
  }
  llvm_unreachable("unknown global kind");
}

static StringRef getElemTypeName(Type type) {
  if (type.isF32())
    return "ELEM_F32";
  if (type.isF16())
    return "ELEM_F16";
  if (type.isInteger(32))
    return "ELEM_I32";
  if (type.isInteger(8))
    return "ELEM_U8";
  return "ELEM_F32";
}

static StringRef getOpKindName(GeneratedOpKind kind) {
  switch (kind) {
  case GeneratedOpKind::Matmul:
    return "OP_MATMUL";
  case GeneratedOpKind::Conv2D:
    return "OP_CONV2D";
  case GeneratedOpKind::DepthwiseConv:
    return "OP_DEPTHWISE_CONV";
  case GeneratedOpKind::Abs:
    return "OP_ABS";
  case GeneratedOpKind::Add:
    return "OP_ADD";
  case GeneratedOpKind::Div:
    return "OP_DIV";
  case GeneratedOpKind::Log:
    return "OP_LOG";
  case GeneratedOpKind::Exp:
    return "OP_EXP";
  case GeneratedOpKind::SoftmaxExp:
    return "OP_SOFTMAX_EXP";
  case GeneratedOpKind::GroupNormalize:
    return "OP_GROUP_NORMALIZE";
  case GeneratedOpKind::GroupReduce:
    return "OP_GROUP_REDUCE";
  case GeneratedOpKind::GroupCenteredReduce:
    return "OP_GROUP_CENTERED_REDUCE";
  case GeneratedOpKind::Im2Col:
    return "OP_IM2COL";
  case GeneratedOpKind::OutputReformat:
    return "OP_OUTPUT_REFORMAT";
  case GeneratedOpKind::Split:
    return "OP_SPLIT";
  case GeneratedOpKind::Mul:
    return "OP_MUL";
  case GeneratedOpKind::Neg:
    return "OP_NEG";
  case GeneratedOpKind::Pow:
    return "OP_POW";
  case GeneratedOpKind::ReduceMax:
    return "OP_REDUCE_MAX";
  case GeneratedOpKind::ReduceSum:
    return "OP_REDUCE_SUM";
  case GeneratedOpKind::Relu:
    return "OP_RELU";
  case GeneratedOpKind::Reshape:
    return "OP_RESHAPE";
  case GeneratedOpKind::Sigmoid:
    return "OP_SIGMOID";
  case GeneratedOpKind::Slice:
    return "OP_SLICE";
  case GeneratedOpKind::Sqrt:
    return "OP_SQRT";
  case GeneratedOpKind::Sub:
    return "OP_SUB";
  case GeneratedOpKind::Transpose:
    return "OP_TRANSPOSE";
  case GeneratedOpKind::AllReduceMax:
    return "OP_ALL_REDUCE_MAX";
  case GeneratedOpKind::AllReduceSum:
    return "OP_ALL_REDUCE_SUM";
  case GeneratedOpKind::Copy:
    return "OP_COPY";
  }
  llvm_unreachable("unknown op kind");
}

// Keep descriptor array padding consistent with the MAPS tensor-rank limit.
static void emitPaddedI64Array(ArrayRef<int64_t> values,
                               llvm::raw_ostream &os) {
  os << "{";
  for (unsigned i = 0; i < kGeneratedDescriptorMaxDims; ++i) {
    if (i != 0)
      os << ", ";
    os << (i < values.size() ? values[i] : 0);
  }
  os << "}";
}

static void emitPaddedU64Array(ArrayRef<uint64_t> values,
                               llvm::raw_ostream &os) {
  os << "{";
  for (unsigned i = 0; i < kGeneratedDescriptorMaxDims; ++i) {
    if (i != 0)
      os << ", ";
    os << (i < values.size() ? values[i] : 0);
  }
  os << "}";
}

static void emitSliceDesc(const GeneratedSlice &slice,
                          llvm::raw_ostream &os) {
  os << "  {\n";
  os << "    .slice_id = " << slice.sliceId << ",\n";
  os << "    .global_id = " << slice.globalId << ",\n";
  os << "    .global_kind = " << getGlobalKindName(slice.globalKind) << ",\n";
  os << "    .owner_hartid = " << slice.ownerHartid << ",\n";
  os << "    .full_offset = ";
  emitPaddedI64Array(slice.fullOffset, os);
  os << ",\n";
  os << "    .rank = " << slice.shape.size() << ",\n";
  os << "    .shape = ";
  emitPaddedI64Array(slice.shape, os);
  os << ",\n";
  os << "    .elem_type = " << getElemTypeName(slice.elementType) << ",\n";
  os << "    .elem_bytes = " << slice.elemBytes << ",\n";
  os << "    .l1_offset_bytes = " << slice.l1OffsetBytes << ",\n";
  os << "    .slot_bytes = " << slice.slotBytes << ",\n";
  os << "    .num_slots = " << slice.numSlots << ",\n";
  os << "    .strides_bytes = ";
  emitPaddedU64Array(slice.stridesBytes, os);
  os << ",\n";
  os << "  }";
}

static void emitSubsliceDesc(const MagiaApplicationDescription &application,
                             const GeneratedSubslice &subslice,
                             llvm::raw_ostream &os) {
  const GeneratedSlice &slice = application.slices[subslice.sliceIndex];

  os << "{\n";
  os << "      .slice_id = " << slice.sliceId << ",\n";
  os << "      .offset = ";
  emitPaddedI64Array(subslice.offset, os);
  os << ",\n";
  os << "      .rank = " << subslice.shape.size() << ",\n";
  os << "      .shape = ";
  emitPaddedI64Array(subslice.shape, os);
  os << ",\n";
  os << "      .elem_type = " << getElemTypeName(subslice.elementType) << ",\n";
  os << "      .elem_bytes = " << subslice.elemBytes << ",\n";
  os << "      .strides_bytes = ";
  emitPaddedU64Array(subslice.stridesBytes, os);
  os << ",\n";
  os << "    }";
}

static void emitTensorSubSlice(ArrayRef<int64_t> shape,
                               ArrayRef<uint64_t> stridesBytes,
                               llvm::raw_ostream &os) {
  os << "{\n";
  os << "      .rank = " << shape.size() << "u,\n";
  os << "      .num_elems = " << product(shape) << "u,\n";
  os << "      .dims = {";
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << "{0u, " << shape[i] << "u, " << stridesBytes[i] << "u}";
  }
  os << "},\n";
  os << "    }";
}

static void emitL2ReadDesc(const MagiaApplicationDescription &application,
                           const GeneratedL2Read &read,
                           llvm::raw_ostream &os) {
  const GeneratedL2Global &global = application.l2Globals[read.l2GlobalIndex];
  const GeneratedSubslice &dst = application.subslices[read.dstSubsliceIndex];

  os << "  {\n";
  os << "    .global_id = " << global.globalId << ",\n";
  os << "    .global_kind = " << getGlobalKindName(global.globalKind) << ",\n";
  os << "    .src_l2_addr = (uint32_t)";
  uint64_t srcOffset = read.srcL2Address - global.l2Address;
  emitL2GlobalAddress(application, global, srcOffset, os);
  os << ",\n";
  os << "    .src_l2_token_stride_bytes = "
     << read.srcL2TokenStrideBytes << ",\n";
  os << "    .dst = ";
  emitSubsliceDesc(application, dst, os);
  os << ",\n";
  os << "    .copy_src = ";
  emitTensorSubSlice(dst.shape, global.stridesBytes, os);
  os << ",\n";
  os << "    .copy_dst = ";
  emitTensorSubSlice(dst.shape, dst.stridesBytes, os);
  os << ",\n";
  os << "  }";
}

static void emitRecvDesc(const MagiaApplicationDescription &application,
                         const GeneratedRecv &recv,
                         llvm::raw_ostream &os) {
  const GeneratedSubslice &dst = application.subslices[recv.dstSubsliceIndex];

  os << "  {\n";
  os << "    .transition_id = " << recv.transitionId << ",\n";
  os << "    .src_hartid = " << recv.srcHartid << ",\n";
  os << "    .dst_hartid = " << recv.dstHartid << ",\n";
  os << "    .dst = ";
  emitSubsliceDesc(application, dst, os);
  os << ",\n";
  os << "    .producer_idx = " << recv.producerIdx << ",\n";
  os << "  }";
}

// Decides which receives the magia-v3 runtime gathers in place: slices whose
// receives tile them exactly, each on its own FIFO sub-ring, get a shared
// payload ring that mirrors the slice's planned storage.  The receiver's FIFO
// layout and every sender's payload mode are emitted from this one decision.
//
// The runtime registers each gathered slice in its slice alias table, indexed
// by (slice id, token slot), and traps outside the default MAPS_ALIAS_MAX_SLICES
// and MAPS_ALIAS_MAX_SLOTS of maps_utils.h; generated tiles assert these.
static constexpr unsigned kRuntimeAliasMaxSlices = 64;
static constexpr unsigned kRuntimeAliasMaxSlots = 4;

static bool isSliceInplaceEligible(const MagiaApplicationDescription &application,
                                   const GeneratedTilePlan &plan,
                                   unsigned sliceIndex) {
  auto readsSlice = [&](unsigned subsliceIndex) {
    return application.subslices[subsliceIndex].sliceIndex == sliceIndex;
  };
  for (unsigned readIndex : plan.l2ReadIndices)
    if (readsSlice(application.l2Reads[readIndex].dstSubsliceIndex))
      return false;
  for (unsigned sendIndex : plan.sendIndices)
    if (readsSlice(application.sends[sendIndex].srcSubsliceIndex))
      return false;
  for (unsigned writeIndex : plan.l2WriteIndices)
    if (readsSlice(application.l2Writes[writeIndex].srcSubsliceIndex))
      return false;
  for (unsigned opIndex : plan.opIndices) {
    const GeneratedOp &op = application.ops[opIndex];
    if (op.kind != GeneratedOpKind::AllReduceMax &&
        op.kind != GeneratedOpKind::AllReduceSum && !op.isCollective)
      continue;
    if (llvm::any_of(op.inputSubsliceIndices, readsSlice) ||
        llvm::any_of(op.outputSubsliceIndices, readsSlice))
      return false;
  }
  return true;
}

static bool isRecvInplaceEligible(const MagiaApplicationDescription &application,
                                  const GeneratedTilePlan &plan,
                                  unsigned position) {
  const GeneratedRecv &recv = application.recvs[plan.recvIndices[position]];
  const GeneratedSubslice &dst = application.subslices[recv.dstSubsliceIndex];
  const GeneratedSlice &slice = application.slices[dst.sliceIndex];
  if (slice.globalKind == GeneratedGlobalKind::Initializer)
    return false;
  if (dst.shape.size() != slice.shape.size() || dst.shape.empty() ||
      dst.elemBytes != slice.elemBytes)
    return false;

  uint64_t packedStride = dst.elemBytes;
  for (unsigned d = dst.shape.size(); d-- > 0;) {
    if (dst.offset[d] != 0 || dst.shape[d] != slice.shape[d] ||
        dst.stridesBytes[d] != slice.stridesBytes[d] ||
        slice.stridesBytes[d] != packedStride)
      return false;
    packedStride *= dst.shape[d];
  }
  for (unsigned i = 0; i < plan.recvIndices.size(); ++i)
    if (i != position &&
        application.subslices[application.recvs[plan.recvIndices[i]].dstSubsliceIndex]
                .sliceIndex == dst.sliceIndex)
      return false;
  return isSliceInplaceEligible(application, plan, dst.sliceIndex);
}

static bool isSliceGatherEligible(const MagiaApplicationDescription &application,
                                  const GeneratedTilePlan &plan,
                                  unsigned sliceIndex) {
  const GeneratedSlice &slice = application.slices[sliceIndex];
  if (slice.globalKind == GeneratedGlobalKind::Initializer || slice.shape.empty() ||
      slice.sliceId >= kRuntimeAliasMaxSlices ||
      application.numTokenSlots > kRuntimeAliasMaxSlots ||
      !isSliceInplaceEligible(application, plan, sliceIndex))
    return false;

  uint64_t covered = 0;
  for (unsigned i = 0; i < plan.recvIndices.size(); ++i) {
    const GeneratedSubslice &dst =
        application.subslices[application.recvs[plan.recvIndices[i]].dstSubsliceIndex];
    if (dst.sliceIndex != sliceIndex)
      continue;
    if (dst.shape.size() != slice.shape.size() || dst.elemBytes != slice.elemBytes)
      return false;
    for (unsigned d = 0; d < dst.shape.size(); ++d)
      if (dst.stridesBytes[d] != slice.stridesBytes[d] ||
          dst.offset[d] + dst.shape[d] > slice.shape[d])
        return false;
    for (unsigned j = 0; j < i; ++j) {
      const GeneratedSubslice &other = application.subslices
          [application.recvs[plan.recvIndices[j]].dstSubsliceIndex];
      if (other.sliceIndex != sliceIndex)
        continue;
      bool overlap = true;
      for (unsigned d = 0; d < dst.shape.size(); ++d)
        if (other.offset[d] + other.shape[d] <= dst.offset[d] ||
            dst.offset[d] + dst.shape[d] <= other.offset[d])
          overlap = false;
      if (overlap)
        return false;
    }
    covered += product(dst.shape);
  }
  return covered != 0 && covered == product(slice.shape);
}

static void planGatheredFifoLayout(const MagiaApplicationDescription &application,
                                   GeneratedTilePlan &plan) {
  SmallVector<unsigned> payloadRings(plan.recvIndices.size(), 0);
  SmallVector<uint64_t> ringBytes;
  for (unsigned i = 0; i < plan.recvIndices.size(); ++i) {
    // TransferCollector gives each receive its own sub-ring, numbered by its
    // position on the destination tile.
    assert(application.recvs[plan.recvIndices[i]].producerIdx == i &&
           "receive producer index is not its position on the tile");
    unsigned sliceIndex =
        application.subslices[application.recvs[plan.recvIndices[i]].dstSubsliceIndex]
            .sliceIndex;
    bool first = true;
    for (unsigned j = 0; j < i; ++j)
      first &= application.subslices[application.recvs[plan.recvIndices[j]]
                                         .dstSubsliceIndex]
                   .sliceIndex != sliceIndex;
    if (!first || isRecvInplaceEligible(application, plan, i) ||
        !isSliceGatherEligible(application, plan, sliceIndex))
      continue;
    ringBytes.push_back(application.slices[sliceIndex].slotBytes);
    for (unsigned j = i; j < plan.recvIndices.size(); ++j) {
      const GeneratedRecv &recv = application.recvs[plan.recvIndices[j]];
      if (application.subslices[recv.dstSubsliceIndex].sliceIndex == sliceIndex)
        payloadRings[j] = ringBytes.size();
    }
  }
  if (ringBytes.empty())
    return;
  plan.fifoPayloadRings = std::move(payloadRings);
  plan.fifoSharedRingBytes = std::move(ringBytes);
}

void planGatheredFifoLayouts(MagiaApplicationDescription &application) {
  if (application.target != "magia-v3")
    return;
  for (auto &[hartid, plan] : application.tilePlans)
    planGatheredFifoLayout(application, plan);
}

static bool isGatheredSend(const MagiaApplicationDescription &application,
                           const GeneratedSend &send) {
  auto it = application.tilePlans.find(send.dstHartid);
  if (it == application.tilePlans.end())
    llvm_unreachable("send targets a tile without a plan");
  const GeneratedTilePlan &plan = it->second;
  return !plan.fifoPayloadRings.empty() &&
         plan.fifoPayloadRings[send.producerIdx] != 0;
}

static void emitSendDesc(const MagiaApplicationDescription &application,
                         const GeneratedSend &send,
                         llvm::raw_ostream &os) {
  const GeneratedSubslice &src = application.subslices[send.srcSubsliceIndex];
  const GeneratedSubslice &dst = application.subslices[send.dstSubsliceIndex];

  os << "  {\n";
  os << "    .transition_id = " << send.transitionId << ",\n";
  os << "    .src_hartid = " << send.srcHartid << ",\n";
  os << "    .dst_hartid = " << send.dstHartid << ",\n";
  os << "    .src = ";
  emitSubsliceDesc(application, src, os);
  os << ",\n";
  os << "    .dst = ";
  emitSubsliceDesc(application, dst, os);
  os << ",\n";
  os << "    .dst_l1_data_base = " << send.dstL1DataBase << ",\n";
  os << "    .dst_slice_l1_offset_bytes = "
     << send.dstSliceL1OffsetBytes << ",\n";
  os << "    .dst_slice_slot_bytes = " << send.dstSliceSlotBytes << ",\n";
  os << "    .copy_src = ";
  emitTensorSubSlice(src.shape, src.stridesBytes, os);
  os << ",\n";
  os << "    .copy_dst = ";
  emitTensorSubSlice(dst.shape, dst.stridesBytes, os);
  os << ",\n";
  os << "    .producer_idx = " << send.producerIdx << ",\n";
  if (application.target == "magia-v3") {
    // Builds with MAPS_FIFO_INPLACE_RECV=0 ignore the shared payload rings.
    bool gathered = isGatheredSend(application, send);
    os << "    .dst_payload_mode = "
       << (gathered ? "MAPS_FIFO_INPLACE_RECV ? FIFO_PAYLOAD_SHARED : "
                      "FIFO_PAYLOAD_INLINE"
                    : "FIFO_PAYLOAD_INLINE")
       << ",\n";
    os << "    .dst_offset_bytes = "
       << (gathered ? computeOffsetBytes(dst.offset, dst.stridesBytes) : 0)
       << "u,\n";
  }
  os << "  }";
}

static void emitSubsliceArray(const MagiaApplicationDescription &application,
                              ArrayRef<unsigned> subsliceIndices,
                              unsigned paddedSize,
                              llvm::raw_ostream &os) {
  os << "{";
  for (unsigned i = 0; i < paddedSize; ++i) {
    if (i != 0)
      os << ", ";
    if (i < subsliceIndices.size())
      emitSubsliceDesc(application, application.subslices[subsliceIndices[i]], os);
    else
      os << "{0}";
  }
  os << "}";
}

using CollectiveKey = std::tuple<unsigned, unsigned, unsigned>;

static CollectiveKey getCollectiveKey(const GeneratedOp &op) {
  return {op.collectiveIdentity.stageId, op.collectiveIdentity.layerIndex,
          op.collectiveIdentity.groupId};
}

static unsigned getCollectiveFlagBase(const MagiaApplicationDescription &application,
                                      const GeneratedOp &target) {
  std::map<CollectiveKey, const GeneratedOp *> collectives;
  for (const GeneratedOp &op : application.ops)
    if (op.isCollective)
      collectives.try_emplace(getCollectiveKey(op), &op);

  unsigned flagBase = 0;
  for (const auto &[key, op] : collectives) {
    if (key == getCollectiveKey(target))
      return flagBase;
    flagBase += 2 * application.numTokenSlots * op->collectiveParticipants.size();
  }
  llvm_unreachable("collective operation has no stable flag allocation");
}

static const GeneratedOp &getParticipantCollectiveOp(
    const MagiaApplicationDescription &application, const GeneratedOp &target,
    unsigned participantHartid) {
  for (const GeneratedOp &op : application.ops)
    if (op.isCollective && op.ownerHartid == participantHartid &&
        getCollectiveKey(op) == getCollectiveKey(target))
      return op;
  llvm_unreachable("collective participant has no operation descriptor");
}

static uint64_t getSubsliceL1Offset(const MagiaApplicationDescription &application,
                                    unsigned subsliceIndex) {
  const GeneratedSubslice &subslice = application.subslices[subsliceIndex];
  const GeneratedSlice &slice = application.slices[subslice.sliceIndex];
  return slice.l1OffsetBytes +
         computeOffsetBytes(subslice.offset, subslice.stridesBytes);
}

static uint64_t getFifoSlotDataSize(const MagiaApplicationDescription &application,
                                    unsigned recvIndex);

// Bytes of the magia-v3 variable FIFO layout or, with `shared`, of the shared
// layout the runtime builds when the tile gathers slices in place (members keep
// metadata-only slots, plus the payload map and shared rings).
static void emitFifoVariantBytes(const MagiaApplicationDescription &application,
                                 const GeneratedTilePlan &plan, bool shared,
                                 llvm::raw_ostream &os) {
  unsigned numProducers = plan.recvIndices.size();
  os << "FIFO_HEADER_SIZE + " << numProducers
     << "u * FIFO_RING_STATE_SIZE + " << numProducers + 1
     << "u * FIFO_SLOT_OFFSET_SIZE";
  for (unsigned i = 0; i < numProducers; ++i) {
    uint64_t dataBytes = shared && plan.fifoPayloadRings[i] != 0
                             ? 0
                             : getFifoSlotDataSize(application, plan.recvIndices[i]);
    os << " + MAPS_NUM_TOKEN_SLOTS * (FIFO_SLOT_META_SIZE + ((" << dataBytes
       << "u + 3u) & ~3u))";
  }
  if (!shared)
    return;
  os << " + " << numProducers << "u * FIFO_PAYLOAD_RING_SIZE";
  for (uint64_t bytes : plan.fifoSharedRingBytes)
    os << " + MAPS_NUM_TOKEN_SLOTS * ((" << bytes << "u + 3u) & ~3u)";
}

// Bytes the magia-v3 FIFO takes on this tile.  Builds with
// MAPS_FIFO_INPLACE_RECV=0 ignore the shared payload rings and build the
// variable layout instead.
static void emitFifoLayoutBytes(const MagiaApplicationDescription &application,
                                const GeneratedTilePlan &plan,
                                llvm::raw_ostream &os) {
  if (plan.fifoSharedRingBytes.empty()) {
    emitFifoVariantBytes(application, plan, /*shared=*/false, os);
    return;
  }
  os << "(MAPS_FIFO_INPLACE_RECV ? (";
  emitFifoVariantBytes(application, plan, /*shared=*/true, os);
  os << ") : (";
  emitFifoVariantBytes(application, plan, /*shared=*/false, os);
  os << "))";
}

static void emitL1DataBaseOffset(const MagiaApplicationDescription &application,
                                 int64_t hartid, llvm::raw_ostream &os) {
  const GeneratedTilePlan &plan = application.tilePlans.find(hartid)->second;
  if (application.target != "magia-v3") {
    uint64_t slotDataSize = 0;
    for (unsigned recvIndex : plan.recvIndices)
      slotDataSize = std::max(
          slotDataSize, getFifoSlotDataSize(application, recvIndex));
    os << "FIFO_HEADER_SIZE + " << plan.recvIndices.size()
       << "u * FIFO_RING_STATE_SIZE + " << plan.recvIndices.size()
       << "u * MAPS_NUM_TOKEN_SLOTS * (FIFO_SLOT_META_SIZE + (("
       << slotDataSize << "u + 3u) & ~3u))";
    return;
  }
  emitFifoLayoutBytes(application, plan, os);
}

static void emitOpDesc(const MagiaApplicationDescription &application,
                       const GeneratedOp &op,
                       llvm::raw_ostream &os) {
  os << "  {\n";
  os << "    .kind = " << getOpKindName(op.kind) << ",\n";
  os << "    .num_inputs = " << op.inputSubsliceIndices.size() << "u,\n";
  os << "    .inputs = ";
  emitSubsliceArray(application, op.inputSubsliceIndices, 5, os);
  os << ",\n";
  os << "    .num_outputs = " << op.outputSubsliceIndices.size() << "u,\n";
  os << "    .outputs = ";
  emitSubsliceArray(application, op.outputSubsliceIndices, 3, os);
  os << ",\n";
  if (op.isCollective) {
    os << "    .collective = {.stage_id = " << op.collectiveIdentity.stageId
       << "u, .layer_index = " << op.collectiveIdentity.layerIndex
       << "u, .group_id = " << op.collectiveIdentity.groupId
       << "u, .flag_base = " << getCollectiveFlagBase(application, op)
       << "u, .num_token_slots = " << application.numTokenSlots
       << "u, .num_participants = " << op.collectiveParticipants.size()
       << "u, .participants = {";
    for (auto participant : llvm::enumerate(op.collectiveParticipants)) {
      if (participant.index() != 0)
        os << ", ";
      const GeneratedOp &participantOp = getParticipantCollectiveOp(
          application, op, participant.value());
      const GeneratedSubslice &input =
          application.subslices[participantOp.inputSubsliceIndices.front()];
      const GeneratedSubslice &output =
          application.subslices[participantOp.outputSubsliceIndices.front()];
      const GeneratedSlice &inputSlice = application.slices[input.sliceIndex];
      const GeneratedSlice &outputSlice = application.slices[output.sliceIndex];
      os << "{.hartid = " << participant.value()
         << "u, .l1_data_base_offset = ";
      emitL1DataBaseOffset(application, participant.value(), os);
      os << ", .input_l1_offset_bytes = "
         << getSubsliceL1Offset(application,
                                participantOp.inputSubsliceIndices.front())
         << "u, .input_slot_bytes = " << inputSlice.slotBytes
         << "u, .output_l1_offset_bytes = "
         << getSubsliceL1Offset(application,
                                participantOp.outputSubsliceIndices.front())
         << "u, .output_slot_bytes = " << outputSlice.slotBytes << "u}";
    }
    os << "}},\n";
  }
  os << "    .params = {";
  for (unsigned i = 0; i < 8; ++i) {
    if (i != 0)
      os << ", ";
    os << op.params[i] << "u";
  }
  os << "},\n";
  os << "  }";
}

static void emitL2WriteDesc(const MagiaApplicationDescription &application,
                            const GeneratedL2Write &write,
                            llvm::raw_ostream &os) {
  const GeneratedL2Global &global = application.l2Globals[write.l2GlobalIndex];
  const GeneratedSubslice &src = application.subslices[write.srcSubsliceIndex];

  os << "  {\n";
  os << "    .global_id = " << global.globalId << ",\n";
  os << "    .global_kind = " << getGlobalKindName(global.globalKind) << ",\n";
  os << "    .src = ";
  emitSubsliceDesc(application, src, os);
  os << ",\n";
  os << "    .dst_l2_addr = (uint32_t)" << getL2GlobalSymbol(global);
  uint64_t dstOffset = write.dstL2Address - global.l2Address;
  if (dstOffset != 0)
    os << " + " << dstOffset << "u";
  os << ",\n";
  os << "    .dst_l2_token_stride_bytes = "
     << write.dstL2TokenStrideBytes << ",\n";
  os << "    .copy_src = ";
  emitTensorSubSlice(src.shape, src.stridesBytes, os);
  os << ",\n";
  os << "    .copy_dst = ";
  emitTensorSubSlice(src.shape, global.stridesBytes, os);
  os << ",\n";
  os << "  }";
}

static void emitTileSliceDescriptors(const MagiaApplicationDescription &application,
                                     const GeneratedTilePlan &plan,
                                     llvm::raw_ostream &os) {
  os << "static const slice_desc_t maps_application_tile_"
     << plan.hartid << "_slices[] = {\n";
  for (unsigned sliceIndex : plan.sliceIndices) {
    emitSliceDesc(application.slices[sliceIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static void emitTileL2ReadDescriptors(const MagiaApplicationDescription &application,
                                      const GeneratedTilePlan &plan,
                                      llvm::raw_ostream &os) {
  if (!plan.initL2ReadIndices.empty()) {
    os << "static const l2_read_desc_t maps_application_tile_"
       << plan.hartid << "_init_l2_reads[] = {\n";
    for (unsigned readIndex : plan.initL2ReadIndices) {
      emitL2ReadDesc(application, application.l2Reads[readIndex], os);
      os << ",\n";
    }
    os << "};\n\n";
  }

  if (!plan.l2ReadIndices.empty()) {
    os << "static const l2_read_desc_t maps_application_tile_"
       << plan.hartid << "_l2_reads[] = {\n";
    for (unsigned readIndex : plan.l2ReadIndices) {
      emitL2ReadDesc(application, application.l2Reads[readIndex], os);
      os << ",\n";
    }
    os << "};\n\n";
  }
}

static void emitTileRecvDescriptors(const MagiaApplicationDescription &application,
                                    const GeneratedTilePlan &plan,
                                    llvm::raw_ostream &os) {
  if (plan.recvIndices.empty())
    return;

  os << "static const fifo_recv_desc_t maps_application_tile_"
     << plan.hartid << "_recvs[] = {\n";
  for (unsigned recvIndex : plan.recvIndices) {
    emitRecvDesc(application, application.recvs[recvIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static void emitTileSendDescriptors(const MagiaApplicationDescription &application,
                                    const GeneratedTilePlan &plan,
                                    llvm::raw_ostream &os) {
  if (plan.sendIndices.empty())
    return;

  os << "static const fifo_send_desc_t maps_application_tile_"
     << plan.hartid << "_sends[] = {\n";
  for (unsigned sendIndex : plan.sendIndices) {
    emitSendDesc(application, application.sends[sendIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static void emitTileOpDescriptors(const MagiaApplicationDescription &application,
                                  const GeneratedTilePlan &plan,
                                  llvm::raw_ostream &os) {
  if (plan.opIndices.empty())
    return;

  os << "static const op_desc_t maps_application_tile_"
     << plan.hartid << "_ops[] = {\n";
  for (unsigned opIndex : plan.opIndices) {
    emitOpDesc(application, application.ops[opIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static void emitTileL2WriteDescriptors(const MagiaApplicationDescription &application,
                                       const GeneratedTilePlan &plan,
                                       llvm::raw_ostream &os) {
  if (plan.l2WriteIndices.empty())
    return;

  os << "static const l2_write_desc_t maps_application_tile_"
     << plan.hartid << "_l2_writes[] = {\n";
  for (unsigned writeIndex : plan.l2WriteIndices) {
    emitL2WriteDesc(application, application.l2Writes[writeIndex], os);
    os << ",\n";
  }
  os << "};\n\n";
}

static uint64_t getFifoSlotDataSize(const MagiaApplicationDescription &application,
                                    unsigned recvIndex) {
  const GeneratedRecv &recv = application.recvs[recvIndex];
  const GeneratedSubslice &dst = application.subslices[recv.dstSubsliceIndex];
  return product(dst.shape) * dst.elemBytes;
}

static void emitFifoSlotDataSizes(const MagiaApplicationDescription &application,
                                  const GeneratedTilePlan &plan,
                                  llvm::raw_ostream &os) {
  if (application.target != "magia-v3" || plan.recvIndices.empty())
    return;

  os << "static const uint32_t maps_application_tile_" << plan.hartid
     << "_fifo_slot_data_sizes[] = {";
  for (unsigned i = 0; i < plan.recvIndices.size(); ++i) {
    if (i != 0)
      os << ", ";
    os << getFifoSlotDataSize(application, plan.recvIndices[i]) << "u";
  }
  os << "};\n\n";

  if (plan.fifoSharedRingBytes.empty())
    return;
  os << "static const uint32_t maps_application_tile_" << plan.hartid
     << "_fifo_payload_rings[] = {";
  for (unsigned i = 0; i < plan.fifoPayloadRings.size(); ++i)
    os << (i != 0 ? ", " : "") << plan.fifoPayloadRings[i] << "u";
  os << "};\n\n";
  os << "static const uint32_t maps_application_tile_" << plan.hartid
     << "_fifo_shared_ring_bytes[] = {";
  for (unsigned i = 0; i < plan.fifoSharedRingBytes.size(); ++i)
    os << (i != 0 ? ", " : "") << plan.fifoSharedRingBytes[i] << "u";
  os << "};\n\n";

  // Gathered slices must fit the runtime's slice alias table.
  unsigned maxSliceId = 0;
  for (unsigned i = 0; i < plan.recvIndices.size(); ++i)
    if (plan.fifoPayloadRings[i] != 0)
      maxSliceId = std::max(
          maxSliceId,
          application.slices[application.subslices
                                 [application.recvs[plan.recvIndices[i]]
                                      .dstSubsliceIndex]
                                     .sliceIndex]
              .sliceId);
  os << "#if MAPS_FIFO_INPLACE_RECV\n";
  os << "_Static_assert(" << maxSliceId
     << "u < MAPS_ALIAS_MAX_SLICES && MAPS_APPLICATION_NUM_TOKEN_SLOTS <= "
        "MAPS_ALIAS_MAX_SLOTS, \"gathered slices exceed the slice alias "
        "table\");\n";
  os << "#endif\n\n";
}

static void emitPlanArrayInitializer(StringRef fieldName, StringRef arraySuffix,
                                     unsigned count, int64_t hartid,
                                     llvm::raw_ostream &os) {
  if (count == 0)
    return;

  os << "  .num_" << fieldName << " = " << count << "u,\n";
  os << "  ." << fieldName << " = maps_application_tile_"
     << hartid << "_" << arraySuffix << ",\n";
}

static void emitApplicationDataDefinitions(
    const MagiaApplicationDescription &application, llvm::raw_ostream &os) {
  os << "#include <stdint.h>\n";
  os << "#include <stddef.h>\n\n";

  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.isConstant)
      continue;
    os << "uint8_t " << getL2GlobalSymbol(global) << "["
       << getL2GlobalAllocationBytes(global)
       << "] __attribute__((aligned(16)";
    if (application.target == "magia-v3")
      os << ", section(\".l2_arena.maps_application\")";
    os << "));\n";
  }

  if (application.randomizeInputs) {
    os << "\nstatic uint32_t maps_application_lcg(uint32_t *state) {\n";
    os << "  *state = *state * 1664525u + 1013904223u;\n";
    os << "  return *state;\n";
    os << "}\n\n";
  }

  os << "static void maps_application_zero_bytes(uint8_t *dst, size_t bytes) {\n";
  os << "  for (size_t i = 0; i < bytes; ++i)\n";
  os << "    dst[i] = 0u;\n";
  os << "}\n\n";

  if (application.randomizeInputs) {
    os << "static void maps_application_fill_f16(uint8_t *dst, size_t bytes,\n";
    os << "                                    uint32_t *state) {\n";
    os << "  static const uint16_t values[] = {\n";
    os << "      0x3c00u, 0x4000u, 0x4200u, 0x4400u,\n";
    os << "      0x4500u, 0x4600u, 0x4700u, 0x4800u,\n";
    os << "  };\n";
    os << "  uint16_t *half = (uint16_t *)dst;\n";
    os << "  size_t elems = bytes / sizeof(uint16_t);\n";
    os << "  for (size_t i = 0; i < elems; ++i)\n";
    os << "    half[i] = values[(maps_application_lcg(state) >> 24) & 7u];\n";
    os << "}\n\n";

    os << "static void maps_application_fill_bytes(uint8_t *dst, size_t bytes,\n";
    os << "                                      uint32_t *state) {\n";
    os << "  for (size_t i = 0; i < bytes; ++i)\n";
    os << "    dst[i] = (uint8_t)(maps_application_lcg(state) >> 24);\n";
    os << "}\n\n";
  }

  os << "void maps_application_init_l2_data(uint32_t seed) {\n";
  if (application.randomizeInputs)
    os << "  uint32_t state = seed;\n";
  else
    os << "  (void)seed;\n";
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.isConstant)
      continue;
    os << "\n";
    if (application.randomizeInputs &&
        (global.globalKind == GeneratedGlobalKind::Input ||
         global.globalKind == GeneratedGlobalKind::Initializer)) {
      if (global.elemBytes == 2) {
        os << "  maps_application_fill_f16(" << getL2GlobalSymbol(global)
           << ", sizeof(" << getL2GlobalSymbol(global) << "), &state);\n";
      } else {
        os << "  maps_application_fill_bytes(" << getL2GlobalSymbol(global)
           << ", sizeof(" << getL2GlobalSymbol(global) << "), &state);\n";
      }
    } else {
      os << "  maps_application_zero_bytes(" << getL2GlobalSymbol(global)
         << ", sizeof(" << getL2GlobalSymbol(global) << "));\n";
    }
  }
  os << "}\n";
}

static std::string uppercase(StringRef value) {
  std::string result = value.str();
  for (char &character : result)
    character = llvm::toUpper(character);
  return result;
}

static std::string escapePrintfText(StringRef value) {
  std::string result;
  for (unsigned char character : value.bytes()) {
    switch (character) {
    case '%':
      result += "%%";
      break;
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
    default:
      if (character >= 0x20 && character <= 0x7e) {
        result += static_cast<char>(character);
      } else {
        result += '\\';
        result += static_cast<char>('0' + ((character >> 6) & 7));
        result += static_cast<char>('0' + ((character >> 3) & 7));
        result += static_cast<char>('0' + (character & 7));
      }
    }
  }
  return result;
}

static void replaceAll(std::string &text, StringRef from, StringRef to) {
  size_t position = 0;
  while ((position = text.find(from.str(), position)) != std::string::npos) {
    text.replace(position, from.size(), to.str());
    position += to.size();
  }
}

static std::string emitWithApplicationSymbols(
    const MagiaApplicationDescription &application,
    llvm::function_ref<void(llvm::raw_ostream &)> emit) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  emit(stream);
  stream.flush();
  replaceAll(text, "MAPS_APPLICATION", uppercase(application.outputStem));
  replaceAll(text, "maps_application", application.outputStem);
  replaceAll(text, ".l2_arena." + application.outputStem,
             ".l2_arena.maps_application");
  replaceAll(text, application.outputStem + "_weights_",
             application.outputStem + "_initializers_");
  return text;
}

void emitApplicationHeader(const MagiaApplicationDescription &application,
                           llvm::raw_ostream &os) {
  std::string guard = uppercase(application.outputStem) + "_H";
  os << "#ifndef " << guard << "\n";
  os << "#define " << guard << "\n\n";
  os << "#include <stdint.h>\n\n";
  os << "#define " << uppercase(application.outputStem) << "_EXECUTION_TOKENS "
     << application.numTokens << "u\n";
  os << "#define " << uppercase(application.outputStem) << "_TOKEN_SLOTS "
     << application.numTokenSlots << "u\n\n";
  os << "int " << application.outputStem << "_run(void);\n";
  os << "void " << application.outputStem << "_handle_input(uint32_t token);\n";
  os << "void " << application.outputStem << "_handle_output(uint32_t token);\n\n";
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.globalKind == GeneratedGlobalKind::Input)
      os << "uint8_t *" << application.outputStem << "_input_"
         << normalizeSnakeCase(global.globalName) << "(uint32_t token);\n";
    if (global.globalKind == GeneratedGlobalKind::Output)
      os << "const uint8_t *" << application.outputStem << "_output_"
         << normalizeSnakeCase(global.globalName) << "(uint32_t token);\n";
  }
  os << "\n#endif /* " << guard << " */\n";
}

void emitApplicationDataSource(const MagiaApplicationDescription &application,
                                llvm::raw_ostream &os) {
  os << emitWithApplicationSymbols(application, [&](llvm::raw_ostream &stream) {
    emitGeneratedPrelude(application, stream);
    emitL2GlobalDeclarations(application, stream);
  });
  os << "\n";
  os << emitWithApplicationSymbols(
      application, [&](llvm::raw_ostream &stream) { emitApplicationDataDefinitions(application, stream); });
  os << "\n";
  for (const GeneratedTilePlan *plan : getSortedTilePlans(application)) {
    os << "extern const fifo_tile_plan_t " << application.outputStem
       << "_tile_" << getPaddedPhysicalTileId(application, plan->hartid)
       << "_plan;\n";
  }
  uint64_t physicalTileCount =
      static_cast<uint64_t>(application.meshWidth) * application.meshHeight;
  os << "static const fifo_tile_plan_t *const " << application.outputStem
     << "_tile_plans[" << physicalTileCount << "] = {\n";
  for (const GeneratedTilePlan *plan : getSortedTilePlans(application))
    os << "  [" << plan->hartid << "] = &" << application.outputStem
       << "_tile_" << getPaddedPhysicalTileId(application, plan->hartid)
       << "_plan,\n";
  os << "};\n";
  os << "\nvoid " << application.outputStem
     << "_initialize_data(uint32_t hartid) {\n";
  os << "  " << application.outputStem << "_init_tensors(hartid);\n";
  os << "}\n\n";
  os << "uint32_t " << application.outputStem << "_build_plan(\n";
  os << "    fifo_tile_plan_t *plan, uint32_t hartid,\n";
  os << "    maps_operation_runtime_t *runtime) {\n";
  os << "  if (hartid >= " << physicalTileCount << "u || !"
     << application.outputStem << "_tile_plans[hartid])\n";
  os << "    return 0u;\n";
  os << "  *plan = *" << application.outputStem << "_tile_plans[hartid];\n";
  os << "  plan->l1_data_base += get_l1_base(hartid);\n";
  os << "  plan->operation_runtime = runtime;\n";
  os << "  return 1u;\n";
  os << "}\n\n";
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Input &&
        global.globalKind != GeneratedGlobalKind::Output)
      continue;
    StringRef direction =
        global.globalKind == GeneratedGlobalKind::Input ? "input" : "output";
    if (global.globalKind == GeneratedGlobalKind::Output)
      os << "const ";
    os << "uint8_t *" << application.outputStem << "_" << direction << "_"
       << normalizeSnakeCase(global.globalName) << "(uint32_t token) {\n";
    os << "  return " << application.outputStem << "_l2_global_" << global.descriptionIndex
       << " + token * " << global.tokenStrideBytes << "u;\n";
    os << "}\n\n";
  }
}

static void emitTileDescriptorL1Offset(const MagiaApplicationDescription &application,
                                       const GeneratedTilePlan &plan,
                                       llvm::raw_ostream &os) {
  os << "((";
  emitL1DataBaseOffset(application, plan.hartid, os);
  os << " + " << plan.l1DataBytes << "u + 15u) & ~15u)";
}

static void emitApplicationTilePlanConstructor(
    const MagiaApplicationDescription &application, const GeneratedTilePlan &plan,
    llvm::raw_ostream &os) {
  if (application.target == "magia-v3") {
    os << "_Static_assert(";
    emitL1DataBaseOffset(application, plan.hartid, os);
    os << " + " << plan.l1DataBytes
       << "u <= MAPS_V3_TASK_SCRATCH_OFFSET, "
          "\"FIFO and planned L1 data overlap task scratch\");\n\n";
  }
  os << "const fifo_tile_plan_t maps_application_tile_"
     << getPaddedPhysicalTileId(application, plan.hartid) << "_plan = {\n";
  os << "  .hartid = " << plan.hartid << "u,\n";
  os << "  .l1_data_base = ";
  emitL1DataBaseOffset(application, plan.hartid, os);
  os << ",\n";
  os << "  .ready_flags_base = MAPS_APPLICATION_READY_FLAGS_OFFSET,\n";
  os << "  .ready_flags_count = MAPS_APPLICATION_READY_FLAGS_WORDS,\n";
  os << "  .num_token_slots = MAPS_APPLICATION_NUM_TOKEN_SLOTS,\n";
  os << "  .fifo = {\n";
  os << "    .l1_offset_bytes = 0u,\n";
  os << "    .num_producers = " << plan.recvIndices.size() << "u,\n";
  os << "    .num_slots = MAPS_APPLICATION_NUM_TOKEN_SLOTS,\n";
  if (application.target == "magia-v3" && !plan.recvIndices.empty()) {
    os << "    .slot_data_sizes = maps_application_tile_" << plan.hartid
       << "_fifo_slot_data_sizes,\n";
    if (!plan.fifoSharedRingBytes.empty()) {
      os << "    .payload_rings = maps_application_tile_" << plan.hartid
         << "_fifo_payload_rings,\n";
      os << "    .num_shared_rings = " << plan.fifoSharedRingBytes.size()
         << "u,\n";
      os << "    .shared_ring_bytes = maps_application_tile_" << plan.hartid
         << "_fifo_shared_ring_bytes,\n";
    }
  }
  if (application.target != "magia-v3") {
    uint64_t slotDataSize = 0;
    for (unsigned recvIndex : plan.recvIndices)
      slotDataSize = std::max(
          slotDataSize, getFifoSlotDataSize(application, recvIndex));
    os << "    .slot_data_size = " << slotDataSize << "u,\n";
  }
  os << "  },\n";
  emitPlanArrayInitializer("slices", "slices", plan.sliceIndices.size(),
                           plan.hartid, os);
  emitPlanArrayInitializer("init_l2_reads", "init_l2_reads",
                           plan.initL2ReadIndices.size(), plan.hartid, os);
  emitPlanArrayInitializer("l2_reads", "l2_reads",
                           plan.l2ReadIndices.size(), plan.hartid, os);
  emitPlanArrayInitializer("recvs", "recvs", plan.recvIndices.size(),
                           plan.hartid, os);
  emitPlanArrayInitializer("ops", "ops", plan.opIndices.size(), plan.hartid,
                           os);
  emitPlanArrayInitializer("sends", "sends", plan.sendIndices.size(),
                           plan.hartid, os);
  emitPlanArrayInitializer("l2_writes", "l2_writes",
                           plan.l2WriteIndices.size(), plan.hartid, os);
  if (application.target == "magia-v3") {
    // Descriptors are copied to L1 right after the planned slice data.
    os << "  .descriptor_l1_offset = ";
    emitTileDescriptorL1Offset(application, plan, os);
    os << ",\n";
  }
  os << "};\n";
  if (application.target == "magia-v3") {
    os << "\n_Static_assert(";
    emitTileDescriptorL1Offset(application, plan, os);
    auto addArray = [&](StringRef suffix, size_t count) {
      if (count != 0)
        os << " + ((sizeof(maps_application_tile_" << plan.hartid << "_"
           << suffix << ") + 15u) & ~15u)";
    };
    addArray("slices", plan.sliceIndices.size());
    addArray("init_l2_reads", plan.initL2ReadIndices.size());
    addArray("l2_reads", plan.l2ReadIndices.size());
    addArray("recvs", plan.recvIndices.size());
    addArray("ops", plan.opIndices.size());
    addArray("sends", plan.sendIndices.size());
    addArray("l2_writes", plan.l2WriteIndices.size());
    addArray("fifo_slot_data_sizes", plan.recvIndices.size());
    os << " <= MAPS_V3_TASK_SCRATCH_OFFSET, "
          "\"L1 descriptor copy overlaps task scratch\");\n";
  }
}

void emitApplicationTileSource(const MagiaApplicationDescription &application,
                               const GeneratedTilePlan &plan,
                               llvm::raw_ostream &os) {
  os << emitWithApplicationSymbols(application, [&](llvm::raw_ostream &stream) {
    emitGeneratedPrelude(application, stream);
    emitL2GlobalDeclarations(application, stream);
    emitTileSliceDescriptors(application, plan, stream);
    emitTileL2ReadDescriptors(application, plan, stream);
    emitTileRecvDescriptors(application, plan, stream);
    emitFifoSlotDataSizes(application, plan, stream);
    emitTileOpDescriptors(application, plan, stream);
    emitTileSendDescriptors(application, plan, stream);
    emitTileL2WriteDescriptors(application, plan, stream);
    emitApplicationTilePlanConstructor(application, plan, stream);
  });
}

void emitApplicationRunner(const MagiaApplicationDescription &application,
                           llvm::raw_ostream &os) {
  SmallVector<GeneratedSpatzTask> spatzTasks =
      getRequiredSpatzTasks(application);
  os << "#include \"" << application.outputStem << ".h\"\n\n";
  if (!spatzTasks.empty())
    os << "#include \"" << application.outputStem << "_task_bin.h\"\n";
  os << "#include \"fsync.h\"\n";
  os << "#include \"redmule.h\"\n";
  os << "#include \"tile.h\"\n";
  os << "#include \"utils/maps_operations.h\"\n";
  os << "#include \"utils/maps_utils_v2.h\"\n\n";
  os << "_Static_assert(MAPS_OP_ABI_VERSION == "
     << kGeneratedOperationAbiVersion
     << "u, \"incompatible MAGIA operation ABI\");\n";
  if (!spatzTasks.empty()) {
    os << "_Static_assert(MAPS_KERNEL_ABI_VERSION == "
       << kGeneratedKernelAbiVersion
       << "u, \"incompatible MAGIA kernel ABI\");\n";
    os << "_Static_assert(MAPS_TASK_BUNDLE_ABI_VERSION == "
       << kGeneratedTaskBundleVersion
       << "u, \"incompatible Spatz task-bundle ABI\");\n\n";
    os << "_Static_assert(MAPS_OPERATION_TASK_SCRATCH_BYTES == "
       << kGeneratedTaskScratchBytes
       << "u, \"incompatible Spatz task scratch contract\");\n\n";
  }
  os << "#define " << uppercase(application.outputStem) << "_WAIT_MODE WFE\n\n";
  os << "extern void " << application.outputStem
     << "_initialize_data(uint32_t hartid);\n";
  os << "extern uint32_t " << application.outputStem << "_build_plan(\n";
  os << "    fifo_tile_plan_t *plan, uint32_t hartid,\n";
  os << "    maps_operation_runtime_t *runtime);\n\n";
  os << "int " << application.outputStem << "_run(void) {\n";
  os << "  const uint32_t hartid = get_hartid();\n";
  os << "  idma_init();\n";
  os << "  redmule_init();\n";
  os << "  fsync_init();\n";
  os << "  eu_init();\n";
  os << "  eu_clear_events(0xffffffffu);\n";
  os << "  eu_fsync_init(0u);\n";
  os << "  eu_idma_init(0u);\n";
  os << "  eu_redmule_init(0u);\n\n";
  os << "  maps_operation_runtime_t runtime;\n";
  if (!spatzTasks.empty()) {
    os << "  runtime.kernel_abi_version = " << kGeneratedKernelAbiVersion
       << "u;\n";
    os << "  runtime.task_bundle_abi_version = " << kGeneratedTaskBundleVersion
       << "u;\n";
    os << "  runtime.spatz_binary_start = SPATZ_BINARY_START;\n";
    os << "  runtime.spatz_params = (void *)(get_l1_base(hartid) + "
          "MAPS_OPERATION_TASK_SCRATCH_OFFSET);\n";
    os << "  runtime.spatz_params_bytes = MAPS_OPERATION_TASK_SCRATCH_BYTES;\n";
    os << "  runtime.spatz_initialized = 0u;\n";
    for (const GeneratedSpatzTask &task : spatzTasks)
      os << "  runtime." << task.runtimeField << " = " << task.symbol << ";\n";
  }
  os << "  fifo_tile_plan_t plan;\n";
  os << "  const uint32_t active = " << application.outputStem
     << "_build_plan(&plan, hartid, &runtime);\n";
  if (!spatzTasks.empty()) {
    os << "  if (maps_operation_runtime_init(&runtime) != 0)\n";
    os << "    return -1;\n";
  }
  os << "  if (active)\n";
  os << "    maps_fifo_init(&plan);\n\n";
  os << "  fsync_sync_level(MAX_SYNC_LVL - 1, 0);\n";
  os << "  eu_fsync_wait(" << uppercase(application.outputStem)
     << "_WAIT_MODE);\n";
  os << "  " << application.outputStem << "_initialize_data(hartid);\n";
  os << "  if (hartid == 0u) {\n";
  os << "    for (uint32_t token = 0u; token < "
     << uppercase(application.outputStem) << "_EXECUTION_TOKENS; ++token)\n";
  os << "      " << application.outputStem << "_handle_input(token);\n";
  os << "  }\n";
  os << "  fsync_sync_global();\n";
  os << "  eu_fsync_wait(" << uppercase(application.outputStem)
     << "_WAIT_MODE);\n";
  if (application.target == "magia-v3") {
    os << "  if (active)\n";
    os << "    maps_fifo_plan_to_l1(&plan);\n";
  }
  os << "  if (active)\n";
  os << "    maps_fifo_init_tile(&plan);\n";
  os << "  fsync_sync_global();\n";
  os << "  eu_fsync_wait(" << uppercase(application.outputStem)
     << "_WAIT_MODE);\n";
  os << "  if (active)\n";
  os << "    maps_fifo_run_tile_tokens(\n";
  os << "        &plan, " << uppercase(application.outputStem)
     << "_EXECUTION_TOKENS);\n";
  os << "  fsync_sync_global();\n";
  os << "  eu_fsync_wait(" << uppercase(application.outputStem)
     << "_WAIT_MODE);\n";
  os << "  if (active)\n";
  os << "    maps_fifo_flush_experiment_trace(\n";
  os << "        &plan, " << uppercase(application.outputStem)
     << "_EXECUTION_TOKENS);\n";
  os << "  if (hartid == 0u) {\n";
  os << "    for (uint32_t token = 0u; token < "
     << uppercase(application.outputStem) << "_EXECUTION_TOKENS; ++token)\n";
  os << "      " << application.outputStem << "_handle_output(token);\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";
  os << "int main(void) { return " << application.outputStem << "_run(); }\n";
}

void emitInitialApplicationSource(const MagiaApplicationDescription &application,
                                  llvm::raw_ostream &os) {
  os << "/* Only customize this source file. MAPS owns every other emitted "
        "file. */\n";
  os << "#include \"" << application.outputStem << ".h\"\n\n";
  os << "#include <stddef.h>\n";
  os << "#include \"utils/printf.h\"\n\n";
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.runtimeInputFile.empty())
      continue;
    os << "extern const uint8_t "
       << getRuntimeInputSymbol(application, global, "start") << "[];\n";
  }
  os << "\n";
  os << "void " << application.outputStem << "_handle_input(uint32_t token) {\n";
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Input)
      continue;
    std::string name = normalizeSnakeCase(global.globalName);
    os << "  uint8_t *" << name << " = " << application.outputStem << "_input_"
       << name << "(token);\n";
    if (!global.runtimeInputFile.empty()) {
      os << "  const uint8_t *" << name << "_data = "
         << getRuntimeInputSymbol(application, global, "start") << ";\n";
      os << "  for (size_t index = 0; index < " << global.tensorBytes
         << "u; ++index)\n";
      os << "    " << name << "[index] = " << name
         << "_data[index + token * " << global.tensorBytes << "u];\n";
    } else {
      os << "  for (size_t index = 0; index < " << global.tensorBytes
         << "u; ++index)\n";
      os << "    " << name << "[index] = (uint8_t)(index + token * 17u + "
         << global.globalId << "u * 31u);\n";
    }
  }
  os << "}\n\n";
  os << "void " << application.outputStem << "_handle_output(uint32_t token) {\n";
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Output)
      continue;
    std::string name = normalizeSnakeCase(global.globalName);
    os << "  const uint8_t *" << name << " = " << application.outputStem << "_output_"
       << name << "(token);\n";
    os << "  uint32_t " << name << "_checksum = 2166136261u;\n";
    os << "  for (size_t index = 0; index < " << global.tensorBytes
       << "u; ++index)\n";
    os << "    " << name << "_checksum = (" << name << "_checksum ^ " << name
       << "[index]) * 16777619u;\n";
    os << "  printf(\"" << application.outputStem << " token %u output "
       << escapePrintfText(global.globalName)
       << " checksum: %08x\\n\", token, " << name
       << "_checksum);\n";
  }
  os << "}\n";
}

void emitApplicationCMake(const MagiaApplicationDescription &application,
                          llvm::raw_ostream &os) {
  SmallVector<GeneratedSpatzTask> spatzTasks =
      getRequiredSpatzTasks(application);
  os << "set(MAPS_APPLICATION " << application.outputStem << ")\n\n";
  if (!spatzTasks.empty()) {
    os << "set(MAPS_SPATZ_KERNELS_DIR ${CMAKE_SOURCE_DIR}/kernels)\n";
    os << "add_spatz_task(\n";
    os << "  TEST_NAME ${MAPS_APPLICATION}\n";
    os << "  TASK_SOURCES\n";
    for (const GeneratedSpatzTask &task : spatzTasks)
      os << "    ${MAPS_SPATZ_KERNELS_DIR}/" << task.source << "\n";
    os << "  FIRST_TASK_NAME " << spatzTasks.front().name << "\n";
    os << "  INCLUDE_DIRS\n";
    for (const GeneratedSpatzTask &task : spatzTasks)
      os << "    ${MAPS_SPATZ_KERNELS_DIR}/" << task.includeDirectory << "\n";
    os << ")\n\n";
  }
  os << "add_executable(" << application.outputStem << "\n";
  os << "  src/application.c\n";
  os << "  src/" << application.outputStem << ".c\n";
  os << "  src/" << application.outputStem << "_runner.c\n";
  for (const GeneratedTilePlan *plan : getSortedTilePlans(application))
    os << "  src/tiles/" << getApplicationTileFilename(application, plan->hartid)
       << "\n";
  os << ")\n";
  os << "configure_file(\n";
  os << "  src/" << application.outputStem << "_initializers.S.in\n";
  os << "  ${CMAKE_CURRENT_BINARY_DIR}/" << application.outputStem
     << "_initializers.S\n";
  os << "  @ONLY\n";
  os << ")\n";
  os << "target_sources(" << application.outputStem
     << " PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/" << application.outputStem
     << "_initializers.S)\n";
  os << "target_include_directories(" << application.outputStem
     << " PUBLIC include)\n";
  os << "target_compile_options(" << application.outputStem << " PRIVATE -O2)\n";
  os << "set_property(SOURCE src/" << application.outputStem
     << "_runner.c APPEND PROPERTY COMPILE_OPTIONS -O2)\n";
  os << "target_link_libraries(" << application.outputStem
     << " PUBLIC runtime hal idma_lib redmule_lib fsync_lib eu_lib";
  if (application.target != "magia-v3")
    os << " kernels_redmule kernels_generic";
  os << ")\n";
  if (!spatzTasks.empty()) {
    os << "target_embed_spatz_task(\n";
    os << "  TARGET ${MAPS_APPLICATION}\n";
    os << "  TASK_NAME ${MAPS_APPLICATION}\n";
    os << ")\n";
    os << "target_compile_definitions(${MAPS_APPLICATION} PRIVATE\n";
    for (const GeneratedSpatzTask &task : spatzTasks)
      os << "  " << task.featureMacro << "=1\n";
    os << ")\n";
  }
}

void emitApplicationReadme(const MagiaApplicationDescription &application,
                           llvm::raw_ostream &os) {
  os << "# " << application.outputStem << "\n\n";
  os << "Generated from `" << application.sourceModelName
     << "` for the `" << application.target << "` Target and a "
     << application.meshWidth << " x "
     << application.meshHeight << " planned Mesh.\n\n";
  os << "Copy this directory into the MAGIA SDK application/test tree. Add "
        "the following line to its parent `CMakeLists.txt`:\n\n";
  os << "```cmake\nadd_subdirectory(" << application.outputStem << ")\n```\n\n";
  os << "Configure the SDK normally, then build this application with:\n\n";
  os << "```sh\ncmake --build build --target " << application.outputStem
     << "\n```\n\n";
  os << "Only `src/application.c` is developer-owned. Edit that file to "
        "replace the generated input and output hooks; all other files are "
        "generated by MAPS.\n";
}

void emitApplicationInitializersAssembly(const MagiaApplicationDescription &application,
                                         llvm::raw_ostream &os) {
  std::string start = application.outputStem + "_initializers_start";
  std::string end = application.outputStem + "_initializers_end";
  std::string size = application.outputStem + "_initializers_size";
  os << ".section "
     << (application.target == "magia-v3" ? ".l2_bulk.maps_initializers"
                                           : ".maps_initializers")
     << ",\"a\",@progbits\n";
  os << ".balign " << application.weightsAlignment << "\n";
  os << ".global " << start << "\n";
  os << ".global " << end << "\n";
  os << ".global " << size << "\n";
  os << start << ":\n";
  os << ".incbin \"@CMAKE_CURRENT_SOURCE_DIR@/data/" << application.outputStem
     << ".initializers.bin\"\n";
  os << end << ":\n";
  os << ".set " << size << ", " << end << " - " << start << "\n";
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.runtimeInputFile.empty())
      continue;
    start = getRuntimeInputSymbol(application, global, "start");
    end = getRuntimeInputSymbol(application, global, "end");
    size = getRuntimeInputSymbol(application, global, "size");
    os << "\n.section "
       << (application.target == "magia-v3" ? ".l2_bulk.maps_runtime_inputs"
                                             : ".maps_runtime_inputs")
       << ",\"a\",@progbits\n";
    os << ".balign " << kGeneratedL2AlignmentBytes << "\n";
    os << ".global " << start << "\n";
    os << ".global " << end << "\n";
    os << ".global " << size << "\n";
    os << start << ":\n";
    os << ".incbin \"@CMAKE_CURRENT_SOURCE_DIR@/data/"
       << global.runtimeInputAsset << "\"\n";
    os << end << ":\n";
    os << ".set " << size << ", " << end << " - " << start << "\n";
  }
}

} // namespace mlir::maps
