#include "ApplicationDescription.h"
#include "collectors/CollectorSupport.h"

#include "llvm/ADT/StringExtras.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <utility>

namespace mlir::maps {

std::string normalizeSnakeCase(StringRef value) {
  std::string normalized;
  bool separator = false;
  bool previousWasLowerOrDigit = false;
  for (char character : value) {
    if (!llvm::isAlnum(character)) {
      separator = !normalized.empty();
      previousWasLowerOrDigit = false;
      continue;
    }
    bool uppercase = llvm::isUpper(character);
    if (!normalized.empty() &&
        (separator || (uppercase && previousWasLowerOrDigit)) &&
        normalized.back() != '_')
      normalized.push_back('_');
    normalized.push_back(llvm::toLower(character));
    separator = false;
    previousWasLowerOrDigit =
        llvm::isLower(character) || llvm::isDigit(character);
  }
  while (!normalized.empty() && normalized.back() == '_')
    normalized.pop_back();
  if (normalized.empty())
    return "application";
  if (llvm::isDigit(normalized.front()))
    normalized.insert(0, "application_");
  return normalized;
}

static unsigned getOrCreateGlobalId(MagiaApplicationDescription &application,
                                    StringRef globalName) {
  auto existing = application.globalIds.find(globalName);
  if (existing != application.globalIds.end())
    return existing->second;

  unsigned globalId = application.nextGlobalId++;
  application.globalIds[globalName] = globalId;
  return globalId;
}

static unsigned allocateSliceIdForHart(MagiaApplicationDescription &application,
                                       int64_t hartid) {
  unsigned sliceId = application.nextSliceIdByHart[hartid];
  application.nextSliceIdByHart[hartid] = sliceId + 1;
  return sliceId;
}


unsigned addGeneratedSlice(MagiaApplicationDescription &application, GeneratedSlice slice) {
  /*
  Add slice to plan
  */
  slice.descriptionIndex = application.slices.size();
  slice.globalId = getOrCreateGlobalId(application, slice.globalName);
  slice.sliceId = allocateSliceIdForHart(application, slice.ownerHartid);
  application.slices.push_back(std::move(slice));
  return application.slices.back().descriptionIndex;
}


unsigned addGeneratedSubslice(MagiaApplicationDescription &application, GeneratedSubslice subslice) {
  /*
  Add subslice to plan
  */
  subslice.descriptionIndex = application.subslices.size();
  application.subslices.push_back(std::move(subslice));
  return application.subslices.back().descriptionIndex;
}


GeneratedSubslice makeWholeSubslice(const GeneratedSlice &slice) {
  /*
  Create a subslice consisting of the full slice. Used for L2 to L1 transfers for example
  */

  GeneratedSubslice subslice;
  subslice.sliceIndex = slice.descriptionIndex;
  subslice.offset.assign(slice.shape.size(), 0);
  subslice.shape = slice.shape;
  subslice.elementType = slice.elementType;
  subslice.elemBytes = slice.elemBytes;
  subslice.stridesBytes = slice.stridesBytes;
  return subslice;
}

uint64_t computeOffsetBytes(ArrayRef<int64_t> offsets,
                            ArrayRef<uint64_t> stridesBytes) {
  uint64_t result = 0;
  for (size_t i = 0; i < offsets.size() && i < stridesBytes.size(); ++i)
    result += static_cast<uint64_t>(offsets[i]) * stridesBytes[i];
  return result;
}

unsigned addGeneratedL2Read(MagiaApplicationDescription &application, GeneratedL2Read read) {
  /*
  Add L2 read descriptor to plan
  */

  read.descriptionIndex = application.l2Reads.size();
  application.l2Reads.push_back(read);
  return application.l2Reads.back().descriptionIndex;

}


unsigned getOrCreateTransitionId(MagiaApplicationDescription &application, StringRef channel) {
  auto existing = application.transitionIds.find(channel);
  if (existing != application.transitionIds.end())
    return existing->second;

  unsigned transitionId = application.nextTransitionId++;
  application.transitionIds[channel] = transitionId;
  return transitionId;
}

unsigned addGeneratedRecv(MagiaApplicationDescription &application, GeneratedRecv recv) {
  recv.descriptionIndex = application.recvs.size();
  application.recvs.push_back(recv);
  return application.recvs.back().descriptionIndex;
}

unsigned addGeneratedSend(MagiaApplicationDescription &application, GeneratedSend send) {
  send.descriptionIndex = application.sends.size();
  application.sends.push_back(send);
  return application.sends.back().descriptionIndex;
}

unsigned addGeneratedOp(MagiaApplicationDescription &application, GeneratedOp op) {
  op.descriptionIndex = application.ops.size();
  application.ops.push_back(op);
  return application.ops.back().descriptionIndex;
}

SmallVector<GeneratedSpatzTask>
getRequiredSpatzTasks(const MagiaApplicationDescription &application) {
  if (application.target != "magia-v3")
    return {};
  bool needsAdd = false;
  bool needsMul = false;
  bool needsMatmul = false;
  bool needsRelu = false;
  bool needsSigmoid = false;
  bool needsMaxPool = false;
  bool needsDepthwiseConv = false;
  bool needsTranspose = false;
  bool needsSoftmaxExp = false;
  bool needsGroupReduce = false;
  bool needsGroupCenteredReduce = false;
  bool needsGroupNormalize = false;
  bool needsReduceSum = false;
  bool needsReduceMax = false;
  bool needsBinaryBcast = false;
  for (const GeneratedOp &op : application.ops) {
    needsSigmoid |= op.kind == GeneratedOpKind::Sigmoid;
    needsMaxPool |= op.kind == GeneratedOpKind::MaxPool;
    needsDepthwiseConv |= op.kind == GeneratedOpKind::DepthwiseConv;
    needsTranspose |= op.kind == GeneratedOpKind::Transpose &&
                      !op.inputSubsliceIndices.empty() &&
                      application.subslices[op.inputSubsliceIndices.front()]
                          .elementType.isF16();
    if (op.deviceName != "spatz")
      continue;
    needsAdd |= op.kind == GeneratedOpKind::Add;
    needsMul |= op.kind == GeneratedOpKind::Mul;
    needsMatmul |= op.kind == GeneratedOpKind::Matmul;
    needsRelu |= op.kind == GeneratedOpKind::Relu;
    needsSoftmaxExp |= op.kind == GeneratedOpKind::SoftmaxExp;
    needsGroupReduce |= op.kind == GeneratedOpKind::GroupReduce;
    needsGroupCenteredReduce |= op.kind == GeneratedOpKind::GroupCenteredReduce;
    needsGroupNormalize |= op.kind == GeneratedOpKind::GroupNormalize;
    needsReduceSum |= op.kind == GeneratedOpKind::ReduceSum ||
                      op.kind == GeneratedOpKind::AllReduceSum;
    needsReduceMax |= op.kind == GeneratedOpKind::ReduceMax ||
                      op.kind == GeneratedOpKind::AllReduceMax;
    needsBinaryBcast |= op.kind == GeneratedOpKind::Sub || op.kind == GeneratedOpKind::Div;
  }

  SmallVector<GeneratedSpatzTask> tasks;
  if (needsAdd)
    tasks.push_back({GeneratedOpKind::Add, "add_fp16_spatz_task",
                     "add/fp16/spatz/spatz_task/add_fp16_spatz_task.c", "add/fp16/spatz/include",
                     "ADD_FP16_SPATZ_TASK", "add_fp16_task",
                     "MAPS_HAS_ADD_SPATZ_TASK"});
  if (needsMul)
    tasks.push_back({GeneratedOpKind::Mul, "mul_bcast_fp16_spatz_task",
                     "mul/fp16/spatz/spatz_task/mul_bcast_fp16_spatz_task.c", "mul/fp16/spatz/include",
                     "MUL_BCAST_FP16_SPATZ_TASK", "mul_bcast_fp16_task",
                     "MAPS_HAS_MUL_SPATZ_TASK"});
  if (needsMatmul)
    tasks.push_back({GeneratedOpKind::Matmul, "matmul_rows_fp16_spatz_task",
                     "matmul/fp16/spatz/spatz_task/matmul_rows_fp16_spatz_task.c", "matmul/fp16/spatz/include",
                     "MATMUL_ROWS_FP16_SPATZ_TASK", "matmul_fp16_task",
                     "MAPS_HAS_MATMUL_SPATZ_TASK"});
  if (needsTranspose)
    tasks.push_back({GeneratedOpKind::Transpose, "transpose_strided_fp16_spatz_task",
                     "transpose/fp16/spatz/spatz_task/transpose_strided_fp16_spatz_task.c", "transpose/fp16/spatz/include",
                     "TRANSPOSE_STRIDED_FP16_SPATZ_TASK", "transpose_fp16_task",
                     "MAPS_HAS_TRANSPOSE_SPATZ_TASK"});
  if (needsMaxPool)
    tasks.push_back({GeneratedOpKind::MaxPool, "maxpool2d_fp16_spatz_task",
                     "maxpool2d/fp16/spatz/spatz_task/maxpool2d_fp16_spatz_task.c", "maxpool2d/fp16/spatz/include",
                     "MAXPOOL2D_FP16_SPATZ_TASK", "maxpool_fp16_task",
                     "MAPS_HAS_MAXPOOL_SPATZ_TASK"});
  if (needsDepthwiseConv)
    tasks.push_back({GeneratedOpKind::DepthwiseConv, "depthwise_conv_fp16_spatz_task",
                     "depthwise_conv/fp16/spatz/spatz_task/depthwise_conv_fp16_spatz_task.c",
                     "depthwise_conv/fp16/spatz/include",
                     "DEPTHWISE_CONV_FP16_SPATZ_TASK", "depthwise_conv_fp16_task",
                     "MAPS_HAS_DEPTHWISE_CONV_SPATZ_TASK"});
  if (needsSigmoid)
    tasks.push_back({GeneratedOpKind::Sigmoid, "sigmoid_fp16_spatz_task",
                     "sigmoid/fp16/spatz/spatz_task/sigmoid_fp16_spatz_task.c", "sigmoid/fp16/spatz/include",
                     "SIGMOID_FP16_SPATZ_TASK", "sigmoid_fp16_task",
                     "MAPS_HAS_SIGMOID_SPATZ_TASK"});
  if (needsRelu)
    tasks.push_back({GeneratedOpKind::Relu, "relu_fp16_spatz_task",
                     "relu/fp16/spatz/spatz_task/relu_fp16_spatz_task.c", "relu/fp16/spatz/include",
                     "RELU_FP16_SPATZ_TASK", "relu_fp16_task",
                     "MAPS_HAS_RELU_SPATZ_TASK"});
  if (needsSoftmaxExp)
    tasks.push_back({GeneratedOpKind::SoftmaxExp, "softmax_exp_fp16_spatz_task",
                     "softmax_exp/fp16/spatz/spatz_task/softmax_exp_fp16_spatz_task.c",
                     "softmax_exp/fp16/spatz/include", "SOFTMAX_EXP_FP16_SPATZ_TASK",
                     "softmax_exp_fp16_task", "MAPS_HAS_SOFTMAX_EXP_SPATZ_TASK"});
  if (needsGroupReduce)
    tasks.push_back({GeneratedOpKind::GroupReduce, "group_reduce_fp16_spatz_task",
                     "group_reduce/fp16/spatz/spatz_task/group_reduce_fp16_spatz_task.c",
                     "group_reduce/fp16/spatz/include", "GROUP_REDUCE_FP16_SPATZ_TASK",
                     "group_reduce_fp16_task", "MAPS_HAS_GROUP_REDUCE_SPATZ_TASK"});
  if (needsGroupCenteredReduce)
    tasks.push_back({GeneratedOpKind::GroupCenteredReduce,
                     "group_centered_reduce_fp16_spatz_task",
                     "group_centered_reduce/fp16/spatz/spatz_task/group_centered_reduce_fp16_spatz_task.c",
                     "group_centered_reduce/fp16/spatz/include",
                     "GROUP_CENTERED_REDUCE_FP16_SPATZ_TASK",
                     "group_centered_reduce_fp16_task",
                     "MAPS_HAS_GROUP_CENTERED_REDUCE_SPATZ_TASK"});
  if (needsGroupNormalize)
    tasks.push_back({GeneratedOpKind::GroupNormalize,
                     "group_normalize_fp16_spatz_task",
                     "group_normalize/fp16/spatz/spatz_task/group_normalize_fp16_spatz_task.c",
                     "group_normalize/fp16/spatz/include", "GROUP_NORMALIZE_FP16_SPATZ_TASK",
                     "group_normalize_fp16_task",
                     "MAPS_HAS_GROUP_NORMALIZE_SPATZ_TASK"});
  if (needsReduceSum)
    tasks.push_back({GeneratedOpKind::ReduceSum, "reducesum_fp16_spatz_task",
                     "reducesum/fp16/spatz/spatz_task/reducesum_fp16_spatz_task.c",
                     "reducesum/fp16/spatz/include", "REDUCESUM_FP16_SPATZ_TASK",
                     "reducesum_fp16_task", "MAPS_HAS_REDUCE_SUM_SPATZ_TASK"});
  if (needsReduceMax)
    tasks.push_back({GeneratedOpKind::AllReduceMax, "reducemax_fp16_spatz_task",
                     "reducemax/fp16/spatz/spatz_task/reducemax_fp16_spatz_task.c",
                     "reducemax/fp16/spatz/include", "REDUCEMAX_FP16_SPATZ_TASK",
                     "reducemax_fp16_task", "MAPS_HAS_REDUCE_MAX_SPATZ_TASK"});
  if (needsBinaryBcast)
    tasks.push_back({GeneratedOpKind::Sub, "binary_bcast_fp16_spatz_task",
                     "binary_bcast/fp16/spatz/spatz_task/binary_bcast_fp16_spatz_task.c",
                     "binary_bcast/fp16/spatz/include", "BINARY_BCAST_FP16_SPATZ_TASK",
                     "binary_bcast_fp16_task", "MAPS_HAS_BINARY_BCAST_SPATZ_TASK"});
  return tasks;
}

LogicalResult validateSpatzTaskCoverage(
    const MagiaApplicationDescription &application) {
  if (application.target != "magia-v3")
    return success();
  for (const GeneratedOp &op : application.ops) {
    if (op.deviceName == "spatz" && op.kind != GeneratedOpKind::Add &&
        op.kind != GeneratedOpKind::Mul &&
        op.kind != GeneratedOpKind::Matmul &&
        op.kind != GeneratedOpKind::Relu &&
        op.kind != GeneratedOpKind::SoftmaxExp &&
        op.kind != GeneratedOpKind::GroupReduce &&
        op.kind != GeneratedOpKind::GroupCenteredReduce &&
        op.kind != GeneratedOpKind::GroupNormalize &&
        op.kind != GeneratedOpKind::ReduceSum &&
        op.kind != GeneratedOpKind::ReduceMax &&
        op.kind != GeneratedOpKind::AllReduceMax &&
        op.kind != GeneratedOpKind::AllReduceSum && op.kind != GeneratedOpKind::Sub &&
        op.kind != GeneratedOpKind::Div) {
      llvm::errs() << "MAGIA-v3 has no Tile-Local Spatz Task for "
                   << static_cast<unsigned>(op.kind) << "\n";
      return failure();
    }
  }
  return success();
}

unsigned addGeneratedL2Write(MagiaApplicationDescription &application, GeneratedL2Write write) {
  write.descriptionIndex = application.l2Writes.size();
  application.l2Writes.push_back(write);
  return application.l2Writes.back().descriptionIndex;
}

static std::string getL2GlobalKey(GeneratedGlobalKind globalKind,
                                  StringRef globalName) {
  return (Twine(static_cast<int>(globalKind)) + ":" + globalName).str();
}


void noteTile(MagiaApplicationDescription &application, int64_t hartid) {
  /*
  Add tile to plan
  */

  auto &plan = application.tilePlans[hartid];
  plan.hartid = hartid;
}


uint64_t alignTo(uint64_t value, uint64_t alignment) {
  /*
  Consider L1 allignment
  */
  return ((value + alignment - 1) / alignment) * alignment;
}


uint64_t product(ArrayRef<int64_t> values) {
  uint64_t result = 1;
  for (int64_t value : values)
    result *= static_cast<uint64_t>(value);
  return result;
}


uint64_t computeSlotBytes(const GeneratedSlice &slice) {
  /*
  number of bytes for a slot
  */
  return product(slice.shape) * slice.elemBytes;
}


unsigned getElementByteWidth(Type type) {
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isF16())
    return 2;
  if (type.isInteger(8))
    return 1;
  return 0;
}


SmallVector<uint64_t> computeContiguousStridesBytes(ArrayRef<int64_t> shape, unsigned elemBytes) {
  /*
  Returns the number of bytes per strided dimension considering the last dimension as the contiguous dimension

  e.g.
  shape = [3, 4, 5]
  elemBytes = 2

  dim 2 stride = 2
  dim 1 stride = 5 * 2 = 10
  dim 0 stride = 4 * 10 = 40

  result = [40, 10, 2]

  */
  SmallVector<uint64_t> strides(shape.size(), elemBytes);
  if (shape.empty())
    return strides;

  for (int64_t dim = static_cast<int64_t>(shape.size()) - 2; dim >= 0; --dim)
    strides[dim] = strides[dim + 1] * static_cast<uint64_t>(shape[dim + 1]);

  return strides;
}


LogicalResult getOrCreateL2Global(MagiaApplicationDescription &application,
                                  StringRef globalName,
                                  GeneratedGlobalKind globalKind,
                                  RankedTensorType tensorType,
                                  unsigned numTokens,
                                  unsigned &l2GlobalIndex,
                                  const L2SourceInfo *sourceInfo) {

  /*
  check if an L2 allocation exists for the current global tensor. If not, create it
  */
  
  // build a unique key (for example, Input:arg1)
  std::string key = getL2GlobalKey(globalKind, globalName);
  
  // check if key exists, if it already exists, use it
  auto existing = application.l2GlobalIndices.find(key);
  if (existing != application.l2GlobalIndices.end()) {
    l2GlobalIndex = existing->second;
    return success();
  }

  // else
  // get bytes per element
  unsigned elemBytes = getElementByteWidth(tensorType.getElementType());

  // Create the global allocation
  GeneratedL2Global global;
  global.descriptionIndex = application.l2Globals.size();
  global.globalId = getOrCreateGlobalId(application, globalName);
  global.globalName = globalName.str();
  global.globalKind = globalKind;
  global.shape.append(tensorType.getShape().begin(), tensorType.getShape().end());
  global.elementType = tensorType.getElementType();
  global.elemBytes = elemBytes;
  global.tensorBytes = product(global.shape) * elemBytes;
  global.tokenStrideBytes = alignTo(global.tensorBytes, kGeneratedL2AlignmentBytes);
  global.numTokens = numTokens;
  global.stridesBytes = computeContiguousStridesBytes(global.shape, global.elemBytes);

  if (globalKind == GeneratedGlobalKind::Initializer && sourceInfo &&
      sourceInfo->isConstant) {
    global.isConstant = true;
    global.packedImageOffset = sourceInfo->packedImageOffset;
    global.packedByteSize = sourceInfo->packedByteSize;
    global.packedDtype = sourceInfo->packedDtype;
    global.packedShape = sourceInfo->packedShape;
    global.requiredAlignment = application.weightsAlignment;

    if (!application.hasPackedWeights ||
        global.packedImageOffset % global.requiredAlignment != 0 ||
        global.packedImageOffset + global.packedByteSize > application.weightsSize ||
        global.packedShape != global.shape ||
        global.packedByteSize != global.tensorBytes)
      return failure();

    global.l2Address = application.l2Base + global.packedImageOffset;
    application.l2GlobalIndices[key] = global.descriptionIndex;
    application.l2Globals.push_back(std::move(global));
    l2GlobalIndex = application.l2Globals.back().descriptionIndex;
    return success();
  }

  if (globalKind == GeneratedGlobalKind::Initializer &&
      (application.hasPackedWeights || !application.randomizeInputs))
    return failure();

  // allign
  application.nextL2Address = alignTo(application.nextL2Address,
                                kGeneratedL2AlignmentBytes);
  global.l2Address = application.nextL2Address;

  // Return error if memory is exceeded
  uint64_t allocationBytes = global.tokenStrideBytes * global.numTokens;
  if (global.l2Address + allocationBytes > application.l2End) {
    llvm::errs() << "L2 runtime allocation for '" << globalName << "' needs "
                 << allocationBytes << " bytes, exceeding remaining runtime capacity "
                 << application.l2End - global.l2Address << " bytes\n";
    return failure();
  }

  // set starting adress for next allocation
  application.nextL2Address = global.l2Address + allocationBytes;

  // Advance the allocator
  application.l2GlobalIndices[key] = global.descriptionIndex;
  application.l2Globals.push_back(std::move(global));
  l2GlobalIndex = application.l2Globals.back().descriptionIndex;

  return success();

}


LogicalResult allocateSliceInTileL1(MagiaApplicationDescription &application, unsigned sliceIndex) {
  /*
  Allocates space for a tile slice in tile's L1
  */
  GeneratedSlice &slice = application.slices[sliceIndex];

  noteTile(application, slice.ownerHartid);
  GeneratedTilePlan &plan = application.tilePlans[slice.ownerHartid];

  // Get the amount of bytes the slice needs
  slice.slotBytes = computeSlotBytes(slice);

  // Consider L1 allignment
  plan.nextL1OffsetBytes = alignTo(plan.nextL1OffsetBytes, kGeneratedL1AlignmentBytes);

  // get the adress for next tile's starting point
  slice.l1OffsetBytes = plan.nextL1OffsetBytes;

  // Guard for out-of-memory
  uint64_t allocationBytes = slice.slotBytes * slice.numSlots;
  uint64_t endOffset = slice.l1OffsetBytes + allocationBytes;
  if (plan.l1DataBaseOffset + endOffset > kGeneratedReadyFlagsOffset)
    return failure();

  // Populate plan with info derived
  plan.nextL1OffsetBytes = endOffset;
  plan.l1DataBytes = std::max(plan.l1DataBytes, endOffset);
  plan.sliceIndices.push_back(sliceIndex);
  return success();

}

SmallVector<const GeneratedTilePlan *>
getSortedTilePlans(const MagiaApplicationDescription &application) {
  SmallVector<const GeneratedTilePlan *> plans;
  plans.reserve(application.tilePlans.size());
  for (const auto &entry : application.tilePlans)
    plans.push_back(&entry.second);
  llvm::sort(plans, [](const GeneratedTilePlan *lhs,
                       const GeneratedTilePlan *rhs) {
    return lhs->hartid < rhs->hartid;
  });
  return plans;
}

} // namespace mlir::maps
