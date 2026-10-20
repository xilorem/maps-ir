#include "OpCollector.h"

#include "SliceCollector.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/Twine.h"

#include <optional>
#include <set>

namespace mlir::maps {

static std::optional<GeneratedOpKind> getGenericOpKind(Operation *genericOp) {
  std::optional<GeneratedOpKind> kind;

  genericOp->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "arith.addf")
      kind = GeneratedOpKind::Add;
    if (name == "math.log")
      kind = GeneratedOpKind::Log;
    if (name == "math.exp")
      kind = GeneratedOpKind::Exp;
  });

  return kind;
}

static std::optional<GeneratedOpKind> getKernelOpKind(StringRef kind) {
  if (kind == "MATMUL")
    return GeneratedOpKind::Matmul;
  if (kind == "CONV2D")
    return GeneratedOpKind::Conv2D;
  if (kind == "MAX_POOL")
    return GeneratedOpKind::MaxPool;
  if (kind == "DEPTHWISE_CONV")
    return GeneratedOpKind::DepthwiseConv;
  if (kind == "ABS")
    return GeneratedOpKind::Abs;
  if (kind == "ADD")
    return GeneratedOpKind::Add;
  if (kind == "DIV")
    return GeneratedOpKind::Div;
  if (kind == "EXP")
    return GeneratedOpKind::Exp;
  if (kind == "SOFTMAX_EXP")
    return GeneratedOpKind::SoftmaxExp;
  if (kind == "GROUP_NORMALIZE")
    return GeneratedOpKind::GroupNormalize;
  if (kind == "GROUP_REDUCE")
    return GeneratedOpKind::GroupReduce;
  if (kind == "GROUP_CENTERED_REDUCE")
    return GeneratedOpKind::GroupCenteredReduce;
  if (kind == "IM2COL")
    return GeneratedOpKind::Im2Col;
  if (kind == "OUTPUT_REFORMAT")
    return GeneratedOpKind::OutputReformat;
  if (kind == "SPLIT")
    return GeneratedOpKind::Split;
  if (kind == "LOG")
    return GeneratedOpKind::Log;
  if (kind == "MUL")
    return GeneratedOpKind::Mul;
  if (kind == "NEG")
    return GeneratedOpKind::Neg;
  if (kind == "POW")
    return GeneratedOpKind::Pow;
  if (kind == "REDUCE_MAX")
    return GeneratedOpKind::ReduceMax;
  if (kind == "REDUCE_SUM")
    return GeneratedOpKind::ReduceSum;
  if (kind == "RELU")
    return GeneratedOpKind::Relu;
  if (kind == "RESHAPE")
    return GeneratedOpKind::Reshape;
  if (kind == "SIGMOID")
    return GeneratedOpKind::Sigmoid;
  if (kind == "SLICE")
    return GeneratedOpKind::Slice;
  if (kind == "SQRT")
    return GeneratedOpKind::Sqrt;
  if (kind == "SUB")
    return GeneratedOpKind::Sub;
  if (kind == "TRANSPOSE")
    return GeneratedOpKind::Transpose;
  if (kind == "AllReduceMax")
    return GeneratedOpKind::AllReduceMax;
  if (kind == "AllReduceSum")
    return GeneratedOpKind::AllReduceSum;
  return std::nullopt;
}

static void fillElementwiseParams(GeneratedOp &generatedOp,
                                  RankedTensorType outputType) {
  ArrayRef<int64_t> shape = outputType.getShape();
  generatedOp.params[0] = shape.empty() ? 1u : static_cast<uint32_t>(shape[0]);
  generatedOp.params[1] =
      shape.size() < 2 ? 1u : static_cast<uint32_t>(shape[1]);
  generatedOp.params[2] = static_cast<uint32_t>(product(shape));
}

static LogicalResult fillCollectiveDescriptor(Operation *op,
                                              int64_t tileId,
                                              GeneratedOp &generatedOp) {
  auto stageId = op->getAttrOfType<IntegerAttr>("collective_stage_id");
  auto layerIndex = op->getAttrOfType<IntegerAttr>("collective_layer_index");
  auto groupId = op->getAttrOfType<IntegerAttr>("collective_group_id");
  auto participants =
      op->getAttrOfType<DenseI64ArrayAttr>("collective_participants");
  if (!stageId || !layerIndex || !groupId || !participants ||
      stageId.getInt() < 0 || layerIndex.getInt() < 0 || groupId.getInt() < 0)
    return failure();

  std::set<int64_t> uniqueParticipants;
  for (int64_t participant : participants.asArrayRef()) {
    if (participant < 0 || !uniqueParticipants.insert(participant).second)
      return failure();
    generatedOp.collectiveParticipants.push_back(
        static_cast<unsigned>(participant));
  }
  if (generatedOp.collectiveParticipants.empty() ||
      uniqueParticipants.find(tileId) == uniqueParticipants.end())
    return failure();

  generatedOp.isCollective = true;
  generatedOp.collectiveIdentity = {
      static_cast<unsigned>(stageId.getInt()),
      static_cast<unsigned>(layerIndex.getInt()),
      static_cast<unsigned>(groupId.getInt()),
  };
  return success();
}

LogicalResult collectComputeOp(Operation *op, RunCollectionState &state,
                               MagiaApplicationDescription &application) {
  StringRef name = op->getName().getStringRef();
  bool isMatmul = name == "linalg.matmul";
  bool isGeneric = name == "linalg.generic";
  bool isKernel = name == "maps.kernel";
  if (!isMatmul && !isGeneric && !isKernel)
    return success();

  Operation *tileOp = findParentNamed(op, "maps.tile");
  if (!tileOp)
    return failure();

  auto tileId = tileOp->getAttrOfType<IntegerAttr>("tile_id");
  if (!tileId)
    return failure();

  if (op->getNumResults() == 0)
    return failure();

  auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!outputType)
    return failure();

  GeneratedOp generatedOp;
  generatedOp.ownerHartid = tileId.getInt();
  if (auto device = op->getAttrOfType<StringAttr>("device_name"))
    generatedOp.deviceName = device.getValue().str();
  if (isKernel) {
    auto kind = op->getAttrOfType<StringAttr>("kind");
    auto params = op->getAttrOfType<DenseI64ArrayAttr>("params");
    auto generatedKind = kind ? getKernelOpKind(kind.getValue()) : std::nullopt;
    if (!generatedKind || !params || params.size() > 8)
      return failure();
    if (*generatedKind == GeneratedOpKind::Conv2D &&
        (op->getNumOperands() < 2 || op->getNumOperands() > 3 ||
         params.size() != 8))
      return failure();
    generatedOp.kind = *generatedKind;
    for (auto indexed : llvm::enumerate(params.asArrayRef()))
      generatedOp.params[indexed.index()] =
          static_cast<uint32_t>(indexed.value());
    if ((*generatedKind == GeneratedOpKind::AllReduceMax ||
         *generatedKind == GeneratedOpKind::AllReduceSum) &&
        failed(fillCollectiveDescriptor(op, tileId.getInt(), generatedOp)))
      return failure();
  } else if (isMatmul) {
    generatedOp.kind = GeneratedOpKind::Matmul;
    if (op->getNumOperands() < 2)
      return failure();
    auto lhsType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto rhsType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    if (!lhsType || !rhsType || lhsType.getRank() < 2 || rhsType.getRank() < 2)
      return failure();
    generatedOp.params[0] = static_cast<uint32_t>(lhsType.getShape()[0]);
    generatedOp.params[1] = static_cast<uint32_t>(lhsType.getShape()[1]);
    generatedOp.params[2] = static_cast<uint32_t>(rhsType.getShape()[1]);
  } else {
    std::optional<GeneratedOpKind> kind = getGenericOpKind(op);
    if (!kind)
      return success();
    generatedOp.kind = *kind;
    fillElementwiseParams(generatedOp, outputType);
  }

  unsigned numOutputs = op->getNumResults();
  unsigned numInputs = isKernel ? op->getNumOperands()
                                : op->getNumOperands() - numOutputs;
  for (unsigned i = 0; i < numInputs; ++i) {
    auto input = state.valueSubsliceIndices.find(op->getOperand(i));
    if (input == state.valueSubsliceIndices.end())
      return failure();
    generatedOp.inputSubsliceIndices.push_back(input->second);
  }

  for (unsigned resultIndex = 0; resultIndex < numOutputs; ++resultIndex) {
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(resultIndex).getType());
    if (!resultType)
      return failure();
    std::string valueKey = getProducedValueKey(state, op->getResult(resultIndex));
    std::string uniqueKey =
        (Twine("op:") + Twine(tileId.getInt()) + ":" + valueKey).str();
    std::string outputFullOffsetName =
        (Twine("output_full_offset_") + Twine(resultIndex)).str();
    SmallVector<int64_t> outputFullOffset =
        getDenseI64Array(op, outputFullOffsetName);
    unsigned outputSliceIndex = getOrCreateProducedSlice(
        application, state.producedSliceIndices, uniqueKey, valueKey,
        GeneratedGlobalKind::Intermediate, tileId.getInt(), resultType,
        outputFullOffset);
    if (outputSliceIndex == std::numeric_limits<unsigned>::max())
      return failure();

    unsigned outputSubsliceIndex =
        addWholeSubsliceForSlice(application, outputSliceIndex);
    generatedOp.outputSubsliceIndices.push_back(outputSubsliceIndex);
    state.valueSubsliceIndices[op->getResult(resultIndex)] = outputSubsliceIndex;
  }

  unsigned opIndex = addGeneratedOp(application, generatedOp);
  noteTile(application, tileId.getInt());
  application.tilePlans[tileId.getInt()].opIndices.push_back(opIndex);
  return success();
}

} // namespace mlir::maps
