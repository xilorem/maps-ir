#include "maps/Conversion/MapsToD2M/ComputeOpRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include <array>

namespace mlir::maps {
namespace {

class ExpComputeEmitter final : public ComputeOpEmitter {
public:
  bool match(Value value) const override {
    return isa_and_nonnull<math::ExpOp>(value.getDefiningOp());
  }

  FailureOr<std::optional<Value>>
  lower(Value value, RankedTensorType outputType,
        tensor::ExtractSliceOp outputSlice,
        ComputeOpLoweringContext &ctx) const override {
    auto exp = cast<math::ExpOp>(value.getDefiningOp());
    auto loweredInput = ctx.lowerValue(exp.getOperand(), outputType, outputSlice);
    if (failed(loweredInput) || !*loweredInput)
      return loweredInput;

    auto expGeneric =
        ctx.builder.createExpGeneric(exp.getLoc(), **loweredInput, outputType);
    if (failed(expGeneric))
      return failure();
    return std::optional<Value>(expGeneric->getResult(0));
  }
};

class MatmulComputeEmitter final : public ComputeOpEmitter {
public:
  bool match(Value value) const override {
    return isa_and_nonnull<linalg::MatmulOp>(value.getDefiningOp());
  }

  FailureOr<std::optional<Value>>
  lower(Value value, RankedTensorType outputType,
        tensor::ExtractSliceOp outputSlice,
        ComputeOpLoweringContext &ctx) const override {
    auto matmul = cast<linalg::MatmulOp>(value.getDefiningOp());
    auto lhsType = cast<RankedTensorType>(matmul.getInputs()[0].getType());
    auto rhsType = cast<RankedTensorType>(matmul.getInputs()[1].getType());
    auto loweredLhs =
        ctx.lowerValue(matmul.getInputs()[0], lhsType, tensor::ExtractSliceOp());
    auto loweredRhs = ctx.lowerValue(matmul.getInputs()[1], rhsType, outputSlice);
    if (failed(loweredLhs) || failed(loweredRhs) || !*loweredLhs || !*loweredRhs)
      return std::optional<Value>();

    auto matmulGeneric =
        ctx.builder.createMatmulGeneric(value.getLoc(), **loweredLhs, **loweredRhs,
                                        outputType);
    if (failed(matmulGeneric))
      return failure();
    return std::optional<Value>(matmulGeneric->getResult(0));
  }
};

class LinalgGenericComputeEmitter final : public ComputeOpEmitter {
public:
  bool match(Value value) const override {
    return isa_and_nonnull<linalg::GenericOp>(value.getDefiningOp());
  }

  FailureOr<std::optional<Value>>
  lower(Value value, RankedTensorType outputType,
        tensor::ExtractSliceOp outputSlice,
        ComputeOpLoweringContext &ctx) const override {
    auto generic = cast<linalg::GenericOp>(value.getDefiningOp());
    SmallVector<Value> loweredInputs;
    loweredInputs.reserve(generic.getNumDpsInputs());
    for (Value input : generic.getDpsInputs()) {
      auto loweredInput = ctx.lowerValue(
          input, cast<RankedTensorType>(input.getType()), tensor::ExtractSliceOp());
      if (failed(loweredInput) || !*loweredInput)
        return loweredInput;
      loweredInputs.push_back(**loweredInput);
    }

    auto loweredGeneric = ctx.builder.createLinalgGeneric(
        value.getLoc(), loweredInputs, outputType, generic);
    if (failed(loweredGeneric))
      return failure();
    return std::optional<Value>(loweredGeneric->getResult(0));
  }
};

class RootAddSendEmitter final : public SendValueEmitter {
public:
  bool match(Value value) const override {
    return isa_and_nonnull<arith::AddFOp>(value.getDefiningOp());
  }

  FailureOr<std::optional<SmallVector<Value>>>
  lower(Value value, RankedTensorType outputType,
        tensor::ExtractSliceOp outputSlice,
        ComputeOpLoweringContext &ctx) const override {
    auto add = cast<arith::AddFOp>(value.getDefiningOp());
    auto concat = add.getLhs().getDefiningOp<tensor::ConcatOp>();
    Value bias = add.getRhs();
    if (!concat) {
      concat = add.getRhs().getDefiningOp<tensor::ConcatOp>();
      bias = add.getLhs();
    }

    if (!concat) {
      auto lhsType = cast<RankedTensorType>(add.getLhs().getType());
      auto rhsType = cast<RankedTensorType>(add.getRhs().getType());
      auto lhsValue = ctx.lowerValue(add.getLhs(), lhsType, tensor::ExtractSliceOp());
      auto rhsValue = ctx.lowerValue(add.getRhs(), rhsType, tensor::ExtractSliceOp());
      if (failed(lhsValue) || failed(rhsValue) || !*lhsValue || !*rhsValue)
        return std::optional<SmallVector<Value>>();

      auto addGeneric = ctx.builder.createAddGeneric(add.getLoc(), **lhsValue,
                                                     **rhsValue, outputType);
      if (failed(addGeneric))
        return failure();
      return std::optional<SmallVector<Value>>(
          SmallVector<Value>{addGeneric->getResult(0)});
    }

    SmallVector<Value> shardResults;
    int64_t dim = static_cast<int64_t>(concat.getDim());
    int64_t offset = 0;
    auto biasSource = ctx.getSourceValue(bias);
    if (failed(biasSource))
      return std::optional<SmallVector<Value>>();

    for (Value concatInput : concat.getInputs()) {
      auto logicalInputType = cast<RankedTensorType>(concatInput.getType());
      auto concatInputValue =
          ctx.lowerValue(concatInput, logicalInputType, tensor::ExtractSliceOp());
      if (failed(concatInputValue) || !*concatInputValue)
        return std::optional<SmallVector<Value>>();

      SmallVector<int64_t> staticOffsets(logicalInputType.getRank(), 0);
      SmallVector<int64_t> staticSizes(logicalInputType.getShape().begin(),
                                       logicalInputType.getShape().end());
      staticOffsets[dim] = offset;
      offset += logicalInputType.getDimSize(dim);

      Value biasSlice = ctx.createStaticSlice(add.getLoc(), *biasSource,
                                              staticOffsets, staticSizes);
      biasSlice = ctx.materializeContiguous(add.getLoc(), biasSlice);
      auto tiledBias = ctx.builder.getTiledValue(biasSlice);
      if (failed(tiledBias))
        return add.emitOpError("failed to materialize bias shard");

      auto addGeneric = ctx.builder.createAddGeneric(add.getLoc(), **concatInputValue,
                                                     *tiledBias,
                                                     logicalInputType);
      if (failed(addGeneric))
        return failure();
      shardResults.push_back(addGeneric->getResult(0));
    }

    return std::optional<SmallVector<Value>>(shardResults);
  }
};

ArrayRef<const ComputeOpEmitter *> getComputeEmitters() {
  static const ExpComputeEmitter expEmitter;
  static const MatmulComputeEmitter matmulEmitter;
  static const LinalgGenericComputeEmitter genericEmitter;
  static const std::array<const ComputeOpEmitter *, 3> emitters = {
      &expEmitter, &matmulEmitter, &genericEmitter};
  return emitters;
}

ArrayRef<const SendValueEmitter *> getSendValueEmitters() {
  static const RootAddSendEmitter addEmitter;
  static const std::array<const SendValueEmitter *, 1> emitters = {&addEmitter};
  return emitters;
}

} // namespace

FailureOr<std::optional<Value>>
lowerRegisteredComputeOp(Value value, RankedTensorType outputType,
                         tensor::ExtractSliceOp outputSlice,
                         ComputeOpLoweringContext &ctx) {
  for (const ComputeOpEmitter *emitter : getComputeEmitters()) {
    if (!emitter->match(value))
      continue;
    return emitter->lower(value, outputType, outputSlice, ctx);
  }
  return failure();
}

FailureOr<std::optional<SmallVector<Value>>>
lowerRegisteredSendValue(Value value, RankedTensorType outputType,
                         tensor::ExtractSliceOp outputSlice,
                         ComputeOpLoweringContext &ctx) {
  if (auto loweredOp =
          lowerRegisteredComputeOp(value, outputType, outputSlice, ctx);
      succeeded(loweredOp) && *loweredOp) {
    return std::optional<SmallVector<Value>>(SmallVector<Value>{**loweredOp});
  }

  for (const SendValueEmitter *emitter : getSendValueEmitters()) {
    if (!emitter->match(value))
      continue;
    return emitter->lower(value, outputType, outputSlice, ctx);
  }

  return failure();
}

} // namespace mlir::maps
