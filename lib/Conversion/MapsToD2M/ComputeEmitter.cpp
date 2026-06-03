#include "maps/Conversion/MapsToD2M/ComputeEmitter.h"

#include "maps/Conversion/MapsToD2M/ComputeOpRegistry.h"
#include "maps/Conversion/MapsToD2M/D2MGenericBuilder.h"
#include "maps/Conversion/MapsToD2M/D2MTypeUtils.h"
#include "maps/Conversion/MapsToD2M/SpatialEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <functional>

namespace mlir::maps {
namespace {

static bool isResidualComputeProducer(Value value) {
  // check if the value is produced by an operation that we consider compute
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return false;

  return isa<linalg::MatmulOp, linalg::GenericOp, math::ExpOp,
             arith::AddFOp>(definingOp);
}

static bool isHoistableNonComputeOp(Operation *op) {
  return isa<tensor::ExtractSliceOp, tensor::EmptyOp, tensor::InsertSliceOp,
             tensor::ConcatOp, arith::ConstantOp>(op);
}

static Value createDirectStaticSlice(OpBuilder &builder, Location loc, Value value,
                                     ArrayRef<int64_t> offsets,
                                     ArrayRef<int64_t> sizes) {
  SmallVector<int64_t> composedOffsets(offsets.begin(), offsets.end());
  SmallVector<int64_t> strides(offsets.size(), 1);

  while (auto parentSlice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
    auto parentOffsets = parentSlice.getStaticOffsets();
    auto parentStrides = parentSlice.getStaticStrides();
    for (size_t i = 0; i < composedOffsets.size(); ++i)
      composedOffsets[i] =
          parentOffsets[i] + composedOffsets[i] * parentStrides[i];
    value = parentSlice.getSource();
  }

  SmallVector<OpFoldResult> mixedOffsets;
  SmallVector<OpFoldResult> mixedSizes;
  SmallVector<OpFoldResult> mixedStrides;
  for (int64_t offset : composedOffsets)
    mixedOffsets.push_back(builder.getIndexAttr(offset));
  for (int64_t size : sizes)
    mixedSizes.push_back(builder.getIndexAttr(size));
  for (int64_t stride : strides)
    mixedStrides.push_back(builder.getIndexAttr(stride));

  return builder.create<tensor::ExtractSliceOp>(loc, value, mixedOffsets,
                                                mixedSizes, mixedStrides);
}

static Value materializeContiguousTensor(OpBuilder &builder, Location loc,
                                         Value value) {
  auto type = cast<RankedTensorType>(value.getType());
  Value output = builder.create<tensor::EmptyOp>(loc, type.getShape(),
                                                 type.getElementType());
  SmallVector<OpFoldResult> offsets(type.getRank(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides(type.getRank(), builder.getIndexAttr(1));
  for (int64_t size : type.getShape())
    sizes.push_back(builder.getIndexAttr(size));
  return builder.create<tensor::InsertSliceOp>(loc, value, output, offsets,
                                               sizes, strides);
}

static void eraseBlockOps(PatternRewriter &rewriter, Block &body) {
  SmallVector<Operation *> nestedOps;
  for (Operation &nested : llvm::make_early_inc_range(body))
    nestedOps.push_back(&nested);

  for (Operation *nested : llvm::reverse(nestedOps)) {
    nested->dropAllDefinedValueUses();
    rewriter.eraseOp(nested);
  }
}

} // namespace

FailureOr<bool> lowerComputeTileProgram(
    maps::TileOp tile, DenseMap<Attribute, ChannelInfo> &channels,
    DenseMap<Attribute, Value> &channelValues,
    DenseMap<Attribute, Value> &logicalChannelValues,
    DenseMap<Attribute, SmallVector<Value>> &channelValueLists,
    DenseMap<Attribute, Value> &slotValues,
    DenseMap<Operation *, mlir::tt::ttcore::CoreRangeAttr> &tileCoreRanges,
    DenseMap<int64_t, SmallVector<Attribute>> &stageChannels,
    DenseMap<Operation *, int64_t> &tileStageIds, PatternRewriter &rewriter) {

  SmallVector<maps::SendOp> sends; // all compute tiles should have at least one send op in the top level

  // collect tile outs
  for (Operation &nested : tile.getBody().front()) {
    if (auto send = dyn_cast<maps::SendOp>(nested))
      sends.push_back(send);
  }

  auto getSourceValue = [&](Value value) -> FailureOr<Value> {
    /* small lambda for resolving the source for a given value;
    It handles three cases
    - value comes from a maps.recv from another tile: we look for the the corresponding channel info and return the value that will be used to send the data from the other tile to this tile
    - value comes from a maps.recv from the host: we look for the corresponding channel info and return the send value that will be used to send the data from the host to the device
    - value comes from a maps.load: we look for the corresponding slot value
    */

    // comes from maps.recv case
    if (auto recv = value.getDefiningOp<maps::RecvOp>()) {
      auto channelIt = channels.find(recv.getChannelAttr());
      if (channelIt == channels.end())
        return recv.emitOpError("missing channel info");
      
      // comes from host case
      if (channelIt->second.srcHartId == -1)
        return channelIt->second.send.getValue();

      // comes from another tile case
      auto valueIt = channelValues.find(recv.getChannelAttr());
      if (valueIt == channelValues.end())
        return failure();
      return valueIt->second;
    }

    // comes from maps.load case
    if (auto load = value.getDefiningOp<maps::LoadOp>()) {
      auto slotIt = slotValues.find(load.getSlotAttr());
      if (slotIt == slotValues.end())
        return load.emitOpError("expected lowered slot value");
      return slotIt->second;
    }

    return value;
  };

  DenseMap<Value, Value> hoistedValues; // value map to avoid recloning the same value multiple times when hoisting
  std::function<FailureOr<Value>(Value)> hoistValue = [&](Value value) -> FailureOr<Value> {
    if (auto recv = value.getDefiningOp<maps::RecvOp>()) {
      auto channelIt = channels.find(recv.getChannelAttr());
      if (channelIt == channels.end())
        return recv.emitOpError("missing channel info");
      if (channelIt->second.srcHartId == -1)
        return channelIt->second.send.getValue();

      auto valueIt = logicalChannelValues.find(recv.getChannelAttr());
      if (valueIt == logicalChannelValues.end())
        return failure();
      return valueIt->second;
    }

    if (auto load = value.getDefiningOp<maps::LoadOp>()) {
      auto slotIt = slotValues.find(load.getSlotAttr());
      if (slotIt == slotValues.end())
        return load.emitOpError("expected lowered slot value");
      return slotIt->second;
    }

    auto cached = hoistedValues.find(value);
    if (cached != hoistedValues.end())
      return cached->second;

    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || definingOp->getBlock() != &tile.getBody().front())
      return value;
    if (!isHoistableNonComputeOp(definingOp)) {
      definingOp->emitOpError(
          "unsupported producer escaped D2M lowering; compute values must "
          "lower through registered emitters instead of being hoisted");
      return failure();
    }

    IRMapping mapping;
    for (Value operand : definingOp->getOperands()) {
      auto hoistedOperand = hoistValue(operand);
      if (failed(hoistedOperand))
        return failure();
      mapping.map(operand, *hoistedOperand);
    }

    rewriter.setInsertionPoint(tile);
    Operation *cloned = rewriter.clone(*definingOp, mapping);
    for (auto [original, replacement] :
         llvm::zip_equal(definingOp->getResults(), cloned->getResults())) {
      hoistedValues[original] = replacement;
    }
    return hoistedValues.lookup(value);
  };

  D2MGenericBuilder builder(rewriter.getContext(), rewriter);
  auto getLoweredInputValue = [&](Value value) -> FailureOr<Value> {
    if (auto recv = value.getDefiningOp<maps::RecvOp>()) {
      auto channelIt = channels.find(recv.getChannelAttr());
      if (channelIt == channels.end())
        return recv.emitOpError("missing channel info");
      if (channelIt->second.srcHartId == -1)
        return channelIt->second.send.getValue();

      auto valueIt = channelValues.find(recv.getChannelAttr());
      if (valueIt == channelValues.end())
        return failure();
      return valueIt->second;
    }

    if (auto load = value.getDefiningOp<maps::LoadOp>()) {
      auto slotIt = slotValues.find(load.getSlotAttr());
      if (slotIt == slotValues.end())
        return load.emitOpError("expected lowered slot value");
      return slotIt->second;
    }

    return value;
  };
  auto wrapIfPlaced = [&](Value value) -> FailureOr<Value> {
    auto generic = value.getDefiningOp<mlir::tt::d2m::GenericOp>();
    if (!generic)
      return value;
    auto coreRangeIt = tileCoreRanges.find(tile.getOperation());
    if (coreRangeIt == tileCoreRanges.end())
      return tile.emitOpError("missing placement for compute tile");
    return wrapGenericInSpatial(generic, coreRangeIt->second, rewriter);
  };
  auto materializeLogicalValue = [&](Location loc, Value value,
                                     RankedTensorType logicalType)
      -> FailureOr<Value> {
    auto valueType = dyn_cast<RankedTensorType>(value.getType());
    if (!valueType)
      return failure();
    if (!valueType.getEncoding())
      return value;

    auto logicalOutput = rewriter.create<mlir::tt::d2m::EmptyOp>(
        loc, logicalType.getShape(), logicalType.getElementType(),
        logicalType.getEncoding());
    auto toLayout = rewriter.create<mlir::tt::d2m::ToLayoutOp>(
        loc, value, logicalOutput.getResult());
    return toLayout.getResult(0);
  };
  using OptionalValue = std::optional<Value>;
  std::function<FailureOr<OptionalValue>(
      Value, RankedTensorType, tensor::ExtractSliceOp)> lowerValue;

  lowerValue = [&](Value value, RankedTensorType outputType,
                   tensor::ExtractSliceOp outputSlice)
      -> FailureOr<OptionalValue> {
    if (auto recv = value.getDefiningOp<maps::RecvOp>()) {
      auto sourceValue = getLoweredInputValue(recv.getResult());
      if (failed(sourceValue))
        return OptionalValue();
      if (auto tiledType =
              dyn_cast<RankedTensorType>((*sourceValue).getType());
          tiledType && tiledType.getEncoding() &&
          isa<mlir::tt::ttcore::TileType>(tiledType.getElementType())) {
        return OptionalValue(*sourceValue);
      }
      auto tiledInput = builder.getTiledValue(*sourceValue);
      if (failed(tiledInput))
        return recv.emitOpError("failed to materialize tiled recv input");
      return OptionalValue(*tiledInput);
    }

    if (auto load = value.getDefiningOp<maps::LoadOp>()) {
      auto sourceValue = getLoweredInputValue(load.getResult());
      if (failed(sourceValue))
        return OptionalValue();
      auto tiledInput = builder.getTiledValue(*sourceValue);
      if (failed(tiledInput))
        return load.emitOpError("failed to materialize tiled loaded input");
      return OptionalValue(*tiledInput);
    }

    if (auto empty = value.getDefiningOp<tensor::EmptyOp>()) {
      auto logicalType = cast<RankedTensorType>(empty.getType());
      auto tiledType = getTiledDeviceTensorType(rewriter.getContext(),
                                                logicalType);
      auto tiledEmpty = rewriter.create<mlir::tt::d2m::EmptyOp>(
          empty.getLoc(), tiledType.getShape(), tiledType.getElementType(),
          tiledType.getEncoding());
      return OptionalValue(tiledEmpty.getResult());
    }

    if (auto insertSlice = value.getDefiningOp<tensor::InsertSliceOp>()) {
      auto sourceType =
          cast<RankedTensorType>(insertSlice.getSource().getType());
      auto destType = cast<RankedTensorType>(insertSlice.getDest().getType());

      auto loweredSource =
          lowerValue(insertSlice.getSource(), sourceType, tensor::ExtractSliceOp());
      auto loweredDest =
          lowerValue(insertSlice.getDest(), destType, tensor::ExtractSliceOp());
      if (failed(loweredSource) || failed(loweredDest) || !*loweredSource ||
          !*loweredDest)
        return OptionalValue();

      auto logicalSource =
          materializeLogicalValue(insertSlice.getLoc(), **loweredSource, sourceType);
      auto logicalDest =
          materializeLogicalValue(insertSlice.getLoc(), **loweredDest, destType);
      if (failed(logicalSource) || failed(logicalDest))
        return insertSlice.emitOpError(
            "failed to materialize logical tensor assembly inputs");

      auto assembled = rewriter.create<tensor::InsertSliceOp>(
          insertSlice.getLoc(), *logicalSource, *logicalDest,
          insertSlice.getMixedOffsets(), insertSlice.getMixedSizes(),
          insertSlice.getMixedStrides());
      auto tiledAssembled = builder.getTiledValue(assembled.getResult());
      if (failed(tiledAssembled))
        return insertSlice.emitOpError(
            "failed to materialize tiled tensor assembly result");
      return OptionalValue(*tiledAssembled);
    }

    ComputeOpLoweringContext opCtx{
        .tile = tile,
        .builder = builder,
        .rewriter = rewriter,
        .getSourceValue = getSourceValue,
        .hoistValue = hoistValue,
        .lowerValue = lowerValue,
        .createStaticSlice =
            [&](Location loc, Value source, ArrayRef<int64_t> offsets,
                ArrayRef<int64_t> sizes) {
              return createDirectStaticSlice(rewriter, loc, source, offsets, sizes);
            },
        .materializeContiguous = [&](Location loc, Value source) {
          return materializeContiguousTensor(rewriter, loc, source);
        }};
    if (auto loweredOp = lowerRegisteredComputeOp(value, outputType, outputSlice,
                                                  opCtx);
        succeeded(loweredOp)) {
      return loweredOp;
    }
    if (isResidualComputeProducer(value)) {
      value.getDefiningOp()->emitOpError(
          "unsupported compute producer for recursive D2M lowering; add a "
          "registered compute emitter instead of falling back to tilization");
      return failure();
    }

    auto sourceValue = hoistValue(value);
    if (failed(sourceValue))
      return OptionalValue();

    Value input = *sourceValue;
    if (outputSlice) {
      input = createDirectStaticSlice(
          rewriter, value.getLoc(), input, outputSlice.getStaticOffsets(),
          outputSlice.getStaticSizes());
      input = materializeContiguousTensor(rewriter, value.getLoc(), input);
    }
    auto tiledInput = builder.getTiledValue(input);
    if (failed(tiledInput))
      return tile.emitOpError("failed to materialize tiled compute input");
    return OptionalValue(*tiledInput);
  };

  ComputeOpLoweringContext opCtx{
      .tile = tile,
      .builder = builder,
      .rewriter = rewriter,
      .getSourceValue = getSourceValue,
      .hoistValue = hoistValue,
      .lowerValue = lowerValue,
      .createStaticSlice =
          [&](Location loc, Value source, ArrayRef<int64_t> offsets,
              ArrayRef<int64_t> sizes) {
            return createDirectStaticSlice(rewriter, loc, source, offsets, sizes);
          },
      .materializeContiguous = [&](Location loc, Value source) {
        return materializeContiguousTensor(rewriter, loc, source);
      }};

  for (maps::SendOp send : sends) {
    auto outputType = cast<RankedTensorType>(send.getValue().getType());
    Value sendValue = send.getValue();
    auto sendSlice = sendValue.getDefiningOp<tensor::ExtractSliceOp>();
    if (sendSlice)
      sendValue = sendSlice.getSource();

    std::optional<Value> hoistedSendValue;
    if (!isResidualComputeProducer(sendValue)) {
      auto hoistedValue = hoistValue(send.getValue());
      if (failed(hoistedValue))
        return success(false);
      hoistedSendValue = *hoistedValue;
      logicalChannelValues[send.getChannelAttr()] = *hoistedSendValue;
    }

    rewriter.setInsertionPoint(tile);
    if (auto loweredValues =
            lowerRegisteredSendValue(sendValue, outputType, sendSlice, opCtx);
        succeeded(loweredValues)) {
      if (!*loweredValues)
        return success(false);

      SmallVector<Value> wrappedValues;
      wrappedValues.reserve((**loweredValues).size());
      for (Value loweredValue : **loweredValues) {
        auto wrappedValue = wrapIfPlaced(loweredValue);
        if (failed(wrappedValue))
          return failure();
        wrappedValues.push_back(*wrappedValue);
      }
      channelValues[send.getChannelAttr()] = wrappedValues.front();
      channelValueLists[send.getChannelAttr()] = wrappedValues;
    } else {
      if (isResidualComputeProducer(sendValue)) { // if the residual value is produced by a compute op, fail
        Operation *producer = sendValue.getDefiningOp();
        return send.emitOpError()
               << "unsupported compute producer '"
               << producer->getName().getStringRef()
               << "' for maps.send; add a registered compute/send emitter "
                  "instead of falling back to tilization";
      }

      if (!hoistedSendValue) {
        send.emitOpError("missing hoisted non-compute send value");
        return failure();
      }

      auto tiledValue = builder.getTiledValue(*hoistedSendValue);
      if (failed(tiledValue))
        return send.emitOpError("failed to materialize generic send value");
      channelValues[send.getChannelAttr()] = *tiledValue;
      channelValueLists[send.getChannelAttr()] = {*tiledValue};
    }

    auto stageIt = tileStageIds.find(tile.getOperation());
    if (stageIt != tileStageIds.end())
      stageChannels[stageIt->second].push_back(send.getChannelAttr());
  }

  eraseBlockOps(rewriter, tile.getBody().front());
  rewriter.eraseOp(tile);
  return true;
}

} // namespace mlir::maps
