#include "maps/Conversion/MapsToD2M/MapsPatternLowering.h"

#include "maps/Conversion/MapsToD2M/D2MGenericBuilder.h"
#include "maps/Conversion/MapsToD2M/D2MTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"

#include <functional>

namespace mlir::maps {
namespace {

static bool isElementwiseAddGeneric(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 2 || generic.getNumDpsInits() != 1)
    return false;

  auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!resultType)
    return false;

  auto identityMap =
      AffineMap::getMultiDimIdentityMap(resultType.getRank(), generic.getContext());
  if (llvm::any_of(generic.getIndexingMapsArray(),
                   [&](AffineMap map) { return map != identityMap; }))
    return false;

  Block &body = generic.getRegion().front();
  auto add = dyn_cast<arith::AddFOp>(body.front());
  if (!add)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(add->getNextNode());
  if (!yield || yield.getNumOperands() != 1 || yield.getOperand(0) != add.getResult())
    return false;
  return add->getNextNode() == yield && yield->getNextNode() == nullptr;
}

static bool isElementwiseExpGeneric(linalg::GenericOp generic) {
  if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
    return false;

  auto resultType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
  if (!resultType)
    return false;

  auto identityMap =
      AffineMap::getMultiDimIdentityMap(resultType.getRank(), generic.getContext());
  if (llvm::any_of(generic.getIndexingMapsArray(),
                   [&](AffineMap map) { return map != identityMap; }))
    return false;

  Block &body = generic.getRegion().front();
  auto exp = dyn_cast<math::ExpOp>(body.front());
  if (!exp)
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(exp->getNextNode());
  if (!yield || yield.getNumOperands() != 1 || yield.getOperand(0) != exp.getResult())
    return false;
  return exp->getNextNode() == yield && yield->getNextNode() == nullptr;
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

static void inlineRegionOp(PatternRewriter &rewriter, Operation *op) {
  // take a block outside the region it belongs, useful for wrapper ops like module
  rewriter.inlineBlockBefore(&op->getRegion(0).front(), op);
  rewriter.eraseOp(op);
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

static void eraseInputSends(ModuleOp module, PatternRewriter &rewriter) {
  SmallVector<maps::SendOp> sendsToErase;
  module.walk([&](maps::SendOp send) {
    if (static_cast<int64_t>(send.getSrcHartid()) == -1)
      sendsToErase.push_back(send);
  });

  for (maps::SendOp send : sendsToErase)
    rewriter.eraseOp(send);
}

static LogicalResult lowerInitTile(maps::TileOp tile, PatternRewriter &rewriter) {
  Block &body = tile.getBody().front();
  body.dropAllDefinedValueUses();
  eraseBlockOps(rewriter, body);
  rewriter.eraseOp(tile);
  return success();
}

static FailureOr<bool> lowerComputeTile(
    maps::TileOp tile, DenseMap<Attribute, ChannelInfo> &channels,
    DenseMap<Attribute, Value> &channelValues,
    DenseMap<Attribute, Value> &logicalChannelValues,
    DenseMap<Attribute, SmallVector<Value>> &channelValueLists,
    DenseMap<Attribute, Value> &slotValues,
    DenseMap<int64_t, SmallVector<Attribute>> &stageChannels,
    DenseMap<Operation *, int64_t> &tileStageIds,
    PatternRewriter &rewriter) {
  SmallVector<maps::SendOp> sends;
  for (Operation &nested : tile.getBody().front()) {
    if (auto send = dyn_cast<maps::SendOp>(nested))
      sends.push_back(send);
  }

  auto getSourceValue = [&](Value value) -> FailureOr<Value> {
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

  DenseMap<Value, Value> hoistedValues;
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
  using OptionalValue = std::optional<Value>;
  std::function<FailureOr<OptionalValue>(
      Value, RankedTensorType, tensor::ExtractSliceOp)> lowerValue;

  lowerValue = [&](Value value, RankedTensorType outputType,
                   tensor::ExtractSliceOp outputSlice)
      -> FailureOr<OptionalValue> {
    if (auto exp = value.getDefiningOp<math::ExpOp>()) {
      auto loweredInput = lowerValue(exp.getOperand(), outputType, outputSlice);
      if (failed(loweredInput) || !*loweredInput)
        return loweredInput;

      auto expGeneric =
          builder.createExpGeneric(exp.getLoc(), **loweredInput, outputType);
      if (failed(expGeneric))
        return failure();
      return OptionalValue(expGeneric->getResult(0));
    }

    if (auto matmul = value.getDefiningOp<linalg::MatmulOp>()) {
      auto lhsSource = hoistValue(matmul.getInputs()[0]);
      auto rhsSource = hoistValue(matmul.getInputs()[1]);
      if (failed(lhsSource) || failed(rhsSource))
        return OptionalValue();

      Value rhsValue = *rhsSource;
      if (outputSlice) {
        auto rhsType = cast<RankedTensorType>(rhsValue.getType());
        SmallVector<int64_t> offsets(rhsType.getRank(), 0);
        SmallVector<int64_t> sizes(rhsType.getShape().begin(),
                                   rhsType.getShape().end());
        offsets.back() = outputSlice.getStaticOffsets().back();
        sizes.back() = outputSlice.getStaticSizes().back();
        rhsValue = createDirectStaticSlice(rewriter, value.getLoc(), rhsValue,
                                           offsets, sizes);
        rhsValue = materializeContiguousTensor(rewriter, value.getLoc(), rhsValue);
      }

          auto tiledLhs = builder.getTiledValue(*lhsSource);
          auto tiledRhs = builder.getTiledValue(rhsValue);
          if (failed(tiledLhs) || failed(tiledRhs))
            return tile.emitOpError("failed to materialize tiled matmul inputs");
          auto matmulGeneric = builder.createMatmulGeneric(
              value.getLoc(), *tiledLhs, *tiledRhs, outputType);
          if (failed(matmulGeneric))
            return failure();
          return OptionalValue(matmulGeneric->getResult(0));
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

  for (maps::SendOp send : sends) {
    auto hoistedSendValue = hoistValue(send.getValue());
    if (failed(hoistedSendValue))
      return success(false);
    logicalChannelValues[send.getChannelAttr()] = *hoistedSendValue;

    auto outputType = cast<RankedTensorType>(send.getValue().getType());
    Value sendValue = send.getValue();
    auto sendSlice = sendValue.getDefiningOp<tensor::ExtractSliceOp>();
    if (sendSlice)
      sendValue = sendSlice.getSource();

    if (isa_and_nonnull<math::ExpOp, linalg::MatmulOp>(
            sendValue.getDefiningOp())) {
      rewriter.setInsertionPoint(tile);
      auto loweredValue = lowerValue(sendValue, outputType, sendSlice);
      if (failed(loweredValue))
        return failure();
      if (!*loweredValue)
        return success(false);
      channelValues[send.getChannelAttr()] = **loweredValue;
      channelValueLists[send.getChannelAttr()] = {**loweredValue};
    } else if (auto add = sendValue.getDefiningOp<arith::AddFOp>()) {
      auto concat = add.getLhs().getDefiningOp<tensor::ConcatOp>();
      Value bias = add.getRhs();
      if (!concat) {
        concat = add.getRhs().getDefiningOp<tensor::ConcatOp>();
        bias = add.getLhs();
      }

      if (concat) {
        SmallVector<Value> shardResults;
        int64_t dim = static_cast<int64_t>(concat.getDim());
        int64_t offset = 0;
        auto biasSource = getSourceValue(bias);
        if (failed(biasSource))
          return success(false);

        for (Value concatInput : concat.getInputs()) {
          rewriter.setInsertionPoint(tile);
          auto concatInputValue = hoistValue(concatInput);
          if (failed(concatInputValue))
            return success(false);

          auto inputSource = builder.getTiledValue(*concatInputValue);
          if (failed(inputSource))
            return send.emitOpError("failed to materialize concat shard");

          auto logicalInputType = cast<RankedTensorType>(concatInput.getType());
          SmallVector<int64_t> staticOffsets(logicalInputType.getRank(), 0);
          SmallVector<int64_t> staticSizes(logicalInputType.getShape().begin(),
                                           logicalInputType.getShape().end());
          staticOffsets[dim] = offset;
          offset += logicalInputType.getDimSize(dim);

          Value biasSlice = createDirectStaticSlice(
              rewriter, send.getLoc(), *biasSource, staticOffsets, staticSizes);
          biasSlice =
              materializeContiguousTensor(rewriter, send.getLoc(), biasSlice);
          auto tiledBias = builder.getTiledValue(biasSlice);
          if (failed(tiledBias))
            return send.emitOpError("failed to materialize bias shard");

          auto addGeneric = builder.createAddGeneric(send.getLoc(), *inputSource,
                                                     *tiledBias,
                                                     logicalInputType);
          if (failed(addGeneric))
            return failure();
          shardResults.push_back(addGeneric->getResult(0));
        }

        channelValues[send.getChannelAttr()] = shardResults.front();
        channelValueLists[send.getChannelAttr()] = shardResults;
      } else {
        rewriter.setInsertionPoint(tile);
        auto lhsValue = hoistValue(add.getLhs());
        auto rhsValue = hoistValue(add.getRhs());
        if (failed(lhsValue) || failed(rhsValue))
          return success(false);

        auto lhsSource = builder.getTiledValue(*lhsValue);
        auto rhsSource = builder.getTiledValue(*rhsValue);
        if (failed(lhsSource) || failed(rhsSource))
          return send.emitOpError("failed to materialize add inputs");

        auto addGeneric = builder.createAddGeneric(send.getLoc(), *lhsSource,
                                                   *rhsSource, outputType);
        if (failed(addGeneric))
          return failure();
        channelValues[send.getChannelAttr()] = addGeneric->getResult(0);
        channelValueLists[send.getChannelAttr()] = {addGeneric->getResult(0)};
      }
    } else {
      rewriter.setInsertionPoint(tile);
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

} // namespace


// main entry point
LogicalResult lowerMapsProgramToD2M(ModuleOp module,
                                    MapsProgramInfo &program,
                                    ChannelLoweringState &state) {
  PatternRewriter rewriter(module.getContext()); //
  DenseMap<Attribute, Value> logicalChannelValues;
  DenseMap<Operation *, int64_t> tileStageIds;
  for (StageInfo &stage : program.stages) {
    for (TileInfo &tile : stage.tiles)
      tileStageIds[tile.op.getOperation()] = stage.stageId;
  }

  // Inline the MAPS structural wrappers.
  // setInsertionPoint sets the "cursor" of the patternRewriter to the specific op loc
  rewriter.setInsertionPoint(program.init); 
  inlineRegionOp(rewriter, program.init);
  rewriter.setInsertionPoint(program.run);
  inlineRegionOp(rewriter, program.run);
  for (StageInfo &stage : program.stages) {
    rewriter.setInsertionPoint(stage.op);
    inlineRegionOp(rewriter, stage.op);
  }

  // Drop storage-only tiles now that slot values have been analyzed.
  SmallVector<maps::TileOp> initTiles;
  SmallVector<maps::TileOp> pendingTiles;

  module.walk([&](maps::TileOp tile) {
    if (isInitStorageTile(tile)) {
      initTiles.push_back(tile);
      return;
    }
    pendingTiles.push_back(tile);
  });

  for (maps::TileOp tile : initTiles) {
    // entry point for lowering initialization tiles (no compute)
    if (failed(lowerInitTile(tile, rewriter)))
      return failure();
  }

  // Lower compute tiles in dependency order without relying on greedy cleanup.
  while (!pendingTiles.empty()) {
    SmallVector<maps::TileOp> deferredTiles;
    bool madeProgress = false;

    for (maps::TileOp tile : pendingTiles) {
      auto lowered = lowerComputeTile(tile, program.channels, state.values,
                                      logicalChannelValues, state.valueLists,
                                      program.slotValues,
                                      state.stageChannels, tileStageIds,
                                      rewriter);
      if (failed(lowered))
        return failure();
      if (*lowered) {
        madeProgress = true;
        continue;
      }
      deferredTiles.push_back(tile);
    }

    if (!madeProgress)
      return deferredTiles.front().emitOpError(
          "failed to resolve compute tile dependencies");
    pendingTiles = std::move(deferredTiles);
  }

  // Host-to-tile sends are represented directly by their source values.
  eraseInputSends(module, rewriter);
  return success();
}

} // namespace mlir::maps
