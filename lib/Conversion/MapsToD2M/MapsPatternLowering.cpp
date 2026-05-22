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

namespace mlir::maps {
namespace {

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

  D2MGenericBuilder builder(rewriter.getContext(), rewriter);

  for (maps::SendOp send : sends) {
    auto outputType = cast<RankedTensorType>(send.getValue().getType());
    Value sendValue = send.getValue();
    auto sendSlice = sendValue.getDefiningOp<tensor::ExtractSliceOp>();
    if (sendSlice)
      sendValue = sendSlice.getSource();

    if (auto exp = sendValue.getDefiningOp<math::ExpOp>()) {
      auto matmul = exp.getOperand().getDefiningOp<linalg::MatmulOp>();
      if (!matmul)
        return send.emitOpError("expected exp(matmul(...)) producer");

      auto lhsSource = getSourceValue(matmul.getInputs()[0]);
      auto rhsSource = getSourceValue(matmul.getInputs()[1]);
      if (failed(lhsSource) || failed(rhsSource))
        return success(false);

      Value rhsValue = *rhsSource;
      if (sendSlice) {
        auto rhsType = cast<RankedTensorType>(rhsValue.getType());
        SmallVector<int64_t> offsets(rhsType.getRank(), 0);
        SmallVector<int64_t> sizes(rhsType.getShape().begin(),
                                   rhsType.getShape().end());
        offsets.back() = sendSlice.getStaticOffsets().back();
        sizes.back() = sendSlice.getStaticSizes().back();
        rewriter.setInsertionPoint(tile);
        rhsValue = createDirectStaticSlice(rewriter, send.getLoc(), rhsValue,
                                           offsets, sizes);
        rhsValue =
            materializeContiguousTensor(rewriter, send.getLoc(), rhsValue);
      }

      rewriter.setInsertionPoint(tile);
      auto tiledLhs = builder.getTiledValue(*lhsSource);
      auto tiledRhs = builder.getTiledValue(rhsValue);
      if (failed(tiledLhs) || failed(tiledRhs))
        return send.emitOpError("failed to materialize tiled matmul inputs");

      auto matmulGeneric = builder.createMatmulGeneric(send.getLoc(), *tiledLhs,
                                                       *tiledRhs, outputType);
      if (failed(matmulGeneric))
        return failure();
      auto expGeneric = builder.createExpGeneric(send.getLoc(),
                                                 matmulGeneric->getResult(0),
                                                 outputType);
      if (failed(expGeneric))
        return failure();

      channelValues[send.getChannelAttr()] = expGeneric->getResult(0);
      channelValueLists[send.getChannelAttr()] = {expGeneric->getResult(0)};
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
          auto concatInputValue = getSourceValue(concatInput);
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
        auto lhsValue = getSourceValue(add.getLhs());
        auto rhsValue = getSourceValue(add.getRhs());
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
      return send.emitOpError("unsupported compute tile send pattern");
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

LogicalResult lowerMapsProgramToD2M(ModuleOp module,
                                    MapsProgramInfo &program,
                                    ChannelLoweringState &state) {
  PatternRewriter rewriter(module.getContext());
  DenseMap<Operation *, int64_t> tileStageIds;
  for (StageInfo &stage : program.stages) {
    for (TileInfo &tile : stage.tiles)
      tileStageIds[tile.op.getOperation()] = stage.stageId;
  }

  // Inline the MAPS structural wrappers.
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
    if (failed(lowerInitTile(tile, rewriter)))
      return failure();
  }

  // Lower compute tiles in dependency order without relying on greedy cleanup.
  while (!pendingTiles.empty()) {
    SmallVector<maps::TileOp> deferredTiles;
    bool madeProgress = false;

    for (maps::TileOp tile : pendingTiles) {
      auto lowered = lowerComputeTile(tile, program.channels, state.values,
                                      state.valueLists, program.slotValues,
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
