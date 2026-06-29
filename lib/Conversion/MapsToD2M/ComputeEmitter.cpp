#include "maps/Conversion/MapsToD2M/ComputeEmitter.h"

#include "maps/Conversion/MapsToD2M/ComputeOpRegistry.h"
#include "maps/Conversion/MapsToD2M/D2MGenericBuilder.h"
#include "maps/Conversion/MapsToD2M/D2MTypeUtils.h"
#include "maps/Conversion/MapsToD2M/SpatialEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseSet.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

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

static bool isTileToTileChannel(const ChannelInfo &channel) {
  return channel.srcHartId != -1 && channel.dstHartId != -1;
}

static RankedTensorType getGlobalSemaphoreBackingType(Operation *op) {
  MLIRContext *ctx = op->getContext();
  SmallVector<int64_t> gridShape = {8, 8};
  SmallVector<int64_t> shardShape = {1, 1};
  SmallVector<int64_t> dimAlignments = {1, 1};
  auto i64Type = IntegerType::get(ctx, 64);
  auto intervalsType = RankedTensorType::get({2, 2}, i64Type);
  auto collapsedIntervals =
      DenseIntElementsAttr::get(intervalsType, ArrayRef<int64_t>{0, 1, 1, 2});
  auto layout = mlir::tt::ttcore::MetalLayoutAttr::get(
      ctx, gridShape, mlir::tt::ttcore::OOBVal::Undef,
      mlir::tt::ttcore::MemorySpace::DeviceL1,
      mlir::tt::ttcore::TensorMemoryLayout::Sharded, collapsedIntervals,
      dimAlignments);
  auto elementType = IntegerType::get(ctx, 32,
                                      IntegerType::SignednessSemantics::Unsigned);
  return RankedTensorType::get(layout.getDeviceShape(gridShape, shardShape),
                               elementType, layout);
}

static Value getOrCreateChannelSemaphore(
    maps::SendOp send, DenseMap<Attribute, ChannelInfo> &channels,
    DenseMap<Attribute, Value> &channelSemaphores, PatternRewriter &rewriter) {
  Attribute channel = send.getChannelAttr();
  auto channelIt = channels.find(channel);
  if (channelIt == channels.end() || !isTileToTileChannel(channelIt->second))
    return {};

  auto semIt = channelSemaphores.find(channel);
  if (semIt != channelSemaphores.end())
    return semIt->second;

  rewriter.setInsertionPoint(send->getParentOp());
  auto backingType = getGlobalSemaphoreBackingType(send.getOperation());
  auto backing = rewriter.create<mlir::tt::d2m::EmptyOp>(
      send.getLoc(), backingType.getShape(), backingType.getElementType(),
      backingType.getEncoding());
  auto semType = mlir::tt::d2m::GlobalSemaphoreType::get(rewriter.getContext());
  auto initialValue = IntegerAttr::get(backingType.getElementType(), 0);
  auto sem = rewriter.create<mlir::tt::d2m::CreateGlobalSemaphoreOp>(
      send.getLoc(), semType, backing.getResult(), initialValue);
  channelSemaphores[channel] = sem.getResult();
  return sem.getResult();
}

struct RecvSemaphore {
  Attribute channel;
  Value semaphore;
  Value anchor;
};

static SmallVector<RecvSemaphore> collectRecvSemaphores(
    maps::TileOp tile, DenseMap<Attribute, ChannelInfo> &channels,
    DenseMap<Attribute, Value> &channelSemaphores,
    DenseMap<Attribute, Value> &channelValues) {
  SmallVector<RecvSemaphore> semaphores;
  DenseSet<Value> seen;

  for (Operation &nested : tile.getBody().front()) {
    auto recv = dyn_cast<maps::RecvOp>(nested);
    if (!recv)
      continue;

    auto channelIt = channels.find(recv.getChannelAttr());
    if (channelIt == channels.end() || !isTileToTileChannel(channelIt->second))
      continue;

    auto semIt = channelSemaphores.find(recv.getChannelAttr());
    if (semIt == channelSemaphores.end())
      continue;
    auto anchorIt = channelValues.find(recv.getChannelAttr());
    if (anchorIt == channelValues.end())
      continue;
    if (seen.insert(semIt->second).second)
      semaphores.push_back({.channel = recv.getChannelAttr(),
                            .semaphore = semIt->second,
                            .anchor = anchorIt->second});
  }

  return semaphores;
}

static Value emitSemaphoreWaits(Location loc, ArrayRef<RecvSemaphore> semaphores,
                                mlir::tt::ttcore::CoreRangeAttr coreRange,
                                PatternRewriter &rewriter) {
  if (semaphores.empty())
    return {};

  Value anchor = semaphores.front().anchor;
  SmallVector<Value> semaphoreValues;
  semaphoreValues.reserve(semaphores.size());
  for (RecvSemaphore semaphore : semaphores)
    semaphoreValues.push_back(semaphore.semaphore);

  auto threads = rewriter.getArrayAttr(mlir::tt::d2m::ThreadAttr::get(
      rewriter.getContext(), mlir::tt::d2m::ThreadType::Datamovement));
  auto generic = rewriter.create<mlir::tt::d2m::GenericOp>(
      loc, TypeRange{anchor.getType()}, ValueRange{}, ValueRange{anchor},
      semaphoreValues, getFrontendGrid(rewriter.getContext()),
      rewriter.getArrayAttr({}), rewriter.getAffineMapArrayAttr({}),
      rewriter.getArrayAttr({}), threads,
      /*fabricConnectionConfig=*/nullptr, /*regionsCount=*/1);

  {
    PatternRewriter::InsertionGuard guard(rewriter);
    Region &region = generic.getRegion(0);
    Block *block = rewriter.createBlock(&region);
    rewriter.setInsertionPointToEnd(block);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (RecvSemaphore semaphore : semaphores)
      rewriter.create<mlir::tt::d2m::SemaphoreWaitOp>(
          loc, semaphore.semaphore, one);
    rewriter.create<mlir::tt::d2m::YieldOp>(loc, ValueRange{anchor});
  }

  rewriter.setInsertionPoint(generic);
  auto spatial = rewriter.create<mlir::tt::d2m::SpatialOp>(
      loc, TypeRange{anchor.getType()}, ValueRange{}, ValueRange{anchor},
      rewriter.getArrayAttr(coreRange), /*regionsCount=*/1);
  Block *regionBlock = rewriter.createBlock(&spatial.getRegions().front());
  generic->moveBefore(regionBlock, regionBlock->end());
  rewriter.setInsertionPointToEnd(regionBlock);
  rewriter.create<mlir::tt::d2m::SpatialYieldOp>(
      loc, TypeRange{}, generic.getResults(), ArrayRef<NamedAttribute>{});
  rewriter.setInsertionPointAfter(spatial);
  return spatial.getResult(0);
}

static FailureOr<Value> materializeTileToTilePayload(Location loc, Value source,
                                                     PatternRewriter &rewriter) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType)
    return failure();
  auto sourceLayout =
      dyn_cast_if_present<mlir::tt::ttcore::MetalLayoutAttr>(
          sourceType.getEncoding());
  if (!sourceLayout)
    return failure();

  auto dramLayout = mlir::tt::ttcore::MetalLayoutAttr::get(
      rewriter.getContext(), sourceLayout.getLogicalShape(),
      mlir::tt::ttcore::OOBVal::Undef,
      mlir::tt::ttcore::MemorySpace::DeviceDRAM,
      mlir::tt::ttcore::TensorMemoryLayout::Interleaved,
      sourceLayout.getCollapsedIntervals(), sourceLayout.getDimAlignments());
  SmallVector<int64_t> tileShape;
  if (auto tileType =
          dyn_cast<mlir::tt::ttcore::TileType>(sourceType.getElementType()))
    llvm::append_range(tileShape, tileType.getShape());
  SmallVector<int64_t> unitGrid = {1, 1};
  auto dramType =
      RankedTensorType::get(dramLayout.getDeviceShape(unitGrid, tileShape),
                            sourceType.getElementType(), dramLayout);
  auto dram = rewriter.create<mlir::tt::d2m::EmptyOp>(
      loc, dramType.getShape(), dramType.getElementType(),
      dramType.getEncoding());
  auto toLayout = rewriter.create<mlir::tt::d2m::ToLayoutOp>(
      loc, source, dram.getResult());
  return toLayout.getResult(0);
}

static Value emitSemaphoreSignal(Location loc, Value semaphore, Value anchor,
                                 mlir::tt::ttcore::CoreRangeAttr senderCoreRange,
                                 mlir::tt::ttcore::CoreRangeAttr dstCoreRange,
                                 PatternRewriter &rewriter) {
  if (!semaphore || !anchor)
    return {};

  auto threads = rewriter.getArrayAttr(mlir::tt::d2m::ThreadAttr::get(
      rewriter.getContext(), mlir::tt::d2m::ThreadType::Datamovement));
  auto generic = rewriter.create<mlir::tt::d2m::GenericOp>(
      loc, TypeRange{anchor.getType()}, ValueRange{}, ValueRange{anchor},
      ValueRange{semaphore}, getFrontendGrid(rewriter.getContext()),
      rewriter.getArrayAttr({}), rewriter.getAffineMapArrayAttr({}),
      rewriter.getArrayAttr({}), threads,
      /*fabricConnectionConfig=*/nullptr, /*regionsCount=*/1);

  {
    PatternRewriter::InsertionGuard guard(rewriter);
    Region &region = generic.getRegion(0);
    Block *block = rewriter.createBlock(&region);
    rewriter.setInsertionPointToEnd(block);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto dst = dstCoreRange.getStartCoord();
    Value dstY = rewriter.create<arith::ConstantIndexOp>(loc, dst.getY());
    Value dstX = rewriter.create<arith::ConstantIndexOp>(loc, dst.getX());
    rewriter.create<mlir::tt::d2m::SemaphoreIncOp>(
        loc, semaphore, one, ValueRange{dstY, dstX});
    rewriter.create<mlir::tt::d2m::YieldOp>(loc, ValueRange{anchor});
  }

  rewriter.setInsertionPoint(generic);
  auto spatial = rewriter.create<mlir::tt::d2m::SpatialOp>(
      loc, TypeRange{anchor.getType()}, ValueRange{}, ValueRange{anchor},
      rewriter.getArrayAttr(senderCoreRange), /*regionsCount=*/1);
  Block *regionBlock = rewriter.createBlock(&spatial.getRegions().front());
  generic->moveBefore(regionBlock, regionBlock->end());
  rewriter.setInsertionPointToEnd(regionBlock);
  rewriter.create<mlir::tt::d2m::SpatialYieldOp>(
      loc, TypeRange{}, generic.getResults(), ArrayRef<NamedAttribute>{});
  rewriter.setInsertionPointAfter(spatial);
  return spatial.getResult(0);
}

struct CompositeAssemblyPiece {
  Value value;
  RankedTensorType type;
};

static FailureOr<std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>>
collectCompositeAssembly(Value value) {
  auto insertSlice = value.getDefiningOp<tensor::InsertSliceOp>();
  if (!insertSlice)
    return std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>();

  auto destType = dyn_cast<RankedTensorType>(insertSlice.getDest().getType());
  auto sourceType = dyn_cast<RankedTensorType>(insertSlice.getSource().getType());
  if (!destType || !sourceType || sourceType.getRank() != destType.getRank())
    return std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>();
  int64_t concatDim = -1;
  auto offsets = insertSlice.getStaticOffsets();
  auto sizes = insertSlice.getStaticSizes();
  auto strides = insertSlice.getStaticStrides();
  if (llvm::is_contained(offsets, ShapedType::kDynamic) ||
      llvm::is_contained(sizes, ShapedType::kDynamic) ||
      llvm::is_contained(strides, ShapedType::kDynamic))
    return std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>();
  auto destShape = destType.getShape();
  for (size_t index = 0; index < offsets.size(); ++index) {
    int64_t offset = offsets[index];
    int64_t size = sizes[index];
    int64_t stride = strides[index];
    int64_t destSize = destShape[index];
    if (stride != 1)
      return std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>();
    if (offset != 0 || size != destSize) {
      if (concatDim != -1)
        return std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>();
      concatDim = static_cast<int64_t>(index);
    }
  }
  if (concatDim == -1)
    return std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>();

  SmallVector<CompositeAssemblyPiece> pieces;
  int64_t expectedOffset = 0;
  std::function<LogicalResult(Value)> walkAssembly = [&](Value current) {
    if (current.getDefiningOp<tensor::EmptyOp>())
      return success();

    auto currentInsert = current.getDefiningOp<tensor::InsertSliceOp>();
    if (!currentInsert)
      return failure();

    if (failed(walkAssembly(currentInsert.getDest())))
      return failure();

    auto currentDestType =
        dyn_cast<RankedTensorType>(currentInsert.getDest().getType());
    auto currentSourceType =
        dyn_cast<RankedTensorType>(currentInsert.getSource().getType());
    if (!currentDestType || !currentSourceType ||
        currentSourceType.getRank() != currentDestType.getRank())
      return failure();
    auto currentOffsets = currentInsert.getStaticOffsets();
    auto currentSizes = currentInsert.getStaticSizes();
    auto currentStrides = currentInsert.getStaticStrides();
    if (llvm::is_contained(currentOffsets, ShapedType::kDynamic) ||
        llvm::is_contained(currentSizes, ShapedType::kDynamic) ||
        llvm::is_contained(currentStrides, ShapedType::kDynamic))
      return failure();
    auto currentDestShape = currentDestType.getShape();
    for (size_t index = 0; index < currentOffsets.size(); ++index) {
      int64_t offset = currentOffsets[index];
      int64_t size = currentSizes[index];
      int64_t stride = currentStrides[index];
      int64_t destSize = currentDestShape[index];
      if (stride != 1)
        return failure();
      if (static_cast<int64_t>(index) == concatDim) {
        if (offset != expectedOffset)
          return failure();
        expectedOffset += size;
        continue;
      }
      if (offset != 0 || size != destSize)
        return failure();
    }

    pieces.push_back(
        {.value = currentInsert.getSource(), .type = currentSourceType});
    return success();
  };

  if (failed(walkAssembly(value)))
    return std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>();
  if (expectedOffset != destType.getShape()[concatDim])
    return std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>();
  return std::optional<std::pair<int64_t, SmallVector<CompositeAssemblyPiece>>>(
      std::make_pair(concatDim, std::move(pieces)));
}

} // namespace

FailureOr<bool> lowerComputeTileProgram(
    maps::TileOp tile, DenseMap<Attribute, ChannelInfo> &channels,
    DenseMap<Attribute, Value> &channelValues,
    DenseMap<Attribute, Value> &logicalChannelValues,
    DenseMap<Attribute, SmallVector<Value>> &channelValueLists,
    DenseMap<Attribute, Value> &channelSemaphores,
    DenseMap<Attribute, Value> &slotValues,
    DenseMap<Operation *, mlir::tt::ttcore::CoreRangeAttr> &tileCoreRanges,
    DenseMap<int64_t, mlir::tt::ttcore::CoreRangeAttr> &tileIdCoreRanges,
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
  SmallVector<RecvSemaphore> recvSemaphores =
      collectRecvSemaphores(tile, channels, channelSemaphores, channelValues);
  if (!recvSemaphores.empty()) {
    auto coreRangeIt = tileCoreRanges.find(tile.getOperation());
    if (coreRangeIt == tileCoreRanges.end())
      return tile.emitOpError("missing placement for compute tile");
    rewriter.setInsertionPoint(tile);
    Value waitedAnchor =
        emitSemaphoreWaits(tile.getLoc(), recvSemaphores, coreRangeIt->second,
                           rewriter);
    if (waitedAnchor)
      channelValues[recvSemaphores.front().channel] = waitedAnchor;
  }
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

    auto scalarDeviceType =
        getScalarDeviceTensorType(rewriter.getContext(), logicalType);
    if (valueType == scalarDeviceType)
      return value;

    auto logicalOutput = rewriter.create<mlir::tt::d2m::EmptyOp>(
        loc, scalarDeviceType.getShape(), scalarDeviceType.getElementType(),
        scalarDeviceType.getEncoding());
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

      auto compositeAssembly = collectCompositeAssembly(value);
      if (failed(compositeAssembly))
        return failure();
      if (*compositeAssembly) {
        auto [concatDim, pieces] = **compositeAssembly;
        SmallVector<Value> scalarInputs;
        SmallVector<int64_t> logicalSizes;
        scalarInputs.reserve(pieces.size());
        logicalSizes.reserve(pieces.size());
        for (const CompositeAssemblyPiece &piece : pieces) {
          auto loweredPiece =
              lowerValue(piece.value, piece.type, tensor::ExtractSliceOp());
          if (failed(loweredPiece) || !*loweredPiece)
            return OptionalValue();

          auto scalarPiece = materializeLogicalValue(insertSlice.getLoc(),
                                                     **loweredPiece, piece.type);
          if (failed(scalarPiece))
            return insertSlice.emitOpError(
                "failed to materialize stitched input shard");
          scalarInputs.push_back(*scalarPiece);
          logicalSizes.push_back(piece.type.getShape()[concatDim]);
        }

        auto scalarDestType = getScalarDeviceTensorType(rewriter.getContext(),
                                                        destType);
        auto stitched = rewriter.create<mlir::tt::d2m::CompositeViewOp>(
            insertSlice.getLoc(), scalarDestType, scalarInputs, concatDim,
            rewriter.getDenseI64ArrayAttr(logicalSizes));
        auto tiledAssembled = builder.getTiledValue(stitched.getResult());
        if (failed(tiledAssembled))
          return insertSlice.emitOpError(
              "failed to materialize tiled stitched tensor");
        return OptionalValue(*tiledAssembled);
      }

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

      auto sourceValueType = cast<RankedTensorType>((*logicalSource).getType());
      int64_t sourcePrefixRank =
          sourceValueType.getRank() - static_cast<int64_t>(sourceType.getRank());
      SmallVector<OpFoldResult> mixedOffsets(sourcePrefixRank,
                                             rewriter.getIndexAttr(0));
      SmallVector<OpFoldResult> mixedSizes;
      SmallVector<OpFoldResult> mixedStrides(sourcePrefixRank,
                                             rewriter.getIndexAttr(1));
      for (int64_t i = 0; i < sourcePrefixRank; ++i)
        mixedSizes.push_back(rewriter.getIndexAttr(sourceValueType.getShape()[i]));
      llvm::append_range(mixedOffsets, insertSlice.getMixedOffsets());
      llvm::append_range(mixedSizes, insertSlice.getMixedSizes());
      llvm::append_range(mixedStrides, insertSlice.getMixedStrides());

      auto assembled = rewriter.create<tensor::InsertSliceOp>(
          insertSlice.getLoc(), *logicalSource, *logicalDest,
          mixedOffsets, mixedSizes, mixedStrides);
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
    Value sendSemaphore =
        getOrCreateChannelSemaphore(send, channels, channelSemaphores, rewriter);
    std::optional<mlir::tt::ttcore::CoreRangeAttr> dstCoreRange;
    auto channelIt = channels.find(send.getChannelAttr());
    if (channelIt != channels.end() && isTileToTileChannel(channelIt->second)) {
      auto dstIt = tileIdCoreRanges.find(channelIt->second.dstHartId);
      if (dstIt == tileIdCoreRanges.end())
        return send.emitOpError("missing destination tile placement");
      dstCoreRange = dstIt->second;
    }

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
        Value channelValue = *wrappedValue;
        if (sendSemaphore && dstCoreRange) {
          auto srcCoreRangeIt = tileCoreRanges.find(tile.getOperation());
          if (srcCoreRangeIt == tileCoreRanges.end())
            return tile.emitOpError("missing placement for compute tile");
          rewriter.setInsertionPointAfter(
              wrappedValue->getDefiningOp());
          auto tileToTilePayload =
              materializeTileToTilePayload(send.getLoc(), *wrappedValue,
                                           rewriter);
          if (failed(tileToTilePayload))
            return send.emitOpError(
                "failed to materialize tile-to-tile payload");
          Value signaledValue =
              emitSemaphoreSignal(send.getLoc(), sendSemaphore,
                                  *tileToTilePayload,
                                  srcCoreRangeIt->second, *dstCoreRange,
                                  rewriter);
          if (signaledValue)
            channelValue = signaledValue;
        }
        wrappedValues.push_back(channelValue);
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
