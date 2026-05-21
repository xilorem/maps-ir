#include "maps/Conversion/Passes.h"
#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Transforms/DialectConversion.h"

#include <algorithm>
#include <limits>

namespace mlir::maps {
#define GEN_PASS_DEF_CONVERTMAPSTOD2M
#include "maps/Conversion/Passes.h.inc"

namespace {

struct ChannelInfo {
  FlatSymbolRefAttr name;
  int64_t srcHartId = -1;
  int64_t dstHartId = -1;
  RankedTensorType type;
  maps::SendOp send;
  SmallVector<maps::RecvOp> recvs;
};

struct TileInfo {
  int64_t tileId = -1;
  SmallVector<int64_t> coords;
  maps::TileOp op;
};

struct StageInfo {
  int64_t stageId = -1;
  StringRef sourceName;
  maps::StageOp op;
  SmallVector<TileInfo> tiles;
};

struct MapsProgramInfo {
  maps::InitOp init;
  maps::RunOp run;
  SmallVector<StageInfo> stages;
  DenseMap<Attribute, ChannelInfo> channels;
  DenseMap<Attribute, Value> slotValues;
};

} // namespace


namespace{

static bool isInitStorageTile(maps::TileOp op) {
  SmallVector<maps::RecvOp> recvs;
  SmallVector<maps::StoreOp> stores;

  for (Operation &nested : op.getBody().front()) {
    if (auto recv = dyn_cast<maps::RecvOp>(nested)) {
      recvs.push_back(recv);
      continue;
    }
    if (auto store = dyn_cast<maps::StoreOp>(nested)) {
      stores.push_back(store);
      continue;
    }
    return false;
  }

  if (recvs.size() != stores.size())
    return false;

  for (auto [recv, store] : llvm::zip_equal(recvs, stores)) {
    if (store.getValue() != recv.getResult())
      return false;
  }

  return true;
}

static int64_t ceilDiv(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

static mlir::tt::ttcore::GridAttr getGridForTensor(MLIRContext *ctx,
                                                   RankedTensorType type) {
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() < 2)
    return mlir::tt::ttcore::GridAttr::get(ctx, {1, 1});

  return mlir::tt::ttcore::GridAttr::get(
      ctx, {ceilDiv(shape[shape.size() - 2], 32),
            ceilDiv(shape[shape.size() - 1], 32)});
}

static mlir::tt::ttcore::MetalLayoutAttr
getL1ShardedLayout(MLIRContext *ctx, RankedTensorType type) {
  return mlir::tt::ttcore::MetalLayoutAttr::get(
      ctx, type.getShape(), mlir::tt::ttcore::OOBVal::Undef,
      mlir::tt::ttcore::MemorySpace::DeviceL1,
      mlir::tt::ttcore::TensorMemoryLayout::Sharded);
}

static RankedTensorType getDeviceTensorType(MLIRContext *ctx,
                                            RankedTensorType type) {
  if (type.getEncoding())
    return type;
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               getL1ShardedLayout(ctx, type));
}

static RankedTensorType getTiledDeviceTensorType(MLIRContext *ctx,
                                                 RankedTensorType type) {
  if (type.getEncoding() &&
      isa<mlir::tt::ttcore::TileType>(type.getElementType()))
    return type;

  auto layout = getL1ShardedLayout(ctx, type);
  constexpr std::array<int64_t, 2> defaultTileShape =
      mlir::tt::ttcore::TileType::getDefaultShape();
  SmallVector<int64_t> tileShape(defaultTileShape.begin(),
                                 defaultTileShape.end());
  Type tileElementType =
      mlir::tt::ttcore::TileType::get(type.getElementType(), tileShape);
  SmallVector<int64_t> unshardedShape = layout.getPhysicalShape(tileShape);
  SmallVector<int64_t> gridShape(unshardedShape.size(), 1);
  SmallVector<int64_t> deviceShape = layout.getDeviceShape(gridShape, tileShape);
  return RankedTensorType::get(deviceShape, tileElementType, layout);
}

static RankedTensorType getShardTensorType(RankedTensorType type) {
  auto layout = cast<mlir::tt::ttcore::MetalLayoutAttr>(type.getEncoding());
  return RankedTensorType::get(layout.getShardShape(type), type.getElementType());
}

static mlir::tt::ttcore::GridAttr getFrontendGrid(MLIRContext *ctx) {
  return mlir::tt::ttcore::GridAttr::get(ctx, {1, 1});
}

static SmallVector<AffineMap> getIdentityAffineMapsArray(OpBuilder &builder,
                                                         size_t arity,
                                                         size_t rank) {
  return SmallVector<AffineMap>(arity, builder.getMultiDimIdentityMap(rank));
}

static SmallVector<mlir::Attribute>
getElementwiseIteratorTypesArray(OpBuilder &builder, size_t rank) {
  auto parallel = mlir::tt::ttcore::IteratorTypeAttr::get(
      builder.getContext(), mlir::tt::ttcore::IteratorType::Parallel);
  return SmallVector<mlir::Attribute>(rank, parallel);
}

static SmallVector<mlir::utils::IteratorType>
getLinalgParallelIterators(size_t rank) {
  return SmallVector<mlir::utils::IteratorType>(
      rank, mlir::utils::IteratorType::parallel);
}

static SmallVector<AffineMap> getMatmulAffineMapsArray(OpBuilder &builder,
                                                       size_t rank) {
  assert(rank >= 2 && "matmul rank must be >= 2");
  MLIRContext *ctx = builder.getContext();
  SmallVector<AffineExpr> lhsExprs;
  SmallVector<AffineExpr> rhsExprs;
  SmallVector<AffineExpr> outExprs;

  for (unsigned i = 0; i < rank - 2; ++i) {
    lhsExprs.push_back(builder.getAffineDimExpr(i));
    rhsExprs.push_back(builder.getAffineDimExpr(i));
    outExprs.push_back(builder.getAffineDimExpr(i));
  }

  lhsExprs.push_back(builder.getAffineDimExpr(rank - 2));
  lhsExprs.push_back(builder.getAffineDimExpr(rank));
  rhsExprs.push_back(builder.getAffineDimExpr(rank));
  rhsExprs.push_back(builder.getAffineDimExpr(rank - 1));
  outExprs.push_back(builder.getAffineDimExpr(rank - 2));
  outExprs.push_back(builder.getAffineDimExpr(rank - 1));

  return SmallVector<AffineMap>{
      AffineMap::get(rank + 1, 0, lhsExprs, ctx),
      AffineMap::get(rank + 1, 0, rhsExprs, ctx),
      AffineMap::get(rank + 1, 0, outExprs, ctx)};
}

static SmallVector<mlir::Attribute>
getMatmulIteratorTypesArray(OpBuilder &builder, size_t rank) {
  assert(rank >= 2 && "matmul rank must be >= 2");
  auto parallel = mlir::tt::ttcore::IteratorTypeAttr::get(
      builder.getContext(), mlir::tt::ttcore::IteratorType::Parallel);
  auto reduction = mlir::tt::ttcore::IteratorTypeAttr::get(
      builder.getContext(), mlir::tt::ttcore::IteratorType::Reduction);

  SmallVector<mlir::Attribute> result(rank, parallel);
  result.push_back(reduction);
  return result;
}

static SmallVector<mlir::utils::IteratorType>
getLinalgMatmulIterators(size_t rank) {
  SmallVector<mlir::utils::IteratorType> result(
      rank, mlir::utils::IteratorType::parallel);
  result.push_back(mlir::utils::IteratorType::reduction);
  return result;
}

static SmallVector<Value>
createGenericBlockArguments(OpBuilder &builder, Location loc,
                            mlir::tt::d2m::GenericOp generic,
                            TypeRange inputs, TypeRange outputs) {
  SmallVector<Value> operands;

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto shardType = getShardTensorType(cast<RankedTensorType>(inputs[i]));
    Value genericOperand = generic->getOperand(i);
    SmallVector<Value> indices =
        mlir::tt::d2m::utils::buildGridIndices(builder, loc,
                                               generic.getIndexingMap(i));
    Value buffer =
        builder.create<tensor::EmptyOp>(loc, shardType.getShape(),
                                        shardType.getElementType())
            .getResult();
    Value loaded = builder
                       .create<mlir::tt::d2m::RemoteLoadOp>(
                           loc, shardType, buffer, genericOperand, indices)
                       .getResult();
    operands.push_back(loaded);
  }

  for (Type output : outputs) {
    auto shardType = getShardTensorType(cast<RankedTensorType>(output));
    operands.push_back(builder.create<tensor::EmptyOp>(
        loc, shardType.getShape(), shardType.getElementType()));
  }

  return operands;
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
  mixedOffsets.reserve(composedOffsets.size());
  mixedSizes.reserve(sizes.size());
  mixedStrides.reserve(strides.size());
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
static void updateTensorResultEncodings(Operation *op, MLIRContext *ctx) {
  Attribute encoding;
  for (Value operand : op->getOperands()) {
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (operandType && operandType.getEncoding()) {
      encoding = operandType.getEncoding();
      break;
    }
  }

  if (!encoding && !isa<tensor::EmptyOp>(op))
    return;

  for (Value result : op->getResults()) {
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType || resultType.getEncoding())
      continue;

    result.setType(RankedTensorType::get(resultType.getShape(),
                                         resultType.getElementType(),
                                         getL1ShardedLayout(ctx, resultType)));
  }
}

static BlockArgument getRootBlockArgument(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (def->getNumOperands() != 1)
      return {};
    value = def->getOperand(0);
  }
  return dyn_cast<BlockArgument>(value);
}

static void annotateForwardFunc(func::FuncOp func, MapsProgramInfo &program,
                                MLIRContext *ctx) {
  func->setAttr("tt.function_type", StringAttr::get(ctx, "forward_device"));
  if (func.getSymName() == "main")
    func.setSymName("forward");

  DenseSet<unsigned> parameterArgs;
  if (program.init) {
    program.init.walk([&](maps::SendOp send) {
      if (BlockArgument arg = getRootBlockArgument(send.getValue()))
        parameterArgs.insert(arg.getArgNumber());
    });
  }

  for (BlockArgument arg : func.getArguments()) {
    auto argType = parameterArgs.contains(arg.getArgNumber())
                       ? mlir::tt::ttcore::ArgumentType::Parameter
                       : mlir::tt::ttcore::ArgumentType::Input;
    func.setArgAttr(arg.getArgNumber(), mlir::tt::ttcore::ArgumentTypeAttr::name,
                    mlir::tt::ttcore::ArgumentTypeAttr::get(ctx, argType));
  }
}

static bool hasStaticSlice(tensor::ExtractSliceOp slice) {
  return llvm::none_of(slice.getStaticOffsets(), [](int64_t value) {
    return ShapedType::isDynamic(value);
  }) &&
         llvm::none_of(slice.getStaticSizes(), [](int64_t value) {
           return ShapedType::isDynamic(value);
         }) &&
         llvm::none_of(slice.getStaticStrides(), [](int64_t value) {
           return ShapedType::isDynamic(value);
         });
}

static bool isIdentityStaticSlice(ArrayRef<int64_t> offsets,
                                  ArrayRef<int64_t> sizes,
                                  ArrayRef<int64_t> strides,
                                  RankedTensorType type) {
  return llvm::all_of(llvm::enumerate(offsets), [](auto it) {
           return it.value() == 0;
         }) &&
         llvm::all_of(strides, [](int64_t stride) { return stride == 1; }) &&
         llvm::equal(sizes, type.getShape());
}

static bool isParameterArgument(func::FuncOp func, BlockArgument arg) {
  auto attr = func.getArgAttrOfType<mlir::tt::ttcore::ArgumentTypeAttr>(
      arg.getArgNumber(), mlir::tt::ttcore::ArgumentTypeAttr::name);
  return attr && attr.getValue() == mlir::tt::ttcore::ArgumentType::Parameter;
}

static std::string getSliceKey(BlockArgument arg, tensor::ExtractSliceOp slice) {
  std::string key = std::to_string(arg.getArgNumber());
  auto appendList = [&](ArrayRef<int64_t> values) {
    key.push_back(':');
    for (int64_t value : values) {
      key.append(std::to_string(value));
      key.push_back(',');
    }
  };
  appendList(slice.getStaticOffsets());
  appendList(slice.getStaticSizes());
  appendList(slice.getStaticStrides());
  return key;
}

static LogicalResult liftParameterSlices(func::FuncOp func) {
  struct SliceInfo {
    BlockArgument arg;
    tensor::ExtractSliceOp slice;
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> strides;
  };

  SmallVector<SliceInfo> slices;
  func.walk([&](tensor::ExtractSliceOp slice) {
    auto arg = dyn_cast<BlockArgument>(slice.getSource());
    auto sourceType = dyn_cast<RankedTensorType>(slice.getSource().getType());
    if (!arg || !sourceType || !hasStaticSlice(slice) ||
        !isParameterArgument(func, arg))
      return;
    slices.push_back({arg,
                      slice,
                      SmallVector<int64_t>(slice.getStaticOffsets()),
                      SmallVector<int64_t>(slice.getStaticSizes()),
                      SmallVector<int64_t>(slice.getStaticStrides())});
  });

  llvm::sort(slices, [](const SliceInfo &lhs, const SliceInfo &rhs) {
    if (lhs.arg.getArgNumber() != rhs.arg.getArgNumber())
      return lhs.arg.getArgNumber() < rhs.arg.getArgNumber();
    if (lhs.offsets != rhs.offsets)
      return std::lexicographical_compare(lhs.offsets.begin(), lhs.offsets.end(),
                                          rhs.offsets.begin(), rhs.offsets.end());
    if (lhs.sizes != rhs.sizes)
      return std::lexicographical_compare(lhs.sizes.begin(), lhs.sizes.end(),
                                          rhs.sizes.begin(), rhs.sizes.end());
    return std::lexicographical_compare(lhs.strides.begin(), lhs.strides.end(),
                                        rhs.strides.begin(), rhs.strides.end());
  });

  llvm::StringMap<BlockArgument> liftedArgs;
  for (const SliceInfo &info : slices) {
    tensor::ExtractSliceOp slice = info.slice;
    auto arg = info.arg;
    auto sourceType = cast<RankedTensorType>(slice.getSource().getType());
    auto offsets = slice.getStaticOffsets();
    auto sizes = slice.getStaticSizes();
    auto strides = slice.getStaticStrides();
    if (isIdentityStaticSlice(offsets, sizes, strides, sourceType)) {
      slice.replaceAllUsesWith(slice.getSource());
      continue;
    }

    std::string key = getSliceKey(arg, slice);
    auto it = liftedArgs.find(key);
    if (it == liftedArgs.end()) {
      unsigned argIndex = func.getNumArguments();
      DictionaryAttr attrs = func.getArgAttrDict(arg.getArgNumber());
      if (failed(func.insertArgument(argIndex, slice.getType(), attrs,
                                     slice.getLoc())))
        return failure();
      it = liftedArgs.try_emplace(key, func.getArgument(argIndex)).first;
    }
    slice.replaceAllUsesWith(it->second);
  }

  return success();
}

static LogicalResult eraseUnusedParameterArgs(func::FuncOp func) {
  SmallVector<unsigned> argsToErase;
  for (BlockArgument arg : func.getArguments()) {
    if (arg.use_empty() && isParameterArgument(func, arg))
      argsToErase.push_back(arg.getArgNumber());
  }

  for (auto it = argsToErase.rbegin(); it != argsToErase.rend(); ++it) {
    if (failed(func.eraseArgument(*it)))
      return failure();
  }

  return success();
}

static LogicalResult reorderParameterArgsByFirstUse(func::FuncOp func) {
  DenseMap<Operation *, unsigned> opOrder;
  unsigned nextOrder = 0;
  func.walk([&](Operation *op) { opOrder[op] = nextOrder++; });

  struct ArgUseInfo {
    BlockArgument arg;
    unsigned firstUseOrder;
  };

  SmallVector<ArgUseInfo> parameterArgs;
  for (BlockArgument arg : func.getArguments()) {
    if (!isParameterArgument(func, arg))
      continue;

    unsigned firstUseOrder = std::numeric_limits<unsigned>::max();
    for (OpOperand &use : arg.getUses())
      firstUseOrder = std::min(firstUseOrder, opOrder[use.getOwner()]);
    parameterArgs.push_back({arg, firstUseOrder});
  }

  auto desiredOrder = parameterArgs;
  llvm::sort(desiredOrder, [](const ArgUseInfo &lhs, const ArgUseInfo &rhs) {
    if (lhs.firstUseOrder != rhs.firstUseOrder)
      return lhs.firstUseOrder < rhs.firstUseOrder;
    return lhs.arg.getArgNumber() < rhs.arg.getArgNumber();
  });

  if (llvm::equal(parameterArgs, desiredOrder,
                  [](const ArgUseInfo &lhs, const ArgUseInfo &rhs) {
                    return lhs.arg == rhs.arg;
                  }))
    return success();

  SmallVector<BlockArgument> replacementArgs;
  replacementArgs.reserve(desiredOrder.size());
  for (const ArgUseInfo &info : desiredOrder) {
    unsigned argIndex = func.getNumArguments();
    DictionaryAttr attrs = func.getArgAttrDict(info.arg.getArgNumber());
    if (failed(func.insertArgument(argIndex, info.arg.getType(), attrs,
                                   func.getLoc())))
      return failure();
    replacementArgs.push_back(func.getArgument(argIndex));
  }

  for (auto [oldInfo, newArg] : llvm::zip_equal(desiredOrder, replacementArgs))
    oldInfo.arg.replaceAllUsesWith(newArg);

  SmallVector<unsigned> argsToErase;
  for (const ArgUseInfo &info : parameterArgs)
    argsToErase.push_back(info.arg.getArgNumber());
  for (auto it = argsToErase.rbegin(); it != argsToErase.rend(); ++it) {
    if (failed(func.eraseArgument(*it)))
      return failure();
  }

  return success();
}

static bool foldIdentityMaterialization(tensor::InsertSliceOp insertSlice) {
  auto sourceType = dyn_cast<RankedTensorType>(insertSlice.getSource().getType());
  auto resultType = dyn_cast<RankedTensorType>(insertSlice.getType());
  auto empty = insertSlice.getDest().getDefiningOp<tensor::EmptyOp>();
  if (!sourceType || !resultType || !empty || sourceType != resultType)
    return false;

  if (!isIdentityStaticSlice(insertSlice.getStaticOffsets(),
                             insertSlice.getStaticSizes(),
                             insertSlice.getStaticStrides(), sourceType))
    return false;

  insertSlice.replaceAllUsesWith(insertSlice.getSource());
  return true;
}

static bool foldIdentitySlice(tensor::ExtractSliceOp slice) {
  auto sourceType = dyn_cast<RankedTensorType>(slice.getSource().getType());
  if (!sourceType || !hasStaticSlice(slice))
    return false;
  if (!isIdentityStaticSlice(slice.getStaticOffsets(), slice.getStaticSizes(),
                             slice.getStaticStrides(), sourceType))
    return false;
  slice.replaceAllUsesWith(slice.getSource());
  return true;
}

struct ComputeTileOpLowering : public OpConversionPattern<maps::TileOp> {
  DenseMap<Attribute, ChannelInfo> &channels;
  DenseMap<Attribute, Value> &channelValues;
  DenseMap<Attribute, SmallVector<Value>> &channelValueLists;
  DenseMap<Attribute, Value> &slotValues;

  ComputeTileOpLowering(MLIRContext *ctx,
                        DenseMap<Attribute, ChannelInfo> &channels,
                        DenseMap<Attribute, Value> &channelValues,
                        DenseMap<Attribute, SmallVector<Value>> &channelValueLists,
                        DenseMap<Attribute, Value> &slotValues)
      : OpConversionPattern<maps::TileOp>(ctx), channels(channels),
        channelValues(channelValues), channelValueLists(channelValueLists),
        slotValues(slotValues) {}

  LogicalResult matchAndRewrite(maps::TileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (isInitStorageTile(op))
      return failure();

    SmallVector<maps::SendOp> sends;
    for (Operation &nested : op.getBody().front()) {
      if (auto send = dyn_cast<maps::SendOp>(nested))
        sends.push_back(send);
    }

    if (sends.empty())
      return op.emitOpError("expected compute tile to produce at least one maps.send");

    MLIRContext *ctx = rewriter.getContext();
    auto threads = rewriter.getArrayAttr(
        mlir::tt::d2m::ThreadAttr::get(ctx, mlir::tt::d2m::ThreadType::Unified));

    auto getSourceValue = [&](Value value) -> FailureOr<Value> {
      if (auto recv = value.getDefiningOp<maps::RecvOp>()) {
        auto it = channels.find(recv.getChannelAttr());
        if (it == channels.end())
          return recv.emitOpError("expected known channel");
        if (it->second.srcHartId == -1)
          return it->second.send.getValue();
        auto valueIt = channelValues.find(recv.getChannelAttr());
        if (valueIt == channelValues.end())
          return recv.emitOpError("expected lowered channel value");
        return valueIt->second;
      }
      if (auto load = value.getDefiningOp<maps::LoadOp>()) {
        auto it = slotValues.find(load.getSlotAttr());
        if (it == slotValues.end())
          return load.emitOpError("expected lowered slot value");
        return it->second;
      }
      return value;
    };

    auto getTiledValue = [&](Value value) -> FailureOr<Value> {
      auto source = getSourceValue(value);
      if (failed(source))
        return failure();
      value = *source;

      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type)
        return failure();
      if (type.getEncoding() && isa<mlir::tt::ttcore::TileType>(type.getElementType()))
        return value;

      auto tiledType = getTiledDeviceTensorType(ctx, type);
      rewriter.setInsertionPoint(op);
      auto output = rewriter.create<mlir::tt::d2m::EmptyOp>(
          value.getLoc(), tiledType.getShape(), tiledType.getElementType(),
          tiledType.getEncoding());
      auto toLayout = rewriter.create<mlir::tt::d2m::ToLayoutOp>(
          value.getLoc(), value, output.getResult());
      return toLayout.getResult(0);
    };

    auto createGeneric = [&](Location loc, ArrayRef<Value> inputs,
                             RankedTensorType outputType,
                             ArrayRef<AffineMap> indexingMaps,
                             ArrayRef<Attribute> iteratorTypes,
                             Value initialOutput = Value())
        -> mlir::tt::d2m::GenericOp {
      SmallVector<int64_t> blockFactors(iteratorTypes.size(), 1);
      SmallVector<Attribute> blockFactorAttrs;
      blockFactorAttrs.reserve(blockFactors.size());
      for (int64_t factor : blockFactors)
        blockFactorAttrs.push_back(rewriter.getI64IntegerAttr(factor));
      rewriter.setInsertionPoint(op);
      Value output = initialOutput;
      if (!output) {
        output = rewriter.create<mlir::tt::d2m::EmptyOp>(
            loc, outputType.getShape(), outputType.getElementType(),
            outputType.getEncoding());
      }
      auto generic = rewriter.create<mlir::tt::d2m::GenericOp>(
          loc, TypeRange{outputType}, inputs, ValueRange{output},
          ValueRange{}, getFrontendGrid(ctx),
          rewriter.getArrayAttr(blockFactorAttrs),
          rewriter.getAffineMapArrayAttr(indexingMaps),
          rewriter.getArrayAttr(iteratorTypes), threads,
          /*fabricConnectionConfig=*/nullptr, /*regionsCount=*/1);
      generic->setAttr("maps.tile_id", op.getTileIdAttr());
      generic->setAttr("maps.coords", op.getCoordsAttr());
      return generic;
    };

    auto populateElementwiseGeneric = [&](mlir::tt::d2m::GenericOp generic,
                                          function_ref<Value(OpBuilder &, Location,
                                                             ValueRange)> bodyBuilder) {
      auto insertPoint = rewriter.saveInsertionPoint();
      rewriter.startOpModification(generic);
      {
        Region &region = generic->getRegion(0);
        Block *block = rewriter.createBlock(&region);
        SmallVector<Value> blockArgs = createGenericBlockArguments(
            rewriter, generic.getLoc(), generic,
            TypeRange(generic.getInputs().getTypes()),
            TypeRange(generic.getOutputs().getTypes()));
        rewriter.setInsertionPointToEnd(block);

        auto shardType = cast<RankedTensorType>(blockArgs.back().getType());
        auto linalg = rewriter.create<linalg::GenericOp>(
            generic.getLoc(), TypeRange{shardType},
            ValueRange(blockArgs).drop_back(),
            ValueRange{blockArgs.back()},
            getIdentityAffineMapsArray(rewriter, blockArgs.size(),
                                       shardType.getRank()),
            getLinalgParallelIterators(shardType.getRank()),
            [&](OpBuilder &builder, Location loc, ValueRange bbArgs) {
              Value yielded = bodyBuilder(builder, loc, bbArgs);
              builder.create<linalg::YieldOp>(loc, yielded);
            });

        SmallVector<Value> outputIndices =
            mlir::tt::d2m::utils::buildGridIndices(rewriter, generic.getLoc(),
                                                   generic.getIndexingMap(
                                                       generic.getNumOperands() - 1));
        Value stored = rewriter
                           .create<mlir::tt::d2m::RemoteStoreOp>(
                               generic.getLoc(), generic.getResult(0).getType(),
                               generic.getOutputs().front(), outputIndices,
                               linalg.getResult(0))
                           .getResult();
        rewriter.create<mlir::tt::d2m::YieldOp>(generic.getLoc(), stored);
      }
      rewriter.finalizeOpModification(generic);
      rewriter.restoreInsertionPoint(insertPoint);
    };

    auto createExpGeneric = [&](Location loc, Value input,
                                RankedTensorType logicalOutputType)
        -> mlir::tt::d2m::GenericOp {
      auto outputType = getTiledDeviceTensorType(ctx, logicalOutputType);
      auto generic = createGeneric(
          loc, {input}, outputType,
          getIdentityAffineMapsArray(rewriter, /*arity=*/2, /*rank=*/2),
          getElementwiseIteratorTypesArray(rewriter, /*rank=*/2));
      populateElementwiseGeneric(generic,
                                 [&](OpBuilder &builder, Location bodyLoc,
                                     ValueRange bbArgs) {
                                   return builder
                                       .create<mlir::tt::d2m::TileExpOp>(
                                           bodyLoc, bbArgs.front().getType(),
                                           bbArgs.front())
                                       .getResult();
                                 });
      return generic;
    };

    auto createAddGeneric = [&](Location loc, Value lhs, Value rhs,
                                RankedTensorType logicalOutputType)
        -> mlir::tt::d2m::GenericOp {
      auto outputType = getTiledDeviceTensorType(ctx, logicalOutputType);
      auto generic = createGeneric(
          loc, {lhs, rhs}, outputType,
          getIdentityAffineMapsArray(rewriter, /*arity=*/3, /*rank=*/2),
          getElementwiseIteratorTypesArray(rewriter, /*rank=*/2));
      populateElementwiseGeneric(generic,
                                 [&](OpBuilder &builder, Location bodyLoc,
                                     ValueRange bbArgs) {
                                   return builder
                                       .create<mlir::tt::d2m::TileAddOp>(
                                           bodyLoc, bbArgs.front().getType(),
                                           bbArgs[0], bbArgs[1])
                                       .getResult();
                                 });
      return generic;
    };

    auto createMatmulGeneric = [&](Location loc, Value lhs, Value rhs,
                                   RankedTensorType logicalOutputType)
        -> mlir::tt::d2m::GenericOp {
      auto outputType = getTiledDeviceTensorType(ctx, logicalOutputType);
      auto generic = createGeneric(
          loc, {lhs, rhs}, outputType,
          getMatmulAffineMapsArray(rewriter, /*rank=*/2),
          getMatmulIteratorTypesArray(rewriter, /*rank=*/2));

      auto insertPoint = rewriter.saveInsertionPoint();
      rewriter.startOpModification(generic);
      {
        Region &region = generic.getRegion(0);
        Block *block = rewriter.createBlock(&region);
        SmallVector<Value> blockArgs = createGenericBlockArguments(
            rewriter, generic.getLoc(), generic,
            TypeRange(generic.getInputs().getTypes()),
            TypeRange(generic.getOutputs().getTypes()));
        rewriter.setInsertionPointToEnd(block);

        auto shardType = cast<RankedTensorType>(blockArgs.back().getType());
        auto linalg = rewriter.create<linalg::GenericOp>(
            generic.getLoc(), TypeRange{shardType},
            ValueRange(blockArgs).take_front(2), ValueRange{blockArgs.back()},
            getMatmulAffineMapsArray(rewriter, /*rank=*/2),
            getLinalgMatmulIterators(/*rank=*/2),
            [&](OpBuilder &builder, Location bodyLoc, ValueRange bbArgs) {
              Value yielded = builder
                                  .create<mlir::tt::d2m::TileMatmulOp>(
                                      bodyLoc, bbArgs.back().getType(),
                                      bbArgs[0], bbArgs[1], bbArgs[2])
                                  .getResult();
              builder.create<linalg::YieldOp>(bodyLoc, yielded);
            });

        SmallVector<Value> outputIndices =
            mlir::tt::d2m::utils::buildGridIndices(rewriter, generic.getLoc(),
                                                   generic.getIndexingMap(2));
        Value stored = rewriter
                           .create<mlir::tt::d2m::RemoteStoreOp>(
                               generic.getLoc(), generic.getResult(0).getType(),
                               generic.getOutputs().front(), outputIndices,
                               linalg.getResult(0))
                           .getResult();
        rewriter.create<mlir::tt::d2m::YieldOp>(generic.getLoc(), stored);
      }
      rewriter.finalizeOpModification(generic);
      rewriter.restoreInsertionPoint(insertPoint);
      return generic;
    };

    for (maps::SendOp send : sends) {
      auto outputType = dyn_cast<RankedTensorType>(send.getValue().getType());
      if (!outputType)
        return send.emitOpError("expected tensor send value");

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
          return failure();

        Value rhsValue = *rhsSource;
        if (sendSlice) {
          auto rhsType = cast<RankedTensorType>(rhsValue.getType());
          rewriter.setInsertionPoint(op);
          SmallVector<int64_t> offsets(rhsType.getRank(), 0);
          SmallVector<int64_t> sizes(rhsType.getShape().begin(),
                                     rhsType.getShape().end());
          offsets.back() = sendSlice.getStaticOffsets().back();
          sizes.back() = sendSlice.getStaticSizes().back();
          rhsValue =
              createDirectStaticSlice(rewriter, send.getLoc(), rhsValue, offsets,
                                      sizes);
          rhsValue =
              materializeContiguousTensor(rewriter, send.getLoc(), rhsValue);
        }

        auto tiledLhs = getTiledValue(*lhsSource);
        auto tiledRhs = getTiledValue(rhsValue);
        if (failed(tiledLhs) || failed(tiledRhs))
          return send.emitOpError("failed to materialize tiled matmul inputs");

        auto matmulGeneric =
            createMatmulGeneric(send.getLoc(), *tiledLhs, *tiledRhs, outputType);
        auto expGeneric =
            createExpGeneric(send.getLoc(), matmulGeneric.getResult(0), outputType);
        channelValues[send.getChannelAttr()] = expGeneric.getResult(0);
        channelValueLists[send.getChannelAttr()] = {expGeneric.getResult(0)};
        continue;
      }

      if (auto add = sendValue.getDefiningOp<arith::AddFOp>()) {
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
            return failure();

          for (Value concatInput : concat.getInputs()) {
            auto inputSource = getTiledValue(concatInput);
            if (failed(inputSource))
              return send.emitOpError("failed to materialize concat shard");

            auto logicalInputType =
                cast<RankedTensorType>(concatInput.getType());
            SmallVector<int64_t> staticOffsets(logicalInputType.getRank(), 0);
            SmallVector<int64_t> staticSizes(logicalInputType.getShape().begin(),
                                             logicalInputType.getShape().end());
            staticOffsets[dim] = offset;
            offset += logicalInputType.getDimSize(dim);

            rewriter.setInsertionPoint(op);
            Value biasSlice = createDirectStaticSlice(
                rewriter, send.getLoc(), *biasSource, staticOffsets,
                staticSizes);
            biasSlice =
                materializeContiguousTensor(rewriter, send.getLoc(), biasSlice);
            auto tiledBias = getTiledValue(biasSlice);
            if (failed(tiledBias))
              return send.emitOpError("failed to materialize bias shard");

            auto addGeneric = createAddGeneric(send.getLoc(), *inputSource,
                                               *tiledBias, logicalInputType);
            shardResults.push_back(addGeneric.getResult(0));
          }

          channelValues[send.getChannelAttr()] = shardResults.front();
          channelValueLists[send.getChannelAttr()] = shardResults;
          continue;
        }

        auto lhsSource = getTiledValue(add.getLhs());
        auto rhsSource = getTiledValue(add.getRhs());
        if (failed(lhsSource) || failed(rhsSource))
          return send.emitOpError("failed to materialize add inputs");
        auto addGeneric =
            createAddGeneric(send.getLoc(), *lhsSource, *rhsSource, outputType);
        channelValues[send.getChannelAttr()] = addGeneric.getResult(0);
        channelValueLists[send.getChannelAttr()] = {addGeneric.getResult(0)};
        continue;
      }

      return send.emitOpError("unsupported compute tile send pattern");
    }

    rewriter.eraseOp(op);
    return success();
  }
};


static FailureOr<MapsProgramInfo> collectMapsProgram(ModuleOp module){
  MapsProgramInfo info;

  // register init
  module.walk([&](maps::InitOp op){
    if (info.init)
      op.emitError("expected at most one maps.init");
    info.init = op;
  });


  module.walk([&](maps::RunOp op){
    if (info.run)
      op.emitError("expected at most one maps.init");
    info.run = op;
  });


  module.walk([&](maps::StageOp stage){
    StageInfo stageInfo;
    stageInfo.op = stage;
    stageInfo.stageId = static_cast<int64_t>(stage.getStageId());
    stage.walk([&](maps::TileOp tile){
      TileInfo tileInfo;
      tileInfo.op = tile;
      tileInfo.tileId = static_cast<int64_t>(tile.getTileId());
      tileInfo.coords = static_cast<SmallVector<int64_t>>(tile.getCoords());
      stageInfo.tiles.push_back(tileInfo);
    });
    info.stages.push_back(stageInfo);

  }); 

  module.walk([&](maps::SendOp op){
    FlatSymbolRefAttr channelName = op.getChannelAttr();

    ChannelInfo channel;
    channel.name = channelName;
    channel.dstHartId = static_cast<int64_t>(op.getDstHartid());
    channel.srcHartId = static_cast<int64_t>(op.getSrcHartid());
    channel.type = dyn_cast<RankedTensorType>(op.getValue().getType());
    channel.send = op;

    info.channels[channelName] = channel;
  });

  module.walk([&](maps::RecvOp op){
    Attribute channelName = op.getChannelAttr();
    auto it = info.channels.find(channelName);

    if (it == info.channels.end()) {
      op.emitError("recv without matching send for channel ") << channelName;
      return;
    }
    it->second.recvs.push_back(op);
  });

  module.walk([&](maps::TileOp tile) {
    if (!isInitStorageTile(tile))
      return;

    for (Operation &nested : tile.getBody().front()) {
      auto store = dyn_cast<maps::StoreOp>(nested);
      if (!store)
        continue;

      auto recv = store.getValue().getDefiningOp<maps::RecvOp>();
      if (!recv)
        continue;

      auto channelIt = info.channels.find(recv.getChannelAttr());
      if (channelIt == info.channels.end())
        continue;

      info.slotValues[store.getSlotAttr()] = channelIt->second.send.getValue();
    }
  });

  return info;
}


namespace {
struct InitOpLowering : public OpConversionPattern<maps::InitOp> {
  using OpConversionPattern<maps::InitOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(maps::InitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();

    rewriter.inlineBlockBefore(&body, op);
    rewriter.eraseOp(op);
    return success();
  }
  };

struct RunOpLowering : public OpConversionPattern<maps::RunOp> {
  using OpConversionPattern<maps::RunOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(maps::RunOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();

    rewriter.inlineBlockBefore(&body, op);
    rewriter.eraseOp(op);
    return success();
  }
};

struct StageOpLowering : public OpConversionPattern<maps::StageOp> {
  using OpConversionPattern<maps::StageOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(maps::StageOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();

    rewriter.inlineBlockBefore(&body, op);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace


  struct InitTileOpLowering : public OpConversionPattern<maps::TileOp> {
    using OpConversionPattern<maps::TileOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(maps::TileOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
      if (!isInitStorageTile(op))
        return failure();

      SmallVector<Operation *> nestedOps;
      for (Operation &nested : llvm::make_early_inc_range(op.getBody().front()))
        nestedOps.push_back(&nested);

      for (Operation *nested : nestedOps)
        rewriter.eraseOp(nested);

      rewriter.eraseOp(op);
      return success();
    }
  };
}


struct LoadOpLowering : public OpConversionPattern<maps::LoadOp> {
  DenseMap<Attribute, Value> &slotValues;

  LoadOpLowering(MLIRContext *ctx, DenseMap<Attribute, Value> &slotValues)
      : OpConversionPattern<maps::LoadOp>(ctx), slotValues(slotValues) {}

  LogicalResult matchAndRewrite(maps::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto it = slotValues.find(op.getSlotAttr());
    if (it == slotValues.end())
      return failure();

    rewriter.replaceOp(op, it->second);
    return success();
  }
};

struct RecvOpLowering : public OpConversionPattern<maps::RecvOp> {
  DenseMap<Attribute, ChannelInfo> &channels;

  RecvOpLowering(MLIRContext *ctx, DenseMap<Attribute, ChannelInfo> &channels)
      : OpConversionPattern<maps::RecvOp>(ctx), channels(channels) {}

  LogicalResult matchAndRewrite(maps::RecvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto it = channels.find(op.getChannelAttr());
    if (it == channels.end() || it->second.srcHartId != -1)
      return failure();

    rewriter.replaceOp(op, it->second.send.getValue());
    return success();
  }
};

struct SendOpLowering : public OpConversionPattern<maps::SendOp> {
  using OpConversionPattern<maps::SendOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(maps::SendOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (static_cast<int64_t>(op.getSrcHartid()) != -1)
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};


namespace {
struct ConvertMapsToD2MPass: impl::ConvertMapsToD2MBase<ConvertMapsToD2MPass> {
  void runOnOperation() override {

    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    FailureOr<MapsProgramInfo> info = collectMapsProgram(module);
    if (failed(info)) {
      signalPassFailure();
    return;
    }
    MapsProgramInfo &program = *info;

    module.walk([&](func::FuncOp func) {
      annotateForwardFunc(func, program, ctx);
    });
    
    ConversionTarget target(*ctx);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    // Target dialects.
    target.addLegalDialect<mlir::tt::ttcore::TTCoreDialect>();
    target.addLegalDialect<mlir::tt::d2m::D2MDialect>();

    // Source dialect
    target.addIllegalDialect<mlir::maps::MapsDialect>();

    target.addDynamicallyLegalOp<maps::SendOp>(
        [](maps::SendOp op) {
          return static_cast<int64_t>(op.getSrcHartid()) != -1;
        });
    target.addDynamicallyLegalOp<maps::RecvOp>(
        [&](maps::RecvOp op) {
          auto it = program.channels.find(op.getChannelAttr());
          return it == program.channels.end() || it->second.srcHartId != -1;
        });
    target.addLegalOp<maps::StoreOp>(); // temporary
    target.addDynamicallyLegalOp<maps::LoadOp>(
        [&](maps::LoadOp op) {
          return !program.slotValues.contains(op.getSlotAttr());
        });


    // mark as legal ops that aren't mentioned
    target.markUnknownOpDynamicallyLegal([](Operation *){
      return true;
    });

    // register patterns
    DenseMap<Attribute, Value> channelValues;
    DenseMap<Attribute, SmallVector<Value>> channelValueLists;
    RewritePatternSet patterns(ctx);
    patterns.add<InitOpLowering, RunOpLowering, StageOpLowering,
                 InitTileOpLowering>(ctx);
    patterns.add<LoadOpLowering>(ctx, program.slotValues);
    patterns.add<RecvOpLowering>(ctx, program.channels);
    patterns.add<SendOpLowering>(ctx);
    patterns.add<ComputeTileOpLowering>(ctx, program.channels, channelValues,
                                        channelValueLists, program.slotValues);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))){
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(ctx);
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(liftParameterSlices(func))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<maps::RecvOp> recvsToReplace;
    SmallVector<Value> outputValues;
    SmallVector<Type> outputTypes;
    module.walk([&](maps::RecvOp op) {
      auto channelIt = program.channels.find(op.getChannelAttr());
      if (channelValues.contains(op.getChannelAttr()) &&
          channelIt != program.channels.end() && channelIt->second.dstHartId == -1) {
        auto listIt = channelValueLists.find(op.getChannelAttr());
        ValueRange values = listIt == channelValueLists.end()
                                ? ValueRange(channelValues[op.getChannelAttr()])
                                : ValueRange(listIt->second);
        for (Value output : values) {
          outputValues.push_back(output);
          outputTypes.push_back(output.getType());
        }
        recvsToReplace.push_back(op);
      }
    });

    if (!outputValues.empty()) {
      auto func = outputValues.front().getParentRegion()
                      ->getParentOfType<func::FuncOp>();
      if (!func) {
        module.emitError("failed to find enclosing function for lowered outputs");
        signalPassFailure();
        return;
      }

      SmallVector<func::ReturnOp> returnsToReplace;
      func.walk([&](func::ReturnOp op) {
        if (op.getNumOperands() == 0)
          returnsToReplace.push_back(op);
      });
      for (func::ReturnOp ret : returnsToReplace) {
        rewriter.setInsertionPoint(ret);
        SmallVector<Value> returnedValues;
        SmallVector<Type> returnedTypes;
        for (Value output : outputValues) {
          auto type = dyn_cast<RankedTensorType>(output.getType());
          if (!type || !type.getEncoding()) {
            returnedValues.push_back(output);
            returnedTypes.push_back(output.getType());
            continue;
          }

          auto layout = cast<mlir::tt::ttcore::MetalLayoutAttr>(type.getEncoding());
          Type hostElementType = type.getElementType();
          if (auto tileType =
                  dyn_cast<mlir::tt::ttcore::TileType>(hostElementType))
            hostElementType = tileType.getElementType();
          auto hostType =
              RankedTensorType::get(layout.getLogicalShape(), hostElementType);
          auto hostOutput = rewriter.create<mlir::tt::d2m::EmptyOp>(
              output.getLoc(), hostType.getShape(), hostType.getElementType(),
              /*encoding=*/nullptr);
          auto toHost = rewriter.create<mlir::tt::d2m::ToLayoutOp>(
              output.getLoc(), output, hostOutput.getResult());
          returnedValues.push_back(toHost.getResult(0));
          returnedTypes.push_back(hostType);
        }

        func.setFunctionType(rewriter.getFunctionType(func.getArgumentTypes(),
                                                      returnedTypes));
        rewriter.replaceOpWithNewOp<func::ReturnOp>(ret, returnedValues);
      }
    }

    for (maps::RecvOp recv : recvsToReplace) {
      rewriter.setInsertionPoint(recv);
      auto listIt = channelValueLists.find(recv.getChannelAttr());
      if (listIt != channelValueLists.end() && listIt->second.size() > 1) {
        auto resultType = cast<RankedTensorType>(recv.getResult().getType());
        auto replacement = rewriter.create<tensor::EmptyOp>(
            recv.getLoc(), resultType.getShape(), resultType.getElementType());
        rewriter.replaceOp(recv, replacement.getResult());
      } else {
        rewriter.replaceOp(recv, channelValues[recv.getChannelAttr()]);
      }
    }

    SmallVector<maps::SendOp> sendsToErase;
    module.walk([&](maps::SendOp op) {
      auto channelIt = program.channels.find(op.getChannelAttr());
      if (channelValues.contains(op.getChannelAttr()) &&
          channelIt != program.channels.end() && channelIt->second.dstHartId == -1)
        sendsToErase.push_back(op);
    });
    for (maps::SendOp send : sendsToErase) {
      rewriter.eraseOp(send);
    }

    bool changed = true;
    while (changed) {
      changed = false;
      module.walk([&](tensor::ExtractSliceOp slice) {
        if (foldIdentitySlice(slice))
          changed = true;
      });
      module.walk([&](tensor::InsertSliceOp insertSlice) {
        if (foldIdentityMaterialization(insertSlice))
          changed = true;
      });

      SmallVector<Operation *> deadOps;
      module.walk([&](Operation *op) {
        if (!op->use_empty())
          return;
        if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp, tensor::EmptyOp>(
                op))
          deadOps.push_back(op);
      });
      for (Operation *op : deadOps) {
        rewriter.eraseOp(op);
        changed = true;
      }
    }

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(eraseUnusedParameterArgs(func))) {
        signalPassFailure();
        return;
      }
      if (failed(reorderParameterArgsByFirstUse(func))) {
        signalPassFailure();
        return;
      }
    }

    bool hasMapsOp = false;
    module.walk([&](Operation *op) {
      if (isa<ModuleOp>(op))
        return;
      if (isa<maps::MapsDialect>(op->getDialect())) {
        op->emitError("failed to convert maps operation");
        hasMapsOp = true;
      }
    });
    if (hasMapsOp)
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::maps
