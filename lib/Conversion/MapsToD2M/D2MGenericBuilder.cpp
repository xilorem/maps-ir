#include "maps/Conversion/MapsToD2M/D2MGenericBuilder.h"

#include "maps/Conversion/MapsToD2M/D2MTypeUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

namespace mlir::maps {

D2MGenericBuilder::D2MGenericBuilder(MLIRContext *ctx,
                                     PatternRewriter &rewriter)
    : ctx(ctx), rewriter(rewriter) {}

FailureOr<Value> D2MGenericBuilder::getTiledValue(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return failure();
  if (type.getEncoding() && isa<mlir::tt::ttcore::TileType>(type.getElementType()))
    return value;

  auto tiledType = getTiledDeviceTensorType(ctx, type);
  auto output = rewriter.create<mlir::tt::d2m::EmptyOp>(
      value.getLoc(), tiledType.getShape(), tiledType.getElementType(),
      tiledType.getEncoding());
  auto toLayout = rewriter.create<mlir::tt::d2m::ToLayoutOp>(
      value.getLoc(), value, output.getResult());
  return toLayout.getResult(0);
}

SmallVector<Value> D2MGenericBuilder::createGenericBlockArguments(
    mlir::tt::d2m::GenericOp generic, TypeRange inputs, TypeRange outputs) {
  SmallVector<Value> operands;

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto shardType = getShardTensorType(cast<RankedTensorType>(inputs[i]));
    Value genericOperand = generic->getOperand(i);
    SmallVector<Value> indices = mlir::tt::d2m::utils::buildGridIndices(
        rewriter, generic.getLoc(), generic.getIndexingMap(i));
    Value buffer = rewriter
                       .create<tensor::EmptyOp>(generic.getLoc(),
                                                shardType.getShape(),
                                                shardType.getElementType())
                       .getResult();
    Value loaded = rewriter
                       .create<mlir::tt::d2m::RemoteLoadOp>(
                           generic.getLoc(), shardType, buffer, genericOperand,
                           indices)
                       .getResult();
    operands.push_back(loaded);
  }

  for (Type output : outputs) {
    auto shardType = getShardTensorType(cast<RankedTensorType>(output));
    operands.push_back(rewriter.create<tensor::EmptyOp>(
        generic.getLoc(), shardType.getShape(), shardType.getElementType()));
  }

  return operands;
}

mlir::tt::d2m::GenericOp D2MGenericBuilder::createGeneric(
    Location loc, ArrayRef<Value> inputs, RankedTensorType outputType,
    ArrayRef<AffineMap> indexingMaps, ArrayRef<Attribute> iteratorTypes) {
  SmallVector<Attribute> blockFactorAttrs(iteratorTypes.size(),
                                          rewriter.getI64IntegerAttr(1));
  auto output = rewriter.create<mlir::tt::d2m::EmptyOp>(
      loc, outputType.getShape(), outputType.getElementType(),
      outputType.getEncoding());
  auto threads = rewriter.getArrayAttr(
      mlir::tt::d2m::ThreadAttr::get(ctx, mlir::tt::d2m::ThreadType::Unified));

  return rewriter.create<mlir::tt::d2m::GenericOp>(
      loc, TypeRange{outputType}, inputs, ValueRange{output}, ValueRange{},
      getFrontendGrid(ctx), rewriter.getArrayAttr(blockFactorAttrs),
      rewriter.getAffineMapArrayAttr(indexingMaps),
      rewriter.getArrayAttr(iteratorTypes), threads,
      /*fabricConnectionConfig=*/nullptr, /*regionsCount=*/1);
}

void D2MGenericBuilder::populateElementwiseGeneric(
    mlir::tt::d2m::GenericOp generic,
    function_ref<Value(OpBuilder &, Location, ValueRange)> bodyBuilder) {
  auto insertPoint = rewriter.saveInsertionPoint();
  rewriter.startOpModification(generic);
  {
    Region &region = generic->getRegion(0);
    Block *block = rewriter.createBlock(&region);
    SmallVector<Value> blockArgs = createGenericBlockArguments(
        generic, TypeRange(generic.getInputs().getTypes()),
        TypeRange(generic.getOutputs().getTypes()));
    rewriter.setInsertionPointToEnd(block);

    auto shardType = cast<RankedTensorType>(blockArgs.back().getType());
    auto linalg = rewriter.create<linalg::GenericOp>(
        generic.getLoc(), TypeRange{shardType}, ValueRange(blockArgs).drop_back(),
        ValueRange{blockArgs.back()},
        getIdentityAffineMapsArray(rewriter, blockArgs.size(), shardType.getRank()),
        getLinalgParallelIterators(shardType.getRank()),
        [&](OpBuilder &builder, Location loc, ValueRange bbArgs) {
          Value yielded = bodyBuilder(builder, loc, bbArgs);
          builder.create<linalg::YieldOp>(loc, yielded);
        });

    SmallVector<Value> outputIndices = mlir::tt::d2m::utils::buildGridIndices(
        rewriter, generic.getLoc(),
        generic.getIndexingMap(generic.getNumOperands() - 1));
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
}

FailureOr<mlir::tt::d2m::GenericOp> D2MGenericBuilder::createExpGeneric(
    Location loc, Value input, RankedTensorType logicalOutputType) {
  auto outputType = getTiledDeviceTensorType(ctx, logicalOutputType);
  auto generic = createGeneric(
      loc, {input}, outputType, getIdentityAffineMapsArray(rewriter, 2, 2),
      getElementwiseIteratorTypesArray(rewriter, 2));
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
}

FailureOr<mlir::tt::d2m::GenericOp> D2MGenericBuilder::createAddGeneric(
    Location loc, Value lhs, Value rhs, RankedTensorType logicalOutputType) {
  auto outputType = getTiledDeviceTensorType(ctx, logicalOutputType);
  auto generic = createGeneric(
      loc, {lhs, rhs}, outputType, getIdentityAffineMapsArray(rewriter, 3, 2),
      getElementwiseIteratorTypesArray(rewriter, 2));
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
}

FailureOr<mlir::tt::d2m::GenericOp> D2MGenericBuilder::createMatmulGeneric(
    Location loc, Value lhs, Value rhs, RankedTensorType logicalOutputType) {
  auto outputType = getTiledDeviceTensorType(ctx, logicalOutputType);
  auto generic = createGeneric(loc, {lhs, rhs}, outputType,
                               getMatmulAffineMapsArray(rewriter, 2),
                               getMatmulIteratorTypesArray(rewriter, 2));

  auto insertPoint = rewriter.saveInsertionPoint();
  rewriter.startOpModification(generic);
  {
    Region &region = generic.getRegion(0);
    Block *block = rewriter.createBlock(&region);
    SmallVector<Value> blockArgs = createGenericBlockArguments(
        generic, TypeRange(generic.getInputs().getTypes()),
        TypeRange(generic.getOutputs().getTypes()));
    rewriter.setInsertionPointToEnd(block);

    auto shardType = cast<RankedTensorType>(blockArgs.back().getType());
    auto linalg = rewriter.create<linalg::GenericOp>(
        generic.getLoc(), TypeRange{shardType}, ValueRange(blockArgs).take_front(2),
        ValueRange{blockArgs.back()}, getMatmulAffineMapsArray(rewriter, 2),
        getLinalgMatmulIterators(2),
        [&](OpBuilder &builder, Location bodyLoc, ValueRange bbArgs) {
          Value yielded = builder
                              .create<mlir::tt::d2m::TileMatmulOp>(
                                  bodyLoc, bbArgs.back().getType(), bbArgs[0],
                                  bbArgs[1], bbArgs[2])
                              .getResult();
          builder.create<linalg::YieldOp>(bodyLoc, yielded);
        });

    SmallVector<Value> outputIndices = mlir::tt::d2m::utils::buildGridIndices(
        rewriter, generic.getLoc(), generic.getIndexingMap(2));
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
}

} // namespace mlir::maps
