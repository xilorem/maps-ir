#include "maps/Conversion/MapsToD2M/D2MTypeUtils.h"

#include <array>

namespace mlir::maps {

static int64_t ceilDiv(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

mlir::tt::ttcore::GridAttr getGridForTensor(MLIRContext *ctx,
                                            RankedTensorType type) {
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() < 2)
    return mlir::tt::ttcore::GridAttr::get(ctx, {1, 1});

  return mlir::tt::ttcore::GridAttr::get(
      ctx, {ceilDiv(shape[shape.size() - 2], 32),
            ceilDiv(shape[shape.size() - 1], 32)});
}

mlir::tt::ttcore::MetalLayoutAttr getL1ShardedLayout(MLIRContext *ctx,
                                                     RankedTensorType type) {
  return mlir::tt::ttcore::MetalLayoutAttr::get(
      ctx, type.getShape(), mlir::tt::ttcore::OOBVal::Undef,
      mlir::tt::ttcore::MemorySpace::DeviceL1,
      mlir::tt::ttcore::TensorMemoryLayout::Sharded);
}

RankedTensorType getTiledDeviceTensorType(MLIRContext *ctx,
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

RankedTensorType getShardTensorType(RankedTensorType type) {
  auto layout = cast<mlir::tt::ttcore::MetalLayoutAttr>(type.getEncoding());
  return RankedTensorType::get(layout.getShardShape(type), type.getElementType());
}

mlir::tt::ttcore::GridAttr getFrontendGrid(MLIRContext *ctx) {
  return mlir::tt::ttcore::GridAttr::get(ctx, {1, 1});
}

SmallVector<AffineMap> getIdentityAffineMapsArray(OpBuilder &builder,
                                                  size_t arity, size_t rank) {
  return SmallVector<AffineMap>(arity, builder.getMultiDimIdentityMap(rank));
}

SmallVector<Attribute> getElementwiseIteratorTypesArray(OpBuilder &builder,
                                                        size_t rank) {
  auto parallel = mlir::tt::ttcore::IteratorTypeAttr::get(
      builder.getContext(), mlir::tt::ttcore::IteratorType::Parallel);
  return SmallVector<Attribute>(rank, parallel);
}

SmallVector<mlir::utils::IteratorType> getLinalgParallelIterators(size_t rank) {
  return SmallVector<mlir::utils::IteratorType>(
      rank, mlir::utils::IteratorType::parallel);
}

SmallVector<AffineMap> getMatmulAffineMapsArray(OpBuilder &builder,
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

SmallVector<Attribute> getMatmulIteratorTypesArray(OpBuilder &builder,
                                                   size_t rank) {
  assert(rank >= 2 && "matmul rank must be >= 2");
  auto parallel = mlir::tt::ttcore::IteratorTypeAttr::get(
      builder.getContext(), mlir::tt::ttcore::IteratorType::Parallel);
  auto reduction = mlir::tt::ttcore::IteratorTypeAttr::get(
      builder.getContext(), mlir::tt::ttcore::IteratorType::Reduction);

  SmallVector<Attribute> result(rank, parallel);
  result.push_back(reduction);
  return result;
}

SmallVector<mlir::utils::IteratorType> getLinalgMatmulIterators(size_t rank) {
  SmallVector<mlir::utils::IteratorType> result(
      rank, mlir::utils::IteratorType::parallel);
  result.push_back(mlir::utils::IteratorType::reduction);
  return result;
}

} // namespace mlir::maps
