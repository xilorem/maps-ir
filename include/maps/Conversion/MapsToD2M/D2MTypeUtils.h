#ifndef MAPS_CONVERSION_MAPSTOD2M_D2MTYPEUTILS_H
#define MAPS_CONVERSION_MAPSTOD2M_D2MTYPEUTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::maps {

mlir::tt::ttcore::MetalLayoutAttr getL1ShardedLayout(MLIRContext *ctx,
                                                     RankedTensorType type);
RankedTensorType getTiledDeviceTensorType(MLIRContext *ctx,
                                          RankedTensorType type);
RankedTensorType getShardTensorType(RankedTensorType type);
mlir::tt::ttcore::GridAttr getGridForTensor(MLIRContext *ctx,
                                            RankedTensorType type);
mlir::tt::ttcore::GridAttr getFrontendGrid(MLIRContext *ctx);

SmallVector<AffineMap> getIdentityAffineMapsArray(OpBuilder &builder,
                                                  size_t arity, size_t rank);
SmallVector<Attribute> getElementwiseIteratorTypesArray(OpBuilder &builder,
                                                        size_t rank);
SmallVector<mlir::utils::IteratorType> getLinalgParallelIterators(size_t rank);
SmallVector<AffineMap> getMatmulAffineMapsArray(OpBuilder &builder,
                                                size_t rank);
SmallVector<Attribute> getMatmulIteratorTypesArray(OpBuilder &builder,
                                                   size_t rank);
SmallVector<mlir::utils::IteratorType> getLinalgMatmulIterators(size_t rank);

} // namespace mlir::maps

#endif
