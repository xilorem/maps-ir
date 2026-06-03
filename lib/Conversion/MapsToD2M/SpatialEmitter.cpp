#include "maps/Conversion/MapsToD2M/SpatialEmitter.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

namespace mlir::maps {

FailureOr<Value> wrapGenericInSpatial(mlir::tt::d2m::GenericOp generic,
                                      mlir::tt::ttcore::CoreRangeAttr coreRange,
                                      PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(generic);
  auto spatial = rewriter.create<mlir::tt::d2m::SpatialOp>(
      generic.getLoc(), generic->getResultTypes(), generic.getInputs(),
      generic.getOutputs(), rewriter.getArrayAttr(coreRange), /*regionsCount=*/1);

  Block *regionBlock = rewriter.createBlock(&spatial.getRegions().front());
  generic->moveBefore(regionBlock, regionBlock->end());
  rewriter.setInsertionPointToEnd(regionBlock);
  rewriter.create<mlir::tt::d2m::SpatialYieldOp>(generic.getLoc(), TypeRange{},
                                                 generic.getResults(),
                                                 ArrayRef<NamedAttribute>{});

  for (auto [genericResult, spatialResult] :
       llvm::zip_equal(generic.getResults(), spatial.getResults())) {
    genericResult.replaceUsesWithIf(spatialResult, [&](OpOperand &use) {
      return use.getOwner()->getParentRegion() != &spatial.getRegions().front();
    });
  }

  if (spatial.getNumResults() != 1)
    return failure();
  return spatial.getResult(0);
}

} // namespace mlir::maps
