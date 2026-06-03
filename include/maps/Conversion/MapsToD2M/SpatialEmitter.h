#ifndef MAPS_CONVERSION_MAPSTOD2M_SPATIALEMITTER_H
#define MAPS_CONVERSION_MAPSTOD2M_SPATIALEMITTER_H

#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::maps {

FailureOr<Value> wrapGenericInSpatial(mlir::tt::d2m::GenericOp generic,
                                      mlir::tt::ttcore::CoreRangeAttr coreRange,
                                      PatternRewriter &rewriter);

} // namespace mlir::maps

#endif
