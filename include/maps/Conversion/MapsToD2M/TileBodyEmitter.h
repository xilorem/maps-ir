#ifndef MAPS_CONVERSION_MAPSTOD2M_TILEBODYEMITTER_H
#define MAPS_CONVERSION_MAPSTOD2M_TILEBODYEMITTER_H

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::maps {

class TileBodyOpEmitter {
public:
  virtual ~TileBodyOpEmitter() = default;
  virtual bool match(Operation *op) const = 0;
  virtual FailureOr<Value> lower(PatternRewriter &rewriter, Operation *op,
                                 IRMapping &mapping) const = 0;
};

FailureOr<Value> lowerTileBodyOp(PatternRewriter &rewriter, Operation *op,
                                 IRMapping &mapping);

} // namespace mlir::maps

#endif
