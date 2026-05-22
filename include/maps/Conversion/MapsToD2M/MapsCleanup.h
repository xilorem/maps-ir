#ifndef MAPS_CONVERSION_MAPSTOD2M_MAPSCLEANUP_H
#define MAPS_CONVERSION_MAPSTOD2M_MAPSCLEANUP_H

#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::maps {

void cleanupGeneratedIR(ModuleOp module);
LogicalResult verifyNoMapsOps(ModuleOp module);

} // namespace mlir::maps

#endif
