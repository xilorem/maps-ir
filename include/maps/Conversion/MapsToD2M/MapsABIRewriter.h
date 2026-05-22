#ifndef MAPS_CONVERSION_MAPSTOD2M_MAPSABIREWRITER_H
#define MAPS_CONVERSION_MAPSTOD2M_MAPSABIREWRITER_H

#include "maps/Conversion/MapsToD2M/MapsPatternLowering.h"

namespace mlir::maps {

LogicalResult rewriteForwardABI(ModuleOp module, MapsProgramInfo &program);
LogicalResult normalizeParameterSlices(ModuleOp module);
LogicalResult rewriteOutputABI(ModuleOp module, MapsProgramInfo &program,
                               const ChannelLoweringState &channels);

} // namespace mlir::maps

#endif
