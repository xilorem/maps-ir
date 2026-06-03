#ifndef MAPS_CONVERSION_MAPSTOD2M_FINISHEMITTER_H
#define MAPS_CONVERSION_MAPSTOD2M_FINISHEMITTER_H

#include "maps/Conversion/MapsToD2M/ComputeEmitter.h"

namespace mlir::maps {

LogicalResult lowerFinishTransfers(ModuleOp module, MapsProgramInfo &program,
                                   const ChannelLoweringState &channels);

} // namespace mlir::maps

#endif
