#ifndef MAPS_CONVERSION_MAPSTOD2M_MAPSPATTERNLOWERING_H
#define MAPS_CONVERSION_MAPSTOD2M_MAPSPATTERNLOWERING_H

#include "maps/Conversion/MapsToD2M/ComputeEmitter.h"

namespace mlir::maps {

LogicalResult lowerMapsProgramToD2M(ModuleOp module,
                                    MapsProgramInfo &program,
                                    ChannelLoweringState &state);

} // namespace mlir::maps

#endif
