#ifndef MAPS_CONVERSION_MAPSTOD2M_MAPSPATTERNLOWERING_H
#define MAPS_CONVERSION_MAPSTOD2M_MAPSPATTERNLOWERING_H

#include "maps/Conversion/MapsToD2M/MapsProgramAnalysis.h"

namespace mlir::maps {

struct ChannelLoweringState {
  DenseMap<Attribute, Value> values;
  DenseMap<Attribute, SmallVector<Value>> valueLists;
  DenseMap<int64_t, SmallVector<Attribute>> stageChannels;
};

LogicalResult lowerMapsProgramToD2M(ModuleOp module,
                                    MapsProgramInfo &program,
                                    ChannelLoweringState &state);

} // namespace mlir::maps

#endif
