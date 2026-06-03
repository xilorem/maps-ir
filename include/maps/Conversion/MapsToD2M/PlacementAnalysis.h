#ifndef MAPS_CONVERSION_MAPSTOD2M_PLACEMENTANALYSIS_H
#define MAPS_CONVERSION_MAPSTOD2M_PLACEMENTANALYSIS_H

#include "maps/Conversion/MapsToD2M/MapsProgramAnalysis.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include <vector>

namespace mlir::maps {

struct TilePlacementInfo {
  int64_t tileId = -1;
  SmallVector<int64_t> coords;
  mlir::tt::ttcore::CoreRangeAttr coreRange;
  SmallVector<const TileProgramInfo *> tilePrograms;
};

struct PlacementGroupInfo {
  mlir::tt::ttcore::CoreRangeAttr coreRange;
  SmallVector<const TileProgramInfo *> tilePrograms;
};

struct MapsPlacementInfo {
  std::vector<TilePlacementInfo> tilePlacements;
  std::vector<PlacementGroupInfo> placementGroups;
};

FailureOr<MapsPlacementInfo> analyzeMapsPlacement(const MapsProgramInfo &program);
void printMapsPlacementAnalysis(raw_ostream &os, const MapsPlacementInfo &placement);

} // namespace mlir::maps

#endif
