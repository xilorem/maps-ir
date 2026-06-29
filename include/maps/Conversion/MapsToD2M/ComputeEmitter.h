#ifndef MAPS_CONVERSION_MAPSTOD2M_COMPUTEEMITTER_H
#define MAPS_CONVERSION_MAPSTOD2M_COMPUTEEMITTER_H

#include "maps/Conversion/MapsToD2M/MapsProgramAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::maps {

struct ChannelLoweringState {
  DenseMap<Attribute, Value> values;
  DenseMap<Attribute, SmallVector<Value>> valueLists;
  DenseMap<Attribute, Value> semaphores;
  DenseMap<int64_t, SmallVector<Attribute>> stageChannels;
};

FailureOr<bool> lowerComputeTileProgram(
    maps::TileOp tile, DenseMap<Attribute, ChannelInfo> &channels,
    DenseMap<Attribute, Value> &channelValues,
    DenseMap<Attribute, Value> &logicalChannelValues,
    DenseMap<Attribute, SmallVector<Value>> &channelValueLists,
    DenseMap<Attribute, Value> &channelSemaphores,
    DenseMap<Attribute, Value> &slotValues,
    DenseMap<Operation *, mlir::tt::ttcore::CoreRangeAttr> &tileCoreRanges,
    DenseMap<int64_t, mlir::tt::ttcore::CoreRangeAttr> &tileIdCoreRanges,
    DenseMap<int64_t, SmallVector<Attribute>> &stageChannels,
    DenseMap<Operation *, int64_t> &tileStageIds, PatternRewriter &rewriter);

} // namespace mlir::maps

#endif
