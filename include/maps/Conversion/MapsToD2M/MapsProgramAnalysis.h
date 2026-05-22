#ifndef MAPS_CONVERSION_MAPSTOD2M_MAPSPROGRAMANALYSIS_H
#define MAPS_CONVERSION_MAPSTOD2M_MAPSPROGRAMANALYSIS_H

#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::maps {

struct ChannelInfo {
  FlatSymbolRefAttr name;
  int64_t srcHartId = -1;
  int64_t dstHartId = -1;
  RankedTensorType type;
  maps::SendOp send;
  SmallVector<maps::RecvOp> recvs;
};

struct TileInfo {
  int64_t tileId = -1;
  SmallVector<int64_t> coords;
  maps::TileOp op;
};

struct StageInfo {
  int64_t stageId = -1;
  maps::StageOp op;
  SmallVector<TileInfo> tiles;
};

struct MapsProgramInfo {
  maps::InitOp init;
  maps::RunOp run;
  SmallVector<StageInfo> stages;
  DenseMap<Attribute, ChannelInfo> channels;
  DenseMap<Attribute, Value> slotValues;
};

FailureOr<MapsProgramInfo> analyzeMapsProgram(ModuleOp module);
LogicalResult verifyMapsProgram(MapsProgramInfo &program);
bool isInitStorageTile(maps::TileOp tile);

} // namespace mlir::maps

#endif
