#ifndef MAPS_CONVERSION_MAPSTOD2M_MAPSPROGRAMANALYSIS_H
#define MAPS_CONVERSION_MAPSTOD2M_MAPSPROGRAMANALYSIS_H

#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/IR/BuiltinOps.h"

#include <vector>

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

struct TileProgramInfo {
  int64_t stageId = -1;
  int64_t tileId = -1;
  SmallVector<int64_t> coords;
  maps::TileOp op;
  bool isInitStorage = false;
  SmallVector<Operation *> ops;
  SmallVector<maps::RecvOp> recvs;
  SmallVector<maps::LoadOp> loads;
  SmallVector<maps::StoreOp> stores;
  SmallVector<maps::SendOp> sends;
};

struct StageInfo {
  int64_t stageId = -1;
  maps::StageOp op;
  std::vector<TileInfo> tiles;
  std::vector<TileProgramInfo> tilePrograms;
};

struct InitTransferInfo {
  FlatSymbolRefAttr channel;
  int64_t dstHartId = -1;
  RankedTensorType type;
  Value source;
  maps::SendOp send;
};

struct FinishTransferInfo {
  FlatSymbolRefAttr channel;
  int64_t srcHartId = -1;
  RankedTensorType type;
  maps::SendOp send;
  SmallVector<maps::RecvOp> recvs;
};

struct SlotBindingInfo {
  FlatSymbolRefAttr slot;
  int64_t tileId = -1;
  FlatSymbolRefAttr sourceChannel;
  Value source;
  maps::StoreOp store;
  maps::RecvOp recv;
};

struct MapsProgramInfo {
  maps::InitOp init;
  maps::RunOp run;
  std::vector<StageInfo> stages;
  std::vector<TileProgramInfo> tilePrograms;
  std::vector<InitTransferInfo> initTransfers;
  std::vector<FinishTransferInfo> finishTransfers;
  std::vector<SlotBindingInfo> slotBindings;
  DenseMap<Attribute, ChannelInfo> channels;
  DenseMap<Attribute, Value> slotValues;
};

FailureOr<MapsProgramInfo> analyzeMapsProgram(ModuleOp module);
LogicalResult verifyMapsProgram(MapsProgramInfo &program);
bool isInitStorageTile(maps::TileOp tile);
void printMapsProgramAnalysis(raw_ostream &os, const MapsProgramInfo &program);

} // namespace mlir::maps

#endif
