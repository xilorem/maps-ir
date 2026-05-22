#include "maps/Conversion/MapsToD2M/MapsProgramAnalysis.h"

namespace mlir::maps {

bool isInitStorageTile(maps::TileOp tile) {
  SmallVector<maps::RecvOp> recvs;
  SmallVector<maps::StoreOp> stores;

  for (Operation &nested : tile.getBody().front()) {
    if (auto recv = dyn_cast<maps::RecvOp>(nested)) {
      recvs.push_back(recv);
      continue;
    }
    if (auto store = dyn_cast<maps::StoreOp>(nested)) {
      stores.push_back(store);
      continue;
    }
    return false;
  }

  if (recvs.size() != stores.size())
    return false;

  for (auto [recv, store] : llvm::zip_equal(recvs, stores)) {
    if (store.getValue() != recv.getResult())
      return false;
  }

  return true;
}

FailureOr<MapsProgramInfo> analyzeMapsProgram(ModuleOp module) {
  MapsProgramInfo program;
  bool failed = false;

  module.walk([&](maps::InitOp op) {
    if (program.init) {
      op.emitError("expected at most one maps.init");
      failed = true;
      return;
    }
    program.init = op;
  });

  module.walk([&](maps::RunOp op) {
    if (program.run) {
      op.emitError("expected at most one maps.run");
      failed = true;
      return;
    }
    program.run = op;
  });

  module.walk([&](maps::StageOp stage) {
    StageInfo stageInfo;
    stageInfo.op = stage;
    stageInfo.stageId = static_cast<int64_t>(stage.getStageId());
    stage.walk([&](maps::TileOp tile) {
      TileInfo tileInfo;
      tileInfo.op = tile;
      tileInfo.tileId = static_cast<int64_t>(tile.getTileId());
      tileInfo.coords = static_cast<SmallVector<int64_t>>(tile.getCoords());
      stageInfo.tiles.push_back(tileInfo);
    });
    program.stages.push_back(stageInfo);
  });

  module.walk([&](maps::SendOp op) {
    Attribute channelName = op.getChannelAttr();
    if (program.channels.contains(channelName)) {
      op.emitError("duplicate send for channel ") << channelName;
      failed = true;
      return;
    }

    ChannelInfo channel;
    channel.name = op.getChannelAttr();
    channel.srcHartId = static_cast<int64_t>(op.getSrcHartid());
    channel.dstHartId = static_cast<int64_t>(op.getDstHartid());
    channel.type = cast<RankedTensorType>(op.getValue().getType());
    channel.send = op;
    program.channels.try_emplace(channelName, std::move(channel));
  });

  module.walk([&](maps::RecvOp op) {
    auto it = program.channels.find(op.getChannelAttr());
    if (it == program.channels.end()) {
      op.emitError("recv without matching send for channel ")
          << op.getChannelAttr();
      failed = true;
      return;
    }
    it->second.recvs.push_back(op);
  });

  module.walk([&](maps::TileOp tile) {
    if (!isInitStorageTile(tile))
      return;

    for (Operation &nested : tile.getBody().front()) {
      auto store = dyn_cast<maps::StoreOp>(nested);
      if (!store)
        continue;

      auto recv = store.getValue().getDefiningOp<maps::RecvOp>();
      auto it = program.channels.find(recv.getChannelAttr());
      program.slotValues[store.getSlotAttr()] = it->second.send.getValue();
    }
  });

  if (failed)
    return failure();
  return program;
}

LogicalResult verifyMapsProgram(MapsProgramInfo &program) {
  for (StageInfo &stage : program.stages) {
    for (TileInfo &tile : stage.tiles) {
      if (isInitStorageTile(tile.op))
        continue;

      bool hasSend = llvm::any_of(tile.op.getBody().front(), [](Operation &nested) {
        return isa<maps::SendOp>(nested);
      });
      if (!hasSend)
        return tile.op.emitOpError(
            "expected compute tile to produce at least one maps.send");
    }
  }

  return success();
}

} // namespace mlir::maps
