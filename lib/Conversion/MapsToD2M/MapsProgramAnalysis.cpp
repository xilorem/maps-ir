#include "maps/Conversion/MapsToD2M/MapsProgramAnalysis.h"

namespace mlir::maps {

static TileProgramInfo buildTileProgramInfo(maps::TileOp tile,
                                            int64_t stageId = -1) {
  TileProgramInfo tileProgram;
  tileProgram.stageId = stageId;
  tileProgram.tileId = static_cast<int64_t>(tile.getTileId());
  tileProgram.coords = static_cast<SmallVector<int64_t>>(tile.getCoords());
  tileProgram.op = tile;
  tileProgram.isInitStorage = isInitStorageTile(tile);

  for (Operation &nested : tile.getBody().front()) {
    tileProgram.ops.push_back(&nested);
    if (auto recv = dyn_cast<maps::RecvOp>(nested)) {
      tileProgram.recvs.push_back(recv);
      continue;
    }
    if (auto load = dyn_cast<maps::LoadOp>(nested)) {
      tileProgram.loads.push_back(load);
      continue;
    }
    if (auto store = dyn_cast<maps::StoreOp>(nested)) {
      tileProgram.stores.push_back(store);
      continue;
    }
    if (auto send = dyn_cast<maps::SendOp>(nested))
      tileProgram.sends.push_back(send);
  }

  return tileProgram;
}

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
      stageInfo.tilePrograms.push_back(buildTileProgramInfo(tile, stageInfo.stageId));
    });
    program.stages.push_back(stageInfo);
  });

  if (program.init) {
    program.init.walk([&](maps::TileOp tile) {
      program.tilePrograms.push_back(buildTileProgramInfo(tile));
    });
  }

  for (StageInfo &stage : program.stages)
    llvm::append_range(program.tilePrograms, stage.tilePrograms);

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

    int64_t srcHartId = static_cast<int64_t>(op.getSrcHartid());
    int64_t dstHartId = static_cast<int64_t>(op.getDstHartid());
    auto type = cast<RankedTensorType>(op.getValue().getType());

    if (srcHartId == -1) {
      program.initTransfers.push_back(
          {op.getChannelAttr(), dstHartId, type, op.getValue(), op});
      return;
    }

    if (dstHartId == -1)
      program.finishTransfers.push_back(
          {op.getChannelAttr(), srcHartId, type, op, {}});
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

    auto finishIt = llvm::find_if(
        program.finishTransfers, [&](FinishTransferInfo &finishTransfer) {
          return finishTransfer.channel == op.getChannelAttr();
        });
    if (finishIt != program.finishTransfers.end())
      finishIt->recvs.push_back(op);
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
      program.slotBindings.push_back(
          {store.getSlotAttr(), static_cast<int64_t>(tile.getTileId()),
           recv.getChannelAttr(), it->second.send.getValue(), store, recv});
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

void printMapsProgramAnalysis(raw_ostream &os, const MapsProgramInfo &program) {
  os << "maps.program\n";
  os << "  init_transfers: " << program.initTransfers.size() << "\n";
  for (const InitTransferInfo &transfer : program.initTransfers) {
    os << "    - channel: " << transfer.channel << ", dst_tile: "
       << transfer.dstHartId << ", type: " << transfer.type << "\n";
  }

  os << "  slot_bindings: " << program.slotBindings.size() << "\n";
  for (const SlotBindingInfo &binding : program.slotBindings) {
    os << "    - slot: " << binding.slot << ", tile: " << binding.tileId
       << ", channel: " << binding.sourceChannel << "\n";
  }

  os << "  tile_programs: " << program.tilePrograms.size() << "\n";
  for (const TileProgramInfo &tileProgram : program.tilePrograms) {
    os << "    - tile: " << tileProgram.tileId << ", stage: "
       << tileProgram.stageId << ", coords: [";
    llvm::interleaveComma(tileProgram.coords, os);
    os << "], kind: "
       << (tileProgram.isInitStorage ? "init_storage" : "compute")
       << ", recvs: " << tileProgram.recvs.size()
       << ", loads: " << tileProgram.loads.size()
       << ", stores: " << tileProgram.stores.size()
       << ", sends: " << tileProgram.sends.size() << "\n";
  }

  os << "  finish_transfers: " << program.finishTransfers.size() << "\n";
  for (const FinishTransferInfo &transfer : program.finishTransfers) {
    os << "    - channel: " << transfer.channel << ", src_tile: "
       << transfer.srcHartId << ", recvs: " << transfer.recvs.size()
       << ", type: " << transfer.type << "\n";
  }
}

} // namespace mlir::maps
