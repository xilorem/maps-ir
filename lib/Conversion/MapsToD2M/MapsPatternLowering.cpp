#include "maps/Conversion/MapsToD2M/MapsPatternLowering.h"

#include "maps/Conversion/MapsToD2M/ComputeEmitter.h"
#include "maps/Conversion/MapsToD2M/InitEmitter.h"
#include "maps/Conversion/MapsToD2M/PlacementAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::maps {
namespace {

static void inlineRegionOp(PatternRewriter &rewriter, Operation *op) {
  // take a block outside the region it belongs, useful for wrapper ops like module
  rewriter.inlineBlockBefore(&op->getRegion(0).front(), op);
  rewriter.eraseOp(op);
}

} // namespace


// main entry point
LogicalResult lowerMapsProgramToD2M(ModuleOp module,
                                    MapsProgramInfo &program,
                                    ChannelLoweringState &state) {
  PatternRewriter rewriter(module.getContext()); //
  DenseMap<Attribute, Value> logicalChannelValues;
  DenseMap<Operation *, int64_t> tileStageIds;
  DenseMap<Operation *, mlir::tt::ttcore::CoreRangeAttr> tileCoreRanges;
  for (StageInfo &stage : program.stages) {
    for (TileInfo &tile : stage.tiles)
      tileStageIds[tile.op.getOperation()] = stage.stageId;
  }

  auto placement = analyzeMapsPlacement(program);
  if (failed(placement))
    return failure();
  for (const TilePlacementInfo &tilePlacement : placement->tilePlacements) {
    for (const TileProgramInfo *tileProgram : tilePlacement.tilePrograms) {
      if (!tileProgram->isInitStorage)
        tileCoreRanges[const_cast<maps::TileOp &>(tileProgram->op).getOperation()] =
            tilePlacement.coreRange;
    }
  }

  // Inline the MAPS structural wrappers.
  // setInsertionPoint sets the "cursor" of the patternRewriter to the specific op loc
  rewriter.setInsertionPoint(program.init); 
  inlineRegionOp(rewriter, program.init);
  rewriter.setInsertionPoint(program.run);
  inlineRegionOp(rewriter, program.run);
  for (StageInfo &stage : program.stages) {
    rewriter.setInsertionPoint(stage.op);
    inlineRegionOp(rewriter, stage.op);
  }

  // Drop storage-only tiles now that slot values have been analyzed.
  SmallVector<maps::TileOp> pendingTiles;

  module.walk([&](maps::TileOp tile) {
    if (!isInitStorageTile(tile))
      pendingTiles.push_back(tile);
  });

  for (const TileProgramInfo &tileProgram : program.tilePrograms) {
    if (!tileProgram.isInitStorage)
      continue;
    if (failed(lowerInitTileProgram(tileProgram, rewriter)))
      return failure();
  }

  // Lower compute tiles in dependency order without relying on greedy cleanup.
  while (!pendingTiles.empty()) {
    SmallVector<maps::TileOp> deferredTiles;
    bool madeProgress = false;

    for (maps::TileOp tile : pendingTiles) {
      auto lowered = lowerComputeTileProgram(
          tile, program.channels, state.values, logicalChannelValues,
          state.valueLists, program.slotValues, tileCoreRanges,
          state.stageChannels,
          tileStageIds, rewriter);
      if (failed(lowered))
        return failure();
      if (*lowered) {
        madeProgress = true;
        continue;
      }
      deferredTiles.push_back(tile);
    }

    if (!madeProgress)
      return deferredTiles.front().emitOpError(
          "failed to resolve compute tile dependencies");
    pendingTiles = std::move(deferredTiles);
  }

  // Host-to-tile sends are represented directly by their source values.
  eraseHostToTileSends(module, rewriter);
  return success();
}

} // namespace mlir::maps
