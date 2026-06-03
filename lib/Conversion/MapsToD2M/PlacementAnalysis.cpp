#include "maps/Conversion/MapsToD2M/PlacementAnalysis.h"

namespace mlir::maps {
namespace {

static FailureOr<mlir::tt::ttcore::CoreRangeAttr>
getSingletonCoreRange(MLIRContext *ctx, ArrayRef<int64_t> coords) {
  if (coords.size() != 2)
    return failure();
  if (coords[0] < 0 || coords[1] < 0)
    return failure();

  auto coordAttr =
      mlir::tt::ttcore::CoreCoordAttr::get(ctx, coords[1], coords[0]);
  return mlir::tt::ttcore::CoreRangeAttr::get(ctx, coordAttr, coordAttr);
}

static bool sameCoords(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  return llvm::equal(lhs, rhs);
}

} // namespace

FailureOr<MapsPlacementInfo> analyzeMapsPlacement(const MapsProgramInfo &program) {
  MapsPlacementInfo placement;
  DenseMap<int64_t, size_t> tilePlacementById;

  for (const TileProgramInfo &tileProgram : program.tilePrograms) {
    auto &tileOpRef = const_cast<maps::TileOp &>(tileProgram.op);
    auto *tileOp = tileOpRef.getOperation();
    auto coreRange = getSingletonCoreRange(tileOp->getContext(),
                                           tileProgram.coords);
    if (failed(coreRange))
      return const_cast<Operation *>(tileOp)->emitOpError(
          "expected tile coords to contain two non-negative values");

    auto [it, inserted] = tilePlacementById.try_emplace(
        tileProgram.tileId, placement.tilePlacements.size());
    if (inserted) {
      TilePlacementInfo tilePlacement;
      tilePlacement.tileId = tileProgram.tileId;
      tilePlacement.coords = tileProgram.coords;
      tilePlacement.coreRange = *coreRange;
      tilePlacement.tilePrograms.push_back(&tileProgram);
      placement.tilePlacements.push_back(std::move(tilePlacement));
      continue;
    }

    TilePlacementInfo &tilePlacement = placement.tilePlacements[it->second];
    if (!sameCoords(tilePlacement.coords, tileProgram.coords))
      return const_cast<Operation *>(tileOp)->emitOpError(
          "expected all tile programs with the same tile_id to use identical coords");
    tilePlacement.tilePrograms.push_back(&tileProgram);
  }

  placement.placementGroups.reserve(placement.tilePlacements.size());
  for (TilePlacementInfo &tilePlacement : placement.tilePlacements) {
    PlacementGroupInfo group;
    group.coreRange = tilePlacement.coreRange;
    llvm::append_range(group.tilePrograms, tilePlacement.tilePrograms);
    placement.placementGroups.push_back(std::move(group));
  }

  llvm::sort(placement.tilePlacements, [](const TilePlacementInfo &lhs,
                                          const TilePlacementInfo &rhs) {
    return lhs.tileId < rhs.tileId;
  });
  llvm::sort(placement.placementGroups, [](const PlacementGroupInfo &lhs,
                                           const PlacementGroupInfo &rhs) {
    auto lhsStart = lhs.coreRange.getStartCoord();
    auto rhsStart = rhs.coreRange.getStartCoord();
    return std::pair<int64_t, int64_t>{lhsStart.getY(), lhsStart.getX()} <
           std::pair<int64_t, int64_t>{rhsStart.getY(), rhsStart.getX()};
  });

  return placement;
}

void printMapsPlacementAnalysis(raw_ostream &os,
                                const MapsPlacementInfo &placement) {
  os << "maps.placement\n";
  os << "  tile_placements: " << placement.tilePlacements.size() << "\n";
  for (const TilePlacementInfo &tilePlacement : placement.tilePlacements) {
    auto start = tilePlacement.coreRange.getStartCoord();
    auto end = tilePlacement.coreRange.getEndCoord();
    os << "    - tile: " << tilePlacement.tileId << ", coords: [";
    llvm::interleaveComma(tilePlacement.coords, os);
    os << "], core_range: #ttcore.core_range<(" << start.getY() << ", "
       << start.getX() << "), (" << end.getY() << ", " << end.getX()
       << ")>, programs: " << tilePlacement.tilePrograms.size() << "\n";
  }

  os << "  placement_groups: " << placement.placementGroups.size() << "\n";
  for (const PlacementGroupInfo &group : placement.placementGroups) {
    auto start = group.coreRange.getStartCoord();
    auto end = group.coreRange.getEndCoord();
    os << "    - core_range: #ttcore.core_range<(" << start.getY() << ", "
       << start.getX() << "), (" << end.getY() << ", " << end.getX()
       << ")>, tile_programs: [";
    llvm::interleaveComma(group.tilePrograms, os, [&](const TileProgramInfo *program) {
      os << "{tile: " << program->tileId << ", stage: " << program->stageId
         << "}";
    });
    os << "]\n";
  }
}

} // namespace mlir::maps
