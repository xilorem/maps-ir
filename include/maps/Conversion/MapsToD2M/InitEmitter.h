#ifndef MAPS_CONVERSION_MAPSTOD2M_INITEMITTER_H
#define MAPS_CONVERSION_MAPSTOD2M_INITEMITTER_H

#include "maps/Conversion/MapsToD2M/MapsProgramAnalysis.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::maps {

LogicalResult lowerInitTileProgram(const TileProgramInfo &tileProgram,
                                   PatternRewriter &rewriter);
void eraseHostToTileSends(ModuleOp module, PatternRewriter &rewriter);

} // namespace mlir::maps

#endif
