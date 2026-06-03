#include "maps/Conversion/Passes.h"
#include "maps/Conversion/MapsToD2M/MapsProgramAnalysis.h"
#include "maps/Conversion/MapsToD2M/PlacementAnalysis.h"

namespace mlir::maps {
#define GEN_PASS_DEF_PRINTMAPSPLACEMENTANALYSIS
#include "maps/Conversion/Passes.h.inc"

namespace {

struct PrintMapsPlacementAnalysisPass
    : public impl::PrintMapsPlacementAnalysisBase<
          PrintMapsPlacementAnalysisPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<MapsProgramInfo> program = analyzeMapsProgram(module);
    if (failed(program))
      return signalPassFailure();
    if (failed(verifyMapsProgram(*program)))
      return signalPassFailure();

    FailureOr<MapsPlacementInfo> placement = analyzeMapsPlacement(*program);
    if (failed(placement))
      return signalPassFailure();
    printMapsPlacementAnalysis(llvm::outs(), *placement);
  }
};

} // namespace
} // namespace mlir::maps
