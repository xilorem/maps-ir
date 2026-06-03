#include "maps/Conversion/Passes.h"
#include "maps/Conversion/MapsToD2M/MapsProgramAnalysis.h"

namespace mlir::maps {
#define GEN_PASS_DEF_PRINTMAPSPROGRAMANALYSIS
#include "maps/Conversion/Passes.h.inc"

namespace {

struct PrintMapsProgramAnalysisPass
    : public impl::PrintMapsProgramAnalysisBase<PrintMapsProgramAnalysisPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<MapsProgramInfo> program = analyzeMapsProgram(module);
    if (failed(program))
      return signalPassFailure();
    if (failed(verifyMapsProgram(*program)))
      return signalPassFailure();
    printMapsProgramAnalysis(llvm::outs(), *program);
  }
};

} // namespace
} // namespace mlir::maps
