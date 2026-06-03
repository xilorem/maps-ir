// Main entry point for the MAPS to D2M conversion pipeline.

#include "maps/Conversion/Passes.h"
#include "maps/Conversion/MapsToD2M/MapsABIRewriter.h"
#include "maps/Conversion/MapsToD2M/MapsCleanup.h"
#include "maps/Conversion/MapsToD2M/MapsPatternLowering.h"
#include "maps/Conversion/MapsToD2M/MapsProgramAnalysis.h"

namespace mlir::maps {
#define GEN_PASS_DEF_CONVERTMAPSTOD2M
#include "maps/Conversion/Passes.h.inc"

namespace {

struct ConvertMapsToD2MPass
    : public impl::ConvertMapsToD2MBase<ConvertMapsToD2MPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Analyze the MAPS program.
    FailureOr<MapsProgramInfo> program = analyzeMapsProgram(module);
    if (failed(program))
      return signalPassFailure();

    // Verify the supported MAPS structure.
    if (failed(verifyMapsProgram(*program)))
      return signalPassFailure();

    // Rewrite the forward ABI before lowering.
    if (failed(rewriteForwardABI(module, *program)))
      return signalPassFailure();

    // Lower MAPS ops to D2M.
    ChannelLoweringState channels;
    if (failed(lowerMapsProgramToD2M(module, *program, channels)))
      return signalPassFailure();

    // Lift any late static parameter slices introduced during lowering.
    if (failed(normalizeParameterSlices(module)))
      return signalPassFailure();

    // Rewrite lowered host-visible outputs.
    if (failed(rewriteOutputABI(module, *program, channels)))
      return signalPassFailure();

    // Drop cleanup-only tensor scaffolding.
    cleanupGeneratedIR(module);

    // Cleanup can remove the last users of original parameter tensors, so run
    // the slice normalization again to keep the lowered ABI minimal.
    if (failed(normalizeParameterSlices(module)))
      return signalPassFailure();

    // Verify no MAPS ops remain.
    if (failed(verifyNoMapsOps(module)))
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::maps
