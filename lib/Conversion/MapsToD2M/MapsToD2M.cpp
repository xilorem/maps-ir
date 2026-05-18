#include "maps/Conversion/Passes.h"

namespace mlir::maps {
#define GEN_PASS_DEF_CONVERTMAPSTOD2M
#include "maps/Conversion/Passes.h.inc"

namespace {
struct ConvertMapsToD2MPass
    : impl::ConvertMapsToD2MBase<ConvertMapsToD2MPass> {
  void runOnOperation() override {
    // Intentionally no-op for registration smoke test.
  }
};
} // namespace
} // namespace mlir::maps