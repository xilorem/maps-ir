#ifndef MAPS_PASS
#define MAPS_PASS

include "mlir/Pass/PassBase.td"

def ConvertMapsToD2M : Pass<"convert-maps-to-d2m", "mlir::ModuleOp"> {
  let summary = "Convert MAPS IR to TT-MLIR D2M";
  let dependentDialects = [
    "::mlir::maps::MapsDialect",
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::tt::ttcore::TTCoreDialect",
    "::mlir::tt::d2m::D2MDialect"
  ];
}

#endif // MAPS_PASS
