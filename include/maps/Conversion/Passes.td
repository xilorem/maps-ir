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

def PrintMapsProgramAnalysis
    : Pass<"print-maps-program-analysis", "mlir::ModuleOp"> {
  let summary = "Print structured MAPS program analysis";
  let dependentDialects = [
    "::mlir::maps::MapsDialect"
  ];
}

def PrintMapsPlacementAnalysis
    : Pass<"print-maps-placement-analysis", "mlir::ModuleOp"> {
  let summary = "Print structured MAPS placement analysis";
  let dependentDialects = [
    "::mlir::maps::MapsDialect",
    "::mlir::tt::ttcore::TTCoreDialect"
  ];
}

def LowerStandaloneLinalgElementwise
    : Pass<"lower-standalone-linalg-elementwise", "mlir::func::FuncOp"> {
  let summary = "Lower standalone memref elementwise linalg.generic ops";
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tt::d2m::D2MDialect"
  ];
}

#endif // MAPS_PASS
