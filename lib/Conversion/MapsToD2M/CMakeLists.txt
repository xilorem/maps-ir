add_mlir_library(MLIRMapsToD2M
  ComputeEmitter.cpp
  ComputeOpRegistry.cpp
  D2MGenericBuilder.cpp
  D2MTypeUtils.cpp
  FinishEmitter.cpp
  InitEmitter.cpp
  LowerStandaloneLinalgElementwise.cpp
  MapsABIRewriter.cpp
  MapsCleanup.cpp
  MapsPatternLowering.cpp
  MapsProgramAnalysis.cpp
  MapsToD2M.cpp
  PlacementAnalysis.cpp
  PrintMapsProgramAnalysis.cpp
  PrintMapsPlacementAnalysis.cpp
  SpatialEmitter.cpp
  TileBodyEmitter.cpp

  DEPENDS
  MLIRMapsConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRMapsDialect
  TTMLIR::MLIRTTTransforms
  TTMLIR::MLIRTTCoreDialect
  TTMLIR::MLIRD2MDialect
  TTMLIR::MLIRD2MUtils
  TTMLIR::MLIRTTIRDialect
  TTMLIR::MLIRTTNNDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
)

target_include_directories(MLIRMapsToD2M
  PUBLIC
  ${TTMLIR_INCLUDE_DIRS}
)
