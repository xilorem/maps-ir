add_mlir_library(MLIRMapsToD2M
  D2MGenericBuilder.cpp
  D2MTypeUtils.cpp
  MapsABIRewriter.cpp
  MapsCleanup.cpp
  MapsPatternLowering.cpp
  MapsProgramAnalysis.cpp
  MapsToD2M.cpp

  DEPENDS
  MLIRMapsConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRMapsDialect
  TTMLIR::MLIRTTCoreDialect
  TTMLIR::MLIRD2MDialect
  TTMLIR::MLIRD2MUtils
  TTMLIR::MLIRTTIRDialect
  TTMLIR::MLIRTTNNDialect
  MLIRIR
  MLIRPass
)

target_include_directories(MLIRMapsToD2M
  PUBLIC
  ${TTMLIR_INCLUDE_DIRS}
)
