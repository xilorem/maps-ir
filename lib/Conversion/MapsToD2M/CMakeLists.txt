add_mlir_library(MLIRMapsToD2M
  MapsToD2M.cpp

  DEPENDS
  MLIRMapsConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRMapsDialect
  MLIRIR
  MLIRPass
)

target_include_directories(MLIRMapsToD2M
  PUBLIC
  ${TTMLIR_INCLUDE_DIRS}
)