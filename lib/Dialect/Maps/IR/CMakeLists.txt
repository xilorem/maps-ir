add_mlir_dialect_library(MLIRMapsDialect
  MapsDialect.cpp
  MapsOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/maps/Dialect/Maps

  DEPENDS
  MLIRMapsOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRUBDialect
)