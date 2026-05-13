#include "maps/Dialect/Maps/IR/Maps.h"

using namespace mlir;
using namespace mlir::maps;

#include "maps/Dialect/Maps/IR/MapsOpsDialect.cpp.inc"

void mlir::maps::MapsDialect::initialize() {
    addOperations<
    
#define GET_OP_LIST
#include "maps/Dialect/Maps/IR/MapsOps.cpp.inc"
    >();
}