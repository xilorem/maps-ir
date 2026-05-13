

#ifndef MAPS_BASE
#define MAPS_BASE

include "mlir/IR/OpBase.td"

def Maps_Dialect : Dialect {
  let name = "maps";
  let cppNamespace = "::mlir::maps";
  let description = [{
    "The MAPS dialect is a collection of operations and types that represent "
    "the core abstractions of the MAPS IR. It includes operations for "
    "representing pipelines, stages, tiles, and communication between tiles. "
    "The MAPS dialect is designed to be a low-level representation that can be "
    "used as a target for high-level transformations and optimizations."
  }];
}

#endif // MAPS_BASE