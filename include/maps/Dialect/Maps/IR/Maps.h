

#ifndef MLIR_DIALECT_MAPS_H_
#define MLIR_DIALECT_MAPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"




#include "maps/Dialect/Maps/IR/MapsOpsDialect.h.inc"



#define GET_OP_CLASSES
#include "maps/Dialect/Maps/IR/MapsOps.h.inc"


#endif // MLIR_DIALECT_MAPS_H_