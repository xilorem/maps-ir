#ifndef MAPS_CONVERSION_PASSES_H
#define MAPS_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"

namespace mlir::maps {

#define GEN_PASS_DECL
#include "maps/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "maps/Conversion/Passes.h.inc"

} // namespace mlir::maps

#endif