#ifndef MAPS_CONVERSION_MAPSTOD2M_D2MGENERICBUILDER_H
#define MAPS_CONVERSION_MAPSTOD2M_D2MGENERICBUILDER_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

namespace mlir::maps {

class D2MGenericBuilder {
public:
  D2MGenericBuilder(MLIRContext *ctx, PatternRewriter &rewriter);

  FailureOr<Value> getTiledValue(Value value);

  FailureOr<mlir::tt::d2m::GenericOp>
  createMatmulGeneric(Location loc, Value lhs, Value rhs,
                      RankedTensorType logicalOutputType);

  FailureOr<mlir::tt::d2m::GenericOp>
  createExpGeneric(Location loc, Value input,
                   RankedTensorType logicalOutputType);

  FailureOr<mlir::tt::d2m::GenericOp>
  createAddGeneric(Location loc, Value lhs, Value rhs,
                   RankedTensorType logicalOutputType);

private:
  SmallVector<Value> createGenericBlockArguments(mlir::tt::d2m::GenericOp generic,
                                                 TypeRange inputs,
                                                 TypeRange outputs);
  mlir::tt::d2m::GenericOp createGeneric(Location loc, ArrayRef<Value> inputs,
                                         RankedTensorType outputType,
                                         ArrayRef<AffineMap> indexingMaps,
                                         ArrayRef<Attribute> iteratorTypes);
  void populateElementwiseGeneric(
      mlir::tt::d2m::GenericOp generic,
      function_ref<Value(OpBuilder &, Location, ValueRange)> bodyBuilder);

  MLIRContext *ctx;
  PatternRewriter &rewriter;
};

} // namespace mlir::maps

#endif
