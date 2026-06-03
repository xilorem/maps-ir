#ifndef MAPS_CONVERSION_MAPSTOD2M_COMPUTEOPREGISTRY_H
#define MAPS_CONVERSION_MAPSTOD2M_COMPUTEOPREGISTRY_H

#include "maps/Conversion/MapsToD2M/D2MGenericBuilder.h"
#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <optional>

namespace mlir::maps {

struct ComputeOpLoweringContext {
  maps::TileOp tile;
  D2MGenericBuilder &builder;
  PatternRewriter &rewriter;
  std::function<FailureOr<Value>(Value)> getSourceValue;
  std::function<FailureOr<Value>(Value)> hoistValue;
  std::function<FailureOr<std::optional<Value>>(Value, RankedTensorType,
                                                tensor::ExtractSliceOp)>
      lowerValue;
  std::function<Value(Location, Value, ArrayRef<int64_t>, ArrayRef<int64_t>)>
      createStaticSlice;
  std::function<Value(Location, Value)> materializeContiguous;
};

class ComputeOpEmitter {
public:
  virtual ~ComputeOpEmitter() = default;
  virtual bool match(Value value) const = 0;
  virtual FailureOr<std::optional<Value>>
  lower(Value value, RankedTensorType outputType,
        tensor::ExtractSliceOp outputSlice,
        ComputeOpLoweringContext &ctx) const = 0;
};

class SendValueEmitter {
public:
  virtual ~SendValueEmitter() = default;
  virtual bool match(Value value) const = 0;
  virtual FailureOr<std::optional<SmallVector<Value>>>
  lower(Value value, RankedTensorType outputType,
        tensor::ExtractSliceOp outputSlice,
        ComputeOpLoweringContext &ctx) const = 0;
};

FailureOr<std::optional<Value>>
lowerRegisteredComputeOp(Value value, RankedTensorType outputType,
                         tensor::ExtractSliceOp outputSlice,
                         ComputeOpLoweringContext &ctx);

FailureOr<std::optional<SmallVector<Value>>>
lowerRegisteredSendValue(Value value, RankedTensorType outputType,
                         tensor::ExtractSliceOp outputSlice,
                         ComputeOpLoweringContext &ctx);

} // namespace mlir::maps

#endif
