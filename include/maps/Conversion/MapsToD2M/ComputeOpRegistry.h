#ifndef MAPS_CONVERSION_MAPSTOD2M_COMPUTEOPREGISTRY_H
#define MAPS_CONVERSION_MAPSTOD2M_COMPUTEOPREGISTRY_H

#include "maps/Conversion/MapsToD2M/D2MGenericBuilder.h"
#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <optional>

namespace mlir::maps {

struct ComputeOpLoweringContext {
  /* This struct contains all the information needed to lower a compute operation
  inside a maps.tile. It is passed to the registered compute op emitters to perform the lowering
  */

  maps::TileOp tile; // which tile computes the op
  D2MGenericBuilder &builder; // custom builder for creating d2m.generic ops
  PatternRewriter &rewriter; // generic mlir rewrite helper
  std::function<FailureOr<Value>(Value)> getSourceValue; // given a value, what's the source value that should be used for the lowering 
                                                         //(e.g. for recv and load ops we need to look for the corresponding channel/slot value)
  std::function<FailureOr<Value>(Value)> hoistValue;  
  std::function<FailureOr<std::optional<Value>>(Value, RankedTensorType,
                                                tensor::ExtractSliceOp)>
      lowerValue; // recursive lowering function
  std::function<Value(Location, Value, ArrayRef<int64_t>, ArrayRef<int64_t>)>
      createStaticSlice; // conceptually, source tensor + offsets + sizes -> tensor.extract_slice
  std::function<Value(Location, Value)> materializeContiguous; // takes a tensor value, often a slice/view-like value, and materializes it into a standalone contiguous tensor.
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
