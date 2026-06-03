#include "maps/Conversion/MapsToD2M/TileBodyEmitter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

#include <array>

namespace mlir::maps {
namespace {

FailureOr<Value> lookupMappedValue(IRMapping &mapping, Value value) {
  if (Value mapped = mapping.lookupOrNull(value))
    return mapped;
  return failure();
}

class ConstantTileBodyEmitter final : public TileBodyOpEmitter {
public:
  bool match(Operation *op) const override {
    return isa<arith::ConstantOp>(op);
  }

  FailureOr<Value> lower(PatternRewriter &rewriter, Operation *op,
                         IRMapping &mapping) const override {
    auto constant = cast<arith::ConstantOp>(op);
    auto cloned = rewriter.create<arith::ConstantOp>(
        constant.getLoc(), constant.getType(), constant.getValueAttr());
    mapping.map(constant.getResult(), cloned.getResult());
    return cloned.getResult();
  }
};

template <typename SourceOp, typename TargetOp>
class BinaryTileBodyEmitter final : public TileBodyOpEmitter {
public:
  bool match(Operation *op) const override { return isa<SourceOp>(op); }

  FailureOr<Value> lower(PatternRewriter &rewriter, Operation *op,
                         IRMapping &mapping) const override {
    auto source = cast<SourceOp>(op);
    auto lhs = lookupMappedValue(mapping, source.getLhs());
    auto rhs = lookupMappedValue(mapping, source.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    auto lowered = rewriter.create<TargetOp>(source.getLoc(), (*lhs).getType(),
                                             *lhs, *rhs);
    mapping.map(source.getResult(), lowered.getResult());
    return lowered.getResult();
  }
};

template <typename SourceOp, typename TargetOp>
class UnaryTileBodyEmitter final : public TileBodyOpEmitter {
public:
  bool match(Operation *op) const override { return isa<SourceOp>(op); }

  FailureOr<Value> lower(PatternRewriter &rewriter, Operation *op,
                         IRMapping &mapping) const override {
    auto source = cast<SourceOp>(op);
    auto input = lookupMappedValue(mapping, source.getOperand());
    if (failed(input))
      return failure();
    auto lowered =
        rewriter.create<TargetOp>(source.getLoc(), (*input).getType(), *input);
    mapping.map(source.getResult(), lowered.getResult());
    return lowered.getResult();
  }
};

ArrayRef<const TileBodyOpEmitter *> getTileBodyEmitters() {
  static const ConstantTileBodyEmitter constantEmitter;
  static const BinaryTileBodyEmitter<arith::AddFOp, mlir::tt::d2m::TileAddOp>
      addEmitter;
  static const BinaryTileBodyEmitter<arith::SubFOp, mlir::tt::d2m::TileSubOp>
      subEmitter;
  static const BinaryTileBodyEmitter<arith::MulFOp, mlir::tt::d2m::TileMulOp>
      mulEmitter;
  static const BinaryTileBodyEmitter<arith::DivFOp, mlir::tt::d2m::TileDivOp>
      divEmitter;
  static const UnaryTileBodyEmitter<arith::NegFOp,
                                    mlir::tt::d2m::TileNegativeOp>
      negEmitter;
  static const UnaryTileBodyEmitter<math::ExpOp, mlir::tt::d2m::TileExpOp>
      expEmitter;
  static const std::array<const TileBodyOpEmitter *, 7> emitters = {
      &constantEmitter, &addEmitter, &subEmitter, &mulEmitter,
      &divEmitter,      &negEmitter, &expEmitter};
  return emitters;
}

} // namespace

FailureOr<Value> lowerTileBodyOp(PatternRewriter &rewriter, Operation *op,
                                 IRMapping &mapping) {
  for (const TileBodyOpEmitter *emitter : getTileBodyEmitters()) {
    if (!emitter->match(op))
      continue;
    return emitter->lower(rewriter, op, mapping);
  }
  return failure();
}

} // namespace mlir::maps
