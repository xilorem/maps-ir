#include "maps/Conversion/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"

namespace mlir::maps {
#define GEN_PASS_DEF_LOWERSTANDALONELINALGELEMENTWISE
#include "maps/Conversion/Passes.h.inc"

namespace {

static bool hasIdentityMaps(linalg::GenericOp generic) {
  if (!generic.hasPureTensorSemantics() && !generic.hasPureBufferSemantics())
    return false;
  unsigned rank = generic.getNumLoops();
  for (AffineMap map : generic.getIndexingMapsArray()) {
    if (!map.isIdentity() || map.getNumResults() != rank)
      return false;
  }
  return true;
}

static bool isStandaloneMemrefAdd(linalg::GenericOp generic) {
  if (!generic.hasPureBufferSemantics() || generic.getNumDpsInputs() != 2 ||
      generic.getNumDpsInits() != 1 || generic.getNumResults() != 0 ||
      !generic.getNumLoops())
    return false;
  if (!llvm::all_of(generic.getDpsInputs(), [](Value value) {
        return isa<MemRefType>(value.getType());
      }))
    return false;
  if (!isa<MemRefType>(generic.getDpsInits()[0].getType()) || !hasIdentityMaps(generic))
    return false;

  Block &body = generic.getRegion().front();
  Operation &front = body.front();
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (auto add = dyn_cast<arith::AddFOp>(front))
    return yield && add.getLhs() == body.getArgument(0) &&
           add.getRhs() == body.getArgument(1) &&
           yield.getValues() == ValueRange{add.getResult()};
  if (auto add = dyn_cast<mlir::tt::d2m::TileAddOp>(front))
    return yield && add.getLhs() == body.getArgument(0) &&
           add.getRhs() == body.getArgument(1) &&
           yield.getValues() == ValueRange{add.getResult()};
  return false;
}

static bool isStandaloneMemrefExp(linalg::GenericOp generic) {
  if (!generic.hasPureBufferSemantics() || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || generic.getNumResults() != 0 ||
      !generic.getNumLoops())
    return false;
  if (!isa<MemRefType>(generic.getDpsInputs()[0].getType()) ||
      !isa<MemRefType>(generic.getDpsInits()[0].getType()) || !hasIdentityMaps(generic))
    return false;

  Block &body = generic.getRegion().front();
  Operation &front = body.front();
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (auto exp = dyn_cast<math::ExpOp>(front))
    return yield && exp.getOperand() == body.getArgument(0) &&
           yield.getValues() == ValueRange{exp.getResult()};
  if (auto exp = dyn_cast<mlir::tt::d2m::TileExpOp>(front))
    return yield && exp.getOperand() == body.getArgument(0) &&
           yield.getValues() == ValueRange{exp.getResult()};
  return false;
}

template <typename ScalarBuilder>
static LogicalResult lowerGenericToAffine(linalg::GenericOp generic,
                                          ScalarBuilder buildScalar) {
  auto outputType = dyn_cast<MemRefType>(generic.getDpsInits()[0].getType());
  if (!outputType || !outputType.hasStaticShape())
    return failure();

  OpBuilder builder(generic);
  SmallVector<Value> ivs;
  auto buildLoopNest = [&](auto &self, unsigned dim) -> void {
    if (dim == outputType.getRank()) {
      SmallVector<Value> loadedInputs;
      loadedInputs.reserve(generic.getNumDpsInputs());
      for (Value input : generic.getDpsInputs())
        loadedInputs.push_back(
            builder.create<memref::LoadOp>(generic.getLoc(), input, ivs));
      Value scalar = buildScalar(builder, generic.getLoc(), loadedInputs);
      builder.create<memref::StoreOp>(generic.getLoc(), scalar,
                                      generic.getDpsInits()[0], ivs);
      return;
    }

    Value lower = builder.create<arith::ConstantIndexOp>(generic.getLoc(), 0);
    Value upper = builder.create<arith::ConstantIndexOp>(generic.getLoc(),
                                                         outputType.getDimSize(dim));
    Value step = builder.create<arith::ConstantIndexOp>(generic.getLoc(), 1);
    auto loop =
        builder.create<scf::ForOp>(generic.getLoc(), lower, upper, step);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    ivs.push_back(loop.getInductionVar());
    self(self, dim + 1);
    ivs.pop_back();
  };

  buildLoopNest(buildLoopNest, 0);
  generic.erase();
  return success();
}

struct LowerStandaloneLinalgElementwisePass
    : public impl::LowerStandaloneLinalgElementwiseBase<
          LowerStandaloneLinalgElementwisePass> {
  void runOnOperation() override {
    SmallVector<linalg::GenericOp> worklist;
    getOperation()->walk([&](linalg::GenericOp generic) {
      if (isStandaloneMemrefAdd(generic) || isStandaloneMemrefExp(generic))
        worklist.push_back(generic);
    });

    for (linalg::GenericOp generic : worklist) {
      LogicalResult result = failure();
      if (isStandaloneMemrefAdd(generic)) {
        Operation &front = generic.getRegion().front().front();
        result = lowerGenericToAffine(
            generic, [&](OpBuilder &builder, Location loc, ValueRange inputs) {
              if (isa<arith::AddFOp>(front))
                return Value(builder
                                 .create<arith::AddFOp>(loc, inputs[0], inputs[1])
                                 .getResult());
              return Value(builder
                               .create<mlir::tt::d2m::TileAddOp>(
                                   loc, inputs[0].getType(), inputs[0], inputs[1])
                               .getResult());
            });
      } else if (isStandaloneMemrefExp(generic)) {
        Operation &front = generic.getRegion().front().front();
        result = lowerGenericToAffine(
            generic, [&](OpBuilder &builder, Location loc, ValueRange inputs) {
              if (isa<math::ExpOp>(front))
                return Value(
                    builder.create<math::ExpOp>(loc, inputs[0]).getResult());
              return Value(builder
                               .create<mlir::tt::d2m::TileExpOp>(
                                   loc, inputs[0].getType(), inputs[0])
                               .getResult());
            });
      }

      if (failed(result))
        return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::maps
