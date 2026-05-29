#include "maps/Conversion/MapsToD2M/MapsCleanup.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::maps {
namespace {

static bool hasStaticSlice(tensor::ExtractSliceOp slice) {
  return llvm::none_of(slice.getStaticOffsets(), ShapedType::isDynamic) &&
         llvm::none_of(slice.getStaticSizes(), ShapedType::isDynamic) &&
         llvm::none_of(slice.getStaticStrides(), ShapedType::isDynamic);
}

static bool isIdentityStaticSlice(ArrayRef<int64_t> offsets,
                                  ArrayRef<int64_t> sizes,
                                  ArrayRef<int64_t> strides,
                                  RankedTensorType type) {
  return llvm::all_of(offsets, [](int64_t offset) { return offset == 0; }) &&
         llvm::all_of(strides, [](int64_t stride) { return stride == 1; }) &&
         llvm::equal(sizes, type.getShape());
}

static bool foldIdentitySlice(tensor::ExtractSliceOp slice) {
  if (slice->getNumOperands() == 0 || !slice->getOperand(0))
    return false;
  auto sourceType = dyn_cast<RankedTensorType>(slice->getOperand(0).getType());
  if (!sourceType || !hasStaticSlice(slice))
    return false;
  if (!isIdentityStaticSlice(slice.getStaticOffsets(), slice.getStaticSizes(),
                             slice.getStaticStrides(), sourceType)) {
    return false;
  }

  slice.replaceAllUsesWith(slice->getOperand(0));
  return true;
}

static bool foldIdentityMaterialization(tensor::InsertSliceOp insertSlice) {
  auto sourceType = dyn_cast<RankedTensorType>(insertSlice.getSource().getType());
  auto resultType = dyn_cast<RankedTensorType>(insertSlice.getType());
  auto empty = insertSlice.getDest().getDefiningOp<tensor::EmptyOp>();
  if (!sourceType || !resultType || !empty || sourceType != resultType)
    return false;
  if (isa_and_nonnull<tensor::ExtractSliceOp>(insertSlice.getSource().getDefiningOp()))
    return false;
  if (!isIdentityStaticSlice(insertSlice.getStaticOffsets(),
                             insertSlice.getStaticSizes(),
                             insertSlice.getStaticStrides(), sourceType)) {
    return false;
  }

  insertSlice.replaceAllUsesWith(insertSlice.getSource());
  return true;
}

} // namespace

void cleanupGeneratedIR(ModuleOp module) {
  IRRewriter rewriter(module.getContext());

  bool changed = true;
  while (changed) {
    changed = false;

    module.walk([&](tensor::ExtractSliceOp slice) {
      if (foldIdentitySlice(slice))
        changed = true;
    });
    module.walk([&](tensor::InsertSliceOp insertSlice) {
      if (foldIdentityMaterialization(insertSlice))
        changed = true;
    });

    SmallVector<Operation *> deadOps;
    module.walk([&](Operation *op) {
      if (op->use_empty() &&
          isa<tensor::ExtractSliceOp, tensor::InsertSliceOp, tensor::EmptyOp>(op)) {
        deadOps.push_back(op);
      }
    });

    for (Operation *op : deadOps) {
      rewriter.eraseOp(op);
      changed = true;
    }
  }
}

LogicalResult verifyNoMapsOps(ModuleOp module) {
  bool hasMapsOp = false;
  module.walk([&](Operation *op) {
    if (isa<ModuleOp>(op))
      return;
    if (isa<maps::MapsDialect>(op->getDialect())) {
      op->emitError("failed to convert maps operation");
      hasMapsOp = true;
    }
  });
  return success(!hasMapsOp);
}

} // namespace mlir::maps
