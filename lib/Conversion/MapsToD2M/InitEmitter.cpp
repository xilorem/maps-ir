#include "maps/Conversion/MapsToD2M/InitEmitter.h"

namespace mlir::maps {
namespace {

static void eraseBlockOps(PatternRewriter &rewriter, Block &body) {
  SmallVector<Operation *> nestedOps;
  for (Operation &nested : llvm::make_early_inc_range(body))
    nestedOps.push_back(&nested);

  for (Operation *nested : llvm::reverse(nestedOps)) {
    nested->dropAllDefinedValueUses();
    rewriter.eraseOp(nested);
  }
}

} // namespace

LogicalResult lowerInitTileProgram(const TileProgramInfo &tileProgram,
                                   PatternRewriter &rewriter) {
  if (!tileProgram.isInitStorage)
    return const_cast<Operation *>(const_cast<maps::TileOp &>(tileProgram.op).getOperation())
        ->emitOpError("expected init storage tile program");

  auto &tileOp = const_cast<maps::TileOp &>(tileProgram.op);
  Block &body = tileOp.getBody().front();
  body.dropAllDefinedValueUses();
  eraseBlockOps(rewriter, body);
  rewriter.eraseOp(tileOp);
  return success();
}

void eraseHostToTileSends(ModuleOp module, PatternRewriter &rewriter) {
  SmallVector<maps::SendOp> sendsToErase;
  module.walk([&](maps::SendOp send) {
    if (static_cast<int64_t>(send.getSrcHartid()) == -1)
      sendsToErase.push_back(send);
  });

  for (maps::SendOp send : sendsToErase)
    rewriter.eraseOp(send);
}

} // namespace mlir::maps
