#include "maps/Conversion/Passes.h"
#include "maps/Dialect/Maps/IR/Maps.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::maps {
#define GEN_PASS_DEF_CONVERTMAPSTOD2M
#include "maps/Conversion/Passes.h.inc"

namespace {

struct ChannelInfo {
  FlatSymbolRefAttr name;
  int64_t srcHartId = -1;
  int64_t dstHartId = -1;
  RankedTensorType type;
  maps::SendOp send;
  SmallVector<maps::RecvOp> recvs;
};

struct TileInfo {
  int64_t tileId = -1;
  SmallVector<int64_t> coords;
  maps::TileOp op;
};

struct StageInfo {
  int64_t stageId = -1;
  StringRef sourceName;
  maps::StageOp op;
  SmallVector<TileInfo> tiles;
};

struct MapsProgramInfo {
  maps::InitOp init;
  maps::RunOp run;
  SmallVector<StageInfo> stages;
  DenseMap<Attribute, ChannelInfo> channels;
};

} // namespace


static FailureOr<MapsProgramInfo> collectMapsProgram(ModuleOp module){
  MapsProgramInfo info;

  // register init
  module.walk([&](maps::InitOp op){
    if (info.init)
      op.emitError("expected at most one maps.init");
    info.init = op;
  });


  module.walk([&](maps::RunOp op){
    if (info.run)
      op.emitError("expected at most one maps.init");
    info.run = op;
  });


  module.walk([&](maps::StageOp stage){
    StageInfo stageInfo;
    stageInfo.op = stage;
    stageInfo.stageId = static_cast<int64_t>(stage.getStageId());
    stage.walk([&](maps::TileOp tile){
      TileInfo tileInfo;
      tileInfo.op = tile;
      tileInfo.tileId = static_cast<int64_t>(tile.getTileId());
      tileInfo.coords = static_cast<SmallVector<int64_t>>(tile.getCoords());
      stageInfo.tiles.push_back(tileInfo);
    });
    info.stages.push_back(stageInfo);

  }); 

  module.walk([&](maps::SendOp op){
    FlatSymbolRefAttr channelName = op.getChannelAttr();

    ChannelInfo channel;
    channel.name = channelName;
    channel.dstHartId = static_cast<int64_t>(op.getDstHartid());
    channel.srcHartId = static_cast<int64_t>(op.getSrcHartid());
    channel.type = dyn_cast<RankedTensorType>(op.getValue().getType());
    channel.send = op;

    info.channels[channelName] = channel;
  });

  module.walk([&](maps::RecvOp op){
    Attribute channelName = op.getChannelAttr();
    auto it = info.channels.find(channelName);

    if (it == info.channels.end()) {
      op.emitError("recv without matching send for channel ") << channelName;
      return;
    }
    it->second.recvs.push_back(op);
  });

  return info;
}


namespace {
struct InitOpLowering : public OpConversionPattern<maps::InitOp> {
  using OpConversionPattern<maps::InitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(maps::InitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();

    rewriter.setInsertionPoint(op);

    SmallVector<Operation *> opsToMove;
    for (Operation &nested : llvm::make_early_inc_range(body)) {
      if (isa<maps::SendOp, maps::RecvOp, maps::TileOp>(nested))
        continue;

      opsToMove.push_back(&nested);
    }

    for (Operation *nested : opsToMove)
      nested->moveBefore(op);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace


namespace {
struct ConvertMapsToD2MPass
    : impl::ConvertMapsToD2MBase<ConvertMapsToD2MPass> {
  void runOnOperation() override {

    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    FailureOr<MapsProgramInfo> info = collectMapsProgram(module);
    if (failed(info)) {
      signalPassFailure();
    return;
  }
    
    ConversionTarget target(*ctx);
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();

    // Target dialects.
    target.addLegalDialect<mlir::tt::ttcore::TTCoreDialect>();
    target.addLegalDialect<mlir::tt::d2m::D2MDialect>();

    // Source dialect
    target.addIllegalDialect<mlir::maps::MapsDialect>();


    // mark as legal ops that aren't mentioned
    target.markUnknownOpDynamicallyLegal([](Operation *){
      return true;
    });

    RewritePatternSet patterns(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))){
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::maps