#include "maps/Conversion/MapsToD2M/FinishEmitter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::maps {

LogicalResult lowerFinishTransfers(ModuleOp module, MapsProgramInfo &program,
                                   const ChannelLoweringState &channels) {
  IRRewriter rewriter(module.getContext());

  SmallVector<maps::RecvOp> recvsToReplace;
  SmallVector<Value> outputValues;
  module.walk([&](maps::RecvOp op) {
    auto channelIt = program.channels.find(op.getChannelAttr());
    if (!channels.values.contains(op.getChannelAttr()) ||
        channelIt == program.channels.end() || channelIt->second.dstHartId != -1)
      return;

    auto listIt = channels.valueLists.find(op.getChannelAttr());
    ValueRange values =
        listIt == channels.valueLists.end()
            ? ValueRange(channels.values.lookup(op.getChannelAttr()))
            : ValueRange(listIt->second);
    llvm::append_range(outputValues, values);
    recvsToReplace.push_back(op);
  });

  if (!outputValues.empty()) {
    auto func = module.lookupSymbol<func::FuncOp>("forward");
    if (!func)
      func = *module.getOps<func::FuncOp>().begin();

    SmallVector<func::ReturnOp> returnsToReplace;
    func.walk([&](func::ReturnOp op) {
      if (op.getNumOperands() == 0)
        returnsToReplace.push_back(op);
    });

    for (func::ReturnOp ret : returnsToReplace) {
      rewriter.setInsertionPoint(ret);
      SmallVector<Value> returnedValues;
      SmallVector<Type> returnedTypes;
      for (Value output : outputValues) {
        auto type = dyn_cast<RankedTensorType>(output.getType());
        if (!type || !type.getEncoding()) {
          returnedValues.push_back(output);
          returnedTypes.push_back(output.getType());
          continue;
        }

        auto layout = cast<mlir::tt::ttcore::MetalLayoutAttr>(type.getEncoding());
        Type hostElementType = type.getElementType();
        if (auto tileType = dyn_cast<mlir::tt::ttcore::TileType>(hostElementType))
          hostElementType = tileType.getElementType();
        auto hostType =
            RankedTensorType::get(layout.getLogicalShape(), hostElementType);
        auto hostOutput = rewriter.create<mlir::tt::d2m::EmptyOp>(
            output.getLoc(), hostType.getShape(), hostType.getElementType(),
            /*encoding=*/nullptr);
        auto toHost = rewriter.create<mlir::tt::d2m::ToLayoutOp>(
            output.getLoc(), output, hostOutput.getResult());
        returnedValues.push_back(toHost.getResult(0));
        returnedTypes.push_back(hostType);
      }

      func.setFunctionType(
          rewriter.getFunctionType(func.getArgumentTypes(), returnedTypes));
      rewriter.replaceOpWithNewOp<func::ReturnOp>(ret, returnedValues);
    }
  }

  for (maps::RecvOp recv : recvsToReplace) {
    rewriter.setInsertionPoint(recv);
    auto listIt = channels.valueLists.find(recv.getChannelAttr());
    if (listIt != channels.valueLists.end() && listIt->second.size() > 1) {
      auto resultType = cast<RankedTensorType>(recv.getResult().getType());
      auto replacement = rewriter.create<tensor::EmptyOp>(
          recv.getLoc(), resultType.getShape(), resultType.getElementType());
      rewriter.replaceOp(recv, replacement.getResult());
      continue;
    }

    rewriter.replaceOp(recv, channels.values.lookup(recv.getChannelAttr()));
  }

  SmallVector<maps::SendOp> sendsToErase;
  module.walk([&](maps::SendOp op) {
    auto channelIt = program.channels.find(op.getChannelAttr());
    if (channels.values.contains(op.getChannelAttr()) &&
        channelIt != program.channels.end() && channelIt->second.dstHartId == -1)
      sendsToErase.push_back(op);
  });

  for (maps::SendOp send : sendsToErase)
    rewriter.eraseOp(send);

  return success();
}

} // namespace mlir::maps
