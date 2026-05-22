#include "maps/Conversion/MapsToD2M/MapsABIRewriter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include <limits>

namespace mlir::maps {
namespace {

static BlockArgument getRootBlockArgument(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (def->getNumOperands() != 1)
      return {};
    value = def->getOperand(0);
  }
  return dyn_cast<BlockArgument>(value);
}

static void annotateForwardFunc(func::FuncOp func, MapsProgramInfo &program) {
  MLIRContext *ctx = func.getContext();
  func->setAttr("tt.function_type", StringAttr::get(ctx, "forward_device"));
  if (func.getSymName() == "main")
    func.setSymName("forward");

  DenseSet<unsigned> parameterArgs;
  if (program.init) {
    program.init.walk([&](maps::SendOp send) {
      if (BlockArgument arg = getRootBlockArgument(send.getValue()))
        parameterArgs.insert(arg.getArgNumber());
    });
  }

  for (BlockArgument arg : func.getArguments()) {
    auto argType = parameterArgs.contains(arg.getArgNumber())
                       ? mlir::tt::ttcore::ArgumentType::Parameter
                       : mlir::tt::ttcore::ArgumentType::Input;
    func.setArgAttr(arg.getArgNumber(), mlir::tt::ttcore::ArgumentTypeAttr::name,
                    mlir::tt::ttcore::ArgumentTypeAttr::get(ctx, argType));
  }
}

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

static bool isParameterArgument(func::FuncOp func, BlockArgument arg) {
  auto attr = func.getArgAttrOfType<mlir::tt::ttcore::ArgumentTypeAttr>(
      arg.getArgNumber(), mlir::tt::ttcore::ArgumentTypeAttr::name);
  return attr && attr.getValue() == mlir::tt::ttcore::ArgumentType::Parameter;
}

static std::string getSliceKey(BlockArgument arg, tensor::ExtractSliceOp slice) {
  std::string key = std::to_string(arg.getArgNumber());
  auto appendList = [&](ArrayRef<int64_t> values) {
    key.push_back(':');
    for (int64_t value : values) {
      key.append(std::to_string(value));
      key.push_back(',');
    }
  };
  appendList(slice.getStaticOffsets());
  appendList(slice.getStaticSizes());
  appendList(slice.getStaticStrides());
  return key;
}

static LogicalResult liftParameterSlices(func::FuncOp func) {
  struct SliceInfo {
    BlockArgument arg;
    tensor::ExtractSliceOp slice;
  };

  SmallVector<SliceInfo> slices;
  func.walk([&](tensor::ExtractSliceOp slice) {
    auto arg = dyn_cast<BlockArgument>(slice.getSource());
    if (!arg || !isParameterArgument(func, arg) || !hasStaticSlice(slice))
      return;
    slices.push_back({arg, slice});
  });

  llvm::sort(slices, [](const SliceInfo &lhs, const SliceInfo &rhs) {
    if (lhs.arg.getArgNumber() != rhs.arg.getArgNumber())
      return lhs.arg.getArgNumber() < rhs.arg.getArgNumber();
    return lhs.slice->isBeforeInBlock(rhs.slice);
  });

  llvm::StringMap<BlockArgument> liftedArgs;
  for (SliceInfo &info : slices) {
    auto sourceType = cast<RankedTensorType>(info.slice.getSource().getType());
    if (isIdentityStaticSlice(info.slice.getStaticOffsets(),
                              info.slice.getStaticSizes(),
                              info.slice.getStaticStrides(), sourceType)) {
      info.slice.replaceAllUsesWith(info.slice.getSource());
      continue;
    }

    std::string key = getSliceKey(info.arg, info.slice);
    auto it = liftedArgs.find(key);
    if (it == liftedArgs.end()) {
      unsigned argIndex = func.getNumArguments();
      DictionaryAttr attrs = func.getArgAttrDict(info.arg.getArgNumber());
      if (failed(func.insertArgument(argIndex, info.slice.getType(), attrs,
                                     info.slice.getLoc())))
        return failure();
      it = liftedArgs.try_emplace(key, func.getArgument(argIndex)).first;
    }

    info.slice.replaceAllUsesWith(it->second);
  }

  return success();
}

static LogicalResult eraseUnusedParameterArgs(func::FuncOp func) {
  SmallVector<unsigned> argsToErase;
  for (BlockArgument arg : func.getArguments()) {
    if (arg.use_empty() && isParameterArgument(func, arg))
      argsToErase.push_back(arg.getArgNumber());
  }

  for (auto it = argsToErase.rbegin(); it != argsToErase.rend(); ++it) {
    if (failed(func.eraseArgument(*it)))
      return failure();
  }

  return success();
}

static LogicalResult reorderParameterArgsByFirstUse(func::FuncOp func) {
  DenseMap<Operation *, unsigned> opOrder;
  unsigned nextOrder = 0;
  func.walk([&](Operation *op) { opOrder[op] = nextOrder++; });

  struct ArgUseInfo {
    BlockArgument arg;
    unsigned firstUseOrder;
  };

  SmallVector<ArgUseInfo> parameterArgs;
  for (BlockArgument arg : func.getArguments()) {
    if (!isParameterArgument(func, arg))
      continue;

    unsigned firstUseOrder = std::numeric_limits<unsigned>::max();
    for (OpOperand &use : arg.getUses())
      firstUseOrder = std::min(firstUseOrder, opOrder[use.getOwner()]);
    parameterArgs.push_back({arg, firstUseOrder});
  }

  auto desiredOrder = parameterArgs;
  llvm::sort(desiredOrder, [](const ArgUseInfo &lhs, const ArgUseInfo &rhs) {
    if (lhs.firstUseOrder != rhs.firstUseOrder)
      return lhs.firstUseOrder < rhs.firstUseOrder;
    return lhs.arg.getArgNumber() < rhs.arg.getArgNumber();
  });

  if (llvm::equal(parameterArgs, desiredOrder,
                  [](const ArgUseInfo &lhs, const ArgUseInfo &rhs) {
                    return lhs.arg == rhs.arg;
                  })) {
    return success();
  }

  SmallVector<BlockArgument> replacementArgs;
  for (const ArgUseInfo &info : desiredOrder) {
    unsigned argIndex = func.getNumArguments();
    DictionaryAttr attrs = func.getArgAttrDict(info.arg.getArgNumber());
    if (failed(func.insertArgument(argIndex, info.arg.getType(), attrs,
                                   func.getLoc())))
      return failure();
    replacementArgs.push_back(func.getArgument(argIndex));
  }

  for (auto [oldInfo, newArg] : llvm::zip_equal(desiredOrder, replacementArgs))
    oldInfo.arg.replaceAllUsesWith(newArg);

  SmallVector<unsigned> argsToErase;
  for (const ArgUseInfo &info : parameterArgs)
    argsToErase.push_back(info.arg.getArgNumber());
  for (auto it = argsToErase.rbegin(); it != argsToErase.rend(); ++it) {
    if (failed(func.eraseArgument(*it)))
      return failure();
  }

  return success();
}

} // namespace

LogicalResult rewriteForwardABI(ModuleOp module, MapsProgramInfo &program) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    annotateForwardFunc(func, program);
    if (failed(liftParameterSlices(func)))
      return failure();
    if (failed(eraseUnusedParameterArgs(func)))
      return failure();
    if (failed(reorderParameterArgsByFirstUse(func)))
      return failure();
  }

  return success();
}

LogicalResult normalizeParameterSlices(ModuleOp module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (failed(liftParameterSlices(func)))
      return failure();
    if (failed(eraseUnusedParameterArgs(func)))
      return failure();
    if (failed(reorderParameterArgsByFirstUse(func)))
      return failure();
  }

  return success();
}

LogicalResult rewriteOutputABI(ModuleOp module, MapsProgramInfo &program,
                               const ChannelLoweringState &channels) {
  IRRewriter rewriter(module.getContext());

  SmallVector<maps::RecvOp> recvsToReplace;
  SmallVector<Value> outputValues;
  module.walk([&](maps::RecvOp op) {
    auto channelIt = program.channels.find(op.getChannelAttr());
    if (!channels.values.contains(op.getChannelAttr()) ||
        channelIt == program.channels.end() || channelIt->second.dstHartId != -1) {
      return;
    }

    auto listIt = channels.valueLists.find(op.getChannelAttr());
    ValueRange values = listIt == channels.valueLists.end()
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
        channelIt != program.channels.end() && channelIt->second.dstHartId == -1) {
      sendsToErase.push_back(op);
    }
  });

  for (maps::SendOp send : sendsToErase)
    rewriter.eraseOp(send);

  return success();
}

} // namespace mlir::maps
