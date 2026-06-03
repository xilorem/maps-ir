#include "maps/Conversion/MapsToD2M/MapsABIRewriter.h"
#include "maps/Conversion/MapsToD2M/FinishEmitter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
  func.walk([&](Operation *op) {
    auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
    if (!slice || op->getNumOperands() == 0)
      return;
    Value source = op->getOperand(0);
    if (!source)
      return;
    auto arg = dyn_cast<BlockArgument>(source);
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
    auto sourceType = cast<RankedTensorType>(info.arg.getType());
    if (isIdentityStaticSlice(info.slice.getStaticOffsets(),
                              info.slice.getStaticSizes(),
                              info.slice.getStaticStrides(), sourceType)) {
      info.slice.replaceAllUsesWith(info.arg);
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
  return lowerFinishTransfers(module, program, channels);
}

} // namespace mlir::maps
