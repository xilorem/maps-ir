#include "IRCollector.h"

#include "collectors/CollectorSupport.h"
#include "collectors/L2Collector.h"
#include "collectors/OpCollector.h"
#include "collectors/SliceCollector.h"
#include "collectors/TransferCollector.h"

#include "mlir/IR/Operation.h"

#include <algorithm>
#include <limits>


namespace mlir::maps {

static LogicalResult collectExecutionContract(ModuleOp module,
                                              MagiaApplicationDescription &application) {
  auto numTokenSlots =
      module->getAttrOfType<IntegerAttr>("maps.num_token_slots");
  if (!numTokenSlots || numTokenSlots.getInt() <= 0 ||
      static_cast<uint64_t>(numTokenSlots.getInt()) >
          std::numeric_limits<unsigned>::max())
    return failure();
  application.numTokenSlots = static_cast<unsigned>(numTokenSlots.getInt());
  return success();
}

static LogicalResult collectDeploymentIdentity(ModuleOp module,
                                               MagiaApplicationDescription &application) {
  if (auto name = module->getAttrOfType<StringAttr>("maps.name"))
    application.sourceModelName = name.getValue().str();
  if (auto target = module->getAttrOfType<StringAttr>("maps.target")) {
    if (target.getValue() != application.target)
      return failure();
  }

  auto meshWidth = module->getAttrOfType<IntegerAttr>("maps.mesh_width");
  auto meshHeight = module->getAttrOfType<IntegerAttr>("maps.mesh_height");
  if (meshWidth && meshHeight && meshWidth.getInt() > 0 &&
      meshHeight.getInt() > 0 &&
      static_cast<uint64_t>(meshWidth.getInt()) <=
          std::numeric_limits<unsigned>::max() &&
      static_cast<uint64_t>(meshHeight.getInt()) <=
          std::numeric_limits<unsigned>::max()) {
    application.meshWidth = static_cast<unsigned>(meshWidth.getInt());
    application.meshHeight = static_cast<unsigned>(meshHeight.getInt());
  } else if (meshWidth || meshHeight) {
    return failure();
  }
  return success();
}

static LogicalResult collectBundleMetadata(ModuleOp module,
                                           MagiaApplicationDescription &application) {
  auto bundle = module->getAttrOfType<DictionaryAttr>("maps.bundle");
  if (!bundle)
    return success();

  auto schemaVersion = bundle.getAs<IntegerAttr>("schema_version");
  auto weightsFile = bundle.getAs<StringAttr>("weights_file");
  auto weightsSize = bundle.getAs<IntegerAttr>("weights_size");
  auto weightsSha256 = bundle.getAs<StringAttr>("weights_sha256");
  auto alignment = bundle.getAs<IntegerAttr>("alignment");
  auto endianness = bundle.getAs<StringAttr>("endianness");
  if (!schemaVersion || schemaVersion.getInt() != 1 || !weightsFile ||
      !weightsSize || weightsSize.getInt() < 0 || !weightsSha256 ||
      !alignment || alignment.getInt() <= 0 ||
      (alignment.getInt() & (alignment.getInt() - 1)) != 0 ||
      !endianness || endianness.getValue() != "little")
    return failure();
  if (weightsFile.getValue().contains('"') ||
      weightsFile.getValue().contains('\n') ||
      weightsSha256.getValue().size() != 64)
    return failure();

  application.hasPackedWeights = true;
  application.weightsFile = weightsFile.getValue().str();
  application.weightsSize = weightsSize.getInt();
  application.weightsSha256 = weightsSha256.getValue().str();
  application.weightsAlignment = alignment.getInt();
  if (application.l2Base % application.weightsAlignment != 0)
    return failure();

  uint64_t initializerCapacity = application.l2End - application.l2Base;
  if (application.target == "magia-v3")
    initializerCapacity = kMagiaV3L2BulkBytes;
  if (auto capacity = module->getAttrOfType<IntegerAttr>(
          "maps.l2_capacity_bytes")) {
    if (capacity.getInt() <= 0)
      return failure();
    initializerCapacity = std::min(initializerCapacity,
                                   static_cast<uint64_t>(capacity.getInt()));
    if (application.target != "magia-v3") {
      uint64_t configuredEnd = application.l2Base + capacity.getInt();
      application.l2End = std::min(application.l2End, configuredEnd);
    }
  }
  if (application.weightsSize > initializerCapacity) {
    llvm::errs() << "packed initializer image (" << application.weightsSize
                 << " bytes) exceeds "
                 << (application.target == "magia-v3" ? "L2 bulk" : "L2 allocation")
                 << " capacity (" << initializerCapacity << " bytes)\n";
    return failure();
  }
  // V3 initializers are emitted into .l2_bulk and addressed by ELF symbols.
  // Its bounded application-data window contains only mutable runtime buffers.
  // V2 retains its existing combined initializer/runtime allocation contract.
  application.nextL2Address = application.target == "magia-v3"
      ? application.l2Base
      : alignTo(application.l2Base + application.weightsSize,
                kGeneratedL2AlignmentBytes);
  return success();
}


static LogicalResult collectInitRegion(Operation *initOp, MagiaApplicationDescription &application) {

  llvm::StringMap<L2SourceInfo> sources; // llvm hash map optimized for llvm strings
  collectL2Sources(initOp, sources);

  WalkResult result = initOp->walk([&](Operation *op) -> WalkResult {

    // search for maps.store operations
    if (op->getName().getStringRef() != "maps.store")
      return WalkResult::advance();

    if (failed(collectInitializerStore(op, sources, application)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();

}


static LogicalResult collectRunRegion(Operation *runOp, MagiaApplicationDescription &application) {
  
  llvm::StringMap<L2SourceInfo> sources;
  collectL2Sources(runOp, sources);

  RunCollectionState state;
  if (failed(collectTileRecvs(runOp, state, application)))
    return failure();
  
  // Collect the host-side insertion offsets of outputs. The operation is the
  // parent of the run.op as the collection happens after running.
  collectOutputInsertOffsets(runOp->getParentOp(), state.outputInserts);

  WalkResult result = runOp->walk([&](Operation *op) -> WalkResult {
    StringRef name = op->getName().getStringRef();

    if (name == "maps.recv") {
      if (failed(collectInputRecv(op, sources, state, application)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (name == "maps.load") {
      if (failed(collectLoad(op, state, application)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (name == "tensor.extract_slice") {
      if (failed(collectExtractSliceAlias(op, state, application)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (name == "tensor.insert_slice") {
      if (failed(collectInsertSliceAlias(op, state, application)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    /* TODO: create a maps.execute dialect so that we can directly
    branch with that instead of linalg operations.
    This would become something like:
    
    if (name == "maps.execute"){
      if (failed(collectComputeOp(op, state, application)))
        return WalkResult::interrupt();
      return WalkResult::advance();}

    This can also be helpful to account for ops with 
    intra-stage copies like softmax
    */

    if (name == "linalg.matmul" || name == "linalg.generic" ||
        name == "maps.kernel") {
      if (failed(collectComputeOp(op, state, application)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (name != "maps.send")
      return WalkResult::advance();

    if (failed(collectProducedSend(op, state, application)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}


// Top level collector function
LogicalResult collectMagiaApplicationDescription(ModuleOp module, MagiaApplicationDescription &application) {
  if (failed(collectDeploymentIdentity(module, application)) ||
      failed(collectExecutionContract(module, application)) ||
      failed(collectBundleMetadata(module, application)))
    return failure();
                                        
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    StringRef name = op->getName().getStringRef();

    // init region collection
    if (name == "maps.init") {
      if (failed(collectInitRegion(op, application)))
        return WalkResult::interrupt();
    }

    // run region collector
    if (name == "maps.run") {
      if (failed(collectRunRegion(op, application)))
        return WalkResult::interrupt();
    }
  
    return WalkResult::advance();
  });

  // return failure if any collection phase encountered a bad pattern
  if (result.wasInterrupted())
    return failure();

  return success();
}


} // namespace mlir::maps
