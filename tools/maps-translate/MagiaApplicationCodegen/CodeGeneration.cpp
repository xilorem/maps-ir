#include "CodeGeneration.h"
#include "Emitter.h"
#include "IRCollector.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::maps {

static llvm::cl::opt<unsigned> magiaNumTokens(
    "maps-magia-num-tokens",
    llvm::cl::desc("number of runtime tokens described by generated MAGIA L2 "
                   "input/output buffers"),
    llvm::cl::init(kDefaultNumTokens));

static llvm::cl::opt<std::string> magiaOutputStem(
    "maps-magia-output-stem",
    llvm::cl::desc("MAGIA Application name and generated C symbol stem"),
    llvm::cl::init("application"));

static llvm::cl::opt<uint64_t> magiaL2Base(
    "maps-magia-l2-base",
    llvm::cl::desc("first target L2 address available to generated artifacts"),
    llvm::cl::init(kMagiaL2Base));

static llvm::cl::opt<uint64_t> magiaL2End(
    "maps-magia-l2-end",
    llvm::cl::desc("exclusive target L2 address limit"),
    llvm::cl::init(kMagiaL2End));

static llvm::cl::opt<bool> magiaRandomizeInputs(
    "maps-magia-test-randomize-inputs",
    llvm::cl::desc("test-only pseudo-random initialization of synthetic inputs"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> magiaWeightsFile(
    "maps-magia-weights-file",
    llvm::cl::desc("assembly .incbin path overriding bundle weights_file"),
    llvm::cl::init(""));

static llvm::cl::list<std::string> magiaRuntimeInputs(
    "maps-magia-runtime-input",
    llvm::cl::desc("Runtime Input assignment in original-tensor-name=file form"),
    llvm::cl::ZeroOrMore);

static void configureGeneratedRuntime(
    MagiaApplicationDescription &application, StringRef target) {
  application.target = target.str();
  application.numTokens = magiaNumTokens;
  application.outputStem = magiaOutputStem;
  application.l2Base = magiaL2Base;
  application.l2End = magiaL2End;
  application.nextL2Address = application.l2Base;
  application.randomizeInputs = magiaRandomizeInputs;
}

static LogicalResult writeTextArtifact(
    StringRef path,
    llvm::function_ref<void(llvm::raw_ostream &)> emit) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "failed to open '" << path << "': " << error.message()
                 << "\n";
    return failure();
  }
  emit(stream);
  stream.flush();
  if (stream.has_error()) {
    llvm::errs() << "failed to write '" << path << "'\n";
    return failure();
  }
  return success();
}

static LogicalResult writeBinaryArtifact(StringRef path, StringRef contents) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "failed to open '" << path << "': " << error.message()
                 << "\n";
    return failure();
  }
  stream.write(contents.data(), contents.size());
  stream.flush();
  if (stream.has_error()) {
    llvm::errs() << "failed to write '" << path << "'\n";
    return failure();
  }
  return success();
}

static std::string applicationArtifactPath(StringRef outputDirectory,
                                       StringRef filename) {
  llvm::SmallString<256> path(outputDirectory);
  llvm::sys::path::append(path, filename);
  return path.str().str();
}

static std::string sha256(StringRef contents) {
  llvm::SHA256 digest;
  digest.update(contents);
  return llvm::toHex(digest.final(), /*LowerCase=*/true);
}

static std::optional<std::string> manifestDtype(Type type) {
  if (type.isF16())
    return "float16";
  if (type.isF32())
    return "float32";
  if (type.isF64())
    return "float64";
  if (type.isBF16())
    return "bfloat16";
  if (auto integer = dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return "bool";
    if (integer.getWidth() == 8 && integer.isUnsigned())
      return "uint8";
    if (integer.getWidth() == 32)
      return "int32";
    if (integer.getWidth() == 64)
      return "int64";
  }
  return std::nullopt;
}

struct GeneratedArtifact {
  std::string path;
  std::string role;
};

static FailureOr<llvm::json::Array>
buildArtifactEntries(ArrayRef<GeneratedArtifact> artifacts,
                     StringRef outputDirectory) {
  llvm::json::Array entries;
  for (const GeneratedArtifact &artifact : artifacts) {
    auto contentsOrError = llvm::MemoryBuffer::getFile(
        applicationArtifactPath(outputDirectory, artifact.path), /*IsText=*/false,
        /*RequiresNullTerminator=*/false);
    if (!contentsOrError)
      return failure();
    StringRef contents = (*contentsOrError)->getBuffer();
    entries.push_back(llvm::json::Object{
        {"path", artifact.path},
        {"role", artifact.role},
        {"byte_size", static_cast<int64_t>(contents.size())},
        {"sha256", sha256(contents)},
    });
  }
  return entries;
}

static FailureOr<llvm::json::Object>
applicationTensorEntry(const GeneratedL2Global &global) {
  std::optional<std::string> dtype = manifestDtype(global.elementType);
  if (!dtype)
    return failure();
  llvm::json::Array shape;
  for (int64_t dimension : global.shape)
    shape.push_back(dimension);
  llvm::json::Object entry{
      {"id", global.globalId},
      {"original_name", global.globalName},
      {"normalized_name", normalizeSnakeCase(global.globalName)},
      {"dtype", *dtype},
      {"shape", std::move(shape)},
      {"byte_size", static_cast<int64_t>(global.tensorBytes)},
  };
  if (global.globalKind == GeneratedGlobalKind::Input) {
    if (global.runtimeInputFile.empty())
      entry["data"] = llvm::json::Object{{"kind", "synthetic"}};
    else
      entry["data"] = llvm::json::Object{
          {"kind", "supplied"},
          {"path", "data/" + global.runtimeInputAsset},
      };
  }
  return entry;
}

static LogicalResult writeApplicationManifest(const MagiaApplicationDescription &application,
                                              StringRef outputDirectory) {
  llvm::json::Array inputs;
  llvm::json::Array outputs;
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Input &&
        global.globalKind != GeneratedGlobalKind::Output)
      continue;
    FailureOr<llvm::json::Object> entry = applicationTensorEntry(global);
    if (failed(entry))
      return failure();
    if (global.globalKind == GeneratedGlobalKind::Input)
      inputs.push_back(std::move(*entry));
    else
      outputs.push_back(std::move(*entry));
  }

  llvm::json::Array activeTiles;
  uint64_t maxTileDataBytes = 0;
  for (const GeneratedTilePlan *tilePlan : getSortedTilePlans(application)) {
    activeTiles.push_back(tilePlan->hartid);
    maxTileDataBytes = std::max(maxTileDataBytes, tilePlan->l1DataBytes);
  }
  llvm::json::Array tasks;
  SmallVector<GeneratedSpatzTask> requiredTasks =
      getRequiredSpatzTasks(application);
  for (const GeneratedSpatzTask &task : requiredTasks)
    tasks.push_back(task.name);

  std::string name = application.outputStem;
  SmallVector<GeneratedArtifact> generatedArtifacts{
      {"CMakeLists.txt", "build_definition"},
      {"README.md", "integration_guide"},
      {"include/" + name + ".h", "application_interface"},
      {"src/" + name + ".c", "application_data"},
      {"src/" + name + "_runner.c", "application_runner"},
      {"src/" + name + "_initializers.S.in", "initializer_assembly"},
      {"data/" + name + ".initializers.bin", "initializer_data"},
  };
  for (const GeneratedTilePlan *tilePlan : getSortedTilePlans(application))
    generatedArtifacts.push_back(
        {"src/tiles/" +
             getApplicationTileFilename(application, tilePlan->hartid),
         "tile_plan"});
  for (const GeneratedL2Global &global : application.l2Globals)
    if (!global.runtimeInputFile.empty())
      generatedArtifacts.push_back(
          {"data/" + global.runtimeInputAsset, "runtime_input_data"});
  FailureOr<llvm::json::Array> generatedFiles =
      buildArtifactEntries(generatedArtifacts, outputDirectory);
  if (failed(generatedFiles))
    return failure();
  llvm::json::Array userOwnedFiles{llvm::json::Object{
      {"path", "src/application.c"},
      {"role", "application_source"},
  }};
  llvm::json::Array trustRootFiles{llvm::json::Object{
      {"path", "manifest.json"},
      {"role", "application_manifest"},
  }};

  llvm::json::Object abi{
      {"operation", kGeneratedOperationAbiVersion},
      {"descriptor", kGeneratedDescriptorAbiVersion},
  };
  llvm::json::Object memory{
      {"initializers_bytes", static_cast<int64_t>(application.weightsSize)},
      {"required_l2_bytes",
       static_cast<int64_t>(application.nextL2Address - application.l2Base)},
      {"max_tile_l1_bytes", static_cast<int64_t>(maxTileDataBytes)},
  };
  if (application.target == "magia-v3") {
    abi["kernel"] = kGeneratedKernelAbiVersion;
    abi["task_bundle"] = kGeneratedTaskBundleVersion;
    memory["initializers_region"] = "l2_bulk";
    memory["runtime_region"] = "l2_arena";
    uint64_t taskScratchBytes =
        requiredTasks.empty() ? 0ull : kGeneratedTaskScratchBytes;
    memory["task_scratch_bytes"] = static_cast<int64_t>(taskScratchBytes);
    memory["max_tile_l1_bytes"] = static_cast<int64_t>(
        maxTileDataBytes + taskScratchBytes);
  }
  llvm::json::Object manifest{
      {"schema_version", 1},
      {"application",
       llvm::json::Object{{"name", name}, {"target", application.target}}},
      {"source_model", application.sourceModelName},
      {"planned_mesh", llvm::json::Object{{"width", application.meshWidth},
                                          {"height", application.meshHeight}}},
      {"abi", std::move(abi)},
      {"execution",
       llvm::json::Object{{"tokens", application.numTokens},
                          {"token_slots", application.numTokenSlots}}},
      {"active_physical_tiles", std::move(activeTiles)},
      {"tasks", std::move(tasks)},
      {"tensors", llvm::json::Object{{"inputs", std::move(inputs)},
                                     {"outputs", std::move(outputs)}}},
      {"entry_points",
       llvm::json::Object{
           {"run", name + "_run"},
           {"handle_input", name + "_handle_input"},
           {"handle_output", name + "_handle_output"},
       }},
      {"memory", std::move(memory)},
      {"files", llvm::json::Object{{"generated", std::move(*generatedFiles)},
                                   {"trust_root", std::move(trustRootFiles)},
                                   {"user_owned", std::move(userOwnedFiles)}}},
  };
  std::string manifestText =
      llvm::formatv("{0:2}\n", llvm::json::Value(std::move(manifest))).str();
  return writeTextArtifact(
      applicationArtifactPath(outputDirectory, "manifest.json"),
      [&](llvm::raw_ostream &os) { os << manifestText; });
}

static LogicalResult configureRuntimeInputs(MagiaApplicationDescription &application) {
  llvm::StringMap<GeneratedL2Global *> inputs;
  llvm::StringSet<> normalizedNames;
  for (GeneratedL2Global &global : application.l2Globals) {
    if (global.globalKind != GeneratedGlobalKind::Input)
      continue;
    if (!normalizedNames.insert(normalizeSnakeCase(global.globalName)).second) {
      llvm::errs() << "Runtime Input names do not have unique normalized forms\n";
      return failure();
    }
    inputs[global.globalName] = &global;
  }
  for (StringRef assignment : magiaRuntimeInputs) {
    auto [name, path] = assignment.split('=');
    auto found = inputs.find(name);
    if (path.empty() || found == inputs.end() ||
        !found->second->runtimeInputFile.empty()) {
      llvm::errs() << "invalid Runtime Input assignment '" << assignment
                   << "'\n";
      return failure();
    }
    auto contentsOrError = llvm::MemoryBuffer::getFile(
        path, /*IsText=*/false, /*RequiresNullTerminator=*/false);
    if (!contentsOrError ||
        (*contentsOrError)->getBufferSize() !=
            found->second->tensorBytes * application.numTokens) {
      llvm::errs() << "invalid Runtime Input data for '" << name << "'\n";
      return failure();
    }
    found->second->runtimeInputFile = path.str();
    found->second->runtimeInputAsset =
        application.outputStem + ".input_" + normalizeSnakeCase(name) + ".bin";
  }
  return success();
}

LogicalResult generateMagiaApplication(ModuleOp module,
                                       StringRef outputDirectory,
                                       StringRef target) {
  if (llvm::sys::fs::exists(outputDirectory)) {
    llvm::errs() << "application output already exists: " << outputDirectory
                 << "\n";
    return failure();
  }
  MagiaApplicationDescription application;
  configureGeneratedRuntime(application, target);
  application.outputStem = normalizeSnakeCase(application.outputStem);
  if (application.l2End <= application.l2Base) {
    llvm::errs() << "invalid generated application L2 range\n";
    return failure();
  }
  if (failed(collectMagiaApplicationDescription(module, application))) {
    llvm::errs() << "failed to collect generated application descriptors\n";
    return failure();
  }
  if (application.sourceModelName.empty() || application.meshWidth == 0 ||
      application.meshHeight == 0 || !application.hasPackedWeights ||
      application.numTokens == 0) {
    llvm::errs() << "incomplete generated application metadata\n";
    return failure();
  }
  if (failed(validateSpatzTaskCoverage(application)))
    return failure();
  if (failed(configureRuntimeInputs(application)))
    return failure();
  planGatheredFifoLayouts(application);

  StringRef sourceInitializers = magiaWeightsFile.empty()
                                     ? StringRef(application.weightsFile)
                                     : StringRef(magiaWeightsFile);
  auto initializersOrError = llvm::MemoryBuffer::getFile(
      sourceInitializers, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!initializersOrError) {
    llvm::errs() << "failed to read packed Initializers '" << sourceInitializers
                 << "': " << initializersOrError.getError().message() << "\n";
    return failure();
  }
  StringRef initializers = (*initializersOrError)->getBuffer();
  if (initializers.size() != application.weightsSize) {
    llvm::errs() << "packed Initializers size mismatch: expected "
                 << application.weightsSize << " bytes, found " << initializers.size()
                 << "\n";
    return failure();
  }
  if (sha256(initializers) != application.weightsSha256) {
    llvm::errs() << "packed Initializers checksum mismatch\n";
    return failure();
  }

  for (StringRef directory : {StringRef(), StringRef("include"),
                              StringRef("src"), StringRef("src/tiles"),
                              StringRef("data")}) {
    std::string path = directory.empty()
                           ? outputDirectory.str()
                           : applicationArtifactPath(outputDirectory, directory);
    std::error_code error = llvm::sys::fs::create_directories(path);
    if (error) {
      llvm::errs() << "failed to create application directory '" << path
                   << "': " << error.message() << "\n";
      return failure();
    }
  }

  std::string name = application.outputStem;
  auto writeApplicationText =
      [&](StringRef path, llvm::function_ref<void(llvm::raw_ostream &)> emit) {
        return writeTextArtifact(applicationArtifactPath(outputDirectory, path),
                                 emit);
      };
  if (failed(writeApplicationText(
          "CMakeLists.txt",
          [&](llvm::raw_ostream &os) { emitApplicationCMake(application, os); })) ||
      failed(writeApplicationText(
          "README.md",
          [&](llvm::raw_ostream &os) { emitApplicationReadme(application, os); })) ||
      failed(writeApplicationText(
          "include/" + name + ".h",
          [&](llvm::raw_ostream &os) { emitApplicationHeader(application, os); })) ||
      failed(writeApplicationText("src/" + name + ".c",
                                  [&](llvm::raw_ostream &os) {
                                    emitApplicationDataSource(application, os);
                                  })) ||
      failed(writeApplicationText(
          "src/" + name + "_runner.c",
          [&](llvm::raw_ostream &os) { emitApplicationRunner(application, os); })) ||
      failed(writeApplicationText("src/application.c",
                                  [&](llvm::raw_ostream &os) {
                                    emitInitialApplicationSource(application, os);
                                  })) ||
      failed(writeApplicationText(
          "src/" + name + "_initializers.S.in", [&](llvm::raw_ostream &os) {
            emitApplicationInitializersAssembly(application, os);
          })))
    return failure();
  for (const GeneratedTilePlan *tilePlan : getSortedTilePlans(application)) {
    std::string path =
        "src/tiles/" + getApplicationTileFilename(application, tilePlan->hartid);
    if (failed(writeApplicationText(path, [&](llvm::raw_ostream &os) {
          emitApplicationTileSource(application, *tilePlan, os);
        })))
      return failure();
  }
  if (failed(writeBinaryArtifact(
          applicationArtifactPath(outputDirectory,
                              "data/" + name + ".initializers.bin"),
          initializers)))
    return failure();
  for (const GeneratedL2Global &global : application.l2Globals) {
    if (global.runtimeInputFile.empty())
      continue;
    auto inputOrError = llvm::MemoryBuffer::getFile(
        global.runtimeInputFile, /*IsText=*/false,
        /*RequiresNullTerminator=*/false);
    if (!inputOrError ||
        failed(writeBinaryArtifact(
            applicationArtifactPath(outputDirectory,
                                "data/" + global.runtimeInputAsset),
            (*inputOrError)->getBuffer())))
      return failure();
  }
  if (failed(writeApplicationManifest(application, outputDirectory)))
    return failure();
  return success();
}

} // namespace mlir::maps
