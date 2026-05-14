#include "maps/Dialect/Maps/IR/Maps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::maps {


static Type parseElementType(const llvm::json::Object &tensor,
                             OpBuilder &builder) {
    std::optional<StringRef> dtype = tensor.getString("dtype");

    if (!dtype)
        return builder.getF32Type();

    if (*dtype == "f32" || *dtype == "float32")
        return builder.getF32Type();

    if (*dtype == "f64" || *dtype == "float64")
        return builder.getF64Type();

    if (*dtype == "f16" || *dtype == "float16")
        return builder.getF16Type();

    if (*dtype == "bf16")
        return builder.getBF16Type();

    if (*dtype == "i32")
        return builder.getI32Type();

    if (*dtype == "i64")
        return builder.getI64Type();

    return {};
}


static RankedTensorType parseTensorType(const llvm::json::Object &tensor,
                                        OpBuilder &builder) {
    auto *dimsJson = tensor.getArray("dims");
    if (!dimsJson)
        return {};

    SmallVector<int64_t> dims;

    for (const llvm::json::Value &dimJson : *dimsJson) {
        std::optional<int64_t> dim = dimJson.getAsInteger();
        if (!dim)
        return {};

        dims.push_back(*dim);
    }

    Type elementType = parseElementType(tensor, builder);
    if (!elementType)
        return {};

    return RankedTensorType::get(dims, elementType);
}


static std::optional<llvm::SetVector<int64_t>> collectInputTensorIds(const llvm::json::Array &stages){

    llvm::SetVector<int64_t> ids;

    for (const llvm::json::Value &stageValue : stages) {
        auto *stage = stageValue.getAsObject();
        auto *layers = stage->getArray("layers");

        for (const llvm::json::Value &layerValue : *layers) {
            auto *layer = layerValue.getAsObject();
            auto *inputs = layer->getArray("inputs");

            for (const llvm::json::Value &inputValue : *inputs){
            auto *input = inputValue.getAsObject();

            std::optional<int64_t> tensorId = input->getInteger("tensor_id");
            auto *source = input->getObject("source");

            if (source->getInteger("base_addr")) // gonna change this in maps later
            ids.insert(*tensorId);
            }
        }
    }

    return ids;
}

static OwningOpRef<Operation *> importJsonToMaps(StringRef input, 
    MLIRContext *ctx) {

    ctx->loadDialect<maps::MapsDialect, 
        func::FuncDialect,
        tensor::TensorDialect,
        linalg::LinalgDialect,
        math::MathDialect,
        arith::ArithDialect>();

    auto json = llvm::json::parse(input);
    if (!json) {
        llvm::errs() << llvm::toString(json.takeError()) << "\n";
        return {};
    }

    auto *root = json->getAsObject();
    auto *tensors = root->getArray("tensors");
    auto *stages = root->getArray("stages");

    std::optional<StringRef> netName = root->getString("name");

    auto inputTensorIds = *collectInputTensorIds(*stages);

    OpBuilder builder(ctx);
    Location loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    module->setAttr("maps.name", builder.getStringAttr(*netName));

    SmallVector<Type> inputTypes;

    for (int64_t tensorId : inputTensorIds) {
        auto *tensor = (*tensors)[tensorId].getAsObject();
        RankedTensorType type = parseTensorType(*tensor, builder);
        inputTypes.push_back(type);
  }

  builder.setInsertionPointToStart(module.getBody());

  FunctionType funcType = builder.getFunctionType(inputTypes, {});
  func::FuncOp mainFunc = func::FuncOp::create(builder, loc, "main", funcType);

  Block *entry = mainFunc.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  func::ReturnOp::create(builder, loc);

  return OwningOpRef<Operation *>(module.getOperation());
}

void registerJsonToMapsTranslation() {
    TranslateToMLIRRegistration reg(
        "json-to-maps",
        "import MAPS JSON as MAPS Dialect",
        importJsonToMaps,
        [](DialectRegistry &registry) {
            registry.insert<maps::MapsDialect,
                            func::FuncDialect,
                            tensor::TensorDialect,
                            linalg::LinalgDialect,
                            math::MathDialect,
                            arith::ArithDialect>();
    });
}
} // namespace mlir::maps