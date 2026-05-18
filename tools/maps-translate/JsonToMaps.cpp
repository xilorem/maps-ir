// Translation registration function
#include "mlir/Tools/mlir-translate/Translation.h"

// Dialects
#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// IR constructs
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"

// JSON parsing
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

// Abstract data types
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseSet.h"


using namespace mlir;

namespace mlir::maps {

struct SliceDim {
    int64_t start;
    int64_t length;
};

struct Fragment {
    std::string name;
    int64_t tensorId;
    int64_t srcHartId;
    int64_t dstHartId;
    SmallVector<SliceDim> srcDims;
    SmallVector<SliceDim> dstDims;
};

struct Tile{
    int64_t hartId;
    int64_t x;
    int64_t y;
};

struct Stage {
    int64_t stageId;
    std::string symName;
};



static Type parseElementType(const llvm::json::Object &tensor,
                             OpBuilder &builder) {
    std::optional<StringRef> dtype = tensor.getString("dtype");
    
    // fallback to f32 if dtype is not specified, TODO: need to add it in the maps tool
    if (!dtype)
        return builder.getF32Type();

    if (*dtype == "f32" || *dtype == "float32" || *dtype == "fp32")
        return builder.getF32Type();

    if (*dtype == "f64" || *dtype == "float64" || *dtype == "fp64")
        return builder.getF64Type();

    if (*dtype == "f16" || *dtype == "float16" || *dtype == "fp16")
        return builder.getF16Type();

    if (*dtype == "bf16" || *dtype == "brainfloat16")
        return builder.getBF16Type();

    if (*dtype == "i32" || *dtype == "int32" || *dtype == "integer32")
        return builder.getI32Type();

    if (*dtype == "i64" || *dtype == "int64" || *dtype == "integer64")
        return builder.getI64Type();

    return {};
}


static RankedTensorType parseTensorType(const llvm::json::Object &tensor,
                                        OpBuilder &builder) {

    // e.g. %A: tensor<64x128xf16>

    // get tensor dimensions
    auto *dimsJson = tensor.getArray("dims");
    SmallVector<int64_t> dims;

    for (const llvm::json::Value &dimJson : *dimsJson) {
        std::optional<int64_t> dim = dimJson.getAsInteger();
        if (!dim)
            return {};

        dims.push_back(*dim);
    }

    // get tensor element type
    Type elementType = parseElementType(tensor, builder);
    if (!elementType)
        return {};

    return RankedTensorType::get(dims, elementType);
}


static std::optional<llvm::SetVector<int64_t>> collectExtOutputTensorIds(const llvm::json::Array &stages){
    
    // to get the output tensors, we check if a tensor is produced but never consumed
    llvm::SetVector<int64_t> produced;
    llvm::SetVector<int64_t> consumed;
    llvm::SetVector<int64_t> ids;

    for (const llvm::json::Value &stageValue : stages) {
        auto *stage = stageValue.getAsObject();
        auto *layers = stage->getArray("layers");

        for (const llvm::json::Value &layerValue : *layers) {
            auto *layer = layerValue.getAsObject();
            auto *inputs = layer->getArray("inputs");
            auto *outputs = layer->getArray("outputs");

            for (const llvm::json::Value &inputValue : *inputs){
                auto *input = inputValue.getAsObject();
                std::optional<int64_t> tensorId = input->getInteger("tensor_id");
                if (tensorId)
                    consumed.insert(*tensorId);
            }

            for (const llvm::json::Value &outputValue : *outputs){
                auto *output = outputValue.getAsObject();
                std::optional<int64_t> tensorId = output->getInteger("tensor_id");
                if (tensorId)
                    produced.insert(*tensorId);
            }
        }
    }

    for (int64_t tensorId : produced) {
        if (!consumed.contains(tensorId))
            ids.insert(tensorId);
    }

    return ids;
}


static std::optional<SmallVector<SliceDim>> parseSliceDims(const llvm::json::Object &slice) {
    SmallVector<SliceDim> result;
    // auto rank = slice.getInteger("rank");
    auto *dims = slice.getArray("dims");

    for (const llvm::json::Value &dimValue : *dims) {
        auto *dim = dimValue.getAsObject();
        auto start = dim->getInteger("start");
        auto length = dim->getInteger("length");

        result.push_back({*start, *length});
    }
    return result;
}


static std::optional<SmallVector<Fragment>> getInitFragments(const llvm::json::Array &initializations) {
    SmallVector<Fragment> result;
    
    for (const llvm::json::Value &initValue : initializations) {
        auto *init = initValue.getAsObject();

        auto tensorId = init->getInteger("tensor_id");
        auto *fragments = init->getArray("fragments");
        auto name = init->getString("name");

        for (const llvm::json::Value &fragmentValue : *fragments) {
            auto *fragment = fragmentValue.getAsObject();
            if (!fragment)
                return std::nullopt;
            
            
            auto srcHartId = fragment->getInteger("src_hartid");
            auto dstHartId = fragment->getInteger("dst_hartid");
            auto *srcSlice = fragment->getObject("src_slice");
            auto *dstSlice = fragment->getObject("dst_slice");
            auto fragName = (name->str() + "_from_L2_to_tile" + std::to_string(*dstHartId));

            auto srcDims = parseSliceDims(*srcSlice);
            auto dstDims = parseSliceDims(*dstSlice);

            result.push_back(Fragment{
                fragName,
                *tensorId,
                *srcHartId,
                *dstHartId,
                *srcDims,
                *dstDims,
            });
        }
    }
    return result;
}


static std::optional<SmallVector<Tile>> getTileObjs(const llvm::json::Array &tiles) {
    SmallVector<Tile> result;
    
    for (const llvm::json::Value &tileValue : tiles) {
        auto *tile = tileValue.getAsObject();

        auto tileId = tile->getInteger("tile_id");
        auto x = tile->getInteger("x");
        auto y = tile->getInteger("y");
        result.push_back(Tile{
            *tileId,
            *x,
            *y
        });
    }
    return result;
}


static std::optional<SmallVector<Stage>>getStageObjs(const llvm::json::Array &stages) {
    SmallVector<Stage> result;

    // TODO: need to add stage index 
    for (auto indexed : llvm::enumerate(stages)) {
        auto *stage = indexed.value().getAsObject();
        if (!stage)
            return std::nullopt;

        std::string symName = ("stage" + std::to_string(indexed.index()));

        result.push_back(Stage{
            static_cast<int64_t>(indexed.index()),
            symName,
        });
    }

    return result;
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

    // create main function
    builder.setInsertionPointToStart(module.getBody());

    FunctionType funcType = builder.getFunctionType(inputTypes, {});
    func::FuncOp mainFunc = func::FuncOp::create(builder, loc, "main", funcType);

    Block *mainEntry = mainFunc.addEntryBlock();

    // start of main block
    builder.setInsertionPointToStart(mainEntry);

    // create empty tensors for output tensors
    auto extOutputTensorIds = *collectExtOutputTensorIds(*stages);
    llvm::DenseMap<int64_t, Value> OutputTensorInitValues;

    for (int64_t tensorId : extOutputTensorIds) {
        auto *tensor = (*tensors)[tensorId].getAsObject();
        RankedTensorType type = parseTensorType(*tensor, builder);
        auto emptyTensor = tensor::EmptyOp::create(builder, loc,type.getShape(), type.getElementType());
        OutputTensorInitValues[tensorId] = emptyTensor.getResult();
    }


    // start of init block for data movement of initializers inside tiles
    auto init = InitOp::create(builder, loc);
    init.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&init.getBody().front());

    // obtain initializer fragments from json
    auto *initializations = root->getArray("initializations");
    auto initFragments = getInitFragments(*initializations);

    // map input tensor ids to main function arguments
    llvm::DenseMap<int64_t, Value> inputTensorValues;
    for (auto indexed : llvm::enumerate(inputTensorIds)) {
        int64_t argIndex = indexed.index();
        int64_t tensorId = indexed.value();
        inputTensorValues[tensorId] = mainEntry->getArgument(argIndex);
    }

    // extract slices from input tensors for each fragment
    for (const Fragment &fragment : *initFragments) {
        Value source = inputTensorValues.lookup(fragment.tensorId);
        
        SmallVector<int64_t> shape;
        SmallVector<OpFoldResult> offsets;
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides;

        for (const SliceDim &dim : fragment.srcDims) {
            shape.push_back(dim.length);
            offsets.push_back(builder.getIndexAttr(dim.start));
            sizes.push_back(builder.getIndexAttr(dim.length));
            strides.push_back(builder.getIndexAttr(1));
        }


        auto sourceType = cast<RankedTensorType>(source.getType());
        auto sliceType = RankedTensorType::get(shape, sourceType.getElementType());

        // create extract_slice op for the fragment, considering tile usage
        Value slice = tensor::ExtractSliceOp::create(
            builder, 
            loc, 
            sliceType, 
            source, 
            offsets, 
            sizes, 
            strides
        );

        // create send op for initializers fragments
        maps::SendOp::create(
            builder, 
            loc, 
            slice, 
            SymbolRefAttr::get(ctx, fragment.name),
            builder.getI64IntegerAttr(fragment.srcHartId),
            builder.getI64IntegerAttr(fragment.dstHartId)
        );
    }

    // start of tiles receiving init data
    auto *mesh = root->getObject("mesh");
    auto *tiles = mesh->getArray("tiles");
    auto tileObjs = getTileObjs(*tiles);

    for (const Tile &tileObj : *tileObjs) {

        auto tile = TileOp::create(
            builder,
            loc,
            builder.getStringAttr("tile" + std::to_string(tileObj.hartId)),
            builder.getI64IntegerAttr(tileObj.hartId),
            builder.getDenseI64ArrayAttr({tileObj.x, tileObj.y})
        );

        // start tile block
        tile.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&tile.getBody().front());

        // check which tiles receive which fragments
        for (const Fragment &fragment : *initFragments) {
            if (fragment.dstHartId != tileObj.hartId)
                continue; // tile doesn't receive this fragment

            auto *tensor = (*tensors)[fragment.tensorId].getAsObject();
            SmallVector<int64_t> shape;
            for (const SliceDim &dim : fragment.dstDims)
                shape.push_back(dim.length);

            Type elementType = parseElementType(*tensor, builder);
            Type resultType = RankedTensorType::get(shape, elementType);

            // create recv op for the fragment
            maps::RecvOp::create(
                builder,
                loc,
                resultType,
                StringRef(fragment.name),
                fragment.srcHartId,
                fragment.dstHartId
            );
        }

        builder.setInsertionPointAfter(tile);
        // end of tile block
    }
    // end of tiles receiving init data

    // end of init region
    builder.setInsertionPointAfter(init);


    // start of run region
    auto run = RunOp::create(builder, loc);
    run.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&run.getBody().front());


    // start of stages regions
    auto stageObjs = getStageObjs(*stages);

    for (const Stage &stageObj : *stageObjs) {

        auto stage = StageOp::create(
        builder,
        loc,
        builder.getI64IntegerAttr(stageObj.stageId),
        builder.getStringAttr(stageObj.symName)
    );
        stage.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&stage.getBody().front());
        

        for (const Tile &tileObj : *tileObjs) {

            auto tile = TileOp::create(
                builder,
                loc,
                builder.getStringAttr("tile" + std::to_string(tileObj.hartId)),
                builder.getI64IntegerAttr(tileObj.hartId),
                builder.getDenseI64ArrayAttr({tileObj.x, tileObj.y})
            );

            // start tile block
            tile.getBody().emplaceBlock();
            builder.setInsertionPointToStart(&tile.getBody().front());

           

            builder.setInsertionPointAfter(tile);
            // end of tile block
        }

        builder.setInsertionPointAfter(stage);
    }

    builder.setInsertionPointAfter(run);
    // end of run region

    // place main's return op at the end of the block
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
