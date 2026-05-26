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
    std::string name;      // channel name
    std::string slotName;  // local tile storage name
    int64_t tensorId;
    int64_t srcHartId;
    int64_t dstHartId;
    SmallVector<SliceDim> srcDims;
    SmallVector<SliceDim> dstDims;
};

struct TransitionFragment {
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
    int64_t x0;
    int64_t y0;
    int64_t width;
    int64_t height;
};


static bool isInitializerTensor(const llvm::json::Object &tensor) {
    return tensor.getBoolean("is_initializer").value_or(false);
}


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

static Attribute parseJsonAttribute(const llvm::json::Value &value,
                                    OpBuilder &builder) {
    if (auto string = value.getAsString())
        return builder.getStringAttr(*string);
    if (auto integer = value.getAsInteger())
        return builder.getI64IntegerAttr(*integer);
    if (auto number = value.getAsNumber())
        return builder.getF64FloatAttr(*number);
    if (auto boolean = value.getAsBoolean())
        return builder.getBoolAttr(*boolean);
    if (auto *array = value.getAsArray()) {
        SmallVector<Attribute> elements;
        for (const llvm::json::Value &element : *array) {
            Attribute attr = parseJsonAttribute(element, builder);
            if (!attr)
                return {};
            elements.push_back(attr);
        }
        return builder.getArrayAttr(elements);
    }
    if (auto *object = value.getAsObject()) {
        SmallVector<NamedAttribute> attributes;
        for (const auto &entry : *object) {
            Attribute attr = parseJsonAttribute(entry.second, builder);
            if (!attr)
                return {};
            attributes.push_back(builder.getNamedAttr(entry.first, attr));
        }
        return builder.getDictionaryAttr(attributes);
    }
    return {};
}

static DictionaryAttr parseJsonObject(const llvm::json::Object &object,
                                      OpBuilder &builder) {
    SmallVector<NamedAttribute> attributes;
    for (const auto &entry : object) {
        Attribute attr = parseJsonAttribute(entry.second, builder);
        if (!attr)
            return {};
        attributes.push_back(builder.getNamedAttr(entry.first, attr));
    }
    return builder.getDictionaryAttr(attributes);
}


static bool isTileInStage(const Tile &tile, const Stage &stage) {
    return tile.x >= stage.x0 &&
           tile.x < stage.x0 + stage.width &&
           tile.y >= stage.y0 &&
           tile.y < stage.y0 + stage.height;
}


static std::optional<int64_t> getTensorId(const llvm::json::Value &value) {
    auto *object = value.getAsObject();
    if (!object)
        return std::nullopt;
    return object->getInteger("tensor_id");
}

static SliceDim partitionDim(int64_t size, int64_t parts, int64_t index) {
    int64_t base = size / parts;
    int64_t remainder = size % parts;
    return {index * base + std::min(index, remainder),
            base + (index < remainder ? 1 : 0)};
}


static SmallVector<int64_t> getLocalStartsForSourceTile(ArrayRef<TransitionFragment> fragments,
                                                        const TransitionFragment &selected) {
    SmallVector<int64_t> starts;
    starts.reserve(selected.srcDims.size());
    for (const SliceDim &dim : selected.srcDims)
        starts.push_back(dim.start);

    for (const TransitionFragment &fragment : fragments) {
        if (fragment.tensorId != selected.tensorId ||
            fragment.srcHartId != selected.srcHartId ||
            fragment.srcDims.size() != starts.size())
            continue;

        for (auto indexed : llvm::enumerate(fragment.srcDims))
            starts[indexed.index()] = std::min(starts[indexed.index()], indexed.value().start);
    }

    return starts;
}


static unsigned getConcatDim(ArrayRef<TransitionFragment> fragments) {
    if (fragments.empty())
        return 0;

    for (auto indexed : llvm::enumerate(fragments.front().dstDims)) {
        int64_t start = indexed.value().start;
        bool differs = llvm::any_of(fragments, [&](const TransitionFragment &fragment) {
            return fragment.dstDims[indexed.index()].start != start;
        });
        if (differs)
            return indexed.index();
    }

    return 0;
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

static std::optional<RankedTensorType>
parseTileOutputType(const llvm::json::Object &tensor,
                    const llvm::json::Object &output, const Tile &tile,
                    Type elementType) {
    auto *dimsJson = tensor.getArray("dims");
    auto *layout = output.getObject("layout");
    auto *submesh = layout ? layout->getObject("submesh") : nullptr;
    auto *meshX = layout ? layout->getObject("mesh_x") : nullptr;
    auto *meshY = layout ? layout->getObject("mesh_y") : nullptr;
    if (!dimsJson || !layout || !submesh || !meshX || !meshY)
        return std::nullopt;

    auto x0 = submesh->getInteger("x0");
    auto y0 = submesh->getInteger("y0");
    auto width = submesh->getInteger("width");
    auto logicalWidth = layout->getInteger("logical_width");
    auto logicalHeight = layout->getInteger("logical_height");
    if (!x0 || !y0 || !width || !logicalWidth || !logicalHeight ||
        *logicalWidth <= 0 || *logicalHeight <= 0)
        return std::nullopt;

    int64_t ordinal = (tile.y - *y0) * *width + tile.x - *x0;
    int64_t logicalX = ordinal % *logicalWidth;
    int64_t logicalY = ordinal / *logicalWidth;
    if (ordinal < 0 || logicalY >= *logicalHeight)
        return std::nullopt;

    SmallVector<int64_t> dims;
    for (const llvm::json::Value &dimValue : *dimsJson) {
        auto dim = dimValue.getAsInteger();
        if (!dim)
            return std::nullopt;
        dims.push_back(*dim);
    }

    auto shardDimension = [&](const llvm::json::Object &mapping, int64_t parts,
                              int64_t index) -> bool {
        auto mode = mapping.getInteger("mode");
        if (!mode)
            return false;
        if (*mode != 1)
            return true;
        auto axis = mapping.getInteger("tensor_axis");
        if (!axis || *axis < 0 || static_cast<size_t>(*axis) >= dims.size())
            return false;
        dims[*axis] = partitionDim(dims[*axis], parts, index).length;
        return true;
    };

    if (!shardDimension(*meshX, *logicalWidth, logicalX) ||
        !shardDimension(*meshY, *logicalHeight, logicalY))
        return std::nullopt;
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
            
            auto channelName = name->str() + "_from_L2_to_tile" + std::to_string(*dstHartId);

            std::string slotBase = name->str();
            if (llvm::StringRef(slotBase).starts_with("init_"))
                slotBase = slotBase.substr(5);

            auto slotName = slotBase + "_tile" + std::to_string(*dstHartId);
            auto srcDims = parseSliceDims(*srcSlice);
            auto dstDims = parseSliceDims(*dstSlice);

            result.push_back(Fragment{
                channelName,
                slotName,
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


static std::optional<SmallVector<TransitionFragment>> getTransitionFragments(const llvm::json::Array &transitions) {
    SmallVector<TransitionFragment> result;
    
    for (const llvm::json::Value &transitionValue : transitions) {
        auto *transition = transitionValue.getAsObject();
        if (!transition)
            return std::nullopt;

        auto tensorId = transition->getInteger("tensor_id");
        auto *fragments = transition->getArray("fragments");
        auto name = transition->getString("name");

        for (const llvm::json::Value &fragmentValue : *fragments) {
            auto *fragment = fragmentValue.getAsObject();
            if (!fragment)
                return std::nullopt;
            
            auto srcHartId = fragment->getInteger("src_hartid");
            auto dstHartId = fragment->getInteger("dst_hartid");
            auto *srcSlice = fragment->getObject("src_slice");
            auto *dstSlice = fragment->getObject("dst_slice");
            auto fragName = (name->str() + "_from_tile" + std::to_string(*srcHartId) +
                             "_to_tile" + std::to_string(*dstHartId));

            auto srcDims = parseSliceDims(*srcSlice);
            auto dstDims = parseSliceDims(*dstSlice);

            result.push_back(TransitionFragment{
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

static std::optional<SmallVector<TransitionFragment>> getFinalizationFragments(
    const llvm::json::Array &finalizations) {
    SmallVector<TransitionFragment> result;

    for (const llvm::json::Value &finalizationValue : finalizations) {
        auto *finalization = finalizationValue.getAsObject();
        if (!finalization)
            return std::nullopt;

        auto tensorId = finalization->getInteger("tensor_id");
        auto *fragments = finalization->getArray("fragments");
        auto name = finalization->getString("name");

        for (const llvm::json::Value &fragmentValue : *fragments) {
            auto *fragment = fragmentValue.getAsObject();
            if (!fragment)
                return std::nullopt;

            auto srcHartId = fragment->getInteger("src_hartid");
            auto dstHartId = fragment->getInteger("dst_hartid");
            auto *srcSlice = fragment->getObject("src_slice");
            auto *dstSlice = fragment->getObject("dst_slice");
            auto fragName = (name->str() + "_from_tile" + std::to_string(*srcHartId) +
                             "_to_L2");

            auto srcDims = parseSliceDims(*srcSlice);
            auto dstDims = parseSliceDims(*dstSlice);

            result.push_back(TransitionFragment{
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
        auto *submesh = stage->getObject("submesh");
        auto x0 = submesh->getInteger("x0");
        auto y0 = submesh->getInteger("y0");
        auto width = submesh->getInteger("width");
        auto height = submesh->getInteger("height");

        result.push_back(Stage{
            static_cast<int64_t>(indexed.index()),
            symName,
            *x0,
            *y0,
            *width,
            *height,
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
    auto *transitions = root->getArray("transitions");
    auto transitionFragments = getTransitionFragments(*transitions);
    auto *finalizations = root->getArray("finalizations");
    auto outputFragments = getFinalizationFragments(*finalizations);

    // map input tensor ids to main function arguments
    llvm::DenseMap<int64_t, Value> inputTensorValues;
    for (auto indexed : llvm::enumerate(inputTensorIds)) {
        int64_t argIndex = indexed.index();
        int64_t tensorId = indexed.value();
        inputTensorValues[tensorId] = mainEntry->getArgument(argIndex);
    }

    // extract slices from input tensors for each fragment
    for (const Fragment &fragment : *initFragments) {
        auto *tensor = (*tensors)[fragment.tensorId].getAsObject();
        if (!isInitializerTensor(*tensor))
            continue;
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
    auto stageObjs = getStageObjs(*stages);
    for (const Tile &tileObj : *tileObjs) {
        bool receivesInitializer = llvm::any_of(*initFragments, [&](const Fragment &fragment) {
            auto *tensor = (*tensors)[fragment.tensorId].getAsObject();
            return fragment.dstHartId == tileObj.hartId &&
                   isInitializerTensor(*tensor);
        });
        if (!receivesInitializer)
            continue;

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
            if (!isInitializerTensor(*tensor))
                continue;

            SmallVector<int64_t> shape;
            for (const SliceDim &dim : fragment.srcDims)
                shape.push_back(dim.length);

            Type elementType = parseElementType(*tensor, builder);
            Type resultType = RankedTensorType::get(shape, elementType);

            // create recv op for the fragment
            auto recv = maps::RecvOp::create(
                builder,
                loc,
                resultType,
                StringRef(fragment.name),
                fragment.srcHartId,
                fragment.dstHartId
            );

            // store result
            maps::StoreOp::create(
                builder,
                loc,
                recv.getResult(),
                SymbolRefAttr::get(ctx, fragment.slotName)
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
    for (const Stage &stageObj : *stageObjs) {

        auto stage = StageOp::create(
            builder,
            loc,
            builder.getI64IntegerAttr(stageObj.stageId),
            builder.getStringAttr(stageObj.symName)
        );
        stage.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&stage.getBody().front());

        // slice and send runtime inputs consumed by this stage before tile bodies
        for (const Fragment &fragment : *initFragments) {
            auto *tensor = (*tensors)[fragment.tensorId].getAsObject();
            if (isInitializerTensor(*tensor))
                continue;

            bool isFragmentInStage = false;
            for (const Tile &tileObj : *tileObjs) {
                if (fragment.dstHartId == tileObj.hartId &&
                    isTileInStage(tileObj, stageObj)) {
                    isFragmentInStage = true;
                    break;
                }
            }
            if (!isFragmentInStage)
                continue;

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

            Value slice = tensor::ExtractSliceOp::create(
                builder, loc, sliceType, source, offsets, sizes, strides);

            maps::SendOp::create(
                builder,
                loc,
                slice,
                SymbolRefAttr::get(ctx, fragment.name),
                builder.getI64IntegerAttr(fragment.srcHartId),
                builder.getI64IntegerAttr(fragment.dstHartId)
            );
        }
        

        for (const Tile &tileObj : *tileObjs) {

            // check if tile is in the stage's submesh
            if (!isTileInStage(tileObj, stageObj))
                continue;


            // if in submesh, create tile block
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

            llvm::DenseMap<int64_t, Value> tileValues;
            llvm::DenseMap<int64_t, SmallVector<std::pair<const TransitionFragment *, Value>>> transitionValues;

            // receive transition fragments from producer tiles
            for (const TransitionFragment &fragment : *transitionFragments) {
                if (fragment.dstHartId != tileObj.hartId)
                    continue;

                auto *tensor = (*tensors)[fragment.tensorId].getAsObject();
                SmallVector<int64_t> shape;
                for (const SliceDim &dim : fragment.srcDims)
                    shape.push_back(dim.length);

                Type elementType = parseElementType(*tensor, builder);
                Type resultType = RankedTensorType::get(shape, elementType);

                Value transitionInput = maps::RecvOp::create(
                    builder,
                    loc,
                    resultType,
                    StringRef(fragment.name),
                    fragment.srcHartId,
                    fragment.dstHartId
                ).getResult();
                transitionValues[fragment.tensorId].push_back({&fragment, transitionInput});
            }

            for (auto &entry : transitionValues) {
                SmallVector<std::pair<const TransitionFragment *, Value>> &values = entry.second;
                if (values.size() == 1) {
                    tileValues[entry.first] = values.front().second;
                    continue;
                }

                SmallVector<TransitionFragment> fragments;
                fragments.reserve(values.size());
                for (const auto &value : values) {
                    fragments.push_back(*value.first);
                }

                SmallVector<unsigned> varyingDims;
                for (auto indexed : llvm::enumerate(fragments.front().dstDims)) {
                    int64_t start = indexed.value().start;
                    if (llvm::any_of(fragments, [&](const TransitionFragment &fragment) {
                            return fragment.dstDims[indexed.index()].start != start;
                        }))
                        varyingDims.push_back(indexed.index());
                }

                if (varyingDims.size() > 1) {
                    SmallVector<int64_t> starts;
                    SmallVector<int64_t> ends;
                    for (const SliceDim &dim : fragments.front().dstDims) {
                        starts.push_back(dim.start);
                        ends.push_back(dim.start + dim.length);
                    }
                    for (const TransitionFragment &fragment : fragments) {
                        for (auto indexed : llvm::enumerate(fragment.dstDims)) {
                            starts[indexed.index()] =
                                std::min(starts[indexed.index()], indexed.value().start);
                            ends[indexed.index()] =
                                std::max(ends[indexed.index()],
                                         indexed.value().start + indexed.value().length);
                        }
                    }

                    auto firstType = cast<RankedTensorType>(values.front().second.getType());
                    SmallVector<int64_t> assembledShape;
                    for (auto indexed : llvm::enumerate(starts))
                        assembledShape.push_back(ends[indexed.index()] - indexed.value());
                    Value assembled = tensor::EmptyOp::create(
                        builder, loc, assembledShape, firstType.getElementType()).getResult();

                    for (const auto &value : values) {
                        SmallVector<OpFoldResult> offsets;
                        SmallVector<OpFoldResult> sizes;
                        SmallVector<OpFoldResult> strides;
                        for (auto indexed : llvm::enumerate(value.first->dstDims)) {
                            offsets.push_back(builder.getIndexAttr(
                                indexed.value().start - starts[indexed.index()]));
                            sizes.push_back(builder.getIndexAttr(indexed.value().length));
                            strides.push_back(builder.getIndexAttr(1));
                        }
                        assembled = tensor::InsertSliceOp::create(
                            builder, loc, value.second, assembled, offsets, sizes, strides);
                    }
                    tileValues[entry.first] = assembled;
                    continue;
                }

                unsigned concatDim = getConcatDim(fragments);
                llvm::sort(values, [concatDim](const auto &lhs, const auto &rhs) {
                    return lhs.first->dstDims[concatDim].start < rhs.first->dstDims[concatDim].start;
                });

                SmallVector<Value> inputs;
                inputs.reserve(values.size());
                for (const auto &value : values)
                    inputs.push_back(value.second);

                auto firstType = cast<RankedTensorType>(inputs.front().getType());
                SmallVector<int64_t> concatShape(firstType.getShape());
                concatShape[concatDim] = 0;
                for (Value input : inputs) {
                    auto inputType = cast<RankedTensorType>(input.getType());
                    concatShape[concatDim] += inputType.getShape()[concatDim];
                }

                auto concatType = RankedTensorType::get(concatShape, firstType.getElementType());
                tileValues[entry.first] = tensor::ConcatOp::create(
                    builder,
                    loc,
                    concatType,
                    concatDim,
                    inputs
                ).getResult();
            }

            // load stored values and receive runtime inputs
            for (const Fragment &fragment : *initFragments) {
                if (fragment.dstHartId != tileObj.hartId)
                    continue;

                auto *tensor = (*tensors)[fragment.tensorId].getAsObject();
                if (isInitializerTensor(*tensor)) {
                    // load init fragments
                    SmallVector<int64_t> shape;
                    for (const SliceDim &dim : fragment.srcDims)
                        shape.push_back(dim.length);

                    Type elementType = parseElementType(*tensor, builder);
                    Type resultType = RankedTensorType::get(shape, elementType);

                    Value localValue = maps::LoadOp::create(
                        builder,
                        loc,
                        resultType,
                        SymbolRefAttr::get(ctx, fragment.slotName)
                    ).getResult();
                    tileValues[fragment.tensorId] = localValue;
                } else {
                    // receive inputs
                    SmallVector<int64_t> shape;
                    for (const SliceDim &dim : fragment.srcDims)
                        shape.push_back(dim.length);
            
                    Type elementType = parseElementType(*tensor, builder);
                    Type resultType = RankedTensorType::get(shape, elementType);

                    Value runtimeInput = maps::RecvOp::create(
                        builder,
                        loc,
                        resultType,
                        StringRef(fragment.name),
                        fragment.srcHartId,
                        fragment.dstHartId
                    ).getResult();
                    tileValues[fragment.tensorId] = runtimeInput;
                }
            }
            
            // emit tile core ops
            auto *stageJson = (*stages)[stageObj.stageId].getAsObject();
            auto *layers = stageJson->getArray("layers");

            for (const llvm::json::Value &layerValue : *layers) {
                auto *layer = layerValue.getAsObject();
                auto *node = layer->getObject("node");
                auto *inputs = layer->getArray("inputs");
                auto *outputs = layer->getArray("outputs");
                auto nodeKind = node->getInteger("kind");
                if (!nodeKind || !inputs || !outputs || outputs->empty()) {
                    llvm::errs() << "unsupported MAPS layer: missing node inputs or outputs\n";
                    return {};
                }

                auto outputTensorId = getTensorId((*outputs)[0]);
                if (!outputTensorId) {
                    llvm::errs() << "unsupported MAPS layer: output has no tensor id\n";
                    return {};
                }

                if (*nodeKind == 0) {
                    if (inputs->size() < 2) {
                        llvm::errs() << "unsupported Matmul node: expected two inputs\n";
                        return {};
                    }

                    auto lhsTensorId = getTensorId((*inputs)[0]);
                    auto rhsTensorId = getTensorId((*inputs)[1]);
                    if (!lhsTensorId || !rhsTensorId) {
                        llvm::errs() << "unsupported Matmul node: missing tensor id\n";
                        return {};
                    }

                    Value lhs = tileValues.lookup(*lhsTensorId);
                    Value rhs = tileValues.lookup(*rhsTensorId);
                    if (!lhs || !rhs) {
                        llvm::errs() << "unsupported Matmul node: missing local input value\n";
                        return {};
                    }

                    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
                    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
                    if (!lhsType || !rhsType || lhsType.getRank() != 2 || rhsType.getRank() != 2) {
                        llvm::errs() << "unsupported Matmul node: expected rank-2 tensor inputs\n";
                        return {};
                    }

                    SmallVector<int64_t> outputShape = {
                        lhsType.getShape()[0],
                        rhsType.getShape()[1],
                    };
                    Type elementType = lhsType.getElementType();
                    auto outputType = RankedTensorType::get(outputShape, elementType);
                    Value initTensor = tensor::EmptyOp::create(
                        builder,
                        loc,
                        outputShape,
                        elementType
                    ).getResult();

                    auto matmul = linalg::MatmulOp::create(
                        builder,
                        loc,
                        TypeRange{outputType},
                        ValueRange{lhs, rhs},
                        ValueRange{initTensor},
                        ArrayRef<NamedAttribute>{}
                    );
                    tileValues[*outputTensorId] = matmul->getResult(0);
                    continue;
                }

                if (*nodeKind == 4) {
                    auto *payload = node->getObject("payload");
                    auto opName = payload ? payload->getString("op_name") : std::nullopt;
                    auto *nodeOutputs = node->getArray("outputs");
                    auto *nodeOutput = nodeOutputs && !nodeOutputs->empty()
                                           ? (*nodeOutputs)[0].getAsObject()
                                           : nullptr;
                    auto *output = (*outputs)[0].getAsObject();
                    if (!opName || !nodeOutput || !output || inputs->empty()) {
                        llvm::errs() << "unsupported transform MAPS node: missing metadata\n";
                        return {};
                    }

                    SmallVector<Value> inputValues;
                    for (const llvm::json::Value &input : *inputs) {
                        auto tensorId = getTensorId(input);
                        Value inputValue =
                            tensorId ? tileValues.lookup(*tensorId) : Value{};
                        if (!inputValue) {
                            llvm::errs() << "unsupported transform MAPS node: missing local "
                                            "input value\n";
                            return {};
                        }
                        inputValues.push_back(inputValue);
                    }

                    Type elementType =
                        cast<RankedTensorType>(inputValues.front().getType())
                            .getElementType();
                    auto outputType =
                        parseTileOutputType(*nodeOutput, *output, tileObj, elementType);
                    if (!outputType) {
                        llvm::errs() << "unsupported transform MAPS node: invalid output "
                                        "layout\n";
                        return {};
                    }

                    OperationState state(loc, maps::TransformOp::getOperationName());
                    state.addOperands(inputValues);
                    state.addTypes(*outputType);
                    state.addAttribute("op_name", builder.getStringAttr(*opName));
                    DictionaryAttr payloadAttr = parseJsonObject(*payload, builder);
                    if (!payloadAttr) {
                        llvm::errs() << "unsupported transform MAPS node: invalid payload\n";
                        return {};
                    }
                    state.addAttribute("payload", payloadAttr);
                    if (auto *attributes = node->getObject("attributes")) {
                        DictionaryAttr attributesAttr =
                            parseJsonObject(*attributes, builder);
                        if (!attributesAttr) {
                            llvm::errs()
                                << "unsupported transform MAPS node: invalid attributes\n";
                            return {};
                        }
                        state.addAttribute("node_attributes", attributesAttr);
                    }
                    tileValues[*outputTensorId] =
                        builder.create(state)->getResult(0);
                    continue;
                }

                if (*nodeKind == 3) {
                    llvm::errs() << "unsupported MAPS operation Conv: export an explicit "
                                    "im2col/matmul decomposition before translation\n";
                    return {};
                }

                if (*nodeKind != 1 || inputs->empty()) {
                    llvm::errs() << "unsupported MAPS node kind " << *nodeKind << "\n";
                    return {};
                }

                auto *payload = node->getObject("payload");
                auto opName = payload ? payload->getString("op_name") : std::nullopt;
                if (!opName) {
                    llvm::errs() << "unsupported elementwise MAPS node: missing op_name\n";
                    return {};
                }

                if (*opName == "Exp") {
                    auto inputTensorId = getTensorId((*inputs)[0]);
                    if (!inputTensorId) {
                        llvm::errs() << "unsupported Exp node: missing tensor id\n";
                        return {};
                    }

                    Value inputValue = tileValues.lookup(*inputTensorId);
                    if (!inputValue) {
                        llvm::errs() << "unsupported Exp node: missing local input value\n";
                        return {};
                    }

                    Value exp = math::ExpOp::create(builder, loc, inputValue).getResult();
                    tileValues[*outputTensorId] = exp;
                    continue;
                }

                if (*opName == "Add") {
                    if (inputs->size() < 2) {
                        llvm::errs() << "unsupported Add node: expected two inputs\n";
                        return {};
                    }

                    auto lhsTensorId = getTensorId((*inputs)[0]);
                    auto rhsTensorId = getTensorId((*inputs)[1]);
                    if (!lhsTensorId || !rhsTensorId) {
                        llvm::errs() << "unsupported Add node: missing tensor id\n";
                        return {};
                    }

                    Value lhs = tileValues.lookup(*lhsTensorId);
                    Value rhs = tileValues.lookup(*rhsTensorId);
                    if (!lhs || !rhs) {
                        llvm::errs() << "unsupported Add node: missing local input value\n";
                        return {};
                    }

                    auto *nodeOutputs = node->getArray("outputs");
                    auto *nodeOutput = nodeOutputs && !nodeOutputs->empty()
                                           ? (*nodeOutputs)[0].getAsObject()
                                           : nullptr;
                    auto *output = (*outputs)[0].getAsObject();
                    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
                    if (!nodeOutput || !output || !lhsType) {
                        llvm::errs() << "unsupported Add node: missing output metadata\n";
                        return {};
                    }
                    auto outputType = parseTileOutputType(
                        *nodeOutput, *output, tileObj, lhsType.getElementType());
                    if (!outputType) {
                        llvm::errs() << "unsupported Add node: invalid output layout\n";
                        return {};
                    }

                    auto broadcastToOutput = [&](Value value) -> Value {
                        auto type = dyn_cast<RankedTensorType>(value.getType());
                        if (!type || type == *outputType)
                            return type ? value : Value{};
                        if (type.getRank() >= outputType->getRank())
                            return {};
                        int64_t rankDifference =
                            outputType->getRank() - type.getRank();
                        for (int64_t i = 0; i < type.getRank(); ++i) {
                            if (type.getShape()[i] !=
                                outputType->getShape()[i + rankDifference])
                                return {};
                        }
                        SmallVector<int64_t> broadcastDimensions;
                        for (int64_t i = 0; i < rankDifference; ++i)
                            broadcastDimensions.push_back(i);
                        Value init = tensor::EmptyOp::create(
                            builder, loc, outputType->getShape(),
                            outputType->getElementType()).getResult();
                        return linalg::BroadcastOp::create(
                                   builder, loc, value, init, broadcastDimensions)
                            .getResults()
                            .front();
                    };
                    Type originalLhsType = lhs.getType();
                    Type originalRhsType = rhs.getType();
                    lhs = broadcastToOutput(lhs);
                    rhs = broadcastToOutput(rhs);
                    if (!lhs || !rhs) {
                        llvm::errs()
                            << "unsupported Add node "
                            << node->getString("name").value_or("<unknown>")
                            << ": operands are not broadcastable to output layout "
                            << *outputType << " from " << originalLhsType
                            << " and " << originalRhsType << "\n";
                        return {};
                    }

                    Value add = arith::AddFOp::create(builder, loc, lhs, rhs).getResult();
                    tileValues[*outputTensorId] = add;
                    continue;
                }

                llvm::errs() << "unsupported elementwise MAPS op " << *opName << "\n";
                return {};
            }

            // send transition fragments produced by this tile
            for (const TransitionFragment &fragment : *transitionFragments) {
                if (fragment.srcHartId != tileObj.hartId)
                    continue;

                Value source = tileValues.lookup(fragment.tensorId);
                if (!source)
                    continue;

                SmallVector<int64_t> shape;
                SmallVector<OpFoldResult> offsets;
                SmallVector<OpFoldResult> sizes;
                SmallVector<OpFoldResult> strides;
                SmallVector<int64_t> localStarts =
                    getLocalStartsForSourceTile(*transitionFragments, fragment);

                for (auto indexed : llvm::enumerate(fragment.srcDims)) {
                    const SliceDim &dim = indexed.value();
                    int64_t localOffset = dim.start - localStarts[indexed.index()];
                    shape.push_back(dim.length);
                    offsets.push_back(builder.getIndexAttr(localOffset));
                    sizes.push_back(builder.getIndexAttr(dim.length));
                    strides.push_back(builder.getIndexAttr(1));
                }

                auto sourceType = cast<RankedTensorType>(source.getType());
                auto sliceType = RankedTensorType::get(shape, sourceType.getElementType());

                Value slice = tensor::ExtractSliceOp::create(
                    builder,
                    loc,
                    sliceType,
                    source,
                    offsets,
                    sizes,
                    strides
                );

                maps::SendOp::create(
                    builder,
                    loc,
                    slice,
                    SymbolRefAttr::get(ctx, fragment.name),
                    builder.getI64IntegerAttr(fragment.srcHartId),
                    builder.getI64IntegerAttr(fragment.dstHartId)
                );
            }

            // send final output fragments back to L2
            for (const TransitionFragment &fragment : *outputFragments) {
                if (fragment.srcHartId != tileObj.hartId)
                    continue;

                Value source = tileValues.lookup(fragment.tensorId);
                if (!source)
                    continue;

                SmallVector<int64_t> shape;
                SmallVector<OpFoldResult> offsets;
                SmallVector<OpFoldResult> sizes;
                SmallVector<OpFoldResult> strides;
                SmallVector<int64_t> localStarts =
                    getLocalStartsForSourceTile(*outputFragments, fragment);

                for (auto indexed : llvm::enumerate(fragment.srcDims)) {
                    const SliceDim &dim = indexed.value();
                    int64_t localOffset = dim.start - localStarts[indexed.index()];
                    shape.push_back(dim.length);
                    offsets.push_back(builder.getIndexAttr(localOffset));
                    sizes.push_back(builder.getIndexAttr(dim.length));
                    strides.push_back(builder.getIndexAttr(1));
                }

                auto sourceType = cast<RankedTensorType>(source.getType());
                auto sliceType = RankedTensorType::get(shape, sourceType.getElementType());

                Value slice = tensor::ExtractSliceOp::create(
                    builder,
                    loc,
                    sliceType,
                    source,
                    offsets,
                    sizes,
                    strides
                );

                maps::SendOp::create(
                    builder,
                    loc,
                    slice,
                    SymbolRefAttr::get(ctx, fragment.name),
                    builder.getI64IntegerAttr(fragment.srcHartId),
                    builder.getI64IntegerAttr(fragment.dstHartId)
                );
            }

            builder.setInsertionPointAfter(tile);
            // end of tile block
        }

        builder.setInsertionPointAfter(stage);
    }

    builder.setInsertionPointAfter(run);
    // end of run region

    // receive final output fragments from tiles and assemble them in L2
    llvm::DenseMap<int64_t, Value> outputTensorValues = OutputTensorInitValues;
    for (const TransitionFragment &fragment : *outputFragments) {
        auto *tensor = (*tensors)[fragment.tensorId].getAsObject();
        SmallVector<int64_t> shape;
        SmallVector<OpFoldResult> offsets;
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides;

        for (const SliceDim &dim : fragment.dstDims) {
            shape.push_back(dim.length);
            offsets.push_back(builder.getIndexAttr(dim.start));
            sizes.push_back(builder.getIndexAttr(dim.length));
            strides.push_back(builder.getIndexAttr(1));
        }

        Type elementType = parseElementType(*tensor, builder);
        Type resultType = RankedTensorType::get(shape, elementType);

        Value outputSlice = maps::RecvOp::create(
            builder,
            loc,
            resultType,
            StringRef(fragment.name),
            fragment.srcHartId,
            fragment.dstHartId
        ).getResult();

        Value currentOutput = outputTensorValues.lookup(fragment.tensorId);
        Value updatedOutput = tensor::InsertSliceOp::create(
            builder,
            loc,
            outputSlice,
            currentOutput,
            offsets,
            sizes,
            strides,
            ArrayRef<NamedAttribute>{}
        ).getResult();
        outputTensorValues[fragment.tensorId] = updatedOutput;
    }

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
