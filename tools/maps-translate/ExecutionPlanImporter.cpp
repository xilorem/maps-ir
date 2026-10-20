#include "ExecutionPlanImporter.h"

// Dialects
#include "maps/Dialect/Maps/IR/Maps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// IR constructs
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"

// JSON parsing
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

// Abstract data types
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/DenseSet.h"


using namespace mlir;

namespace mlir::maps {

struct SliceDim {
    int64_t start;
    int64_t length;
};

struct L2SourceChannelBinding {
    std::string channelName;
    std::string slotName;  // local tile storage name
    int64_t tensorId;
    int64_t srcHartId;
    int64_t dstHartId;
    SmallVector<SliceDim> srcDims;
    SmallVector<SliceDim> dstDims;
    int64_t transitionId = -1;
    int64_t destinationStageId = -1;
    int64_t destinationLayerIndex = -1;
    int64_t destinationInputIndex = -1;
};

struct TileSourceChannelBinding {
    std::string channelName;
    int64_t tensorId;
    int64_t srcHartId;
    int64_t dstHartId;
    SmallVector<SliceDim> srcDims;
    SmallVector<SliceDim> dstDims;
    SmallVector<SliceDim> srcParentDims;
    SmallVector<SliceDim> dstParentDims;
    bool hasRelativeSubSlices;
    int64_t transitionId = -1;
    int64_t sourceStageId = -1;
    int64_t destinationStageId = -1;
    std::string globalName;
};

using TileSourceChannelBindingVector = SmallVector<TileSourceChannelBinding, 0>;

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
    bool hasExplicitTileIds;
    llvm::DenseSet<int64_t> tileIds;
    llvm::DenseMap<int64_t, int64_t> physicalToVirtualTileId;
};


static bool isInitializerTensor(const llvm::json::Object &tensor) {
    return tensor.getBoolean("is_initializer").value_or(false);
}


static Type parseElementType(const llvm::json::Object &tensor,
                             OpBuilder &builder) {
    std::optional<StringRef> dtype = tensor.getString("dtype");
    if (auto *initializer = tensor.getObject("initializer")) {
        if (auto packedDtype = initializer->getString("dtype"))
            dtype = packedDtype;
    }
    
    if (!dtype) {
        if (std::optional<int64_t> elemBytes = tensor.getInteger("elem_bytes")) {
            if (*elemBytes == 2)
                return builder.getF16Type();
            if (*elemBytes == 4)
                return builder.getF32Type();
            if (*elemBytes == 8)
                return builder.getF64Type();
        }
        return builder.getF32Type();
    }

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

static std::optional<SmallVector<int64_t>>
parseIntArrayField(const llvm::json::Object &object, StringRef fieldName) {
    auto *array = object.getArray(fieldName);
    if (!array)
        return std::nullopt;

    SmallVector<int64_t> values;
    values.reserve(array->size());
    for (const llvm::json::Value &entry : *array) {
        auto value = entry.getAsInteger();
        if (!value)
            return std::nullopt;
        values.push_back(*value);
    }
    return values;
}

static Value createZeroValue(OpBuilder &builder, Location loc, Type type) {
    return arith::ConstantOp::create(builder, loc, builder.getZeroAttr(type));
}

static Value createIndexConstant(OpBuilder &builder, Location loc, int64_t value) {
    return arith::ConstantIndexOp::create(builder, loc, value);
}

static int64_t ceilDivNonNegative(int64_t dividend, int64_t divisor) {
    assert(dividend >= 0 && divisor > 0);
    return (dividend + divisor - 1) / divisor;
}

static Value emitElementwiseAdd(Location loc, Value lhs, Value rhs,
                                RankedTensorType outputType,
                                OpBuilder &builder) {
    Value init = tensor::EmptyOp::create(
        builder, loc, outputType.getShape(),
        outputType.getElementType()).getResult();
    auto identityMap =
        AffineMap::getMultiDimIdentityMap(outputType.getRank(), builder.getContext());
    SmallVector<AffineMap> indexingMaps = {
        identityMap,
        identityMap,
        identityMap,
    };
    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(), utils::IteratorType::parallel);
    return linalg::GenericOp::create(
               builder,
               loc,
               outputType,
               ValueRange{lhs, rhs},
               ValueRange{init},
               indexingMaps,
               iteratorTypes,
               [&](OpBuilder &nestedBuilder, Location nestedLoc,
                   ValueRange blockArgs) {
                   Value add = arith::AddFOp::create(
                       nestedBuilder, nestedLoc, blockArgs[0], blockArgs[1]);
                   linalg::YieldOp::create(nestedBuilder, nestedLoc, add);
               })
        .getResult(0);
}

static Value emitElementwiseExp(Location loc, Value input,
                                RankedTensorType outputType,
                                OpBuilder &builder) {
    Value init = tensor::EmptyOp::create(
        builder, loc, outputType.getShape(),
        outputType.getElementType()).getResult();
    auto identityMap =
        AffineMap::getMultiDimIdentityMap(outputType.getRank(), builder.getContext());
    SmallVector<AffineMap> indexingMaps = {
        identityMap,
        identityMap,
    };
    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(), utils::IteratorType::parallel);
    return linalg::GenericOp::create(
               builder,
               loc,
               outputType,
               ValueRange{input},
               ValueRange{init},
               indexingMaps,
               iteratorTypes,
               [&](OpBuilder &nestedBuilder, Location nestedLoc,
                   ValueRange blockArgs) {
                   Value exp = math::ExpOp::create(
                       nestedBuilder, nestedLoc, blockArgs[0]);
                   linalg::YieldOp::create(nestedBuilder, nestedLoc, exp);
               })
        .getResult(0);
}

static Value emitElementwiseLog(Location loc, Value input,
                                RankedTensorType outputType,
                                OpBuilder &builder) {
    Value init = tensor::EmptyOp::create(
        builder, loc, outputType.getShape(),
        outputType.getElementType()).getResult();
    auto identityMap =
        AffineMap::getMultiDimIdentityMap(outputType.getRank(), builder.getContext());
    SmallVector<AffineMap> indexingMaps = {
        identityMap,
        identityMap,
    };
    SmallVector<utils::IteratorType> iteratorTypes(
        outputType.getRank(), utils::IteratorType::parallel);
    return linalg::GenericOp::create(
               builder,
               loc,
               outputType,
               ValueRange{input},
               ValueRange{init},
               indexingMaps,
               iteratorTypes,
               [&](OpBuilder &nestedBuilder, Location nestedLoc,
                   ValueRange blockArgs) {
                   Value log = math::LogOp::create(
                       nestedBuilder, nestedLoc, blockArgs[0]);
                   linalg::YieldOp::create(nestedBuilder, nestedLoc, log);
               })
        .getResult(0);
}

static FailureOr<Value> emitOutputReformat(Location loc, Value inputValue,
                                           RankedTensorType outputType,
                                           OpBuilder &builder) {
    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());
    if (!inputType || inputType.getRank() != 2 || outputType.getRank() != 4)
        return failure();

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (llvm::any_of(inputShape, ShapedType::isDynamic) ||
        llvm::any_of(outputShape, ShapedType::isDynamic))
        return failure();

    int64_t batch = outputShape[0];
    int64_t channels = outputShape[1];
    int64_t height = outputShape[2];
    int64_t width = outputShape[3];
    if (inputShape[0] != batch * height * width || inputShape[1] != channels)
        return failure();

    Value assembled = tensor::EmptyOp::create(
        builder, loc, outputShape, outputType.getElementType()).getResult();
    auto columnType = RankedTensorType::get(
        {inputShape[0], 1}, outputType.getElementType());
    auto flatType = RankedTensorType::get(
        {inputShape[0]}, outputType.getElementType());
    auto channelType = RankedTensorType::get(
        {batch, 1, height, width}, outputType.getElementType());
    SmallVector<ReassociationIndices> collapseReassociation = {{0, 1}};
    SmallVector<ReassociationIndices> expandReassociation = {{0, 1, 2, 3}};

    for (int64_t channel = 0; channel < channels; ++channel) {
        Value columnSlice = tensor::ExtractSliceOp::create(
            builder,
            loc,
            columnType,
            inputValue,
            ArrayRef<OpFoldResult>{builder.getIndexAttr(0),
                                   builder.getIndexAttr(channel)},
            ArrayRef<OpFoldResult>{builder.getIndexAttr(inputShape[0]),
                                   builder.getIndexAttr(1)},
            ArrayRef<OpFoldResult>{builder.getIndexAttr(1),
                                   builder.getIndexAttr(1)});
        Value flatColumn = tensor::CollapseShapeOp::create(
            builder, loc, flatType, columnSlice, collapseReassociation);
        Value reshapedChannel = tensor::ExpandShapeOp::create(
            builder, loc, channelType, flatColumn, expandReassociation);
        assembled = tensor::InsertSliceOp::create(
            builder,
            loc,
            reshapedChannel,
            assembled,
            ArrayRef<OpFoldResult>{builder.getIndexAttr(0),
                                   builder.getIndexAttr(channel),
                                   builder.getIndexAttr(0),
                                   builder.getIndexAttr(0)},
            ArrayRef<OpFoldResult>{builder.getIndexAttr(batch),
                                   builder.getIndexAttr(1),
                                   builder.getIndexAttr(height),
                                   builder.getIndexAttr(width)},
            ArrayRef<OpFoldResult>{builder.getIndexAttr(1),
                                   builder.getIndexAttr(1),
                                   builder.getIndexAttr(1),
                                   builder.getIndexAttr(1)});
    }

    return assembled;
}

static FailureOr<Value> emitWeightPack(Location loc, Value inputValue,
                                       RankedTensorType outputType,
                                       OpBuilder &builder) {
    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());
    if (!inputType || inputType.getRank() != 4 || outputType.getRank() != 2)
        return failure();

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (llvm::any_of(inputShape, ShapedType::isDynamic) ||
        llvm::any_of(outputShape, ShapedType::isDynamic))
        return failure();

    int64_t outChannels = inputShape[0];
    int64_t inChannels = inputShape[1];
    int64_t kernelHeight = inputShape[2];
    int64_t kernelWidth = inputShape[3];
    if (outputShape[0] != inChannels * kernelHeight * kernelWidth ||
        outputShape[1] != outChannels)
        return failure();

    auto transposedType = RankedTensorType::get(
        {inChannels, kernelHeight, kernelWidth, outChannels},
        outputType.getElementType());
    Value transposedInit = tensor::EmptyOp::create(
        builder, loc, transposedType.getShape(),
        transposedType.getElementType()).getResult();
    Value transposed = linalg::TransposeOp::create(
        builder, loc, inputValue, transposedInit,
        ArrayRef<int64_t>{1, 2, 3, 0}).getResult()[0];
    SmallVector<ReassociationIndices> reassociation = {{0, 1, 2}, {3}};
    return tensor::CollapseShapeOp::create(
        builder, loc, outputType, transposed, reassociation).getResult();
}

static FailureOr<Value> emitIm2Col(Location loc, Value inputValue,
                                   const llvm::json::Object &payload,
                                   RankedTensorType outputType,
                                   OpBuilder &builder) {
    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());
    if (!inputType || inputType.getRank() != 4 || outputType.getRank() != 2)
        return failure();

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (llvm::any_of(inputShape, ShapedType::isDynamic) ||
        llvm::any_of(outputShape, ShapedType::isDynamic))
        return failure();

    auto kernelShape = parseIntArrayField(payload, "kernel_shape");
    auto pads = parseIntArrayField(payload, "pads");
    auto strides = parseIntArrayField(payload, "strides");
    auto dilations = parseIntArrayField(payload, "dilations");
    if (!kernelShape || !pads || !strides || !dilations ||
        kernelShape->size() != 2 || pads->size() != 4 || strides->size() != 2 ||
        dilations->size() != 2)
        return failure();

    int64_t batch = inputShape[0];
    int64_t channels = inputShape[1];
    int64_t inputHeight = inputShape[2];
    int64_t inputWidth = inputShape[3];
    int64_t kernelHeight = (*kernelShape)[0];
    int64_t kernelWidth = (*kernelShape)[1];
    int64_t padTop = (*pads)[0];
    int64_t padLeft = (*pads)[1];
    int64_t padBottom = (*pads)[2];
    int64_t padRight = (*pads)[3];
    int64_t strideY = (*strides)[0];
    int64_t strideX = (*strides)[1];
    int64_t dilationY = (*dilations)[0];
    int64_t dilationX = (*dilations)[1];

    int64_t outputHeight =
        (inputHeight + padTop + padBottom - dilationY * (kernelHeight - 1) - 1) /
            strideY +
        1;
    int64_t outputWidth =
        (inputWidth + padLeft + padRight - dilationX * (kernelWidth - 1) - 1) /
            strideX +
        1;
    if (outputShape[0] != batch * outputHeight * outputWidth ||
        outputShape[1] != channels * kernelHeight * kernelWidth)
        return failure();

    Value assembled = tensor::EmptyOp::create(
        builder, loc, outputShape, outputType.getElementType()).getResult();
    Value zero = createZeroValue(builder, loc, outputType.getElementType());
    assembled = linalg::FillOp::create(builder, loc, zero, assembled).getResult(0);

    auto spatialType = RankedTensorType::get(
        {batch, 1, outputHeight, outputWidth}, outputType.getElementType());
    auto flatColumnType = RankedTensorType::get(
        {batch * outputHeight * outputWidth}, outputType.getElementType());
    auto outputColumnType = RankedTensorType::get(
        {batch * outputHeight * outputWidth, 1}, outputType.getElementType());
    SmallVector<ReassociationIndices> collapseSpatial = {{0, 1, 2, 3}};
    SmallVector<ReassociationIndices> expandColumn = {{0, 1}};

    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t kernelY = 0; kernelY < kernelHeight; ++kernelY) {
            for (int64_t kernelX = 0; kernelX < kernelWidth; ++kernelX) {
                int64_t column = (channel * kernelHeight + kernelY) * kernelWidth + kernelX;
                Value columnTensor = tensor::EmptyOp::create(
                    builder, loc, spatialType.getShape(),
                    spatialType.getElementType()).getResult();
                columnTensor =
                    linalg::FillOp::create(builder, loc, zero, columnTensor).getResult(0);

                int64_t startY = kernelY * dilationY - padTop;
                int64_t startX = kernelX * dilationX - padLeft;
                int64_t validOutYStart =
                    startY < 0 ? ceilDivNonNegative(-startY, strideY) : 0;
                int64_t validOutXStart =
                    startX < 0 ? ceilDivNonNegative(-startX, strideX) : 0;
                int64_t validOutYEnd =
                    std::min(outputHeight,
                             (inputHeight - 1 - startY) / strideY + 1);
                int64_t validOutXEnd =
                    std::min(outputWidth,
                             (inputWidth - 1 - startX) / strideX + 1);

                if (validOutYStart < validOutYEnd && validOutXStart < validOutXEnd) {
                    int64_t validHeight = validOutYEnd - validOutYStart;
                    int64_t validWidth = validOutXEnd - validOutXStart;
                    int64_t inputStartY = startY + validOutYStart * strideY;
                    int64_t inputStartX = startX + validOutXStart * strideX;

                    auto patchType = RankedTensorType::get(
                        {batch, 1, validHeight, validWidth},
                        outputType.getElementType());
                    Value patch = tensor::ExtractSliceOp::create(
                        builder,
                        loc,
                        patchType,
                        inputValue,
                        ArrayRef<OpFoldResult>{builder.getIndexAttr(0),
                                               builder.getIndexAttr(channel),
                                               builder.getIndexAttr(inputStartY),
                                               builder.getIndexAttr(inputStartX)},
                        ArrayRef<OpFoldResult>{builder.getIndexAttr(batch),
                                               builder.getIndexAttr(1),
                                               builder.getIndexAttr(validHeight),
                                               builder.getIndexAttr(validWidth)},
                        ArrayRef<OpFoldResult>{builder.getIndexAttr(1),
                                               builder.getIndexAttr(1),
                                               builder.getIndexAttr(strideY),
                                               builder.getIndexAttr(strideX)});
                    columnTensor = tensor::InsertSliceOp::create(
                        builder,
                        loc,
                        patch,
                        columnTensor,
                        ArrayRef<OpFoldResult>{builder.getIndexAttr(0),
                                               builder.getIndexAttr(0),
                                               builder.getIndexAttr(validOutYStart),
                                               builder.getIndexAttr(validOutXStart)},
                        ArrayRef<OpFoldResult>{builder.getIndexAttr(batch),
                                               builder.getIndexAttr(1),
                                               builder.getIndexAttr(validHeight),
                                               builder.getIndexAttr(validWidth)},
                        ArrayRef<OpFoldResult>{builder.getIndexAttr(1),
                                               builder.getIndexAttr(1),
                                               builder.getIndexAttr(1),
                                               builder.getIndexAttr(1)});
                }

                Value flatColumn = tensor::CollapseShapeOp::create(
                    builder, loc, flatColumnType, columnTensor, collapseSpatial);
                Value outputColumn = tensor::ExpandShapeOp::create(
                    builder, loc, outputColumnType, flatColumn, expandColumn);
                assembled = tensor::InsertSliceOp::create(
                    builder,
                    loc,
                    outputColumn,
                    assembled,
                    ArrayRef<OpFoldResult>{builder.getIndexAttr(0),
                                           builder.getIndexAttr(column)},
                    ArrayRef<OpFoldResult>{
                        builder.getIndexAttr(batch * outputHeight * outputWidth),
                        builder.getIndexAttr(1)},
                    ArrayRef<OpFoldResult>{builder.getIndexAttr(1),
                                           builder.getIndexAttr(1)});
            }
        }
    }

    return assembled;
}

static FailureOr<Value> emitTransformAsStandardOps(
    Location loc, StringRef opName, ValueRange inputValues,
    const llvm::json::Object &payload, RankedTensorType outputType,
    OpBuilder &builder) {
    if (opName == "Im2Col") {
        if (inputValues.size() != 1)
            return failure();
        return emitIm2Col(loc, inputValues.front(), payload, outputType, builder);
    }

    if (opName == "WeightPack") {
        if (inputValues.size() != 1)
            return failure();
        return emitWeightPack(loc, inputValues.front(), outputType, builder);
    }

    if (opName == "OutputReformat") {
        if (inputValues.size() != 1)
            return failure();
        return emitOutputReformat(loc, inputValues.front(), outputType, builder);
    }

    return failure();
}


static bool isTileInStage(const Tile &tile, const Stage &stage) {
    if (stage.hasExplicitTileIds)
        return stage.tileIds.contains(tile.hartId);

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


static SmallVector<int64_t> getLocalStartsForSourceTile(ArrayRef<TileSourceChannelBinding> bindings,
                                                        const TileSourceChannelBinding &selected) {
    SmallVector<int64_t> starts;
    starts.reserve(selected.srcDims.size());
    for (const SliceDim &dim : selected.srcDims)
        starts.push_back(dim.start);

    for (const TileSourceChannelBinding &binding : bindings) {
        if (binding.tensorId != selected.tensorId ||
            binding.srcHartId != selected.srcHartId ||
            binding.srcDims.size() != starts.size())
            continue;

        for (auto indexed : llvm::enumerate(binding.srcDims))
            starts[indexed.index()] = std::min(starts[indexed.index()], indexed.value().start);
    }

    return starts;
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
                    const Stage &stage, Type elementType);

static std::optional<SmallVector<SliceDim>>
parseTileOutputSlice(const llvm::json::Object &tensor,
                     const llvm::json::Object &output, const Tile &tile,
                     const Stage &stage) {
    auto *dimsJson = tensor.getArray("dims");
    auto *layout = output.getObject("layout");
    auto *submesh = layout ? layout->getObject("submesh") : nullptr;
    auto *meshX = layout ? layout->getObject("mesh_x") : nullptr;
    auto *meshY = layout ? layout->getObject("mesh_y") : nullptr;
    if (!dimsJson || !layout || !submesh || !meshX || !meshY)
        return std::nullopt;

    auto logicalWidth = layout->getInteger("logical_width");
    auto logicalHeight = layout->getInteger("logical_height");
    if (!logicalWidth || !logicalHeight ||
        *logicalWidth <= 0 || *logicalHeight <= 0)
        return std::nullopt;

    std::optional<int64_t> ordinal;
    auto x0 = submesh->getInteger("x0");
    auto y0 = submesh->getInteger("y0");
    auto width = submesh->getInteger("width");
    if (x0 && y0 && width) {
        ordinal = (tile.y - *y0) * *width + tile.x - *x0;
    } else if (auto *tileIds = submesh->getArray("tile_ids")) {
        int64_t layoutTileId = tile.hartId;
        auto virtualTileIt = stage.physicalToVirtualTileId.find(tile.hartId);
        if (virtualTileIt != stage.physicalToVirtualTileId.end())
            layoutTileId = virtualTileIt->second;

        for (auto indexed : llvm::enumerate(*tileIds)) {
            auto tileId = indexed.value().getAsInteger();
            if (!tileId)
                return std::nullopt;
            if (*tileId == layoutTileId) {
                ordinal = static_cast<int64_t>(indexed.index());
                break;
            }
        }
    }
    if (!ordinal)
        return std::nullopt;

    if (*ordinal < 0 || *ordinal >= *logicalWidth * *logicalHeight)
        return std::nullopt;
    int64_t logicalX = *ordinal % *logicalWidth;
    int64_t logicalY = *ordinal / *logicalWidth;

    SmallVector<SliceDim> dims;
    for (const llvm::json::Value &dimValue : *dimsJson) {
        auto dim = dimValue.getAsInteger();
        if (!dim)
            return std::nullopt;
        dims.push_back({0, *dim});
    }

    auto shardDimension = [&](const llvm::json::Object &mapping, int64_t parts,
                              int64_t index) -> bool {
        constexpr int64_t shardMode = 1;
        auto mode = mapping.getInteger("mode");
        if (!mode)
            return false;
        int64_t shardGranularity = 1;
        if (const llvm::json::Value *granularity =
                mapping.get("shard_granularity")) {
            auto parsedGranularity = granularity->getAsInteger();
            if (!parsedGranularity)
                return false;
            shardGranularity = *parsedGranularity;
        }
        if (shardGranularity <= 0)
            return false;
        if (*mode != shardMode)
            return shardGranularity == 1;
        auto axis = mapping.getInteger("tensor_axis");
        if (!axis || *axis < 0 || static_cast<size_t>(*axis) >= dims.size())
            return false;
        SliceDim &dim = dims[*axis];
        if (dim.length % shardGranularity != 0)
            return false;
        SliceDim unitRange = partitionDim(
            dim.length / shardGranularity, parts, index);
        dim = {
            dim.start + unitRange.start * shardGranularity,
            unitRange.length * shardGranularity,
        };
        return true;
    };

    if (!shardDimension(*meshX, *logicalWidth, logicalX) ||
        !shardDimension(*meshY, *logicalHeight, logicalY))
        return std::nullopt;
    return dims;
}

static std::optional<RankedTensorType>
parseTileOutputType(const llvm::json::Object &tensor,
                    const llvm::json::Object &output, const Tile &tile,
                    const Stage &stage, Type elementType) {
    auto dims = parseTileOutputSlice(tensor, output, tile, stage);
    if (!dims)
        return std::nullopt;

    SmallVector<int64_t> shape;
    shape.reserve(dims->size());
    for (const SliceDim &dim : *dims)
        shape.push_back(dim.length);
    return RankedTensorType::get(shape, elementType);
}

static std::optional<StringRef>
getDescriptorKernelKind(const llvm::json::Object &payload) {
    auto workKind = payload.getString("work_kind");
    auto opName = payload.getString("op_name");
    StringRef kind = workKind ? *workKind : opName.value_or("");
    if (kind == "Im2Col")
        kind = "IM2COL";
    else if (kind == "OutputReformat")
        kind = "OUTPUT_REFORMAT";
    else if (kind == "GEMM")
        kind = "MATMUL";
    if (kind.empty() && payload.getBoolean("transpose_w").has_value())
        kind = "MATMUL";

    if (kind == "AllReduceSum" || kind == "AllReduceMax")
        return kind;

    static constexpr StringLiteral supportedKinds[] = {
        "MAX_POOL",     "ABS",          "ADD",          "DEPTHWISE_CONV", "DIV",
        "EXP",          "SOFTMAX_EXP",  "GROUP_NORMALIZE", "GROUP_REDUCE",
        "GROUP_CENTERED_REDUCE", "IM2COL", "OUTPUT_REFORMAT", "SPLIT", "LOG",
        "MATMUL",       "MUL",          "NEG",            "POW",
        "REDUCE_MAX",
        "REDUCE_SUM",   "RELU",         "RESHAPE",        "SIGMOID",
        "SLICE",        "SQRT",         "SUB",            "TRANSPOSE",
    };
    for (StringRef supported : supportedKinds) {
        if (kind == supported)
            return kind;
    }
    return std::nullopt;
}

static int64_t getF32ParameterBits(double value) {
    llvm::APFloat parameter(static_cast<float>(value));
    return static_cast<int64_t>(parameter.bitcastToAPInt().getZExtValue());
}

static int64_t packU16Pair(int64_t first, int64_t second) {
    if (first < 0 || first > UINT16_MAX || second < 0 || second > UINT16_MAX)
        return -1;
    return first | (second << 16);
}

static std::optional<SmallVector<int64_t>>
getDescriptorKernelParams(StringRef kind, const llvm::json::Object &payload) {
    SmallVector<int64_t> params;

    if (kind == "MAX_POOL") {
        auto kernel = parseIntArrayField(payload, "kernel_shape");
        auto strides = parseIntArrayField(payload, "strides");
        auto pads = parseIntArrayField(payload, "pads");
        if (!kernel || kernel->size()!=2 || !strides || strides->size()!=2 || !pads || pads->size()!=4)
            return std::nullopt;
        params.append(*kernel);
        params.append(*strides);
        params.append(*pads);
    } else if (kind == "IM2COL") {
        auto kernelShape = parseIntArrayField(payload, "kernel_shape");
        auto strides = parseIntArrayField(payload, "strides");
        auto dilations = parseIntArrayField(payload, "dilations");
        auto pads = parseIntArrayField(payload, "pads");
        auto *input = payload.getObject("x");
        auto inputDims = input ? parseIntArrayField(*input, "dims") : std::nullopt;
        if (!kernelShape || kernelShape->size() != 2 || !strides ||
            strides->size() != 2 || !dilations || dilations->size() != 2 ||
            !pads || pads->size() != 4 || !inputDims || inputDims->size() != 4)
            return std::nullopt;
        int64_t outputHeight =
            ((*inputDims)[2] + (*pads)[0] + (*pads)[2] -
             (*dilations)[0] * ((*kernelShape)[0] - 1) - 1) /
                (*strides)[0] +
            1;
        int64_t outputWidth =
            ((*inputDims)[3] + (*pads)[1] + (*pads)[3] -
             (*dilations)[1] * ((*kernelShape)[1] - 1) - 1) /
                (*strides)[1] +
            1;
        params.append({packU16Pair((*kernelShape)[0], (*kernelShape)[1]),
                       packU16Pair((*strides)[0], (*strides)[1]),
                       packU16Pair((*dilations)[0], (*dilations)[1]),
                       packU16Pair((*pads)[0], (*pads)[1]),
                       packU16Pair((*pads)[2], (*pads)[3]),
                       packU16Pair((*inputDims)[2], (*inputDims)[3]),
                       packU16Pair(outputHeight, outputWidth)});
        params.push_back(payload.getBoolean("channels_first").value_or(false));
        if (llvm::is_contained(params, -1))
            return std::nullopt;
    } else if (kind == "DEPTHWISE_CONV") {
        auto strides = parseIntArrayField(payload, "strides");
        auto dilations = parseIntArrayField(payload, "dilations");
        auto pads = parseIntArrayField(payload, "pads");
        auto *input = payload.getObject("x");
        auto *weight = payload.getObject("w");
        auto inputDims = input ? parseIntArrayField(*input, "dims") : std::nullopt;
        auto weightDims = weight ? parseIntArrayField(*weight, "dims") : std::nullopt;
        if (!strides || strides->size() != 2 || !dilations ||
            dilations->size() != 2 || !pads || pads->size() != 4 ||
            !inputDims || inputDims->size() != 4 || !weightDims ||
            weightDims->size() != 4 || (*inputDims)[1] <= 0 ||
            (*weightDims)[0] % (*inputDims)[1] != 0)
            return std::nullopt;
        params.append(*strides);
        params.append(*dilations);
        params.append(*pads);
        // The lower padding is not needed by the tile-local loop. Preserve the
        // right padding in slot 7 and use slot 6 for the channel multiplier so
        // unaligned output-channel shards can rebase their input channels.
        params[6] = (*weightDims)[0] / (*inputDims)[1];
    } else if (kind == "REDUCE_MAX" || kind == "REDUCE_SUM") {
        auto axis = payload.getInteger("axis");
        if (!axis)
            return std::nullopt;
        params.push_back(*axis);
    } else if (kind == "GROUP_REDUCE" ||
               kind == "GROUP_CENTERED_REDUCE") {
        auto groups = payload.getInteger("num_groups");
        auto elementsPerGroup = payload.getInteger("element_count_per_group");
        auto channelsPerGroup = payload.getInteger("channel_count_per_group");
        if (!groups || !elementsPerGroup || !channelsPerGroup)
            return std::nullopt;
        params.push_back(*groups);
        params.push_back(*elementsPerGroup);
        params.push_back(*channelsPerGroup);
    } else if (kind == "GROUP_NORMALIZE") {
        auto groups = payload.getInteger("num_groups");
        auto epsilon = payload.getNumber("epsilon");
        auto elementsPerGroup = payload.getInteger("element_count_per_group");
        auto channelsPerGroup = payload.getInteger("channel_count_per_group");
        if (!groups || !epsilon || !elementsPerGroup || !channelsPerGroup)
            return std::nullopt;
        params.push_back(*groups);
        params.push_back(getF32ParameterBits(*epsilon));
        params.push_back(*elementsPerGroup);
        params.push_back(*channelsPerGroup);
    } else if (kind == "SPLIT") {
        auto axis = payload.getInteger("axis");
        if (!axis)
            return std::nullopt;
        params.push_back(*axis);
    } else if (kind == "MUL") {
        if (auto factor = payload.getNumber("factor"))
            params.push_back(getF32ParameterBits(*factor));
    } else if (kind == "TRANSPOSE") {
        auto permutation = parseIntArrayField(payload, "permutation");
        if (!permutation || permutation->size() > 8)
            return std::nullopt;
        params.append(*permutation);
    } else if (kind == "SLICE") {
        auto offsets = parseIntArrayField(payload, "offsets");
        if (!offsets || offsets->size() > 8)
            return std::nullopt;
        params.append(*offsets);
    }

    return params;
}

static FailureOr<Value> emitDescriptorKernel(
    Location loc, const llvm::json::Object &node,
    const llvm::json::Object &layer, unsigned layerIndex,
    const llvm::json::Array &inputs, const llvm::json::Object &output,
    const Tile &tile, const Stage &stage, OpBuilder &builder,
    llvm::DenseMap<int64_t, Value> &tileValues) {
    auto *payload = node.getObject("payload");
    auto *nodeOutputs = node.getArray("outputs");
    auto *nodeOutput =
        nodeOutputs && !nodeOutputs->empty() ? (*nodeOutputs)[0].getAsObject()
                                            : nullptr;
    if (!payload || !nodeOutput)
        return failure();

    auto kind = getDescriptorKernelKind(*payload);
    if (!kind)
        return failure();
    auto params = getDescriptorKernelParams(*kind, *payload);
    if (!params)
        return failure();

    SmallVector<Value> inputValues;
    for (const llvm::json::Value &input : inputs) {
        auto tensorId = getTensorId(input);
        Value value = tensorId ? tileValues.lookup(*tensorId) : Value{};
        if (!value)
            return failure();
        inputValues.push_back(value);
    }
    if (inputValues.empty())
        return failure();

    auto inputType = dyn_cast<RankedTensorType>(inputValues.front().getType());
    auto outputType =
        inputType ? parseTileOutputType(*nodeOutput, output, tile, stage,
                                        inputType.getElementType())
                  : std::nullopt;
    auto outputSlice = parseTileOutputSlice(*nodeOutput, output, tile, stage);
    if (!outputType || !outputSlice)
        return failure();

    if (*kind == "MATMUL") {
        auto lhsType = dyn_cast<RankedTensorType>(inputValues[0].getType());
        auto rhsType = inputValues.size() > 1
                           ? dyn_cast<RankedTensorType>(inputValues[1].getType())
                           : RankedTensorType{};
        if (!lhsType || !rhsType || lhsType.getRank() < 2 ||
            rhsType.getRank() < 2)
            return failure();
        int64_t batchVolume = 1;
        int64_t matrixRows = lhsType.getShape()[lhsType.getRank() - 2];
        int64_t matrixDepth = lhsType.getShape()[lhsType.getRank() - 1];
        bool transposeW = payload->getBoolean("transpose_w").value_or(false);
        int64_t matrixColumns =
            rhsType.getShape()[rhsType.getRank() - (transposeW ? 2 : 1)];
        bool convOutput = outputType->getRank() == 4 && lhsType.getRank() == 2 &&
                          rhsType.getRank() == 2;
        if (convOutput) {
            matrixRows = outputType->getShape()[1];
            matrixColumns = outputType->getShape()[2] * outputType->getShape()[3];
        } else {
            for (int64_t dimension : outputType->getShape().drop_back(2))
                batchVolume *= dimension;
        }
        params->clear();
        params->append(
            {matrixRows, matrixDepth, matrixColumns, batchVolume,
             batchVolume > 1, batchVolume > 1, convOutput,
             transposeW});
    }

    OperationState state(loc, maps::KernelOp::getOperationName());
    state.addOperands(inputValues);
    state.addTypes(*outputType);
    state.addAttribute("kind", builder.getStringAttr(*kind));
    state.addAttribute("params", builder.getDenseI64ArrayAttr(*params));
    SmallVector<int64_t> outputFullOffset;
    for (const SliceDim &dim : *outputSlice)
        outputFullOffset.push_back(dim.start);
    state.addAttribute("output_full_offset_0",
                       builder.getDenseI64ArrayAttr(outputFullOffset));
    if (auto deviceName = layer.getString("device_name"))
        state.addAttribute("device_name", builder.getStringAttr(*deviceName));
    if (*kind == "AllReduceSum" || *kind == "AllReduceMax") {
        auto *groups = layer.getArray("collective_groups");
        if (!groups)
            return failure();

        bool foundGroup = false;
        for (auto groupEntry : llvm::enumerate(*groups)) {
            auto *group = groupEntry.value().getAsObject();
            auto *tileIds = group ? group->getArray("tile_ids") : nullptr;
            if (!tileIds)
                return failure();

            SmallVector<int64_t> participants;
            bool tileParticipates = false;
            for (const llvm::json::Value &tileIdValue : *tileIds) {
                auto tileId = tileIdValue.getAsInteger();
                if (!tileId)
                    return failure();
                participants.push_back(*tileId);
                tileParticipates |= *tileId == tile.hartId;
            }
            if (!tileParticipates)
                continue;
            if (foundGroup || participants.empty())
                return failure();

            foundGroup = true;
            state.addAttribute("collective_stage_id",
                               builder.getI64IntegerAttr(stage.stageId));
            state.addAttribute("collective_layer_index",
                               builder.getI64IntegerAttr(layerIndex));
            state.addAttribute(
                "collective_group_id",
                builder.getI64IntegerAttr(groupEntry.index()));
            state.addAttribute(
                "collective_participants",
                builder.getDenseI64ArrayAttr(participants));
        }
        if (!foundGroup)
            return failure();
    }
    return builder.create(state)->getResult(0);
}

static FailureOr<SmallVector<Value>> emitSplitDescriptorKernel(
    Location loc, const llvm::json::Object &node,
    const llvm::json::Object &layer, const llvm::json::Array &inputs,
    const llvm::json::Array &outputs, const Tile &tile, const Stage &stage,
    OpBuilder &builder, llvm::DenseMap<int64_t, Value> &tileValues) {
    auto *payload = node.getObject("payload");
    auto *nodeOutputs = node.getArray("outputs");
    if (!payload || !nodeOutputs || nodeOutputs->size() != outputs.size())
        return failure();
    auto params = getDescriptorKernelParams("SPLIT", *payload);
    if (!params)
        return failure();

    SmallVector<Value> inputValues;
    for (const llvm::json::Value &input : inputs) {
        auto tensorId = getTensorId(input);
        Value value = tensorId ? tileValues.lookup(*tensorId) : Value{};
        if (!value)
            return failure();
        inputValues.push_back(value);
    }
    if (inputValues.size() != 1)
        return failure();
    auto inputType = dyn_cast<RankedTensorType>(inputValues.front().getType());
    if (!inputType)
        return failure();

    SmallVector<Type> outputTypes;
    SmallVector<SmallVector<int64_t>> outputFullOffsets;
    for (unsigned index = 0; index < outputs.size(); ++index) {
        auto *nodeOutput = (*nodeOutputs)[index].getAsObject();
        auto *output = outputs[index].getAsObject();
        auto outputType = nodeOutput && output
            ? parseTileOutputType(*nodeOutput, *output, tile, stage,
                                  inputType.getElementType())
            : std::nullopt;
        auto outputSlice = nodeOutput && output
            ? parseTileOutputSlice(*nodeOutput, *output, tile, stage)
            : std::nullopt;
        if (!outputType || !outputSlice)
            return failure();
        outputTypes.push_back(*outputType);
        SmallVector<int64_t> fullOffset;
        for (const SliceDim &dim : *outputSlice)
            fullOffset.push_back(dim.start);
        outputFullOffsets.push_back(std::move(fullOffset));
    }

    OperationState state(loc, maps::KernelOp::getOperationName());
    state.addOperands(inputValues);
    state.addTypes(outputTypes);
    state.addAttribute("kind", builder.getStringAttr("SPLIT"));
    state.addAttribute("params", builder.getDenseI64ArrayAttr(*params));
    for (auto indexed : llvm::enumerate(outputFullOffsets))
        state.addAttribute(
            (Twine("output_full_offset_") + Twine(indexed.index())).str(),
            builder.getDenseI64ArrayAttr(indexed.value()));
    if (auto deviceName = layer.getString("device_name"))
        state.addAttribute("device_name", builder.getStringAttr(*deviceName));
    Operation *kernel = builder.create(state);
    return SmallVector<Value>(kernel->getResults());
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


static std::optional<SmallVector<SliceDim>> parseSliceDimArray(const llvm::json::Array &dims) {
    SmallVector<SliceDim> result;

    for (const llvm::json::Value &dimValue : dims) {
        auto *dim = dimValue.getAsObject();
        if (!dim)
            return std::nullopt;
        auto start = dim->getInteger("start");
        auto length = dim->getInteger("length");
        if (!start || !length)
            return std::nullopt;

        result.push_back({*start, *length});
    }
    return result;
}


static std::optional<SmallVector<SliceDim>> parseSliceDims(const llvm::json::Object &slice) {
    auto *dims = slice.getArray("dims");
    if (!dims)
        return std::nullopt;
    return parseSliceDimArray(*dims);
}


static DenseI64ArrayAttr getSliceDimsAttr(OpBuilder &builder,
                                          ArrayRef<SliceDim> dims) {
    SmallVector<int64_t> flattened;
    flattened.reserve(dims.size() * 2);
    for (const SliceDim &dim : dims) {
        flattened.push_back(dim.start);
        flattened.push_back(dim.length);
    }
    return builder.getDenseI64ArrayAttr(flattened);
}


static void setTransferSubSliceAttrs(Operation *op,
                                       const TileSourceChannelBinding &binding,
                                       OpBuilder &builder) {
    if (!binding.hasRelativeSubSlices)
        return;

    op->setAttr("src_parent_slice", getSliceDimsAttr(builder, binding.srcParentDims));
    op->setAttr("src_subslice", getSliceDimsAttr(builder, binding.srcDims));
    op->setAttr("dst_parent_slice", getSliceDimsAttr(builder, binding.dstParentDims));
    op->setAttr("dst_subslice", getSliceDimsAttr(builder, binding.dstDims));
}


static std::optional<std::pair<SmallVector<SliceDim>, SmallVector<SliceDim>>>
parseTensorSubSlice(const llvm::json::Object &subslice) {
    auto *parent = subslice.getObject("parent");
    auto *dims = subslice.getArray("dims");
    if (!parent || !dims)
        return std::nullopt;

    auto parentDims = parseSliceDims(*parent);
    auto relativeDims = parseSliceDimArray(*dims);
    if (!parentDims || !relativeDims)
        return std::nullopt;

    return std::make_pair(*parentDims, *relativeDims);
}



static std::optional<SmallVector<L2SourceChannelBinding>>
getInitializerChannelBindings(const llvm::json::Array &stages) {
    SmallVector<L2SourceChannelBinding> result;

    for (auto stageEntry : llvm::enumerate(stages)) {
        auto *stage = stageEntry.value().getAsObject();
        auto *layers = stage->getArray("layers");

        for (auto layerEntry : llvm::enumerate(*layers)) {
            auto *layer = layerEntry.value().getAsObject();
            auto *inputs = layer->getArray("inputs");

            for (auto inputEntry : llvm::enumerate(*inputs)) {
                auto *input = inputEntry.value().getAsObject();
                auto tensorId = input->getInteger("tensor_id");
                auto *source = input->getObject("source");
                if (source->getString("kind") != "INITIALIZER")
                    continue;

                auto *destinations = source->getArray("destinations");
                for (auto destinationEntry : llvm::enumerate(*destinations)) {
                    auto *destination =
                        destinationEntry.value().getAsObject();
                    auto tileId = destination->getInteger("tile_id");
                    auto *tensorSlice =
                        destination->getObject("tensor_slice");
                    auto dims = parseSliceDims(*tensorSlice);
                    if (!tileId || !dims)
                        return std::nullopt;

                    std::string name =
                        "initializer_stage_" +
                        std::to_string(stageEntry.index()) + "_layer_" +
                        std::to_string(layerEntry.index()) + "_input_" +
                        std::to_string(inputEntry.index()) + "_destination_" +
                        std::to_string(destinationEntry.index());
                    result.push_back(L2SourceChannelBinding{
                        name,
                        name + "_slot",
                        *tensorId,
                        -1,
                        *tileId,
                        *dims,
                        *dims,
                        -1,
                        static_cast<int64_t>(stageEntry.index()),
                        static_cast<int64_t>(layerEntry.index()),
                        static_cast<int64_t>(inputEntry.index()),
                    });
                }
            }
        }
    }

    return result;
}



struct ExecutionPlanChannelBindings {
    SmallVector<L2SourceChannelBinding> inputs;
    TileSourceChannelBindingVector intermediates;
    TileSourceChannelBindingVector outputs;
};

static std::optional<ExecutionPlanChannelBindings>
getExecutionPlanChannelBindings(const llvm::json::Array &transitions,
                                const llvm::json::Array &tensors) {
    ExecutionPlanChannelBindings result;

    for (auto transitionEntry : llvm::enumerate(transitions)) {
        auto *transition = transitionEntry.value().getAsObject();
        auto kind = transition->getString("kind");
        auto tensorId = transition->getInteger("tensor_id");
        if (!kind || !tensorId)
            return std::nullopt;

        std::string transitionName =
            "transition_" + std::to_string(transitionEntry.index());
        if (*kind == "INPUT") {
            auto destinationStageId =
                transition->getInteger("destination_stage_id");
            auto *destinations = transition->getArray("destinations");
            if (!destinationStageId || !destinations)
                return std::nullopt;
            for (auto destinationEntry : llvm::enumerate(*destinations)) {
                auto *destination = destinationEntry.value().getAsObject();
                auto tileId = destination->getInteger("tile_id");
                auto *tensorSlice =
                    destination->getObject("tensor_slice");
                auto dims = tensorSlice
                    ? parseSliceDims(*tensorSlice)
                    : std::nullopt;
                if (!tileId || !dims)
                    return std::nullopt;

                result.inputs.push_back(L2SourceChannelBinding{
                    transitionName + "_destination_" +
                        std::to_string(destinationEntry.index()),
                    "",
                    *tensorId,
                    -1,
                    *tileId,
                    *dims,
                    *dims,
                    static_cast<int64_t>(transitionEntry.index()),
                    *destinationStageId,
                });
            }
            continue;
        }

        if (*kind == "INTERMEDIATE") {
            auto sourceStageId =
                transition->getInteger("source_stage_id");
            auto destinationStageId =
                transition->getInteger("destination_stage_id");
            auto *transfers = transition->getArray("transfers");
            if (!sourceStageId || !destinationStageId || !transfers)
                return std::nullopt;
            for (auto transferEntry : llvm::enumerate(*transfers)) {
                auto *transfer = transferEntry.value().getAsObject();
                auto sourceTileId =
                    transfer->getInteger("source_tile_id");
                auto destinationTileId =
                    transfer->getInteger("destination_tile_id");
                auto *sourceSubSlice =
                    transfer->getObject("source_subslice");
                auto *destinationSubSlice =
                    transfer->getObject("destination_subslice");
                auto parsedSource = sourceSubSlice
                    ? parseTensorSubSlice(*sourceSubSlice)
                    : std::nullopt;
                auto parsedDestination = destinationSubSlice
                    ? parseTensorSubSlice(*destinationSubSlice)
                    : std::nullopt;
                if (!sourceTileId || !destinationTileId ||
                    !parsedSource || !parsedDestination)
                    return std::nullopt;

                result.intermediates.push_back(TileSourceChannelBinding{
                    transitionName + "_transfer_" +
                        std::to_string(transferEntry.index()),
                    *tensorId,
                    *sourceTileId,
                    *destinationTileId,
                    parsedSource->second,
                    parsedDestination->second,
                    parsedSource->first,
                    parsedDestination->first,
                    true,
                    static_cast<int64_t>(transitionEntry.index()),
                    *sourceStageId,
                    *destinationStageId,
                });
            }
            continue;
        }

        if (*kind == "OUTPUT") {
            auto sourceStageId =
                transition->getInteger("source_stage_id");
            auto *sources = transition->getArray("sources");
            if (!sourceStageId || !sources)
                return std::nullopt;
            auto *tensor = tensors[*tensorId].getAsObject();
            auto tensorName = tensor ? tensor->getString("name") : std::nullopt;
            if (!tensorName)
                return std::nullopt;
            for (auto sourceEntry : llvm::enumerate(*sources)) {
                auto *source = sourceEntry.value().getAsObject();
                auto tileId = source->getInteger("tile_id");
                auto *tensorSlice = source->getObject("tensor_slice");
                auto dims = tensorSlice
                    ? parseSliceDims(*tensorSlice)
                    : std::nullopt;
                if (!tileId || !dims)
                    return std::nullopt;

                result.outputs.push_back(TileSourceChannelBinding{
                    transitionName + "_source_" +
                        std::to_string(sourceEntry.index()),
                    *tensorId,
                    *tileId,
                    -1,
                    *dims,
                    *dims,
                    *dims,
                    *dims,
                    false,
                    static_cast<int64_t>(transitionEntry.index()),
                    *sourceStageId,
                    -1,
                    tensorName->str(),
                });
            }
            continue;
        }

        return std::nullopt;
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
        if (!submesh)
            return std::nullopt;

        llvm::DenseSet<int64_t> tileIds;
        llvm::DenseMap<int64_t, int64_t> physicalToVirtualTileId;

        auto *virtualToPhysical = stage->getObject("virtual_to_physical");
        if (virtualToPhysical) {
            for (const auto &entry : *virtualToPhysical) {
                int64_t virtualTileId;
                StringRef virtualTileIdString = entry.first;
                if (virtualTileIdString.getAsInteger(10, virtualTileId))
                    return std::nullopt;
                auto physicalTileId = entry.second.getAsInteger();
                if (!physicalTileId)
                    return std::nullopt;
                tileIds.insert(*physicalTileId);
                physicalToVirtualTileId[*physicalTileId] = virtualTileId;
            }
        } else if (auto *submeshTileIds = submesh->getArray("tile_ids")) {
            for (const llvm::json::Value &tileIdValue : *submeshTileIds) {
                auto tileId = tileIdValue.getAsInteger();
                if (!tileId)
                    return std::nullopt;
                tileIds.insert(*tileId);
            }
        }

        bool hasExplicitTileIds = !tileIds.empty();
        int64_t x0Value = 0;
        int64_t y0Value = 0;
        int64_t widthValue = 0;
        int64_t heightValue = 0;
        if (!hasExplicitTileIds) {
            auto x0 = submesh->getInteger("x0");
            auto y0 = submesh->getInteger("y0");
            auto width = submesh->getInteger("width");
            auto height = submesh->getInteger("height");
            if (!x0 || !y0 || !width || !height)
                return std::nullopt;
            x0Value = *x0;
            y0Value = *y0;
            widthValue = *width;
            heightValue = *height;
        }

        result.push_back(Stage{
            static_cast<int64_t>(indexed.index()),
            symName,
            x0Value,
            y0Value,
            widthValue,
            heightValue,
            hasExplicitTileIds,
            std::move(tileIds),
            std::move(physicalToVirtualTileId),
        });
    }

    return result;
}


static std::optional<llvm::SetVector<int64_t>>
collectInputTensorIds(const llvm::json::Array &stages,
                      const llvm::json::Array &transitions) {

    llvm::SetVector<int64_t> ids;

    for (const llvm::json::Value &transitionValue : transitions) {
        auto *transition = transitionValue.getAsObject();
        if (transition->getString("kind") == "INPUT")
            ids.insert(*transition->getInteger("tensor_id"));
    }

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

            if (source->getString("kind") == "INITIALIZER")
                ids.insert(*tensorId);
            }
        }
    }

    return ids;
}

static OwningOpRef<Operation *>
importExecutionPlanDocument(StringRef input, MLIRContext *ctx) {

    ctx->loadDialect<maps::MapsDialect, 
        func::FuncDialect,
        tensor::TensorDialect,
        linalg::LinalgDialect,
        math::MathDialect,
        scf::SCFDialect,
        arith::ArithDialect>();

    auto json = llvm::json::parse(input);
    if (!json) {
        llvm::errs() << llvm::toString(json.takeError()) << "\n";
        return {};
    }

    auto *root = json->getAsObject();
    auto *tensors = root->getArray("tensors");
    auto *stages = root->getArray("stages");
    auto *transitions = root->getArray("transitions");

    std::optional<StringRef> netName = root->getString("name");

    auto inputTensorIds = *collectInputTensorIds(*stages, *transitions);

    OpBuilder builder(ctx);
    Location loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);
    module->setAttr("maps.name", builder.getStringAttr(*netName));
    if (auto target = root->getString("target"))
        module->setAttr("maps.target", builder.getStringAttr(*target));

    auto *execution = root->getObject("execution");
    auto numTokenSlots = execution
        ? execution->getInteger("num_token_slots")
        : std::optional<int64_t>();
    if (!numTokenSlots || *numTokenSlots <= 0) {
        llvm::errs() << "missing or invalid execution.num_token_slots\n";
        return {};
    }
    module->setAttr("maps.num_token_slots",
                    builder.getI64IntegerAttr(*numTokenSlots));

    if (auto *bundle = root->getObject("bundle")) {
        DictionaryAttr bundleAttr = parseJsonObject(*bundle, builder);
        if (!bundleAttr) {
            llvm::errs() << "invalid deployment bundle metadata\n";
            return {};
        }
        module->setAttr("maps.bundle", bundleAttr);
    }

    if (auto *mesh = root->getObject("mesh")) {
        if (auto width = mesh->getInteger("width"))
            module->setAttr("maps.mesh_width",
                            builder.getI64IntegerAttr(*width));
        if (auto height = mesh->getInteger("height"))
            module->setAttr("maps.mesh_height",
                            builder.getI64IntegerAttr(*height));
        if (auto *l2 = mesh->getObject("l2_memory")) {
            if (auto capacity = l2->getInteger("size"))
                module->setAttr("maps.l2_capacity_bytes",
                                builder.getI64IntegerAttr(*capacity));
        }
    }

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

    for (auto indexed : llvm::enumerate(inputTensorIds)) {
        auto *tensor = (*tensors)[indexed.value()].getAsObject();
        if (!tensor)
            return {};
        if (auto name = tensor->getString("name"))
            mainFunc.setArgAttr(indexed.index(), "maps.global_name",
                                builder.getStringAttr(*name));
        if (auto *initializer = tensor->getObject("initializer")) {
            DictionaryAttr initializerAttr = parseJsonObject(*initializer, builder);
            if (!initializerAttr) {
                llvm::errs() << "invalid initializer metadata for tensor "
                             << indexed.value() << "\n";
                return {};
            }
            mainFunc.setArgAttr(indexed.index(), "maps.initializer",
                                initializerAttr);
        }
    }

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

    auto l2SourceBindings = getInitializerChannelBindings(*stages);
    auto executionPlanBindings =
        getExecutionPlanChannelBindings(*transitions, *tensors);
    if (!l2SourceBindings || !executionPlanBindings)
        return {};
    l2SourceBindings->append(
        executionPlanBindings->inputs.begin(),
        executionPlanBindings->inputs.end());
    TileSourceChannelBindingVector intermediateBindings =
        std::move(executionPlanBindings->intermediates);
    TileSourceChannelBindingVector outputBindings =
        std::move(executionPlanBindings->outputs);

    // map input tensor ids to main function arguments
    llvm::DenseMap<int64_t, Value> inputTensorValues;
    for (auto indexed : llvm::enumerate(inputTensorIds)) {
        int64_t argIndex = indexed.index();
        int64_t tensorId = indexed.value();
        inputTensorValues[tensorId] = mainEntry->getArgument(argIndex);
    }

    // extract slices from input tensors for each binding
    for (const L2SourceChannelBinding &binding : *l2SourceBindings) {
        auto *tensor = (*tensors)[binding.tensorId].getAsObject();
        if (!isInitializerTensor(*tensor))
            continue;
        Value source = inputTensorValues.lookup(binding.tensorId);
        
        SmallVector<int64_t> shape;
        SmallVector<OpFoldResult> offsets;
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides;

        for (const SliceDim &dim : binding.srcDims) {
            shape.push_back(dim.length);
            offsets.push_back(builder.getIndexAttr(dim.start));
            sizes.push_back(builder.getIndexAttr(dim.length));
            strides.push_back(builder.getIndexAttr(1));
        }


        auto sourceType = cast<RankedTensorType>(source.getType());
        auto sliceType = RankedTensorType::get(shape, sourceType.getElementType());

        // create extract_slice op for the binding, considering tile usage
        Value slice = tensor::ExtractSliceOp::create(
            builder, 
            loc, 
            sliceType, 
            source, 
            offsets, 
            sizes, 
            strides
        );

        // create send op for initializer bindings
        maps::SendOp::create(
            builder, 
            loc, 
            slice, 
            SymbolRefAttr::get(ctx, binding.channelName),
            builder.getI64IntegerAttr(binding.srcHartId),
            builder.getI64IntegerAttr(binding.dstHartId)
        );
    }

    // start of tiles receiving init data
    auto *mesh = root->getObject("mesh");
    auto *tiles = mesh->getArray("tiles");
    auto tileObjs = getTileObjs(*tiles);
    auto stageObjs = getStageObjs(*stages);
    for (const Tile &tileObj : *tileObjs) {
        bool receivesInitializer = llvm::any_of(*l2SourceBindings, [&](const L2SourceChannelBinding &binding) {
            auto *tensor = (*tensors)[binding.tensorId].getAsObject();
            return binding.dstHartId == tileObj.hartId &&
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

        // check which tiles receive which bindings
        for (const L2SourceChannelBinding &binding : *l2SourceBindings) {
            if (binding.dstHartId != tileObj.hartId)
                continue; // tile doesn't receive this binding

            auto *tensor = (*tensors)[binding.tensorId].getAsObject();
            if (!isInitializerTensor(*tensor))
                continue;

            SmallVector<int64_t> shape;
            for (const SliceDim &dim : binding.srcDims)
                shape.push_back(dim.length);

            Type elementType = parseElementType(*tensor, builder);
            Type resultType = RankedTensorType::get(shape, elementType);

            // create recv op for the binding
            auto recv = maps::RecvOp::create(
                builder,
                loc,
                resultType,
                StringRef(binding.channelName),
                binding.srcHartId,
                binding.dstHartId
            );

            // store result
            maps::StoreOp::create(
                builder,
                loc,
                recv.getResult(),
                SymbolRefAttr::get(ctx, binding.slotName)
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
        for (const L2SourceChannelBinding &binding : *l2SourceBindings) {
            auto *tensor = (*tensors)[binding.tensorId].getAsObject();
            if (isInitializerTensor(*tensor))
                continue;
            if (binding.destinationStageId >= 0 &&
                binding.destinationStageId != stageObj.stageId)
                continue;

            bool isBindingInStage = false;
            for (const Tile &tileObj : *tileObjs) {
                if (binding.dstHartId == tileObj.hartId &&
                    isTileInStage(tileObj, stageObj)) {
                    isBindingInStage = true;
                    break;
                }
            }
            if (!isBindingInStage)
                continue;

            Value source = inputTensorValues.lookup(binding.tensorId);

            SmallVector<int64_t> shape;
            SmallVector<OpFoldResult> offsets;
            SmallVector<OpFoldResult> sizes;
            SmallVector<OpFoldResult> strides;

            for (const SliceDim &dim : binding.srcDims) {
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
                SymbolRefAttr::get(ctx, binding.channelName),
                builder.getI64IntegerAttr(binding.srcHartId),
                builder.getI64IntegerAttr(binding.dstHartId)
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
            llvm::DenseMap<int64_t, SmallVector<std::pair<const TileSourceChannelBinding *, Value>>> transitionValues;
            llvm::DenseMap<int64_t, Value> transitionInputValues;

            // receive intermediate channel bindings from producer tiles
            for (const TileSourceChannelBinding &binding : intermediateBindings) {
                if (binding.dstHartId != tileObj.hartId)
                    continue;
                if (binding.destinationStageId >= 0 &&
                    binding.destinationStageId != stageObj.stageId)
                    continue;

                auto *tensor = (*tensors)[binding.tensorId].getAsObject();
                SmallVector<int64_t> shape;
                for (const SliceDim &dim : binding.srcDims)
                    shape.push_back(dim.length);

                Type elementType = parseElementType(*tensor, builder);
                Type resultType = RankedTensorType::get(shape, elementType);

                auto recv = maps::RecvOp::create(
                    builder,
                    loc,
                    resultType,
                    StringRef(binding.channelName),
                    binding.srcHartId,
                    binding.dstHartId
                );
                setTransferSubSliceAttrs(recv.getOperation(), binding, builder);
                Value transitionInput = recv.getResult();
                int64_t bindingId = binding.transitionId >= 0
                    ? binding.transitionId
                    : binding.tensorId;
                transitionValues[bindingId].push_back(
                    {&binding, transitionInput});
            }

            for (auto &entry : transitionValues) {
                SmallVector<std::pair<const TileSourceChannelBinding *, Value>> &values = entry.second;
                if (values.size() == 1) {
                    const TileSourceChannelBinding &binding =
                        *values.front().first;
                    if (binding.transitionId >= 0)
                        transitionInputValues[binding.transitionId] =
                            values.front().second;
                    else
                        tileValues[binding.tensorId] =
                            values.front().second;
                    continue;
                }

                TileSourceChannelBindingVector bindings;
                bindings.reserve(values.size());
                for (const auto &value : values) {
                    bindings.push_back(*value.first);
                }

                SmallVector<int64_t> starts;
                SmallVector<int64_t> ends;

                for (const SliceDim &dim : bindings.front().dstDims) {
                    starts.push_back(dim.start);
                    ends.push_back(dim.start + dim.length);
                }

                for (const TileSourceChannelBinding &binding : bindings) {
                    for (auto indexed : llvm::enumerate(binding.dstDims)) {
                        starts[indexed.index()] =
                            std::min(starts[indexed.index()], indexed.value().start);

                        ends[indexed.index()] =
                            std::max(ends[indexed.index()],
                                    indexed.value().start + indexed.value().length);
                    }
                }

                auto firstType = cast<RankedTensorType>(values.front().second.getType());

                SmallVector<int64_t> assembledShape;
                for (auto indexed : llvm::enumerate(starts)) {
                    assembledShape.push_back(ends[indexed.index()] - indexed.value());
                }

                Value assembled = tensor::EmptyOp::create(
                    builder,
                    loc,
                    assembledShape,
                    firstType.getElementType()
                ).getResult();

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
                        builder,
                        loc,
                        value.second,
                        assembled,
                        offsets,
                        sizes,
                        strides
                    ).getResult();
                }

                const TileSourceChannelBinding &binding = bindings.front();
                if (binding.transitionId >= 0)
                    transitionInputValues[binding.transitionId] = assembled;
                else
                    tileValues[binding.tensorId] = assembled;
                continue;
            }

            // load stored values and receive runtime inputs
            for (const L2SourceChannelBinding &binding : *l2SourceBindings) {
                if (binding.dstHartId != tileObj.hartId)
                    continue;
                if (binding.destinationStageId >= 0 &&
                    binding.destinationStageId != stageObj.stageId)
                    continue;

                auto *tensor = (*tensors)[binding.tensorId].getAsObject();
                if (isInitializerTensor(*tensor)) {
                    if (binding.destinationLayerIndex >= 0)
                        continue;
                    // load init bindings
                    SmallVector<int64_t> shape;
                    for (const SliceDim &dim : binding.srcDims)
                        shape.push_back(dim.length);

                    Type elementType = parseElementType(*tensor, builder);
                    Type resultType = RankedTensorType::get(shape, elementType);

                    Value localValue = maps::LoadOp::create(
                        builder,
                        loc,
                        resultType,
                        SymbolRefAttr::get(ctx, binding.slotName)
                    ).getResult();
                    tileValues[binding.tensorId] = localValue;
                } else {
                    // receive inputs
                    SmallVector<int64_t> shape;
                    for (const SliceDim &dim : binding.srcDims)
                        shape.push_back(dim.length);
            
                    Type elementType = parseElementType(*tensor, builder);
                    Type resultType = RankedTensorType::get(shape, elementType);

                    Value runtimeInput = maps::RecvOp::create(
                        builder,
                        loc,
                        resultType,
                        StringRef(binding.channelName),
                        binding.srcHartId,
                        binding.dstHartId
                    ).getResult();
                    if (binding.transitionId >= 0)
                        transitionInputValues[binding.transitionId] =
                            runtimeInput;
                    else
                        tileValues[binding.tensorId] = runtimeInput;
                }
            }
            
            // emit tile core ops
            auto *stageJson = (*stages)[stageObj.stageId].getAsObject();
            auto *layers = stageJson->getArray("layers");

            for (auto layerEntry : llvm::enumerate(*layers)) {
                auto *layer = layerEntry.value().getAsObject();
                auto *node = layer->getObject("node");
                auto *inputs = layer->getArray("inputs");
                auto *outputs = layer->getArray("outputs");
                auto nodeKind = node->getInteger("kind");
                if (!nodeKind || !inputs || !outputs || outputs->empty()) {
                    llvm::errs() << "unsupported MAPS layer: missing node inputs or outputs\n";
                    return {};
                }

                for (auto inputEntry : llvm::enumerate(*inputs)) {
                    auto *input = inputEntry.value().getAsObject();
                    auto tensorId = input->getInteger("tensor_id");
                    auto *source = input->getObject("source");
                    auto sourceKind = source->getString("kind");
                    if (sourceKind == "TRANSITION") {
                        auto transitionId =
                            source->getInteger("transition_id");
                        Value transitionInput = transitionId
                            ? transitionInputValues.lookup(*transitionId)
                            : Value{};
                        if (!tensorId || !transitionInput) {
                            llvm::errs()
                                << "transition input is unavailable on stage "
                                << stageObj.stageId << " tile "
                                << tileObj.hartId << "\n";
                            return {};
                        }
                        tileValues[*tensorId] = transitionInput;
                        continue;
                    }
                    if (sourceKind != "INITIALIZER")
                        continue;

                    const L2SourceChannelBinding *initializer = nullptr;
                    for (const L2SourceChannelBinding &binding : *l2SourceBindings) {
                        if (binding.dstHartId == tileObj.hartId &&
                            binding.destinationStageId == stageObj.stageId &&
                            binding.destinationLayerIndex ==
                                static_cast<int64_t>(layerEntry.index()) &&
                            binding.destinationInputIndex ==
                                static_cast<int64_t>(inputEntry.index())) {
                            initializer = &binding;
                            break;
                        }
                    }
                    if (!tensorId || !initializer) {
                        llvm::errs()
                            << "initializer input is unavailable on stage "
                            << stageObj.stageId << " tile "
                            << tileObj.hartId << "\n";
                        return {};
                    }
                    auto *tensor =
                        (*tensors)[initializer->tensorId].getAsObject();
                    SmallVector<int64_t> shape;
                    for (const SliceDim &dim : initializer->srcDims)
                        shape.push_back(dim.length);
                    Type elementType = parseElementType(*tensor, builder);
                    Type resultType =
                        RankedTensorType::get(shape, elementType);
                    tileValues[*tensorId] = maps::LoadOp::create(
                        builder,
                        loc,
                        resultType,
                        SymbolRefAttr::get(ctx, initializer->slotName)
                    ).getResult();
                }

                auto outputTensorId = getTensorId((*outputs)[0]);
                if (!outputTensorId) {
                    llvm::errs() << "unsupported MAPS layer: output has no tensor id\n";
                    return {};
                }

                if (*nodeKind == 0) {
                    auto *payload = node->getObject("payload");
                    bool transposeW =
                        payload &&
                        payload->getBoolean("transpose_w").value_or(false);
                    auto deviceName = layer->getString("device_name");
                    bool useDescriptor = deviceName && *deviceName == "spatz";
                    if (useDescriptor || transposeW || inputs->size() == 3) {
                        auto *output = (*outputs)[0].getAsObject();
                        auto kernel = output
                            ? emitDescriptorKernel(
                                  loc, *node, *layer, layerEntry.index(),
                                  *inputs, *output, tileObj,
                                  stageObj, builder, tileValues)
                            : FailureOr<Value>(failure());
                        if (failed(kernel)) {
                            llvm::errs()
                                << "unsupported Matmul descriptor\n";
                            return {};
                        }
                        tileValues[*outputTensorId] = *kernel;
                        continue;
                    }

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
                    auto workKind =
                        payload ? payload->getString("work_kind") : std::nullopt;
                    auto *nodeOutputs = node->getArray("outputs");
                    auto *nodeOutput = nodeOutputs && !nodeOutputs->empty()
                                           ? (*nodeOutputs)[0].getAsObject()
                                           : nullptr;
                    auto *output = (*outputs)[0].getAsObject();
                    if (workKind == "SPLIT") {
                        auto kernels = emitSplitDescriptorKernel(
                            loc, *node, *layer, *inputs, *outputs, tileObj,
                            stageObj, builder, tileValues);
                        if (failed(kernels) || kernels->size() != outputs->size()) {
                            llvm::errs() << "unsupported Split descriptor\n";
                            return {};
                        }
                        for (unsigned index = 0; index < outputs->size(); ++index) {
                            auto tensorId = getTensorId((*outputs)[index]);
                            if (!tensorId) {
                                llvm::errs() << "Split output is missing tensor id\n";
                                return {};
                            }
                            tileValues[*tensorId] = (*kernels)[index];
                        }
                        continue;
                    }
                    if ((workKind &&
                        (*workKind == "RESHAPE" || *workKind == "SLICE" ||
                         *workKind == "TRANSPOSE")) ||
                        (opName && (*opName == "Im2Col" ||
                                    *opName == "OutputReformat"))) {
                        auto kernel = output
                            ? emitDescriptorKernel(
                                  loc, *node, *layer, layerEntry.index(),
                                  *inputs, *output, tileObj,
                                  stageObj, builder, tileValues)
                            : FailureOr<Value>(failure());
                        if (failed(kernel)) {
                            llvm::errs()
                                << "unsupported transform descriptor work_kind="
                                << *workKind << "\n";
                            return {};
                        }
                        tileValues[*outputTensorId] = *kernel;
                        continue;
                    }
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
                        parseTileOutputType(
                            *nodeOutput, *output, tileObj, stageObj, elementType);
                    if (!outputType) {
                        llvm::errs() << "unsupported transform MAPS node: invalid output "
                                        "layout\n";
                        return {};
                    }

                    auto transformed = emitTransformAsStandardOps(
                        loc, *opName, inputValues, *payload, *outputType, builder);
                    if (failed(transformed)) {
                        llvm::errs() << "unsupported transform MAPS node op_name="
                                     << *opName << "\n";
                        return {};
                    }
                    tileValues[*outputTensorId] = *transformed;
                    continue;
                }

                if (*nodeKind == 3) {
                    auto *payload = node->getObject("payload");
                    auto workKind =
                        payload ? payload->getString("work_kind") : std::nullopt;
                    auto *nodeInputs = node->getArray("inputs");
                    auto *nodeOutputs = node->getArray("outputs");
                    auto *nodeOutput =
                        nodeOutputs && !nodeOutputs->empty()
                            ? (*nodeOutputs)[0].getAsObject()
                            : nullptr;
                    auto *output = (*outputs)[0].getAsObject();
                    if (workKind == "DEPTHWISE_CONV" || workKind == "MAX_POOL") {
                        auto kernel = output
                            ? emitDescriptorKernel(
                                  loc, *node, *layer, layerEntry.index(),
                                  *inputs, *output, tileObj,
                                  stageObj, builder, tileValues)
                            : FailureOr<Value>(failure());
                        if (failed(kernel)) {
                            llvm::errs()
                                << "unsupported DepthwiseConv descriptor\n";
                            return {};
                        }
                        tileValues[*outputTensorId] = *kernel;
                        continue;
                    }
                    if (!payload || workKind != "CONV2D" || !nodeInputs ||
                        nodeInputs->size() < 2 || nodeInputs->size() > 3 ||
                        !nodeOutput || !output ||
                        inputs->size() != nodeInputs->size()) {
                        llvm::errs() << "unsupported Conv2D node: invalid metadata\n";
                        return {};
                    }

                    SmallVector<Value> inputValues;
                    for (const llvm::json::Value &input : *inputs) {
                        auto tensorId = getTensorId(input);
                        Value value =
                            tensorId ? tileValues.lookup(*tensorId) : Value{};
                        if (!value) {
                            llvm::errs()
                                << "unsupported Conv2D node: missing local input value\n";
                            return {};
                        }
                        inputValues.push_back(value);
                    }

                    auto inputType =
                        dyn_cast<RankedTensorType>(inputValues[0].getType());
                    auto weightType =
                        dyn_cast<RankedTensorType>(inputValues[1].getType());
                    auto outputType = inputType
                        ? parseTileOutputType(
                              *nodeOutput, *output, tileObj, stageObj,
                              inputType.getElementType())
                        : std::nullopt;
                    auto outputSlice =
                        parseTileOutputSlice(
                            *nodeOutput, *output, tileObj, stageObj);
                    auto strides = parseIntArrayField(*payload, "strides");
                    auto dilations = parseIntArrayField(*payload, "dilations");
                    auto pads = parseIntArrayField(*payload, "pads");
                    auto *globalInput = (*nodeInputs)[0].getAsObject();
                    auto *globalWeight = (*nodeInputs)[1].getAsObject();
                    auto globalInputDims =
                        globalInput
                            ? parseIntArrayField(*globalInput, "dims")
                            : std::nullopt;
                    auto globalWeightDims =
                        globalWeight
                            ? parseIntArrayField(*globalWeight, "dims")
                            : std::nullopt;
                    if (!inputType || !weightType || inputType.getRank() != 4 ||
                        weightType.getRank() != 4 || !outputType ||
                        outputType->getRank() != 4 || !outputSlice ||
                        outputSlice->size() != 4 || !strides ||
                        strides->size() != 2 || !dilations ||
                        dilations->size() != 2 || !pads || pads->size() != 4 ||
                        !globalInputDims || globalInputDims->size() != 4 ||
                        !globalWeightDims || globalWeightDims->size() != 4) {
                        llvm::errs() << "unsupported Conv2D node: invalid geometry\n";
                        return {};
                    }
                    if (!isa<FloatType>(inputType.getElementType()) ||
                        weightType.getElementType() != inputType.getElementType() ||
                        outputType->getElementType() != inputType.getElementType() ||
                        (inputValues.size() == 3 &&
                         cast<RankedTensorType>(inputValues[2].getType())
                                 .getElementType() != inputType.getElementType())) {
                        llvm::errs() << "unsupported Conv2D node: inputs and output "
                                        "must use one floating-point type\n";
                        return {};
                    }

                    const SliceDim outputH = (*outputSlice)[2];
                    const SliceDim outputW = (*outputSlice)[3];
                    int64_t theoreticalHStart =
                        outputH.start * (*strides)[0] - (*pads)[0];
                    int64_t theoreticalWStart =
                        outputW.start * (*strides)[1] - (*pads)[1];
                    int64_t theoreticalHEnd = theoreticalHStart;
                    int64_t theoreticalWEnd = theoreticalWStart;
                    if (outputH.length > 0) {
                        theoreticalHEnd =
                            (outputH.start + outputH.length - 1) *
                                (*strides)[0] -
                            (*pads)[0] +
                            (*dilations)[0] * ((*globalWeightDims)[2] - 1) +
                            1;
                    }
                    if (outputW.length > 0) {
                        theoreticalWEnd =
                            (outputW.start + outputW.length - 1) *
                                (*strides)[1] -
                            (*pads)[1] +
                            (*dilations)[1] * ((*globalWeightDims)[3] - 1) +
                            1;
                    }
                    SmallVector<int64_t> localPadding = {
                        std::max<int64_t>(0, -theoreticalHStart),
                        std::max<int64_t>(0, -theoreticalWStart),
                        std::max<int64_t>(
                            0, theoreticalHEnd - (*globalInputDims)[2]),
                        std::max<int64_t>(
                            0, theoreticalWEnd - (*globalInputDims)[3]),
                    };

                    SmallVector<int64_t> params = {
                        (*strides)[0], (*strides)[1],
                        (*dilations)[0], (*dilations)[1],
                    };
                    params.append(localPadding);

                    OperationState state(loc, maps::KernelOp::getOperationName());
                    state.addOperands(inputValues);
                    state.addTypes(*outputType);
                    state.addAttribute("kind", builder.getStringAttr("CONV2D"));
                    state.addAttribute(
                        "params", builder.getDenseI64ArrayAttr(params));
                    Operation *kernel = builder.create(state);
                    tileValues[*outputTensorId] = kernel->getResult(0);
                    continue;
                }

                if (*nodeKind == 2 || *nodeKind == 255) {
                    auto *output = (*outputs)[0].getAsObject();
                    auto kernel = output
                        ? emitDescriptorKernel(
                              loc, *node, *layer, layerEntry.index(), *inputs,
                              *output, tileObj, stageObj, builder, tileValues)
                        : FailureOr<Value>(failure());
                    if (failed(kernel)) {
                        llvm::errs() << "unsupported MAPS descriptor node kind "
                                     << *nodeKind << "\n";
                        return {};
                    }
                    tileValues[*outputTensorId] = *kernel;
                    continue;
                }

                if (*nodeKind != 1 || inputs->empty()) {
                    llvm::errs() << "unsupported MAPS node kind " << *nodeKind << "\n";
                    return {};
                }

                auto *payload = node->getObject("payload");
                auto opName = payload ? payload->getString("op_name") : std::nullopt;
                auto deviceName = layer->getString("device_name");
                if (deviceName == "spatz") {
                    auto *output = (*outputs)[0].getAsObject();
                    auto kernel = output
                        ? emitDescriptorKernel(
                              loc, *node, *layer, layerEntry.index(), *inputs,
                              *output, tileObj, stageObj, builder, tileValues)
                        : FailureOr<Value>(failure());
                    if (failed(kernel)) {
                        llvm::errs() << "unsupported Spatz-assigned operation\n";
                        return {};
                    }
                    tileValues[*outputTensorId] = *kernel;
                    continue;
                }
                if (!opName ||
                    (*opName != "Exp" && *opName != "Log" &&
                     *opName != "Add")) {
                    auto *output = (*outputs)[0].getAsObject();
                    auto kernel = output
                        ? emitDescriptorKernel(
                              loc, *node, *layer, layerEntry.index(), *inputs,
                              *output, tileObj, stageObj, builder, tileValues)
                        : FailureOr<Value>(failure());
                    if (succeeded(kernel)) {
                        tileValues[*outputTensorId] = *kernel;
                        continue;
                    }
                }
                if (!opName) {
                    llvm::errs() << "unsupported elementwise MAPS node: missing op_name\n";
                    return {};
                }

                if (*opName == "Exp" || *opName == "Log") {
                    auto inputTensorId = getTensorId((*inputs)[0]);
                    if (!inputTensorId) {
                        llvm::errs() << "unsupported " << *opName
                                     << " node: missing tensor id\n";
                        return {};
                    }

                    Value inputValue = tileValues.lookup(*inputTensorId);
                    if (!inputValue) {
                        llvm::errs() << "unsupported " << *opName
                                     << " node: missing local input value\n";
                        return {};
                    }

                    auto inputType = dyn_cast<RankedTensorType>(inputValue.getType());
                    if (!inputType) {
                        llvm::errs() << "unsupported " << *opName
                                     << " node: expected ranked tensor input\n";
                        return {};
                    }
                    Value result = *opName == "Exp"
                                       ? emitElementwiseExp(
                                             loc, inputValue, inputType, builder)
                                       : emitElementwiseLog(
                                             loc, inputValue, inputType, builder);
                    tileValues[*outputTensorId] = result;
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
                        *nodeOutput, *output, tileObj, stageObj,
                        lhsType.getElementType());
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

                    Value add = emitElementwiseAdd(
                        loc, lhs, rhs, *outputType, builder);
                    tileValues[*outputTensorId] = add;
                    continue;
                }

                llvm::errs() << "unsupported elementwise MAPS op " << *opName << "\n";
                return {};
            }

            // send intermediate channel bindings produced by this tile
            for (const TileSourceChannelBinding &binding : intermediateBindings) {
                if (binding.srcHartId != tileObj.hartId)
                    continue;
                if (binding.sourceStageId >= 0 &&
                    binding.sourceStageId != stageObj.stageId)
                    continue;

                Value source = tileValues.lookup(binding.tensorId);
                if (!source)
                    continue;

                SmallVector<int64_t> shape;
                SmallVector<OpFoldResult> offsets;
                SmallVector<OpFoldResult> sizes;
                SmallVector<OpFoldResult> strides;
                SmallVector<int64_t> localStarts =
                    getLocalStartsForSourceTile(intermediateBindings, binding);

                for (auto indexed : llvm::enumerate(binding.srcDims)) {
                    const SliceDim &dim = indexed.value();
                    int64_t localOffset = binding.hasRelativeSubSlices
                        ? dim.start
                        : dim.start - localStarts[indexed.index()];
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

                auto send = maps::SendOp::create(
                    builder,
                    loc,
                    slice,
                    SymbolRefAttr::get(ctx, binding.channelName),
                    builder.getI64IntegerAttr(binding.srcHartId),
                    builder.getI64IntegerAttr(binding.dstHartId)
                );
                setTransferSubSliceAttrs(send.getOperation(), binding, builder);
            }

            // send final output bindings back to L2
            for (const TileSourceChannelBinding &binding : outputBindings) {
                if (binding.srcHartId != tileObj.hartId)
                    continue;
                if (binding.sourceStageId >= 0 &&
                    binding.sourceStageId != stageObj.stageId)
                    continue;

                Value source = tileValues.lookup(binding.tensorId);
                if (!source)
                    continue;

                SmallVector<int64_t> shape;
                SmallVector<OpFoldResult> offsets;
                SmallVector<OpFoldResult> sizes;
                SmallVector<OpFoldResult> strides;
                SmallVector<int64_t> localStarts =
                    getLocalStartsForSourceTile(outputBindings, binding);

                for (auto indexed : llvm::enumerate(binding.srcDims)) {
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

                auto send = maps::SendOp::create(
                    builder,
                    loc,
                    slice,
                    SymbolRefAttr::get(ctx, binding.channelName),
                    builder.getI64IntegerAttr(binding.srcHartId),
                    builder.getI64IntegerAttr(binding.dstHartId)
                );
                send->setAttr("maps.global_name",
                              builder.getStringAttr(binding.globalName));
            }

            builder.setInsertionPointAfter(tile);
            // end of tile block
        }

        builder.setInsertionPointAfter(stage);
    }

    builder.setInsertionPointAfter(run);
    // end of run region

    // receive final output bindings from tiles and assemble them in L2
    llvm::DenseMap<int64_t, Value> outputTensorValues = OutputTensorInitValues;
    for (const TileSourceChannelBinding &binding : outputBindings) {
        auto *tensor = (*tensors)[binding.tensorId].getAsObject();
        SmallVector<int64_t> shape;
        SmallVector<OpFoldResult> offsets;
        SmallVector<OpFoldResult> sizes;
        SmallVector<OpFoldResult> strides;

        for (const SliceDim &dim : binding.dstDims) {
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
            StringRef(binding.channelName),
            binding.srcHartId,
            binding.dstHartId
        ).getResult();

        Value currentOutput = outputTensorValues.lookup(binding.tensorId);
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
        outputTensorValues[binding.tensorId] = updatedOutput;
    }

    // place main's return op at the end of the block
    func::ReturnOp::create(builder, loc);

    return OwningOpRef<Operation *>(module.getOperation());
}

OwningOpRef<Operation *> importExecutionPlan(StringRef input,
                                             MLIRContext *context) {
    return importExecutionPlanDocument(input, context);
}
} // namespace mlir::maps
