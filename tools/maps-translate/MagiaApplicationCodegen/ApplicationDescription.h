#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_DESCRIPTION_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_DESCRIPTION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <string>

namespace mlir::maps {

// MAGIA memory layout constants used by generated applications. The
// description keeps these values in one place so collection and emission agree on
// allocation size, alignment, and token buffering.
static constexpr uint64_t kMagiaL1UsableBytes = 896ull * 1024ull;
static constexpr uint64_t kGeneratedL1DataBaseOffset = 0ull;
static constexpr uint64_t kGeneratedL1AlignmentBytes = 16ull;
static constexpr uint64_t kGeneratedTaskScratchBytes = 0x10000ull;
static constexpr uint64_t kGeneratedReadyFlagsOffset = 0xD0000ull;
static constexpr unsigned kGeneratedReadyFlagsWords = 4096;
static constexpr unsigned kGeneratedDescriptorMaxDims = 6;
static constexpr unsigned kGeneratedOperationAbiVersion = 2;
static constexpr unsigned kGeneratedDescriptorAbiVersion = 1;
static constexpr unsigned kGeneratedKernelAbiVersion = 2;
static constexpr unsigned kGeneratedTaskBundleVersion = 2;
static constexpr uint64_t kMagiaV3L2BulkBytes = 0x0C000000ull;
static constexpr uint64_t kMagiaL2Base = 0xCC010000ull;
static constexpr uint64_t kMagiaL2End = 0xCCF10000ull;
static constexpr uint64_t kGeneratedL2AlignmentBytes = 16ull;
static constexpr unsigned kDefaultNumTokens = 1;

enum class GeneratedGlobalKind {
  Input,
  Output,
  Initializer,
  Intermediate,
};

enum class GeneratedOpKind {
  Matmul,
  Conv2D,
  DepthwiseConv,
  Abs,
  Add,
  Div,
  Log,
  Exp,
  SoftmaxExp,
  GroupNormalize,
  GroupReduce,
  GroupCenteredReduce,
  Im2Col,
  OutputReformat,
  Split,
  Mul,
  Neg,
  Pow,
  ReduceMax,
  ReduceSum,
  Relu,
  Reshape,
  Sigmoid,
  Slice,
  Sqrt,
  Sub,
  Transpose,
  AllReduceMax,
  AllReduceSum,
  Copy,
};

// A slice is the L1 allocation owned by a tile.  It points back to the logical
// global tensor and records the offset of this tile-local view.
struct GeneratedSlice {
  unsigned descriptionIndex = 0;
  unsigned sliceId = 0;
  unsigned globalId = 0;
  std::string globalName;
  GeneratedGlobalKind globalKind = GeneratedGlobalKind::Initializer;
  int64_t ownerHartid = -1;
  SmallVector<int64_t> fullOffset;
  SmallVector<int64_t> shape;
  Type elementType;
  unsigned elemBytes = 0;
  uint64_t l1OffsetBytes = 0;
  uint64_t slotBytes = 0;
  unsigned numSlots = 1;
  SmallVector<uint64_t> stridesBytes;
};

// Sub-slices describe copy endpoints inside an already allocated GeneratedSlice.
struct GeneratedSubslice {
  unsigned descriptionIndex = 0;
  unsigned sliceIndex = 0;
  SmallVector<int64_t> offset;
  SmallVector<int64_t> shape;
  Type elementType;
  unsigned elemBytes = 0;
  SmallVector<uint64_t> stridesBytes;
};

// Host-visible tensors are allocated once in the generated L2 region.  Multiple
// tile reads can reference the same GeneratedL2Global by global name and kind.
struct GeneratedL2Global {
  unsigned descriptionIndex = 0;
  unsigned globalId = 0;
  std::string globalName;
  GeneratedGlobalKind globalKind = GeneratedGlobalKind::Input;
  SmallVector<int64_t> shape;
  Type elementType;
  unsigned elemBytes = 0;
  uint64_t tensorBytes = 0;
  uint64_t l2Address = 0;
  uint64_t tokenStrideBytes = 0;
  unsigned numTokens = 1;
  SmallVector<uint64_t> stridesBytes;
  bool isConstant = false;
  uint64_t packedImageOffset = 0;
  uint64_t packedByteSize = 0;
  std::string packedDtype;
  SmallVector<int64_t> packedShape;
  uint64_t requiredAlignment = kGeneratedL2AlignmentBytes;
  std::string runtimeInputFile;
  std::string runtimeInputAsset;
};

struct GeneratedL2Read {
  unsigned descriptionIndex = 0;
  unsigned l2GlobalIndex = 0;
  unsigned dstSubsliceIndex = 0;
  uint64_t srcL2Address = 0;
  uint64_t srcL2TokenStrideBytes = 0;
};

struct GeneratedRecv {
  unsigned descriptionIndex = 0;
  unsigned transitionId = 0;
  int64_t srcHartid = -1;
  int64_t dstHartid = -1;
  unsigned dstSubsliceIndex = 0;
  // Index of this producer's destination-local FIFO sub-ring.
  unsigned producerIdx = 0;
};

struct GeneratedSend {
  unsigned descriptionIndex = 0;
  unsigned transitionId = 0;
  int64_t srcHartid = -1;
  int64_t dstHartid = -1;
  unsigned srcSubsliceIndex = 0;
  unsigned dstSubsliceIndex = 0;
  uint64_t dstL1DataBase = kGeneratedL1DataBaseOffset;
  uint64_t dstSliceL1OffsetBytes = 0;
  uint64_t dstSliceSlotBytes = 0;
  // Must match the destination recv's FIFO sub-ring index.
  unsigned producerIdx = 0;
};

struct GeneratedCollectiveIdentity {
  unsigned stageId = 0;
  unsigned layerIndex = 0;
  unsigned groupId = 0;
};

struct GeneratedOp {
  unsigned descriptionIndex = 0;
  int64_t ownerHartid = -1;
  GeneratedOpKind kind = GeneratedOpKind::Copy;
  std::string deviceName;
  SmallVector<unsigned> inputSubsliceIndices;
  SmallVector<unsigned> outputSubsliceIndices;
  bool isCollective = false;
  GeneratedCollectiveIdentity collectiveIdentity;
  SmallVector<unsigned> collectiveParticipants;
  uint32_t params[8] = {};
};

struct GeneratedSpatzTask {
  GeneratedOpKind kind;
  StringRef name;
  StringRef source;
  StringRef includeDirectory;
  StringRef symbol;
  StringRef runtimeField;
  StringRef featureMacro;
};

struct GeneratedL2Write {
  unsigned descriptionIndex = 0;
  unsigned l2GlobalIndex = 0;
  unsigned srcSubsliceIndex = 0;
  uint64_t dstL2Address = 0;
  uint64_t dstL2TokenStrideBytes = 0;
};

struct GeneratedTilePlan {
  int64_t hartid = -1;
  uint64_t l1DataBaseOffset = kGeneratedL1DataBaseOffset;
  uint64_t nextL1OffsetBytes = 0;
  uint64_t l1DataBytes = 0;
  SmallVector<unsigned> sliceIndices;
  SmallVector<unsigned> initL2ReadIndices;
  SmallVector<unsigned> l2ReadIndices;
  SmallVector<unsigned> recvIndices;
  SmallVector<unsigned> sendIndices;
  SmallVector<unsigned> opIndices;
  SmallVector<unsigned> l2WriteIndices;
  // magia-v3 FIFO gather layout, set by planGatheredFifoLayouts.  When any
  // slice is gathered: per destination-local producer, the 1-based shared
  // payload ring it gathers into, or 0 when its payloads stay in its own slots.
  SmallVector<unsigned> fifoPayloadRings;
  // Bytes per slot of each shared payload ring (the gathered slice's slot size).
  SmallVector<uint64_t> fifoSharedRingBytes;
};

struct MagiaApplicationDescription {
  std::string target = "magia-v2";
  std::string outputStem = "application";
  std::string sourceModelName;
  unsigned meshWidth = 0;
  unsigned meshHeight = 0;
  std::string weightsFile;
  std::string weightsSha256;
  uint64_t weightsSize = 0;
  uint64_t weightsAlignment = kGeneratedL2AlignmentBytes;
  uint64_t l2Base = kMagiaL2Base;
  uint64_t l2End = kMagiaL2End;
  bool hasPackedWeights = false;
  bool randomizeInputs = false;
  unsigned numTokens = kDefaultNumTokens;
  unsigned numTokenSlots = 0;
  unsigned nextGlobalId = 0;
  unsigned nextTransitionId = 0;
  llvm::StringMap<unsigned> globalIds;
  llvm::DenseMap<int64_t, unsigned> nextSliceIdByHart;
  llvm::StringMap<unsigned> l2GlobalIndices;
  llvm::StringMap<unsigned> transitionIds;
  llvm::StringMap<unsigned> initializerSliceIndices;
  SmallVector<GeneratedSlice, 0> slices;
  SmallVector<GeneratedSubslice, 0> subslices;
  SmallVector<GeneratedL2Global, 0> l2Globals;
  SmallVector<GeneratedL2Read, 0> l2Reads;
  SmallVector<GeneratedRecv, 0> recvs;
  SmallVector<GeneratedSend, 0> sends;
  SmallVector<GeneratedOp, 0> ops;
  SmallVector<GeneratedL2Write, 0> l2Writes;
  llvm::MapVector<int64_t, GeneratedTilePlan> tilePlans;
  uint64_t nextL2Address = kMagiaL2Base;
};

unsigned addGeneratedSlice(MagiaApplicationDescription &application, GeneratedSlice slice);
unsigned addGeneratedSubslice(MagiaApplicationDescription &application,
                              GeneratedSubslice subslice);
GeneratedSubslice makeWholeSubslice(const GeneratedSlice &slice);
uint64_t computeOffsetBytes(ArrayRef<int64_t> offsets,
                            ArrayRef<uint64_t> stridesBytes);
unsigned addGeneratedL2Read(MagiaApplicationDescription &application, GeneratedL2Read read);
unsigned getOrCreateTransitionId(MagiaApplicationDescription &application, StringRef channel);
unsigned addGeneratedRecv(MagiaApplicationDescription &application, GeneratedRecv recv);
unsigned addGeneratedSend(MagiaApplicationDescription &application, GeneratedSend send);
unsigned addGeneratedOp(MagiaApplicationDescription &application, GeneratedOp op);
SmallVector<GeneratedSpatzTask>
getRequiredSpatzTasks(const MagiaApplicationDescription &application);
LogicalResult validateSpatzTaskCoverage(
    const MagiaApplicationDescription &application);
unsigned addGeneratedL2Write(MagiaApplicationDescription &application, GeneratedL2Write write);
void noteTile(MagiaApplicationDescription &application, int64_t hartid);
uint64_t alignTo(uint64_t value, uint64_t alignment);
uint64_t product(ArrayRef<int64_t> values);
std::string normalizeSnakeCase(StringRef value);
uint64_t computeSlotBytes(const GeneratedSlice &slice);
unsigned getElementByteWidth(Type type);
SmallVector<uint64_t> computeContiguousStridesBytes(ArrayRef<int64_t> shape,
                                                    unsigned elemBytes);
LogicalResult getOrCreateL2Global(MagiaApplicationDescription &application,
                                  StringRef globalName,
                                  GeneratedGlobalKind globalKind,
                                  RankedTensorType tensorType,
                                  unsigned numTokens,
                                  unsigned &l2GlobalIndex,
                                  const struct L2SourceInfo *sourceInfo = nullptr);
LogicalResult allocateSliceInTileL1(MagiaApplicationDescription &application,
                                    unsigned sliceIndex);
SmallVector<const GeneratedTilePlan *>
getSortedTilePlans(const MagiaApplicationDescription &application);

} // namespace mlir::maps

#endif
