# maps-ir

`maps-ir` is an out-of-tree MLIR project for the MAPS tool. 

## Prerequisites

- CMake 3.20 or newer
- A C++17 compiler
- LLVM and MLIR development packages built with CMake package files

The build uses normal CMake package discovery for LLVM and MLIR. Point CMake at
your MLIR install with either `CMAKE_PREFIX_PATH`, or explicit `MLIR_DIR` and
`LLVM_DIR` values.

For an LLVM/MLIR install rooted at `/path/to/llvm-install`, the package
directories are typically:

- `/path/to/llvm-install/lib/cmake/llvm`
- `/path/to/llvm-install/lib/cmake/mlir`

## Build

Configure and build the local tools:

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm

cmake --build build --target maps-translate
```


## Run The Example

Generate MAPS MLIR from the bundled example JSON:

```bash
mkdir -p generated

./build/tools/maps-translate/maps-translate \
  --json-to-maps examples/magia_example.pipeline.json \
  > generated/magia_example.pipeline.mlir
```

The generated file is regular MAPS MLIR. It may contain standard MLIR dialect
ops such as `tensor`, `linalg`, `arith`, and `math` inside `maps.*` regions.
For example, repeated affine map aliases such as
`#map = affine_map<(d0, d1) -> (d0, d1)>` are normal MLIR assembly for
`linalg.generic` indexing maps.

To check that the generated MLIR parses:

```bash
./build/tools/maps-opt/maps-opt generated/magia_example.pipeline.mlir \
  > generated/magia_example.pipeline.roundtrip.mlir
```

## Repository Layout

- `include/maps/Dialect/Maps/IR`: MAPS dialect declarations
- `lib/Dialect/Maps/IR`: MAPS dialect implementation
- `tools/maps-translate`: JSON-to-MAPS importer
- `tools/maps-opt`: MAPS optimizer driver
- `test`: MLIR tests
- `examples`: input examples
