cmake_minimum_required(VERSION 3.20)
project(maps-ir LANGUAGES CXX C)

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(TTMLIR REQUIRED CONFIG
  HINTS /home/ivan/repos/tt-mlir/build/lib/cmake/ttmlir
)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddMLIR)
include(AddLLVM)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS}) 
include_directories(${MLIR_INCLUDE_DIRS}) 
include_directories(${PROJECT_SOURCE_DIR}/include) 
include_directories(${PROJECT_BINARY_DIR}/include) 
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
