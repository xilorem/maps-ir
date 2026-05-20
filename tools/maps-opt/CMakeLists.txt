add_llvm_executable(maps-opt
  maps-opt.cpp
)

llvm_update_compile_flags(maps-opt)

target_link_directories(maps-opt
  PRIVATE
  /home/ivan/repos/tt-mlir/build/lib
  /opt/ttmlir-toolchain/lib
  /opt/ttmlir-toolchain/lib64
)

target_link_libraries(maps-opt
  PRIVATE
  MLIRMapsDialect
  MLIRMapsToD2M
  TTMLIR::MLIRTTCoreDialect
  TTMLIR::MLIRD2MDialect
  TTMLIR::MLIRD2MUtils
  TTMLIR::MLIRTTIRDialect
  TTMLIR::MLIRTTNNDialect
  MLIRRegisterAllDialects
  MLIROptLib
  MLIRIR
  MLIRSupport
)
