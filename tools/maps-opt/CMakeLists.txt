add_llvm_executable(maps-opt
  maps-opt.cpp
)

llvm_update_compile_flags(maps-opt)

target_link_libraries(maps-opt
  PRIVATE
  MLIRMapsDialect
  MLIRRegisterAllDialects
  MLIROptLib
  MLIRIR
  MLIRSupport
)
