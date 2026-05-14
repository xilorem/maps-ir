add_llvm_executable(maps-translate
  maps-translate.cpp
  JsonToMaps.cpp
)

llvm_update_compile_flags(maps-translate)

target_link_libraries(maps-translate
  PRIVATE
  MLIRMapsDialect
  MLIRMapsDialect
  MLIRTranslateLib
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRArithDialect
)
