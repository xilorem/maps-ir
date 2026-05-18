set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MapsConversion)
add_public_tablegen_target(MLIRMapsConversionPassIncGen)

add_subdirectory(MapsToD2M)