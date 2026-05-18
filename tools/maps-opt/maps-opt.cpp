#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"

#include "ttmlir/RegisterAll.h"

#include "maps/Dialect/Maps/IR/Maps.h"
#include "maps/Conversion/Passes.h"

int main(int argc, char **argv){
    mlir::DialectRegistry registry;
    
    mlir::registerAllDialects(registry);
    mlir::tt::registerAllDialects(registry);
    mlir::tt::registerAllExtensions(registry);
    
    mlir::maps::registerMapsConversionPasses();

    registry.insert<mlir::maps::MapsDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Maps optimizer driver\n", registry)
    );
}