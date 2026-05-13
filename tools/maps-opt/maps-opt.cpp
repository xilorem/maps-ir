#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"



#include "maps/Dialect/Maps/IR/Maps.h"

int main(int argc, char **argv){
    mlir::DialectRegistry registry;
    
    mlir::registerAllDialects(registry);
    
    registry.insert<mlir::maps::MapsDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Maps optimizer driver\n", registry)
    );
}