#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"


#include "maps/Dialect/Maps/IR/Maps.h"
#include "maps/Conversion/Passes.h"

int main(int argc, char **argv){
    mlir::DialectRegistry registry;
    
    mlir::registerAllDialects(registry);
    registry.insert<mlir::tt::ttcore::TTCoreDialect,
                    mlir::tt::d2m::D2MDialect>();

    
    mlir::maps::registerMapsConversionPasses();

    registry.insert<mlir::maps::MapsDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Maps optimizer driver\n", registry)
    );
}
