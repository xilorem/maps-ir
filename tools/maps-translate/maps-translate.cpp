#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

namespace mlir::maps {
void registerJsonToMapsTranslation();
}

int main(int argc, char **argv) {
  mlir::maps::registerJsonToMapsTranslation();
  return failed(mlir::mlirTranslateMain(argc, argv, "MAPS translation-driver"));
}