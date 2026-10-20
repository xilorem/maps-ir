#ifndef MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_EMITTER_H
#define MAPS_TOOLS_MAPS_TRANSLATE_MAGIAAPPLICATIONCODEGEN_EMITTER_H

#include "ApplicationDescription.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir::maps {

void planGatheredFifoLayouts(MagiaApplicationDescription &application);
void emitApplicationHeader(const MagiaApplicationDescription &application,
                           llvm::raw_ostream &os);
void emitApplicationDataSource(const MagiaApplicationDescription &application,
                                llvm::raw_ostream &os);
void emitApplicationTileSource(const MagiaApplicationDescription &application,
                               const GeneratedTilePlan &plan,
                               llvm::raw_ostream &os);
void emitApplicationRunner(const MagiaApplicationDescription &application,
                           llvm::raw_ostream &os);
void emitInitialApplicationSource(const MagiaApplicationDescription &application,
                                  llvm::raw_ostream &os);
void emitApplicationCMake(const MagiaApplicationDescription &application,
                          llvm::raw_ostream &os);
void emitApplicationReadme(const MagiaApplicationDescription &application,
                           llvm::raw_ostream &os);
void emitApplicationInitializersAssembly(const MagiaApplicationDescription &application,
                                         llvm::raw_ostream &os);
std::string getApplicationTileFilename(const MagiaApplicationDescription &application,
                                       int64_t hartid);

} // namespace mlir::maps

#endif
