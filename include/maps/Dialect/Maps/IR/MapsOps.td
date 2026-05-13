

#ifndef MAPS_OPS
#define MAPS_OPS

include "maps/Dialect/Maps/IR/MapsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class Maps_Op<string mnemonic, list<Trait> traits = []>:
    Op<Maps_Dialect, mnemonic, traits>;


def StageOp : Maps_Op<"stage", [SingleBlock, NoTerminator]>{
    let summary = "A stage operation";
    let description = [{
        A stage operation represents a single stage in a pipeline. 
    }];
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        I64Attr:$stage_id);

    let regions = (region SizedRegion<1>:$body);

    // let hasVerifier = 1;

    let assemblyFormat = [{
        $sym_name attr-dict-with-keyword $body
    }];

}

def TileOp : Maps_Op<"tile">{
    let summary = "A tile operation";
}

def InitOp : Maps_Op<"init">{
    let summary = "An init operation";
}

def ReadOp : Maps_Op<"read">{
    let summary = "A read operation";
}

def SendOp : Maps_Op<"send">{
    let summary = "A send operation";
}

def RecieveOp : Maps_Op<"recv">{
    let summary = "A recieve operation";
}

def ReturnOp : Maps_Op<"return">{
    let summary = "A return operation";
}

#endif // MAPS_OPS