// RUN: maps-opt %s -convert-maps-to-d2m | FileCheck %s

module {
  func.func @main(%weight: tensor<128x32xf32>) {
    maps.init {
      maps.send %weight, @init_weight_to_tile0
        {dst_hartid = 0 : i64, src_hartid = -1 : i64}
        : tensor<128x32xf32>
      maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
        %0 = maps.recv @init_weight_to_tile0
          {dst_hartid = 0 : i64, src_hartid = -1 : i64}
          : tensor<128x32xf32>
        maps.store %0, @weight_tile0 : tensor<128x32xf32>
      }
    }

    maps.run {
      maps.stage @stage0 attributes {stage_id = 0 : i64} {
        maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
          %0 = maps.load @weight_tile0 : tensor<128x32xf32>
          maps.send %0, @weight_to_host
            {dst_hartid = -1 : i64, src_hartid = 0 : i64}
            : tensor<128x32xf32>
        }
      }
    }

    %0 = maps.recv @weight_to_host
      {dst_hartid = -1 : i64, src_hartid = 0 : i64}
      : tensor<128x32xf32>
    return
  }
}

// CHECK-LABEL: module
// CHECK: func.func @forward(%arg0: tensor<128x32xf32> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<128x32xf32>
// CHECK: %[[EMPTY_DEVICE:.*]] = d2m.empty() : tensor<1x1x4x1x!ttcore.tile<32x32, f32>
// CHECK: %[[TO_DEVICE:.*]] = d2m.to_layout %arg0, %[[EMPTY_DEVICE]]
// CHECK: %[[EMPTY_HOST:.*]] = d2m.empty() : tensor<128x32xf32>
// CHECK: %[[TO_HOST:.*]] = d2m.to_layout %[[TO_DEVICE]], %[[EMPTY_HOST]]
// CHECK: return %[[TO_HOST]] : tensor<128x32xf32>
// CHECK-NOT: maps.
