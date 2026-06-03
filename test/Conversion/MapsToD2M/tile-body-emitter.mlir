// RUN: maps-opt %s -convert-maps-to-d2m | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%in: tensor<64x32xf32>, %bias: tensor<64x32xf32>) {
    maps.init {
      maps.send %bias, @init_bias_to_tile0
        {dst_hartid = 0 : i64, src_hartid = -1 : i64}
        : tensor<64x32xf32>
      maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
        %0 = maps.recv @init_bias_to_tile0
          {dst_hartid = 0 : i64, src_hartid = -1 : i64}
          : tensor<64x32xf32>
        maps.store %0, @bias_tile0 : tensor<64x32xf32>
      }
    }

    maps.run {
      maps.stage @stage0 attributes {stage_id = 0 : i64} {
        maps.send %in, @in_to_tile0
          {dst_hartid = 0 : i64, src_hartid = -1 : i64}
          : tensor<64x32xf32>
        maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
          %0 = maps.recv @in_to_tile0
            {dst_hartid = 0 : i64, src_hartid = -1 : i64}
            : tensor<64x32xf32>
          %1 = maps.load @bias_tile0 : tensor<64x32xf32>
          %2 = tensor.empty() : tensor<64x32xf32>
          %3 = linalg.generic {indexing_maps = [#map, #map, #map],
                               iterator_types = ["parallel", "parallel"]}
            ins(%0, %1 : tensor<64x32xf32>, tensor<64x32xf32>)
            outs(%2 : tensor<64x32xf32>) {
            ^bb0(%in0: f32, %in1: f32, %out: f32):
              %4 = math.exp %in0 : f32
              %5 = arith.negf %4 : f32
              %6 = arith.addf %5, %in1 : f32
              linalg.yield %6 : f32
          } -> tensor<64x32xf32>
          maps.send %3, @tile0_to_host
            {dst_hartid = -1 : i64, src_hartid = 0 : i64}
            : tensor<64x32xf32>
        }
      }
    }

    %0 = maps.recv @tile0_to_host
      {dst_hartid = -1 : i64, src_hartid = 0 : i64}
      : tensor<64x32xf32>
    return
  }
}

// CHECK-LABEL: func.func @forward
// CHECK: d2m.spatial
// CHECK: d2m.generic
// CHECK: "d2m.tile_exp"
// CHECK: "d2m.tile_negative"
// CHECK: "d2m.tile_add"
// CHECK-NOT: math.exp
// CHECK-NOT: arith.negf
// CHECK-NOT: arith.addf
// CHECK-NOT: maps.
