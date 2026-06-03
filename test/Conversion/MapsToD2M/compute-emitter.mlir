// RUN: maps-opt %s -convert-maps-to-d2m | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%lhs: tensor<64x128xf32>, %rhs: tensor<128x32xf32>,
                  %bias: tensor<64x32xf32>) {
    maps.init {
      maps.send %rhs, @init_rhs_to_tile0
        {dst_hartid = 0 : i64, src_hartid = -1 : i64}
        : tensor<128x32xf32>
      maps.send %bias, @init_bias_to_tile1
        {dst_hartid = 1 : i64, src_hartid = -1 : i64}
        : tensor<64x32xf32>

      maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
        %0 = maps.recv @init_rhs_to_tile0
          {dst_hartid = 0 : i64, src_hartid = -1 : i64}
          : tensor<128x32xf32>
        maps.store %0, @rhs_tile0 : tensor<128x32xf32>
      }

      maps.tile @tile1 attributes {coords = array<i64: 1, 0>, tile_id = 1 : i64} {
        %0 = maps.recv @init_bias_to_tile1
          {dst_hartid = 1 : i64, src_hartid = -1 : i64}
          : tensor<64x32xf32>
        maps.store %0, @bias_tile1 : tensor<64x32xf32>
      }
    }

    maps.run {
      maps.stage @stage0 attributes {stage_id = 0 : i64} {
        maps.send %lhs, @lhs_to_tile0
          {dst_hartid = 0 : i64, src_hartid = -1 : i64}
          : tensor<64x128xf32>
        maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
          %0 = maps.recv @lhs_to_tile0
            {dst_hartid = 0 : i64, src_hartid = -1 : i64}
            : tensor<64x128xf32>
          %1 = maps.load @rhs_tile0 : tensor<128x32xf32>
          %2 = tensor.empty() : tensor<64x32xf32>
          %3 = linalg.matmul ins(%0, %1 : tensor<64x128xf32>, tensor<128x32xf32>)
            outs(%2 : tensor<64x32xf32>) -> tensor<64x32xf32>
          maps.send %3, @tile0_to_tile1
            {dst_hartid = 1 : i64, src_hartid = 0 : i64}
            : tensor<64x32xf32>
        }
      }

      maps.stage @stage1 attributes {stage_id = 1 : i64} {
        maps.tile @tile1 attributes {coords = array<i64: 1, 0>, tile_id = 1 : i64} {
          %0 = maps.recv @tile0_to_tile1
            {dst_hartid = 1 : i64, src_hartid = 0 : i64}
            : tensor<64x32xf32>
          %1 = maps.load @bias_tile1 : tensor<64x32xf32>
          %2 = tensor.empty() : tensor<64x32xf32>
          %3 = linalg.generic {indexing_maps = [#map, #map, #map],
                               iterator_types = ["parallel", "parallel"]}
            ins(%0, %1 : tensor<64x32xf32>, tensor<64x32xf32>)
            outs(%2 : tensor<64x32xf32>) {
            ^bb0(%in0: f32, %in1: f32, %out: f32):
              %4 = arith.addf %in0, %in1 : f32
              linalg.yield %4 : f32
          } -> tensor<64x32xf32>
          maps.send %3, @tile1_to_host
            {dst_hartid = -1 : i64, src_hartid = 1 : i64}
            : tensor<64x32xf32>
        }
      }
    }

    %0 = maps.recv @tile1_to_host
      {dst_hartid = -1 : i64, src_hartid = 1 : i64}
      : tensor<64x32xf32>
    return
  }
}

// CHECK-LABEL: func.func @forward
// CHECK: %[[LHS_EMPTY:.*]] = d2m.empty() : tensor<1x1x2x4x!ttcore.tile<32x32, f32>
// CHECK: %[[LHS_TILED:.*]] = d2m.to_layout %arg0, %[[LHS_EMPTY]]
// CHECK: %[[RHS_EMPTY:.*]] = d2m.empty() : tensor<1x1x4x1x!ttcore.tile<32x32, f32>
// CHECK: %[[RHS_TILED:.*]] = d2m.to_layout %arg1, %[[RHS_EMPTY]]
// CHECK: %[[MATMUL_SPATIAL:.*]] = d2m.spatial
// CHECK: %[[MATMUL:.*]] = d2m.generic
// CHECK: d2m.remote_load
// CHECK: "d2m.tile_matmul"
// CHECK: %[[BIAS_EMPTY:.*]] = d2m.empty() : tensor<1x1x2x1x!ttcore.tile<32x32, f32>
// CHECK: %[[BIAS_TILED:.*]] = d2m.to_layout %arg2, %[[BIAS_EMPTY]]
// CHECK: %[[ADD_SPATIAL:.*]] = d2m.spatial
// CHECK: %[[ADD_GENERIC:.*]] = d2m.generic
// CHECK: "d2m.tile_add"
// CHECK: %[[HOST_EMPTY:.*]] = d2m.empty() : tensor<64x32xf32>
// CHECK: %[[HOST_OUT:.*]] = d2m.to_layout %[[ADD_SPATIAL]], %[[HOST_EMPTY]]
// CHECK: return %[[HOST_OUT]] : tensor<64x32xf32>
// CHECK-NOT: linalg.matmul
// CHECK-NOT: maps.
