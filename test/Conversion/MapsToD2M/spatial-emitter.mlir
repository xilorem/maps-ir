// RUN: maps-opt %s -convert-maps-to-d2m | FileCheck %s

module {
  func.func @main(%lhs: tensor<64x128xf32>, %rhs: tensor<128x32xf32>) {
    maps.init {
      maps.send %rhs, @init_rhs_to_tile0
        {dst_hartid = 0 : i64, src_hartid = -1 : i64}
        : tensor<128x32xf32>
      maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
        %0 = maps.recv @init_rhs_to_tile0
          {dst_hartid = 0 : i64, src_hartid = -1 : i64}
          : tensor<128x32xf32>
        maps.store %0, @rhs_tile0 : tensor<128x32xf32>
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
// CHECK: %[[SPATIAL:.*]] = d2m.spatial {grid_ranges = [#ttcore.core_range<(0,0), (0,0)>]}
// CHECK: ins(%{{.*}}, %{{.*}}
// CHECK: outs(%{{.*}}
// CHECK: %[[GENERIC:.*]] = d2m.generic
// CHECK: "d2m.tile_matmul"
// CHECK: d2m.spatial_yield %[[GENERIC]]
// CHECK: %[[HOST_EMPTY:.*]] = d2m.empty() : tensor<64x32xf32>
// CHECK: %[[HOST_OUT:.*]] = d2m.to_layout %[[SPATIAL]], %[[HOST_EMPTY]]
// CHECK: return %[[HOST_OUT]] : tensor<64x32xf32>
// CHECK-NOT: maps.
