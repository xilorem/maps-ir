// RUN: maps-opt %s -print-maps-placement-analysis -o /dev/null | FileCheck %s

module {
  func.func @main(%input: tensor<64x128xf32>, %weight: tensor<128x32xf32>,
                  %bias: tensor<64x32xf32>) {
    maps.init {
      %weight_slice = tensor.extract_slice %weight[0, 0] [128, 32] [1, 1]
        : tensor<128x32xf32> to tensor<128x32xf32>
      maps.send %weight_slice, @init_weight_to_tile0
        {dst_hartid = 0 : i64, src_hartid = -1 : i64}
        : tensor<128x32xf32>
      %bias_slice = tensor.extract_slice %bias[0, 0] [64, 32] [1, 1]
        : tensor<64x32xf32> to tensor<64x32xf32>
      maps.send %bias_slice, @init_bias_to_tile1
        {dst_hartid = 1 : i64, src_hartid = -1 : i64}
        : tensor<64x32xf32>

      maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
        %0 = maps.recv @init_weight_to_tile0
          {dst_hartid = 0 : i64, src_hartid = -1 : i64}
          : tensor<128x32xf32>
        maps.store %0, @weight_tile0 : tensor<128x32xf32>
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
        %input_slice = tensor.extract_slice %input[0, 0] [64, 128] [1, 1]
          : tensor<64x128xf32> to tensor<64x128xf32>
        maps.send %input_slice, @input_to_tile0
          {dst_hartid = 0 : i64, src_hartid = -1 : i64}
          : tensor<64x128xf32>

        maps.tile @tile0 attributes {coords = array<i64: 0, 0>, tile_id = 0 : i64} {
          %0 = maps.recv @input_to_tile0
            {dst_hartid = 0 : i64, src_hartid = -1 : i64}
            : tensor<64x128xf32>
          %1 = maps.load @weight_tile0 : tensor<128x32xf32>
          %2 = tensor.empty() : tensor<64x32xf32>
          %3 = linalg.matmul ins(%0, %1 : tensor<64x128xf32>, tensor<128x32xf32>)
            outs(%2 : tensor<64x32xf32>) -> tensor<64x32xf32>
          maps.send %3, @matmul_to_tile1
            {dst_hartid = 1 : i64, src_hartid = 0 : i64}
            : tensor<64x32xf32>
        }
      }

      maps.stage @stage1 attributes {stage_id = 1 : i64} {
        maps.tile @tile1 attributes {coords = array<i64: 1, 0>, tile_id = 1 : i64} {
          %0 = maps.recv @matmul_to_tile1
            {dst_hartid = 1 : i64, src_hartid = 0 : i64}
            : tensor<64x32xf32>
          %1 = maps.load @bias_tile1 : tensor<64x32xf32>
          %2 = tensor.empty() : tensor<64x32xf32>
          %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                                affine_map<(d0, d1) -> (d0, d1)>,
                                                affine_map<(d0, d1) -> (d0, d1)>],
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

// CHECK: maps.placement
// CHECK:   tile_placements: 2
// CHECK:     - tile: 0, coords: [0, 0], core_range: #ttcore.core_range<(0, 0), (0, 0)>, programs: 2
// CHECK:     - tile: 1, coords: [1, 0], core_range: #ttcore.core_range<(0, 1), (0, 1)>, programs: 2
// CHECK:   placement_groups: 2
// CHECK:     - core_range: #ttcore.core_range<(0, 0), (0, 0)>, tile_programs: [{tile: 0, stage: -1}, {tile: 0, stage: 0}]
// CHECK:     - core_range: #ttcore.core_range<(0, 1), (0, 1)>, tile_programs: [{tile: 1, stage: -1}, {tile: 1, stage: 1}]
