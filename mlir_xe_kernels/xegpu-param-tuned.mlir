// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#mapRow = affine_map<(d0) -> (d0 * 64)>
#mapCol = affine_map<(d0) -> (d0 * 64)>

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<128x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<128x1024xf32>) -> memref<128x1024xf32> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %A_gpu = gpu.alloc  host_shared () : memref<128x1024xf16>
    memref.copy %A, %A_gpu : memref<128x1024xf16> to memref<128x1024xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %B, %B_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %C_gpu = gpu.alloc  host_shared () : memref<128x1024xf32>
    memref.copy %C, %C_gpu : memref<128x1024xf32> to memref<128x1024xf32>

    %dimM = memref.dim %C, %c0 : memref<128x1024xf32>
    %dimN = memref.dim %C, %c1 : memref<128x1024xf32>

    %tileSizeM = arith.constant 64 : index
    %tileSizeN = arith.constant 64 : index
    %numTilesX = arith.divui %dimM, %tileSizeM : index
    %numTilesY = arith.divui %dimN, %tileSizeN : index

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%numTilesX, %numTilesY, %c1) threads in (%c8, %c4, %c1) args(%A_gpu : memref<128x1024xf16>, %B_gpu : memref<1024x1024xf16>, %C_gpu : memref<128x1024xf32>)

    // %cast = memref.cast %C_gpu : memref<128x1024xf32> to memref<*xf32>
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()

    gpu.dealloc  %A_gpu : memref<128x1024xf16>
    gpu.dealloc  %B_gpu : memref<1024x1024xf16>
    return %C_gpu : memref<128x1024xf32>
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<subgroup_size = 16>>} {
    gpu.func @test_kernel(%A: memref<128x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<128x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %c24 = arith.constant 24 : index
      %c32 = arith.constant 32 : index
      %c48 = arith.constant 48 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %4 = affine.apply #mapRow(%0)
      %5 = affine.apply #mapCol(%1)
      %2 = gpu.thread_id  x
      %3 = gpu.thread_id  y // Contiguous vectorizable dimension.

      // Block tiles.
      // Created by initial GEMM tiling.
      // Each thread block computes one C tile.
      //
      // PARAM: sizes have to match new GEMM tile size
      // %blockA = memref.subview %A[%4, 0] [16, 1024] [1, 1] : memref<128x1024xf16> to memref<16x1024xf16, strided<[1024, 1], offset: ?>>
      // %blockB = memref.subview %B[0, %5] [1024, 32] [1, 1] : memref<1024x1024xf16> to memref<1024x32xf16, strided<[1024, 1], offset: ?>>
      // %blockC = memref.subview %C[%4, %5] [16, 32] [1, 1] : memref<128x1024xf32> to memref<16x32xf32, strided<[1024, 1], offset: ?>>

      // Thread tile sizes.
      // Each thread will compute <1x1> DPAS tile (<8x16> elements) of C tile.
      %TM = arith.constant 8 : index
      %TN = arith.constant 16 : index

      // Block tile sizes.
      // Parallel dimensions are based on the original tiling size.
      // Reduction dimension tiling is chosen to match thread tile sizes.
      //
      // PARAM: block sizes BM and BN have to match GEMM tile size
      %BM = arith.constant 64 : index
      %BN = arith.constant 64 : index
      %BK = arith.constant 64 : index // == %blockDimK - matches inner block tile dim size
      %numThreadTiles = arith.divui %BK, %TN : index

      // Find size of the GEMM tiles reduction dimension.
      %dimK = memref.dim %A, %c1 : memref<128x1024xf16>
      %numSubTilesK = arith.ceildivsi %dimK, %BK : index

      // Initialize accumulator registers.
      //
      // Each thread loads C tiles it will compute.
      %threadOffsetRow = arith.muli %2, %TM : index
      %threadOffsetCol = arith.muli %3, %TN : index
      %outTileRow = arith.addi %4, %threadOffsetRow : index
      %outTileCol = arith.addi %5, %threadOffsetCol : index
      %tileC = xegpu.create_nd_tdesc %C[%outTileRow, %outTileCol] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %vC = xegpu.load_nd %tileC {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

      xegpu.compile_hint

      %res = scf.for %subtileIv = %c0 to %numSubTilesK step %c1 iter_args(%acc = %vC) -> (vector<8x16xf32>) {
        // Load sub-tiles of A and B tiles from GMEM to SMEM.
        // The sub-tiles are loaded cooperatively using all threads in a threadblock.
        // Find the start position of a sub-tile.
        %subtileOffset = arith.muli %subtileIv, %BK : index

        // %subA = memref.subview %blockA[0, %subtileOffset] [16, 32] [1, 1] : memref<16x1024xf16, strided<[1024, 1], offset: ?>> to memref<16x32xf16, strided<[1024, 1], offset: ?>>
        // %subB = memref.subview %blockB[%subtileOffset, 0] [32, 32] [1, 1] : memref<1024x32xf16, strided<[1024, 1], offset: ?>> to memref<32x32xf16, strided<[1024, 1], offset: ?>>

        // Fetch data from GMEM to SMEM using all threads in a threadblock.
        // Each thread has to load 1 tile of A and B from their block tiles.
        %tColA = arith.addi %subtileOffset, %threadOffsetCol : index
        %tRowB = arith.addi %subtileOffset, %threadOffsetRow : index
        %tileA = xegpu.create_nd_tdesc %A[%outTileRow, %tColA] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
        %tileB = xegpu.create_nd_tdesc %B[%tRowB, %outTileCol] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>

        // Use prefetching to cache all the A and B sub-tiles.
        // They will be shared among threads within the block.
        xegpu.prefetch_nd %tileA {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %tileB {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>

        xegpu.compile_hint

        // Synchronize all threads in a threadblock.
        // Whole A and B sub-tiles are needed to perform computation.
        // Wait for all threads in a threadblock to finish loading A and B tile elements.
        // TOOD: see if needed, HW might enforce cache coherency on its own
        gpu.barrier

        // GEMM computation.
        // TODO: unroll this loop
        %partRes = scf.for %tOffset = %c0 to %numThreadTiles step %c1 iter_args(%valC = %acc) -> (vector<8x16xf32>) {
          %subtileOffsetRow = arith.muli %tOffset, %TM : index
          %subtileOffsetCol = arith.muli %tOffset, %TN : index
          %subACol = arith.addi %subtileOffset, %subtileOffsetCol : index
          %subBRow = arith.addi %subtileOffset, %subtileOffsetRow : index
          %tA = xegpu.create_nd_tdesc %A[%outTileRow, %subACol] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
          %tB = xegpu.create_nd_tdesc %B[%subBRow, %outTileCol] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
          %vA = xegpu.load_nd %tA {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
          %vB = xegpu.load_nd %tB {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
          %dpas = xegpu.dpas %vA, %vB, %valC {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
          scf.yield %dpas : vector<8x16xf32>
        }

        xegpu.compile_hint
        // Synchronize all threads in a threadblock.
        // All current computations have to be finished before SMEM A and B tiles can be
        // replaced with new values (new tiles) from GMEM.
        // TODO: see if needed, cache might be large enough to allow prefetching of the next set of tiles
        // gpu.barrier

        scf.yield %partRes : vector<8x16xf32>
      }

      // Store the final C tile element values.
      xegpu.store_nd %res, %tileC {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>

      gpu.return
    }
  }
  memref.global "private" @matA : memref<128x1024xf16> = dense<1.0>
  memref.global "private" @matB : memref<1024x1024xf16> = dense<1.0>
  memref.global "private" @matC : memref<128x1024xf32> = dense<0.0>
  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.get_global @matA : memref<128x1024xf16>
    %B = memref.get_global @matB : memref<1024x1024xf16>
    %C = memref.get_global @matC : memref<128x1024xf32>
    %2 = call @test(%A, %B, %C) : (memref<128x1024xf16>, memref<1024x1024xf16>, memref<128x1024xf32>) -> memref<128x1024xf32>

    return
  }
}
