// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<128x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<128x1024xf32>) -> memref<128x1024xf32> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %A_gpu = gpu.alloc  host_shared () : memref<128x1024xf16>
    memref.copy %A, %A_gpu : memref<128x1024xf16> to memref<128x1024xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<1024x1024xf16>
    memref.copy %B, %B_gpu : memref<1024x1024xf16> to memref<1024x1024xf16>
    %C_gpu = gpu.alloc  host_shared () : memref<128x1024xf32>
    memref.copy %C, %C_gpu : memref<128x1024xf32> to memref<128x1024xf32>

    %matSizeX = memref.dim %C, %c0 : memref<128x1024xf32>
    %matSizeY = memref.dim %C, %c1 : memref<128x1024xf32>

    %tileX = arith.constant 8 : index
    %tileY = arith.constant 16 : index
    %bDimX = arith.divui %matSizeX, %tileX : index
    %bDimY = arith.divui %matSizeY, %tileY : index

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%bDimX, %bDimY, %c1) threads in (%c1, %c1, %c1) args(%A_gpu : memref<128x1024xf16>, %B_gpu : memref<1024x1024xf16>, %C_gpu : memref<128x1024xf32>)

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
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1024 = arith.constant 1024 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y

      %dimK = memref.dim %A, %c1 : memref<128x1024xf16>

      %tileDimK = arith.constant 16 : index

      // Init C tile values.
      %2 = arith.muli %0, %c8 : index
      %3 = arith.muli %1, %c16 : index
      %tileC = xegpu.create_nd_tdesc %C[%2, %3] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %vC = xegpu.load_nd %tileC {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

      // Number of prefetching stages.
      // NOTE: for simplicity rest of the code is hardcoded for the case: stages=2.
      %numStages = arith.constant 2 : index
      %numTailTiles = arith.subi %numStages, %c1 : index
      %numTailElements = arith.muli %numTailTiles, %tileDimK : index
      %ub = arith.subi %dimK, %numTailElements : index

      %tileA = xegpu.create_nd_tdesc %A[%2, %c0] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %tileB = xegpu.create_nd_tdesc %B[%c0, %3] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>

      xegpu.prefetch_nd %tileA {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
      xegpu.prefetch_nd %tileB {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>

      xegpu.compile_hint

      // each work-group has 1 subgroup. the subgroup calculates a [8x16 = 8x1024 * 1024x16] block
      // Partial GEMM computation.
      %partRes, %lastTileA, %lastTileB = scf.for %arg3 = %c0 to %ub step %tileDimK iter_args(%acc = %vC, %tA = %tileA, %tB = %tileB) -> (vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>) {
        %vA = xegpu.load_nd %tA {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %vB = xegpu.load_nd %tB {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

        xegpu.compile_hint

        %nextTileA = xegpu.update_nd_offset %tA, [%c0, %tileDimK] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %nextTileB = xegpu.update_nd_offset %tB, [%tileDimK, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>

        xegpu.prefetch_nd %nextTileA {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %nextTileB {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>

        xegpu.compile_hint

        %vPartRes = xegpu.dpas %vA, %vB, %acc {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        scf.yield %vPartRes, %nextTileA, %nextTileB : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>
      }

      xegpu.compile_hint

      // Tail GEMM computation.
      %vlA = xegpu.load_nd %lastTileA {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
      %vlB = xegpu.load_nd %lastTileB {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
      %res = xegpu.dpas %vlA, %vlB, %partRes {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

      // Store GEMM result.
      xegpu.store_nd %res, %tileC {mode = vc} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>

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
