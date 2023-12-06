// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#mapRow = affine_map<(d0) -> (d0 * 128)>
#mapCol = affine_map<(d0) -> (d0 * 128)>

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

    %tileSizeM = arith.constant 128 : index
    %tileSizeN = arith.constant 128 : index
    %numTilesX = arith.divui %dimM, %tileSizeM : index
    %numTilesY = arith.divui %dimN, %tileSizeN : index

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%numTilesX, %numTilesY, %c1) threads in (%c4, %c4, %c1) args(%A_gpu : memref<128x1024xf16>, %B_gpu : memref<1024x1024xf16>, %C_gpu : memref<128x1024xf32>)

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
      // Each thread will compute <2x4> DPAS tile (<8x16> elements) of C tile.
      %TM = arith.constant 32 : index
      %TN = arith.constant 32 : index
      %stepTM = arith.constant 8 : index
      %stepTN = arith.constant 16 : index

      // Block tile sizes.
      // Parallel dimensions are based on the original tiling size.
      // Reduction dimension tiling is chosen to match thread tile sizes.
      //
      // PARAM: block sizes BM and BN have to match GEMM tile size
      %BM = arith.constant 128 : index
      %BN = arith.constant 128 : index
      %BK = arith.constant 32 : index // == %blockDimK - matches inner block tile dim size
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
      %otr_0 = arith.addi %outTileRow, %c0 : index
      %otr_1 = arith.addi %outTileRow, %stepTM : index
      %otr_2 = arith.addi %outTileRow, %c16 : index
      %otr_3 = arith.addi %outTileRow, %c24 : index
      %otc_0 = arith.addi %outTileCol, %c0 : index
      %otc_1 = arith.addi %outTileCol, %stepTN : index
      // %otc_2 = arith.addi %outTileCol, %c32 : index
      // %otc_3 = arith.addi %outTileCol, %c48 : index

      %tileC_0_0 = xegpu.create_nd_tdesc %C[%otr_0, %otc_0] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %tileC_0_1 = xegpu.create_nd_tdesc %C[%otr_0, %otc_1] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %tileC_1_0 = xegpu.create_nd_tdesc %C[%otr_1, %otc_0] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %tileC_1_1 = xegpu.create_nd_tdesc %C[%otr_1, %otc_1] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %tileC_2_0 = xegpu.create_nd_tdesc %C[%otr_2, %otc_0] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %tileC_2_1 = xegpu.create_nd_tdesc %C[%otr_2, %otc_1] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %tileC_3_0 = xegpu.create_nd_tdesc %C[%otr_3, %otc_0] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>
      %tileC_3_1 = xegpu.create_nd_tdesc %C[%otr_3, %otc_1] {mode = vc} : memref<128x1024xf32> -> !xegpu.tensor_desc<8x16xf32>

      %tileA_0_0 = xegpu.create_nd_tdesc %A[%otr_0, %c0] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %tileA_0_1 = xegpu.create_nd_tdesc %A[%otr_0, %stepTN] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %tileA_1_0 = xegpu.create_nd_tdesc %A[%otr_1, %c0] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %tileA_1_1 = xegpu.create_nd_tdesc %A[%otr_1, %stepTN] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %tileA_2_0 = xegpu.create_nd_tdesc %A[%otr_2, %c0] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %tileA_2_1 = xegpu.create_nd_tdesc %A[%otr_2, %stepTN] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %tileA_3_0 = xegpu.create_nd_tdesc %A[%otr_3, %c0] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %tileA_3_1 = xegpu.create_nd_tdesc %A[%otr_3, %stepTN] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>

      %tileB_0_0 = xegpu.create_nd_tdesc %B[%c0, %otc_0] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      %tileB_0_1 = xegpu.create_nd_tdesc %B[%c0, %otc_1] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      %tileB_1_0 = xegpu.create_nd_tdesc %B[%stepTN, %otc_0] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      %tileB_1_1 = xegpu.create_nd_tdesc %B[%stepTN, %otc_1] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_2_0 = xegpu.create_nd_tdesc %B[%c32, %otc_0] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_2_1 = xegpu.create_nd_tdesc %B[%c32, %otc_1] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_3_0 = xegpu.create_nd_tdesc %B[%c48, %otc_0] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_3_1 = xegpu.create_nd_tdesc %B[%c48, %otc_1] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_2_0 = xegpu.create_nd_tdesc %B[%c32, %otc_0] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_2_1 = xegpu.create_nd_tdesc %B[%c32, %otc_1] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_2_2 = xegpu.create_nd_tdesc %B[%c32, %otc_2] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_2_3 = xegpu.create_nd_tdesc %B[%c32, %otc_3] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_3_0 = xegpu.create_nd_tdesc %B[%c48, %otc_0] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_3_1 = xegpu.create_nd_tdesc %B[%c48, %otc_1] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_3_2 = xegpu.create_nd_tdesc %B[%c48, %otc_2] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      // %tileB_3_3 = xegpu.create_nd_tdesc %B[%c48, %otc_3] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>

      %vC_0_0 = xegpu.load_nd %tileC_0_0 {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %vC_0_1 = xegpu.load_nd %tileC_0_1 {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %vC_1_0 = xegpu.load_nd %tileC_1_0 {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %vC_1_1 = xegpu.load_nd %tileC_1_1 {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %vC_2_0 = xegpu.load_nd %tileC_2_0 {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %vC_2_1 = xegpu.load_nd %tileC_2_1 {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %vC_3_0 = xegpu.load_nd %tileC_3_0 {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>
      %vC_3_1 = xegpu.load_nd %tileC_3_1 {mode = vc} : !xegpu.tensor_desc<8x16xf32> -> vector<8x16xf32>

      xegpu.compile_hint

      %out:20 = scf.for %subtileIv = %c0 to %numSubTilesK step %c1 iter_args(
        %tA_0_0 = %tileA_0_0,
        %tA_0_1 = %tileA_0_1,
        %tA_1_0 = %tileA_1_0,
        %tA_1_1 = %tileA_1_1,
        %tA_2_0 = %tileA_2_0,
        %tA_2_1 = %tileA_2_1,
        %tA_3_0 = %tileA_3_0,
        %tA_3_1 = %tileA_3_1,

        %tB_0_0 = %tileB_0_0,
        %tB_0_1 = %tileB_0_1,
        %tB_1_0 = %tileB_1_0,
        %tB_1_1 = %tileB_1_1,
        // %tB_1_0 = %tileB_1_0,
        // %tB_1_1 = %tileB_1_1,
        // %tB_1_2 = %tileB_1_2,
        // %tB_1_3 = %tileB_1_3,
        // %tB_2_0 = %tileB_2_0,
        // %tB_2_1 = %tileB_2_1,
        // %tB_2_2 = %tileB_2_2,
        // %tB_2_3 = %tileB_2_3,
        // %tB_3_0 = %tileB_3_0,
        // %tB_3_1 = %tileB_3_1,
        // %tB_3_2 = %tileB_3_2,
        // %tB_3_3 = %tileB_3_3,

        %acc_0_0 = %vC_0_0,
        %acc_0_1 = %vC_0_1,
        %acc_1_0 = %vC_1_0,
        %acc_1_1 = %vC_1_1,
        %acc_2_0 = %vC_2_0,
        %acc_2_1 = %vC_2_1,
        %acc_3_0 = %vC_3_0,
        %acc_3_1 = %vC_3_1
      ) -> (
        !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
        !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, //!xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, //!xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
        vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
      ) {
        // Load sub-tiles of A and B tiles from GMEM to SMEM.
        %vA_0_0 = xegpu.load_nd %tA_0_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %vA_0_1 = xegpu.load_nd %tA_0_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %vA_1_0 = xegpu.load_nd %tA_1_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %vA_1_1 = xegpu.load_nd %tA_1_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %vA_2_0 = xegpu.load_nd %tA_2_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %vA_2_1 = xegpu.load_nd %tA_2_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %vA_3_0 = xegpu.load_nd %tA_3_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %vA_3_1 = xegpu.load_nd %tA_3_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 1}: !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>

        %vB_0_0 = xegpu.load_nd %tB_0_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %vB_0_1 = xegpu.load_nd %tB_0_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %vB_1_0 = xegpu.load_nd %tB_1_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %vB_1_1 = xegpu.load_nd %tB_1_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_1_0 = xegpu.load_nd %tB_1_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_1_1 = xegpu.load_nd %tB_1_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_1_2 = xegpu.load_nd %tB_1_2 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_1_3 = xegpu.load_nd %tB_1_3 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_2_0 = xegpu.load_nd %tB_2_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_2_1 = xegpu.load_nd %tB_2_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_2_2 = xegpu.load_nd %tB_2_2 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_2_3 = xegpu.load_nd %tB_2_3 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_3_0 = xegpu.load_nd %tB_3_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_3_1 = xegpu.load_nd %tB_3_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_3_2 = xegpu.load_nd %tB_3_2 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        // %vB_3_3 = xegpu.load_nd %tB_3_3 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

        %nextTileA_0_0 = xegpu.update_nd_offset %tA_0_0, [%c0, %TN] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %nextTileA_0_1 = xegpu.update_nd_offset %tA_0_1, [%c0, %TN] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %nextTileA_1_0 = xegpu.update_nd_offset %tA_1_0, [%c0, %TN] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %nextTileA_1_1 = xegpu.update_nd_offset %tA_1_1, [%c0, %TN] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %nextTileA_2_0 = xegpu.update_nd_offset %tA_2_0, [%c0, %TN] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %nextTileA_2_1 = xegpu.update_nd_offset %tA_2_1, [%c0, %TN] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %nextTileA_3_0 = xegpu.update_nd_offset %tA_3_0, [%c0, %TN] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %nextTileA_3_1 = xegpu.update_nd_offset %tA_3_1, [%c0, %TN] {mode = vc}: !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>

        %nextTileB_0_0 = xegpu.update_nd_offset %tB_0_0, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        %nextTileB_0_1 = xegpu.update_nd_offset %tB_0_1, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        %nextTileB_1_0 = xegpu.update_nd_offset %tB_1_0, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        %nextTileB_1_1 = xegpu.update_nd_offset %tB_1_1, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_1_0 = xegpu.update_nd_offset %tB_1_0, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_1_1 = xegpu.update_nd_offset %tB_1_1, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_1_2 = xegpu.update_nd_offset %tB_1_2, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_1_3 = xegpu.update_nd_offset %tB_1_3, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_2_0 = xegpu.update_nd_offset %tB_0_0, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_2_1 = xegpu.update_nd_offset %tB_0_1, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_2_2 = xegpu.update_nd_offset %tB_0_2, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_2_3 = xegpu.update_nd_offset %tB_0_3, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_3_0 = xegpu.update_nd_offset %tB_1_0, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_3_1 = xegpu.update_nd_offset %tB_1_1, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_3_2 = xegpu.update_nd_offset %tB_1_2, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        // %nextTileB_3_3 = xegpu.update_nd_offset %tB_1_3, [%TN, %c0] {mode = vc}: !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>

        xegpu.prefetch_nd %nextTileA_0_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %nextTileA_0_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %nextTileA_1_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %nextTileA_1_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %nextTileA_2_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %nextTileA_2_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %nextTileA_3_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %nextTileA_3_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<8x16xf16>

        xegpu.prefetch_nd %nextTileB_0_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        xegpu.prefetch_nd %nextTileB_0_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        xegpu.prefetch_nd %nextTileB_1_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        xegpu.prefetch_nd %nextTileB_1_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_1_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_1_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_1_2 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_1_3 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_2_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_2_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_2_2 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_2_3 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_3_0 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_3_1 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_3_2 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>
        // xegpu.prefetch_nd %nextTileB_3_3 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached}: !xegpu.tensor_desc<16x16xf16>

        xegpu.compile_hint

        // GEMM computation
        %dpas_0_0_temp_0 = xegpu.dpas %vA_0_0, %vB_0_0, %acc_0_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_0_1_temp_0 = xegpu.dpas %vA_0_0, %vB_0_1, %acc_0_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_1_0_temp_0 = xegpu.dpas %vA_1_0, %vB_0_0, %acc_1_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_1_1_temp_0 = xegpu.dpas %vA_1_0, %vB_0_1, %acc_1_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_2_0_temp_0 = xegpu.dpas %vA_2_0, %vB_0_0, %acc_2_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_2_1_temp_0 = xegpu.dpas %vA_2_0, %vB_0_1, %acc_2_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_3_0_temp_0 = xegpu.dpas %vA_3_0, %vB_0_0, %acc_3_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_3_1_temp_0 = xegpu.dpas %vA_3_0, %vB_0_1, %acc_3_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        %dpas_0_0 = xegpu.dpas %vA_0_1, %vB_1_0, %dpas_0_0_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_0_1 = xegpu.dpas %vA_0_1, %vB_1_1, %dpas_0_1_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_1_0 = xegpu.dpas %vA_1_1, %vB_1_0, %dpas_1_0_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_1_1 = xegpu.dpas %vA_1_1, %vB_1_1, %dpas_1_1_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_2_0 = xegpu.dpas %vA_2_1, %vB_1_0, %dpas_2_0_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_2_1 = xegpu.dpas %vA_2_1, %vB_1_1, %dpas_2_1_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_3_0 = xegpu.dpas %vA_3_1, %vB_1_0, %dpas_3_0_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %dpas_3_1 = xegpu.dpas %vA_3_1, %vB_1_1, %dpas_3_1_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        // %dpas_0_0_temp_1 = xegpu.dpas %vA_0_1, %vB_1_0, %dpas_0_0_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_0_1_temp_1 = xegpu.dpas %vA_0_1, %vB_1_1, %dpas_0_1_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_0_2_temp_1 = xegpu.dpas %vA_0_1, %vB_1_2, %dpas_0_2_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_0_3_temp_1 = xegpu.dpas %vA_0_1, %vB_1_3, %dpas_0_3_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_0_temp_1 = xegpu.dpas %vA_1_1, %vB_1_0, %dpas_1_0_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_1_temp_1 = xegpu.dpas %vA_1_1, %vB_1_1, %dpas_1_1_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_2_temp_1 = xegpu.dpas %vA_1_1, %vB_1_2, %dpas_1_2_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_3_temp_1 = xegpu.dpas %vA_1_1, %vB_1_3, %dpas_1_3_temp_0 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        // %dpas_0_0_temp_2 = xegpu.dpas %vA_0_2, %vB_2_0, %dpas_0_0_temp_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_0_1_temp_2 = xegpu.dpas %vA_0_2, %vB_2_1, %dpas_0_1_temp_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_0_2_temp_2 = xegpu.dpas %vA_0_2, %vB_2_2, %dpas_0_2_temp_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_0_3_temp_2 = xegpu.dpas %vA_0_2, %vB_2_3, %dpas_0_3_temp_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_0_temp_2 = xegpu.dpas %vA_1_2, %vB_2_0, %dpas_1_0_temp_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_1_temp_2 = xegpu.dpas %vA_1_2, %vB_2_1, %dpas_1_1_temp_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_2_temp_2 = xegpu.dpas %vA_1_2, %vB_2_2, %dpas_1_2_temp_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_3_temp_2 = xegpu.dpas %vA_1_2, %vB_2_3, %dpas_1_3_temp_1 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        // %dpas_0_0 = xegpu.dpas %vA_0_3, %vB_3_0, %dpas_0_0_temp_2 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_0_1 = xegpu.dpas %vA_0_3, %vB_3_1, %dpas_0_1_temp_2 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_0_2 = xegpu.dpas %vA_0_3, %vB_3_2, %dpas_0_2_temp_2 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_0_3 = xegpu.dpas %vA_0_3, %vB_3_3, %dpas_0_3_temp_2 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_0 = xegpu.dpas %vA_1_3, %vB_3_0, %dpas_1_0_temp_2 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_1 = xegpu.dpas %vA_1_3, %vB_3_1, %dpas_1_1_temp_2 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_2 = xegpu.dpas %vA_1_3, %vB_3_2, %dpas_1_2_temp_2 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        // %dpas_1_3 = xegpu.dpas %vA_1_3, %vB_3_3, %dpas_1_3_temp_2 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>

        xegpu.compile_hint
        // Synchronize all threads in a threadblock.
        // All current computations have to be finished before SMEM A and B tiles can be
        // replaced with new values (new tiles) from GMEM.
        // TODO: see if needed, cache might be large enough to allow prefetching of the next set of tiles
        gpu.barrier

        scf.yield
          %nextTileA_0_0,
          %nextTileA_0_1,
          %nextTileA_1_0,
          %nextTileA_1_1,
          %nextTileA_2_0,
          %nextTileA_2_1,
          %nextTileA_3_0,
          %nextTileA_3_1,

          %nextTileB_0_0,
          %nextTileB_0_1,
          %nextTileB_1_0,
          %nextTileB_1_1,
          // %nextTileB_1_0,
          // %nextTileB_1_1,
          // %nextTileB_1_2,
          // %nextTileB_1_3,
          // %nextTileB_2_0,
          // %nextTileB_2_1,
          // %nextTileB_2_2,
          // %nextTileB_2_3,
          // %nextTileB_3_0,
          // %nextTileB_3_1,
          // %nextTileB_3_2,
          // %nextTileB_3_3,

          %dpas_0_0,
          %dpas_0_1,
          %dpas_1_0,
          %dpas_1_1,
          %dpas_2_0,
          %dpas_2_1,
          %dpas_3_0,
          %dpas_3_1
          : !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>,
            !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, //!xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, //!xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>,
            vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>
      }

      // Store the final C tile element values.
      xegpu.store_nd %out#12, %tileC_0_0 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %out#13, %tileC_0_1 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %out#14, %tileC_1_0 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %out#15, %tileC_1_1 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %out#16, %tileC_2_0 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %out#17, %tileC_2_1 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %out#18, %tileC_3_0 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>
      xegpu.store_nd %out#19, %tileC_3_1 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf32>

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
