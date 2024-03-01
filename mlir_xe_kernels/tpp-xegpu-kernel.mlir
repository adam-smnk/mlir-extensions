// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0, d1) -> (d0 * 128 + d1 * 32)>
#map2 = affine_map<(d0, d1) -> (d0 * 128 + d1 * 32 + 16)>
#map3 = affine_map<(d0) -> (d0 * 32 + 8)>
#map4 = affine_map<(d0) -> (d0 * 32 + 16)>
#map5 = affine_map<(d0) -> (d0 * 32 + 24)>
#map6 = affine_map<(d0, d1) -> (d0 * 32 + d1 * 128)>
#map7 = affine_map<(d0, d1) -> (d0 * 32 + d1 * 128 + 16)>

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<128x1024xf16>, %B: memref<1024x1024xf16>, %C: memref<128x1024xf16>) -> memref<128x1024xf16> attributes {llvm.emit_c_interface} {
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
    %C_gpu = gpu.alloc  host_shared () : memref<128x1024xf16>
    memref.copy %C, %C_gpu : memref<128x1024xf16> to memref<128x1024xf16>

    %cast = memref.cast %C_gpu : memref<128x1024xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c8, %c1, %c1) threads in (%c4, %c4, %c1)  args(%C_gpu : memref<128x1024xf16>, %A_gpu : memref<128x1024xf16>, %B_gpu : memref<1024x1024xf16>)

    // %cast = memref.cast %C_gpu : memref<128x1024xf16> to memref<*xf16>
    // call @printMemrefF16(%cast) : (memref<*xf16>) -> ()

    gpu.dealloc  %A_gpu : memref<128x1024xf16>
    gpu.dealloc  %B_gpu : memref<1024x1024xf16>
    return %C_gpu : memref<128x1024xf16>
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<subgroup_size = 16>>} {
    gpu.func @test_kernel(%arg0: memref<128x1024xf16>, %arg1: memref<128x1024xf16>, %arg2: memref<1024x1024xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c1024 = arith.constant 1024 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = gpu.thread_id  y
      %3 = affine.apply #map(%1)
      %4 = affine.apply #map1(%0, %2)
      %5 = xegpu.create_nd_tdesc %arg0[%3, %4] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %6 = affine.apply #map2(%0, %2)
      %7 = xegpu.create_nd_tdesc %arg0[%3, %6] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %8 = affine.apply #map3(%1)
      %9 = xegpu.create_nd_tdesc %arg0[%8, %4] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %10 = xegpu.create_nd_tdesc %arg0[%8, %6] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %11 = affine.apply #map4(%1)
      %12 = xegpu.create_nd_tdesc %arg0[%11, %4] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %13 = xegpu.create_nd_tdesc %arg0[%11, %6] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %14 = affine.apply #map5(%1)
      %15 = xegpu.create_nd_tdesc %arg0[%14, %4] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %16 = xegpu.create_nd_tdesc %arg0[%14, %6] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %17 = xegpu.load_nd %5 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %18 = xegpu.load_nd %7 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %19 = xegpu.load_nd %9 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %20 = xegpu.load_nd %10 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %21 = xegpu.load_nd %12 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %22 = xegpu.load_nd %13 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %23 = xegpu.load_nd %15 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      %24 = xegpu.load_nd %16 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x16xf16>
      xegpu.compile_hint
      %25 = arith.extf %17 : vector<8x16xf16> to vector<8x16xf32>
      %26 = arith.extf %18 : vector<8x16xf16> to vector<8x16xf32>
      %27 = arith.extf %19 : vector<8x16xf16> to vector<8x16xf32>
      %28 = arith.extf %20 : vector<8x16xf16> to vector<8x16xf32>
      %29 = arith.extf %21 : vector<8x16xf16> to vector<8x16xf32>
      %30 = arith.extf %22 : vector<8x16xf16> to vector<8x16xf32>
      %31 = arith.extf %23 : vector<8x16xf16> to vector<8x16xf32>
      %32 = arith.extf %24 : vector<8x16xf16> to vector<8x16xf32>
      %33 = xegpu.create_nd_tdesc %arg1[%3, 0] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %34 = xegpu.create_nd_tdesc %arg1[%3, 16] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %35 = xegpu.create_nd_tdesc %arg1[%8, 0] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %36 = xegpu.create_nd_tdesc %arg1[%8, 16] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %37 = xegpu.create_nd_tdesc %arg1[%11, 0] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %38 = xegpu.create_nd_tdesc %arg1[%11, 16] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %39 = xegpu.create_nd_tdesc %arg1[%14, 0] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %40 = xegpu.create_nd_tdesc %arg1[%14, 16] {mode = vc} : memref<128x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %41 = affine.apply #map6(%2, %0)
      %42 = xegpu.create_nd_tdesc %arg2[0, %41] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      %43 = affine.apply #map7(%2, %0)
      %44 = xegpu.create_nd_tdesc %arg2[0, %43] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      %45 = xegpu.create_nd_tdesc %arg2[16, %41] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      %46 = xegpu.create_nd_tdesc %arg2[16, %43] {mode = vc} : memref<1024x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      %47:20 = scf.for %arg3 = %c0 to %c1024 step %c32 iter_args(%arg4 = %25, %arg5 = %26, %arg6 = %27, %arg7 = %28, %arg8 = %29, %arg9 = %30, %arg10 = %31, %arg11 = %32, %arg12 = %33, %arg13 = %34, %arg14 = %35, %arg15 = %36, %arg16 = %37, %arg17 = %38, %arg18 = %39, %arg19 = %40, %arg20 = %42, %arg21 = %44, %arg22 = %45, %arg23 = %46) -> (vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>) {
        %56 = xegpu.load_nd %arg12 {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %57 = xegpu.load_nd %arg13 {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %58 = xegpu.load_nd %arg14 {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %59 = xegpu.load_nd %arg15 {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %60 = xegpu.load_nd %arg16 {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %61 = xegpu.load_nd %arg17 {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %62 = xegpu.load_nd %arg18 {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %63 = xegpu.load_nd %arg19 {mode = vc, vnni_axis = 1, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %64 = xegpu.load_nd %arg20 {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %65 = xegpu.load_nd %arg21 {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %66 = xegpu.load_nd %arg22 {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %67 = xegpu.load_nd %arg23 {mode = vc, vnni_axis = 0, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %68 = xegpu.update_nd_offset %arg12, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %69 = xegpu.update_nd_offset %arg13, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %70 = xegpu.update_nd_offset %arg14, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %71 = xegpu.update_nd_offset %arg15, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %72 = xegpu.update_nd_offset %arg16, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %73 = xegpu.update_nd_offset %arg17, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %74 = xegpu.update_nd_offset %arg18, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %75 = xegpu.update_nd_offset %arg19, [%c0, %c32] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %76 = xegpu.update_nd_offset %arg20, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        %77 = xegpu.update_nd_offset %arg21, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        %78 = xegpu.update_nd_offset %arg22, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        %79 = xegpu.update_nd_offset %arg23, [%c32, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        xegpu.prefetch_nd %68 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %69 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %70 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %71 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %72 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %73 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %74 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %75 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<8x16xf16>
        xegpu.prefetch_nd %76 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<16x16xf16>
        xegpu.prefetch_nd %77 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<16x16xf16>
        xegpu.prefetch_nd %78 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<16x16xf16>
        xegpu.prefetch_nd %79 {mode = vc, l1_hint = cached, l2_hint = cached, l3_hint = cached} : !xegpu.tensor_desc<16x16xf16>
        xegpu.compile_hint
        %80 = xegpu.dpas %56, %64, %arg4 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %81 = xegpu.dpas %56, %65, %arg5 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %82 = xegpu.dpas %58, %64, %arg6 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %83 = xegpu.dpas %58, %65, %arg7 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %84 = xegpu.dpas %60, %64, %arg8 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %85 = xegpu.dpas %60, %65, %arg9 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %86 = xegpu.dpas %62, %64, %arg10 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %87 = xegpu.dpas %62, %65, %arg11 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %88 = xegpu.dpas %57, %66, %80 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %89 = xegpu.dpas %57, %67, %81 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %90 = xegpu.dpas %59, %66, %82 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %91 = xegpu.dpas %59, %67, %83 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %92 = xegpu.dpas %61, %66, %84 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %93 = xegpu.dpas %61, %67, %85 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %94 = xegpu.dpas %63, %66, %86 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %95 = xegpu.dpas %63, %67, %87 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        xegpu.compile_hint
        gpu.barrier
        scf.yield %88, %89, %90, %91, %92, %93, %94, %95, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79 : vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
      }
      %48 = arith.truncf %47#0 : vector<8x16xf32> to vector<8x16xf16>
      %49 = arith.truncf %47#1 : vector<8x16xf32> to vector<8x16xf16>
      %50 = arith.truncf %47#2 : vector<8x16xf32> to vector<8x16xf16>
      %51 = arith.truncf %47#3 : vector<8x16xf32> to vector<8x16xf16>
      %52 = arith.truncf %47#4 : vector<8x16xf32> to vector<8x16xf16>
      %53 = arith.truncf %47#5 : vector<8x16xf32> to vector<8x16xf16>
      %54 = arith.truncf %47#6 : vector<8x16xf32> to vector<8x16xf16>
      %55 = arith.truncf %47#7 : vector<8x16xf32> to vector<8x16xf16>
      xegpu.store_nd %48, %5 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %49, %7 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %50, %9 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %51, %10 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %52, %12 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %53, %13 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %54, %15 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %55, %16 {mode = vc, l1_hint = write_back, l2_hint = write_back, l3_hint = write_back} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
  memref.global "private" @matA : memref<128x1024xf16> = dense<1.0>
  memref.global "private" @matB : memref<1024x1024xf16> = dense<0.1>
  memref.global "private" @matC : memref<128x1024xf16> = dense<0.0>
  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.get_global @matA : memref<128x1024xf16>
    %B = memref.get_global @matB : memref<1024x1024xf16>
    %C = memref.get_global @matC : memref<128x1024xf16>
    %2 = call @test(%A, %B, %C) : (memref<128x1024xf16>, memref<1024x1024xf16>, memref<128x1024xf16>) -> memref<128x1024xf16>

    return
  }
}
