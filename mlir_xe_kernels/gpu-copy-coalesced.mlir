// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-llvm.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%in: memref<1024x1024xf32>) -> memref<1024x1024xf32> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %in_gpu = gpu.alloc  host_shared () : memref<1024x1024xf32>
    memref.copy %in, %in_gpu : memref<1024x1024xf32> to memref<1024x1024xf32>
    %out_gpu = gpu.alloc  host_shared () : memref<1024x1024xf32>

    %matSizeX = memref.dim %out_gpu, %c0 : memref<1024x1024xf32>
    %matSizeY = memref.dim %out_gpu, %c1 : memref<1024x1024xf32>

    %tileX = arith.constant 16 : index
    %tileY = arith.constant 16 : index
    %bDimX = arith.divui %matSizeX, %tileX : index
    %bDimY = arith.divui %matSizeY, %tileY : index

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%bDimX, %bDimY, %c1) threads in (%c16, %c1, %c1) args(%in_gpu : memref<1024x1024xf32>, %out_gpu : memref<1024x1024xf32>)

    // %cast = memref.cast %out_gpu : memref<1024x1024xf32> to memref<*xf32>
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()

    gpu.dealloc  %in_gpu : memref<1024x1024xf32>
    return %out_gpu : memref<1024x1024xf32>
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<subgroup_size = 16>>} {
    gpu.func @test_kernel(%in: memref<1024x1024xf32>, %out: memref<1024x1024xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c256 = arith.constant 256 : index
      %c1024 = arith.constant 1024 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %3 = gpu.thread_id  x // Consecutive increase within warp threads.

      %bDimX = arith.constant 16 : index
      %bDimY = arith.constant 16 : index

      %blockOffsetRow = arith.muli %0, %bDimX : index
      %blockOffsetCol = arith.muli %1, %bDimY : index

      %tCol = arith.addi %blockOffsetCol, %3 : index

      // Each thread copies 16 elements from the input matrix with stride
      // equal to the warp size.
      // Coalesced GMEM access.
      scf.for %iv = %c0 to %bDimX step %c1 {
        %elemRow = arith.addi %blockOffsetRow, %iv : index
        %elem = memref.load %in[%elemRow, %tCol] : memref<1024x1024xf32>
        memref.store %elem, %out[%elemRow, %tCol] : memref<1024x1024xf32>
      }

      gpu.return
    }
  }
  memref.global "private" @in : memref<1024x1024xf32> = dense<3.0>
  func.func @main() attributes {llvm.emit_c_interface} {
    %in = memref.get_global @in : memref<1024x1024xf32>
    %2 = call @test(%in) : (memref<1024x1024xf32>) -> memref<1024x1024xf32>

    return
  }
}
