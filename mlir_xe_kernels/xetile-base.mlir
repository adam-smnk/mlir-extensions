// TODO: Add imex-runner commands
// RUN:

// NOTES :
// This example assumes one subgroup per one workgroup and the kernel specifies the computation
// done by a single subgroup.

module @gemm attributes {gpu.container_module} {
  func.func @test(%A: memref<128x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<128x4096xf32>) -> memref<128x4096xf32> attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %A_gpu = gpu.alloc  host_shared () : memref<128x4096xf16>
    memref.copy %A, %A_gpu : memref<128x4096xf16> to memref<128x4096xf16>
    %B_gpu = gpu.alloc  host_shared () : memref<4096x4096xf16>
    memref.copy %B, %B_gpu : memref<4096x4096xf16> to memref<4096x4096xf16>
    %C_gpu = gpu.alloc  host_shared () : memref<128x4096xf32>
    memref.copy %C, %C_gpu : memref<128x4096xf32> to memref<128x4096xf32>

    %matSizeX = memref.dim %C, %c0 : memref<128x4096xf32>
    %matSizeY = memref.dim %C, %c1 : memref<128x4096xf32>

    %tileX = arith.constant 16 : index
    %tileY = arith.constant 32 : index
    %bDimX = arith.divui %matSizeX, %tileX : index
    %bDimY = arith.divui %matSizeY, %tileY : index

    gpu.launch_func  @test_kernel::@test_kernel blocks in (%bDimX, %bDimY, %c1) threads in (%c1, %c1, %c1) args(%A_gpu : memref<128x4096xf16>, %B_gpu : memref<4096x4096xf16>, %C_gpu : memref<128x4096xf32>)

    // %cast = memref.cast %C_gpu : memref<128x4096xf32> to memref<*xf32>
    // call @printMemrefF32(%cast) : (memref<*xf32>) -> ()

    gpu.dealloc  %A_gpu : memref<128x4096xf16>
    gpu.dealloc  %B_gpu : memref<4096x4096xf16>
    return %C_gpu : memref<128x4096xf32>
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%A: memref<128x4096xf16>, %B: memref<4096x4096xf16>, %C: memref<128x4096xf32>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index

      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y

      %tileDimX = arith.constant 16 : index
      %tileDimY = arith.constant 64 : index

      %m = arith.muli %block_id_x, %tileDimX : index
      %n = arith.muli %block_id_y, %tileDimY : index
      // intialize C tile and load it
      %c_init_tile = xetile.init_tile %C[%m, %n] : memref<128x4096xf32> -> !xetile.tile<16x32xf32>
      %c_init_value = xetile.load_tile %c_init_tile  : !xetile.tile<16x32xf32> -> vector<16x32xf32>
      // initalize A and B tiles
      %a_init_tile = xetile.init_tile %A[%m, %c0] : memref<128x4096xf16> -> !xetile.tile<16x64xf16>
      %b_init_tile = xetile.init_tile %B[%c0, %n] : memref<4096x4096xf16> -> !xetile.tile<64x32xf16>

      xetile.prefetch_tile %a_init_tile : !xetile.tile<16x64xf16>
      xetile.prefetch_tile %b_init_tile : !xetile.tile<64x32xf16>

      // compute the value of C tile by iterating over tiles in k-dimension and doing dpas
      %out:3 = scf.for %k = %c0 to %c4096 step %tileDimY
        iter_args(%a_tile = %a_init_tile, %b_tile = %b_init_tile, %c_value = %c_init_value)
        -> (!xetile.tile<16x64xf16>, !xetile.tile<64x32xf16>, vector<16x32xf32>) {

        // load A and B tiles
        %a_value = xetile.load_tile %a_tile : !xetile.tile<16x64xf16> -> vector<16x64xf16>
        %b_value = xetile.load_tile %b_tile : !xetile.tile<64x32xf16> -> vector<64x32xf16>

        // update the offsets for A and B tiles
        %a_next_tile = xetile.update_tile_offset %a_tile, [%c0, %tileDimY]
          : !xetile.tile<16x64xf16>, index, index -> !xetile.tile<16x64xf16>
        %b_next_tile = xetile.update_tile_offset %b_tile, [%tileDimY, %c0]
          : !xetile.tile<64x32xf16>, index, index -> !xetile.tile<64x32xf16>

        // xetile.prefetch_tile %a_next_tile : !xetile.tile<16x64xf16>
        // xetile.prefetch_tile %b_next_tile : !xetile.tile<64x32xf16>

        // perform dpas and accumulate
        %c_new_value = xetile.tile_mma %a_value, %b_value, %c_value
          : vector<16x64xf16>, vector<64x32xf16>, vector<16x32xf32> -> vector<16x32xf32>

        // partial C tile result
        scf.yield %a_next_tile, %b_next_tile, %c_new_value
          : !xetile.tile<16x64xf16>, !xetile.tile<64x32xf16>, vector<16x32xf32>
      }
      // store the final accumulated C tile result back to memory
      xetile.store_tile %out#2, %c_init_tile: vector<16x32xf32>, !xetile.tile<16x32xf32>
      gpu.return
    }
  }
  memref.global "private" @matA : memref<128x4096xf16> = dense<1.0>
  memref.global "private" @matB : memref<4096x4096xf16> = dense<1.0>
  memref.global "private" @matC : memref<128x4096xf32> = dense<0.0>
  func.func @main() attributes {llvm.emit_c_interface} {
    %A = memref.get_global @matA : memref<128x4096xf16>
    %B = memref.get_global @matB : memref<4096x4096xf16>
    %C = memref.get_global @matC : memref<128x4096xf32>
    %2 = call @test(%A, %B, %C) : (memref<128x4096xf16>, memref<4096x4096xf16>, memref<128x4096xf32>) -> memref<128x4096xf32>

    return
  }
}
