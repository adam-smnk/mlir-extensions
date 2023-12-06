
file(STRINGS gemm.mbs.in test_mbs)
file(STRINGS gemm.sizes.in test_sizes)

foreach(MB ${test_mbs})
    foreach(SIZE ${test_sizes})
        configure_file(gemm_xegpu.mlir.in ${IMEX_BINARY_DIR}/benchmarks/gemm/gpu/gemm_xegpu_${MB}_${SIZE}_f16.mlir @ONLY)
        configure_file(gemm_xetile.mlir.in ${IMEX_BINARY_DIR}/benchmarks/gemm/gpu/gemm_xetile_${MB}_${SIZE}_f16.mlir @ONLY)
        configure_file(gemm_xegpu_param.mlir.in ${IMEX_BINARY_DIR}/benchmarks/gemm/gpu/gemm_xegpu_param_${MB}_${SIZE}_f16.mlir @ONLY)
        configure_file(gemm_xegpu_coarse.mlir.in ${IMEX_BINARY_DIR}/benchmarks/gemm/gpu/gemm_xegpu_coarse_${MB}_${SIZE}_f16.mlir @ONLY)
    endforeach()
endforeach()
