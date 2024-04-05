#!/bin/sh

LLVM_ROOT=$(realpath ~/llvm-project)

IMEX_ROOT=$(realpath ~/mlir-extensions)
IMEX_RUNNER=$(realpath ${IMEX_ROOT}/build/bin/imex-runner.py)
KERNELS_DIR=$(realpath ${IMEX_ROOT}/build/bench_kernels)

TPP_ROOT=$(realpath ~/tpp-mlir)
TPP_RUN=$(realpath ${TPP_ROOT}/build/bin/tpp-run)

CONFIG_FILE=$(realpath ${IMEX_ROOT}/bench_kernels/gemm_benchmarks.txt)

# Initial validation.
if ! { [ -d ${IMEX_ROOT} ] || [ -f ${IMEX_RUNNER} ]; }; then
  echo "Missing IMEX repo"
  exit 1
fi
if ! [ -d ${KERNELS_DIR} ]; then
  echo "Missing kernels directory"
  exit 1
fi
if ! { [ -d ${TPP_ROOT} ] || [ -f ${TPP_RUN} ]; }; then
  echo "Missing TPP repo"
  exit 1
fi
if ! [ -f ${CONFIG_FILE} ]; then
  echo "Missing bench config file"
  exit 1
fi

# Benchmark config.
TPP_RUN_FLAGS="-gpu=intel -entry-point-result=void -e entry"

LIB_MLIR=$(realpath ${LLVM_ROOT}/build/lib/libmlir_runner_utils.so)
LIB_IMEX=$(realpath ${IMEX_ROOT}/build/lib/libimex_runner_utils.so)
LIB_SYCL=$(realpath ${IMEX_ROOT}/build/lib/libsycl-runtime.so)

IMEX_SHARED_LIBS="--shared-libs=${LIB_MLIR},${LIB_IMEX},${LIB_SYCL}"
IMEX_RUNNER_FLAGS="--requires=sycl-runtime --runner imex-cpu-runner -e entry --entry-point-result=void"

IMEX_ENV="IMEX_ENABLE_LARGE_REG_FILE=1 IMEX_ENABLE_PROFILING=1 IMEX_PROFILING_RUNS=1000 IMEX_PROFILING_WARMUPS=10"

PIPELINE_FILE=$(realpath ${IMEX_ROOT}/benchmarks/pipelines/linalg-to-gpu.pp)

# Run benchmarks.
while read LINE || [[ -n $LINE ]]; do
  TOKENS=(${LINE})
  FILE=${KERNELS_DIR}/${TOKENS[0]}
  FLAGS=${TOKENS[@]:1}

  # Lower input kernels to XeGPU dialect using TPP.
  # Filter out possible error messages.
  # Convert xegpux (present due to conflict with upstream) to downstream xegpu.
  MLIR_XEGPU=$(exec ${TPP_RUN} ${FILE} ${TPP_RUN_FLAGS} ${FLAGS} 2>&1 | sed '/Error:.*/d' | sed 's/xegpux/xegpu/g' )

  # Use IMEX runner to finalize code generation and benchmark performance.
  echo "${MLIR_XEGPU}" | \
    exec env ${IMEX_ENV} python ${IMEX_RUNNER} ${IMEX_RUNNER_FLAGS} ${IMEX_SHARED_LIBS} \
         --pass-pipeline-file=${PIPELINE_FILE} 2>/dev/null | \
    sed -nE "s/.*kernel execution time.*avg\: ([0-9.]+).*/\\1/p"
done < ${CONFIG_FILE}
