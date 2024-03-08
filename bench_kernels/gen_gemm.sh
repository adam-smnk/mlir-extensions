#!/bin/sh

IMEX_ROOT=$(realpath ~/mlir-extensions)
TEMPLATE_FILE=$(realpath ${IMEX_ROOT}/bench_kernels/gemm_M_N_K.mlir)

# Initial validation.
if ! [ -d ${IMEX_ROOT} ]; then
  echo "Missing IMEX repo"
  exit 1
fi
if ! [ -f ${TEMPLATE_FILE} ]; then
  echo "Missing template MLIR file"
  exit 1
fi

OUT_DIR=$(realpath ${IMEX_ROOT}/build/bench_kernels)
mkdir -p ${OUT_DIR}

# Kernel config.
MBS=( 128 256 512 )
SIZES=( 1024 2048 4096 8192 )

# Generate files.
echo "Generating kernels..."
for MB in "${MBS[@]}"; do
  for SIZE in "${SIZES[@]}"; do
    OUT_FILE="gemm_${MB}_${SIZE}_${SIZE}.mlir"
    echo ${OUT_FILE}
    exec sed -E "s/@DIMS_A@/${MB}x${SIZE}/g" ${TEMPLATE_FILE} | \
         sed -E "s/@DIMS_B@/${SIZE}x${SIZE}/g" | \
         sed -E "s/@DIMS_C@/${MB}x${SIZE}/g" > \
         ${OUT_DIR}/${OUT_FILE}
  done
done
