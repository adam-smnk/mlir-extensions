!A_t = tensor<4096x4096xf16>
!B_t = tensor<4096x4096xf16>
!C_t = tensor<4096x4096xf16>

func.func @entry(%arg0: !A_t, %arg1: !B_t, %arg2: !C_t) -> !C_t {
  %0 = linalg.matmul ins(%arg0, %arg1 : !A_t, !B_t)
                     outs(%arg2 : !C_t) -> !C_t
  return %0 : !C_t
}
