#pragma once

#include "ggml_v2-opencl.h"

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_v2_cl_init_legacy(void);

void ggml_v2_cl_sgemm_wrapper_legacy(const enum ggml_v2_blas_order order, const enum ggml_v2_blas_op trans_a, const enum ggml_v2_blas_op trans_b, const int m, const int n, const int k, const float alpha, const void *host_a, const int lda, const float *host_b, const int ldb, const float beta, float *host_c, const int ldc, const int btype);

#ifdef  __cplusplus
}
#endif
