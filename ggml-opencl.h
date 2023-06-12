#pragma once

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_cl_init(void);

enum ggml_blas_order {
    GGML_BLAS_ORDER_ROW_MAJOR = 101,
    GGML_BLAS_ORDER_COLUMN_MAJOR = 102,
};

enum ggml_blas_op {
    GGML_BLAS_OP_N = 111,
    GGML_BLAS_OP_T = 112,
    GGML_BLAS_OP_C = 113,
};

void ggml_cl_sgemm_wrapper(const enum ggml_blas_order order, const enum ggml_blas_op trans_a, const enum ggml_blas_op trans_b, const int m, const int n, const int k, const float alpha, const void *host_a, const int lda, const float *host_b, const int ldb, const float beta, float *host_c, const int ldc, const int btype);

#ifdef  __cplusplus
}
#endif
