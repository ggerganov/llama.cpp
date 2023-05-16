#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

enum ggml_blas_order {
    GGML_BLAS_ORDER_ROW_MAJOR = 101,
    GGML_BLAS_ORDER_COLUMN_MAJOR = 102,
};

enum ggml_blas_op {
    GGML_BLAS_OP_N = 111,
    GGML_BLAS_OP_T = 112,
    GGML_BLAS_OP_C = 113,
};

void ggml_cl_init(void);

bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

void * ggml_cl_host_malloc(size_t size);
void   ggml_cl_host_free(void * ptr);

void ggml_cl_transform_tensor(struct ggml_tensor * tensor);

void ggml_cl_sgemm_wrapper(const enum ggml_blas_order order, const enum ggml_blas_op trans_a, const enum ggml_blas_op trans_b, const int m, const int n, const int k, const float alpha, const void *host_a, const int lda, const float *host_b, const int ldb, const float beta, float *host_c, const int ldc, const int btype);

#ifdef  __cplusplus
}
#endif
