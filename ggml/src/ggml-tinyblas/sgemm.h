#pragma once
//#include <cstdint>
#include "ggml.h"
#define GGML_COMMON_DECL_C
//#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

// appelé que depuis du c++ (le tinyBLAS backend)

namespace ggml::backend::tinyblas {

    // on est en C++
    //  => on peu avoir autant de fonction que de type.
    // calcule C = Aᵀ * B
    template<bool RUN>
    bool gemm(int64_t m, int64_t n, int64_t k,
              const float *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc,
              int ith=0, int nth=1);
    template<bool RUN>
    bool gemm(int64_t m, int64_t n, int64_t k,
              const ggml_fp16_t *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc,
              int ith=0, int nth=1);
    template<bool RUN>
    bool gemm(int64_t m, int64_t n, int64_t k,
              const ggml_fp16_t *A, int64_t lda, const ggml_fp16_t *B, int64_t ldb, float *C, int64_t ldc,
              int ith=0, int nth=1);
    template<bool RUN>
    bool gemm(int64_t m, int64_t n, int64_t k,
              const ggml_bf16_t *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc,
              int ith=0, int nth=1);
    template<bool RUN>
    bool gemm(int64_t m, int64_t n, int64_t k,
              const ggml_bf16_t *A, int64_t lda, const ggml_bf16_t *B, int64_t ldb, float *C, int64_t ldc,
              int ith=0, int nth=1);
    template<bool RUN>
    bool gemm(int64_t m, int64_t n, int64_t k,
              const block_q8_0 *A, int64_t lda, const block_q8_0 *B, int64_t ldb, float *C, int64_t ldc,
              int ith=0, int nth=1);
    template<bool RUN>
    bool gemm(int64_t m, int64_t n, int64_t k,
              const block_q4_0 *A, int64_t lda, const block_q8_0 *B, int64_t ldb, float *C, int64_t ldc,
              int ith=0, int nth=1);
    template<bool RUN>
    bool gemm(int64_t m, int64_t n, int64_t k,
              const block_q5_0 *A, int64_t lda, const block_q8_0 *B, int64_t ldb, float *C, int64_t ldc,
              int ith=0, int nth=1);
    template<bool RUN>
    bool gemm(int64_t m, int64_t n, int64_t k,
              const block_iq4_nl *A, int64_t lda, const block_q8_0 *B, int64_t ldb, float *C, int64_t ldc,
              int ith=0, int nth=1);
}
