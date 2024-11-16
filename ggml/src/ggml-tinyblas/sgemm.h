// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#pragma once
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

namespace ggml::backend::tinyblas {

    // compute: C = Aᵀ * B
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
