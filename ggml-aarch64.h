// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_row_q8_0_aarch64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k, int nrows_interleaved, int blocklen_per_row);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
size_t quantize_q4_0_aarch64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

block_q4_0x4 make_block_q4_0x4(const block_q4_0 * const in[4], unsigned int block_len, unsigned int xor_mask);
block_q4_0x8 make_block_q4_0x8(const block_q4_0 * const in[8], unsigned int block_len, unsigned int xor_mask);
block_q8_0x4 make_block_q8_0x4(const block_q8_0 * const in[4], unsigned int block_len);
block_q8_0x8 make_block_q8_0x8(const block_q8_0 * const in[8], unsigned int block_len);

// GEMV
void ggml_gemv_q4_0_q8_0_aarch64_sve256     (int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth);
void ggml_gemv_q4_0_q8_0_aarch64_neon       (int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth);
void ggml_gemv_q4_0_q8_0_aarch64_neon_noi8mm(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth);
void ggml_gemv_q8_0_q8_0_aarch64_sve256     (int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth);
void ggml_gemv_q8_0_q8_0_aarch64_neon       (int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth);

// GEMM
void ggml_gemm_q4_0_q8_0_aarch64_sve256     (int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth);
void ggml_gemm_q4_0_q8_0_aarch64_neon       (int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth);
void ggml_gemm_q4_0_q8_0_aarch64_neon_noi8mm(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth);
void ggml_gemm_q8_0_q8_0_aarch64            (int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, int ith, int nth);

#ifdef __cplusplus
}
#endif

