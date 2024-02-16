#pragma once

#include "ggml-impl.h"

// GGML internal header

#include <stdint.h>
#include <stddef.h>

#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    ggml_fp16_t d;          // delta
    ggml_fp16_t m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_fp16_t) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
typedef struct {
    ggml_fp16_t d;         // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    ggml_fp16_t d;         // delta
    ggml_fp16_t m;         // min
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
    ggml_fp16_t d;         // delta
    int8_t  qs[QK8_0];     // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
    float d;               // delta
    float s;               // d * sum(qs[i])
    int8_t  qs[QK8_1];     // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(float) + QK8_1, "wrong q8_1 block size/padding");

//
// Super-block quantization structures
//

// Super-block size
#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    ggml_fp16_t d;           // super-block scale for quantized scales
    ggml_fp16_t dmin;        // super-block scale for quantized mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
#ifdef GGML_QKK_64
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[2];
    ggml_fp16_t d;             // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + 2, "wrong q3_K block size/padding");
#else
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[12];        // scales, quantized with 6 bits
    ggml_fp16_t d;             // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");
#endif

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
#ifdef GGML_QKK_64
typedef struct {
    ggml_fp16_t d[2];          // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + QK_K/2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {
    ggml_fp16_t d;             // super-block scale for quantized scales
    ggml_fp16_t dmin;          // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");
#endif

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
#ifdef GGML_QKK_64
typedef struct {
    ggml_fp16_t d;               // super-block scale
    int8_t  scales[QK_K/16];     // 8-bit block scales
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == sizeof(ggml_fp16_t) + QK_K/2 + QK_K/8 + QK_K/16, "wrong q5_K block size/padding");
#else
typedef struct {
    ggml_fp16_t d;               // super-block scale for quantized scales
    ggml_fp16_t dmin;            // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];   // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");
#endif

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_fp16_t d;           // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

// This is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");

// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_fp16_t d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_fp16_t) + QK_K/8*sizeof(uint16_t), "wrong iq2_xxs block size/padding");

// 2.3125 bpw quants
typedef struct {
    ggml_fp16_t d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == sizeof(ggml_fp16_t) + QK_K/8*sizeof(uint16_t) + QK_K/32, "wrong iq2_xs block size/padding");

// (Almost) "true" 3-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 3.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_fp16_t d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_fp16_t) + 3*(QK_K/8), "wrong iq3_xxs block size/padding");

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_row_q4_0_reference(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int k);
void quantize_row_q4_1_reference(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int k);
void quantize_row_q5_0_reference(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int k);
void quantize_row_q5_1_reference(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int k);
void quantize_row_q8_0_reference(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int k);
void quantize_row_q8_1_reference(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int k);

void quantize_row_q2_K_reference(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int k);
void quantize_row_q3_K_reference(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int k);
void quantize_row_q4_K_reference(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int k);
void quantize_row_q5_K_reference(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int k);
void quantize_row_q6_K_reference(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int k);
void quantize_row_q8_K_reference(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int k);
void quantize_row_iq3_xxs_reference(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int k);

void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);
void quantize_row_iq3_xxs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int k);

// Dequantization
void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
//void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

// Dot product
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

//
// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
//
size_t quantize_iq2_xxs(const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_iq2_xs (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_iq3_xxs(const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_q2_K   (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_q3_K   (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_q4_K   (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_q5_K   (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_q6_K   (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_q4_0   (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_q4_1   (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_q5_0   (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);
size_t quantize_q5_1   (const float * src, void * dst, int nrows, int n_per_row, int64_t * hist, const float * imatrix);

void iq2xs_init_impl(int grid_size);
void iq2xs_free_impl(int grid_size);
void iq3xs_init_impl(int grid_size);
void iq3xs_free_impl(int grid_size);

#ifdef __cplusplus
}
#endif

