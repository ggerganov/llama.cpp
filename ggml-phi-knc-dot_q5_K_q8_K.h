// Formatted with: indent -npcs -nlp -i4 -l300
#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C"
{
#endif

    /* A forward declaration, to keep GCC happy. */
    void ggml_vec_dot_q5_K_q8_K(int n, float *restrict s, size_t bs, const void *restrict vx, size_t bx, const void *restrict vy, size_t by, int nrc);
    // Force an alignment onto these vectors.
    typedef float float32x16_t __attribute__((vector_size (64), aligned(64)));
    typedef int8_t int8x16_t __attribute__((vector_size (16), aligned(16)));
    typedef uint8_t uint8x16_t __attribute__((vector_size (16), aligned(16)));
    typedef int32_t int32x16_t __attribute__((vector_size (64), aligned(64)));

    // Zero out a vector of Floats
    void GGML_F32x16_VEC_ZERO(float32x16_t *target);
    // Convert an FP16 value to FP32(Float).
    float GGML_PHI_FP16_TO_FP32(ggml_fp16_t src);
    // Convert an FP32 value to FP16.
    ggml_fp16_t GGML_PHI_FP32_TO_FP16(float src);
    // Create a 5 bit int vector from a 4 bit vector and a 1 bit vector, both in packed forms.
    void GGML_5bit_Unpack_Unaligned (const uint8x16_t * q4, const uint8_t * q1, uint8x16_t * dst);
    // Multiply a Q5 and Q8 vector against each other, with some scaling.
    void GGML_8X_2xI8x16_2xI8x16_MUL_2xI16x16_S_FMA_I32x16_Unaligned (const int8x16_t *q8, uint8x16_t *q5, const uint8_t *scale, ggml_fp16_t scaleX, float scaleY, float32x16_t *res);

#ifdef  __cplusplus
}
#endif
