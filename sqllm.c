#include "sqllm.h"
#include "ggml.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>


void ggml_vec_dot_q4_sq_fp16(const int n, float * restrict s, const void * restrict v, const ggml_fp16_t * restrict y) {

    const int nb = n / 8;

#ifdef __ARM_NEON

    // pointer initialization
    int32_t * baselut = v;
    int32_t * qweight = baselut + 8; // get start of row
    float * yvector = y;

    // initialize sum
    float16x8_t sumf1 = vdupq_n_f16(0);
    float16x8_t sumf2 = vdupq_n_f16(0);
    float16x8_t sumf3 = vdupq_n_f16(0);
    float16x8_t sumf4 = vdupq_n_f16(0);

    // initialize lookup table
    uint8x16_t lut1 = vld1q_u8(baselut);
    uint8x16_t lut2 = vld1q_u8(baselut+4);
    uint8x16_t lutl = vuzp1q_u8(lut1, lut2);
    uint8x16_t luth = vuzp2q_u8(lut1, lut2);

    for (int i = 0; i < nb; i+=4) {
        // get packed vector
        uint8x16_t m4b = vdupq_n_u8(0x0F);
        uint8x16_t packed_vector = vld1q_u8(&qweight[i]);

        // 4-bit -> 2 8-bit vectors
        uint8x16_t packed_vector_lb = vandq_u8  (packed_vector, m4b);
        uint8x16_t packed_vector_hb = vshrq_n_u8(packed_vector, 4);

        // get separate 8-bit indices (split across two vectors) by interleaving
        uint8x16_t packed_vector_0 = vzip1q_u8(packed_vector_lb, packed_vector_hb);
        uint8x16_t packed_vector_1 = vzip2q_u8(packed_vector_lb, packed_vector_hb);

        //perform table lookups
        uint8x16_t lookup_0l = vqtbl1q_u8 (lutl, packed_vector_0);
        uint8x16_t lookup_0h = vqtbl1q_u8 (luth, packed_vector_0);
        uint8x16_t lookup_1l = vqtbl1q_u8 (lutl, packed_vector_1);
        uint8x16_t lookup_1h = vqtbl1q_u8 (luth, packed_vector_1);

        // interleave lookup values
        float16x8_t lookup_0_z1 = vzip1q_u8(lookup_0l, lookup_0h);
        float16x8_t lookup_0_z2 = vzip2q_u8(lookup_0l, lookup_0h);
        float16x8_t lookup_1_z1 = vzip1q_u8(lookup_1l, lookup_1h);
        float16x8_t lookup_1_z2 = vzip2q_u8(lookup_1l, lookup_1h);

        //load int8 values
        float16x8_t tmp1 = vld1q_f16(&yvector[4*i]);
        float16x8_t tmp2 = vld1q_f16(&yvector[4*i+4]);
        float16x8_t tmp3 = vld1q_f16(&yvector[4*i+8]);
        float16x8_t tmp4 = vld1q_f16(&yvector[4*i+12]);

        //fp16 mul
        sumf1 = vfmaq_f16(sumf1, lookup_0_z1, tmp1);
        sumf2 = vfmaq_f16(sumf2, lookup_0_z2, tmp2);
        sumf3 = vfmaq_f16(sumf3, lookup_1_z1, tmp3);
        sumf4 = vfmaq_f16(sumf4, lookup_1_z2, tmp4);
    }

    float16x8_t sumf5 = vaddq_f16(sumf1, sumf2);
    float16x8_t sumf6 = vaddq_f16(sumf3, sumf4);
    float16x8_t sumf7 = vaddq_f16(sumf5, sumf6);

    float res = 0.0;
    const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 (sumf7));
    const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(sumf7));
    res = (float) vaddvq_f32(vaddq_f32(t0, t1));

    *s = res;

#else

    int32_t * baseptr = v;
    int32_t * qweight = baseptr + 8; // get start of row

    // scalar
    float sumf = 0.0;

    ggml_fp16_t * lut = v;
    for (int i = 0; i < nb; i++) {
        int32_t packed = qweight[i];

        for (int j = 0; j < 8; ++j) {
            const int idx = (packed >> j*4) & 0x0F;
            const ggml_fp16_t val = lut[idx];
            const ggml_fp16_t val2 = y[8*i+j];

            sumf += ggml_fp16_to_fp32(val) * ggml_fp16_to_fp32(val2);
        }
    }

    *s = sumf;

#endif
}
