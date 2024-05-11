/* Xeon PHI IMCI support. */
/* formatted by using emacs, with (M-x set-variable RET c-basic-offset RET 4 RET) executed. */
/* formatted by using emacs, with (M-x set-variable RET indent-tabs-mode RET nil RET) executed. */

// For uint32_t
#include <stdint.h>

// For size_t
#include <stdio.h>

// Yes, we have to tell this header to actually export stuff.
#define GGML_COMMON_IMPL_C
#include "ggml-quants.h"
#include "ggml-impl.h"

// For block_q5_K and block_q8_K.
#include "ggml-common.h"

// For our vector types, and forward declarations.
#include "ggml-phi-knc-dot_q5_K_q8_K.h"

// We can fit 16 float32s in a single vector register.
#define GGML_F32_EPR 16

/* Clear a vector of 16 floats. */
void GGML_F32x16_VEC_ZERO(float32x16_t *target)
{
    uint8_t zero=0;

    __asm__ __volatile__ (
                          "vbroadcastss\t%[Z]%{uint8%},\t%%zmm0\n\t" // use an upscaling operator to clear our register.
                          "vmovaps\t\t%%zmm0,\t%[RES]\n\t"
                          : [RES]  "+m"  (*target)
                          : [Z]     "m"  (zero)
                          : "zmm0", "memory");
}

/* Convert a FP16 to a FP32. */
float GGML_PHI_FP16_TO_FP32(ggml_fp16_t src)
{
    // we only care aboun one result.
    uint32_t mask=0x0001;

    // we declare this as an array, so it ends up in a different memory section.
    float f32[1] __attribute__((aligned(64)));

    __asm__ __volatile__ (
                          "kmov\t%[M],\t%%k1\n\t"
                          "vbroadcastss\t%[SRC]%{float16%},\t%%zmm1%{%%k1%}\n\t"
                          "vmovaps\t\t%%zmm1,\t%[DST]%{%%k1%}\n\t"
                          : [DST] "+m"  (f32)
                          : [SRC]  "m"  (src),
                            [M]    "r"  (mask)
                          : "zmm1", "memory", "k1");
    return f32[0];
}

/* convert many FP16s to FP32s. */
void GGML_PHI_FP16_TO_FP32_ROW(const ggml_fp16_t * x, float * y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = GGML_PHI_FP16_TO_FP32(x[i]);
    }
}

/* Convert a FP32 to a FP16. */
ggml_fp16_t GGML_PHI_FP32_TO_FP16(float src)
{
    uint32_t mask=0x0001;

    // we declare this as an array, so it ends up in a different memory section.
    ggml_fp16_t f16[1] __attribute__((aligned(64)));

    __asm__ __volatile__ (
                          "kmov\t%[M],\t%%k1\n\t"
                          "vbroadcastss\t%[SRC],\t%%zmm2%{%%k1%}\n\t"
                          "vmovaps\t\t%%zmm2%{float16%},\t%[DST]%{%%k1%}\n\t"
                          : [DST]  "+m"  (f16)
                          : [SRC]   "m"  (src),
                            [M]     "r"  (mask)
                          : "zmm2", "memory", "k1");
    return f16[0];
}

/* convert many FP32s to FP16s. */
void GGML_PHI_FP32_TO_FP16_ROW(const float * x, ggml_fp16_t * y, int n)
{
    for (int i = 0; i < n; i++) {
        y[i] = GGML_PHI_FP32_TO_FP16(x[i]);
    }
}

// This function perform two multiplies of an I8x16 and an I8x16 vector into two I16x16 vectors. Then it does an FMA on the scaled result of multiplying the two I16x16 vectors, adding the result into an I32x16. When done, it multiplies this I32x16 by a float, returning a F32x16.
// It loops 8 times. Well, actually four, with an unroll.
// Handles q8 being unaligned.
// Requires q5 to be aligned.
void GGML_8X_2xI8x16_2xI8x16_MUL_2xI16x16_S_FMA_I32x16_Unaligned (const int8x16_t *q8, uint8x16_t *q5, const uint8_t *scale, ggml_fp16_t scaleX, float scaleY, float32x16_t *res)
{
    uint8_t zero = 0;
    uint64_t q8offset=((uint64_t) q8) & 0x3f;

    __asm__ __volatile__ (
                          "vprefetchenta\t(%[RES])\n\t"                       // Issue our memory requests first thing.
                          "vprefetch0\t64(%[SCALE])\n\t"
                          "vprefetch0\t(%[SRC8])\n\t"
                          "vprefetch0\t64(%[SRC8])\n\t"
                          "vprefetch0\t(%[SRC5])\n\t"
                          "mov\t%[SRC8],\t%%r11\n\t"                          // Use r11 to store the address for vloadunpackld.
                          "mov\t%[SRC5],\t%%r8\n\t"
                          "mov\t%[SCALE],\t%%r9\n\t"
                          "mov\t$0,\t%%ecx\n\t"
                          "mov\t%[SRC8],\t%%r15\n\t"                          // Use r12-r15 to store the addresses for vloadunpackhd.
                          "mov\t%[SRC8],\t%%r14\n\t"
                          "mov\t%[SRC8],\t%%r13\n\t"
                          "mov\t%[SRC8],\t%%r12\n\t"
                          "mov\t%[OFFSET],\t%%r10\n\t"
                          "cmp\t$32,%%r10\n\t"                                // Examine OFFSET, and decide which (if any) of the vloadunpackhd invocations needs to be increased by 64.
                          "jl\t20f\n\t"
                          "cmp\t$48,%%r10\n\t"
                          "jl\t21f\n\t"
                          "add\t$64,%%r12\n\t"                                // Greater than 47.
                          "jmp\t18f\n\t"
                          "21:\n\t"
                          "add\t$64,%%r13\n\t"                                // Between 48 and 31.
                          "jmp\t18f\n\t"
                          "20:\n\t"                                           // Less than 32...
                          "cmp\t$16,%%r10\n\t"
                          "jz\t18f\n\t"                                       // Zero.
                          "jl\t23f\n\t"
                          "add\t$64,%%r14\n\t"                                // Between 32 and 15.
                          "jmp\t18f\n\t"
                          "23:\n\t"
                          "add\t$64,%%r15\n\t"                                // Between 16 and zero.
                          "18:\n\t"
                          "vbroadcastss\t%[SCALEY],\t%%zmm3\n\t"              // Load the scale factors coresponding to the two input vectors.
                          "vbroadcastss\t%[SCALEX]%{float16%},\t%%zmm4\n\t"
                          "vmulps\t%%zmm3,\t%%zmm4,\t%%zmm5\n\t"              // Prepare the factor we're going to multiply the result by..
                          "vmovaps\t\t(%[RES]),\t%%zmm6\n\t"                  // Load our inital state from sum..
                          "vpbroadcastd\t%[Z]%{uint8%},\t%%zmm7\n\t"          // Empty our result.
                          "1:\n\t"
                          "inc\t%%ecx\n\t"                                    // We are in our loop, increment our counter.
                          "vloadunpackld\t\t(%%r11)%{sint8%},\t%%zmm8\n\t"    // Load the item we will be multiplying from. Upscale it from int8 to int32.
                          "vloadunpackld\t\t16(%%r11)%{sint8%},\t%%zmm9\n\t"  // Load the item we will be multiplying from. Upscale it from int8 to int32.
                          "vloadunpackld\t\t32(%%r11)%{sint8%},\t%%zmm10\n\t" // Load the item we will be multiplying from. Upscale it from int8 to int32.
                          "vloadunpackld\t\t48(%%r11)%{sint8%},\t%%zmm11\n\t" // Load the item we will be multiplying from. Upscale it from int8 to int32.
                          "vprefetch1\t128(%%r11)\n\t"                        // Prepare for a run-through.
                          "add\t$64,\t%%r11\n\t"
                          "vloadunpackhd\t\t(%%r12)%{sint8%},\t%%zmm8\n\t"    // Load the item we will be multiplying from. Upscale it from int8 to int32.
                          "add\t$64,\t%%r12\n\t"
                          "vloadunpackhd\t\t16(%%r13)%{sint8%},\t%%zmm9\n\t"  // Load the item we will be multiplying from. Upscale it from int8 to int32.
                          "add\t$64,\t%%r13\n\t"
                          "vloadunpackhd\t\t32(%%r14)%{sint8%},\t%%zmm10\n\t" // Load the item we will be multiplying from. Upscale it from int8 to int32.
                          "add\t$64,\t%%r14\n\t"
                          "vloadunpackhd\t\t48(%%r15)%{sint8%},\t%%zmm11\n\t" // Load the item we will be multiplying from. Upscale it from int8 to int32.
                          "add\t$64,\t%%r15\n\t"
                          "vmovdqa32\t\t(%%r8)%{uint8%},\t%%zmm12\n\t"        // Load the item we will be multiplying with. Upscale it from int8 to int32.
                          "vpmulld\t%%zmm8,\t%%zmm12,\t%%zmm13\n\t"           // Perform our 64 bit multiply, low side.
                          "vmovdqa32\t\t16(%%r8)%{uint8%},\t%%zmm14\n\t"      // Load the item we will be multiplying with. Upscale it from int8 to int32.
                          "vpmulld\t%%zmm9,\t%%zmm14,\t%%zmm15\n\t"           // Perform our 64 bit multiply, low side.
                          "vmovdqa32\t\t32(%%r8)%{uint8%},\t%%zmm0\n\t"       // Load the item we will be multiplying with. Upscale it from int8 to int32.
                          "vpmulld\t%%zmm10,\t%%zmm0,\t%%zmm1\n\t"            // Perform our 64 bit multiply, low side.
                          "vmovdqa32\t\t48(%%r8)%{uint8%},\t%%zmm2\n\t"       // Load the item we will be multiplying with. Upscale it from int8 to int32.
                          "vpmulld\t%%zmm11,\t%%zmm2,\t%%zmm3\n\t"            // Perform our 64 bit multiply, low side.
                          "vprefetch1\t64(%%r8)\n\t"                          // Prepare for a run-through.
                          "add\t$64,\t%%r8\n\t"
                          "vpbroadcastd\t(%%r9)%{uint8%},\t%%zmm4\n\t"        // Load the item we will be multiplying by.
                          "vpbroadcastd\t1(%%r9)%{uint8%},\t%%zmm8\n\t"       // Load the item we will be multiplying by.
                          "vprefetch1\t2(%%r9)\n\t"
                          "add\t$2,\t%%r9\n\t"
                          "vprefetch0\t(%%r11)\n\t"                           // Prepare for a run-through.
                          "vprefetch0\t64(%%r11)\n\t"                         // Prepare for a run-through.
                          "vprefetch0\t(%%r8)\n\t"                            // Prepare for a run-through.
                          "vprefetch0\t(%%r9)\n\t"                            // Prepare for a run-through.
                          "cmp\t$4,\t%%ecx\n\t"                               // See if this is our last run-through.
                          "vpmadd231d\t%%zmm13,\t%%zmm4,\t%%zmm7\n\t"         // Perform our multiply-add.
                          "vpmadd231d\t%%zmm15,\t%%zmm4,\t%%zmm7\n\t"         // Perform our multiply-add.
                          "vpmadd231d\t%%zmm1,\t%%zmm8,\t%%zmm7\n\t"          // Perform our multiply-add.
                          "vpmadd231d\t%%zmm3,\t%%zmm8,\t%%zmm7\n\t"          // Perform our multiply-add.
                          "jl\t1b\n\t"
                          "vcvtfxpntdq2ps\t$0,%%zmm7,\t%%zmm9\n\t"            // Convert our ints to floats.
                          "vfmadd231ps\t%%zmm5,\t%%zmm9,\t%%zmm6\n\t"         // Perform a fused multiply add.
                          "vmovaps\t\t%%zmm6,\t(%[RES])\n\t"                  // Save the result.
                          : [RES]   "+r" (res)
                          : [SRC8]   "r" (q8),
                            [OFFSET] "m" (q8offset),
                            [SRC5]   "r" (q5),
                            [SCALE]  "r" (scale),
                            [SCALEX] "m" (scaleX),
                            [SCALEY] "m" (scaleY),
                            [Z]      "m" (zero)
                          : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "cc", "ecx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory");
}

// Unpack 256 unsigned 5 bit values into an 8 bit vector.
// Handles q4 not being aligned correctly.
// Requires dst to be aligned.
void GGML_5bit_Unpack_Unaligned (const uint8x16_t * q4, const uint8_t * q1, uint8x16_t * dst)
{
    uint8_t lowmask = 0x0F;
    uint8_t m=1;
    uint8_t bit5 = 0x10;

    __asm__ __volatile__ (
                          "vprefetch0\t(%[SRC1])\n\t"                          // Issue our memory requests first thing.
                          "vprefetch0\t(%[SRC4])\n\t"
                          "vprefetchenta\t(%[DST])\n\t"
                          "mov\t%[SRC4],\t%%r9\n\t"                            // Load the address of the head of our 4-bit list.
                          "mov\t%[DST],\t%%r8\n\t"                             // Load the address of the head of our destination list.
                          "mov\t$0,%%ecx\n\t"                                  // Initialize our counter.
                          "vpbroadcastd\t%[MASK]%{uint8%},\t%%zmm0\n\t"        // Load our mask.
                          "vpbroadcastd\t%[BIT5]%{uint8},\t%%zmm1\n\t"         // Load the bit we want to add (conditionally).
                          "vpbroadcastd\t%[M]%{uint8%},\t%%zmm2\n\t"           // Select which bit we want to test for. Start with bit 1.
                          "vmovdqa32\t(%[SRC1])%{uint8%},\t%%zmm3\n\t"         // Load 16 sets of 8 packed single bits.
                          "vmovdqa32\t16(%[SRC1])%{uint8%},\t%%zmm4\n\t"       // Load the next 16 sets of 8 packed single bits.

                          "1:\n\t"
                          "inc\t%%ecx\n\t"                                     // We are in the loop. increment the counter.

                          "vptestmd\t%%zmm3,\t%%zmm2,\t%%k1\n\t"               // Test to see if our selected bit is set.
                          "vptestmd\t%%zmm4,\t%%zmm2,\t%%k2\n\t"               // Test to see if our selected bit is set.

                          "vloadunpackld\t\t(%%r9)%{uint8%},\t%%zmm5\n\t"      // Load our odd 4 bit sequences. note that it loads two 4 bit sequences into each zmm value.
                          "vloadunpackhd\t\t16(%%r9)%{uint8%},\t%%zmm5\n\t"    // Load our odd 4 bit sequences. note that it loads two 4 bit sequences into each zmm value.
                          "vpandd\t%%zmm0,\t%%zmm5,\t%%zmm6\n\t"               // Apply a mask, storing the first set of four bits into a vector.
                          "vpord\t%%zmm1,%%zmm6,%%zmm6%{%%k1%}\n\t"            // Turn on bit 5 for all values that passed the prior test.
                          "vmovdqa32\t\t%%zmm6%{uint8%},\t(%%r8)\n\t"          // Save our result.

                          "vloadunpackld\t\t16(%%r9)%{uint8%},\t%%zmm7\n\t"    // Load our odd 4 bit sequences. note that it loads two 4 bit sequences into each zmm value.
                          "vloadunpackhd\t\t32(%%r9)%{uint8%},\t%%zmm7\n\t"    // Load our odd 4 bit sequences. note that it loads two 4 bit sequences into each zmm value.
                          "vprefetch1\t32(%%r9)\n\t"                           // Pull the next set of 4 bit sequences into the L2 cache.
                          "vpandd\t%%zmm0,\t%%zmm7,\t%%zmm8\n\t"               // Apply a mask, storing the next set of four bits into a vector.
                          "vpord\t%%zmm1,%%zmm8,%%zmm8%{%%k2%}\n\t"            // Turn on bit 5 for all values that passed the prior test.
                          "vmovdqa32\t\t%%zmm8%{uint8%},\t16(%%r8)\n\t"        // Save our result.
                          
                          "add\t$32,\t%%r8\n\t"
                          "cmp\t$4,\t%%ecx\n\t"

                          "vpslld\t$1,\t%%zmm2,\t%%zmm2\n\t"                   // Select the next bit to test for.
                          
                          "vptestmd\t%%zmm3,\t%%zmm2,\t%%k1\n\t"               // Perform our test.
                          "vptestmd\t%%zmm4,\t%%zmm2,\t%%k2\n\t"               // Perform our test.
                          "vpsrld\t$4,\t%%zmm5,\t%%zmm6\n\t"                   // Load our even 4 bit sequence.
                          "vpsrld\t$4,\t%%zmm7,\t%%zmm8\n\t"                   // Load our next even 4 bit sequence.
                          "vpord\t%%zmm1,%%zmm6,%%zmm6%{%%k1%}\n\t"            // Turn on bit 5 for all values that passed the prior test.
                          "vpord\t%%zmm1,%%zmm8,%%zmm8%{%%k2%}\n\t"            // Turn on bit 5 for all values that passed the prior test.
                          "vmovdqa32\t\t%%zmm6%{uint8%},\t(%%r8)\n\t"          // Save our result.
                          "vmovdqa32\t\t%%zmm8%{uint8%},\t16(%%r8)\n\t"        // Save our result.
                          "vprefetchenta\t32(%%r8)\n\t"

                          "je\t2f\n\t"

                          "vprefetch0\t32(%%r9)\n\t"
                          "vprefetch1\t96(%%r9)\n\t"
                          "vpslld\t$1,\t%%zmm2,\t%%zmm2\n\t"                   // Select the next bit to test for.
                          "add\t$32,\t%%r9\n\t"
                          "add\t$32,\t%%r8\n\t"
                          "jmp\t1b\n\t"
                          "2:"
                          : [DST]   "+r" (dst)
                          : [SRC4]   "r" (q4),
                            [SRC1]   "r" (q1),
                            [MASK]   "m" (lowmask),
                            [M]      "m" (m),
                            [BIT5]   "m" (bit5)
                          : "zmm0", "zmm1", "zmm2", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "cc", "ecx", "k1", "k2", "r8", "r9", "memory");
}
  
// A function for getting the dot product of two vectors, one of 5 bit resolution, and one of 8.
// Used during inference, if your model prints "llama_model_loader: - type q5_K:  XXX tensors", and XXX is not zero. :)
void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {

    /* Interpret X and Y as vectors. */
    const block_q5_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    /* The number of blocks we will process this in. */
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float32x16_t sums;

    // Clear sums.
    GGML_F32x16_VEC_ZERO(&sums);

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        uint8x16_t q5 [QK_K/16];

        // Combine our 4 and 1 bit vector sets into a 5 bit vector (in 8 bits).
        GGML_5bit_Unpack_Unaligned((const uint8x16_t *)x[i].qs, x[i].qh, q5);

        // Extract scales and mins..
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;

        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];

        // FIXME: while comparing FMA output to the original output, the original had an error. Hunt it down.
        GGML_8X_2xI8x16_2xI8x16_MUL_2xI16x16_S_FMA_I32x16_Unaligned((const int8x16_t *)y[i].qs, q5, scales, x[i].d, y[i].d, &sums);

        const float dmin = GGML_PHI_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }

    for (int l = 0; l < GGML_F32_EPR; ++l) sumf += ((float *)&sums)[l];
    *s = sumf;
}
