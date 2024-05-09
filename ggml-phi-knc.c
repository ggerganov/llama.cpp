/* Xeon PHI IMCI support. */
/* formatted by using emacs, with (M-x set-variable RET c-basic-offset RET 4 RET) executed. */
/* Formatted by using emacs, with (M-x set-variable RET indent-tabs-mode RET nil RET) executed. */

#include <stdint.h>

// For size_t
#include <stdio.h>

// For memcpy.
#include <string.h>

// We can fit 16 of these float32s in a single vector register.
#define GGML_F32_EPR 16

// A vector of 16 floats.
typedef float float32x16_t __attribute__((vector_size (64), aligned (64)));

// A forward declaration, to keep GCC happy...
void ggml_vec_dot_f32(int n, float * restrict s, size_t bs, const float * restrict x, size_t bx, const float * restrict y, size_t by, int nrc);

inline static void GGML_F32x16_VEC_ZERO(float32x16_t *target)
{
    uint8_t zero[4] __attribute__((aligned(64))) = {0,0,0,0};

    __asm__ __volatile__ (
                          "vbroadcastf32x4\t%[Z]%{uint8%},\t%%zmm8\n\t" // use an upscaling operator to clear our value.
                          "vmovnraps\t\t%%zmm8,\t%[RES]\n\t"
                          : [RES]  "+m"  (*target)
                          : [Z]    "m"   (zero)
                          : "zmm8");

}

// Multiply each item in mvec1 with the corresponding item in mvec2, adding the result to the corresponding item in sum. optionally clear the sum before starting. 
inline static void GGML_F32x16_VEC_FMA(const float32x16_t *mvec1, const float32x16_t *mvec2, float32x16_t *sumvec, size_t iterations, int clear)
{
    uint8_t zero[4] __attribute__((aligned(64))) = {0,0,0,0};

    __asm__ __volatile__ (
                          "mov\t%[ITER],%%r8\n\t"                       // how many register sized chunks are we responsible for
                          "mov\t%[VEC1],%%r10\n\t"                      // where do we start work in mvec1?
                          "mov\t%[VEC2],%%r12\n\t"                      // where do we start work in mvec2?
                          "cmp\t$1,%[CLR]\n\t"                          // should we clear the sum before we start?
                          "jne\t4f\n\t"
                          "vbroadcastf32x4\t%[Z]%{uint8%},\t%%zmm0\n\t" // if so, use an upscaling operator to do it.
                          "vprefetchnta\t(%%r10)\n\t"
                          "vprefetchnta\t(%%r12)\n\t"
                          "vprefetch1\t128(%%r10)\n\t"
                          "vprefetch1\t128(%%r12)\n\t"
                          "vprefetch1\t256(%%r10)\n\t"
                          "vprefetch1\t256(%%r12)\n\t"
                          "vprefetch1\t384(%%r10)\n\t"
                          "vprefetch1\t384(%%r12)\n\t"
                          "vprefetch1\t512(%%r10)\n\t"
                          "vprefetch1\t512(%%r12)\n\t"
                          "jmp\t1f\n\t"
                          "4:\n\t"
                          "vprefetch0\t(%[RES])\n\t"
                          "vmovaps\t\t(%[RES]),\t%%zmm0\n\t"            // otherwise, load our inital state from sum..
                          "vprefetchnta\t(%%r10)\n\t"
                          "vprefetchnta\t(%%r12)\n\t"
                          "1:\n\t"
                          "cmp\t$3,\t%%r8\n\t"                          // Compare iterations to three.
                          "jnae\t6f\n\t"                                // If there are not three iterations left, jump to label 6.
                          "vmovaps\t\t(%%r10),\t%%zmm1\n\t"             // Load two vectors.
                          "vmovaps\t\t(%%r12),\t%%zmm2\n\t"
                          "sub\t$3,\t%%r8\n\t"                          // Decrement iterations
                          "vprefetchnta\t192(%%r10)\n\t"                // prefetch the next float32x16_t block (192 bytes ahead)
                          "vprefetchnta\t192(%%r12)\n\t"
                          "vmovaps\t\t64(%%r10),\t%%zmm3\n\t"           // Load two vectors.
                          "vmovaps\t\t64(%%r12),\t%%zmm4\n\t"
                          "vprefetch1\t320(%%r10)\n\t"                  // prefetch the block after the block after the next float32x16_t block (320 bytes ahead)
                          "vprefetch1\t320(%%r12)\n\t"
                          "vmovaps\t\t128(%%r10),\t%%zmm5\n\t"          // Load two vectors.
                          "vmovaps\t\t128(%%r12),\t%%zmm6\n\t"
                          "vprefetch1\t576(%%r10)\n\t"
                          "vprefetch1\t576(%%r12)\n\t"
                          "vprefetch1\t704(%%r10)\n\t"
                          "vprefetch1\t704(%%r12)\n\t"
                          "add\t$192,\t%%r10\n\t"                       // Move to the next float32x16_t block (192 bytes ahead)
                          "add\t$192,\t%%r12\n\t"
                          "vfmadd231ps\t%%zmm1,\t%%zmm2,\t%%zmm0\n\t"   // Perform a fused multiply add
                          "vfmadd231ps\t%%zmm3,\t%%zmm4,\t%%zmm0\n\t"   // Perform a fused multiply add
                          "vfmadd231ps\t%%zmm5,\t%%zmm6,\t%%zmm0\n\t"   // Perform a fused multiply add
                          "jmp\t1b\n\t"                                 // Jump back to the start of the loop
                          "6:\n\t"                                      // we know we are near the tail. handle 2, 1, and 0 cases.
                          "cmp\t$0,\t%%r8\n\t"                          // Compare iterations to zero
                          "je\t2f\n\t"                                  // Jump to label 2 if zero (end of loop)
                          "cmp\t$1,\t%%r8\n\t"                          // Compare iterations to one
                          "vmovaps\t\t(%%r10),\t%%zmm1\n\t"             // Load two vectors.
                          "vmovaps\t\t(%%r12),\t%%zmm2\n\t"
                          "vfmadd231ps\t%%zmm1,\t%%zmm2,\t%%zmm0\n\t"   // Perform a fused multiply add
                          "je\t2f\n\t"                                  // Jump to label 3 if one (end of loop)
                          // No compare. we must be two.
                          "vmovaps\t\t64(%%r10),\t%%zmm3\n\t"           // Load two vectors.
                          "vmovaps\t\t64(%%r12),\t%%zmm4\n\t"
                          "vfmadd231ps\t%%zmm3,\t%%zmm4,\t%%zmm0\n\t"   // Perform a fused multiply add
                          "2:\n\t"                                      // Label for loop end
                          "vmovnraps\t\t%%zmm0,\t(%[RES])\n\t"          // save our results.
                          : [RES]  "+r" (sumvec)
                          : [ITER]  "r"  (iterations),
                            [VEC1]  "r"  (mvec1),
                            [VEC2]  "r"  (mvec2),
                            [CLR]   "r"  (clear),
                            [Z]     "m"  (zero)
                          : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "cc", "memory", "r8", "r10", "r12");
}

// NOTE: x and y inputs must be __attribute__((aligned(64)));
void ggml_vec_dot_f32(int n, float * restrict s, size_t bs, const float * restrict x, size_t bx, const float * restrict y, size_t by, int nrc)
{
    // our sum.
    float32x16_t sum;

    // the number of vector-sized steps we will need to do.
    const uint32_t np = (n & ~(GGML_F32_EPR - 1));

    GGML_F32x16_VEC_FMA((const float32x16_t *)x, (const float32x16_t *)y, &sum, np/GGML_F32_EPR, 1);

    // add the leftovers, that could not be handled by the vector loop.
    if ( n - np != 0 )
        {
            // our extended last part of x.
            float32x16_t v1;
            GGML_F32x16_VEC_ZERO(&v1);
            // our extended last part of y.
            float32x16_t v2;
            GGML_F32x16_VEC_ZERO(&v2);

            memcpy(&v1, &x[np], (n - np)*sizeof(float));
            memcpy(&v2, &y[np], (n - np)*sizeof(float));

            GGML_F32x16_VEC_FMA(&v1,
                                &v2,
                                &sum, 1, 0);
        }

    // reduce sum, and store it in s.
    for (uint32_t i=0; i <GGML_F32_EPR; ++i)
        *s+=((float *)&sum)[i];

}
