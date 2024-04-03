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

// This SIMD unit can work with 32 float32s at once.
#define GGML_F32_STEP 32
// We can fit 16 of these float32s in a single vector register.
#define GGML_F32_EPR 16

/* we force an alignment, because i haven't written unaligned forms of the assembly functions, yet.. */
typedef float float32x16_t __attribute__((vector_size (64), aligned(64)));
typedef int8_t int8x16_t __attribute__((vector_size (16), aligned(16)));
typedef uint8_t uint8x16_t __attribute__((vector_size (16), aligned(16)));
typedef int32_t int32x16_t __attribute__((vector_size (64), aligned(64)));

/* A forward declaration, to keep GCC happy. */
void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy,  size_t by, int nrc);

/* clear a vector of 16 floats. */
inline static void GGML_F32x16_VEC_ZERO(float32x16_t *target)
{
  uint8_t zero=0;

  __asm__ __volatile__ (
                        "vbroadcastss\t%[Z]%{uint8%},\t%%zmm8\n\t" // use an upscaling operator to clear our register.
                        "vmovaps\t\t%%zmm8,\t%[RES]\n\t"
                        : [RES]  "+m"  (*target)
                        : [Z]    "m"   (zero)
                        : "zmm8", "memory");
}

// This function perform two multiplies of an I8x16 and an I8x16 vector into two I16x16 vectors. then does an FMA on the scaled result of multiplying the two I16x16 vectors, adding the result into an I32x16.
// it loops 8 times. well, actually four, with an unroll.
inline static void GGML_8X_2xI8x16_2xI8x16_MUL_2xI16x16_S_FMA_I32x16 (int8x16_t *src11, uint8x16_t *src21, const uint8_t *scale, int32x16_t *res)
{
  uint8_t zero = 0;

  __asm__ __volatile__ (
			"vprefetche0\t(%[SRC11])\n\t"
			"vprefetche0\t(%[SRC21])\n\t"
			"vprefetche0\t(%[SCALE])\n\t"
			"mov\t$0,\t%%ecx\n\t"
			"mov\t%[SRC11],\t%%r12\n\t"
			"mov\t%[SRC21],\t%%r8\n\t"
			"mov\t%[SCALE],\t%%r9\n\t"
			"vpbroadcastd\t%[Z]%{uint8%},\t%%zmm7\n\t"     // empty our result.

			"1:\n\t"
			"inc\t%%ecx\n\t"                               // we are in our loop, increment our counter.
			"cmp\t$4,\t%%ecx\n\t"                          // see if this is our last run-through.
			"vmovdqa32\t\t(%%r12)%{sint8%},\t%%zmm0\n\t"   // load the item we will be multiplying from. upscale it from int8 to int32.
			"vmovdqa32\t\t(%%r8)%{uint8%},\t%%zmm1\n\t"    // load the item we will be multiplying with. upscale it from int8 to int32.
			"vpmulld\t%%zmm0,\t%%zmm1,\t%%zmm2\n\t"        // perform our 64 bit multiply, low side.
			"vpbroadcastd\t(%%r9)%{uint8%},\t%%zmm6\n\t"   // load the item we will be multiplying by.
			"vpmadd231d\t%%zmm2,\t%%zmm6,\t%%zmm7\n\t"     // perform our multiply-add.
			"vmovdqa32\t\t16(%%r12)%{sint8%},\t%%zmm3\n\t" // load the item we will be multiplying from. upscale it from int8 to int32.
			"vmovdqa32\t\t16(%%r8)%{uint8%},\t%%zmm4\n\t"  // load the item we will be multiplying with. upscale it from int8 to int32.
			"vpmulld\t%%zmm3,\t%%zmm4,\t%%zmm5\n\t"        // perform our 64 bit multiply, low side.
			"vpmadd231d\t%%zmm5,\t%%zmm6,\t%%zmm7\n\t"     // perform our multiply-add.
			"vmovdqa32\t\t32(%%r12)%{sint8%},\t%%zmm8\n\t" // load the item we will be multiplying from. upscale it from int8 to int32.
			"vmovdqa32\t\t32(%%r8)%{uint8%},\t%%zmm1\n\t"  // load the item we will be multiplying with. upscale it from int8 to int32.
			"vpmulld\t%%zmm8,\t%%zmm1,\t%%zmm2\n\t"        // perform our 64 bit multiply, low side.
			"vpbroadcastd\t1(%%r9)%{uint8%},\t%%zmm6\n\t"  // load the item we will be multiplying by.
			"vpmadd231d\t%%zmm2,\t%%zmm6,\t%%zmm7\n\t"     // perform our multiply-add.
			"vmovdqa32\t\t48(%%r12)%{sint8%},\t%%zmm3\n\t" // load the item we will be multiplying from. upscale it from int8 to int32.
			"vmovdqa32\t\t48(%%r8)%{uint8%},\t%%zmm4\n\t"  // load the item we will be multiplying with. upscale it from int8 to int32.
			"vpmulld\t%%zmm3,\t%%zmm4,\t%%zmm5\n\t"        // perform our 64 bit multiply, low side.
			"vpmadd231d\t%%zmm5,\t%%zmm6,\t%%zmm7\n\t"     // perform our multiply-add.
			"je\t2f\n\t"                                   // if this is the last time through our loop, jump to 2.
			"vprefetche0\t64(%%r12)\n\t"                   // otherwise, prepare for another run-through.
			"vprefetche0\t64(%%r8)\n\t"
			"vprefetche2\t128(%%r12)\n\t"
			"vprefetche2\t128(%%r8)\n\t"
			"add\t$64,\t%%r12\n\t"
			"add\t$64,\t%%r8\n\t"
			"add\t$2,\t%%r9\n\t"
			"jmp\t1b\n\t"
			"2:\n\t"
			"vmovdqa32\t\t%%zmm7,\t(%[RES])\n\t"           // save the result.
			: [RES]   "+r" (res)
			: [SRC11] "r"  (src11),
			  [SRC21] "r"  (src21),
			  [SCALE] "r"  (scale),
			  [Z]     "m"  (zero)
			: "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "ecx", "r8", "r9", "r12", "memory");
}

// Unpack 256 unsigned 5 bit values into an 8 bit vector.
inline static void GGML_5bit_Unpack (const uint8x16_t * q4, const uint8_t * q1, uint8x16_t * dst)
{
  uint8_t lowmask = 0x0F;
  uint32_t allmask=0xFFFFFFFF;
  uint8_t m=1;
  uint8_t bit5 = 0x10;

  __asm__ __volatile__ (
			"vprefetche0\t(%[SRC1])\n\t"
			"vprefetche0\t(%[SRC4])\n\t"
			"vprefetche1\t64(%[SRC4])\n\t"
			"mov\t%[SRC4],\t%%r12\n\t"                       // load the address of the head of our 4-bit list.
			"mov\t%[DST],\t%%r8\n\t"                         // load the address of the head of our destination list.
			"mov\t$0,%%ecx\n\t"                              // initialize our counter.
			"vmovdqa32\t(%[SRC1])%{uint8%},\t%%zmm6\n\t"     // move 16 packed sets of single bits into the lower 8 bits of zmm6.
			"vmovdqa32\t16(%[SRC1])%{uint8%},\t%%zmm7\n\t"   // move the next 16 packed sets of single bits into the lower 8 bits of zmm7.
			"vpbroadcastd\t%[MASK]%{uint8%},\t%%zmm2\n\t "   // load our mask.
			"vpbroadcastd\t%[BIT5]%{uint8},\t%%zmm9\n\t"     // load the bit we want to add (conditionally).
			"vpbroadcastd\t%[M]%{uint8%},\t%%zmm8\n\t"       // select which bit we want to test for.

			"1:\n\t"
			"inc\t%%ecx\n\t"                                 // we are in the loop. increment the counter.

			"vptestmd\t%%zmm6,\t%%zmm8,\t%%k1\n\t"           // perform our test.
			"vptestmd\t%%zmm7,\t%%zmm8,\t%%k2\n\t"           // perform our test.
			"vmovdqa32\t\t(%%r12)%{uint8%},\t%%zmm0\n\t"     // load our odd 4 bit sequences. note that it loads two 4 bit sequences into each zmm value.
			"vpandd\t%%zmm0,\t%%zmm2,\t%%zmm4\n\t"           // apply a mask, storing the low four bits of vector zmm0 into zmm4.
			"vpaddd\t%%zmm4,%%zmm9,%%zmm4%{%%k1%}\n\t"       // turn on bit 5 for all values that passed the prior test.
			"vmovdqa32\t\t%%zmm4%{uint8%},\t(%%r8)\n\t"      // save our result.
			"vmovdqa32\t\t16(%%r12)%{uint8%},\t%%zmm1\n\t"   // load our odd 4 bit sequences. note that it loads two 4 bit sequences into each zmm value.
			"vpandd\t%%zmm1,\t%%zmm2,\t%%zmm5\n\t"           // apply a mask, storing the next low four bits of vector zmm1 into zmm5.
			"vpaddd\t%%zmm5,%%zmm9,%%zmm5%{%%k2%}\n\t"       // turn on bit 5 for all values that passed the prior test.
			"vmovdqa32\t\t%%zmm5%{uint8%},\t16(%%r8)\n\t"    // save our result.

			"add\t$32,\t%%r8\n\t"
			"cmp\t$4,\t%%ecx\n\t"
			"vpslld\t$1,\t%%zmm8,\t%%zmm8\n\t"               // select which bit we want to test for.

			"vptestmd\t%%zmm6,\t%%zmm8,\t%%k1\n\t"           // perform our test.
			"vptestmd\t%%zmm7,\t%%zmm8,\t%%k2\n\t"           // perform our test.
			"vpsrld\t$4,\t%%zmm0,\t%%zmm4\n\t"               // load our even 4 bit sequence into zmm4.
			"vpaddd\t%%zmm4,%%zmm9,%%zmm4%{%%k1%}\n\t"       // turn on bit 5 for all values that passed the prior test.
			"vmovdqa32\t\t%%zmm4%{uint8%},\t(%%r8)\n\t"      // save our result.
			"vpsrld\t$4,\t%%zmm1,\t%%zmm5\n\t"               // load our even 4 bit sequence into zmm5.
			"vpaddd\t%%zmm5,%%zmm9,%%zmm5%{%%k2%}\n\t"       // turn on bit 5 for all values that passed the prior test.
			"vmovdqa32\t\t%%zmm5%{uint8%},\t16(%%r8)\n\t"    // save our result.

			"je\t2f\n\t"

			"vpslld\t$1,\t%%zmm8,\t%%zmm8\n\t"               // select which bit we want to test for.
			"add\t$32,\t%%r12\n\t"
			"add\t$32,\t%%r8\n\t"
			"jmp\t1b\n\t"
			"2:"
			: [DST]  "+r" (dst)
			: [SRC4]  "r" (q4),
			  [SRC1]  "r" (q1),
			  [MASK]  "m" (lowmask),
			  [M]     "m" (m),
			  [ALL]   "m" (allmask),
			  [BIT5]  "m" (bit5)
			: "zmm0", "zmm1", "zmm2", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "ecx", "k1", "k2", "r12", "r8", "memory"
			);
}
  
// A function for getting the dot product of two vectors, one of 5 bit resolution, and one of 8.
// Used during inference, if your model prints "llama_model_loader: - type q5_K:  XXX tensors", and XXX is not zero. :)
void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {

  /* interpret X and Y as vectors. */
  const block_q5_K * restrict x = vx;
  const block_q8_K * restrict y = vy;

  /* the number of blocks we will process this in. */
  const int nb = n / QK_K;

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];

  const uint8_t * scales = (const uint8_t*)&utmp[0];
  const uint8_t * mins   = (const uint8_t*)&utmp[2];

  float32x16_t sums;

  // clear sums.
  GGML_F32x16_VEC_ZERO(&sums);

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    int8x16_t q8copy [QK_K];
    int32x16_t aux32;
    uint8x16_t q4copyvec [QK_K/32];
    uint8x16_t aux8 [QK_K/16];

    // Fill in our 8 bit vector from y[]. required, because there is no good way to align members of y[], And I haven't mastered unaligned assembly yet...
    memcpy (q8copy, y[i].qs, QK_K);

    // Fill in our 4 bit vector from x[]. required, because there is no good way to align members of x[], And I haven't mastered unaligned assembly yet...
    memcpy (q4copyvec, x[i].qs, QK_K/2);

    // combine our 4 and 1 bit vector sets into an 8 bit value.
    GGML_5bit_Unpack(q4copyvec, x[i].qh, aux8);

    // extract scales and mins..
    memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;

    // FIXME: while comparing FMA output to the original output, the original had an error. hunt it down.
    GGML_8X_2xI8x16_2xI8x16_MUL_2xI16x16_S_FMA_I32x16(q8copy, aux8, scales, &aux32);

    int sumi = 0;
    for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
    const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
    for (int l = 0; l < GGML_F32_EPR; ++l) ((float *)&sums)[l] += d * ((int32_t *)&aux32)[l];
    const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
  }

  for (int l = 0; l < GGML_F32_EPR; ++l) sumf += ((float *)&sums)[l];
  *s = sumf;
}
