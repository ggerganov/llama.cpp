// For uint32_t
#include <stdint.h>

// For size_t
#include <stdio.h>

// Yes, we have to tell this header to actually export stuff.
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

// FIXME: why do we have to import this twice?
#define GGML_COMMON_IMPL_C
// For block_q5_K and block_q8_K. only given the second time.
#include "ggml-common.h"


// This SIMD unit can work with 32 float32s at once.
#define GGML_F32_STEP 32
// We can fit 16 of these float32s in a single vector register.
#define GGML_F32_EPR 16

typedef float float32x8_t __attribute__((vector_size (64)));
typedef int16_t int16x8_t __attribute__((vector_size (32)));
typedef int32_t int32x8_t __attribute__((vector_size (64)));
typedef int16_t int16x16_t __attribute__((vector_size (64)));
typedef int32_t int32x16_t __attribute__((vector_size (128)));

/* A forward declaration, to keep GCC happy. */
void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy,  size_t by, int nrc);

inline static void GGML_F32x8_VEC_ZERO(float32x8_t *target)
{
  uint8_t zero[4] __attribute__((aligned(64))) = {0,0,0,0};
  uint32_t mask=0x000000FF;

  __asm__ __volatile__ (
                        "vbroadcastf32x4\t%[Z]%{uint8%},\t%%zmm8\n\t"        // use an upscaling operator to clear our value.
			"kmov\t%[M],\t%%k1\n\t"
                        "vmovaps\t\t%%zmm8,\t%[RES]%{%%k1%}\n\t"
			: [RES]  "+m"  (*target)
			: [Z]    "m"   (zero),
			  [M]    "r"   (mask)
			: "zmm8", "k1", "memory");
}

inline static void GGML_I32x8_VEC_ZERO(int32x8_t *target)
{
  uint8_t zero[4] __attribute__((aligned(64))) = {0,0,0,0};
  uint32_t mask=0x000000FF;

  __asm__ __volatile__ (
                        "vbroadcastI32x4\t%[Z]%{uint8%},\t%%zmm8\n\t"        // use an upscaling operator to clear our value.
			"kmov\t%[M],\t%%k1\n\t"
                        "vmovaps\t\t%%zmm8,\t%[RES]%{%%k1%}\n\t"
			: [RES]  "+m"  (*target)
			: [Z]    "m"   (zero),
			  [M]    "r"   (mask)
			: "zmm8", "k1", "memory");
}

inline static void GGML_I32x16_VEC_ZERO(int32x8_t *target)
{
  uint8_t zero[4] __attribute__((aligned(64))) = {0,0,0,0};

  __asm__ __volatile__ (
                        "vbroadcastI32x4\t%[Z]%{uint8%},\t%%zmm8\n\t"        // use an upscaling operator to clear our value.
			"kmov\t%[M],\t%%k1\n\t"
                        "vmovaps\t\t%%zmm8,\t%[RES]%{%%k1%}\n\t"
			: [RES]  "+m"  (*target)
			: [Z]    "m"   (zero)
			: "zmm8", "k1", "memory");
}

// perform an eight wide Fused Multiply Add of an I16x8 times scalar S into I32x8.
inline static void GGML_I16x8_S_FMA_I32x8 (int16x8_t *src, int32_t scale, int32x8_t *dest)
{
  uint8_t zero[4] __attribute__((aligned(64))) = {0,0,0,0};
  uint32_t mask=0x000000FF;
  int32_t scaleVec[4] = {scale, scale, scale, scale};

  __asm__ __volatile__ (
			"kmov\t%[M],\t%%k1\n\t"                              // we will only be working with 8 values at a time. le sigh.
			"vmovdqa32\t\t%[SRC]%{sint16%},\t%%zmm0%{%%k1%}\n\t" // load the item we will be summing from. upscale it from int16.
			"vbroadcastI32x4\t%[SCALE],\t%%zmm1\n\t"             // load the item we will be multiplying by.
                        "vmovdqa32\t\t%[RES],\t%%zmm2%{%%k1%}\n\t"           // load the item we will be summing onto.
			"vpmadd231d\t%%zmm0,\t%%zmm1,\t%%zmm2%{%%k1%}\n\t"   // perform our multiply-add.
			"vmovdqa32\t\t%%zmm2,\t%[RES]%{%%k1}\n\t"            // save the result.
			: [RES]   "+m" (*dest)
			: [Z]     "m"  (zero),
			  [M]     "r"  (mask),
			  [SRC]   "m"  (src),
			  [SCALE] "m"  (scaleVec)
			: "zmm0", "zmm1", "zmm2", "k1", "memory");
}

// perform an eight wide Fused Multiply Add of an I16x16 times scalar S into I32x16.
inline static void GGML_I16x16_S_FMA_I32x16 (int16x8_t *src, int32_t scale, int32x8_t *dest)
{
  int32_t scaleVec[4] = {scale, scale, scale, scale};

  __asm__ __volatile__ (
			"vmovdqa32\t\t%[SRC]%{sint16%},\t%%zmm0\n\t" // load the item we will be summing from. upscale it from int16.
			"vbroadcastI32x4\t%[SCALE],\t%%zmm1\n\t"     // load the item we will be multiplying by.
                        "vmovdqa32\t\t%[RES],\t%%zmm2\n\t"           // load the item we will be summing onto.
			"vpmadd231d\t%%zmm0,\t%%zmm1,\t%%zmm2\n\t"   // perform our multiply-add.
			"vmovdqa32\t\t%%zmm2,\t%[RES]\n\t"           // save the result.
			: [RES]   "+m" (*dest)
			: [SRC]   "m"  (src),
			  [SCALE] "m"  (scaleVec)
			: "zmm0", "zmm1", "zmm2", "k1", "memory");
}

void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy,  size_t by, int nrc) {

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

  int8_t aux8[QK_K];
  int16x16_t aux16 __attribute__((aligned(128)));
  float32x16_t sums __attribute__((aligned(64)));
  int32x16_t aux32 __attribute__((aligned(128)));

  GGML_F32x16_VEC_ZERO(&sums);

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t * restrict q4 = x[i].qs;
    const uint8_t * restrict hm = x[i].qh;
    const  int8_t * restrict q8 = y[i].qs;

    GGML_I32x16_VEC_ZERO(&aux32);

    int8_t * restrict a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < QK_K/64; ++j) {
      for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
      for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
      a += 32; m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
      for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
      a += 32; m <<= 1;
      q4 += 32;
    }
    memcpy(utmp, x[i].scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    
    int sumi = 0;
    for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
    a = aux8;
    int is = 0;
    for (int j = 0; j < QK_K/32; ++j) {
      int32_t scale = scales[is++];
      for (int l = 0; l < 16; ++l) ((int16_t *)&aux16)[l] = q8[l] * a[l];
      GGML_I16x8_S_FMA_I32x16 (&aux16, scale, &aux32);
      q8 += 16; a += 16;
      /* FIXME: while comparing FMA output to normal output, the original had an error. hunt it down. */
      for (int l = 0; l < 16; ++l) ((int16_t *)&aux16)[l] = q8[l] * a[l];
      GGML_I16x8_S_FMA_I32x16 (&aux16, scale, &aux32);
      q8 += 16; a += 16;
    }
    const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
    for (int l = 0; l < 16; ++l) ((float *)&sums)[l] += d * ((int32_t *)&aux32)[l];
    const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
  }
  for (int l = 0; l < 16; ++l) sumf += ((float *)&sums)[l];
  *s = sumf;
}
