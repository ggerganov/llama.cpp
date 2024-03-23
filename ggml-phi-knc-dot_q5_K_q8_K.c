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
typedef int16_t int16x16_t __attribute__((vector_size (64)));

/* A forward declaration, to keep GCC happy. */
void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy,  size_t by, int nrc);

inline static void GGML_F32x8_VEC_ZERO(float32x8_t *target)
{
  uint8_t zero[4] __attribute__((aligned(64))) = {0,0,0,0};
  uint32_t mask=0x0000000F;

  __asm__ __volatile__ (
                        "vbroadcastf32x4\t%[Z]%{uint8%},\t%%zmm8\n\t"        // use an upscaling operator to clear our value.
			"kmov\t%[M],\t%%k1\n\t"
                        "vmovaps\t\t%%zmm8,\t%[RES]%{%%k1%}\n\t"
			: [RES]  "+m"  (*target)
			: [Z]    "m"   (zero),
			  [M]    "r"   (mask)
			: "r9", "zmm8", "k1");
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

  int8_t  aux8[QK_K];
  int16_t aux16[8];
  float32x8_t sums;
  int32_t aux32[8];

  //memset(sums, 0, 8*sizeof(float));

  GGML_F32x8_VEC_ZERO(&sums);

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t * restrict q4 = x[i].qs;
    const uint8_t * restrict hm = x[i].qh;
    const  int8_t * restrict q8 = y[i].qs;
    memset(aux32, 0, 8*sizeof(int32_t));
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
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8; a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8; a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8; a += 8;
      for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8 += 8; a += 8;
    }
    const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) ((float *)&sums)[l] += d * aux32[l];
    const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
  }
  for (int l = 0; l < 8; ++l) sumf += ((float *)&sums)[l];
  *s = sumf;
}
