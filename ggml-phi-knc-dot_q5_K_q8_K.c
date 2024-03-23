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
typedef int16 int16x16_t __attribute__((vector_size (64)));

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
  int8_t aux8[QK_K];
  //  int16_t aux16[16];
  int16x16_t aux16;
  float32x8_t sums __attribute__((aligned(64)));

  /* use a vector operation to clear these floats. */
  GGML_F32x8_VEC_ZERO(&sums);

  float sumf = 0;

  for (int i = 0; i < nb; ++i) {
    // quants, 4 low bits.
    const uint8_t * restrict q4 = x[i].qs;
    // quants, 1 high bit.
    const uint8_t * restrict hm = x[i].qh;
    const  int8_t * restrict q8 = y[i].qs;
    int8_t * restrict a = aux8;
    for (int l = 0; l < 32; ++l) {
      a[l+ 0] = q4[l] & 0xF;
      a[l+32] = q4[l]  >> 4;
    }
    for (int is = 0; is < 8; ++is) {
      uint8_t m = 1 << is;
      for (int l = 0; l < 8; ++l) a[8*is + l] -= (hm[l] & m ? 0 : 16);
    }

    const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
    const int8_t * restrict sc = x[i].scales;

    for (int j = 0; j < QK_K/16; ++j) {
      const float dl = d * sc[j];
      for (int l = 0; l < 16; ++l) ((int16 *)&aux16)[l] = q8[l] * a[l];
      for (int l = 0; l <  8; ++l) ((float *)&sums)[l] += dl * (((int16 *)&aux16)[l] + ((int16 *)&aux16)[8+l]);
      q8 += 16; a += 16;
    }
  }
  for (int l = 0; l < 8; ++l) sumf += ((float *)&sums)[l];
  *s = sumf;
}
