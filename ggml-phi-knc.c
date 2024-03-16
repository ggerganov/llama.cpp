#include <immintrin.h>

#include <stdint.h>

#include <stdio.h>

static inline _Bool is_aligned(const void *restrict pointer, size_t byte_count)
{ return (uintptr_t)pointer % byte_count == 0; }

// No, we have an SIMD unit.
// #define GGML_SIMD

// This SIMD unit can work with 32 float32s at once.
#define GGML_F32_STEP 32
// We can fit 16 of these float32s in a single vector register.
#define GGML_F32_EPR 16

// because we are not defining GGML_SIMD, we have to do this ourself.
#define GGML_F32_ARR (GGML_F32_STEP/GGML_F32_EPR)

// a single vector. 128*32=512
typedef float float32x16_t __attribute__((vector_size (128)));
#define GGML_F32x16              float32x16_t

// from chatGPT. nuke this later.
#include <string.h>

inline static void GGML_F32x16_VEC_ZERO(float32x16_t *target)
{
  // we only need a mask16, but register sizes...
  __mmask32 mask=0xFFFFFFFF;

  // FIXME: how do we tell GNU AS to perform upconverts?
  float zero[4] __attribute__((aligned(64))) = {0.0f,0.0f,0.0f,0.0f};

  __asm__ __volatile__ ("movl\t%[M],\t%%eax\n\t"
			"kmov %%eax,\t%%k1\n\t"
			"vbroadcastf32x4\t%[Z],\t%%zmm0%{%%k1%}\n\t"
			"vmovaps\t\t%%zmm0,\t%[RES]%{%%k1%}\n\t"
                       : [RES]  "+m"  (*target)
                       : [M]    "m"   (mask),
                         [Z]    "m"   (zero)
                       : "eax", "k1", "zmm0");
}

// multiply each item in mvec1 with the corresponding item in mvec2, adding the result to the corresponding item in sum.
inline static void GGML_F32x16_VEC_FMA(const float32x16_t *mvec1, const float32x16_t *mvec2, float32x16_t *sumvec, size_t iterations)
{
  // we only need a mask16, but register sizes...
  __mmask32 mask=0xFFFFFFFF;
  __asm__ __volatile__ (
			"vmovaps\t\t(%[RES]),\t%%zmm0\n\t"          // load our initial state..
			"1:\n\t"
			"cmp $0,\t%[ITER]\n\t"                      // Compare iterations to 0
			"je\t2f\n\t"                                // Jump to label 2 if zero (end of loop)
			"vmovaps\t\t(%[VEC1]),\t%%zmm1\n\t"         // Load two vectors.
			"vmovaps\t\t(%[VEC2]),\t%%zmm2\n\t"
			"vfmadd231ps\t%%zmm1,\t%%zmm2,\t%%zmm0\n\t" // Perform a fused multiply add.
			"add $64,\t%[VEC1]\n\t"                     // Move to the next float32x16_t (64 bytes ahead)
			"add $64,\t%[VEC2]\n\t"
			"sub $1,\t%[ITER]\n\t"                      // Decrement iterations
			"jmp 1b\n\t"                                // Jump back to the start of the loop
			"2: \n\t"                                   // Label for loop end
			"vmovaps\t\t%%zmm0,\t(%[RES])\n\t"          // save our results.
			: [RES]  "+r" (sumvec),
			  [ITER] "+r"  (iterations)
			: [M]     "r"  (mask),
			  [VEC1]  "r"  (mvec1),
			  [VEC2]  "r"  (mvec2)
			: "zmm0", "zmm1", "zmm2", "cc", "memory");
}


// NOTE: all inputs must be __attribute__((aligned(64)));
float DotProduct_F32(const float * restrict inVec1, const float * restrict inVec2, uint32_t count)
{
  // our single result, in the end.
  float sumf = 0.0f;

  // our sum.
  float32x16_t sum __attribute__((aligned(64)));

  // the number of vector-sized steps we will need to do.
  const uint32_t np = (count & ~(GGML_F32_EPR - 1));

  GGML_F32x16_VEC_ZERO(&sum);

  // 0 indexed cycle count
  //  for (uint32_t cycle = 0; cycle < (np/GGML_F32_EPR); ++cycle)
  GGML_F32x16_VEC_FMA((float32x16_t *)inVec1, (float32x16_t *)inVec2, &sum, np/GGML_F32_EPR);

  if (count != np)
    {
      printf("handling remainder %u\n",count-np);
      // add the leftovers, that could not be handled by the vector loop.
      // our extended last part of inVec1.
      float32x16_t v1 __attribute__((aligned(64)));
      GGML_F32x16_VEC_ZERO(&v1);
      // our extended last part of inVec2.
      float32x16_t v2 __attribute__((aligned(64)));
      GGML_F32x16_VEC_ZERO(&v2);

      memcpy(&v1, &inVec1[np], (count - np)*sizeof(float));
      memcpy(&v2, &inVec2[np], (count - np)*sizeof(float));

      GGML_F32x16_VEC_FMA(&v1,
                         &v2,
                         &sum, 1);
    }

  // reduce sum0..sumX to sumf
  for (uint32_t i=0; i <GGML_F32_EPR; ++i)
    sumf+=((float *)&sum)[i];

  return sumf;
}
