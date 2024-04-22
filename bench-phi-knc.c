/* bench-phi-knc.c: benchmarks and tests for the Xeon PHI Knights Corner optimizations. */

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* For CLOCK_REALTIME? */
#include <unistd.h>
#include <time.h>

/* For memcpy */
#include <string.h>

/* include the increasingly inacurately named header for our F32 dot product code. */
#include "ggml-phi-knc.h"

/* include the header for our Q8K_Q5K dot product code. */
#include "ggml-phi-knc-dot_q5_K_q8_K.h"

// largest Float32 vectors to get the dot product of.
#define F32_MAXVEC 1024768
// how many benchmarks we will run in total.
#define F32_RUNCOUNT 12
#define F32_ITEMS_PER_RUN {10, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 1024768}

int main(void)
{
  int vecRuns[F32_RUNCOUNT] = F32_ITEMS_PER_RUN;

  // seed the random number generator.
  srand(time(NULL));

  // Run benchmarks for our F32 dot product functions. Benchmark them against a naieve implementation.
  for (uint8_t runCount = 0; runCount < F32_RUNCOUNT; ++runCount)
    {
      struct timespec start, middle, end;
      double vector_time;
      double scalar_time;
      float scalar = 0.0f;
      float vector = 0.0f;

      // Generate random input vector of [-1, 1] values.
      float vec1[F32_MAXVEC] __attribute__((aligned(64)));
      for (int i = 0; i < vecRuns[runCount]; i++)
        vec1[i] = 2 * (0.5 - rand() / (float)RAND_MAX);

      // Generate a second random input vector of [-1, 1] values.
      float vec2[F32_MAXVEC] __attribute__((aligned(64)));
      for (int i = 0; i < vecRuns[runCount]; i++)
        vec2[i] = 2 * (0.5 - rand() / (float)RAND_MAX);

      // on your mark..
      clock_gettime(CLOCK_MONOTONIC, &start);

      // call dot product
      ggml_vec_dot_f32(vecRuns[runCount], &vector, 0, vec1, 0, vec2, 0, 0);

      // save the middle point..
      clock_gettime(CLOCK_MONOTONIC, &middle);

      // do the same work by hand;
      for (int i = 0; i < vecRuns[runCount]; ++i)
        scalar += vec1[i]*vec2[i];

      clock_gettime(CLOCK_MONOTONIC, &end);

      printf("vector\tvs\tscalar (%d items)\n", vecRuns[runCount]);
      printf("%.9f\tvs\t%.9f\n", vector, scalar);

      vector_time = middle.tv_sec - start.tv_sec;
      vector_time += (middle.tv_nsec - start.tv_nsec) / 1000000000.0;

      scalar_time = end.tv_sec - middle.tv_sec;
      scalar_time += (end.tv_nsec - middle.tv_nsec) / 1000000000.0;

      printf("%.9f\tvs\t%.9f\n", vector_time, scalar_time);
    }

  fflush(stdout);

  // Generate a random input vector of 256 4 bit values.
  uint8x16_t q4[8];
  uint8_t * q4ptr = (uint8_t *)q4;
  for (int i = 0; i < 128; i++)
    q4ptr[i] = rand() && 0xFF;

  // Generate a random input vector of 256 1 bit values.
  uint8x16_t q1[2];
  uint8_t * q1ptr = (uint8_t *)q1;
  for (int i = 0; i < 32; i++)
    q1ptr[i] = rand() && 0xFF;

  // Get our reference, unshifted result.
  uint8x16_t q5[16];
  GGML_5bit_Unpack_Unaligned(q4, (uint8_t *)q1, q5);

  printf("successfully got a Q5.\n");

  // Perform alignment tests, for GGML_5bit_Unpack_Unaligned.
  // Try to run GGML_5bit_Unpack_Unaligned with all possible misalignments, and get it to fail.
  for (uint8_t shiftCount = 1; shiftCount < 16; ++shiftCount)
    {
      uint8x16_t q5new[16];
      uint8x16_t q4Shifted[9];

      // create an off-by-shiftCount copy of q4.
      q4ptr = ((uint8_t *)q4Shifted) + shiftCount;
      memcpy (q4ptr, q4, 128);

      // call the unaligned form of this function:
      GGML_5bit_Unpack_Unaligned((uint8x16_t *)q4ptr, (uint8_t *)q1, q5new);

      for (uint32_t byteCount = 0; byteCount < 256; ++byteCount)
       {
         if ( ((uint8_t *)q5new)[byteCount] != ((uint8_t *)q5)[byteCount] )
           {
             printf("whoops!\nshiftCount: %d\nbyteCount: %d\n", shiftCount, byteCount);
             exit (-1);
           }
       }

      printf("Got a Q5 offset by %d\n", shiftCount);
    }

  // Generate a random input vector of 256 8 bit values.
  int8x16_t q8[16];
  int8_t * q8ptr = (int8_t *)q8;
  for (int i = 0; i < 256; i++)
    q8ptr[i] = rand() && 0xFF;

  // Generate eight random scales, one for each pair of sums.
  uint8_t scale[8];
  for (int i = 0; i < 8; i++)
    scale[i] = rand() && 0xFF;

  // Generate a random X scale.
  float rndScaleX = 2 * (0.5 - rand() / (float)RAND_MAX);
  ggml_fp16_t scaleX = GGML_PHI_FP32_TO_FP16(rndScaleX);

  // Display the random X scale. Verifies FP32_TO_FP16_TO_FP32 is working.
  printf("rndScaleX: %f\n", rndScaleX);
  printf("scaleX: %x\n", scaleX);
  printf("newScaleX: %f\n", GGML_PHI_FP16_TO_FP32(scaleX));

  // Generate a random Y scale.
  float scaleY = 2 * (0.5 - rand() / (float)RAND_MAX);
  printf("scaleY: %f\n", scaleY);

  // Create a place for our golden result.
  float32x16_t res;

  // Clear res.
  GGML_F32x16_VEC_ZERO(&res);

  // Generate an initial result, to compare to.
  GGML_8X_2xI8x16_2xI8x16_MUL_2xI16x16_S_FMA_I32x16_Unaligned (q8, q5, scale, scaleX, scaleY, &res);

  // Generate a sum of the result.
  float sum = 0.0f;
  for (int l = 0; l < 16; ++l) sum += ((float *)&res)[l];

  printf("Got a res: %f\n", sum);

  // Perform alignment tests, for GGML_8X_2xI8x16_2xI8x16_MUL_2xI16x16_S_FMA_I32x16_Unaligned.
  // try to run GGML_8X_2xI8x16_2xI8x16_MUL_2xI16x16_S_FMA_I32x16_Unaligned with all possible mis-alignments, and get it to fail.
  for (uint8_t shiftCount = 1; shiftCount < 16; ++shiftCount)
    {
      float32x16_t resNew1;
      int8x16_t q8Shifted[17];

      // Create an off-by-shiftCount copy of q8.
      q8ptr = ((int8_t *)q8Shifted)+shiftCount;
      memcpy (q8ptr, q8, 256);

      // Clear resNew.
      GGML_F32x16_VEC_ZERO(&resNew1);

      // Call the unaligned form of this function:
      GGML_8X_2xI8x16_2xI8x16_MUL_2xI16x16_S_FMA_I32x16_Unaligned ((int8x16_t *)q8ptr, q5, scale, scaleX, scaleY, &resNew1);

      // check the result against our reference.
      for (uint32_t floatCount = 0; floatCount < 64; ++floatCount)
       {
         if ( ((int8_t *)&resNew1)[floatCount] != ((int8_t *)&res)[floatCount] )
           {
             printf("whoops!\nshiftCount: %d\nfloatCount: %d\n", shiftCount, floatCount);
             for (uint32_t row = 0; row < 16 ; ++row)
               {
                 for (int col1 = 0; col1 < 4; ++col1)
                   {
                     printf("%2.2x\t", ((int8_t *)&resNew1)[(4*row)+col1]);
                   }
                 printf(" vs ");
                 for (int col2 = 0; col2 < 4; ++col2)
                   {
                     printf("%2.2x\t", ((int8_t *)&res)[(4*row)+col2]);
                   }
                 printf ("\n");
               }
             exit (-1);
           }
       }

      // Generate a sum of our new result.
      float sumf = 0.0f;
      for (int l = 0; l < 16; ++l) sumf += ((float *)&resNew1)[l];

      printf("Got a res from a Q8 offset by %d: %f\n", ((int)q8ptr) & 0x3F, sumf);
    }

  return 0;
}
