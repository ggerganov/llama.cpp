#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h> /*for CLOCK_REALTIME? */
#include <time.h>

#include "ggml-phi-knc.h"

#define MAXVEC 1024768
#define RUNTOTAL 12
#define RUNS
int main(void)
{
  struct timespec start, middle, end;
  double vector_time;
  double scalar_time;
  float scalar = 0.0f;
  float vector = 0.0f;
  int vecRuns[RUNSTOTAL] = {10, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 1024768};

  for (uint32_t runCount = 0; runCount < RUNTOTAL; ++runCount)
    {
      // Generate random input vector of [-1, 1] values.
      float vec1[MAXVEC] __attribute__((aligned(64)));
      for (int i = 0; i < vecRuns[runCount]; i++)
        vec1[i] = 2 * (0.5 - rand() / (float)RAND_MAX);

      // Generate a second random input vector of [-1, 1] values.
      float vec2[MAXVEC] __attribute__((aligned(64)));
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

  return 0;
}
