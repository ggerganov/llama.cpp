#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

/* A forward declaration, to keep GCC happy. */
void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy,  size_t by, int nrc);

#ifdef  __cplusplus
}
#endif
