#ifndef GGML_KNC_QUANT_H
#define GGML_KNC_QUANT_H
#include <stddef.h>
/* q=2,3,4,6; packed GGML K-blocks, n multiple of 256. No nested workers. */
int ggml_knc_dot_k(int q, int n, float *out, const void *x, const void *y);
#endif
