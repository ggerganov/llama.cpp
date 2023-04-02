#pragma once

// Internal functions exposed for tests and benchmarks

#ifdef  __cplusplus
// restrict not standard in C++
#define restrict
extern "C" {
#endif

typedef void (*dequantize_row_q_t)(const void * restrict x, float * restrict y, int k);
typedef void (*quantize_row_q_t)(const float * restrict x, void * restrict y, int k);
typedef void (*vec_dot_q_t)(const int n, float * restrict s, const void * restrict x, const void * restrict y);

typedef struct {
    dequantize_row_q_t dequantize_row_q;
    quantize_row_q_t   quantize_row_q;
    vec_dot_q_t        vec_dot_q;
} quantize_fns_t;

quantize_fns_t ggml_internal_get_quantize_fn(size_t i);

#ifdef  __cplusplus
}
#endif
