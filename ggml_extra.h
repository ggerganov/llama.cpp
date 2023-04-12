#pragma once

#ifdef  __cplusplus
#include <cstdint>
#include <cstddef>
extern "C" {
#else
#include <stdint.h>
#include <stddef.h>
#endif

#ifdef  __cplusplus
// restrict not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif

void kQuantizeQ4_0(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
size_t kQuantizeQ4_0H(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k, int64_t* hist);

void kQuantizeQ4_1(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
size_t kQuantizeQ4_1H(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k, int64_t* hist);

void kQuantizeQ5_1(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
size_t kQuantizeQ5_1H(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k, int64_t* hist);
void kQuantizeQ5_1_Fast(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
size_t kQuantizeQ5_1H_Fast(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k, int64_t* hist);
void kDequantizeQ5_1(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

void kQuantizeQ4_0K(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void kDequantizeQ4_0K(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);

#ifdef  __cplusplus
}
#endif
