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

#ifdef  __cplusplus
}
#endif
