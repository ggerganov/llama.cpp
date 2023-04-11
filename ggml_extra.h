#pragma once

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef  __cplusplus
// restrict not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif

void kQuantizeQ4_0(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);

void kQuantizeQ4_1(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);

#ifdef  __cplusplus
}
#endif
