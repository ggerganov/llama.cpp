#pragma once

#ifdef  __cplusplus
extern "C" {
#endif

void ggml_vk_init(void);

void ggml_vk_dequantize_row_q4_0(const void * x, float * y, int k);

#ifdef  __cplusplus
}
#endif
