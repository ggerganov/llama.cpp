#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


bool ggml_amx_init(void);

bool ggml_compute_forward_mul_mat_use_amx(struct ggml_tensor * dst);

void ggml_mul_mat_amx(struct ggml_tensor * dst, int nth, int ith, void * wdata, int wsize);

#ifdef __cplusplus
}
#endif
