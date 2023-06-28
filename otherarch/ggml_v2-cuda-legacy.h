#include "ggml_v2.h"

#ifdef  __cplusplus
extern "C" {
#endif

void   ggml_v2_init_cublas_legacy(void);

void   ggml_v2_cuda_mul_mat_legacy(const struct ggml_v2_tensor * src0, const struct ggml_v2_tensor * src1, struct ggml_v2_tensor * dst, void * wdata, size_t wsize);


#ifdef  __cplusplus
}
#endif