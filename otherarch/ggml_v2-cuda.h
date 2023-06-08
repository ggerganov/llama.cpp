#include "ggml_v2.h"

#ifdef  __cplusplus
extern "C" {
#endif

void   ggml_v2_init_cublas(void);

bool   ggml_v2_cuda_can_mul_mat(const struct ggml_v2_tensor * src0, const struct ggml_v2_tensor * src1, struct ggml_v2_tensor * dst);
size_t ggml_v2_cuda_mul_mat_get_wsize(const struct ggml_v2_tensor * src0, const struct ggml_v2_tensor * src1, struct ggml_v2_tensor * dst);
void   ggml_v2_cuda_mul_mat(const struct ggml_v2_tensor * src0, const struct ggml_v2_tensor * src1, struct ggml_v2_tensor * dst, void * wdata, size_t wsize);

// TODO: export these with GGML_V2_API
void * ggml_v2_cuda_host_malloc(size_t size);
void   ggml_v2_cuda_host_free(void * ptr);

void ggml_v2_cuda_transform_tensor(struct ggml_v2_tensor * tensor);

#ifdef  __cplusplus
}
#endif