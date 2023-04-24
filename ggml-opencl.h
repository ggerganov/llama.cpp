#pragma once

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast_c.h>
#define MAX_CL_BUFFERS 16

#ifdef  __cplusplus
extern "C" {
#endif

// Buffer reuse code adapted from cuda implementation by slaren
#define CL_CHECK(err, name)                                                           \
    do {                                                                                \
        cl_int err_ = (err);                                                       \
        if (err_ != CL_SUCCESS) {                                                      \
            fprintf(stderr, "OpenCL %s error %d at %s:%d\n", name, err_, __FILE__, __LINE__);   \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

cl_mem ggml_cl_pool_malloc(size_t size, size_t * actual_size);
void ggml_cl_pool_free(cl_mem mem, size_t size);

cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer);
void ggml_cl_init(void);

void ggml_cl_sgemm_wrapper(const CLBlastLayout order, const CLBlastTranspose trans_a, const CLBlastTranspose trans_b, const int m, const int n, const int k, const float alpha, const void *host_a, const int lda, const float *host_b, const int ldb, const float beta, float *host_c, const int ldc, const int btype);

#ifdef  __cplusplus
}
#endif
