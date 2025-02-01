#ifndef GGML_SYCL_DIAG_MASK
#define GGML_SYCL_DIAG_MASK

#include "common.hpp"

void ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_DIAG_MASK