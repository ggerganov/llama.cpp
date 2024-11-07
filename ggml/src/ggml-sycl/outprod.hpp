#ifndef GGML_SYCL_OUTPROD_HPP
#define GGML_SYCL_OUTPROD_HPP

#include "common.hpp"

void ggml_sycl_op_out_prod(ggml_backend_sycl_context& ctx, const ggml_tensor* src0,
    const ggml_tensor* src1, ggml_tensor* dst);


#endif // GGML_SYCL_OUTPROD_HPP

