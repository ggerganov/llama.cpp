#include "clamp.hpp"

static void clamp_f32(const float * x, float * dst, const float min, const float max, const int k,
                      const sycl::nd_item<3> & item_ct1) {
    const int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    dst[i] = x[i] < min ? min : (x[i] > max ? max : x[i]);
}

static void clamp_f32_sycl(const float * x, float * dst, const float min, const float max, const int k,
                           queue_ptr stream) {
    const int num_blocks = (k + SYCL_CLAMP_BLOCK_SIZE - 1) / SYCL_CLAMP_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_CLAMP_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CLAMP_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) { clamp_f32(x, dst, min, max, k, item_ct1); });
}

inline void ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));
    const dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float *         src0_dd     = static_cast<const float *>(dst->src[0]->data);
    float *               dst_dd      = static_cast<float *>(dst->data);

    clamp_f32_sycl(src0_dd, dst_dd, min, max, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:88: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    SYCL_CHECK(0);
    */
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_clamp(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}
