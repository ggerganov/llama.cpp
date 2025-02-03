#include "sum.hpp"

static void k_sum_rows_f32(const float * x, float * dst, const int ncols, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(1);
    const int col = item_ct1.get_local_id(2);

    float sum = 0.0f;
    for (int i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    if (col == 0) {
        dst[row] = sum;
    }
}

static void sum_rows_f32_sycl(const float * x, float * dst, const int ncols, const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, WARP_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[intel::reqd_sub_group_size(WARP_SIZE)]] { k_sum_rows_f32(x, dst, ncols, item_ct1); });
}

inline void ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));

    const int64_t   ne          = ggml_nelements(dst->src[0]);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float *   src0_dd     = static_cast<const float *>(dst->src[0]->data);
    float *         dst_dd      = static_cast<float *>(dst->data);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

inline void ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));

    const int64_t   ncols       = dst->src[0]->ne[0];
    const int64_t   nrows       = ggml_nrows(dst->src[0]);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float *   src0_dd     = static_cast<const float *>(dst->src[0]->data);
    float *         dst_dd      = static_cast<float *>(dst->data);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_sum(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}

void ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_sum_rows(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}
