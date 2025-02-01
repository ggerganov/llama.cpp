#include "diagmask.hpp"
#include <float.h>

static void diag_mask_inf_f32(const float * x, float * dst, const int ncols, const int rows_per_channel,
                              const int n_past, const sycl::nd_item<3> & item_ct1) {
    const int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1);
    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    if (col >= ncols) {
        return;
    }

    const int i = row * ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i]      = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void diag_mask_inf_f32_sycl(const float * x, float * dst, const int ncols_x, const int nrows_x,
                                   const int rows_per_channel, const int n_past, queue_ptr stream) {
    const sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    const int            block_num_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    const sycl::range<3> block_nums(1, block_num_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
        diag_mask_inf_f32(x, dst, ncols_x, rows_per_channel, n_past, item_ct1);
    });
}

inline void ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(strcmp(dst->buffer->buft->iface.get_name(dst->buffer->buft), GGML_SYCL_NAME "_Split") != 0);

    const int64_t ne00   = dst->src[0]->ne[0];
    const int64_t ne01   = dst->src[0]->ne[1];
    const int     nrows0 = ggml_nrows(dst->src[0]);

    const int       n_past      = ((int32_t *) dst->op_params)[0];
    dpct::queue_ptr main_stream = ctx.stream();
    const float *   src0_dd     = static_cast<const float *>(dst->src[0]->data);
    float *         dst_dd      = static_cast<float *>(dst->data);

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}