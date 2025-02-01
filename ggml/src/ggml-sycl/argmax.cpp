#include "argmax.hpp"

static void argmax_f32_i32_sycl(const float * x, int * dst, const int ncols, const int nrows, queue_ptr stream) {
    const sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t         shared_mem = 256 * sizeof(float);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared_data(sycl::range<1>(shared_mem / sizeof(float)), cgh);
        sycl::local_accessor<int, 1>   shared_indices(sycl::range<1>(shared_mem / sizeof(float)), cgh);

        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
            const int tid = item_ct1.get_local_id(2);
            const int row = item_ct1.get_global_id(1);

            float max_val = -INFINITY;
            int   max_idx = -1;

            for (int col = tid; col < ncols; col += 256) {
                float val = x[row * ncols + col];
                if (val > max_val) {
                    max_val = val;
                    max_idx = col;
                }
            }

            shared_data[tid]    = max_val;
            shared_indices[tid] = max_idx;
            item_ct1.barrier(sycl::access::fence_space::local_space);

            for (int stride = 256 / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    float val1 = shared_data[tid];
                    float val2 = shared_data[tid + stride];
                    if (val2 > val1) {
                        shared_data[tid]    = val2;
                        shared_indices[tid] = shared_indices[tid + stride];
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }

            if (tid == 0) {
                dst[row] = shared_indices[0];
            }
        });
    });
}

static void ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(strcmp(dst->buffer->buft->iface.get_name(dst->buffer->buft), GGML_SYCL_NAME "_Split") != 0);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    dpct::queue_ptr main_stream = ctx.stream();
    const float *   src0_dd     = static_cast<const float *>(dst->src[0]->data);
    int32_t *       dst_dd      = static_cast<int32_t *>(dst->data);
    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_argmax(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}
