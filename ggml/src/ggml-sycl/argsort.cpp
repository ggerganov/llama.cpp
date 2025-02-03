#include "argsort.hpp"

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

template <typename T>
static inline void ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a     = b;
    b     = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ static void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad,
                                              const sycl::nd_item<3> & item_ct1, uint8_t * dpct_local) {
    // bitonic sort
    int col = item_ct1.get_local_id(2);
    int row = item_ct1.get_group(1);

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row   = x + row * ncols;
    auto          dst_row = (int *) dpct_local;

    // initialize indices
    dst_row[col] = col;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols &&
                         (order == GGML_SORT_ORDER_ASC ? x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                                                         x_row[dst_row[col]] < x_row[dst_row[ixj]]))) {
                        ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols &&
                         (order == GGML_SORT_ORDER_ASC ? x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                                                         x_row[dst_row[col]] > x_row[dst_row[ixj]]))) {
                        ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            /*
            DPCT1118:1: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static void argsort_f32_i32_sycl(const float * x, int * dst, const int ncols, const int nrows, ggml_sort_order order,
                                 queue_ptr stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const sycl::range<3> block_dims(1, 1, ncols_pad);
    const sycl::range<3> block_nums(1, nrows, 1);
    const size_t         shared_mem = ncols_pad * sizeof(int);

    if (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                    x, dst, ncols, ncols_pad, item_ct1,
                    dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });
    } else if (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                    x, dst, ncols, ncols_pad, item_ct1,
                    dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
        });
    } else {
        GGML_ABORT("fatal error");
    }
}

inline void ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);

    const int64_t ncols = dst->src[0]->ne[0];
    const int64_t nrows = ggml_nrows(dst->src[0]);

    enum ggml_sort_order order       = (enum ggml_sort_order) dst->op_params[0];
    dpct::queue_ptr      main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float *        src0_dd     = static_cast<const float *>(dst->src[0]->data);
    int32_t *            dst_dd      = static_cast<int32_t *>(dst->data);

    argsort_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, order, main_stream);
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_SYCL_DEBUG("call %s\n", __func__);
    ggml_sycl_op_argsort(ctx, dst);
    GGML_SYCL_DEBUG("call %s done\n", __func__);
}
