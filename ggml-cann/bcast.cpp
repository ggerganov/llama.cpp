#include "bcast.h"

/**
 * Mapping ggml_tensor type to acl_tensor type.
 */
aclDataType type_mapping(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return ACL_FLOAT;
        case GGML_TYPE_F16:
            return ACL_FLOAT16;
        case GGML_TYPE_I8:
            return ACL_INT8;
        case GGML_TYPE_I16:
            return ACL_INT16;
        case GGML_TYPE_I32:
            return ACL_INT32;
        default:
            return ACL_DT_UNDEFINED;
    }
    return ACL_DT_UNDEFINED;
}

/**
 * Transform ggml_tensor to acl_tensor. Note that ggml_tensor dimension order
 * is reversed compared to acl_tensor.
 *
 * If bcast_ne and bcast_stride is nullptr, use ggml_tensor's ne and nb.
 * otherwise, use bcast_ne bcast_stride, which means tensor dims should be
 * changed to satisfy the broadcast. @sa: get_bcast_shape.
 */
aclTensor* create_acl_tensor(const ggml_tensor* tensor, const int64_t* bcast_ne,
                             int64_t* bcast_stride, int64_t bcast_dims) {
    size_t size = ggml_nbytes(tensor);
    void* deviceAddr = nullptr;

    if (tensor->backend == GGML_BACKEND_TYPE_GPU) {
        deviceAddr = tensor->data;
    } else {
        // TODO: Consider quantification.
        GGML_ASSERT(!ggml_is_quantized(tensor->type));
        ACL_CHECK(aclrtMalloc(&deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(deviceAddr, size, tensor->data, size,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // If tensor is bcasted, Up to GGML_MAX_DIMS additional dimensions will be
    // added.
    int64_t acl_ne[GGML_MAX_DIMS * 2], acl_stride[GGML_MAX_DIMS * 2];
    if (bcast_ne == nullptr) {
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            acl_ne[i] = tensor->ne[i];
            // The step size of acl is in elements.
            acl_stride[i] = tensor->nb[i] / tensor->nb[0];
        }
    } else {
        // With bcast
        for (int i = 0; i < bcast_dims; i++) {
            acl_ne[i] = bcast_ne[i];
            acl_stride[i] = bcast_stride[i] / tensor->nb[0];
        }
    }

    int64_t dims = (bcast_dims == 0 ? GGML_MAX_DIMS : bcast_dims);
    aclTensor* acl_tensor =
        aclCreateTensor(acl_ne, dims, type_mapping(tensor->type), acl_stride, 0,
                        aclFormat::ACL_FORMAT_ND, acl_ne, dims, deviceAddr);

    return acl_tensor;
}

/**
 * Add extra dims to satisfy acl kernel's broadcast rules (same as numpy).
 * ggml_tensor dimension order is reversed compared to Python.
 * bcast src1 with src0 though adding a extra dim.
 * for example:
 * src0 -> (32,10,10,10)
 * src1 -> (16,10,10,10)
 * bcast_ne_src0 -> (16,2,10,10,10)
 * bcast_ne_src1 -> (16,1,10,10,10)
 *
 * if dim0 has padding.
 * a -> (2, 2) padding = 2
 *  a: [[1, 2, *, *]
 *      [2, 3, *, *]]
 * nb = (8, 4, 2)
 *
 * if a should bcast with b -> (2, 4)
 * b' -> (2, 2, 2)
 * b : [[1, 2, 3, 4, *, *]
 *      [5, 6, 7, 8, *, *]]
 * nb = (12, 6, 1)
 *
 * after bcast:
 * a' -> (2, 1, 2)
 * a': [[[1, 2], *, *]
 *      [[2, 3], *, *]]
 * nb = (8, 4, 2, 1)
 *
 * b' : [[[1, 2], [3, 4], *, *]
 *       [[5, 6], [7, 8], *, *]]
 * nb = (12, 6, 2, 1)
 *
 * because dim1 in a inserted dim, should add nb for dim1,
 * and all other nb moves to next in order.
 */
int64_t get_bcast_shape(const ggml_tensor* src0, const ggml_tensor* src1,
                        int64_t* bcast_ne_src0, int64_t* bcast_ne_src1,
                        int64_t* bcast_stride_src0,
                        int64_t* bcast_stride_src1) {
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    int bcast_dim_cnt = 0;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        int64_t nr = src0->ne[i] / src1->ne[i];
        bcast_ne_src0[bcast_dim_cnt] = src0->ne[i] / nr;
        bcast_ne_src1[bcast_dim_cnt] = src1->ne[i];
        bcast_stride_src0[bcast_dim_cnt] = src0->nb[i];
        bcast_stride_src1[bcast_dim_cnt] = src1->nb[i];
        bcast_dim_cnt++;
        if (nr != 1) {
            // Need to add an extra dim.
            bcast_ne_src0[bcast_dim_cnt] = nr;
            bcast_ne_src1[bcast_dim_cnt] = 1;
            bcast_stride_src0[bcast_dim_cnt] =
                bcast_stride_src0[bcast_dim_cnt - 1] *
                bcast_ne_src0[bcast_dim_cnt - 1];
            bcast_stride_src1[bcast_dim_cnt] =
                bcast_stride_src1[bcast_dim_cnt - 1] *
                bcast_ne_src1[bcast_dim_cnt - 1];
            bcast_dim_cnt++;
        }
    }
    return bcast_dim_cnt;
}

/**
 * Check if shape are not same, and no dim equals 1.
 * if any dim equals 1, acl kernel will do the broadcast.
 */
bool need_bcast(const ggml_tensor* t0, const ggml_tensor* t1) {
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            return true;
        }
    }
    return false;
}
