#include "ggml-cann.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_arange.h>
#include <aclnnop/aclnn_argsort.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_gelu.h>
#include <aclnnop/aclnn_hardsigmoid.h>
#include <aclnnop/aclnn_hardswish.h>
#include <aclnnop/aclnn_leaky_relu.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_relu.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_tanh.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "ggml-backend-impl.h"
#include "ggml.h"

#define MATRIX_ROW_PADDING 512
#define GGML_CANN_MAX_STREAMS 8

// Error handling macro
#define ACL_CHECK(call)                                                      \
    do {                                                                     \
        aclError err_code = call;                                            \
        if (err_code != 0) {                                                 \
            const char* err_msg = aclGetRecentErrMsg();                      \
            std::cerr << "ACL error calling \"" #call "\", code is "         \
                      << err_code << ", reason is " << err_msg << std::endl; \
            GGML_ASSERT(err_code);                                           \
        }                                                                    \
    } while (0);

static void ggml_cann_set_device(const uint32_t device) {
    // TODO: uncomment these lines after empty context has fixed.
    // int current_device;
    // ACL_CHECK(aclrtGetDevice(&current_device));

    // if (device == current_device) {
    //   return;
    // }
    ACL_CHECK(aclrtSetDevice(device));
}

// Backend context
struct ggml_backend_cann_buffer_type_context {
    uint32_t device;
    std::string name;
};

struct ggml_backend_cann_context {
    uint32_t device;
    std::string name;

    aclrtStream streams[GGML_CANN_MAX_DEVICES][GGML_CANN_MAX_STREAMS] = {
        {nullptr}};

    explicit ggml_backend_cann_context(int device)
        : device(device), name(GGML_CANN_NAME + std::to_string(device)) {}

    ~ggml_backend_cann_context() {
        for (int i = 0; i < GGML_CANN_MAX_DEVICES; ++i) {
            for (int j = 0; j < GGML_CANN_MAX_STREAMS; ++j) {
                if (streams[i][j] != nullptr) {
                    ACL_CHECK(aclrtDestroyStream(streams[i][j]));
                }
            }
        }
    }

    aclrtStream stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtCreateStreamWithConfig(&streams[device][stream], 0,
                                                  ACL_STREAM_FAST_LAUNCH));
        }
        return streams[device][stream];
    }

    aclrtStream stream() { return stream(device, 0); }
};

struct ggml_backend_cann_buffer_context {
    uint32_t device;
    void* dev_ptr = nullptr;
    std::string name;

    ggml_backend_cann_buffer_context(uint32_t device, void* dev_ptr)
        : device(device),
          dev_ptr(dev_ptr),
          name(GGML_CANN_NAME + std::to_string(device)) {}

    //~ggml_backend_cann_buffer_context() { ACL_CHECK(aclrtFree(dev_ptr)); }
};

/**
 * Mapping ggml_tensor type to acl_tensor type.
 */
static aclDataType type_mapping(ggml_type type) {
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
static aclTensor* create_acl_tensor(const ggml_tensor* tensor,
                                    const int64_t* bcast_ne = nullptr,
                                    int64_t* bcast_stride = nullptr,
                                    int64_t bcast_dims = 0) {
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
static int64_t get_bcast_shape(const ggml_tensor* src0, const ggml_tensor* src1,
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
static bool need_bcast(const ggml_tensor* t0, const ggml_tensor* t1) {
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t1->ne[i] != t0->ne[i] && t1->ne[i] != 1) {
            return true;
        }
    }
    return false;
}

// aclnn operators

// Bcast macro to avoid duplicate code.
#define BCAST_SHAPE(src0, src1)                                       \
    int64_t bcast_ne_##src0[GGML_MAX_DIMS * 2];                       \
    int64_t bcast_ne_##src1[GGML_MAX_DIMS * 2];                       \
    int64_t bcast_stride_##src0[GGML_MAX_DIMS * 2];                   \
    int64_t bcast_stride_##src1[GGML_MAX_DIMS * 2];                   \
    int64_t bcast_dims =                                              \
        get_bcast_shape(src0, src1, bcast_ne_##src0, bcast_ne_##src1, \
                        bcast_stride_##src0, bcast_stride_##src1);

#define BCAST_PARAM(src) bcast_ne_##src, bcast_stride_##src, bcast_dims

// TODO: repeat is implemented through add to apply bcast. Optimize it.
static void ggml_cann_repeat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(ggml_can_repeat(src, dst));

    size_t nbytes = ggml_nbytes(dst);
    aclrtStream main_stream = ctx.stream();
    // Set dst to a zero tensor.
    ACL_CHECK(aclrtMemsetAsync(dst->data, nbytes, 0, nbytes, main_stream));

    aclTensor* acl_src;
    aclTensor* acl_dst;

    // Short cut for same shape.
    if (ggml_are_same_shape(src, dst)) {
        ACL_CHECK(aclrtMemcpyAsync(dst->data, nbytes, src->data, nbytes,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, main_stream));
    } else {
        if (need_bcast(dst, src)) {
            BCAST_SHAPE(dst, src);
            acl_dst = create_acl_tensor(dst, BCAST_PARAM(dst));
            acl_src = create_acl_tensor(src, BCAST_PARAM(src));
        } else {
            acl_dst = create_acl_tensor(dst);
            acl_src = create_acl_tensor(src);
        }

        // Add src0 to dst.
        aclScalar* alpha = nullptr;
        int alphaValue = 1;
        alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_INT32);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        void* workspaceAddr = nullptr;

        ACL_CHECK(aclnnInplaceAddGetWorkspaceSize(acl_dst, acl_src, alpha,
                                                  &workspaceSize, &executor));
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                                  ACL_MEM_MALLOC_HUGE_FIRST));
        }

        ACL_CHECK(aclnnInplaceAdd(workspaceAddr, workspaceSize, executor,
                                  main_stream));

        ACL_CHECK(aclDestroyScalar(alpha));
        ACL_CHECK(aclDestroyTensor(acl_src));
        ACL_CHECK(aclDestroyTensor(acl_dst));

        if (workspaceSize > 0) {
            ACL_CHECK(aclrtFree(workspaceAddr));
        }
    }
}

static void ggml_cann_add(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    aclTensor* acl_src0;
    aclTensor* acl_src1;
    aclTensor* acl_dst;

    // Need bcast
    if (!ggml_are_same_shape(src0, src1) && need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = create_acl_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = create_acl_tensor(src1, BCAST_PARAM(src1));
        acl_dst = create_acl_tensor(dst, BCAST_PARAM(src0));
    } else {
        acl_src0 = create_acl_tensor(src0);
        acl_src1 = create_acl_tensor(src1);
        acl_dst = create_acl_tensor(dst);
    }

    aclScalar* alpha = nullptr;
    float alphaValue = 1.0f;
    alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnAddGetWorkspaceSize(acl_src0, acl_src1, alpha, acl_dst,
                                       &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnAdd(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(alpha));
    ACL_CHECK(aclDestroyTensor(acl_src0));
    ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

template <aclnnStatus getWorkspaceSize(const aclTensor*, const aclTensor*,
                                       aclTensor*, uint64_t*, aclOpExecutor**),
          aclnnStatus execute(void*, uint64_t, aclOpExecutor*, aclrtStream)>
static void ggml_cann_mul_div(ggml_backend_cann_context& ctx,
                              ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    aclTensor* acl_src0;
    aclTensor* acl_src1;
    aclTensor* acl_dst;

    // Need bcast
    if (!ggml_are_same_shape(src0, src1) && need_bcast(src0, src1)) {
        BCAST_SHAPE(src0, src1)
        acl_src0 = create_acl_tensor(src0, BCAST_PARAM(src0));
        acl_src1 = create_acl_tensor(src1, BCAST_PARAM(src1));
        acl_dst = create_acl_tensor(dst, BCAST_PARAM(src0));
    } else {
        acl_src0 = create_acl_tensor(src0);
        acl_src1 = create_acl_tensor(src1);
        acl_dst = create_acl_tensor(dst);
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(getWorkspaceSize(acl_src0, acl_src1, acl_dst, &workspaceSize,
                               &executor));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(execute(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyTensor(acl_src0));
    ACL_CHECK(aclDestroyTensor(acl_src1));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

// Activation functions template.
template <aclnnStatus getWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*,
                                       aclOpExecutor**),
          aclnnStatus execute(void*, uint64_t, aclOpExecutor*,
                              const aclrtStream)>
static void ggml_cann_activation(ggml_backend_cann_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(getWorkspaceSize(acl_src, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(execute(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

// Activation functions template for const aclTensors.
template <aclnnStatus getWorkspaceSize(const aclTensor*, const aclTensor*,
                                       uint64_t*, aclOpExecutor**),
          aclnnStatus execute(void*, uint64_t, aclOpExecutor*,
                              const aclrtStream)>
static void ggml_cann_activation(ggml_backend_cann_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(getWorkspaceSize(acl_src, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(execute(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

static void ggml_cann_leaky_relu(ggml_backend_cann_context& ctx,
                                 ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));
    aclScalar* acl_negative_slope =
        aclCreateScalar(&negative_slope, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnLeakyReluGetWorkspaceSize(
        acl_src, acl_negative_slope, acl_dst, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(
        aclnnLeakyRelu(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_negative_slope));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

static void ggml_cann_concat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src0 = dst->src[0];
    ggml_tensor* src1 = dst->src[1];
    aclTensor* acl_src0 = create_acl_tensor(src0);
    aclTensor* acl_src1 = create_acl_tensor(src1);
    aclTensor* acl_dst = create_acl_tensor(dst);

    std::vector<aclTensor*> tmp{acl_src0, acl_src1};
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnCatGetWorkspaceSize(tensorList, 2, acl_dst, &workspaceSize,
                                       &executor));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnCat(workspaceAddr, workspaceSize, executor, main_stream));

    aclDestroyTensorList(tensorList);
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

static void ggml_cann_arange(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float start;
    float stop;
    float step;
    memcpy(&start, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&stop, (float*)dst->op_params + 1, sizeof(float));
    memcpy(&step, (float*)dst->op_params + 2, sizeof(float));

    int64_t steps = (int64_t)std::ceil((stop - start) / step);
    GGML_ASSERT(ggml_nelements(dst) == steps);
    aclTensor* acl_dst = create_acl_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    aclScalar* acl_start = aclCreateScalar(&start, aclDataType::ACL_FLOAT);
    aclScalar* acl_end = aclCreateScalar(&stop, aclDataType::ACL_FLOAT);
    aclScalar* acl_step = aclCreateScalar(&step, aclDataType::ACL_FLOAT);

    ACL_CHECK(aclnnArangeGetWorkspaceSize(acl_start, acl_end, acl_step, acl_dst,
                                          &workspaceSize, &executor));
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnArange(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_start));
    ACL_CHECK(aclDestroyScalar(acl_end));
    ACL_CHECK(aclDestroyScalar(acl_step));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

static void ggml_cann_sqr(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    dst->src[1] = dst->src[0];
    ggml_cann_mul_div<aclnnMulGetWorkspaceSize, aclnnMul>(ctx, dst);
}

static void ggml_cann_clamp(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float min;
    float max;
    memcpy(&min, dst->op_params, sizeof(float));
    memcpy(&max, (float*)dst->op_params + 1, sizeof(float));

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    aclScalar* acl_min = aclCreateScalar(&min, aclDataType::ACL_FLOAT);
    aclScalar* acl_max = aclCreateScalar(&max, aclDataType::ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnClampGetWorkspaceSize(acl_src, acl_min, acl_max, acl_dst,
                                         &workspaceSize, &executor));
    if (workspaceSize > 0)
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(aclnnClamp(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyScalar(acl_min));
    ACL_CHECK(aclDestroyScalar(acl_max));
    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

// TODO: acl kernel only support INT64 for out tensors.
static void ggml_cann_argsort(ggml_backend_cann_context& ctx,
                              ggml_tensor* dst) {
    ggml_tensor* src = dst->src[0];
    enum ggml_sort_order order = (enum ggml_sort_order)dst->op_params[0];

    aclTensor* acl_src = create_acl_tensor(src);
    aclTensor* acl_dst = create_acl_tensor(dst);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    ACL_CHECK(aclnnArgsortGetWorkspaceSize(
        acl_src, 0, (order == GGML_SORT_ORDER_DESC ? true : false), acl_dst,
        &workspaceSize, &executor));
    if (workspaceSize > 0)
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize,
                              ACL_MEM_MALLOC_HUGE_FIRST));

    aclrtStream main_stream = ctx.stream();
    ACL_CHECK(
        aclnnArgsort(workspaceAddr, workspaceSize, executor, main_stream));

    ACL_CHECK(aclDestroyTensor(acl_src));
    ACL_CHECK(aclDestroyTensor(acl_dst));

    if (workspaceSize > 0) {
        ACL_CHECK(aclrtFree(workspaceAddr));
    }
}

// CANN backend.
static ggml_guid_t ggml_backend_cann_guid() {
    static ggml_guid guid = {0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                             0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64};
    return &guid;
}

GGML_CALL bool ggml_backend_is_cann(ggml_backend_t backend) {
    return backend != NULL &&
           ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

GGML_CALL uint32_t ggml_backend_cann_get_device_count() {
    return ggml_cann_get_device_count();
}

GGML_CALL uint32_t ggml_cann_get_device_count() {
    uint32_t device_count;
    if (aclrtGetDeviceCount(&device_count) != ACL_SUCCESS) {
        return 0;
    }
    return device_count;
}

GGML_CALL void ggml_cann_get_device_description(uint32_t device,
                                                char* description,
                                                size_t description_size) {
    ggml_cann_set_device(device);
    const char* soc_name = aclrtGetSocName();
    snprintf(description, description_size, "%s", soc_name);
}

GGML_CALL void ggml_backend_cann_get_device_description(
    uint32_t device, char* description, size_t description_size) {
    ggml_cann_get_device_description(device, description, description_size);
}

GGML_CALL void ggml_backend_cann_get_device_memory(uint32_t device,
                                                   size_t* free,
                                                   size_t* total) {
    ggml_cann_set_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

// Backend buffer
GGML_CALL static const char* ggml_backend_cann_buffer_get_name(
    ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_cann(
    ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cann_buffer_get_name;
}

GGML_CALL static void ggml_backend_cann_buffer_free_buffer(
    ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    ACL_CHECK(aclrtFree(ctx->dev_ptr));
    delete ctx;
}

GGML_CALL static void* ggml_backend_cann_buffer_get_base(
    ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void ggml_backend_cann_buffer_init_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor) {
    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        tensor->backend = tensor->view_src->backend;
        tensor->extra = tensor->view_src->extra;
        return;
    }

    tensor->backend = GGML_BACKEND_TYPE_GPU;

    if (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size =
            ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            size_t memset_size = padded_size - original_size;
            ACL_CHECK(aclrtMemset((char*)tensor->data + original_size,
                                  memset_size, 0, memset_size));
        }
    }
}

GGML_CALL static void ggml_backend_cann_buffer_set_tensor(
    ggml_backend_buffer_t buffer, ggml_tensor* tensor, const void* data,
    size_t offset, size_t size) {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_TYPE_GPU);

    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtMemcpy((char*)tensor->data + offset, size, data, size,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtSynchronizeDevice());
}

GGML_CALL static void ggml_backend_cann_buffer_get_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data,
    size_t offset, size_t size) {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_TYPE_GPU);

    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtMemcpy(data, size, (const char*)tensor->data + offset, size,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(aclrtSynchronizeDevice());
}

GGML_CALL static bool ggml_backend_cann_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
    if (ggml_backend_buffer_is_cann(src->buffer)) {
        ggml_backend_cann_buffer_context* src_ctx =
            (ggml_backend_cann_buffer_context*)src->buffer->context;
        ggml_backend_cann_buffer_context* dst_ctx =
            (ggml_backend_cann_buffer_context*)buffer->context;

        ggml_cann_set_device(src_ctx->device);
        ACL_CHECK(aclrtSynchronizeDevice());
        ggml_cann_set_device(dst_ctx->device);
        ACL_CHECK(aclrtSynchronizeDevice());
        size_t memcpy_size = ggml_nbytes(src);
        ACL_CHECK(aclrtMemcpy((char*)dst->data, memcpy_size,
                              (const char*)src->data, memcpy_size,
                              ACL_MEMCPY_DEVICE_TO_DEVICE));
        ACL_CHECK(aclrtSynchronizeDevice());

        return true;
    }
    return false;
}

GGML_CALL static void ggml_backend_cann_buffer_clear(
    ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cann_buffer_context* ctx =
        (ggml_backend_cann_buffer_context*)buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, buffer->size, value, buffer->size));
    ACL_CHECK(aclrtSynchronizeDevice());
}

static ggml_backend_buffer_i ggml_backend_cann_buffer_interface = {
    /* .get_name        = */ ggml_backend_cann_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cann_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cann_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cann_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_cann_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cann_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cann_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_buffer_clear,
    /* .reset           = */ NULL,
};

// Backend buffer type
GGML_CALL static const char* ggml_backend_cann_buffer_type_name(
    ggml_backend_buffer_type_t buft) {
    ggml_backend_cann_buffer_type_context* ctx =
        (ggml_backend_cann_buffer_type_context*)buft->context;

    return ctx->name.c_str();
}

GGML_CALL static ggml_backend_buffer_t
ggml_backend_cann_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {
    ggml_backend_cann_buffer_type_context* buft_ctx =
        (ggml_backend_cann_buffer_type_context*)buft->context;

    ggml_cann_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1);

    void* dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        fprintf(
            stderr,
            "%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n",
            __func__, size / 1024.0 / 1024.0, buft_ctx->device,
            aclGetRecentErrMsg());
        return nullptr;
    }

    ggml_backend_cann_buffer_context* ctx =
        new ggml_backend_cann_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cann_buffer_interface,
                                    ctx, size);
}

GGML_CALL static size_t ggml_backend_cann_buffer_type_get_alignment(
    ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cann_buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft, const ggml_tensor* tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(
                tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

GGML_CALL static bool ggml_backend_cann_buffer_type_supports_backend(
    ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    if (!ggml_backend_is_cann(backend)) {
        return false;
    }

    ggml_backend_cann_buffer_type_context* buft_ctx =
        (ggml_backend_cann_buffer_type_context*)buft->context;
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return buft_ctx->device == cann_ctx->device;
}

static ggml_backend_buffer_type_i ggml_backend_cann_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cann_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cann_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cann_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cann_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_cann_buffer_type_supports_backend,
    /* .is_host          = */ NULL,
};

GGML_CALL ggml_backend_buffer_type_t
ggml_backend_cann_buffer_type(uint32_t device) {
    // TODO: this is not thread safe
    if (device >= ggml_backend_cann_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type
        ggml_backend_cann_buffer_types[GGML_CANN_MAX_DEVICES];

    static bool ggml_backend_cann_buffer_type_initialized = false;

    if (!ggml_backend_cann_buffer_type_initialized) {
        for (uint32_t i = 0; i < GGML_CANN_MAX_DEVICES; i++) {
            ggml_backend_cann_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cann_buffer_type_interface,
                /* .context  = */
                new ggml_backend_cann_buffer_type_context{
                    i, GGML_CANN_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cann_buffer_type_initialized = true;
    }

    return &ggml_backend_cann_buffer_types[device];
}

// Backend
GGML_CALL static const char* ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return cann_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ACL_CHECK(aclFinalize());

    delete cann_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t
ggml_backend_cann_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    return ggml_backend_cann_buffer_type(cann_ctx->device);
}

GGML_CALL static void ggml_backend_cann_set_tensor_async(ggml_backend_t backend,
                                                         ggml_tensor* tensor,
                                                         const void* data,
                                                         size_t offset,
                                                         size_t size) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    GGML_ASSERT(tensor->buffer->buft ==
                    ggml_backend_cann_buffer_type(cann_ctx->device) &&
                "unsupported buffer type");
    GGML_ASSERT(tensor->backend == GGML_BACKEND_TYPE_GPU);

    ACL_CHECK(aclrtMemcpyAsync((char*)tensor->data + offset, size, data, size,
                               ACL_MEMCPY_HOST_TO_DEVICE, cann_ctx->stream()));
}

GGML_CALL static void ggml_backend_cann_get_tensor_async(
    ggml_backend_t backend, const ggml_tensor* tensor, void* data,
    size_t offset, size_t size) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    GGML_ASSERT(tensor->buffer->buft ==
                    ggml_backend_cann_buffer_type(cann_ctx->device) &&
                "unsupported buffer type");
    GGML_ASSERT(tensor->backend == GGML_BACKEND_TYPE_GPU);

    ACL_CHECK(aclrtMemcpyAsync(data, size, (const char*)tensor->data + offset,
                               size, ACL_MEMCPY_DEVICE_TO_HOST,
                               cann_ctx->stream()));
}

GGML_CALL static bool ggml_backend_cann_cpy_tensor_async(
    ggml_backend_t backend_src, ggml_backend_t backend_dst,
    const ggml_tensor* src, ggml_tensor* dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) ||
                ggml_backend_is_cann(backend_dst));

    if (!ggml_backend_buffer_is_cann(src->buffer) ||
        !ggml_backend_buffer_is_cann(dst->buffer)) {
        return false;
    }

    ggml_backend_cann_context* cann_ctx_src =
        (ggml_backend_cann_context*)backend_src->context;
    ggml_backend_cann_context* cann_ctx_dst =
        (ggml_backend_cann_context*)backend_dst->context;
    GGML_UNUSED(cann_ctx_src);

    // TODO: not support peer copy yet.
    GGML_ASSERT(backend_src == backend_dst);

    // src and dst are on the same backend
    size_t copy_size = ggml_nbytes(dst);
    ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size,
                               ACL_MEMCPY_DEVICE_TO_DEVICE,
                               cann_ctx_dst->stream()));
    return true;
}

GGML_CALL static void ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
}

static bool ggml_cann_compute_forward(ggml_backend_cann_context& ctx,
                                      struct ggml_tensor* dst) {
    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
        case GGML_OP_DUP:
            return false;
        case GGML_OP_ADD:
            ggml_cann_add(ctx, dst);
            break;
        case GGML_OP_ACC:
            return false;
        case GGML_OP_MUL:
            ggml_cann_mul_div<aclnnMulGetWorkspaceSize, aclnnMul>(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cann_mul_div<aclnnDivGetWorkspaceSize, aclnnDiv>(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cann_activation<aclnnGeluGetWorkspaceSize, aclnnGelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cann_activation<aclnnSiluGetWorkspaceSize, aclnnSilu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cann_activation<aclnnGeluGetWorkspaceSize, aclnnGelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cann_activation<aclnnTanhGetWorkspaceSize, aclnnTanh>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cann_activation<aclnnReluGetWorkspaceSize, aclnnRelu>(
                        ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cann_activation<aclnnHardsigmoidGetWorkspaceSize,
                                         aclnnHardsigmoid>(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cann_activation<aclnnHardswishGetWorkspaceSize,
                                         aclnnHardswish>(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
            return false;
        case GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
            return false;
        case GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            return false;
        case GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_SCALE:
            return false;
        case GGML_OP_SQR:
            ggml_cann_sqr(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ALIBI:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
            return false;
        case GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

GGML_CALL static enum ggml_status ggml_backend_cann_graph_compute(
    ggml_backend_t backend, ggml_cgraph* cgraph) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    ggml_cann_set_device(cann_ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor* node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_VIEW ||
            node->op == GGML_OP_NONE) {
            continue;
        }

        bool ok = ggml_cann_compute_forward(*cann_ctx, node);
        if (!ok) {
            fprintf(stderr, "%s: error: op not supported %s (%s)\n", __func__,
                    node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_cann_supports_op(ggml_backend_t backend,
                                                    const ggml_tensor* op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_GET_ROWS:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
            return false;
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
        case GGML_OP_NONE:
            return true;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
            return false;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return true;
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
            return false;
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
            return true;
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ALIBI:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
            return false;
        case GGML_OP_ARGSORT:
            return true;
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
            return false;
        case GGML_OP_ARANGE:
            return true;
        case GGML_OP_TIMESTEP_EMBEDDING:
            return false;
        case GGML_OP_LEAKY_RELU:
            return true;
        default:
            return false;
    }

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_cann_offload_op(ggml_backend_t backend,
                                                   const ggml_tensor* op) {
    GGML_UNUSED(backend);
    GGML_UNUSED(op);
    return false;
}

static ggml_backend_event_t ggml_backend_cann_event_new(
    ggml_backend_t backend) {
    // TODO: cann doesn't support peer copy.
    GGML_UNUSED(backend);
    return nullptr;
}

static void ggml_backend_cann_event_free(ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent)event->context));

    delete event;
}

static void ggml_backend_cann_event_record(ggml_backend_event_t event) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)event->backend->context;

    ACL_CHECK(aclrtRecordEvent((aclrtEvent)event->context, cann_ctx->stream()));
}

static void ggml_backend_cann_event_wait(ggml_backend_t backend,
                                         ggml_backend_event_t event) {
    ggml_backend_cann_context* cann_ctx =
        (ggml_backend_cann_context*)backend->context;

    if (ggml_backend_is_cann(event->backend)) {
        ACL_CHECK(aclrtStreamWaitEvent(cann_ctx->stream(),
                                       (aclrtEvent)event->context));
    } else {
        GGML_ASSERT(false);
    }
}

static void ggml_backend_cann_event_synchronize(ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent)event->context));
}

static ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .get_default_buffer_type = */ ggml_backend_cann_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_cann_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cann_graph_compute,
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_event_new,
    /* .event_free              = */ ggml_backend_cann_event_free,
    /* .event_record            = */ ggml_backend_cann_event_record,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .event_synchronize       = */ ggml_backend_cann_event_synchronize,
};

struct ggml_cann_device_info {
    uint32_t device_count;

    struct cann_device_info {
        int     cc;                 // compute capability
        size_t  smpb;               // max. shared memory per block
        bool    vmm;                // virtual memory support
        size_t  vmm_granularity;    // granularity of virtual memory
        size_t  total_vram;
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {};
};

GGML_CALL void ggml_init_cann() {
    static bool initialized = false;
    ggml_cann_device_info info = {};

    if (!initialized) {
        ACL_CHECK(aclInit(nullptr));
        if (aclrtGetDeviceCount(&info.device_count) != ACL_SUCCESS) {
            initialized = true;
            fprintf(stderr,
                    "%s: no " GGML_CANN_NAME " devices found, " GGML_CANN_NAME
                    " will be disabled\n",
                    __func__);
            return;
        }

        GGML_ASSERT(info.device_count <= GGML_CANN_MAX_DEVICES);
        initialized = true;
    }
}

GGML_CALL ggml_backend_t ggml_backend_cann_init(uint32_t device) {
    ggml_init_cann();

    ggml_cann_set_device(device);

    if (device >= ggml_cann_get_device_count()) {
        fprintf(stderr, "%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cann_context* ctx = new ggml_backend_cann_context(device);

    ggml_backend_t cann_backend =
        new ggml_backend{/* .guid      = */ ggml_backend_cann_guid(),
                         /* .interface = */ ggml_backend_cann_interface,
                         /* .context   = */ ctx};

    return cann_backend;
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_cann_init(const char* params,
                                                           void* user_data) {
    ggml_backend_t cann_backend =
        ggml_backend_cann_init((int)(intptr_t)user_data);
    return cann_backend;

    GGML_UNUSED(params);
}

extern "C" GGML_CALL int ggml_backend_cann_reg_devices();

GGML_CALL int ggml_backend_cann_reg_devices() {
    uint32_t device_count = ggml_cann_get_device_count();
    // initialization
    for (uint32_t i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_CANN_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_cann_init,
                              ggml_backend_cann_buffer_type(i),
                              (void*)(intptr_t)i);
    }
    return device_count;
}