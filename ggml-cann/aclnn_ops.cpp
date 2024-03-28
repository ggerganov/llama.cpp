#include "aclnn_ops.h"

#include <cmath>
#include <cstring>
#include <vector>

// TODO: repeat is implemented through add to apply bcast. Optimize it.
void ggml_cann_repeat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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

void ggml_cann_add(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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

void ggml_cann_leaky_relu(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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

void ggml_cann_concat(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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

void ggml_cann_arange(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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

void ggml_cann_sqr(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
    dst->src[1] = dst->src[0];
    ggml_cann_mul_div<aclnnMulGetWorkspaceSize, aclnnMul>(ctx, dst);
}

void ggml_cann_clamp(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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
void ggml_cann_argsort(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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
