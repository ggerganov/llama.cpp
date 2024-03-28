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

#include "bcast.h"
#include "common.h"

void ggml_cann_repeat(ggml_backend_cann_context& ctx, ggml_tensor* dst);

void ggml_cann_add(ggml_backend_cann_context& ctx, ggml_tensor* dst);

void ggml_cann_leaky_relu(ggml_backend_cann_context& ctx, ggml_tensor* dst);

void ggml_cann_concat(ggml_backend_cann_context& ctx, ggml_tensor* dst);

void ggml_cann_arange(ggml_backend_cann_context& ctx, ggml_tensor* dst);

void ggml_cann_sqr(ggml_backend_cann_context& ctx, ggml_tensor* dst);

void ggml_cann_clamp(ggml_backend_cann_context& ctx, ggml_tensor* dst);

void ggml_cann_argsort(ggml_backend_cann_context& ctx, ggml_tensor* dst);

template <aclnnStatus getWorkspaceSize(const aclTensor*, const aclTensor*,
                                       aclTensor*, uint64_t*, aclOpExecutor**),
          aclnnStatus execute(void*, uint64_t, aclOpExecutor*, aclrtStream)>
void ggml_cann_mul_div(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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
void ggml_cann_activation(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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
void ggml_cann_activation(ggml_backend_cann_context& ctx, ggml_tensor* dst) {
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
