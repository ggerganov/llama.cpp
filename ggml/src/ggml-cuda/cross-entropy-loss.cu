#include "common.cuh"
#include "cross-entropy-loss.cuh"
#include "sum.cuh"

#include <cmath>
#include <cstdint>

static __global__ void cross_entropy_loss_f32(const float * logits, const float * labels, float * dst, const int nclasses, const int k) {
    logits += blockIdx.x*nclasses;
    labels += blockIdx.x*nclasses;

    // Find maximum for softmax:
    float max_logit = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        max_logit = fmaxf(max_logit, logits[i]);
    }
    max_logit = warp_reduce_max(max_logit);

    // Calculate log(softmax(logits)) which is just logits - max:
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        sum += expf(logits[i] - max_logit);
    }
    sum = warp_reduce_sum(sum);
    sum = logf(sum);

    // log(exp(logits - max) / sum) = (logits - max) - log(sum)
    float loss = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        loss += (logits[i] - max_logit - sum) * labels[i];
    }
    loss = -warp_reduce_sum(loss) / (float)k;

    if (threadIdx.x != 0) {
        return;
    }

    dst[blockIdx.x] = loss;
}

static __global__ void cross_entropy_loss_back_f32(const float * logits, const float * labels, const float * loss, float * dst, const int nclasses) {
    extern __shared__ float tmp[];

    float maxval = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[blockIdx.x*nclasses + i];
        maxval = fmaxf(maxval, val);
        tmp[i] = val;
    }
    maxval = warp_reduce_max(maxval);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = expf(tmp[i] - maxval);
        sum += val;
        tmp[i] = val;
    }
    sum = warp_reduce_sum(sum);
    const float sm_scale = 1.0f/sum;

    const float d_by_nrows = *loss/gridDim.x;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        dst[blockIdx.x*nclasses + i] = (tmp[i]*sm_scale - labels[blockIdx.x*nclasses + i])*d_by_nrows;
    }
}

void ggml_cuda_cross_entropy_loss(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const int shmem = 0;

    ggml_cuda_pool_alloc<float> dst_tmp(pool, blocks_num.x);

    cross_entropy_loss_f32<<<blocks_num, blocks_dim, shmem, stream>>>(src0_d, src1_d, dst_tmp.ptr, ne00, nrows);
    CUDA_CHECK(cudaGetLastError());

    // Combine results from individual blocks:
    sum_f32_cuda(pool, dst_tmp.ptr, dst_d, blocks_num.x, stream);
}

void ggml_cuda_cross_entropy_loss_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * opt0 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(opt0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(opt0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * opt0_d = (const float *) opt0->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const int shmem = ne00*sizeof(float);

    cross_entropy_loss_back_f32<<<blocks_num, blocks_dim, shmem, stream>>>(src0_d, src1_d, opt0_d, dst_d, ne00);
}
