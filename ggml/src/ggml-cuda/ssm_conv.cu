#include "ssm_conv.cuh"

template <int block_size>
static __global__ void ssm_conv_f32(
    const float * __restrict__ src0, const float * __restrict__ src1,
    const int src0_nb0, const int src0_nb1, const int src0_nb2,
    const int src1_nb1,
    float * dst,
    const int dst_nb0, const int dst_nb1, const int dst_nb2,
    const int nc, const int ncs, const int nr) {

//    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;
    const int i2 = blockIdx.x;
    const int i3 = threadIdx.y;

    const int ith = tid;
    const int nth = WARP_SIZE;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = min(ir0 + dr, nr);
    const int ir  = ir1 - ir0;

    // {d_conv - 1 + n_t, d_inner, n_seqs}
    // sliding window
    const float * s = (const float *) ((const char *) src0 + ir0*src0_nb1 + i2*src0_nb0 + i3*src0_nb2); // {d_conv, d_inner, n_s}
    const float * c = (const float *) ((const char *) src1 + ir0*src1_nb1); // {d_conv, d_inner}
    float * x = (float *) ((char *) dst + ir0*dst_nb0 + i2*dst_nb1 + i3*dst_nb2); // {d_inner, n_t, n_s}
    // TODO: transpose the output for smaller strides for big batches?
    // d_inner
    for (int i1 = 0; i1 < ir; ++i1) {
        // rowwise dot product
        // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
        float sumf = 0.0f;
        #pragma unroll
        for (int i0 = 0; i0 < nc; ++i0) {
            sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
        }
        x[i1] = sumf;
    }
}

static void ssm_conv_f32_cuda(
    const float * src0, const float * src1,
    const int src0_nb0, const int src0_nb1, const int src0_nb2,
    const int src1_nb1,
    float * dst,
    const int dst_nb0, const int dst_nb1, const int dst_nb2,
    const int nc, const int ncs, const int nr, const int n_t, const int n_s,
    cudaStream_t stream) {

    const dim3 block_dims(WARP_SIZE, n_s, 1);
    const int nblocks = n_t;
    ssm_conv_f32<WARP_SIZE><<<nblocks, block_dims, 0, stream>>>(
        src0, src1,
        src0_nb0, src0_nb1, src0_nb2,
        src1_nb1,
        dst,
        dst_nb0, dst_nb1, dst_nb2,
        nc, ncs, nr);
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // conv_x
    const struct ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    const int nc  = src1->ne[0]; // d_conv
    const int ncs = src0->ne[0]; // d_conv - 1 + n_t
    const int nr  = src0->ne[1]; // d_inner
    const int n_t =  dst->ne[1]; // tokens per sequence
    const int n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ssm_conv_f32_cuda(src0_d, src1_d,
        src0->nb[0], src0->nb[1], src0->nb[2],
        src1->nb[1],
        dst_d,
        dst->nb[0], dst->nb[1], dst->nb[2],
        nc, ncs, nr, n_t, n_s,
        stream);
}
