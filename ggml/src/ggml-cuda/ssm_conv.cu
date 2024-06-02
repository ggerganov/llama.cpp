#include "ssm_conv.cuh"

template <int block_size>
static __global__ void ssm_conv_f32(
    const float * src0, const float * src1, const float * src2,
    const int src0_nb1, const int src0_nb2,
    const int src1_nb0, const int src1_nb1, const int src1_nb2,
    const int src2_nb1,
    float * dst,
    const int dst_nb0, const int dst_nb1, const int dst_nb2,
    const int nc, const int nr, const int n_t, const int n_s) {

//    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    const int ith = tid;
    const int nth = WARP_SIZE;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = min(ir0 + dr, nr);
    const int ir  = ir1 - ir0;

    // TODO: maybe require src0 to have d_conv columns instead of (d_conv - 1)?
    //       This would avoid having to copy into an intermediate buffer, but the state would be bigger.

//    float * s = (float *) params->wdata + (nc*dr + CACHE_LINE_SIZE_F32) * ith;
    extern __shared__ float wdata_f32[]; // work buffer for all threads
    float * s = (float *) wdata_f32 + nc*dr*ith;

    for (int i3 = 0; i3 < n_s; ++i3) {
        float * s0 = (float *) ((char *) src0 + ir0*src0_nb1) + i3*src0_nb2; // {d_conv, d_inner, n_s}

        // copy the state into working memory
        // can't use memcpy because (d_conv) != (d_conv - 1)
        for (int i1 = 0; i1 < ir; ++i1) {
            for (int i0 = 0; i0 < nc - 1; ++i0) {
                s[1 + i0 + i1*nc] = s0[i0 + i1*(nc - 1)];
            }
        }

        for (int i2 = 0; i2 < n_t; ++i2) {
            float * x  = (float *) ((char *)  dst + ir0* dst_nb0 + i2* dst_nb1 + i3* dst_nb2); // {d_inner, n_t, n_s}
            float * x0 = (float *) ((char *) src1 + ir0*src1_nb0 + i2*src1_nb1 + i3*src1_nb2); // {d_inner, n_t, n_s}
            float * c  = (float *) ((char *) src2 + ir0*src2_nb1); // {d_conv, d_inner}

            // shift state left
            //memmove(s, s + 1, (nc*ir - 1) * sizeof(float));
            for (int i4 = 0; i4 < nc*ir - 1; ++i4) {
                s[i4] = s[i4+1];
            }

            // d_inner
            for (int i1 = 0; i1 < ir; ++i1) {
                // insert x on the last column
                s[(nc - 1) + i1*nc] = x0[i1];
            }

            // it seems a little faster when this is separate from the state shift
            for (int i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                float sumf = 0.0f;
                for (int i0 = 0; i0 < nc; ++i0) {
                    int i = i0 + i1*nc;
                    sumf += s[i] * c[i];
                }
                x[i1] = sumf;
            }
        }

        // copy the state out of it
        for (int i1 = 0; i1 < ir; ++i1) {
            for (int i0 = 0; i0 < nc - 1; ++i0) {
                s0[i0 + i1*(nc - 1)] = s[1 + i0 + i1*nc];
            }
        }
    }
}

static void ssm_conv_f32_cuda(
    const float * src0, const float * src1, const float * src2,
    const int src0_nb1, const int src0_nb2,
    const int src1_nb0, const int src1_nb1, const int src1_nb2,
    const int src2_nb1,
    float * dst,
    const int dst_nb0, const int dst_nb1, const int dst_nb2,
    const int nc, const int nr, const int n_t, const int n_s,
    cudaStream_t stream) {

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const int nblocks = 1; // TODO
    const int shmem_size = nc * (nr + WARP_SIZE - 1) * sizeof(float); // TODO

    ssm_conv_f32<WARP_SIZE><<<nblocks, block_dims, shmem_size, stream>>>(
        src0, src1, src2,
        src0_nb1, src0_nb2,
        src1_nb0, src1_nb1, src1_nb2,
        src2_nb1,
        dst,
        dst_nb0, dst_nb1, dst_nb2,
        nc, nr, n_t, n_s);
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // conv_state
    const struct ggml_tensor * src1 = dst->src[1]; // x
    const struct ggml_tensor * src2 = dst->src[2]; // conv1d.weight

    const int nc  = src2->ne[0]; // d_conv
    const int nr  = src0->ne[1]; // d_inner
    const int n_t = src1->ne[1]; // tokens per sequence
    const int n_s = src0->ne[2]; // number of sequences in the batch

    GGML_ASSERT(ggml_are_same_shape(src1, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    const float * src2_d = (const float *)src2->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ssm_conv_f32_cuda(src0_d, src1_d, src2_d,
        src0->nb[1], src0->nb[2],
        src1->nb[0], src1->nb[1], src1->nb[2],
        src2->nb[1],
        dst_d,
        dst->nb[0], dst->nb[1], dst->nb[2],
        nc, nr, n_t, n_s,
        stream);
}
