#include "ssm_conv.cuh"

template <int block_size>
static __global__ void ssm_conv_f32(
    const float * src0, const float * src1, const float * src2, const float * src3,
    const int src0_ne0, const int src0_nb1, const int src0_nb2,
    const int src1_nb0, const int src1_nb1,
    const int src2_nb1, const int src2_nb2,
    const int src3_nb1,
    float * dst,
    const int nc, const int nr, const int n_t, const int n_kv) {

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

    if (n_kv > 1) {
        // multiple sequences means it's hard to know when it's the first time a state is read,
        // so copy them all over to the destination, just to be sure.
        for (int i3 = 0; i3 < n_kv; ++i3) {
            float * s0 = (float *) ((char *) src0 + ir0*src0_nb1 + i3*src0_nb2);
            float * s  = (float *) ((char *)  dst + ir0*src2_nb1 + i3*src2_nb2 + nr*n_t*sizeof(float));
            // can't use memcpy because of d_conv vs d_conv - 1
            for (int i1 = 0; i1 < ir; ++i1) {
                for (int i0 = 0; i0 < nc - 1; ++i0) {
                    // copy s0 to last (d_conv - 1) columns of s
                    s[1 + i0 + i1*nc] = s0[i0 + i1*(nc - 1)];
                }
            }
        }
    }

    for (int i2 = 0; i2 < n_t; ++i2) {
        int32_t * sq = (int32_t *) ((char *) src3 +  i2*src3_nb1); // {n_kv, n_tokens}
        float *   x  = (float *)   ((char *)  dst + ir0*sizeof(float) + i2*(nr*sizeof(float))); // {d_inner, n_tokens}
        float *   s  = (float *)   ((char *)  dst + ir0*src2_nb1 + sq[0]*src2_nb2 + nr*n_t*sizeof(float)); // {d_conv, d_inner, n_kv}
        float *   s0; // {d_conv - 1, d_inner, n_kv}
        float *   x0 = (float *)   ((char *) src1 + ir0*src1_nb0 + i2*src1_nb1); // {d_inner, n_tokens}
        float *   c  = (float *)   ((char *) src2 + ir0*src2_nb1); // {d_conv, d_inner}
        int ne0s0;

        // avoid needing to copy the state for the first token
        if (i2 == 0) {
            s0 = (float *) ((char *) src0 + ir0*src0_nb1 + sq[0]*src0_nb2); // {d_conv - 1, d_inner, n_kv}
            ne0s0 = src0_ne0;
        } else {
            // the source is the last (d_conv - 1) columns of the destination
            s0 = s + 1;
            ne0s0 = nc;
        }

        // d_inner
        for (int i1 = 0; i1 < ir; ++i1) {
            // shift state left
            for (int i0 = 0; i0 < nc - 1; ++i0) {
                s[i0 + i1*nc] = s0[i0 + i1*ne0s0];
            }
            // insert x on the last column
            s[(nc - 1) + i1*nc] = x0[i1];
        }

        // handle copies when there are multiple output states
        for (int i3 = 1; i3 < n_kv; ++i3) {
            int32_t seq = sq[i3];
            if (0 <= seq && seq < n_kv) {
                float * s1 = s + (seq - sq[0])*nc*nr;

                //memcpy(s1, s, nc*ir*sizeof(float));
                for (int i4 = 0; i4 < nc*ir; i4++) {
                    s1[i4] = s[i4];
                }
            } else {
                // stop at negative or too big seq_ids
                break;
            }
        }

        // it seems a little faster when this is separate from the state shift
        for (int i1 = 0; i1 < ir; ++i1) {
            // rowwise dot product
            float sumf = 0.0f;
            for (int i0 = 0; i0 < nc; ++i0) {
                int i = i0 + i1*nc;
                sumf += s[i] * c[i];
            }
            x[i1] = sumf;
        }
    }
}

static void ssm_conv_f32_cuda(
    const float * src0, const float * src1, const float * src2, const float * src3,
    const int src0_ne0, const int src0_nb1, const int src0_nb2,
    const int src1_nb0, const int src1_nb1,
    const int src2_nb1, const int src2_nb2,
    const int src3_nb1,
    float * dst,
    const int nc, const int nr, const int n_t, const int n_kv, cudaStream_t stream) {

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const int nblocks = 1; // TODO

    ssm_conv_f32<WARP_SIZE><<<nblocks, block_dims, 0, stream>>>(
        src0, src1, src2, src3,
        src0_ne0, src0_nb1, src0_nb2,
        src1_nb0, src1_nb1,
        src2_nb1, src2_nb2,
        src3_nb1,
        dst,
        nc, nr, n_t, n_kv);
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // conv_state
    const struct ggml_tensor * src1 = dst->src[1]; // x
    const struct ggml_tensor * src2 = dst->src[2]; // conv1d.weight
    const struct ggml_tensor * src3 = dst->src[3]; // state_seq

    const int nc   = src2->ne[0]; // d_conv
    const int nr   = src0->ne[1]; // d_inner
    const int n_t  = src1->ne[1]; // n_tokens
    const int n_kv = src0->ne[2]; // max number of sequences in the batch

    GGML_ASSERT((nr*n_t) + (nc*nr*n_kv) == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(int32_t));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));
    // for use with the destination state offset between sequences
    GGML_ASSERT(src2->nb[2] == src2->ne[1]*src2->ne[0]*sizeof(float));

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    const float * src2_d = (const float *)src2->data;
    const float * src3_d = (const float *)src3->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ssm_conv_f32_cuda(src0_d, src1_d, src2_d, src3_d,
        src0->ne[0], src0->nb[1], src0->nb[2],
        src1->nb[0], src1->nb[1],
        src2->nb[1], src2->nb[2],
        src3->nb[1],
        dst_d, nc, nr, n_t, n_kv, stream);
}
