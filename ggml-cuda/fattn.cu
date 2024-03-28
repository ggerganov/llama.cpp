#include "fattn.cuh"

#include <mma.h>

static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
#pragma unroll
   for (int mask = 16; mask > 0; mask >>= 1) {
       a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, mask, 32));
   }
   return a;
#else
   GGML_UNUSED(a);
   NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL
}

static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL && CUDART_VERSION >= CUDART_HMAX
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = __hmax2(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
#else
    GGML_UNUSED(x);
    NO_DEVICE_CODE;
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= CC_PASCAL && CUDART_VERSION >= CUDART_HMAX
}

#if __CUDA_ARCH__ >= CC_VOLTA
typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    16, 16, 16, half, nvcuda::wmma::row_major> half16x16_a;
typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    16, 16, 16, half, nvcuda::wmma::row_major> half16x16_b;
typedef nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    16, 16, 16, half, nvcuda::wmma::col_major> half16x16_bT;
typedef nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>                          half16x16_acc;
#endif

// based on metal version
template<int D, int Q, int C> // D head size, Q queries per block, C cache items per block
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        float * __restrict__ dst,
        float scale,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        int ne31,
        int nb31,
        int nb01,
        int nb02,
        int nb03,
        int nb11,
        int nb12,
        int nb13,
        int ne0,
        int ne1,
        int ne2,
        int ne3) {
#if __CUDA_ARCH__ >= CC_VOLTA
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    const int num_warps = blockDim.y; // number of warps
    const int iq3 = blockIdx.z;
    const int iq2 = blockIdx.y;
    const int iq1 = blockIdx.x * Q;

    const int D16 = D/16;
    const int Q16 = Q/16;
    const int C16 = C/16;

    const int NW  = WARP_SIZE;
    const int SH  = (C + Q); // shared memory per simdgroup in (half)

    const int T  = D + num_warps*SH; // shared memory size per query in (half)
    const int T2 = T/2;              // shared memory size per query in (half2)
    const int C2 = C/2;
    const int D2 = D/2;

    extern __shared__  half __flash_attn_f16_shmem[];
    // pq
    half  * sq  = (half  *) (__flash_attn_f16_shmem +              0*D); // holds the query data
    half2 * sq2 = (half2 *) (__flash_attn_f16_shmem +              0*D); // same as above but in half2
    half  * ss  = (half  *) (__flash_attn_f16_shmem + warp_id*SH + 1*D); // scratch buffer for attention and diagonal matrix
    half2 * ss2 = (half2 *) (__flash_attn_f16_shmem + warp_id*SH + 1*D); // same as above but in half2

    half16x16_acc zr;
    half16x16_acc lo[Q16][D16];

    // load heads from Q to shared memory
#pragma unroll
    for (int j0 = 0; j0 < Q; j0 += num_warps) {
        const int j = j0 + warp_id;
        if (j >= Q) {
            break;
        }

        const float2 * q2 = (const float2 *) (q + ((iq1 + j)*nb01 + iq2*nb02 + iq3*nb03));

#pragma unroll
        for (int i0 = 0; i0 < D2; i0 += NW) {
            const int i = i0 + lane_id;
            if (i >= D2) {
                break;
            }

            if (iq1 + j < ne01) {
                sq2[j*T2 + i] = __float22half2_rn(q2[i]);
            } else {
                sq2[j*T2 + i] = make_half2(0.0, 0.0);
            }
        }
    }

    nvcuda::wmma::fill_fragment(zr, 0.0);

    // zero out lo
    for (int j = 0; j < Q16; ++j) {
        for (int i = 0; i < D16; ++i) {
            nvcuda::wmma::fill_fragment(lo[j][i], 0.0);
        }
    }

    // zero out shared memory SH
    for (int j = 0; j < Q; ++j) {
        for (int i0 = 0; i0 < SH; i0 += NW) {
            const int i = i0 + lane_id;
            if (i >= SH) {
                break;
            }

            ss[j*T + i] = 0.0;
        }
    }

    __syncthreads();

    {
        half S = __float2half(0.0f);
        half M[Q];

        for (int i = 0; i < Q; ++i) {
            M[i] = CUDART_MIN_DENORM_FP16;
        }

        // assume K and V are same shape
        const int ne22 = ne12;
        const int ne23 = ne13;

        const int nb21 = nb11;
        const int nb22 = nb12;
        const int nb23 = nb13;

        // broadcast
        const int rk2 = ne02/ne12;
        const int rk3 = ne03/ne13;

        const int rv2 = ne02/ne22;
        const int rv3 = ne03/ne23;

        // k indices
        const int ik2 = iq2 / rk2;
        const int ik3 = iq3 / rk3;

        // v indices
        const int iv2 = iq2 / rv2;
        const int iv3 = iq3 / rv3;

        // load the queries from shared memory into local memory
        //half16x16_a mq[Q16][D16];
        //for (int j = 0; j < Q16; ++j) {
        //    for (int i = 0; i < D16; ++i) {
        //        nvcuda::wmma::load_matrix_sync(mq[j][i], sq + 16*j*T + i*16, T);
        //    }
        //}

        // pointer to the mask
        const half * mp = mask ? (const half *) (mask + iq1*nb31) : nullptr;

        // prepare diagonal scale matrix
        half16x16_b mscale;
        for (int i = 0; i < 16; ++i) {
            ss[i*T + i] = __float2half(scale);
        }
        nvcuda::wmma::load_matrix_sync(mscale, ss, T);

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < ne11; ic0 += C*num_warps) {
            const int ic = ic0 + warp_id*C;
            if (ic >= ne11) {
                break;
            }

            // Q*K^T
            {
#pragma unroll
                for (int cc = 0; cc < C16; ++cc) {
                    half16x16_acc mqk[Q16];
                    for (int j = 0; j < Q16; ++j) {
                        nvcuda::wmma::fill_fragment(mqk[j], 0);
                    }

                    const half * pk = (const half *) ((const char *) k + ((ic + 16*cc)*nb11 + ik2*nb12 + ik3*nb13));

                    for (int i = 0; i < D16; ++i) {
                        half16x16_bT mk; // transposed key
                        nvcuda::wmma::load_matrix_sync(mk, pk + i*16, nb11/sizeof(half));

                        for (int j = 0; j < Q16; ++j) {
                            half16x16_a mq;
                            nvcuda::wmma::load_matrix_sync(mq, sq + 16*j*T + i*16, T);
                            nvcuda::wmma::mma_sync(mqk[j], mq, mk, mqk[j]);
                        }
                    }

                    // mqk = mqk*scale + mask
                    for (int j = 0; j < Q16; ++j) {
                        half16x16_a mqka;
                        half16x16_acc mm;

                        if (mp) {
                            nvcuda::wmma::load_matrix_sync(mm, mp + 16*j*(nb31/sizeof(half)) + ic + 16*cc, nb31/sizeof(half), nvcuda::wmma::mem_row_major);
                        }

                        // convert accumulator to matrix_a
                        nvcuda::wmma::store_matrix_sync(      ss + 16*j*T + 16*cc, mqk[j], T, nvcuda::wmma::mem_row_major);
                        nvcuda::wmma::load_matrix_sync (mqka, ss + 16*j*T + 16*cc, T);

                        nvcuda::wmma::mma_sync(mqk[j], mqka, mscale, mp ? mm : zr);
                        nvcuda::wmma::store_matrix_sync(ss + 16*j*T + 16*cc, mqk[j], T, nvcuda::wmma::mem_row_major);
                    }
                }
            }

            // used to detect blocks full of -INF
            half2 smax = make_half2(-INFINITY, -INFINITY);

            // online softmax
            for (int j = 0; j < Q; ++j) {
                const half m = M[j];

                for (int p0 = 0; p0 < C2; p0 += NW) {
                    const int p = p0 + lane_id;

                    const half2 s = ss2[j*T2 + p];

                    smax = __hmax2(smax, s);
                    M[j] = __hmax(M[j], __hmax(s.x, s.y));
                }

                M[j] = warp_reduce_max(M[j]);

                // local sum
                half2 ls = make_half2(0.0f, 0.0f);
                half2 M2 = make_half2(M[j], M[j]);

                for (int p0 = 0; p0 < C2; p0 += NW) {
                    const int p = p0 + lane_id;

                    const half2 s = ss2[j*T2 + p];

                    const half2 vs = h2exp(s - M2);

                    ls += vs;

                    // the P matrix from the paper (Q rows, C columns)
                    ss2[j*T2 + p] = vs;
                }

                ls = warp_reduce_sum(ls);

                const half ms = hexp(m - M[j]);

                // create a QxQ diagonal matrix for rescaling the output
                if (lane_id == j) {
                    ss[j*T + C + j] = ms;

                    S = S*ms + ls.x + ls.y;
                }
            }

            smax = warp_reduce_max(smax);

            // skip -INF blocks
            if (__hisinf(smax.x) == -1 && __hisinf(smax.y) == -1) {
                continue;
            }

            // O = diag(ms)*O
            for (int j = 0; j < Q16; ++j) {
                half16x16_a mm;
                half16x16_b lob;

                nvcuda::wmma::load_matrix_sync(mm, ss + 16*j*T + C + 16*j, T);

                for (int i = 0; i < D16; ++i) {
                    // convert accumulator to matrix_b
                    nvcuda::wmma::store_matrix_sync(     ss + 16*j*T + C + 16*j, lo[j][i], T, nvcuda::wmma::mem_row_major);
                    nvcuda::wmma::load_matrix_sync (lob, ss + 16*j*T + C + 16*j, T);

                    nvcuda::wmma::mma_sync(lo[j][i], mm, lob, zr);
                }
            }

            // restore zeros
            for (int j = 0; j < Q16; ++j) {
                nvcuda::wmma::store_matrix_sync(ss + 16*j*T + C + 16*j, zr, T, nvcuda::wmma::mem_row_major);
            }

            // O = O + (Q*K^T)*V
            {
                for (int cc = 0; cc < C16; ++cc) {
                    const half * pv = (const half *) ((const char *) v + ((ic + 16*cc)*nb21 + iv2*nb22 + iv3*nb23));

                    for (int j = 0; j < Q16; ++j) {
                        half16x16_a ms;
                        nvcuda::wmma::load_matrix_sync(ms, ss + 16*j*T + 16*cc, T);
                        for (int i = 0; i < D16; ++i) {
                            half16x16_b mv;
                            nvcuda::wmma::load_matrix_sync(mv, pv + i*16, nb21/sizeof(half));
                            nvcuda::wmma::mma_sync(lo[j][i], ms, mv, lo[j][i]);
                        }
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (lane_id < Q) {
            ss[lane_id*T + 0] = S;
            ss[lane_id*T + 1] = M[lane_id];
        }
    }

    // reduce the warps sequentially
    for (int sg = 1; sg < num_warps; ++sg) {
        __syncthreads();

        // each simdgroup stores its output to shared memory, reusing sq
        if (warp_id == sg) {
            for (int j = 0; j < Q16; ++j) {
                for (int i = 0; i < D16; ++i) {
                    nvcuda::wmma::store_matrix_sync(sq + 16*j*T + i*16, lo[j][i], T, nvcuda::wmma::mem_row_major);
                }
            }
        }

        __syncthreads();

        // the first simdgroup accumulates the results from the other simdgroups
        if (warp_id == 0) {
            for (int j = lane_id; j < Q; j += NW) {
                const half S0 = ss[j*T +         0];
                const half S1 = ss[j*T + sg*SH + 0];

                const half M0 = ss[j*T +         1];
                const half M1 = ss[j*T + sg*SH + 1];

                const half M = __hmax(M0, M1);

                const half ms0 = hexp(M0 - M);
                const half ms1 = hexp(M1 - M);

                const half S = S0*ms0 + S1*ms1;

                ss[j*T + 0] = S;
                ss[j*T + 1] = M;

                ss[j*T + C + j        ] = ms0;
                ss[j*T + C + j + sg*SH] = ms1;
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            for (int j = 0; j < Q16; ++j) {
                half16x16_a ms0;
                half16x16_a ms1;
                half16x16_b t;
                half16x16_acc t2;

                nvcuda::wmma::load_matrix_sync(ms0, ss + 16*j*T + C + 16*j,         T);
                nvcuda::wmma::load_matrix_sync(ms1, ss + 16*j*T + C + 16*j + sg*SH, T);

                for (int i = 0; i < D16; ++i) {
                    nvcuda::wmma::load_matrix_sync(t, sq + 16*j*T + i*16, T);
                    nvcuda::wmma::mma_sync(t2, ms1, t, zr);

                    // convert accumulator to matrix_b
                    nvcuda::wmma::store_matrix_sync(   sq + 16*j*T + i*16, lo[j][i], T, nvcuda::wmma::mem_row_major);
                    nvcuda::wmma::load_matrix_sync (t, sq + 16*j*T + i*16, T);

                    nvcuda::wmma::mma_sync(lo[j][i], ms0, t, t2);
                }
            }
        }
    }

    // store result to shared memory (reuse sq)
    if (warp_id == 0) {
        for (int j = 0; j < Q16; ++j) {
            for (int i = 0; i < D16; ++i) {
                nvcuda::wmma::store_matrix_sync(sq + 16*j*T + i*16, lo[j][i], T, nvcuda::wmma::mem_row_major);
            }
        }
    }

    // final rescale with 1/S and store to global memory
    if (warp_id == 0) {
        for (int j = 0; j < Q && iq1 + j < ne01; ++j) {
            const half S = ss[j*T + 0];

            for (int i0 = 0; i0 < D; i0 += NW) {
                const int i = i0 + lane_id;
                if (i >= D) {
                    break;
                }

                dst[(iq3*ne2*ne1 + iq2 + (iq1 + j)*ne1)*D + i] = __half2float(sq[j*T + i] / S);
            }
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_tensor * mask = dst->src[3];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(K->type == GGML_TYPE_F16);
    GGML_ASSERT(V->type == GGML_TYPE_F16);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!mask || mask->ne[1] >= GGML_PAD(Q->ne[1], 16) &&
                                "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    ggml_cuda_set_device(ctx.device);

    const cudaStream_t main_stream = ctx.stream();

    float scale;
    memcpy(&scale, KQV->op_params, sizeof(float));

#define NQPB 16
#define NCPW 128

    const int nqpb = NQPB; // queries per block
    const int ncpw = NCPW; // cache values per warp (does not work for other values)

    GGML_ASSERT(NQPB <= 32);

    const int nwarps_max = 8; // TODO: we don't want to launch too much warps. how much is too much?
                              // TODO: produces wrong results for nwarps > 8 (RTX 2060) - not sure why
    const int nwarps = Q->ne[1] <= nqpb ? std::max(2, std::min((int) K->ne[1]/ncpw, nwarps_max)) + 1 : 1;

    dim3 blocks_num((Q->ne[1] + nqpb - 1) / nqpb, Q->ne[2], Q->ne[3]);
    dim3 block_dim(32, nwarps, 1);

    const size_t shmem = nqpb*(Q->ne[0] + nwarps*(ncpw + nqpb))*(sizeof(float)/2);
    //printf("nwarps: %d, shm: %zu\n", nwarps, shmem);

    switch (Q->ne[0]) {
        case 64:
            {
                flash_attn_ext_f16<64, NQPB, NCPW>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? (const char *) mask->data : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
            } break;
        case 80:
            {
                flash_attn_ext_f16<80, NQPB, NCPW>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? (const char *) mask->data : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
            } break;
        case 96:
            {
                flash_attn_ext_f16<96, NQPB, NCPW>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? (const char *) mask->data : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
            } break;
        case 112:
            {
                flash_attn_ext_f16<112, NQPB, NCPW>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? (const char *) mask->data : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
            } break;
        case 128:
            {
                //const size_t shmem_max = 96*1024;
                //cudaFuncSetAttribute(flash_attn_ext_f16<128, NQPB, NCPW>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_max);

                flash_attn_ext_f16<128, NQPB, NCPW>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? (const char *) mask->data : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
            } break;
        case 256:
            {
                const size_t shmem_max = 64*1024;
                cudaFuncSetAttribute(flash_attn_ext_f16<256, NQPB, NCPW>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_max);

                flash_attn_ext_f16<256, NQPB, NCPW>
                    <<<blocks_num, block_dim, shmem, main_stream>>> (
                            (const char *) Q->data, // Query
                            (const char *) K->data, // Key
                            (const char *) V->data, // Value
                            mask ? (const char *) mask->data : nullptr, // Mask
                            (float *) KQV->data, // dst
                            scale,
                            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
                            K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                            mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
                            Q->nb[1], Q->nb[2], Q->nb[3],
                            K->nb[1], K->nb[2], K->nb[3],
                            KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
                            );
            } break;
        default:
            break;
    }

    CUDA_CHECK(cudaGetLastError());
}
