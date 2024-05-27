#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-vec-f16.cuh"

void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * KQV = dst;
    ggml_tensor * Q   = dst->src[0];

    const int32_t precision = KQV->op_params[2];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    constexpr int cols_per_block  = 1;
    constexpr int parallel_blocks = 4;
    switch (Q->ne[0]) {
        case  64: {
            constexpr int      D = 64;
            constexpr int nwarps = D/WARP_SIZE;
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f16<
                D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_f16<half, D>, false, dequantize_1_f16<half>>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        case 128: {
            constexpr int      D = 128;
            constexpr int nwarps = D/WARP_SIZE;
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f16<
                D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_f16<half, D>, false, dequantize_1_f16<half>>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        case 256: {
            constexpr int      D = 256;
            constexpr int nwarps = D/WARP_SIZE;
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f16<
                D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_f16<half, D>, false, dequantize_1_f16<half>>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        default:
            GGML_ASSERT(false);
            break;
    }
}

template <int cols_per_block, int parallel_blocks>
void launch_fattn_vec_f16_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASE(f16,  64, f16,  q4_0)
    FATTN_VEC_CASE(f16,  64, f16,  q4_1)
    FATTN_VEC_CASE(f16,  64, f16,  q5_0)
    FATTN_VEC_CASE(f16,  64, f16,  q5_1)
    FATTN_VEC_CASE(f16,  64, f16,  q8_0)
    FATTN_VEC_CASE(f16,  64, f16,  f16)

    FATTN_VEC_CASE(f16, 128, q4_0, q4_0)
    FATTN_VEC_CASE(f16, 128, q4_0, q4_1)
    FATTN_VEC_CASE(f16, 128, q4_0, q5_0)
    FATTN_VEC_CASE(f16, 128, q4_0, q5_1)
    FATTN_VEC_CASE(f16, 128, q4_0, q8_0)
    FATTN_VEC_CASE(f16, 128, q4_0, f16)

    FATTN_VEC_CASE(f16, 128, q4_1, q4_0)
    FATTN_VEC_CASE(f16, 128, q4_1, q4_1)
    FATTN_VEC_CASE(f16, 128, q4_1, q5_0)
    FATTN_VEC_CASE(f16, 128, q4_1, q5_1)
    FATTN_VEC_CASE(f16, 128, q4_1, q8_0)
    FATTN_VEC_CASE(f16, 128, q4_1, f16)

    FATTN_VEC_CASE(f16, 128, q5_0, q4_0)
    FATTN_VEC_CASE(f16, 128, q5_0, q4_1)
    FATTN_VEC_CASE(f16, 128, q5_0, q5_0)
    FATTN_VEC_CASE(f16, 128, q5_0, q5_1)
    FATTN_VEC_CASE(f16, 128, q5_0, q8_0)
    FATTN_VEC_CASE(f16, 128, q5_0, f16)

    FATTN_VEC_CASE(f16, 128, q5_1, q4_0)
    FATTN_VEC_CASE(f16, 128, q5_1, q4_1)
    FATTN_VEC_CASE(f16, 128, q5_1, q5_0)
    FATTN_VEC_CASE(f16, 128, q5_1, q5_1)
    FATTN_VEC_CASE(f16, 128, q5_1, q8_0)
    FATTN_VEC_CASE(f16, 128, q5_1, f16)

    FATTN_VEC_CASE(f16, 128, q8_0, q4_0)
    FATTN_VEC_CASE(f16, 128, q8_0, q4_1)
    FATTN_VEC_CASE(f16, 128, q8_0, q5_0)
    FATTN_VEC_CASE(f16, 128, q8_0, q5_1)
    FATTN_VEC_CASE(f16, 128, q8_0, q8_0)
    FATTN_VEC_CASE(f16, 128, q8_0, f16)

    FATTN_VEC_CASE(f16, 128, f16,  q4_0)
    FATTN_VEC_CASE(f16, 128, f16,  q4_1)
    FATTN_VEC_CASE(f16, 128, f16,  q5_0)
    FATTN_VEC_CASE(f16, 128, f16,  q5_1)
    FATTN_VEC_CASE(f16, 128, f16,  q8_0)
    FATTN_VEC_CASE(f16, 128, f16,  f16)
#else
    FATTN_VEC_CASE(f16, 128, q4_0, q4_0)
    FATTN_VEC_CASE(f16, 128, q8_0, q8_0)
    FATTN_VEC_CASE(f16, 128, f16,  f16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    on_no_fattn_vec_case(Q->ne[0]);
}

void ggml_cuda_flash_attn_ext_vec_f16_no_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const int32_t precision = KQV->op_params[2];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block  = 1;
        constexpr int parallel_blocks = 4;
        launch_fattn_vec_f16_64_128<cols_per_block, parallel_blocks>(ctx, dst);
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block  = 2;
        constexpr int parallel_blocks = 4;
        launch_fattn_vec_f16_64_128<cols_per_block, parallel_blocks>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block  = 4;
        constexpr int parallel_blocks = 4;
        launch_fattn_vec_f16_64_128<cols_per_block, parallel_blocks>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 8) {
        constexpr int cols_per_block  = 8;
        constexpr int parallel_blocks = 4;
        launch_fattn_vec_f16_64_128<cols_per_block, parallel_blocks>(ctx, dst);
        return;
    }

    constexpr int cols_per_block  = 8;
    constexpr int parallel_blocks = 1;
    launch_fattn_vec_f16_64_128<cols_per_block, parallel_blocks>(ctx, dst);
}
