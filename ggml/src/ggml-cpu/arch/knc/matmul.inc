/* SPDX-License-Identifier: MIT */
static bool ggml_knc_mq_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * w = dst->src[0];
    const struct ggml_tensor * x = dst->src[1];
    if (params->use_ref || !ggml_knc_mq_enabled() || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
        !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(dst)) {
        return false;
    }
    int bits;
    switch (w->type) {
        case GGML_TYPE_Q3_K:
            bits = 3;
            break;
        case GGML_TYPE_Q4_K:
            bits = 4;
            break;
        case GGML_TYPE_Q5_K:
            bits = 5;
            break;
        case GGML_TYPE_Q6_K:
            bits = 6;
            break;
        default:
            return false;
    }
    const int64_t k = w->ne[0], rows = w->ne[1], cols = x->ne[1];
    if (k <= 0 || k > INT_MAX || k % QK_K || rows <= 0 || cols <= 0 || x->ne[0] != k || dst->ne[0] != rows ||
        dst->ne[1] != cols) {
        return false;
    }
    const size_t blocks        = (size_t) k / QK_K;
    const size_t packed_stride = blocks * sizeof(block_q8_K);
    if ((uint64_t) cols > SIZE_MAX / packed_stride || !params->wdata || params->wsize < (size_t) cols * packed_stride ||
        ((uintptr_t) x->data % _Alignof(float)) || ((uintptr_t) dst->data % _Alignof(float)) ||
        ((uintptr_t) params->wdata % _Alignof(block_q8_K))) {
        return false;
    }
    const int ith = params->ith, nth = params->nth;
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth);
    char *       packed = (char *) params->wdata;
    const size_t b0     = blocks * (size_t) ith / (size_t) nth;
    const size_t b1     = blocks * (size_t) (ith + 1) / (size_t) nth;
    for (int64_t col = 0; col < cols && b1 > b0; ++col) {
        quantize_row_q8_K((const float *) ((const char *) x->data + col * x->nb[1]) + b0 * QK_K,
                          packed + (size_t) col * packed_stride + b0 * sizeof(block_q8_K), (b1 - b0) * QK_K);
    }
    ggml_barrier(params->threadpool);

    /* Partition rows before tiling so a larger tile does not idle workers. */
    const int64_t             first         = rows / nth * ith + (rows % nth * ith) / nth;
    const int64_t             last          = rows / nth * (ith + 1) + (rows % nth * (ith + 1)) / nth;
    const int                 tile_rows     = ggml_knc_mq_rows();
    const ggml_knc_mq_tile_fn kernel        = ggml_knc_mq_kernel(bits);
    uint64_t                  staged_blocks = 0;
    for (int64_t col = 0; col < cols; ++col) {
        float * out = (float *) ((char *) dst->data + col * dst->nb[1]);
        for (int64_t row = first; row < last; row += tile_rows) {
            const size_t nr = (size_t) MIN((int64_t) tile_rows, last - row);
            kernel(blocks, nr, (const uint8_t *) w->data + row * w->nb[1], w->nb[1],
                   (const uint8_t *) packed + (size_t) col * packed_stride, out + row);
            staged_blocks += blocks;
        }
    }
    ggml_knc_mq_count(bits, (uint64_t) (last - first) * cols, staged_blocks, ith == 0);
    return true;
}
