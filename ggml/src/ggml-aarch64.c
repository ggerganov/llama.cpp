#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-aarch64.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include <assert.h>

#define UNUSED GGML_UNUSED

static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_interleave, unsigned int xor_mask) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    for (int i = 0; i < QK4_0 * 2; i++) {
        int src_offset = (i / (4 * blck_size_interleave)) * blck_size_interleave;
        int src_id = (i % (4 * blck_size_interleave)) / blck_size_interleave;
        src_offset += (i % blck_size_interleave);

        out.qs[i] = in[src_id].qs[src_offset] ^ xor_mask;
    }

    return out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
static block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, unsigned int blck_size_interleave, unsigned int xor_mask) {
    block_q4_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    for (int i = 0; i < QK4_0 * 4; i++) {
        int src_offset = (i / (8 * blck_size_interleave)) * blck_size_interleave;
        int src_id = (i % (8 * blck_size_interleave)) / blck_size_interleave;
        src_offset += (i % blck_size_interleave);

        out.qs[i] = in[src_id].qs[src_offset] ^ xor_mask;
    }

    return out;
}

static size_t quantize_q4_0_nr_bl(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, int nrows_interleaved, int blck_size_interleave) {
    assert(n_per_row % QK4_0 == 0);
    const int nb = n_per_row / QK4_0;

    void * out_ptr = NULL;
    if (nrows_interleaved == 8) {
        out_ptr = (block_q4_0x8 *) dst;
    }
    else if (nrows_interleaved == 4) {
        out_ptr = (block_q4_0x4 *) dst;
    }
    assert(nrows_interleaved <= 8);
    block_q4_0 dst_tmp[8];

    for (int b = 0; b < (nrow * n_per_row); b += nrows_interleaved * n_per_row) {

        for (int64_t x = 0; x < nb; x++) {

            for (int i  = 0; i < nrows_interleaved; i++ ) {
                quantize_row_q4_0_ref(src + b + i * n_per_row + x * QK4_0, (block_q4_0 *) dst_tmp + i, QK4_0);
            }

            if (nrows_interleaved == 8) {
                *(block_q4_0x8 *) out_ptr = make_block_q4_0x8(dst_tmp, blck_size_interleave, 0x88);
                out_ptr = (block_q4_0x8 *) out_ptr + 1;
            }
            else if (nrows_interleaved == 4) {
                *(block_q4_0x4 *) out_ptr = make_block_q4_0x4(dst_tmp, blck_size_interleave, 0x88);
                out_ptr = (block_q4_0x4 *) out_ptr + 1;
            }
        }
    }

    return ((nrow * n_per_row) / QK4_0 * sizeof(block_q4_0));
}

size_t quantize_q4_0_4x4(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    UNUSED(quant_weights);
    return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 4, 4);
}

size_t quantize_q4_0_4x8(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    UNUSED(quant_weights);
    return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 4, 8);
}

size_t quantize_q4_0_8x8(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    UNUSED(quant_weights);
    return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 8, 8);
}
