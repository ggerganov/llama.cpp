#ifndef HTP_GET_ROWS_OPS_H
#define HTP_GET_ROWS_OPS_H

#include <stdbool.h>
#include <string.h>

#include "hex-fastdiv.h"
#include "matmul-ops.h"

enum htp_get_rows_kernel_type {
    HTP_GET_ROWS_KERNEL_SAMETYPE = 0,
    HTP_GET_ROWS_KERNEL_TILED,
    HTP_GET_ROWS_KERNEL_FLAT,
};

struct htp_get_rows_kernel_params {
    int32_t  n_threads;
    int32_t  kernel_type;
    int32_t  total_tasks;
    int32_t  tasks_per_thread;
    int32_t  vtcm_size;

    // Fastdiv helpers
    struct fastdiv_values div_ne10;
    struct fastdiv_values div_ne10_ne11;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne03;
};

struct htp_get_rows_vtcm_layout {
    size_t total_bytes;
    size_t off_src1;
    size_t src1_bytes;
    size_t off_src0;
    size_t off_dst;

    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;

    size_t src0_spad_half_size;
    size_t dst_spad_half_size;
};

static inline void htp_get_rows_vtcm_layout_build(
    struct htp_get_rows_vtcm_layout * vtcm_layout,
    int kernel_type,
    int type,
    uint32_t ne00,
    uint32_t total_tasks,
    int idx_type,
    uint32_t n_threads) {

    memset(vtcm_layout, 0, sizeof(*vtcm_layout));

    const size_t elem_size = (idx_type == HTP_TYPE_I64) ? sizeof(int64_t) : sizeof(int32_t);
    const size_t src1_raw_bytes = total_tasks * elem_size;
    const size_t src1_bytes = ((src1_raw_bytes + 127) & ~127) + 128;

    vtcm_layout->off_src1   = 0;
    vtcm_layout->src1_bytes = src1_bytes;

    const size_t spad_base = (src1_bytes + 255) & ~255;

    if (kernel_type == HTP_GET_ROWS_KERNEL_SAMETYPE) {
        vtcm_layout->off_src0    = spad_base;
        vtcm_layout->off_dst     = spad_base;
        vtcm_layout->total_bytes = spad_base;
        return;
    }

    if (kernel_type == HTP_GET_ROWS_KERNEL_TILED) {
        const size_t tile_size   = htp_mm_get_weight_tile_size(type);
        const size_t tile_stride = (tile_size + 127) & ~127;
        const uint32_t n_k_tiles = ne00 / HTP_MM_HMX_TILE_N_COLS;
        const size_t row_tiles_size = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        vtcm_layout->src0_spad_half_size = (row_tiles_size + 255) & ~255;
        vtcm_layout->dst_spad_half_size  = (ne00 * sizeof(float) + 255) & ~255;
        vtcm_layout->src0_bytes_per_thread = 2 * vtcm_layout->src0_spad_half_size;
        vtcm_layout->dst_bytes_per_thread  = 2 * vtcm_layout->dst_spad_half_size;
        vtcm_layout->off_src0 = spad_base;
        vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_bytes_per_thread * n_threads;
        vtcm_layout->total_bytes = vtcm_layout->off_dst + vtcm_layout->dst_bytes_per_thread * n_threads;
        return;
    }

    const uint32_t src0_row_size = htp_tensor_get_row_size(type, ne00);
    size_t src0_row_size_aligned = (src0_row_size + 255) & ~255;
    size_t dst_row_size_aligned  = (ne00 * sizeof(float) + 255) & ~255;

    vtcm_layout->src0_spad_half_size = src0_row_size_aligned;
    vtcm_layout->dst_spad_half_size  = dst_row_size_aligned;

    vtcm_layout->src0_bytes_per_thread = src0_row_size_aligned * 2;
    vtcm_layout->dst_bytes_per_thread  = dst_row_size_aligned * 2;

    vtcm_layout->off_src0 = spad_base;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_bytes_per_thread * n_threads;
    vtcm_layout->total_bytes = vtcm_layout->off_dst + vtcm_layout->dst_bytes_per_thread * n_threads;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_get_rows_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_GET_ROWS_OPS_H
