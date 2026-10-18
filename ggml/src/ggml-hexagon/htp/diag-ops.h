#ifndef HTP_DIAG_OPS_H
#define HTP_DIAG_OPS_H

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"
#include <stdint.h>

struct htp_diag_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_dst;
    size_t src0_bytes_per_thread;
    size_t dst_bytes_per_thread;
};

static inline void htp_diag_vtcm_layout_build(
    struct htp_diag_vtcm_layout * L,
    size_t                        src_batch_size_aligned,
    size_t                        dst_row_size_aligned,
    uint32_t                      n_threads
) {
    L->src0_bytes_per_thread = src_batch_size_aligned;
    L->dst_bytes_per_thread  = dst_row_size_aligned;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes_per_thread * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes_per_thread * n_threads);

    L->total_bytes = off;
}

struct htp_diag_kernel_params {
    uint32_t n_threads;
    uint32_t total_batches;
    uint32_t use_dma;
    uint32_t can_split;
    uint32_t batches_per_chunk;
    uint32_t src_batch_size;
    uint32_t dst_row_size;
    uint32_t src_batch_size_aligned;
    uint32_t dst_row_size_aligned;
    uint32_t vtcm_size;

    struct fastdiv_values div_ne02;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_diag_kernel_params) <= 128, "htp_diag_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_diag_kernel_params) <= 128, "htp_diag_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_DIAG_OPS_H
