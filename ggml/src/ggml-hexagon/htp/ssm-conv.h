#ifndef HTP_SSM_CONV_H
#define HTP_SSM_CONV_H

#include <stdint.h>

#include "hex-fastdiv.h"
#include "htp-ops.h"
#include "htp-vtcm.h"

struct htp_ssm_conv_kernel_params {
    uint32_t n_threads;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;

    uint32_t vtcm_src0_size;
    uint32_t vtcm_src1_size;
    uint32_t vtcm_dst_size;
    uint32_t vtcm_size;
    uint32_t pad;

    struct fastdiv_values div_d_inner_tile;
};

struct htp_ssm_conv_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;
    size_t src0_bytes;
    size_t src1_bytes;
    size_t dst_bytes;
};

static inline void htp_ssm_conv_vtcm_layout_build(
    struct htp_ssm_conv_vtcm_layout * L,
    size_t src0_bytes_per_thread,
    size_t src1_bytes_per_thread,
    size_t dst_bytes_per_thread,
    uint32_t n_threads
) {
    L->src0_bytes = src0_bytes_per_thread * n_threads;
    L->src1_bytes = src1_bytes_per_thread * n_threads;
    L->dst_bytes  = dst_bytes_per_thread * n_threads;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_bytes);
    VTCM_LAYOUT_ALLOC(off, off_src1, L->src1_bytes);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_bytes);
    L->total_bytes = off;
}

#if defined(__cplusplus)
static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SSM_CONV_H
