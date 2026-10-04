#ifndef HTP_CONV2D_OPS_H
#define HTP_CONV2D_OPS_H

#include <stdint.h>

#include "hex-fastdiv.h"
#include "htp-ops.h"

enum htp_conv2d_kernel_type {
    HTP_CONV2D_KERNEL_SCALAR    = 1,
    HTP_CONV2D_KERNEL_HMX       = 2,
    HTP_CONV2D_KERNEL_HVX_WIDTH = 3,
    HTP_CONV2D_KERNEL_HVX_TILES = 4,
};

// HVX_TILES keeps fp16 weight pairs in the DDR scratchpad: 32 * rt_block * k_pad * 2 bytes
#define HTP_CONV2D_HVX_W_MAX (512 * 1024)

// HMX / HVX_TILES path: out[cout][pixel] = W[cout][k] x X[k][pixel], one col tile = 32 output pixels of one output row
struct htp_conv2d_kernel_params {
    uint32_t n_threads;
    uint32_t total_rows;
    uint32_t kernel_type;

    uint32_t cwhn;            // input is channels-contiguous, k order is (kh, kw, cin)
    uint32_t w_kmajor;        // kernel is [k][cout] with cout contiguous, k stride w_kstride
    uint32_t w_kstride;       // kernel k stride (bytes) for w_kmajor
    uint32_t k_size;          // KW*KH*Cin
    uint32_t k_group;         // k is split in n_groups contiguous groups of k_group, each padded to k_group_pad
    uint32_t k_group_pad;
    uint32_t n_groups;
    uint32_t n_k_tiles;       // round_up(n_groups * k_group_pad, 64) / 32
    uint32_t n_row_tiles;     // ceil(Cout / 32)
    uint32_t n_ow_tiles;      // ceil(OW / 32)
    uint32_t ct_chunk;        // col tiles per HMX job
    uint32_t rt_block;        // row tiles per weight block
    uint32_t seg_tiles;       // col tiles per staged input segment
    uint32_t stage_row;       // staged input row stride (floats)
    uint32_t stage_size;      // bytes of one staged input buffer
    uint32_t n_stage;         // 2 when a second stage buffer fits, else 1
    uint32_t wk_chunk;        // K elements per weight packing step

    uint32_t off_w;           // VTCM offsets and sizes (bytes)
    uint32_t off_act;
    uint32_t act_size;
    uint32_t off_out;
    uint32_t out_size;
    uint32_t off_thr;
    uint32_t thr_size;
    uint32_t vtcm_size;

    struct fastdiv_values div_ow_tiles;
    struct fastdiv_values div_oh;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128, "htp_conv2d_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128, "htp_conv2d_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_CONV2D_OPS_H
