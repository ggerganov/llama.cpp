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
    HTP_CONV2D_KERNEL_HMX_WIN   = 5,
};

// HMX_WIN output staging buffers per thread
#define HTP_CONV2D_WIN_OBUF 2

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

// HMX_WIN path: the input is a virtual image of N * IHS rows of SR pixels (zero padding included), pixel u = vr * SR + col.
// Output pixel q = vr * SR + ow reads input pixel q + kh * d1 * SR + kw * d0, so each tap is a shifted window over fp16
// croutons (32 pixels x 32 channels). One col tile = 32 consecutive q. Rows vr % IHS >= OH are not written.
// Header fields must match htp_conv2d_kernel_params.
struct htp_conv2d_win_params {
    uint32_t n_threads;
    uint32_t total_rows;
    uint32_t kernel_type;

    uint32_t cwhn;            // input is channels-contiguous
    uint32_t fold;            // kw taps folded into channels: crouton channel kw * Cin + cin, SR = OW, taps are kh only
    uint32_t w_kmajor;        // kernel cout is contiguous, else kernel is k-contiguous rows per cout
    uint32_t n_cc;            // crouton channel chunks of 32
    uint32_t act_rt_last;     // HMX activation Rt low bits (channel stop) for the last channel chunk
    uint32_t sr;              // virtual row width (IW + 2 * p0)
    uint32_t ihs;             // virtual rows per image (IH + 2 * p1)
    uint32_t max_off;         // input pixel offset of the last tap
    uint32_t n_row_tiles;     // ceil(Cout / 32)
    uint32_t rt_block;        // row tiles per weight block
    uint32_t ct_chunk;        // col tiles per HMX job
    uint32_t n_bc;            // croutons per chunk buffer
    uint32_t seg_b;           // croutons per staging step
    uint32_t seg_t;           // col tiles per output staging step
    uint32_t w_cs;            // cin per weight packing step (k-contiguous kernel) or taps per step (cout-contiguous kernel)
    uint32_t stage_size;      // bytes of one input staging buffer

    uint32_t off_w;           // VTCM offsets and sizes (bytes)
    uint32_t off_act;
    uint32_t act_size;
    uint32_t off_out;
    uint32_t out_size;
    uint32_t off_thr;
    uint32_t thr_size;
    uint32_t vtcm_size;

    struct fastdiv_values div_sr;
    struct fastdiv_values div_ihs;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128, "htp_conv2d_kernel_params is too large for kernel_params blob");
static_assert(sizeof(struct htp_conv2d_win_params) <= 128, "htp_conv2d_win_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_conv2d_kernel_params) <= 128, "htp_conv2d_kernel_params is too large for kernel_params blob");
_Static_assert(sizeof(struct htp_conv2d_win_params) <= 128, "htp_conv2d_win_params is too large for kernel_params blob");
#endif

#endif // HTP_CONV2D_OPS_H
