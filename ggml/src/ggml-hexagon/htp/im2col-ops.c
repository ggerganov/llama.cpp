#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#include "hex-common.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-vtcm.h"
#include "htp-tensor.h"

struct htp_im2col_context {
    struct htp_ops_context * octx;

    uint32_t pe_row_base;                          // first output row index assigned to this dev (DMA path)
    uint32_t pe_nrows;                             // number of output rows assigned to this dev (DMA path)
    uint32_t pe_rows_per_thread;                   // output rows per worker
    uint32_t pe_src_row_bytes;                     // one output row's source, rounded 256
    uint32_t pe_dst_row_bytes;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_size_per_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_size_per_thread;              // 2 * pe_dst_row_bytes

    uint32_t pe_owb;                               // output-col block size
    uint32_t pe_wb;                                // staged source window width
    uint32_t pe_nbpr;                              // blocks per row

    // Precomputed dimensions for worker threads
    uint32_t N;
    uint32_t IC;
    uint32_t ID, IH, IW;
    uint32_t KD, KH, KW;
    uint32_t OD, OH, OW;
    uint32_t patch_stride;

    // Per-thread initial coordinates (removes division from thread prologue)
    uint32_t thread_in[HTP_MAX_NTHREADS];
    uint32_t thread_iod[HTP_MAX_NTHREADS];
    uint32_t thread_ioh[HTP_MAX_NTHREADS];
};

// Per-op VTCM layout for the patch-embed DMA path
struct htp_im2col_vtcm_layout {
    size_t off_src;
    size_t off_dst;
    size_t src_bytes_per_thread;
    size_t dst_bytes_per_thread;
    size_t total_bytes;
};

static inline void htp_im2col_vtcm_layout_build(struct htp_im2col_vtcm_layout * L,
                                                size_t                          src_row_bytes,
                                                size_t                          dst_row_bytes,
                                                uint32_t                        n_threads) {
    L->src_bytes_per_thread = 2 * src_row_bytes + 256;
    L->dst_bytes_per_thread = 2 * dst_row_bytes;

    L->off_src     = 0;
    L->off_dst     = L->off_src + L->src_bytes_per_thread * n_threads;
    L->total_bytes = L->off_dst + L->dst_bytes_per_thread * n_threads;
}

static inline void im2col_step_row(uint32_t * in, uint32_t * ioh, uint32_t OH) {
    (*ioh)++;
    if (*ioh >= OH) {
        *ioh = 0;
        (*in)++;
    }
}

static inline void im2col_step_block(uint32_t * r, uint32_t * in, uint32_t * ioh, uint32_t * c0,
                                     uint32_t owb, uint32_t OW, uint32_t OH) {
    *c0 += owb;
    if (*c0 >= OW) {
        *c0 = 0;
        (*r)++;
        (*ioh)++;
        if (*ioh >= OH) {
            *ioh = 0;
            (*in)++;
        }
    }
}

static uint32_t im2col_blocked_dma_stage(struct htp_im2col_context * ictx, dma_queue * dma_q,
                                         float * srcb, uint32_t in, uint32_t ioh, uint32_t c0) {
    const struct htp_ops_context * octx = ictx->octx;
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    const int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    const int32_t d1 = octx->op_params[5];
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KH = ictx->KH;
    const uint32_t Wb = ictx->pe_wb;
    const dma_addr_t src_data = octx->src[1]->data;
    const int32_t win0 = (int32_t) c0 * s0 - p0;
    uint32_t pushes = 0;

    for (uint32_t ikh = 0; ikh < KH; ikh++) {
        const int32_t iih = (int32_t) ioh * s1 + (int32_t) ikh * d1 - p1;
        if (iih < 0 || iih >= (int32_t) IH) {
            continue;
        }
        const int32_t lo = win0 < 0 ? -win0 : 0;
        int32_t hi = (int32_t) IW - win0;
        if (hi > (int32_t) Wb) {
            hi = (int32_t) Wb;
        }
        if (hi <= lo) {
            continue;
        }
        const uint32_t cpw = (uint32_t) (hi - lo);
        float * vdst = srcb + (size_t) ikh * Wb + lo;
        const dma_addr_t vsrc = src_data + (size_t) (((in * IC) * IH + iih) * IW + (win0 + lo)) * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(vdst, vsrc),
                       (size_t) KH * Wb * sizeof(float), (size_t) IH * IW * sizeof(float),
                       cpw * sizeof(float), IC);
        pushes++;
    }
    return pushes;
}

#define IM2COL_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                                \
static __attribute__((noinline)) void FNAME(                                                            \
    const struct htp_im2col_context * ictx,                                                             \
    DST_CTYPE * restrict dstb, const float * restrict srcb,                                             \
    float * restrict scratch,                                                                           \
    uint32_t nb, uint32_t c0, uint32_t ioh) {                                                           \
    const struct htp_ops_context * octx = ictx->octx;                                                   \
    const int32_t  s0 = octx->op_params[0], s1 = octx->op_params[1];                                    \
    const int32_t  p0 = octx->op_params[2], p1 = octx->op_params[3];                                    \
    const int32_t  d0 = octx->op_params[4], d1 = octx->op_params[5];                                    \
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;                                         \
    const uint32_t KH = ictx->KH, KW = ictx->KW;                                                        \
    const uint32_t Wb = ictx->pe_wb;                                                                     \
    const uint32_t patch_stride = ictx->patch_stride;                                                   \
    const int64_t  win0 = (int64_t) c0 * s0 - p0;                                                       \
                                                                                                        \
    for (uint32_t j = 0; j < nb; j++) {                                                                 \
        const uint32_t iow       = c0 + j;                                                              \
        DST_CTYPE *    dst_patch = dstb + (uint64_t) j * patch_stride;                                  \
        const int64_t  iiw0      = (int64_t) iow * s0 - p0;                                             \
        for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                       \
            const int64_t iih = (int64_t) ioh * s1 + (int64_t) ikh * d1 - p1;                           \
            const int     okh = (iih >= 0 && iih < (int64_t) IH);                                       \
            for (uint32_t iic = 0; iic < IC; iic++) {                                                   \
                DST_CTYPE * out_run = dst_patch + ((uint64_t) iic * KH + ikh) * KW;                    \
                if (!okh) {                                                                             \
                    SPLAT_FN(out_run, 0, KW);                                                           \
                    continue;                                                                           \
                }                                                                                       \
                const float * vrow = srcb + ((uint64_t) (iic * KH + ikh)) * Wb;                        \
                if (d0 == 1) {                                                                          \
                    const int64_t lo = iiw0 < 0 ? -iiw0 : 0;                                            \
                    int64_t       hi = (int64_t) IW - iiw0;                                             \
                    if (hi > (int64_t) KW) {                                                            \
                        hi = (int64_t) KW;                                                              \
                    }                                                                                   \
                    if (hi <= lo) {                                                                     \
                        SPLAT_FN(out_run, 0, KW);                                                       \
                    } else {                                                                            \
                        if (lo > 0) {                                                                   \
                            SPLAT_FN(out_run, 0, (uint32_t) lo);                                        \
                        }                                                                               \
                        COPY_FN((uint8_t *) (out_run + lo),                                             \
                                (const uint8_t *) (vrow + (iiw0 + lo - win0)),                          \
                                (uint32_t) (hi - lo));                                                  \
                        if (hi < (int64_t) KW) {                                                        \
                            SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                            \
                        }                                                                               \
                    }                                                                                   \
                    continue;                                                                           \
                }                                                                                       \
                for (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                           \
                    const uint32_t chunk = MIN(KW - ikw, 64);                                           \
                    for (uint32_t c = 0; c < chunk; c++) {                                              \
                        const int64_t iiw = iiw0 + (int64_t) (ikw + c) * d0;                            \
                        scratch[c] = (iiw < 0 || iiw >= (int64_t) IW) ? 0.0f : vrow[iiw - win0];        \
                    }                                                                                   \
                    COPY_FN((uint8_t *) (out_run + ikw), (const uint8_t *) scratch, chunk);             \
                }                                                                                       \
            }                                                                                           \
        }                                                                                               \
    }                                                                                                   \
}

IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_BLOCKED_COMPUTE_BODY(im2col_blocked_compute_f32, float,    hvx_copy_f32_uu,     hvx_splat_f32_u)

#define IM2COL_BLOCKED_DMA_BODY(FNAME, DST_CTYPE, COMPUTE_FN, DST_ELEM)                               \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                               \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                  \
        struct htp_ops_context *    octx        = ictx->octx;                                          \
        struct htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];                              \
        const uint32_t OH                       = ictx->OH;                                            \
        const uint32_t OW                       = ictx->OW;                                            \
        const uint32_t owb                      = ictx->pe_owb;                                        \
        const uint32_t nbpr                     = ictx->pe_nbpr;                                       \
        const uint32_t patch_stride             = ictx->patch_stride;                                  \
        const dma_addr_t dst_data               = octx->dst->data;                                     \
        dma_queue *    dma_q                    = octx->ctx->dma[ith];                                 \
        uint8_t *      srcb_base                = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread; \
        uint8_t *      dstb_base                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread; \
        float *        srcb2[2]                 = { (float *) srcb_base, (float *) (srcb_base + ictx->pe_src_row_bytes) }; \
        DST_CTYPE *    dstb2[2]                 = { (DST_CTYPE *) dstb_base,                          \
                                                    (DST_CTYPE *) (dstb_base + ictx->pe_dst_row_bytes) }; \
        float *        scratch                  = (float *) (srcb_base + 2 * ictx->pe_src_row_bytes);  \
        const uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;                  \
        const uint32_t row_start                = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;  \
        const uint32_t row_end                  = MIN(row_start + ictx->pe_rows_per_thread, row_end_max); \
        if (row_start >= row_end) {                                                                    \
            return;                                                                                    \
        }                                                                                              \
        const uint32_t nrows_local  = row_end - row_start;                                             \
        const uint32_t total_blocks = nrows_local * nbpr;                                              \
        if (total_blocks == 0) {                                                                       \
            return;                                                                                    \
        }                                                                                              \
        uint32_t cur_ioh   = ictx->thread_ioh[ith];                                                    \
        uint32_t cur_in    = ictx->thread_in[ith];                                                     \
        uint32_t cur_c0    = 0;                                                                        \
        uint32_t cur_r     = row_start;                                                                \
        uint32_t stage_ioh = cur_ioh;                                                                  \
        uint32_t stage_in  = cur_in;                                                                   \
        uint32_t stage_c0  = 0;                                                                        \
        uint32_t stage_r   = cur_r;                                                                    \
        uint32_t src_pushes[2] = {0, 0};                                                               \
        for (uint32_t i = 0; i < total_blocks && i < 2; i++) {                                         \
            dma_queue_push(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);                      \
            src_pushes[i] = im2col_blocked_dma_stage(ictx, dma_q, srcb2[i],                            \
                                                     stage_in, stage_ioh, stage_c0);                   \
            im2col_step_block(&stage_r, &stage_in, &stage_ioh, &stage_c0, owb, OW, OH);                \
        }                                                                                              \
        for (uint32_t bi = 0; bi < total_blocks; bi++) {                                               \
            const uint32_t buf = bi & 1u;                                                              \
            float * srcb = srcb2[buf];                                                                 \
            DST_CTYPE * dstb = dstb2[buf];                                                             \
            const uint32_t nb = MIN(owb, OW - cur_c0);                                                 \
            dma_queue_pop(dma_q);                                                                      \
            for (uint32_t k = 0; k < src_pushes[buf]; k++) {                                           \
                dma_queue_pop(dma_q);                                                                  \
            }                                                                                          \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur_r);                                  \
            COMPUTE_FN(ictx, dstb, srcb, scratch, nb, cur_c0, cur_ioh);                                \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_r);                                   \
            const dma_addr_t ddr = dst_data + (size_t) ((cur_in * OH + cur_ioh) * OW + cur_c0) * patch_stride * (DST_ELEM); \
            dma_queue_push(dma_q, dma_make_data(ddr, dstb),                                            \
                           nb * patch_stride * (DST_ELEM), nb * patch_stride * (DST_ELEM),             \
                           nb * patch_stride * (DST_ELEM), 1);                                         \
            if (bi + 2 < total_blocks) {                                                               \
                src_pushes[buf] = im2col_blocked_dma_stage(ictx, dma_q, srcb,                          \
                                                           stage_in, stage_ioh, stage_c0);             \
                im2col_step_block(&stage_r, &stage_in, &stage_ioh, &stage_c0, owb, OW, OH);            \
            }                                                                                          \
            im2col_step_block(&cur_r, &cur_in, &cur_ioh, &cur_c0, owb, OW, OH);                       \
        }                                                                                              \
        dma_queue_flush(dma_q);                                                                        \
    }

IM2COL_BLOCKED_DMA_BODY(im2col_blocked_dma_f16_thread, uint16_t, im2col_blocked_compute_f16, sizeof(__fp16))
IM2COL_BLOCKED_DMA_BODY(im2col_blocked_dma_f32_thread, float,    im2col_blocked_compute_f32, sizeof(float))

static uint32_t im2col_patchembed_dma_stage(struct htp_im2col_context * ictx, dma_queue * dma_q,
                                            float * srcb, uint32_t in, uint32_t ioh) {
    const struct htp_ops_context * octx = ictx->octx;
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KH = ictx->KH;
    const dma_addr_t src_data = octx->src[1]->data;
    const size_t src_channel_stride  = (size_t) IH * IW * sizeof(float);
    const size_t vtcm_channel_stride = (size_t) KH * IW * sizeof(float);
    uint32_t pushes = 0;

    for (uint32_t ikh = 0; ikh < KH; ikh++) {
        const uint32_t iih = ioh * KH + ikh;
        float * vdst = srcb + (size_t) ikh * IW;
        const dma_addr_t vsrc = src_data +
            (((uint64_t) in * IC) * IH + iih) * IW * sizeof(float);
        dma_queue_push(dma_q, dma_make_data(vdst, vsrc),
                       vtcm_channel_stride, src_channel_stride, IW * sizeof(float), IC);
        pushes++;
    }
    return pushes;
}

static __attribute__((noinline)) void im2col_patchembed_compute_f16(
    const struct htp_im2col_context * ictx,
    uint16_t * restrict dstb, const float * restrict srcb, uint32_t ioh) {
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KH = ictx->KH, KW = ictx->KW;
    const uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    for (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (uint64_t) iow * patch_stride;
        for (uint32_t ikh = 0; ikh < KH; ikh++) {
            const int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            for (uint32_t iic = 0; iic < IC; iic++) {
                uint16_t * out_run = dst_patch + ((uint64_t) iic * KH + ikh) * KW;
                if (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_u16_u(out_run, 0, KW);
                    continue;
                }
                const float * src_run = srcb + ((uint64_t) iic * KH + ikh) * IW + (uint64_t) iow * KW;
                hvx_copy_f16_f32_uu((uint8_t *) out_run, (const uint8_t *) src_run, KW);
            }
        }
    }
}

static __attribute__((noinline)) void im2col_patchembed_compute_f32(
    const struct htp_im2col_context * ictx,
    float * restrict dstb, const float * restrict srcb, uint32_t ioh) {
    const uint32_t IC = ictx->IC, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KH = ictx->KH, KW = ictx->KW;
    const uint32_t OW = ictx->OW, patch_stride = ictx->patch_stride;

    for (uint32_t iow = 0; iow < OW; iow++) {
        float * dst_patch = dstb + (uint64_t) iow * patch_stride;
        for (uint32_t ikh = 0; ikh < KH; ikh++) {
            const int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;
            for (uint32_t iic = 0; iic < IC; iic++) {
                float * out_run = dst_patch + ((uint64_t) iic * KH + ikh) * KW;
                if (iih < 0 || iih >= (int32_t) IH) {
                    hvx_splat_f32_u(out_run, 0, KW);
                    continue;
                }
                const float * src_run = srcb + ((uint64_t) iic * KH + ikh) * IW + (uint64_t) iow * KW;
                hvx_copy_f32_uu((uint8_t *) out_run, (const uint8_t *) src_run, KW);
            }
        }
    }
}

#define IM2COL_PATCHEMBED_DMA_BODY(FNAME, DST_CTYPE, COMPUTE_FN, DST_ELEM)                            \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                               \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                  \
        struct htp_ops_context *    octx        = ictx->octx;                                          \
        struct htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];                              \
        const uint32_t OH                       = ictx->OH;                                            \
        const uint32_t OW                       = ictx->OW;                                            \
        const uint32_t patch_stride             = ictx->patch_stride;                                  \
        const dma_addr_t dst_data               = octx->dst->data;                                     \
        dma_queue *    dma_q                    = octx->ctx->dma[ith];                                 \
        uint8_t *      srcb_base                = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread; \
        uint8_t *      dstb_base                = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread; \
        float *        srcb2[2]                 = { (float *) srcb_base, (float *) (srcb_base + ictx->pe_src_row_bytes) }; \
        DST_CTYPE *    dstb2[2]                 = { (DST_CTYPE *) dstb_base,                          \
                                                    (DST_CTYPE *) (dstb_base + ictx->pe_dst_row_bytes) }; \
        const uint32_t row_end_max              = ictx->pe_row_base + ictx->pe_nrows;                  \
        const uint32_t row_start                = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;  \
        const uint32_t row_end                  = MIN(row_start + ictx->pe_rows_per_thread, row_end_max); \
        if (row_start >= row_end) {                                                                    \
            return;                                                                                    \
        }                                                                                              \
        const uint32_t total_rows = row_end - row_start;                                               \
        if (total_rows == 0) {                                                                         \
            return;                                                                                    \
        }                                                                                              \
        uint32_t cur_ioh   = ictx->thread_ioh[ith];                                                    \
        uint32_t cur_in    = ictx->thread_in[ith];                                                     \
        uint32_t stage_ioh = cur_ioh;                                                                  \
        uint32_t stage_in  = cur_in;                                                                   \
        uint32_t src_pushes[2] = {0, 0};                                                               \
        for (uint32_t i = 0; i < total_rows && i < 2; i++) {                                           \
            dma_queue_push(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);                      \
            src_pushes[i] = im2col_patchembed_dma_stage(ictx, dma_q, srcb2[i],                         \
                                                        stage_in, stage_ioh);                          \
            im2col_step_row(&stage_in, &stage_ioh, OH);                                                \
        }                                                                                              \
        for (uint32_t ri = 0; ri < total_rows; ri++) {                                                 \
            const uint32_t buf = ri & 1u;                                                              \
            float * srcb = srcb2[buf];                                                                 \
            DST_CTYPE * dstb = dstb2[buf];                                                             \
            dma_queue_pop(dma_q);                                                                      \
            for (uint32_t k = 0; k < src_pushes[buf]; k++) {                                           \
                dma_queue_pop(dma_q);                                                                  \
            }                                                                                          \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, row_start + ri);                         \
            COMPUTE_FN(ictx, dstb, srcb, cur_ioh);                                                     \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, row_start + ri);                          \
            const dma_addr_t ddr_row = dst_data + (size_t) (cur_in * OH + cur_ioh) * OW * patch_stride * (DST_ELEM); \
            dma_queue_push(dma_q, dma_make_data(ddr_row, dstb),                                        \
                           OW * patch_stride * (DST_ELEM), OW * patch_stride * (DST_ELEM),             \
                           OW * patch_stride * (DST_ELEM), 1);                                         \
            if (ri + 2 < total_rows) {                                                                 \
                src_pushes[buf] = im2col_patchembed_dma_stage(ictx, dma_q, srcb,                       \
                                                              stage_in, stage_ioh);                    \
                im2col_step_row(&stage_in, &stage_ioh, OH);                                            \
            }                                                                                          \
            im2col_step_row(&cur_in, &cur_ioh, OH);                                                    \
        }                                                                                              \
        dma_queue_flush(dma_q);                                                                        \
    }

IM2COL_PATCHEMBED_DMA_BODY(im2col_patchembed_dma_f16_thread, uint16_t, im2col_patchembed_compute_f16, sizeof(__fp16))
IM2COL_PATCHEMBED_DMA_BODY(im2col_patchembed_dma_f32_thread, float,    im2col_patchembed_compute_f32, sizeof(float))

static bool im2col_use_patchembed_dma(const struct htp_ops_context * octx) {
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    const int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    const int32_t d0 = octx->op_params[4], d1 = octx->op_params[5];
    const int     is_2D = octx->op_params[6] == 1;
    if (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        return false;
    }
    const uint32_t KH = is_2D ? octx->src[0]->ne[1] : 1;
    const uint32_t KW = octx->src[0]->ne[0];
    if (s0 != (int32_t) KW) {
        return false;  // non-overlapping (width)
    }
    if (p0 != 0) {
        return false;  // no padding (width)
    }
    if (d0 != 1) {
        return false;  // no dilation (width)
    }
    if (is_2D) {
        if (s1 != (int32_t) KH) {
            return false;  // non-overlapping (height)
        }
        if (p1 != 0) {
            return false;  // no padding (height)
        }
        if (d1 != 1) {
            return false;  // no dilation (height)
        }
    }
    return true;
}

// Sizes the per-thread 2x(src,dst) VTCM ping-pong for the patch-embed DMA path.
// Returns false if it doesn't fit the VTCM budget (caller falls back).
static bool im2col_patchembed_dma_fits(struct htp_ops_context *    octx,
                                       struct htp_im2col_context * ictx,
                                       uint32_t                    n_threads) {
    const int32_t  is_2D = octx->op_params[6] == 1;
    const uint32_t IC = is_2D ? octx->src[1]->ne[2] : octx->src[1]->ne[1];
    const uint32_t IW = octx->src[1]->ne[0];
    const uint32_t KH = is_2D ? octx->src[0]->ne[1] : 1;
    const uint32_t KW = octx->src[0]->ne[0];
    const uint32_t OW           = octx->dst->ne[1];
    const uint32_t patch_stride = IC * KH * KW;

    ictx->pe_src_row_bytes  = hex_round_up(IC * KH * IW * sizeof(float), 256);
    const uint32_t dst_elem = (octx->dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    ictx->pe_dst_row_bytes  = hex_round_up(OW * patch_stride * dst_elem, 256);

    // 2 src + 2 dst buffers per thread (ping-pong), src region first then dst.
    struct htp_im2col_vtcm_layout L;
    htp_im2col_vtcm_layout_build(&L, ictx->pe_src_row_bytes, ictx->pe_dst_row_bytes, n_threads);
    if (L.total_bytes > octx->ctx->vtcm_size) {
        return false;
    }

    uint8_t * const base        = octx->ctx->vtcm_base;
    ictx->pe_vtcm_src           = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src);
    ictx->pe_vtcm_dst           = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    ictx->pe_src_size_per_thread = (uint32_t) L.src_bytes_per_thread;
    ictx->pe_dst_size_per_thread = (uint32_t) L.dst_bytes_per_thread;
    return true;
}

// Sizes a per-thread 2x(src,dst) VTCM ping-pong for the blocked general kernel.
// Stages Wb=(owb-1)*s0+(KW-1)*d0+1 source cols per (iic,ikh) row and owb patches
// of dst. Picks the largest owb that fits; returns false if even owb=1 does not.
static bool im2col_blocked_dma_fits(struct htp_ops_context *    octx,
                                    struct htp_im2col_context * ictx,
                                    uint32_t                    n_threads) {
    const int32_t  is_2D = octx->op_params[6] == 1;
    const int32_t  s0 = octx->op_params[0];
    const int32_t  d0 = octx->op_params[4];
    const uint32_t IC = is_2D ? octx->src[1]->ne[2] : octx->src[1]->ne[1];
    const uint32_t KH = is_2D ? octx->src[0]->ne[1] : 1;
    const uint32_t KW = octx->src[0]->ne[0];
    const uint32_t OW = octx->dst->ne[1];
    const uint32_t patch_stride = IC * KH * KW;
    const uint32_t dst_elem = (octx->dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);

    for (uint32_t owb = (OW < 256 ? OW : 256); owb >= 1; owb--) {
        const uint32_t Wb = (owb - 1) * (uint32_t) s0 + (KW - 1) * (uint32_t) d0 + 1;
        const uint32_t src_row_bytes = hex_round_up(IC * KH * Wb * sizeof(float), 256);
        const uint32_t dst_row_bytes = hex_round_up(owb * patch_stride * dst_elem, 256);
        struct htp_im2col_vtcm_layout L;
        htp_im2col_vtcm_layout_build(&L, src_row_bytes, dst_row_bytes, n_threads);
        if (L.total_bytes <= octx->ctx->vtcm_size) {
            uint8_t * const base = octx->ctx->vtcm_base;
            ictx->pe_owb                 = owb;
            ictx->pe_wb                  = Wb;
            ictx->pe_nbpr                = (OW + owb - 1) / owb;
            ictx->pe_src_row_bytes       = src_row_bytes;
            ictx->pe_dst_row_bytes       = dst_row_bytes;
            ictx->pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src);
            ictx->pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
            ictx->pe_src_size_per_thread = (uint32_t) L.src_bytes_per_thread;
            ictx->pe_dst_size_per_thread = (uint32_t) L.dst_bytes_per_thread;
            return true;
        }
        if (owb == 1) break;  // avoid unsigned underflow
    }
    return false;
}

static inline void im2col_3d_step_row(uint32_t * in, uint32_t * iod, uint32_t * ioh,
                                      uint32_t OH, uint32_t OD) {
    (*ioh)++;
    if (*ioh >= OH) {
        *ioh = 0;
        (*iod)++;
        if (*iod >= OD) {
            *iod = 0;
            (*in)++;
        }
    }
}

static inline void im2col_3d_step_block(uint32_t * r, uint32_t * in, uint32_t * iod, uint32_t * ioh, uint32_t * c0,
                                        uint32_t owb, uint32_t OW, uint32_t OH, uint32_t OD) {
    *c0 += owb;
    if (*c0 >= OW) {
        *c0 = 0;
        (*r)++;
        (*ioh)++;
        if (*ioh >= OH) {
            *ioh = 0;
            (*iod)++;
            if (*iod >= OD) {
                *iod = 0;
                (*in)++;
            }
        }
    }
}

static uint32_t im2col_3d_blocked_dma_stage(struct htp_im2col_context * ictx, dma_queue * dma_q,
                                            float * srcb, uint32_t in, uint32_t iod, uint32_t ioh, uint32_t c0) {
    struct htp_ops_context * octx = ictx->octx;
    const struct htp_tensor * src1 = octx->src[1];
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    const int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    const int32_t d1 = octx->op_params[7], d2 = octx->op_params[8];
    const uint32_t IC = ictx->IC;
    const uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;
    const uint32_t KD = ictx->KD, KH = ictx->KH;
    const uint32_t Wb = ictx->pe_wb;
    const int64_t win0 = (int64_t) c0 * s0 - p0;
    const size_t src_channel_stride = (size_t) ID * IH * IW * sizeof(float);
    const size_t vtcm_channel_stride = (size_t) KD * KH * Wb * sizeof(float);
    uint32_t pushes = 0;

    for (uint32_t ikd = 0; ikd < KD; ikd++) {
        const int64_t iid = (int64_t) iod * s2 + (int64_t) ikd * d2 - p2;
        if (iid < 0 || iid >= (int64_t) ID) {
            continue;
        }
        for (uint32_t ikh = 0; ikh < KH; ikh++) {
            const int64_t iih = (int64_t) ioh * s1 + (int64_t) ikh * d1 - p1;
            if (iih < 0 || iih >= (int64_t) IH) {
                continue;
            }
            const int64_t lo = win0 < 0 ? -win0 : 0;
            int64_t hi = (int64_t) IW - win0;
            if (hi > (int64_t) Wb) {
                hi = (int64_t) Wb;
            }
            if (hi <= lo) {
                continue;
            }
            float * vdst = srcb + ((size_t) ikd * KH + ikh) * Wb + lo;
            const dma_addr_t vsrc = src1->data +
                (((((uint64_t) in * IC) * ID + iid) * IH + iih) * IW + (win0 + lo)) * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride,
                           (size_t) (hi - lo) * sizeof(float), IC);
            pushes++;
        }
    }
    return pushes;
}

#define IM2COL_3D_BLOCKED_COMPUTE_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN)                          \
static __attribute__((noinline)) void FNAME(const struct htp_im2col_context * ictx,                  \
                                            DST_CTYPE * restrict dstb, const float * restrict srcb,  \
                                            float * restrict scratch,                                \
                                            uint32_t nb, uint32_t c0, uint32_t iod, uint32_t ioh) {  \
    const struct htp_ops_context * octx = ictx->octx;                                                \
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];         \
    const int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];         \
    const int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];         \
    const uint32_t IC = ictx->IC;                                                                    \
    const uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                      \
    const uint32_t KD = ictx->KD, KH = ictx->KH, KW = ictx->KW;                                      \
    const uint32_t patch_stride = ictx->patch_stride;                                                \
    const uint32_t Wb = ictx->pe_wb;                                                                 \
    const int64_t win0 = (int64_t) c0 * s0 - p0;                                                     \
    for (uint32_t j = 0; j < nb; j++) {                                                              \
        const uint32_t iow = c0 + j;                                                                 \
        const int64_t iiw0 = (int64_t) iow * s0 - p0;                                                \
        DST_CTYPE * dst_patch = dstb + (uint64_t) j * patch_stride;                                  \
        for (uint32_t iic = 0; iic < IC; iic++) {                                                    \
            for (uint32_t ikd = 0; ikd < KD; ikd++) {                                                \
                const int64_t iid = (int64_t) iod * s2 + (int64_t) ikd * d2 - p2;                    \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                            \
                    const int64_t iih = (int64_t) ioh * s1 + (int64_t) ikh * d1 - p1;                \
                    DST_CTYPE * out_run = dst_patch +                                                \
                        ((uint64_t) iic * KD * KH + (uint64_t) ikd * KH + ikh) * KW;                 \
                    if (iid < 0 || iid >= (int64_t) ID || iih < 0 || iih >= (int64_t) IH) {          \
                        SPLAT_FN(out_run, 0, KW);                                                    \
                        continue;                                                                    \
                    }                                                                                \
                    const float * vrow = srcb +                                                      \
                        ((uint64_t) iic * KD * KH + (uint64_t) ikd * KH + ikh) * Wb;                 \
                    if (d0 == 1) {                                                                   \
                        const int64_t lo = iiw0 < 0 ? -iiw0 : 0;                                     \
                        int64_t hi = (int64_t) IW - iiw0;                                            \
                        if (hi > (int64_t) KW) {                                                     \
                            hi = (int64_t) KW;                                                       \
                        }                                                                            \
                        if (hi <= lo) {                                                              \
                            SPLAT_FN(out_run, 0, KW);                                                \
                        } else {                                                                     \
                            if (lo > 0) {                                                            \
                                SPLAT_FN(out_run, 0, (uint32_t) lo);                                 \
                            }                                                                        \
                            COPY_FN((uint8_t *) (out_run + lo),                                      \
                                    (const uint8_t *) (vrow + (iiw0 + lo - win0)),                   \
                                    (uint32_t) (hi - lo));                                           \
                            if (hi < (int64_t) KW) {                                                 \
                                SPLAT_FN(out_run + hi, 0, (uint32_t) (KW - hi));                     \
                            }                                                                        \
                        }                                                                            \
                        continue;                                                                    \
                    }                                                                                \
                    for (uint32_t ikw = 0; ikw < KW; ikw += 64) {                                    \
                        const uint32_t chunk = MIN(KW - ikw, 64);                                    \
                        for (uint32_t c = 0; c < chunk; c++) {                                       \
                            const int64_t iiw = iiw0 + (int64_t) (ikw + c) * d0;                     \
                            scratch[c] = (iiw < 0 || iiw >= (int64_t) IW) ? 0.0f : vrow[iiw - win0]; \
                        }                                                                            \
                        COPY_FN((uint8_t *) (out_run + ikw), (const uint8_t *) scratch, chunk);      \
                    }                                                                                \
                }                                                                                    \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
}

IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f16, uint16_t, hvx_copy_f16_f32_uu, hvx_splat_u16_u)
IM2COL_3D_BLOCKED_COMPUTE_BODY(im2col_3d_blocked_compute_f32, float,    hvx_copy_f32_uu,     hvx_splat_f32_u)

#define IM2COL_3D_BLOCKED_DMA_BODY(FNAME, DST_CTYPE, COMPUTE_FN, DST_ELEM)                                     \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                                       \
        (void) nth;                                                                                            \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                          \
        struct htp_ops_context *    octx        = ictx->octx;                                                  \
        struct htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];                                      \
        const struct htp_tensor * restrict dst  = octx->dst;                                                   \
        const uint32_t OD = ictx->OD, OH = ictx->OH, OW = ictx->OW;                                            \
        const uint32_t patch_stride = ictx->patch_stride;                                                      \
        const uint32_t owb          = ictx->pe_owb;                                                            \
        const uint32_t nbpr         = ictx->pe_nbpr;                                                           \
        const dma_addr_t dst_data   = dst->data;                                                               \
        dma_queue * dma_q           = octx->ctx->dma[ith];                                                     \
        uint8_t * srcb_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;                  \
        uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;                  \
        float * srcb2[2]            = { (float *) srcb_base, (float *) (srcb_base + ictx->pe_src_row_bytes) }; \
        DST_CTYPE * dstb2[2]        = { (DST_CTYPE *) dstb_base,                                               \
                                        (DST_CTYPE *) (dstb_base + ictx->pe_dst_row_bytes) };                  \
        float * scratch             = (float *) (srcb_base + 2 * ictx->pe_src_row_bytes);                      \
        const uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;                                      \
        const uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;                      \
        const uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);                  \
        if (row_start >= row_end) {                                                                            \
            return;                                                                                            \
        }                                                                                                      \
        const uint32_t nrows_local  = row_end - row_start;                                                     \
        const uint32_t total_blocks = nrows_local * nbpr;                                                      \
        if (total_blocks == 0) {                                                                               \
            return;                                                                                            \
        }                                                                                                      \
        uint32_t cur_ioh   = ictx->thread_ioh[ith];                                                            \
        uint32_t cur_iod   = ictx->thread_iod[ith];                                                            \
        uint32_t cur_in    = ictx->thread_in[ith];                                                             \
        uint32_t cur_c0    = 0;                                                                                \
        uint32_t cur_r     = row_start;                                                                        \
        uint32_t stage_ioh = cur_ioh;                                                                          \
        uint32_t stage_iod = cur_iod;                                                                          \
        uint32_t stage_in  = cur_in;                                                                           \
        uint32_t stage_c0  = 0;                                                                                \
        uint32_t stage_r   = cur_r;                                                                            \
        uint32_t src_pushes[2] = {0, 0};                                                                       \
        for (uint32_t i = 0; i < total_blocks && i < 2; i++) {                                                 \
            dma_queue_push(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);                              \
            src_pushes[i] = im2col_3d_blocked_dma_stage(ictx, dma_q, srcb2[i],                                 \
                                                         stage_in, stage_iod, stage_ioh, stage_c0);            \
            im2col_3d_step_block(&stage_r, &stage_in, &stage_iod, &stage_ioh, &stage_c0,                       \
                                 owb, OW, OH, OD);                                                             \
        }                                                                                                      \
        for (uint32_t bi = 0; bi < total_blocks; bi++) {                                                       \
            const uint32_t buf = bi & 1u;                                                                      \
            float * srcb = srcb2[buf];                                                                         \
            DST_CTYPE * dstb = dstb2[buf];                                                                     \
            const uint32_t r   = cur_r;                                                                        \
            const uint32_t c0  = cur_c0;                                                                       \
            const uint32_t iod = cur_iod;                                                                      \
            const uint32_t ioh = cur_ioh;                                                                      \
            const uint32_t nb  = MIN(owb, OW - c0);                                                            \
            dma_queue_pop(dma_q);                                                                              \
            for (uint32_t p = 0; p < src_pushes[buf]; p++) {                                                   \
                dma_queue_pop(dma_q);                                                                          \
            }                                                                                                  \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);                                              \
            COMPUTE_FN(ictx, dstb, srcb, scratch, nb, c0, iod, ioh);                                           \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);                                               \
            const dma_addr_t ddr = dst_data +                                                                  \
                ((uint64_t) r * OW + c0) * patch_stride * (DST_ELEM);                                          \
            dma_queue_push(dma_q, dma_make_data(ddr, dstb),                                                    \
                           nb * patch_stride * (DST_ELEM), nb * patch_stride * (DST_ELEM),                     \
                           nb * patch_stride * (DST_ELEM), 1);                                                 \
            if (bi + 2 < total_blocks) {                                                                       \
                src_pushes[buf] = im2col_3d_blocked_dma_stage(ictx, dma_q, srcb,                               \
                                                              stage_in, stage_iod, stage_ioh, stage_c0);       \
                im2col_3d_step_block(&stage_r, &stage_in, &stage_iod, &stage_ioh, &stage_c0,                   \
                                     owb, OW, OH, OD);                                                         \
            }                                                                                                  \
            im2col_3d_step_block(&cur_r, &cur_in, &cur_iod, &cur_ioh, &cur_c0,                                 \
                                 owb, OW, OH, OD);                                                             \
        }                                                                                                      \
        dma_queue_flush(dma_q);                                                                                \
    }

IM2COL_3D_BLOCKED_DMA_BODY(im2col_3d_blocked_dma_f16_thread, uint16_t, im2col_3d_blocked_compute_f16, sizeof(__fp16))
IM2COL_3D_BLOCKED_DMA_BODY(im2col_3d_blocked_dma_f32_thread, float,    im2col_3d_blocked_compute_f32, sizeof(float))
static uint32_t im2col_3d_patchembed_dma_stage(dma_queue * dma_q, const dma_addr_t src_data,
                                               float * srcb, uint32_t in, uint32_t iod, uint32_t ioh,
                                               uint32_t IC, uint32_t ID, uint32_t IH, uint32_t IW,
                                               uint32_t KD, uint32_t KH) {
    const size_t src_channel_stride  = (size_t) ID * IH * IW * sizeof(float);
    const size_t vtcm_channel_stride = (size_t) KD * KH * IW * sizeof(float);
    uint32_t pushes = 0;

    for (uint32_t ikd = 0; ikd < KD; ikd++) {
        const uint32_t iid = iod * KD + ikd;
        for (uint32_t ikh = 0; ikh < KH; ikh++) {
            const uint32_t iih = ioh * KH + ikh;
            float * vdst = srcb + ((size_t) ikd * KH + ikh) * IW;
            const dma_addr_t vsrc = src_data +
                ((((uint64_t) in * IC) * ID + iid) * IH + iih) * IW * sizeof(float);
            dma_queue_push(dma_q, dma_make_data(vdst, vsrc),
                           vtcm_channel_stride, src_channel_stride, IW * sizeof(float), IC);
            pushes++;
        }
    }
    return pushes;
}

static __attribute__((noinline)) void im2col_3d_patchembed_compute_f16(
    const struct htp_im2col_context * ictx,
    uint16_t * restrict dstb, const float * restrict srcb) {
    const uint32_t OW           = ictx->OW;
    const uint32_t IC           = ictx->IC;
    const uint32_t KD           = ictx->KD;
    const uint32_t KH           = ictx->KH;
    const uint32_t KW           = ictx->KW;
    const uint32_t IW           = ictx->IW;
    const uint32_t patch_stride = ictx->patch_stride;

    for (uint32_t iow = 0; iow < OW; iow++) {
        uint16_t * dst_patch = dstb + (uint64_t) iow * patch_stride;
        for (uint32_t iic = 0; iic < IC; iic++) {
            for (uint32_t ikd = 0; ikd < KD; ikd++) {
                for (uint32_t ikh = 0; ikh < KH; ikh++) {
                    uint16_t * out_run = dst_patch +
                        ((uint64_t) iic * KD * KH + (uint64_t) ikd * KH + ikh) * KW;
                    const float * src_run = srcb +
                        (((uint64_t) iic * KD * KH + (uint64_t) ikd * KH + ikh) * IW) +
                        (uint64_t) iow * KW;
                    hvx_copy_f16_f32_uu((uint8_t *) out_run, (const uint8_t *) src_run, KW);
                }
            }
        }
    }
}

static __attribute__((noinline)) void im2col_3d_patchembed_compute_f32(
    const struct htp_im2col_context * ictx,
    float * restrict dstb, const float * restrict srcb) {
    const uint32_t OW           = ictx->OW;
    const uint32_t IC           = ictx->IC;
    const uint32_t KD           = ictx->KD;
    const uint32_t KH           = ictx->KH;
    const uint32_t KW           = ictx->KW;
    const uint32_t IW           = ictx->IW;
    const uint32_t patch_stride = ictx->patch_stride;

    for (uint32_t iow = 0; iow < OW; iow++) {
        float * dst_patch = dstb + (uint64_t) iow * patch_stride;
        for (uint32_t iic = 0; iic < IC; iic++) {
            for (uint32_t ikd = 0; ikd < KD; ikd++) {
                for (uint32_t ikh = 0; ikh < KH; ikh++) {
                    float * out_run = dst_patch +
                        ((uint64_t) iic * KD * KH + (uint64_t) ikd * KH + ikh) * KW;
                    const float * src_run = srcb +
                        (((uint64_t) iic * KD * KH + (uint64_t) ikd * KH + ikh) * IW) +
                        (uint64_t) iow * KW;
                    hvx_copy_f32_uu((uint8_t *) out_run, (const uint8_t *) src_run, KW);
                }
            }
        }
    }
}

#define IM2COL_3D_PATCHEMBED_DMA_BODY(FNAME, DST_CTYPE, COMPUTE_FN, DST_ELEM)                                \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                                     \
        (void) nth;                                                                                          \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                        \
        struct htp_ops_context *    octx        = ictx->octx;                                                \
        struct htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];                                    \
        const struct htp_tensor * restrict src1 = octx->src[1];                                              \
        const struct htp_tensor * restrict dst  = octx->dst;                                                 \
        const uint32_t IC           = ictx->IC;                                                              \
        const uint32_t ID = ictx->ID, IH = ictx->IH, IW = ictx->IW;                                          \
        const uint32_t KD = ictx->KD, KH = ictx->KH;                                                         \
        const uint32_t OD = ictx->OD, OH = ictx->OH, OW = ictx->OW;                                          \
        const uint32_t patch_stride = ictx->patch_stride;                                                    \
        const dma_addr_t src_data   = src1->data;                                                            \
        const dma_addr_t dst_data   = dst->data;                                                             \
        dma_queue * dma_q           = octx->ctx->dma[ith];                                                   \
        const uint32_t row_end_max  = ictx->pe_row_base + ictx->pe_nrows;                                    \
        const uint32_t row_start    = ictx->pe_row_base + ictx->pe_rows_per_thread * ith;                    \
        const uint32_t row_end      = MIN(row_start + ictx->pe_rows_per_thread, row_end_max);                \
        if (row_start >= row_end) {                                                                          \
            return;                                                                                          \
        }                                                                                                    \
        const uint32_t nrows_local  = row_end - row_start;                                                   \
        uint8_t * src_base          = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;                \
        uint8_t * dstb_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;                \
        float * srcb2[2]            = { (float *) src_base, (float *) (src_base + ictx->pe_src_row_bytes) }; \
        DST_CTYPE * dstb2[2]        = { (DST_CTYPE *) dstb_base,                                             \
                                         (DST_CTYPE *) (dstb_base + ictx->pe_dst_row_bytes) };               \
        uint32_t cur_ioh   = ictx->thread_ioh[ith];                                                          \
        uint32_t cur_iod   = ictx->thread_iod[ith];                                                          \
        uint32_t cur_in    = ictx->thread_in[ith];                                                           \
        uint32_t stage_ioh = cur_ioh;                                                                        \
        uint32_t stage_iod = cur_iod;                                                                        \
        uint32_t stage_in  = cur_in;                                                                         \
        uint32_t src_pushes[2] = {0, 0};                                                                     \
        for (uint32_t i = 0; i < nrows_local && i < 2; i++) {                                                \
            dma_queue_push(dma_q, dma_make_data(dst_data, dstb2[i]), 0, 0, 0, 0);                            \
            src_pushes[i] = im2col_3d_patchembed_dma_stage(dma_q, src_data, srcb2[i],                        \
                                                           stage_in, stage_iod, stage_ioh,                   \
                                                           IC, ID, IH, IW, KD, KH);                          \
            im2col_3d_step_row(&stage_in, &stage_iod, &stage_ioh, OH, OD);                                   \
        }                                                                                                    \
        for (uint32_t idx = 0; idx < nrows_local; idx++) {                                                   \
            const uint32_t buf = idx & 1u;                                                                   \
            float *     srcb = srcb2[buf];                                                                   \
            DST_CTYPE * dstb = dstb2[buf];                                                                   \
            const uint32_t r = row_start + idx;                                                              \
            dma_queue_pop(dma_q);                                                                            \
            for (uint32_t p = 0; p < src_pushes[buf]; p++) {                                                 \
                dma_queue_pop(dma_q);                                                                        \
            }                                                                                                \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);                                            \
            COMPUTE_FN(ictx, dstb, srcb);                                                                    \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);                                             \
            const dma_addr_t ddr_row = dst_data + (size_t) r * OW * patch_stride * (DST_ELEM);               \
            dma_queue_push(dma_q, dma_make_data(ddr_row, dstb),                                              \
                           OW * patch_stride * (DST_ELEM), OW * patch_stride * (DST_ELEM),                   \
                           OW * patch_stride * (DST_ELEM), 1);                                               \
            if (idx + 2 < nrows_local) {                                                                     \
                src_pushes[buf] = im2col_3d_patchembed_dma_stage(dma_q, src_data, srcb,                      \
                                                                stage_in, stage_iod, stage_ioh,              \
                                                                IC, ID, IH, IW, KD, KH);                     \
                im2col_3d_step_row(&stage_in, &stage_iod, &stage_ioh, OH, OD);                               \
            }                                                                                                \
            im2col_3d_step_row(&cur_in, &cur_iod, &cur_ioh, OH, OD);                                         \
        }                                                                                                    \
        dma_queue_flush(dma_q);                                                                              \
    }

IM2COL_3D_PATCHEMBED_DMA_BODY(im2col_3d_patchembed_dma_f16_thread, uint16_t, im2col_3d_patchembed_compute_f16, sizeof(__fp16))
IM2COL_3D_PATCHEMBED_DMA_BODY(im2col_3d_patchembed_dma_f32_thread, float,    im2col_3d_patchembed_compute_f32, sizeof(float))
static bool im2col_3d_use_patchembed_dma(const struct htp_ops_context * octx) {
    return octx->op_params[0] == (int32_t) octx->src[0]->ne[0] &&
           octx->op_params[1] == (int32_t) octx->src[0]->ne[1] &&
           octx->op_params[2] == (int32_t) octx->src[0]->ne[2] &&
           octx->op_params[3] == 0 && octx->op_params[4] == 0 && octx->op_params[5] == 0 &&
           octx->op_params[6] == 1 && octx->op_params[7] == 1 && octx->op_params[8] == 1;
}

static bool im2col_3d_mul_u32(uint64_t * value, uint32_t factor) {
    if (factor != 0 && *value > UINT64_MAX / factor) {
        return false;
    }
    *value *= factor;
    return true;
}

static bool im2col_3d_patchembed_dma_fits(struct htp_ops_context *    octx,
                                          struct htp_im2col_context * ictx,
                                          uint32_t                    n_threads) {
    uint64_t src_bytes = (uint32_t) octx->op_params[9];
    uint64_t dst_bytes = octx->dst->ne[1];
    if (!im2col_3d_mul_u32(&src_bytes, octx->src[0]->ne[2]) ||
        !im2col_3d_mul_u32(&src_bytes, octx->src[0]->ne[1]) ||
        !im2col_3d_mul_u32(&src_bytes, octx->src[1]->ne[0]) ||
        !im2col_3d_mul_u32(&src_bytes, sizeof(float)) ||
        !im2col_3d_mul_u32(&dst_bytes, octx->dst->ne[0]) ||
        !im2col_3d_mul_u32(&dst_bytes, octx->dst->type == HTP_TYPE_F16 ? sizeof(__fp16) : sizeof(float)) ||
        src_bytes > octx->ctx->vtcm_size || dst_bytes > octx->ctx->vtcm_size) {
        return false;
    }

    ictx->pe_src_row_bytes = hex_round_up((uint32_t) src_bytes, 256);
    ictx->pe_dst_row_bytes = hex_round_up((uint32_t) dst_bytes, 256);
    struct htp_im2col_vtcm_layout L;
    htp_im2col_vtcm_layout_build(&L, ictx->pe_src_row_bytes, ictx->pe_dst_row_bytes, n_threads);
    if (L.total_bytes > octx->ctx->vtcm_size) {
        return false;
    }

    uint8_t * const base = octx->ctx->vtcm_base;
    ictx->pe_vtcm_src = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src);
    ictx->pe_vtcm_dst = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    ictx->pe_src_size_per_thread = (uint32_t) L.src_bytes_per_thread;
    ictx->pe_dst_size_per_thread = (uint32_t) L.dst_bytes_per_thread;
    return true;
}

static bool im2col_3d_output_size(uint32_t input, uint32_t kernel, int32_t stride, int32_t padding,
                                  int32_t dilation, uint32_t * output) {
    if (stride <= 0 || padding < 0 || dilation <= 0 || kernel == 0) {
        return false;
    }

    const int64_t numerator = (int64_t) input + 2LL * padding - (int64_t) dilation * (kernel - 1) - 1;
    if (numerator < 0) {
        return false;
    }

    const uint64_t size = (uint64_t) (numerator / stride) + 1;
    if (size == 0 || size > UINT32_MAX) {
        return false;
    }

    *output = (uint32_t) size;
    return true;
}

static bool im2col_3d_blocked_dma_fits(struct htp_ops_context *    octx,
                                       struct htp_im2col_context * ictx,
                                       uint32_t                    n_threads) {
    const uint32_t s0 = (uint32_t) octx->op_params[0];
    const uint32_t d0 = (uint32_t) octx->op_params[6];
    const uint32_t IC = (uint32_t) octx->op_params[9];
    const uint32_t KD = octx->src[0]->ne[2], KH = octx->src[0]->ne[1], KW = octx->src[0]->ne[0];
    const uint32_t OW = octx->dst->ne[1], patch_stride = octx->dst->ne[0];
    const uint32_t dst_elem = octx->dst->type == HTP_TYPE_F16 ? sizeof(__fp16) : sizeof(float);

    for (uint32_t owb = MIN(OW, 256); owb >= 1; owb--) {
        uint64_t Wb = (uint64_t) (owb - 1) * s0 + (uint64_t) (KW - 1) * d0 + 1;
        uint64_t src_bytes = IC;
        uint64_t dst_bytes = owb;
        if (Wb > UINT32_MAX ||
            !im2col_3d_mul_u32(&src_bytes, KD) ||
            !im2col_3d_mul_u32(&src_bytes, KH) ||
            !im2col_3d_mul_u32(&src_bytes, (uint32_t) Wb) ||
            !im2col_3d_mul_u32(&src_bytes, sizeof(float)) ||
            !im2col_3d_mul_u32(&dst_bytes, patch_stride) ||
            !im2col_3d_mul_u32(&dst_bytes, dst_elem) ||
            src_bytes > octx->ctx->vtcm_size || dst_bytes > octx->ctx->vtcm_size) {
            if (owb == 1) {
                break;
            }
            continue;
        }

        const uint32_t src_row_bytes = hex_round_up((uint32_t) src_bytes, 256);
        const uint32_t dst_row_bytes = hex_round_up((uint32_t) dst_bytes, 256);
        struct htp_im2col_vtcm_layout L;
        htp_im2col_vtcm_layout_build(&L, src_row_bytes, dst_row_bytes, n_threads);
        if (L.total_bytes <= octx->ctx->vtcm_size) {
            uint8_t * const base = octx->ctx->vtcm_base;
            ictx->pe_owb                 = owb;
            ictx->pe_wb                  = (uint32_t) Wb;
            ictx->pe_nbpr                = (OW + owb - 1) / owb;
            ictx->pe_src_row_bytes       = src_row_bytes;
            ictx->pe_dst_row_bytes       = dst_row_bytes;
            ictx->pe_vtcm_src            = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src);
            ictx->pe_vtcm_dst            = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
            ictx->pe_src_size_per_thread = (uint32_t) L.src_bytes_per_thread;
            ictx->pe_dst_size_per_thread = (uint32_t) L.dst_bytes_per_thread;
            return true;
        }
        if (owb == 1) {
            break;
        }
    }
    return false;
}

int op_im2col_3d(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src1->type != HTP_TYPE_F32 || (dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_F32)) {
        FARF(ERROR, "im2col_3d: only (F32 image -> F16/F32 columns) supported");
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!htp_tensor_is_contiguous(src1, sizeof(float)) ||
        !htp_tensor_is_contiguous(dst, dst->type == HTP_TYPE_F16 ? sizeof(__fp16) : sizeof(float))) {
        return HTP_STATUS_NO_SUPPORT;
    }
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1], s2 = octx->op_params[2];
    const int32_t p0 = octx->op_params[3], p1 = octx->op_params[4], p2 = octx->op_params[5];
    const int32_t d0 = octx->op_params[6], d1 = octx->op_params[7], d2 = octx->op_params[8];
    const int32_t ic = octx->op_params[9];
    if (ic <= 0 || src1->ne[3] % (uint32_t) ic != 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t IC = (uint32_t) ic;
    const uint32_t N  = src1->ne[3] / IC;
    if (N == 0) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t OW, OH, OD;
    if (!im2col_3d_output_size(src1->ne[0], src0->ne[0], s0, p0, d0, &OW) ||
        !im2col_3d_output_size(src1->ne[1], src0->ne[1], s1, p1, d1, &OH) ||
        !im2col_3d_output_size(src1->ne[2], src0->ne[2], s2, p2, d2, &OD)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint64_t src_elements = 1;
    uint64_t patch_stride = IC;
    uint64_t total_rows = N;
    uint64_t dst_ne3 = N;
    if (!im2col_3d_mul_u32(&src_elements, src1->ne[0]) ||
        !im2col_3d_mul_u32(&src_elements, src1->ne[1]) ||
        !im2col_3d_mul_u32(&src_elements, src1->ne[2]) ||
        !im2col_3d_mul_u32(&src_elements, src1->ne[3]) ||
        !im2col_3d_mul_u32(&patch_stride, src0->ne[2]) ||
        !im2col_3d_mul_u32(&patch_stride, src0->ne[1]) ||
        !im2col_3d_mul_u32(&patch_stride, src0->ne[0]) ||
        !im2col_3d_mul_u32(&total_rows, OD) ||
        !im2col_3d_mul_u32(&total_rows, OH) ||
        !im2col_3d_mul_u32(&dst_ne3, OD)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    uint64_t total_patches = total_rows;
    if (!im2col_3d_mul_u32(&total_patches, OW)) {
        return HTP_STATUS_INVAL_PARAMS;
    }
    if (patch_stride > UINT32_MAX || total_rows > UINT32_MAX || total_patches > UINT32_MAX || dst_ne3 > UINT32_MAX ||
        dst->ne[0] != patch_stride || dst->ne[1] != OW || dst->ne[2] != OH || dst->ne[3] != dst_ne3) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (src_elements > src1->size / sizeof(float)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t dst_elem = dst->type == HTP_TYPE_F16 ? sizeof(__fp16) : sizeof(float);
    const uint64_t patch_bytes = patch_stride * dst_elem;
    if (patch_bytes > dst->size || total_patches > dst->size / patch_bytes) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t total_rows_u32 = (uint32_t) total_rows;
    uint32_t row_base = 0;
    uint32_t nrows    = total_rows_u32;
    if (octx->ctx->mdev.count > 1) {
        const uint32_t row_size = dst->nb[2];
        const uint32_t rows_per_chunk =
            (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;
        const struct htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(total_rows_u32, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_base = range.start;
        nrows    = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    struct htp_im2col_context ictx = { 0 };
    ictx.octx         = octx;
    ictx.pe_row_base  = row_base;
    ictx.pe_nrows     = nrows;
    ictx.N            = N;
    ictx.IC           = IC;
    ictx.ID           = src1->ne[2];
    ictx.IH           = src1->ne[1];
    ictx.IW           = src1->ne[0];
    ictx.KD           = src0->ne[2];
    ictx.KH           = src0->ne[1];
    ictx.KW           = src0->ne[0];
    ictx.OD           = OD;
    ictx.OH           = OH;
    ictx.OW           = OW;
    ictx.patch_stride = (uint32_t) patch_stride;

    const uint32_t pth = MIN(octx->n_threads, nrows);
    const uint32_t fit_threads = MIN(octx->n_threads, total_rows_u32);
    ictx.pe_rows_per_thread = (nrows + pth - 1) / pth;

    for (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t rem = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = rem % OH;
        rem /= OH;
        ictx.thread_iod[ith] = rem % OD;
        ictx.thread_in[ith]  = rem / OD;
    }

    const bool exact = im2col_3d_use_patchembed_dma(octx);
    if (exact && im2col_3d_patchembed_dma_fits(octx, &ictx, fit_threads)) {
        work_queue_run(octx->ctx->work_queue,
            dst->type == HTP_TYPE_F16 ? im2col_3d_patchembed_dma_f16_thread
                                      : im2col_3d_patchembed_dma_f32_thread, &ictx, pth);
        return HTP_STATUS_OK;
    }

    if (im2col_3d_blocked_dma_fits(octx, &ictx, fit_threads)) {
        work_queue_run(octx->ctx->work_queue,
            dst->type == HTP_TYPE_F16 ? im2col_3d_blocked_dma_f16_thread
                                      : im2col_3d_blocked_dma_f32_thread, &ictx, pth);
        return HTP_STATUS_OK;
    }

    return HTP_STATUS_NO_SUPPORT;
}

int op_im2col(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src1->type != HTP_TYPE_F32 || (dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_F32)) {
        FARF(ERROR, "im2col: only (F32 image -> F16/F32 columns) supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t  is_2D         = octx->op_params[6] == 1;
    const uint32_t N             = is_2D ? src1->ne[3] : src1->ne[2];
    const uint32_t IC            = is_2D ? src1->ne[2] : src1->ne[1];
    const uint32_t IH            = is_2D ? src1->ne[1] : 1;
    const uint32_t IW            = src1->ne[0];
    const uint32_t KH            = is_2D ? src0->ne[1] : 1;
    const uint32_t KW            = src0->ne[0];
    const uint32_t OH            = is_2D ? dst->ne[2] : 1;
    const uint32_t OW            = dst->ne[1];
    const uint32_t patch_stride = IC * KH * KW;
    const uint32_t total_rows    = N * OH;

    uint32_t row_base = 0;
    uint32_t nrows    = total_rows;
    if (octx->ctx->mdev.count > 1) {
        const uint32_t row_size = dst->nb[2];
        const uint32_t rows_per_chunk =
            (row_size > 0) ? (HEX_L2_LINE_SIZE / hex_gcd_u32(row_size, HEX_L2_LINE_SIZE)) : 1;
        const struct htp_tensor_mdev_range range =
            htp_tensor_mdev_partition(total_rows, htp_tensor_mdev_data_aligned(dst) ? rows_per_chunk : 0,
                                      octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_base = range.start;
        nrows    = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    struct htp_im2col_context ictx = { 0 };
    ictx.octx         = octx;
    ictx.pe_row_base  = row_base;
    ictx.pe_nrows     = nrows;
    ictx.N            = N;
    ictx.IC           = IC;
    ictx.IH           = IH;
    ictx.IW           = IW;
    ictx.KH           = KH;
    ictx.KW           = KW;
    ictx.OH           = OH;
    ictx.OW           = OW;
    ictx.patch_stride = patch_stride;

    const uint32_t pth = MIN(octx->n_threads, nrows);
    const uint32_t fit_threads = MIN(octx->n_threads, total_rows);
    ictx.pe_rows_per_thread = (nrows + pth - 1) / pth;

    for (uint32_t ith = 0; ith < pth; ith++) {
        uint32_t r = row_base + ictx.pe_rows_per_thread * ith;
        ictx.thread_ioh[ith] = r % OH;
        ictx.thread_in[ith]  = r / OH;
    }

    const bool exact = im2col_use_patchembed_dma(octx);
    if (exact && im2col_patchembed_dma_fits(octx, &ictx, fit_threads)) {
        work_queue_run(octx->ctx->work_queue,
            dst->type == HTP_TYPE_F16 ? im2col_patchembed_dma_f16_thread
                                      : im2col_patchembed_dma_f32_thread, &ictx, pth);
        return HTP_STATUS_OK;
    }

    if (im2col_blocked_dma_fits(octx, &ictx, fit_threads)) {
        work_queue_run(octx->ctx->work_queue,
            dst->type == HTP_TYPE_F16 ? im2col_blocked_dma_f16_thread
                                      : im2col_blocked_dma_f32_thread, &ictx, pth);
        return HTP_STATUS_OK;
    }

    return HTP_STATUS_NO_SUPPORT;
}
