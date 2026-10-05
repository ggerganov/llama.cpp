#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>
#include <stdint.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "conv2d-ops.h"
#include "dma-queue.h"
#include "hex-common.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"
#include "hvx-base.h"
#include "hvx-copy.h"
#include "hmx-queue.h"
#include "hmx-utils.h"

struct htp_conv2d_context {
    struct htp_ops_context *                    octx;
    const struct htp_conv2d_kernel_params *     kparams;
};

struct htp_conv2d_hmx_job {
    __fp16 *       output;
    const __fp16 * kernel_tiles;
    const __fp16 * patch_tiles;
    const __fp16 * scales;
    uint32_t       n_row_tiles;
    uint32_t       n_col_tiles;
    uint32_t       n_dot_tiles;
};

static void conv2d_hmx_worker(void * data) {
    struct htp_conv2d_hmx_job * job = (struct htp_conv2d_hmx_job *) data;

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int) job->scales));

    const size_t dot_stride = job->n_dot_tiles * HMX_FP16_TILE_N_ELMS;

    for (uint32_t r = 0; r < job->n_row_tiles; ++r) {
        const __fp16 * row_base = job->kernel_tiles + r * dot_stride;
        const __fp16 * col_base = job->patch_tiles;
        __fp16 * out_tile = job->output + r * job->n_col_tiles * HMX_FP16_TILE_N_ELMS;

        for (uint32_t c = 0; c < job->n_col_tiles; ++c) {
            const __fp16 * row_tiles = row_base;
            const __fp16 * col_tiles = col_base;
            asm volatile(HMX_CLRACC_F16());

            uint32_t rem = job->n_dot_tiles;
            while (rem > 0) {
                const uint32_t n_tiles = MIN(rem, 32);
                const uint32_t range = 2048u * n_tiles - 1;
                asm volatile(HMX_LOAD_MPY_DEEP_F16("%1", "%2", "%0") : : "r"(range), "r"(row_tiles), "r"(col_tiles));
                row_tiles += n_tiles * HMX_FP16_TILE_N_ELMS;
                col_tiles += n_tiles * HMX_FP16_TILE_N_ELMS;
                rem -= n_tiles;
            }

            asm volatile(HMX_STORE_AFTER_F16("%0", "%1") : : "r"(out_tile), "r"(0) : "memory");
            col_base += dot_stride;
            out_tile += HMX_FP16_TILE_N_ELMS;
        }
    }
}

static inline float conv2d_read_kernel(const struct htp_tensor * k, uint32_t kw, uint32_t kh, uint32_t cin, uint32_t cout) {
    const uint8_t * p = (const uint8_t *) (uintptr_t) k->data
        + (size_t) kw   * k->nb[0]
        + (size_t) kh   * k->nb[1]
        + (size_t) cin  * k->nb[2]
        + (size_t) cout * k->nb[3];

    if (k->type == HTP_TYPE_F16) {
        return (float) *(const __fp16 *) p;
    }

    return *(const float *) p;
}

static void conv2d_scalar_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_conv2d_context * ctx = (struct htp_conv2d_context *) data;
    struct htp_ops_context * octx = ctx->octx;

    const struct htp_tensor * kernel = octx->src[0];
    const struct htp_tensor * input  = octx->src[1];
    const struct htp_tensor * dst    = octx->dst;

    const uint32_t KW   = kernel->ne[0];
    const uint32_t KH   = kernel->ne[1];
    const uint32_t Cin  = kernel->ne[2];
    const uint32_t Cout = kernel->ne[3];
    const uint32_t OW   = dst->ne[0];
    const uint32_t OH   = dst->ne[1];
    const uint32_t N    = dst->ne[3];

    const int32_t s0 = octx->op_params[0];
    const int32_t s1 = octx->op_params[1];
    const int32_t p0 = octx->op_params[2];
    const int32_t p1 = octx->op_params[3];
    const int32_t d0 = octx->op_params[4];
    const int32_t d1 = octx->op_params[5];

    const uint32_t total_rows = ctx->kparams->total_rows;
    const uint32_t dr = (total_rows + nth - 1) / nth;
    const uint32_t r0 = dr * ith;
    const uint32_t r1 = MIN(r0 + dr, total_rows);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    for (uint32_t r = r0; r < r1; ++r) {
        const uint32_t oh   = r % OH;
        const uint32_t tmp0 = r / OH;
        const uint32_t cout = tmp0 % Cout;
        const uint32_t n    = tmp0 / Cout;

        for (uint32_t ow = 0; ow < OW; ++ow) {
            float sum = 0.0f;

            for (uint32_t kh = 0; kh < KH; ++kh) {
                const int32_t ih = (int32_t) oh * s1 + (int32_t) kh * d1 - p1;
                if (ih < 0 || ih >= (int32_t) input->ne[1]) {
                    continue;
                }

                for (uint32_t kw = 0; kw < KW; ++kw) {
                    const int32_t iw = (int32_t) ow * s0 + (int32_t) kw * d0 - p0;
                    if (iw < 0 || iw >= (int32_t) input->ne[0]) {
                        continue;
                    }

                    for (uint32_t cin = 0; cin < Cin; ++cin) {
                        const uint8_t * xp = (const uint8_t *) (uintptr_t) input->data
                            + (size_t) iw  * input->nb[0]
                            + (size_t) ih  * input->nb[1]
                            + (size_t) cin * input->nb[2]
                            + (size_t) n   * input->nb[3];

                        sum += conv2d_read_kernel(kernel, kw, kh, cin, cout) * *(const float *) xp;
                    }
                }
            }

            uint8_t * yp = (uint8_t *) (uintptr_t) dst->data
                + (size_t) ow   * dst->nb[0]
                + (size_t) oh   * dst->nb[1]
                + (size_t) cout * dst->nb[2]
                + (size_t) n    * dst->nb[3];
            *(float *) yp = sum;
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

static void conv2d_hvx_width_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_conv2d_context * ctx = (struct htp_conv2d_context *) data;
    struct htp_ops_context * octx = ctx->octx;

    const struct htp_tensor * kernel = octx->src[0];
    const struct htp_tensor * input  = octx->src[1];
    const struct htp_tensor * dst    = octx->dst;

    const uint32_t KW   = kernel->ne[0];
    const uint32_t KH   = kernel->ne[1];
    const uint32_t Cin  = kernel->ne[2];
    const uint32_t Cout = kernel->ne[3];
    const uint32_t OW   = dst->ne[0];
    const uint32_t OH   = dst->ne[1];
    const uint32_t N    = dst->ne[3];
    const int32_t p0 = octx->op_params[2];
    const int32_t p1 = octx->op_params[3];

    const uint32_t total_rows = ctx->kparams->total_rows;
    const uint32_t dr = (total_rows + nth - 1) / nth;
    const uint32_t r0 = dr * ith;
    const uint32_t r1 = MIN(r0 + dr, total_rows);
    const uint32_t k_elems = KW * KH * Cin;
    float wbuf_stack[1024];
    float xbuf_stack[VLEN_FP32] __attribute__((aligned(VLEN)));
    float * wbuf = wbuf_stack;
    float * xbuf = xbuf_stack;

    if (k_elems > 1024) {
        uint8_t * spad = octx->src0_spad.data + (size_t) ith * octx->src0_spad.size_per_thread;
        wbuf = (float *) spad;
        xbuf = (float *) (spad + hex_round_up((size_t) k_elems * sizeof(float), VLEN));
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    for (uint32_t r = r0; r < r1; ++r) {
        const uint32_t oh   = r % OH;
        const uint32_t tmp0 = r / OH;
        const uint32_t cout = tmp0 % Cout;
        const uint32_t n    = tmp0 / Cout;

        uint32_t k = 0;
        for (uint32_t kh = 0; kh < KH; ++kh) {
            for (uint32_t kw = 0; kw < KW; ++kw) {
                for (uint32_t cin = 0; cin < Cin; ++cin) {
                    wbuf[k++] = conv2d_read_kernel(kernel, kw, kh, cin, cout);
                }
            }
        }

        if (p0 == 0 && p1 == 0) {
            const uint32_t ow_vec_end = (OW / VLEN_FP32) * VLEN_FP32;

            for (uint32_t ow = 0; ow < ow_vec_end; ow += VLEN_FP32) {
                HVX_Vector acc = Q6_V_vzero();

                k = 0;
                for (uint32_t kh = 0; kh < KH; ++kh) {
                    const uint32_t ih = oh + kh;
                    for (uint32_t kw = 0; kw < KW; ++kw) {
                        const uint32_t iw = ow + kw;
                        for (uint32_t cin = 0; cin < Cin; ++cin) {
                            const uint8_t * xp = (const uint8_t *) (uintptr_t) input->data
                                + (size_t) iw  * input->nb[0]
                                + (size_t) ih  * input->nb[1]
                                + (size_t) cin * input->nb[2]
                                + (size_t) n   * input->nb[3];
                            const HVX_Vector x = hvx_vmemu(xp);
                            const HVX_Vector w = hvx_vec_splat_f32(wbuf[k++]);
                            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                        }
                    }
                }

                uint8_t * yp = (uint8_t *) (uintptr_t) dst->data
                    + (size_t) ow   * dst->nb[0]
                    + (size_t) oh   * dst->nb[1]
                    + (size_t) cout * dst->nb[2]
                    + (size_t) n    * dst->nb[3];
                hvx_vec_store_u(yp, VLEN_FP32 * sizeof(float), Q6_Vsf_equals_Vqf32(acc));
            }

            if (ow_vec_end < OW) {
                const uint32_t ow = ow_vec_end;
                const uint32_t n_tail = OW - ow_vec_end;
                HVX_Vector acc = Q6_V_vzero();

                k = 0;
                for (uint32_t kh = 0; kh < KH; ++kh) {
                    const uint32_t ih = oh + kh;
                    for (uint32_t kw = 0; kw < KW; ++kw) {
                        const uint32_t iw = ow + kw;
                        for (uint32_t cin = 0; cin < Cin; ++cin) {
                            const uint8_t * xp = (const uint8_t *) (uintptr_t) input->data
                                + (size_t) iw  * input->nb[0]
                                + (size_t) ih  * input->nb[1]
                                + (size_t) cin * input->nb[2]
                                + (size_t) n   * input->nb[3];
                            memset(xbuf, 0, VLEN);
                            memcpy(xbuf, xp, n_tail * sizeof(float));
                            const HVX_Vector x = *(const HVX_Vector *) xbuf;
                            const HVX_Vector w = hvx_vec_splat_f32(wbuf[k++]);
                            acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                        }
                    }
                }

                uint8_t * yp = (uint8_t *) (uintptr_t) dst->data
                    + (size_t) ow   * dst->nb[0]
                    + (size_t) oh   * dst->nb[1]
                    + (size_t) cout * dst->nb[2]
                    + (size_t) n    * dst->nb[3];
                hvx_vec_store_u(yp, n_tail * sizeof(float), Q6_Vsf_equals_Vqf32(acc));
            }
            continue;
        }

        uint32_t ow = 0;
        const uint32_t vector_start = (uint32_t) MAX(0, p0);
        const int32_t vector_end_i = MIN((int32_t) OW, (int32_t) input->ne[0] - (int32_t) KW + p0 + 1);
        const uint32_t vector_end = (uint32_t) MAX(0, vector_end_i);

        for (; ow < MIN(vector_start, OW); ++ow) {
            float sum = 0.0f;

            k = 0;
            for (uint32_t kh = 0; kh < KH; ++kh) {
                const int32_t ih = (int32_t) oh + (int32_t) kh - p1;
                if (ih < 0 || ih >= (int32_t) input->ne[1]) {
                    k += KW * Cin;
                    continue;
                }

                for (uint32_t kw = 0; kw < KW; ++kw) {
                    const int32_t iw = (int32_t) ow + (int32_t) kw - p0;
                    if (iw < 0 || iw >= (int32_t) input->ne[0]) {
                        k += Cin;
                        continue;
                    }

                    for (uint32_t cin = 0; cin < Cin; ++cin) {
                        const uint8_t * xp = (const uint8_t *) (uintptr_t) input->data
                            + (size_t) iw  * input->nb[0]
                            + (size_t) ih  * input->nb[1]
                            + (size_t) cin * input->nb[2]
                            + (size_t) n   * input->nb[3];
                        sum += wbuf[k++] * *(const float *) xp;
                    }
                }
            }

            uint8_t * yp = (uint8_t *) (uintptr_t) dst->data
                + (size_t) ow   * dst->nb[0]
                + (size_t) oh   * dst->nb[1]
                + (size_t) cout * dst->nb[2]
                + (size_t) n    * dst->nb[3];
            *(float *) yp = sum;
        }

        for (; ow + VLEN_FP32 <= vector_end; ow += VLEN_FP32) {
            HVX_Vector acc = Q6_V_vzero();

            k = 0;
            for (uint32_t kh = 0; kh < KH; ++kh) {
                const int32_t ih = (int32_t) oh + (int32_t) kh - p1;
                if (ih < 0 || ih >= (int32_t) input->ne[1]) {
                    k += KW * Cin;
                    continue;
                }

                for (uint32_t kw = 0; kw < KW; ++kw) {
                    const int32_t iw = (int32_t) ow + (int32_t) kw - p0;
                    for (uint32_t cin = 0; cin < Cin; ++cin) {
                        const uint8_t * xp = (const uint8_t *) (uintptr_t) input->data
                            + (size_t) iw  * input->nb[0]
                            + (size_t) ih  * input->nb[1]
                            + (size_t) cin * input->nb[2]
                            + (size_t) n   * input->nb[3];
                        const HVX_Vector x = hvx_vmemu(xp);
                        const HVX_Vector w = hvx_vec_splat_f32(wbuf[k++]);
                        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                }
            }

            uint8_t * yp = (uint8_t *) (uintptr_t) dst->data
                + (size_t) ow   * dst->nb[0]
                + (size_t) oh   * dst->nb[1]
                + (size_t) cout * dst->nb[2]
                + (size_t) n    * dst->nb[3];
            hvx_vec_store_u(yp, VLEN_FP32 * sizeof(float), Q6_Vsf_equals_Vqf32(acc));
        }

        if (ow < vector_end) {
            const uint32_t n_tail = vector_end - ow;
            HVX_Vector acc = Q6_V_vzero();

            k = 0;
            for (uint32_t kh = 0; kh < KH; ++kh) {
                const int32_t ih = (int32_t) oh + (int32_t) kh - p1;
                if (ih < 0 || ih >= (int32_t) input->ne[1]) {
                    k += KW * Cin;
                    continue;
                }

                for (uint32_t kw = 0; kw < KW; ++kw) {
                    const int32_t iw = (int32_t) ow + (int32_t) kw - p0;
                    for (uint32_t cin = 0; cin < Cin; ++cin) {
                        const uint8_t * xp = (const uint8_t *) (uintptr_t) input->data
                            + (size_t) iw  * input->nb[0]
                            + (size_t) ih  * input->nb[1]
                            + (size_t) cin * input->nb[2]
                            + (size_t) n   * input->nb[3];
                        memset(xbuf, 0, VLEN);
                        memcpy(xbuf, xp, n_tail * sizeof(float));
                        const HVX_Vector x = *(const HVX_Vector *) xbuf;
                        const HVX_Vector w = hvx_vec_splat_f32(wbuf[k++]);
                        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(x, w));
                    }
                }
            }

            uint8_t * yp = (uint8_t *) (uintptr_t) dst->data
                + (size_t) ow   * dst->nb[0]
                + (size_t) oh   * dst->nb[1]
                + (size_t) cout * dst->nb[2]
                + (size_t) n    * dst->nb[3];
            hvx_vec_store_u(yp, n_tail * sizeof(float), Q6_Vsf_equals_Vqf32(acc));
            ow += n_tail;
        }

        for (; ow < OW; ++ow) {
            float sum = 0.0f;

            k = 0;
            for (uint32_t kh = 0; kh < KH; ++kh) {
                const int32_t ih = (int32_t) oh + (int32_t) kh - p1;
                if (ih < 0 || ih >= (int32_t) input->ne[1]) {
                    k += KW * Cin;
                    continue;
                }

                for (uint32_t kw = 0; kw < KW; ++kw) {
                    const int32_t iw = (int32_t) ow + (int32_t) kw - p0;
                    if (iw < 0 || iw >= (int32_t) input->ne[0]) {
                        k += Cin;
                        continue;
                    }

                    for (uint32_t cin = 0; cin < Cin; ++cin) {
                        const uint8_t * xp = (const uint8_t *) (uintptr_t) input->data
                            + (size_t) iw  * input->nb[0]
                            + (size_t) ih  * input->nb[1]
                            + (size_t) cin * input->nb[2]
                            + (size_t) n   * input->nb[3];
                        sum += wbuf[k++] * *(const float *) xp;
                    }
                }
            }

            uint8_t * yp = (uint8_t *) (uintptr_t) dst->data
                + (size_t) ow   * dst->nb[0]
                + (size_t) oh   * dst->nb[1]
                + (size_t) cout * dst->nb[2]
                + (size_t) n    * dst->nb[3];
            *(float *) yp = sum;
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

struct htp_conv2d_hmx_context {
    struct htp_ops_context *                octx;
    const struct htp_conv2d_kernel_params * kp;
    __fp16 *                                w_tiles;
    __fp16 *                                act[2];
    __fp16 *                                out[2];
    uint8_t *                               thr_base;
    uint32_t *                              w2;
    uint32_t                                t_base;
    uint32_t                                t_end;
    uint32_t                                rt0;
    uint32_t                                n_rt;
    int32_t                                 pack_chunk;
    int32_t                                 unpack_chunk;
    uint32_t                                pack_tpt;
    uint32_t                                unpack_tpt;
};

// DMA hangs if a stride is less than the row size, even for a single row
static inline bool conv2d_dma_push(dma_queue * q, dma_data d, size_t dst_stride, size_t src_stride, size_t row, size_t nrows) {
    if (nrows == 1) {
        dst_stride = src_stride = row;
    }
    return dma_queue_push(q, d, dst_stride, src_stride, row, nrows);
}

// col tiles [t0, t1) of chunk c for thread ith, returns first col tile of the chunk
static inline uint32_t conv2d_hmx_chunk_range(const struct htp_conv2d_hmx_context * hc, uint32_t c, uint32_t tpt,
                                              uint32_t ith, uint32_t * t0, uint32_t * t1) {
    const uint32_t cs = hc->t_base + c * hc->kp->ct_chunk;
    const uint32_t ce = MIN(cs + hc->kp->ct_chunk, hc->t_end);
    *t0 = MIN(cs + ith * tpt, ce);
    *t1 = MIN(*t0 + tpt, ce);
    return cs;
}

static void conv2d_hmx_wpack_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_hmx_context * hc = (const struct htp_conv2d_hmx_context *) data;
    struct htp_ops_context *                octx = hc->octx;
    const struct htp_conv2d_kernel_params * kp   = hc->kp;
    const struct htp_tensor *               kernel = octx->src[0];

    const uint32_t Cout   = kernel->ne[3];
    const uint32_t K      = kp->k_size;
    const uint32_t k_pad  = kp->n_k_tiles * HMX_FP16_TILE_N_COLS;
    const uint32_t WK     = kp->wk_chunk;
    const bool     is_f16 = kernel->type == HTP_TYPE_F16;
    const uint32_t es     = is_f16 ? sizeof(__fp16) : sizeof(float);

    dma_queue * dmaq = octx->ctx->dma[ith];
    uint8_t *   thr  = hc->thr_base + (size_t) ith * kp->thr_size;
    __fp16 *    wbuf = (__fp16 *) thr;
    float *     wraw = (float *) (thr + (size_t) WK * 32 * sizeof(__fp16));
    uint8_t *   wdma = is_f16 ? (uint8_t *) wbuf : (uint8_t *) wraw;

    const HVX_Vector zero = Q6_V_vzero();

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ith);

    for (uint32_t rt = ith; rt < hc->n_rt; rt += nth) {
        const uint32_t co0 = (hc->rt0 + rt) * 32;
        const uint32_t nco = MIN(32, Cout - co0);

        for (uint32_t kc = 0; kc < k_pad; kc += WK) {
            const uint32_t kv = MIN(WK, K - kc);
            const uint32_t hd = MIN(WK, k_pad - kc);

            conv2d_dma_push(dmaq, dma_make_data(wdma, kernel->data + (size_t) co0 * kernel->nb[3] + (size_t) kc * es),
                           WK * es, kernel->nb[3], kv * es, nco);
            dma_queue_flush(dmaq);

            for (uint32_t r = 0; r < 32; r++) {
                HVX_Vector *       dv = (HVX_Vector *) (wbuf + (size_t) r * WK);
                const HVX_Vector * sv = (const HVX_Vector *) (wraw + (size_t) r * WK);
                for (uint32_t j = 0; j < hd; j += 64) {
                    HVX_Vector v = zero;
                    if (r < nco && j < kv) {
                        v = is_f16 ? dv[j / 64] : hvx_vec_f32_to_f16(sv[j / 32], sv[j / 32 + 1]);
                        if (kv - j < 64) {
                            v = Q6_V_vmux_QVV(Q6_Q_vsetq_R((kv - j) * sizeof(__fp16)), v, zero);
                        }
                    }
                    dv[j / 64] = v;
                }
            }

            __fp16 * tiles = hc->w_tiles + ((size_t) rt * kp->n_k_tiles + kc / 32) * HMX_FP16_TILE_N_ELMS;
            hmx_interleave_cols_to_tiles(tiles, wbuf, 32, hd, WK, 1, 0, 32);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ith);
}

// Stage Cin*KH input rows for G col tiles: stage[(cin*KH + kh)*SR + j] = x[n][cin][ih(kh)][ow0 - p0 + j], zero outside
static void conv2d_hmx_stage(const struct htp_ops_context * octx, const struct htp_conv2d_kernel_params * kp,
                             dma_queue * dmaq, float * stage, uint32_t n, uint32_t oh, uint32_t ow0, uint32_t G) {
    const struct htp_tensor * kernel = octx->src[0];
    const struct htp_tensor * input  = octx->src[1];

    const uint32_t KW  = kernel->ne[0];
    const uint32_t KH  = kernel->ne[1];
    const uint32_t Cin = kernel->ne[2];
    const int32_t  IW  = input->ne[0];
    const int32_t  IH  = input->ne[1];
    const int32_t  s1  = octx->op_params[1];
    const int32_t  p0  = octx->op_params[2];
    const int32_t  p1  = octx->op_params[3];
    const int32_t  d0  = octx->op_params[4];
    const int32_t  d1  = octx->op_params[5];
    const uint32_t SR  = kp->stage_row;

    const int32_t Wb  = 32 * G + (KW - 1) * d0;
    const int32_t iw0 = (int32_t) ow0 - p0;
    const int32_t lo  = MAX(0, -iw0);
    const int32_t hi  = MIN(Wb, IW - iw0);

    for (uint32_t kh = 0; kh < KH; kh++) {
        const int32_t ih   = (int32_t) oh * s1 + (int32_t) kh * d1 - p1;
        float *       rows = stage + (size_t) kh * SR;

        if (ih < 0 || ih >= IH || hi <= lo) {
            for (uint32_t c = 0; c < Cin; c++) {
                hvx_splat_f32_u(rows + (size_t) c * KH * SR, 0.0f, Wb);
            }
            continue;
        }

        const dma_addr_t src = input->data + (size_t) n * input->nb[3] + (size_t) ih * input->nb[1] + (size_t) (iw0 + lo) * sizeof(float);
        while (!conv2d_dma_push(dmaq, dma_make_data(rows + lo, src), (size_t) KH * SR * sizeof(float), input->nb[2],
                               (size_t) (hi - lo) * sizeof(float), Cin)) {
            dma_queue_pop(dmaq);
        }

        if (lo > 0 || hi < Wb) {
            for (uint32_t c = 0; c < Cin; c++) {
                float * row = rows + (size_t) c * KH * SR;
                if (lo > 0) {
                    hvx_splat_f32_u(row, 0.0f, lo);
                }
                if (hi < Wb) {
                    hvx_splat_f32_u(row + hi, 0.0f, Wb - hi);
                }
            }
        }
    }
}

// One weight-side tile column: vector i holds k rows 2i and 2i+1 interleaved over 32 pixels
static inline void conv2d_hmx_pack_tile(__fp16 * restrict dst, const float * restrict src, uint32_t K, uint32_t KW,
                                        uint32_t d0, uint32_t SR, uint32_t n_vec) {
    HVX_Vector * restrict out  = (HVX_Vector *) dst;
    const HVX_Vector      zero = Q6_V_vzero();
    const uint32_t        wrap = SR - KW * d0;

    const float * p  = src;
    uint32_t      kw = 0;
    uint32_t      i  = 0;

    for (; i < K / 2; i++) {
        const HVX_Vector a = hvx_vmemu(p);
        p += d0;
        if (++kw == KW) {
            kw = 0;
            p += wrap;
        }
        const HVX_Vector b = hvx_vmemu(p);
        p += d0;
        if (++kw == KW) {
            kw = 0;
            p += wrap;
        }
        out[i] = hvx_vec_f32_to_f16_shuff(a, b);
    }
    if (K & 1) {
        out[i++] = hvx_vec_f32_to_f16_shuff(hvx_vmemu(p), zero);
    }
    for (; i < n_vec; i++) {
        out[i] = zero;
    }
}

// CWHN: stage16[kh*SR + j*Cin + cin] = fp16(x[n][ih(kh)][ow0 - p0 + j][cin]), zero outside
static void conv2d_hmx_stage_cwhn(const struct htp_ops_context * octx, const struct htp_conv2d_kernel_params * kp,
                                  dma_queue * dmaq, float * stage, uint32_t n, uint32_t oh, uint32_t ow0, uint32_t G) {
    const struct htp_tensor * kernel = octx->src[0];
    const struct htp_tensor * input  = octx->src[1];

    const uint32_t KW  = kernel->ne[0];
    const uint32_t KH  = kernel->ne[1];
    const uint32_t Cin = kernel->ne[2];
    const int32_t  IW  = input->ne[0];
    const int32_t  IH  = input->ne[1];
    const int32_t  s1  = octx->op_params[1];
    const int32_t  p0  = octx->op_params[2];
    const int32_t  p1  = octx->op_params[3];
    const int32_t  d0  = octx->op_params[4];
    const int32_t  d1  = octx->op_params[5];
    const uint32_t SR  = kp->stage_row;

    const int32_t Wb  = 32 * G + (KW - 1) * d0;
    const int32_t iw0 = (int32_t) ow0 - p0;
    const int32_t lo  = MAX(0, -iw0);
    const int32_t hi  = MIN(Wb, IW - iw0);

    for (uint32_t kh = 0; kh < KH; kh++) {
        const int32_t ih  = (int32_t) oh * s1 + (int32_t) kh * d1 - p1;
        float *       row = stage + (size_t) kh * SR;

        if (ih < 0 || ih >= IH || hi <= lo) {
            hvx_splat_f32_u(row, 0.0f, Wb * Cin);
            continue;
        }

        const dma_addr_t src  = input->data + (size_t) n * input->nb[3] + (size_t) ih * input->nb[1] + (size_t) (iw0 + lo) * input->nb[0];
        const size_t     size = (size_t) (hi - lo) * Cin * sizeof(float);
        while (!dma_queue_push(dmaq, dma_make_data(row + lo * Cin, src), size, size, size, 1)) {
            dma_queue_pop(dmaq);
        }

        if (lo > 0) {
            hvx_splat_f32_u(row, 0.0f, lo * Cin);
        }
        if (hi < Wb) {
            hvx_splat_f32_u(row + hi * Cin, 0.0f, (Wb - hi) * Cin);
        }
    }
}

static void conv2d_hmx_dma_wait(struct htp_thread_trace * tr, dma_queue * dmaq) {
    htp_trace_event_start(tr, HTP_TRACE_EVT_DMA, 0);
    dma_queue_flush(dmaq);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_DMA, 0);
}

static void conv2d_hmx_cwhn_f16(__fp16 * restrict dst, const float * restrict src, uint32_t KH, uint32_t SR, uint32_t n) {
    for (uint32_t kh = 0; kh < KH; kh++) {
        hvx_copy_f16_f32_aa((uint8_t *) (dst + (size_t) kh * SR), (const uint8_t *) (src + (size_t) kh * SR), n);
    }
}

static uint32_t conv2d_hmx_issue_seg(const struct htp_ops_context * octx, const struct htp_conv2d_kernel_params * kp,
                                     dma_queue * dmaq, float * stage, uint32_t t, uint32_t t1) {
    const uint32_t OH    = octx->dst->ne[1];
    const uint32_t n_owt = kp->n_ow_tiles;
    const uint32_t row   = fastdiv(t, &kp->div_ow_tiles);
    const uint32_t owt   = t - row * n_owt;
    const uint32_t G     = MIN(MIN(kp->seg_tiles, t1 - t), n_owt - owt);
    const uint32_t n     = fastdiv(row, &kp->div_oh);
    const uint32_t oh    = row - n * OH;

    if (kp->cwhn) {
        conv2d_hmx_stage_cwhn(octx, kp, dmaq, stage, n, oh, owt * 32, G);
    } else {
        conv2d_hmx_stage(octx, kp, dmaq, stage, n, oh, owt * 32, G);
    }
    return G;
}

static inline void conv2d_vshuff_w(HVX_Vector * a, HVX_Vector * b) {
    const HVX_VectorPair p = Q6_W_vshuff_VVR(*b, *a, -4);
    *a = Q6_V_lo_W(p);
    *b = Q6_V_hi_W(p);
}

static inline void conv2d_transpose_finish(HVX_Vector * restrict scratch, HVX_Vector * restrict lo, HVX_Vector * restrict hi);

// 32x32 word transpose: out[i] word j = in[j * in_stride] word i; out 0..15 -> lo, 16..31 -> hi
static inline void conv2d_transpose_w32(const HVX_Vector * restrict in, uint32_t in_stride, HVX_Vector * restrict scratch,
                                        HVX_Vector * restrict lo, HVX_Vector * restrict hi) {
    for (uint32_t i = 0; i < 8; i++) {
        HVX_Vector v0 = in[(i +  0) * in_stride];
        HVX_Vector v1 = in[(i +  8) * in_stride];
        HVX_Vector v2 = in[(i + 16) * in_stride];
        HVX_Vector v3 = in[(i + 24) * in_stride];
        conv2d_vshuff_w(&v0, &v2);
        conv2d_vshuff_w(&v1, &v3);
        conv2d_vshuff_w(&v0, &v1);
        conv2d_vshuff_w(&v2, &v3);
        scratch[i +  0] = v0;
        scratch[i +  8] = v1;
        scratch[i + 16] = v2;
        scratch[i + 24] = v3;
    }
    conv2d_transpose_finish(scratch, lo, hi);
}

static inline void conv2d_transpose_finish(HVX_Vector * restrict scratch, HVX_Vector * restrict lo, HVX_Vector * restrict hi) {
    for (uint32_t g = 0; g < 4; g++) {
        HVX_Vector w0 = scratch[8 * g + 0];
        HVX_Vector w1 = scratch[8 * g + 1];
        HVX_Vector w2 = scratch[8 * g + 2];
        HVX_Vector w3 = scratch[8 * g + 3];
        HVX_Vector w4 = scratch[8 * g + 4];
        HVX_Vector w5 = scratch[8 * g + 5];
        HVX_Vector w6 = scratch[8 * g + 6];
        HVX_Vector w7 = scratch[8 * g + 7];
        conv2d_vshuff_w(&w0, &w4);
        conv2d_vshuff_w(&w1, &w5);
        conv2d_vshuff_w(&w2, &w6);
        conv2d_vshuff_w(&w3, &w7);
        conv2d_vshuff_w(&w0, &w2);
        conv2d_vshuff_w(&w1, &w3);
        conv2d_vshuff_w(&w4, &w6);
        conv2d_vshuff_w(&w5, &w7);
        conv2d_vshuff_w(&w0, &w1);
        conv2d_vshuff_w(&w2, &w3);
        conv2d_vshuff_w(&w4, &w5);
        conv2d_vshuff_w(&w6, &w7);
        HVX_Vector * o = (g < 2 ? lo : hi) + 8 * (g & 1);
        o[0] = w0;
        o[1] = w1;
        o[2] = w2;
        o[3] = w3;
        o[4] = w4;
        o[5] = w5;
        o[6] = w6;
        o[7] = w7;
    }
}

// Same transpose, but pixel j is base + j * stride. Skips the vin store/reload.
static inline void conv2d_transpose_strided(const __fp16 * restrict base, uint32_t stride, HVX_Vector * restrict scratch,
                                            HVX_Vector * restrict lo, HVX_Vector * restrict hi) {
    for (uint32_t i = 0; i < 8; i++) {
        HVX_Vector v0 = hvx_vmemu(base + (i +  0) * stride);
        HVX_Vector v1 = hvx_vmemu(base + (i +  8) * stride);
        HVX_Vector v2 = hvx_vmemu(base + (i + 16) * stride);
        HVX_Vector v3 = hvx_vmemu(base + (i + 24) * stride);
        conv2d_vshuff_w(&v0, &v2);
        conv2d_vshuff_w(&v1, &v3);
        conv2d_vshuff_w(&v0, &v1);
        conv2d_vshuff_w(&v2, &v3);
        scratch[i +  0] = v0;
        scratch[i +  8] = v1;
        scratch[i + 16] = v2;
        scratch[i + 24] = v3;
    }
    conv2d_transpose_finish(scratch, lo, hi);
}

// CWHN: k is n_groups groups of k_group contiguous values (per kh, or per (kh, kw) if d0 > 1), each padded to k_group_pad.
// A full 64-wide block is one vector per pixel and is transposed in place. A partial block is two masked loads.
static void conv2d_hmx_pack_tile_cwhn(__fp16 * restrict dst, const __fp16 * restrict src, HVX_Vector * restrict vin,
                                      HVX_Vector * restrict scratch, const struct htp_conv2d_kernel_params * kp,
                                      uint32_t gpr, uint32_t Cin, uint32_t d0) {
    const uint32_t k_pad = kp->n_k_tiles * HMX_FP16_TILE_N_COLS;
    const uint32_t KG    = kp->k_group;
    const uint32_t KGp   = kp->k_group_pad;
    const uint32_t NG    = kp->n_groups;
    const uint32_t SR    = kp->stage_row;

    HVX_Vector * restrict out  = (HVX_Vector *) dst;
    const HVX_VectorPred  q64  = Q6_Q_vsetq_R(64);

    uint32_t g    = 0;
    uint32_t gw   = 0;
    uint32_t r    = 0;
    size_t   roff = 0;
    size_t   goff = 0;

    for (uint32_t b = 0; b < k_pad / 64; b++) {
        const __fp16 * pa[2];
        uint32_t       va[2];
        for (uint32_t h = 0; h < 2; h++) {
            va[h] = g < NG && r < KG ? MIN(32, KG - r) : 0;
            pa[h] = g < NG ? src + goff + r - 32 * h : src;
            r += 32;
            if (r >= KGp) {
                r = 0;
                g++;
                if (++gw == gpr) {
                    gw   = 0;
                    roff += SR;
                    goff = roff;
                } else {
                    goff += d0 * Cin;
                }
            }
        }

        if (va[0] == 32 && va[1] == 32 && pa[0] == pa[1]) {
            conv2d_transpose_strided(pa[0], Cin, scratch, out + 32 * b, out + 32 * b + 16);
            continue;
        } else {
            const HVX_VectorPred q0 = Q6_Q_vsetq_R(va[0] * sizeof(__fp16));
            const HVX_VectorPred q1 = Q6_Q_and_QQn(Q6_Q_vsetq2_R(64 + va[1] * sizeof(__fp16)), q64);
            for (uint32_t j = 0; j < 32; j++) {
                const HVX_Vector v0 = hvx_vmemu(pa[0] + j * Cin);
                const HVX_Vector v1 = hvx_vmemu(pa[1] + j * Cin);
                vin[j] = Q6_V_vmux_QVV(q0, v0, Q6_V_vand_QV(q1, v1));
            }
        }

        conv2d_transpose_w32(vin, 1, scratch, out + 32 * b, out + 32 * b + 16);
    }
}

// Kernel stored as [k][cout] (cout contiguous): DMA k rows of 64 cout, transpose into two row tiles
static void conv2d_hmx_wpack_kmajor_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_hmx_context * hc = (const struct htp_conv2d_hmx_context *) data;
    struct htp_ops_context *                octx = hc->octx;
    const struct htp_conv2d_kernel_params * kp   = hc->kp;
    const struct htp_tensor *               kernel = octx->src[0];

    const uint32_t Cout   = kernel->ne[3];
    const uint32_t k_pad  = kp->n_k_tiles * HMX_FP16_TILE_N_COLS;
    const uint32_t WK     = kp->wk_chunk;
    const uint32_t KG     = kp->k_group;
    const uint32_t KGp    = kp->k_group_pad;
    const uint32_t NG     = kp->n_groups;
    const size_t   ks     = kp->w_kstride;
    const bool     is_f16 = kernel->type == HTP_TYPE_F16;
    const uint32_t es     = is_f16 ? sizeof(__fp16) : sizeof(float);
    const uint32_t n_cb   = (hc->n_rt + 1) / 2;

    dma_queue *  dmaq    = octx->ctx->dma[ith];
    uint8_t *    wraw    = hc->thr_base + (size_t) ith * kp->thr_size;
    HVX_Vector * vin     = (HVX_Vector *) (wraw + (size_t) WK * 64 * sizeof(float));
    HVX_Vector * scratch = vin + 32;
    HVX_Vector * sink    = scratch + 32;

    const HVX_Vector zero = Q6_V_vzero();

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ith);

    uint32_t owner = 0;
    for (uint32_t cb = 0; cb < n_cb; cb++) {
        for (uint32_t g = 0; g < NG; g++) {
            const uint32_t span = g + 1 == NG ? k_pad - g * KGp : KGp;
            for (uint32_t kc = 0; kc < span; kc += WK) {
                const bool mine = owner == ith;
                owner = owner + 1 == nth ? 0 : owner + 1;
                if (!mine) {
                    continue;
                }

                const uint32_t rt  = 2 * cb;
                const uint32_t co0 = (hc->rt0 + rt) * 32;
                const uint32_t nco = MIN(64, Cout - co0);
                const uint32_t nk  = kc < KG ? MIN(WK, KG - kc) : 0;
                const uint32_t hk  = MIN(WK, span - kc);

                if (nk > 0) {
                    conv2d_dma_push(dmaq, dma_make_data(wraw, kernel->data + ((size_t) g * KG + kc) * ks + (size_t) co0 * es),
                                    64 * es, ks, nco * es, nk);
                    dma_queue_flush(dmaq);
                }

                for (uint32_t kt = 0; kt < hk / 32; kt++) {
                    for (uint32_t j = 0; j < 32; j++) {
                        const uint32_t k = kt * 32 + j;
                        HVX_Vector     v = zero;
                        if (k < nk) {
                            const HVX_Vector * sv = (const HVX_Vector *) (wraw + (size_t) k * 64 * es);
                            v = is_f16 ? sv[0] : hvx_vec_f32_to_f16(sv[0], sv[1]);
                            if (nco < 64) {
                                v = Q6_V_vmux_QVV(Q6_Q_vsetq_R(nco * sizeof(__fp16)), v, zero);
                            }
                        }
                        vin[j] = v;
                    }

                    const uint32_t ktg = (g * KGp + kc) / 32 + kt;
                    HVX_Vector *   lo  = (HVX_Vector *) (hc->w_tiles + ((size_t) rt * kp->n_k_tiles + ktg) * HMX_FP16_TILE_N_ELMS);
                    HVX_Vector *   hi  = rt + 1 < hc->n_rt ?
                        (HVX_Vector *) (hc->w_tiles + ((size_t) (rt + 1) * kp->n_k_tiles + ktg) * HMX_FP16_TILE_N_ELMS) : sink;
                    conv2d_transpose_w32(vin, 1, scratch, lo, hi);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_PREP, (uint16_t) ith);
}

static void conv2d_hmx_pack(const struct htp_conv2d_hmx_context * hc, uint32_t ith) {
    struct htp_ops_context *                octx = hc->octx;
    const struct htp_conv2d_kernel_params * kp   = hc->kp;
    const struct htp_tensor *               kernel = octx->src[0];

    const uint32_t KW    = kernel->ne[0];
    const uint32_t KH    = kernel->ne[1];
    const uint32_t Cin   = kernel->ne[2];
    const uint32_t d0    = octx->op_params[4];
    const uint32_t n_owt = kp->n_ow_tiles;
    const uint32_t n_vec = kp->n_k_tiles * (HMX_FP16_TILE_N_ELMS / 64);

    dma_queue * dmaq  = octx->ctx->dma[ith];
    uint8_t *   thr   = hc->thr_base + (size_t) ith * kp->thr_size;
    float *     stages[2] = {
        (float *) thr,
        (float *) (thr + kp->stage_size),
    };
    __fp16 *    act   = hc->act[hc->pack_chunk & 1];

    // CWHN: fp16 stage (+256 bytes of over-read slack), then 32 + 32 vectors for the transpose
    const uint32_t n_stage = kp->n_stage > 1 ? 2 : 1;
    __fp16 *     stage16 = (__fp16 *) (thr + (size_t) n_stage * kp->stage_size + (size_t) 2 * 32 * kp->seg_tiles * 32 * sizeof(float));
    HVX_Vector * vin     = (HVX_Vector *) ((uint8_t *) stage16 + kp->stage_size / 2 + 256);
    HVX_Vector * scratch = vin + 32;
    const uint32_t gpr   = kp->n_groups == KH ? 1 : KW;

    uint32_t       t0, t1;
    const uint32_t cs = conv2d_hmx_chunk_range(hc, hc->pack_chunk, hc->pack_tpt, ith, &t0, &t1);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) ith);

    uint32_t t  = t0;
    uint32_t bi = 0;
    bool     ready = false;
    while (t < t1) {
        const uint32_t row = fastdiv(t, &kp->div_ow_tiles);
        const uint32_t owt = t - row * n_owt;
        const uint32_t G   = MIN(MIN(kp->seg_tiles, t1 - t), n_owt - owt);
        float *        stage = stages[bi];

        if (!ready) {
            conv2d_hmx_issue_seg(octx, kp, dmaq, stage, t, t1);
            conv2d_hmx_dma_wait(tr, dmaq);
        }

        // n_stage 2: DMA the next segment while HVX packs this one
        const uint32_t tnext = t + G;
        if (n_stage > 1 && tnext < t1) {
            conv2d_hmx_issue_seg(octx, kp, dmaq, stages[bi ^ 1], tnext, t1);
        }

        if (kp->cwhn) {
            const uint32_t Wb = 32 * G + (KW - 1) * d0;
            conv2d_hmx_cwhn_f16(stage16, stage, KH, kp->stage_row, Wb * Cin);
            for (uint32_t g = 0; g < G; g++) {
                __fp16 * tile = act + (size_t) (t - cs + g) * n_vec * 64;
                conv2d_hmx_pack_tile_cwhn(tile, stage16 + (size_t) g * 32 * Cin, vin, scratch, kp, gpr, Cin, d0);
            }
        } else {
            for (uint32_t g = 0; g < G; g++) {
                __fp16 * tile = act + (size_t) (t - cs + g) * n_vec * 64;
                conv2d_hmx_pack_tile(tile, stage + g * 32, kp->k_size, KW, d0, kp->stage_row, n_vec);
            }
        }

        if (n_stage > 1 && tnext < t1) {
            conv2d_hmx_dma_wait(tr, dmaq);
            bi    ^= 1;
            ready  = true;
        } else {
            ready = false;
        }
        t = tnext;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) ith);
}

static void conv2d_hmx_unpack(const struct htp_conv2d_hmx_context * hc, uint32_t ith) {
    struct htp_ops_context *                octx = hc->octx;
    const struct htp_conv2d_kernel_params * kp   = hc->kp;
    const struct htp_tensor *               dst  = octx->dst;

    const uint32_t OW    = dst->ne[0];
    const uint32_t OH    = dst->ne[1];
    const uint32_t Cout  = dst->ne[2];
    const uint32_t n_owt = kp->n_ow_tiles;
    const uint32_t OS    = kp->seg_tiles * 32;

    dma_queue * dmaq      = octx->ctx->dma[ith];
    uint8_t *   thr       = hc->thr_base + (size_t) ith * kp->thr_size + (size_t) (kp->n_stage > 1 ? 2 : 1) * kp->stage_size;
    float *     ostage[2] = {
        (float *) thr,
        (float *) (thr + (size_t) 32 * OS * sizeof(float)),
    };
    const __fp16 * out = hc->out[hc->unpack_chunk & 1];

    uint32_t       t0, t1;
    const uint32_t cs  = conv2d_hmx_chunk_range(hc, hc->unpack_chunk, hc->unpack_tpt, ith, &t0, &t1);
    const uint32_t nct = MIN(cs + kp->ct_chunk, hc->t_end) - cs;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) ith);

    uint32_t pending = 0;
    uint32_t b       = 0;

    for (uint32_t t = t0; t < t1;) {
        const uint32_t row = fastdiv(t, &kp->div_ow_tiles);
        const uint32_t owt = t - row * n_owt;
        const uint32_t G   = MIN(MIN(kp->seg_tiles, t1 - t), n_owt - owt);
        const uint32_t n   = fastdiv(row, &kp->div_oh);
        const uint32_t oh  = row - n * OH;
        const uint32_t ow0 = owt * 32;
        const uint32_t nv  = MIN(G * 32, OW - ow0);

        for (uint32_t rt = 0; rt < hc->n_rt; rt++) {
            const uint32_t co0 = (hc->rt0 + rt) * 32;
            const uint32_t nco = MIN(32, Cout - co0);

            if (pending == 2) {
                dma_queue_pop(dmaq);
                pending--;
            }

            float * ob = ostage[b];
            for (uint32_t g = 0; g < G; g++) {
                const HVX_Vector * tv = (const HVX_Vector *) (out + ((size_t) rt * nct + (t - cs + g)) * HMX_FP16_TILE_N_ELMS);
                for (uint32_t r = 0; r < 16; r++) {
                    const HVX_VectorPair pr = hvx_vec_f16_to_f32_shuff(tv[r]);
                    *(HVX_Vector *) (ob + (size_t) (2 * r + 0) * OS + g * 32) = Q6_V_lo_W(pr);
                    *(HVX_Vector *) (ob + (size_t) (2 * r + 1) * OS + g * 32) = Q6_V_hi_W(pr);
                }
            }

            const dma_addr_t ddr = dst->data + (size_t) n * dst->nb[3] + (size_t) oh * dst->nb[1] + (size_t) co0 * dst->nb[2] + (size_t) ow0 * sizeof(float);
            while (!conv2d_dma_push(dmaq, dma_make_data(ddr, ob), dst->nb[2], (size_t) OS * sizeof(float), (size_t) nv * sizeof(float), nco)) {
                dma_queue_pop(dmaq);
                pending--;
            }
            pending++;
            b ^= 1;
        }
        t += G;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) ith);
}

// HVX_TILES: w2[rt][k/2][co] = (w[co][k], w[co][k+1]) fp16 pairs in DDR, taken from the row-side weight tiles
static void conv2d_hvx_wpairs_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_hmx_context * hc = (const struct htp_conv2d_hmx_context *) data;
    const struct htp_conv2d_kernel_params * kp = hc->kp;

    const uint32_t n_kt = kp->n_k_tiles;

    HVX_Vector * vin     = (HVX_Vector *) (hc->thr_base + (size_t) ith * kp->thr_size);
    HVX_Vector * scratch = vin + 32;
    HVX_Vector * sink    = scratch + 32;

    for (uint32_t rt = ith; rt < hc->n_rt; rt += nth) {
        for (uint32_t kt = 0; kt < n_kt; kt++) {
            const HVX_Vector * tv = (const HVX_Vector *) (hc->w_tiles + ((size_t) rt * n_kt + kt) * HMX_FP16_TILE_N_ELMS);
            for (uint32_t r = 0; r < 16; r++) {
                const HVX_Vector d = Q6_Vh_vdeal_Vh(tv[r]);
                vin[2 * r + 0] = d;
                vin[2 * r + 1] = Q6_V_vror_VR(d, 64);
            }
            HVX_Vector * wp = (HVX_Vector *) (hc->w2 + ((size_t) rt * n_kt + kt) * 16 * 32);
            conv2d_transpose_w32(vin, 1, scratch, wp, sink);
        }
    }
}

// acc (sf on v79+, qf32 before) += x.hf[2j] * w.hf[2j] + x.hf[2j+1] * w.hf[2j+1]
static inline HVX_Vector conv2d_dmpyacc(HVX_Vector acc, HVX_Vector x, HVX_Vector w) {
#if __HVX_ARCH__ >= 79
    return Q6_Vsf_vdmpyacc_VsfVhfVhf(acc, x, w);
#else
    const HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(x, w);
    return Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(p), Q6_V_hi_W(p)));
#endif
}

static inline HVX_Vector conv2d_acc_to_sf(HVX_Vector acc) {
#if __HVX_ARCH__ >= 79
    return acc;
#else
    return Q6_Vsf_equals_Vqf32(acc);
#endif
}

// HVX_TILES: out[co][px] = sum over k pairs of vdmpy(act pair vector, splat(w2[co][pair])), 4 cout x 2 col tiles per pass
static void conv2d_hvx_compute(const struct htp_conv2d_hmx_context * hc, uint32_t ith) {
    struct htp_ops_context *                octx = hc->octx;
    const struct htp_conv2d_kernel_params * kp   = hc->kp;
    const struct htp_tensor *               dst  = octx->dst;

    const uint32_t OW    = dst->ne[0];
    const uint32_t OH    = dst->ne[1];
    const uint32_t Cout  = dst->ne[2];
    const uint32_t n_owt = kp->n_ow_tiles;
    const uint32_t OS    = kp->seg_tiles * 32;
    const uint32_t n_vec = kp->n_k_tiles * 16;
    const uint32_t np    = kp->n_groups * kp->k_group_pad / 2;

    dma_queue * dmaq      = octx->ctx->dma[ith];
    uint8_t *   thr       = hc->thr_base + (size_t) ith * kp->thr_size + (size_t) (kp->n_stage > 1 ? 2 : 1) * kp->stage_size;
    float *     ostage[2] = {
        (float *) thr,
        (float *) (thr + (size_t) 32 * OS * sizeof(float)),
    };
    const __fp16 * act = hc->act[hc->pack_chunk & 1];

    uint32_t       t0, t1;
    const uint32_t cs = conv2d_hmx_chunk_range(hc, hc->pack_chunk, hc->pack_tpt, ith, &t0, &t1);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);

    uint32_t pending = 0;
    uint32_t b       = 0;

    for (uint32_t t = t0; t < t1;) {
        const uint32_t row = fastdiv(t, &kp->div_ow_tiles);
        const uint32_t owt = t - row * n_owt;
        const uint32_t G   = MIN(MIN(kp->seg_tiles, t1 - t), n_owt - owt);
        const uint32_t n   = fastdiv(row, &kp->div_oh);
        const uint32_t oh  = row - n * OH;
        const uint32_t ow0 = owt * 32;
        const uint32_t nv  = MIN(G * 32, OW - ow0);

        for (uint32_t rt = 0; rt < hc->n_rt; rt++) {
            const uint32_t co0 = (hc->rt0 + rt) * 32;
            const uint32_t nco = MIN(32, Cout - co0);

            if (pending == 2) {
                dma_queue_pop(dmaq);
                pending--;
            }

            float *        ob = ostage[b];
            const uint32_t ov = OS / 32;
            uint32_t       g  = 0;
            for (; g + 1 < G; g += 2) {
                const HVX_Vector * a0 = (const HVX_Vector *) (act + (size_t) (t - cs + g) * n_vec * 64);
                const HVX_Vector * a1 = a0 + n_vec;
                for (uint32_t c = 0; c < nco; c += 4) {
                    const uint32_t * w = hc->w2 + (size_t) rt * n_vec * 32 + c;
                    HVX_Vector s00 = Q6_V_vzero(), s01 = Q6_V_vzero(), s02 = Q6_V_vzero(), s03 = Q6_V_vzero();
                    HVX_Vector s10 = Q6_V_vzero(), s11 = Q6_V_vzero(), s12 = Q6_V_vzero(), s13 = Q6_V_vzero();
                    for (uint32_t i = 0; i < np; i++) {
                        const HVX_Vector x0 = a0[i];
                        const HVX_Vector x1 = a1[i];
                        const HVX_Vector w0 = Q6_V_vsplat_R(w[i * 32 + 0]);
                        const HVX_Vector w1 = Q6_V_vsplat_R(w[i * 32 + 1]);
                        const HVX_Vector w2 = Q6_V_vsplat_R(w[i * 32 + 2]);
                        const HVX_Vector w3 = Q6_V_vsplat_R(w[i * 32 + 3]);
                        s00 = conv2d_dmpyacc(s00, x0, w0);
                        s01 = conv2d_dmpyacc(s01, x0, w1);
                        s02 = conv2d_dmpyacc(s02, x0, w2);
                        s03 = conv2d_dmpyacc(s03, x0, w3);
                        s10 = conv2d_dmpyacc(s10, x1, w0);
                        s11 = conv2d_dmpyacc(s11, x1, w1);
                        s12 = conv2d_dmpyacc(s12, x1, w2);
                        s13 = conv2d_dmpyacc(s13, x1, w3);
                    }
                    HVX_Vector * o = (HVX_Vector *) (ob + (size_t) c * OS + g * 32);
                    o[0 * ov]     = conv2d_acc_to_sf(s00);
                    o[1 * ov]     = conv2d_acc_to_sf(s01);
                    o[2 * ov]     = conv2d_acc_to_sf(s02);
                    o[3 * ov]     = conv2d_acc_to_sf(s03);
                    o[0 * ov + 1] = conv2d_acc_to_sf(s10);
                    o[1 * ov + 1] = conv2d_acc_to_sf(s11);
                    o[2 * ov + 1] = conv2d_acc_to_sf(s12);
                    o[3 * ov + 1] = conv2d_acc_to_sf(s13);
                }
            }
            if (g < G) {
                const HVX_Vector * a0 = (const HVX_Vector *) (act + (size_t) (t - cs + g) * n_vec * 64);
                for (uint32_t c = 0; c < nco; c += 8) {
                    const uint32_t * w = hc->w2 + (size_t) rt * n_vec * 32 + c;
                    HVX_Vector s0 = Q6_V_vzero(), s1 = Q6_V_vzero(), s2 = Q6_V_vzero(), s3 = Q6_V_vzero();
                    HVX_Vector s4 = Q6_V_vzero(), s5 = Q6_V_vzero(), s6 = Q6_V_vzero(), s7 = Q6_V_vzero();
                    for (uint32_t i = 0; i < np; i++) {
                        const HVX_Vector x = a0[i];
                        s0 = conv2d_dmpyacc(s0, x, Q6_V_vsplat_R(w[i * 32 + 0]));
                        s1 = conv2d_dmpyacc(s1, x, Q6_V_vsplat_R(w[i * 32 + 1]));
                        s2 = conv2d_dmpyacc(s2, x, Q6_V_vsplat_R(w[i * 32 + 2]));
                        s3 = conv2d_dmpyacc(s3, x, Q6_V_vsplat_R(w[i * 32 + 3]));
                        s4 = conv2d_dmpyacc(s4, x, Q6_V_vsplat_R(w[i * 32 + 4]));
                        s5 = conv2d_dmpyacc(s5, x, Q6_V_vsplat_R(w[i * 32 + 5]));
                        s6 = conv2d_dmpyacc(s6, x, Q6_V_vsplat_R(w[i * 32 + 6]));
                        s7 = conv2d_dmpyacc(s7, x, Q6_V_vsplat_R(w[i * 32 + 7]));
                    }
                    HVX_Vector * o = (HVX_Vector *) (ob + (size_t) c * OS + g * 32);
                    o[0 * ov] = conv2d_acc_to_sf(s0);
                    o[1 * ov] = conv2d_acc_to_sf(s1);
                    o[2 * ov] = conv2d_acc_to_sf(s2);
                    o[3 * ov] = conv2d_acc_to_sf(s3);
                    o[4 * ov] = conv2d_acc_to_sf(s4);
                    o[5 * ov] = conv2d_acc_to_sf(s5);
                    o[6 * ov] = conv2d_acc_to_sf(s6);
                    o[7 * ov] = conv2d_acc_to_sf(s7);
                }
            }

            const dma_addr_t ddr = dst->data + (size_t) n * dst->nb[3] + (size_t) oh * dst->nb[1] + (size_t) co0 * dst->nb[2] + (size_t) ow0 * sizeof(float);
            while (!conv2d_dma_push(dmaq, dma_make_data(ddr, ob), dst->nb[2], (size_t) OS * sizeof(float), (size_t) nv * sizeof(float), nco)) {
                dma_queue_pop(dmaq);
                pending--;
            }
            pending++;
            b ^= 1;
        }
        t += G;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ith);
}

static void conv2d_hvx_phase_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_hmx_context * hc = (const struct htp_conv2d_hmx_context *) data;

    conv2d_hmx_pack(hc, ith);
    conv2d_hvx_compute(hc, ith);
    dma_queue_flush(hc->octx->ctx->dma[ith]);
}

static void conv2d_hmx_phase_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_conv2d_hmx_context * hc = (const struct htp_conv2d_hmx_context *) data;

    if (hc->unpack_chunk >= 0) {
        conv2d_hmx_unpack(hc, ith);
    }
    if (hc->pack_chunk >= 0) {
        conv2d_hmx_pack(hc, ith);
    }
    dma_queue_flush(hc->octx->ctx->dma[ith]);
}

static inline uint32_t conv2d_hmx_tpt(const struct htp_conv2d_hmx_context * hc, int32_t c, uint32_t nth) {
    if (c < 0) {
        return 0;
    }
    const uint32_t cs  = hc->t_base + (uint32_t) c * hc->kp->ct_chunk;
    const uint32_t nct = MIN(hc->kp->ct_chunk, hc->t_end - cs);
    return (nct + nth - 1) / nth;
}

static void conv2d_hmx_run_phase(struct htp_conv2d_hmx_context * hc, int32_t pack, int32_t unpack, uint32_t nth) {
    hc->pack_chunk   = pack;
    hc->unpack_chunk = unpack;
    hc->pack_tpt     = conv2d_hmx_tpt(hc, pack, nth);
    hc->unpack_tpt   = conv2d_hmx_tpt(hc, unpack, nth);
    work_queue_run(hc->octx->ctx->work_queue, conv2d_hmx_phase_thread, hc, nth);
}

static int conv2d_hmx(struct htp_ops_context * octx, const struct htp_conv2d_kernel_params * kp) {
    const bool use_hmx = kp->kernel_type == HTP_CONV2D_KERNEL_HMX;
    if (use_hmx && !octx->ctx->hmx_enabled) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (!use_hmx && (size_t) 32 * kp->rt_block * kp->n_k_tiles * 32 * sizeof(__fp16) > octx->ctx->ddr_spad_size) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (kp->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const struct htp_tensor * dst = octx->dst;
    const uint32_t OH = dst->ne[1];
    const uint32_t N  = dst->ne[3];

    uint32_t row_base = 0;
    uint32_t n_rows   = N * OH;
    if (octx->ctx->mdev.count > 1) {
        uint32_t rows_per_chunk = 0;
        if (htp_tensor_mdev_data_aligned(dst) && (dst->nb[2] % HTP_TENSOR_MDEV_LINE_SIZE) == 0) {
            rows_per_chunk = HTP_TENSOR_MDEV_LINE_SIZE / hex_gcd_u32(dst->nb[1], HTP_TENSOR_MDEV_LINE_SIZE);
            if (N > 1 && OH % rows_per_chunk != 0) {
                rows_per_chunk = 0;
            }
        }
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            n_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_base = range.start;
        n_rows   = range.count;
    }
    if (n_rows == 0) {
        return HTP_STATUS_OK;
    }

    uint8_t * base = (uint8_t *) octx->ctx->vtcm_base;
    __fp16 *  scales = (__fp16 *) base;
    if (use_hmx) {
        hmx_init_column_scales(scales, Q6_V_vsplat_R(0x3c00));
    }

    // HVX_TILES packs and computes each chunk in one phase, so it needs a single act buffer and no out buffers
    struct htp_conv2d_hmx_context hc = {
        .octx     = octx,
        .kp       = kp,
        .w_tiles  = VTCM_LAYOUT_PTR(__fp16, base, kp->off_w),
        .act      = { VTCM_LAYOUT_PTR(__fp16, base, kp->off_act), VTCM_LAYOUT_PTR(__fp16, base, kp->off_act + (use_hmx ? kp->act_size : 0)) },
        .out      = { VTCM_LAYOUT_PTR(__fp16, base, kp->off_out), VTCM_LAYOUT_PTR(__fp16, base, kp->off_out + kp->out_size) },
        .thr_base = VTCM_LAYOUT_PTR(uint8_t, base, kp->off_thr),
        .w2       = (uint32_t *) octx->ctx->ddr_spad_base,
        .t_base   = row_base * kp->n_ow_tiles,
        .t_end    = (row_base + n_rows) * kp->n_ow_tiles,
    };

    const uint32_t nth      = kp->n_threads;
    const int32_t  n_chunks = (int32_t) ((hc.t_end - hc.t_base + kp->ct_chunk - 1) / kp->ct_chunk);

    struct htp_conv2d_hmx_job job;

    for (uint32_t rt0 = 0; rt0 < kp->n_row_tiles; rt0 += kp->rt_block) {
        hc.rt0  = rt0;
        hc.n_rt = MIN(kp->rt_block, kp->n_row_tiles - rt0);
        work_queue_run(octx->ctx->work_queue, kp->w_kmajor ? conv2d_hmx_wpack_kmajor_thread : conv2d_hmx_wpack_thread, &hc, nth);

        if (!use_hmx) {
            work_queue_run(octx->ctx->work_queue, conv2d_hvx_wpairs_thread, &hc, nth);
            for (int32_t c = 0; c < n_chunks; c++) {
                hc.pack_chunk   = c;
                hc.unpack_chunk = -1;
                hc.pack_tpt     = conv2d_hmx_tpt(&hc, c, nth);
                work_queue_run(octx->ctx->work_queue, conv2d_hvx_phase_thread, &hc, nth);
            }
            continue;
        }

        conv2d_hmx_run_phase(&hc, 0, -1, nth);

        for (int32_t c = 0; c < n_chunks; c++) {
            const uint32_t cs = hc.t_base + (uint32_t) c * kp->ct_chunk;

            job.output       = hc.out[c & 1];
            job.kernel_tiles = hc.w_tiles;
            job.patch_tiles  = hc.act[c & 1];
            job.scales       = scales;
            job.n_row_tiles  = hc.n_rt;
            job.n_col_tiles  = MIN(kp->ct_chunk, hc.t_end - cs);
            job.n_dot_tiles  = kp->n_k_tiles;
            hmx_queue_push(octx->ctx->hmx_queue, hmx_queue_make_desc(conv2d_hmx_worker, &job));

            const int32_t pack   = c + 1 < n_chunks ? c + 1 : -1;
            const int32_t unpack = c - 1;
            if (pack >= 0 || unpack >= 0) {
                conv2d_hmx_run_phase(&hc, pack, unpack, nth);
            }

            hmx_queue_pop(octx->ctx->hmx_queue);
        }

        conv2d_hmx_run_phase(&hc, -1, n_chunks - 1, nth);
    }

    return HTP_STATUS_OK;
}

int op_conv_2d(struct htp_ops_context * octx) {
    const struct htp_tensor * kernel = octx->src[0];
    const struct htp_tensor * input  = octx->src[1];
    const struct htp_tensor * dst    = octx->dst;

    if ((kernel->type != HTP_TYPE_F16 && kernel->type != HTP_TYPE_F32) ||
        input->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        FARF(ERROR, "conv_2d: only F16/F32 kernel, F32 input and F32 dst are supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_conv2d_kernel_params * kparams =
        (const struct htp_conv2d_kernel_params *) octx->kernel_params;

    if (kparams->total_rows == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->kernel_type == HTP_CONV2D_KERNEL_HMX || kparams->kernel_type == HTP_CONV2D_KERNEL_HVX_TILES) {
        const int status = conv2d_hmx(octx, kparams);
        if (status == HTP_STATUS_OK) {
            return HTP_STATUS_OK;
        }
    }

    const uint32_t n_threads = MIN(kparams->n_threads, octx->n_threads);
    struct htp_conv2d_context ctx = {
        .octx    = octx,
        .kparams = kparams,
    };

    if (kparams->kernel_type == HTP_CONV2D_KERNEL_HVX_WIDTH) {
        const size_t k_elems = (size_t) kernel->ne[0] * kernel->ne[1] * kernel->ne[2];
        if (k_elems > 1024) {
            const size_t wbuf_bytes = hex_round_up(k_elems * sizeof(float), VLEN);
            octx->src0_spad.size_per_thread = wbuf_bytes + VLEN;
            octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
            if (octx->src0_spad.size > octx->ctx->vtcm_size) {
                work_queue_run(octx->ctx->work_queue, conv2d_scalar_thread, &ctx, n_threads);
                return HTP_STATUS_OK;
            }
            octx->src0_spad.data = octx->ctx->vtcm_base;
            octx->src0_spad.src  = NULL;
        }

        work_queue_run(octx->ctx->work_queue, conv2d_hvx_width_thread, &ctx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, conv2d_scalar_thread, &ctx, n_threads);
    }
    return HTP_STATUS_OK;
}
