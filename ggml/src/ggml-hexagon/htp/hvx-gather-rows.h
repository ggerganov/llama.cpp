#pragma once

// Copy short f32 or f16 rows through VTCM: DMA source spans in, rearrange with vgather, DMA result out.
// Row i of dst has a[i][0..na) followed by b[i][0..nb); dst rows are contiguous.

#include "hex-common.h"
#include "hex-profile.h"
#include "hvx-utils.h"
#include "dma-queue.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "work-queue.h"

struct hvx_gather_rows_task {
    struct htp_ops_context * octx;
    const struct htp_tensor * a;
    const struct htp_tensor * b;
    dma_addr_t dst;
    uint32_t na, nb, ne;
    uint32_t r_start, rows;
    uint32_t rg, vg;
    uint32_t rows_per_thread;
    uint32_t nr_max;
    uint32_t v1, span1;
    uint32_t region_size, out_off, slice;
    bool b_dense;
};

static inline void hvx_gather_rows_dma(dma_queue * q, dma_addr_t dst, dma_addr_t src, size_t size) {
    if (!dma_queue_push(q, dma_make_data(dst, src), size, size, size, 1)) {
        dma_queue_flush(q);
        dma_queue_push(q, dma_make_data(dst, src), size, size, size, 1);
    }
}

static inline void hvx_gather_rows_thread_f32(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct hvx_gather_rows_task * task = (const struct hvx_gather_rows_task *) data;
    const struct htp_tensor * a = task->a;
    const struct htp_tensor * b = task->b;

    const uint32_t r_offset = MIN(ith * task->rows_per_thread, task->rows);
    const uint32_t r_beg    = task->r_start + r_offset;
    const uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_thread, task->rows);
    if (r_beg >= r_end) {
        return;
    }

    struct htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * base   = task->octx->ctx->vtcm_base + ith * task->slice;
    HVX_Vector * tab = (HVX_Vector *) base;
    HVX_Vector * inc = tab + task->vg;
    HVX_Vector * cur = inc + task->vg;
    uint8_t * region = (uint8_t *) (cur + task->vg);
    uint8_t * out    = region + task->out_off;

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    const int32_t s_a = task->rg * a->nb[1];
    const int32_t s_b = b ? task->rg * b->nb[1] : 0;
    const uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int32_t t_buf[32] __attribute__((aligned(128)));
    int32_t s_buf[32] __attribute__((aligned(128)));
    for (uint32_t k = 0; k < task->vg; k++) {
        for (uint32_t lane = 0; lane < 32; lane++) {
            if (col < task->na) {
                t_buf[lane] = row * a->nb[1] + col * 4;
                s_buf[lane] = s_a;
            } else {
                t_buf[lane] = task->v1 + (col - task->na) * task->span1 + row * b_nb1;
                s_buf[lane] = s_b;
            }
            if (++col == task->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(const HVX_Vector *) t_buf;
        inc[k] = *(const HVX_Vector *) s_buf;
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_queue * q = task->octx->ctx->dma[ith];

    for (uint32_t r = r_beg; r < r_end; r += task->nr_max) {
        const uint32_t nr = MIN(task->nr_max, r_end - r);

        hvx_gather_rows_dma(q, (dma_addr_t) region, a->data + r * a->nb[1], (nr - 1) * a->nb[1] + task->na * 4);
        if (b) {
            if (task->b_dense) {
                hvx_gather_rows_dma(q, (dma_addr_t) (region + task->v1), b->data + r * b->nb[1], (nr - 1) * b->nb[1] + task->nb * 4);
            } else {
                for (uint32_t k = 0; k < task->nb; k++) {
                    hvx_gather_rows_dma(q, (dma_addr_t) (region + task->v1 + k * task->span1), b->data + k * b->nb[0] + r * b->nb[1], (nr - 1) * b->nb[1] + 4);
                }
            }
        }
        dma_queue_flush(q);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        const uint32_t nvec = (nr * task->ne + 31) >> 5;
        if (task->vg == 1) {
            HVX_Vector vcur = tab[0];
            const HVX_Vector vinc = inc[0];
            #pragma unroll(4)
            for (uint32_t v = 0; v < nvec; v++) {
                Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) region, task->region_size, vcur);
                vcur = Q6_Vw_vadd_VwVw(vcur, vinc);
            }
        } else {
            for (uint32_t k = 0; k < task->vg; k++) {
                cur[k] = tab[k];
            }
            uint32_t k = 0;
            #pragma unroll(4)
            for (uint32_t v = 0; v < nvec; v++) {
                Q6_vgather_ARMVw((HVX_Vector *) (out + v * 128), (size_t) region, task->region_size, cur[k]);
                cur[k] = Q6_Vw_vadd_VwVw(cur[k], inc[k]);
                if (++k == task->vg) {
                    k = 0;
                }
            }
        }

        hvx_gather_sync(out);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        hvx_gather_rows_dma(q, task->dst + r * task->ne * 4, (dma_addr_t) out, nr * task->ne * 4);
        dma_queue_flush(q);
    }
}

static inline void hvx_gather_rows_thread_f16(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;
    const struct hvx_gather_rows_task * task = (const struct hvx_gather_rows_task *) data;
    const struct htp_tensor * a = task->a;
    const struct htp_tensor * b = task->b;

    const uint32_t r_offset = MIN(ith * task->rows_per_thread, task->rows);
    const uint32_t r_beg    = task->r_start + r_offset;
    const uint32_t r_end    = task->r_start + MIN(r_offset + task->rows_per_thread, task->rows);
    if (r_beg >= r_end) {
        return;
    }

    struct htp_thread_trace * tr = &task->octx->ctx->trace[ith];

    uint8_t * base   = task->octx->ctx->vtcm_base + ith * task->slice;
    HVX_Vector * tab = (HVX_Vector *) base;
    HVX_Vector * inc = tab + task->vg;
    HVX_Vector * cur = inc + task->vg;
    uint8_t * region = (uint8_t *) (cur + task->vg);
    uint8_t * out    = region + task->out_off;

    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);
    const int16_t s_a = (int16_t) (task->rg * a->nb[1]);
    const int16_t s_b = b ? (int16_t) (task->rg * b->nb[1]) : 0;
    const uint32_t b_nb1 = b ? b->nb[1] : 0;
    uint32_t row = 0;
    uint32_t col = 0;
    int16_t t_buf[64] __attribute__((aligned(128)));
    int16_t s_buf[64] __attribute__((aligned(128)));
    for (uint32_t k = 0; k < task->vg; k++) {
        for (uint32_t lane = 0; lane < 64; lane++) {
            if (col < task->na) {
                t_buf[lane] = (int16_t) (row * a->nb[1] + col * 2);
                s_buf[lane] = s_a;
            } else {
                t_buf[lane] = (int16_t) (task->v1 + (col - task->na) * task->span1 + row * b_nb1);
                s_buf[lane] = s_b;
            }
            if (++col == task->ne) {
                col = 0;
                row++;
            }
        }
        tab[k] = *(const HVX_Vector *) t_buf;
        inc[k] = *(const HVX_Vector *) s_buf;
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, (uint16_t) r_beg);

    dma_queue * q = task->octx->ctx->dma[ith];

    for (uint32_t r = r_beg; r < r_end; r += task->nr_max) {
        const uint32_t nr = MIN(task->nr_max, r_end - r);

        hvx_gather_rows_dma(q, (dma_addr_t) region, a->data + r * a->nb[1], (nr - 1) * a->nb[1] + task->na * 2);
        if (b) {
            if (task->b_dense) {
                hvx_gather_rows_dma(q, (dma_addr_t) (region + task->v1), b->data + r * b->nb[1], (nr - 1) * b->nb[1] + task->nb * 2);
            } else {
                for (uint32_t k = 0; k < task->nb; k++) {
                    hvx_gather_rows_dma(q, (dma_addr_t) (region + task->v1 + k * task->span1), b->data + k * b->nb[0] + r * b->nb[1], (nr - 1) * b->nb[1] + 2);
                }
            }
        }
        dma_queue_flush(q);

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
        const uint32_t nvec = (nr * task->ne + 63) >> 6;
        if (task->vg == 1) {
            HVX_Vector vcur = tab[0];
            const HVX_Vector vinc = inc[0];
            #pragma unroll(4)
            for (uint32_t v = 0; v < nvec; v++) {
                Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (size_t) region, task->region_size, vcur);
                vcur = Q6_Vh_vadd_VhVh(vcur, vinc);
            }
        } else {
            for (uint32_t k = 0; k < task->vg; k++) {
                cur[k] = tab[k];
            }
            uint32_t k = 0;
            #pragma unroll(4)
            for (uint32_t v = 0; v < nvec; v++) {
                Q6_vgather_ARMVh((HVX_Vector *) (out + v * 128), (size_t) region, task->region_size, cur[k]);
                cur[k] = Q6_Vh_vadd_VhVh(cur[k], inc[k]);
                if (++k == task->vg) {
                    k = 0;
                }
            }
        }

        hvx_gather_sync(out);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

        hvx_gather_rows_dma(q, task->dst + r * task->ne * 2, (dma_addr_t) out, nr * task->ne * 2);
        dma_queue_flush(q);
    }
}

static inline bool hvx_gather_rows_sync(struct htp_ops_context * octx, const struct htp_tensor * a, uint32_t na,
                                        const struct htp_tensor * b, uint32_t nb, dma_addr_t dst, uint32_t rows) {
    if (a->type != HTP_TYPE_F32 && a->type != HTP_TYPE_F16) {
        return false;
    }
    if (b && b->type != a->type) {
        return false;
    }
    const uint32_t elem_size  = (a->type == HTP_TYPE_F16) ? 2 : 4;
    const uint32_t vlen_elems = 128 / elem_size;
    const uint32_t ne         = na + nb;
    if (ne == 0 || ne > vlen_elems || rows == 0 ||
        a->nb[0] != elem_size || (a->nb[1] % elem_size) != 0 ||
        (b && ((b->nb[0] % elem_size) != 0 || (b->nb[1] % elem_size) != 0))) {
        return false;
    }

    struct hvx_gather_rows_task task;
    task.octx = octx;
    task.a    = a;
    task.b    = b;
    task.dst  = dst;
    task.na   = na;
    task.nb   = nb;
    task.ne   = ne;
    task.vg   = ne / hex_gcd_u32(ne, vlen_elems);
    task.rg   = task.vg * vlen_elems / ne;

    uint32_t r_start = 0;
    uint32_t r_count = rows;
    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            rows, task.rg, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        r_start = range.start;
        r_count = range.count;
    }

    if (r_count == 0) {
        return true;
    }

    task.r_start = r_start;
    task.rows    = r_count;

    const uint32_t n_threads = octx->n_threads;
    task.slice = (uint32_t) (octx->ctx->vtcm_size / n_threads) & ~127u;
    task.rows_per_thread = hex_round_up((r_count + n_threads - 1) / n_threads, task.rg);

    const bool b_dense = b && (b->nb[0] == elem_size);
    task.b_dense       = b_dense;

    const uint32_t fixed   = 3 * task.vg * 128 + 3 * 128;
    const uint32_t per_row = a->nb[1] + (b ? (b_dense ? b->nb[1] : nb * b->nb[1]) : 0) + ne * elem_size;
    if (task.slice <= fixed + per_row * task.rg) {
        return false;
    }
    task.nr_max = MIN((task.slice - fixed) / per_row, task.rows_per_thread);
    if (elem_size == 2) {
        const uint32_t per_row_region = a->nb[1] + (b ? (b_dense ? b->nb[1] : nb * b->nb[1]) : 0);
        if (per_row_region > 0) {
            task.nr_max = MIN(task.nr_max, (32768u > 256u ? (32768u - 256u) / per_row_region : 0));
        }
    }
    task.nr_max = task.nr_max / task.rg * task.rg;
    if (task.nr_max == 0) {
        return false;
    }

    task.v1 = hex_round_up((task.nr_max - 1) * a->nb[1] + na * elem_size, 128);
    if (b_dense) {
        task.span1       = elem_size;
        task.region_size = task.v1 + (task.nr_max - 1) * b->nb[1] + nb * elem_size;
    } else {
        task.span1       = b ? (task.nr_max - 1) * b->nb[1] + elem_size : 0;
        task.region_size = task.v1 + nb * task.span1;
    }
    if (elem_size == 2 && task.region_size > 32768) {
        return false;
    }
    task.out_off = hex_round_up(task.region_size, 128);

    work_queue_func_t worker_func = (elem_size == 2) ? hvx_gather_rows_thread_f16 : hvx_gather_rows_thread_f32;
    work_queue_run(octx->ctx->work_queue, worker_func, &task, n_threads);
    return true;
}
