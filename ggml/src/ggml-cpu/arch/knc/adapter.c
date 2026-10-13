#include "knc-mq.h"
/* Bind real GGML operations to the unchanged GNU IMCI 0.2.1 core.
 * GGML/OpenMP owns the worker team. Never call the standalone GEMV wrapper here.
 */
#include "ggml.h"
#include "../../quants.h"
#include "../../vec.h"
#include "ggml-knc.h"
#include "quant.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK_LAYOUT(n, x) typedef char n[(x) ? 1 : -1]
CHECK_LAYOUT(knc_qk, QK_K == 256);
CHECK_LAYOUT(knc_q5, sizeof(block_q5_K) == 176);
CHECK_LAYOUT(knc_q8, sizeof(block_q8_K) == 292);
CHECK_LAYOUT(knc_d, offsetof(block_q8_K, d) == 0);
CHECK_LAYOUT(knc_qs, offsetof(block_q8_K, qs) == 4);
CHECK_LAYOUT(knc_sum, offsetof(block_q8_K, bsums) == 260);
CHECK_LAYOUT(knc_dmin, offsetof(block_q5_K, dmin) == 2);
CHECK_LAYOUT(knc_scales, offsetof(block_q5_K, scales) == 4);
CHECK_LAYOUT(knc_qh, offsetof(block_q5_K, qh) == 16);
CHECK_LAYOUT(knc_q5qs, offsetof(block_q5_K, qs) == 48);
CHECK_LAYOUT(knc_ws, sizeof(ggml_knc_workspace) == 576);

CHECK_LAYOUT(ext_2_scales, offsetof(block_q2_K, scales) == 0);
CHECK_LAYOUT(ext_2_qs, offsetof(block_q2_K, qs) == 16);
CHECK_LAYOUT(ext_2_d, offsetof(block_q2_K, d) == 80);
CHECK_LAYOUT(ext_2_dmin, offsetof(block_q2_K, dmin) == 82);
CHECK_LAYOUT(ext_3_hmask, offsetof(block_q3_K, hmask) == 0);
CHECK_LAYOUT(ext_3_qs, offsetof(block_q3_K, qs) == 32);
CHECK_LAYOUT(ext_3_scales, offsetof(block_q3_K, scales) == 96);
CHECK_LAYOUT(ext_3_d, offsetof(block_q3_K, d) == 108);
CHECK_LAYOUT(ext_4_d, offsetof(block_q4_K, d) == 0);
CHECK_LAYOUT(ext_4_dmin, offsetof(block_q4_K, dmin) == 2);
CHECK_LAYOUT(ext_4_scales, offsetof(block_q4_K, scales) == 4);
CHECK_LAYOUT(ext_4_qs, offsetof(block_q4_K, qs) == 16);
CHECK_LAYOUT(ext_6_ql, offsetof(block_q6_K, ql) == 0);
CHECK_LAYOUT(ext_6_qh, offsetof(block_q6_K, qh) == 128);
CHECK_LAYOUT(ext_6_scales, offsetof(block_q6_K, scales) == 192);
CHECK_LAYOUT(ext_6_d, offsetof(block_q6_K, d) == 208);
CHECK_LAYOUT(ext_2_size, sizeof(block_q2_K) == 84);
CHECK_LAYOUT(ext_3_size, sizeof(block_q3_K) == 110);
CHECK_LAYOUT(ext_4_size, sizeof(block_q4_K) == 144);
CHECK_LAYOUT(ext_6_size, sizeof(block_q6_K) == 210);

static __thread ggml_knc_workspace workspace;
static int                         tracing = 0, optimized = 1, extended = 1;
static uint64_t q5_calls, q5_elements, f32_calls, f32_unaligned, q6_calls, generic_calls, ext_calls[7];

void ggml_knc_trace_enable(int enabled) {
    tracing = enabled;
    ggml_knc_mq_trace(enabled);
}

void ggml_knc_optimized_enable(int enabled) {
    optimized = enabled;
    ggml_knc_mq_optimized(enabled);
}

void ggml_knc_extended_enable(int enabled) {
    extended = enabled;
}

void ggml_knc_trace_report(void) {
    fprintf(stderr, "KNC_IMCI_EXT {\"q2_calls\":%llu,\"q3_calls\":%llu,\"q4_calls\":%llu,\"q6_calls\":%llu}\n",
            (unsigned long long) ext_calls[2], (unsigned long long) ext_calls[3], (unsigned long long) ext_calls[4],
            (unsigned long long) ext_calls[6]);
    fprintf(
        stderr,
        "KNC_KERNEL_TRACE {\"q5_imci_calls\":%llu,\"q5_elements\":%llu,\"f32_imci_calls\":%llu,\"f32_unaligned_scalar_calls\":%llu,\"q6_scalar_calls\":%llu,\"reference_calls\":%llu}\n",
        (unsigned long long) q5_calls, (unsigned long long) q5_elements, (unsigned long long) f32_calls,
        (unsigned long long) f32_unaligned, (unsigned long long) q6_calls, (unsigned long long) generic_calls);
}

#define TRACE(c, n)                                                     \
    do {                                                                \
        if (tracing)                                                    \
            __atomic_fetch_add(&(c), (uint64_t) (n), __ATOMIC_RELAXED); \
    } while (0)

static void require(int ok) {
    if (!ok) {
        fputs("KNC adapter rejected invalid dot arguments\n", stderr);
        abort();
    }
}

extern void ggml_vec_dot_q5_K_q8_K_generic(int, float *, size_t, const void *, size_t, const void *, size_t, int);
extern void ggml_vec_dot_q6_K_q8_K_generic(int, float *, size_t, const void *, size_t, const void *, size_t, int);
extern void ggml_knc_vec_dot_f32_generic(int, float *, size_t, const float *, size_t, const float *, size_t, int);

void ggml_vec_dot_q5_K_q8_K(int n,
                            float * restrict s,
                            size_t bs,
                            const void * restrict x,
                            size_t bx,
                            const void * restrict y,
                            size_t by,
                            int    nrc) {
    require(n >= 0 && nrc == 1);
    if (!optimized) {
        TRACE(generic_calls, 1);
        ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, x, bx, y, by, nrc);
        return;
    }
    require(ggml_knc_dot_q5_K_q8_K(n, s, x, y, &workspace) == GGML_KNC_OK);
    TRACE(q5_calls, 1);
    TRACE(q5_elements, n);
}

#define EXT(Q)                                                                                                       \
    extern void ggml_vec_dot_q##Q##_K_q8_K_generic(int, float *, size_t, const void *, size_t, const void *, size_t, \
                                                   int);                                                             \
    void        ggml_vec_dot_q##Q##_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict x, size_t bx, \
                                           const void * restrict y, size_t by, int nrc) {                            \
        require(n >= 0 && n % 256 == 0 && nrc == 1);                                                          \
        if (!optimized || !extended) {                                                                        \
            TRACE(generic_calls, 1);                                                                          \
            if (Q == 6)                                                                                       \
                TRACE(q6_calls, 1);                                                                           \
            ggml_vec_dot_q##Q##_K_q8_K_generic(n, s, bs, x, bx, y, by, nrc);                                  \
            return;                                                                                           \
        }                                                                                                     \
        require(ggml_knc_dot_k(Q, n, s, x, y) == 0);                                                          \
        TRACE(ext_calls[Q], 1);                                                                               \
    }

EXT(2)
EXT(3) EXT(4) EXT(6) void ggml_vec_dot_f32(int n,
                                           float * restrict s,
                                           size_t bs,
                                           const float * restrict x,
                                           size_t bx,
                                           const float * restrict y,
                                           size_t by,
                                           int    nrc) {
    require(n >= 0 && nrc == 1 && s && (!n || (x && y)));
    if (!optimized) {
        TRACE(generic_calls, 1);
        ggml_knc_vec_dot_f32_generic(n, s, bs, x, bx, y, by, nrc);
        return;
    }
    if (((uintptr_t) x & 63) || ((uintptr_t) y & 63)) {
        TRACE(f32_unaligned, 1);
        ggml_knc_vec_dot_f32_generic(n, s, bs, x, bx, y, by, nrc);
        return;
    }
    require(ggml_knc_dot_f32(n, s, x, y) == GGML_KNC_OK);
    TRACE(f32_calls, 1);
}
