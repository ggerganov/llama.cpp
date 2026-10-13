/* SPDX-License-Identifier: MIT
 * MPSS GNU K1OM / Knights Corner IMCI checked public wrappers.
 * No Intel compiler flags or global no-alias assertion are required.
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif
#include "ggml-knc.h"

#include "ggml-knc-internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef char knc_workspace_size_check[
    sizeof(ggml_knc_workspace) == GGML_KNC_WORKSPACE_BYTES ? 1 : -1];
typedef char knc_q4_alignment_check[
    offsetof(ggml_knc_workspace, internal_q4) % 64 == 0 ? 1 : -1];
typedef char knc_qh_alignment_check[
    offsetof(ggml_knc_workspace, internal_qh) % 64 == 0 ? 1 : -1];
typedef char knc_q8_alignment_check[
    offsetof(ggml_knc_workspace, internal_q8) % 64 == 0 ? 1 : -1];
typedef char knc_binary32_check[sizeof(float) == 4 ? 1 : -1];

static int knc_is_aligned(const void *p, size_t alignment) {
    return ((uintptr_t) p & (alignment - 1)) == 0;
}

static int knc_range_valid(const void *p, size_t bytes) {
    return bytes == 0 || (p != NULL && (uintptr_t) p <= UINTPTR_MAX - bytes);
}

/* Called only after checking that neither half-open range wraps. */
static int knc_overlap(const void *a, size_t na, const void *b, size_t nb) {
    return na && nb && (uintptr_t) a < (uintptr_t) b + nb &&
                      (uintptr_t) b < (uintptr_t) a + na;
}

void *ggml_knc_aligned_alloc(size_t bytes) {
    void *p = NULL;
    if (bytes == 0 || bytes > SIZE_MAX - (GGML_KNC_ALIGNMENT - 1)) return NULL;
    if (posix_memalign(&p, GGML_KNC_ALIGNMENT, bytes) != 0) return NULL;
    memset(p, 0, bytes);
    return p;
}

void ggml_knc_aligned_free(void *p) { free(p); }
size_t ggml_knc_workspace_size(void) { return sizeof(ggml_knc_workspace); }

ggml_knc_workspace *ggml_knc_workspace_alloc(size_t workers) {
    if (workers == 0 || workers > SIZE_MAX / sizeof(ggml_knc_workspace)) return NULL;
    return (ggml_knc_workspace *) ggml_knc_aligned_alloc(workers * sizeof(ggml_knc_workspace));
}

void ggml_knc_workspace_free(ggml_knc_workspace *p) { ggml_knc_aligned_free(p); }

static int knc_packed_sizes(size_t n, size_t *xb, size_t *yb) {
    size_t blocks;
    if (n % GGML_KNC_QK_K != 0) return GGML_KNC_ESIZE;
    blocks = n / GGML_KNC_QK_K;
    if (blocks > SIZE_MAX / GGML_KNC_Q8_K_BYTES) return GGML_KNC_ESIZE;
    *xb = blocks * GGML_KNC_Q5_K_BYTES;
    *yb = blocks * GGML_KNC_Q8_K_BYTES;
    return GGML_KNC_OK;
}

int ggml_knc_dot_q5_K_q8_K(size_t n, float *out, const void *x,
                         const void *y, ggml_knc_workspace *w) {
    size_t xb, yb;
    int rc = knc_packed_sizes(n, &xb, &yb);
    if (rc != GGML_KNC_OK) return rc;
    if (!out) return GGML_KNC_EINVAL;
    if (!knc_is_aligned(out, sizeof(float))) return GGML_KNC_EALIGN;
    if (!knc_range_valid(out, sizeof(float))) return GGML_KNC_ESIZE;
    if (n == 0) { *out = 0.0f; return GGML_KNC_OK; }
    if (!x || !y || !w) return GGML_KNC_EINVAL;
    if (!knc_is_aligned(w, GGML_KNC_ALIGNMENT)) return GGML_KNC_EALIGN;
    if (!knc_range_valid(x, xb) || !knc_range_valid(y, yb) ||
        !knc_range_valid(w, sizeof(*w))) return GGML_KNC_ESIZE;
    if (knc_overlap(out, sizeof(*out), x, xb) ||
        knc_overlap(out, sizeof(*out), y, yb) ||
        knc_overlap(out, sizeof(*out), w, sizeof(*w)) ||
        knc_overlap(w, sizeof(*w), x, xb) ||
        knc_overlap(w, sizeof(*w), y, yb)) return GGML_KNC_EINVAL;
    *out = ggml_knc_dot_packed_unchecked(n / GGML_KNC_QK_K,
                                  (const uint8_t *) x, (const uint8_t *) y, w);
    return GGML_KNC_OK;
}

int ggml_knc_matvec_q5_K_q8_K(size_t n, size_t rows, const void *x,
                            size_t row_stride, const void *y, float *out,
                            ggml_knc_workspace *workspaces, size_t workers) {
    size_t xb, yb, xextent, outbytes, wsbytes;
    ptrdiff_t r;
    int rc = knc_packed_sizes(n, &xb, &yb);
    if (rc != GGML_KNC_OK) return rc;
    if (rows == 0) return GGML_KNC_OK;
    if (workers == 0 || workers > INT_MAX) return GGML_KNC_ETHREADS;
#ifdef _OPENMP
    /* get_level also detects a serialized one-thread outer region. */
    if (omp_get_level() != 0) return GGML_KNC_ETHREADS;
#else
    if (workers != 1) return GGML_KNC_ETHREADS;
#endif
    if (rows > (size_t) PTRDIFF_MAX || rows > SIZE_MAX / sizeof(float) ||
        workers > SIZE_MAX / sizeof(*workspaces)) return GGML_KNC_ESIZE;
    if (!out) return GGML_KNC_EINVAL;
    if (!knc_is_aligned(out, sizeof(float))) return GGML_KNC_EALIGN;
    outbytes = rows * sizeof(float);
    if (!knc_range_valid(out, outbytes)) return GGML_KNC_ESIZE;
    if (n == 0) {
        for (r = 0; r < (ptrdiff_t) rows; ++r) out[r] = 0.0f;
        return GGML_KNC_OK;
    }
    if (row_stride < xb) return GGML_KNC_ESIZE;
    if (rows - 1 > (SIZE_MAX - xb) / row_stride) return GGML_KNC_ESIZE;
    xextent = (rows - 1) * row_stride + xb;
    wsbytes = workers * sizeof(*workspaces);
    if (!x || !y || !workspaces) return GGML_KNC_EINVAL;
    if (!knc_is_aligned(workspaces, GGML_KNC_ALIGNMENT)) return GGML_KNC_EALIGN;
    if (!knc_range_valid(x, xextent) || !knc_range_valid(y, yb) ||
        !knc_range_valid(workspaces, wsbytes)) return GGML_KNC_ESIZE;
    if (knc_overlap(out, outbytes, x, xextent) ||
        knc_overlap(out, outbytes, y, yb) ||
        knc_overlap(out, outbytes, workspaces, wsbytes) ||
        knc_overlap(workspaces, wsbytes, x, xextent) ||
        knc_overlap(workspaces, wsbytes, y, yb)) return GGML_KNC_EINVAL;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads((int) workers)
#endif
    for (r = 0; r < (ptrdiff_t) rows; ++r) {
        size_t worker = 0;
#ifdef _OPENMP
        worker = (size_t) omp_get_thread_num();
#endif
        out[r] = ggml_knc_dot_packed_unchecked(n / GGML_KNC_QK_K,
                    (const uint8_t *) x + (size_t) r * row_stride,
                    (const uint8_t *) y, &workspaces[worker]);
    }
    return GGML_KNC_OK;
}

int ggml_knc_dot_f32(size_t n, float *out, const float *x, const float *y) {
    size_t bytes;
    if (!out) return GGML_KNC_EINVAL;
    if (!knc_is_aligned(out, sizeof(float))) return GGML_KNC_EALIGN;
    if (!knc_range_valid(out, sizeof(float)) || n > SIZE_MAX / sizeof(float)) return GGML_KNC_ESIZE;
    if (n == 0) { *out = 0.0f; return GGML_KNC_OK; }
    if (!x || !y) return GGML_KNC_EINVAL;
    if (!knc_is_aligned(x, GGML_KNC_ALIGNMENT) ||
        !knc_is_aligned(y, GGML_KNC_ALIGNMENT)) return GGML_KNC_EALIGN;
    bytes = n * sizeof(float);
    if (!knc_range_valid(x, bytes) || !knc_range_valid(y, bytes)) return GGML_KNC_ESIZE;
    if (knc_overlap(out, sizeof(*out), x, bytes) ||
        knc_overlap(out, sizeof(*out), y, bytes)) return GGML_KNC_EINVAL;
    *out = ggml_knc_dot_f32_unchecked(n, x, y);
    return GGML_KNC_OK;
}
