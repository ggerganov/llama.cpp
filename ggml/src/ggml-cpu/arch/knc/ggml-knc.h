/* SPDX-License-Identifier: MIT
 * Standalone Knights Corner IMCI kernels. See README for integration gates.
 */
#ifndef GGML_KNC_H
#define GGML_KNC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_KNC_ALIGNMENT 64
#define GGML_KNC_QK_K 256
#define GGML_KNC_Q5_K_BYTES 176
#define GGML_KNC_Q8_K_BYTES 292
#define GGML_KNC_WORKSPACE_BYTES 576

enum ggml_knc_status {
    GGML_KNC_OK = 0,
    GGML_KNC_EINVAL = 1,
    GGML_KNC_EALIGN = 2,
    GGML_KNC_ESIZE = 3,
    GGML_KNC_ETHREADS = 4,
    GGML_KNC_ENOMEM = 5
};

/* Public solely to allow fixed, compiler-aligned stack/arena storage. Fields
 * are implementation-private; do not prefill them. Packed vector loads use
 * the first three 64-byte-aligned arrays; scale broadcasts read aligned
 * int32 fields. Fixed aligned reduction output is separate stack scratch.
 * sizeof is a multiple of 64, so an aligned array is safe per-worker storage.
 */
typedef struct __attribute__((aligned(GGML_KNC_ALIGNMENT))) ggml_knc_workspace {
    uint8_t internal_q4[128];
    uint8_t internal_qh[64];
    int8_t internal_q8[256];
    int16_t internal_bsums[16];
    int32_t internal_scales[8];
    int32_t internal_mins[8];
} ggml_knc_workspace;

/* Preparation/teardown only. Allocation is 64-byte aligned, zero-filled and
 * first-touched; bytes==0 or a size overflow returns NULL. Free NULL is safe.
 * A model integration must also allocate tensor/KV arenas with 64-byte bases
 * and align any F32 subviews used by ggml_knc_dot_f32.
 */
void *ggml_knc_aligned_alloc(size_t bytes);
void ggml_knc_aligned_free(void *ptr);
size_t ggml_knc_workspace_size(void);
ggml_knc_workspace *ggml_knc_workspace_alloc(size_t worker_count);
void ggml_knc_workspace_free(ggml_knc_workspace *workspaces);

/* QK_K=256, little-endian standard GGML packed layout; no requantization.
 * x holds n/256 blocks of 176 bytes; y holds n/256 blocks of 292 bytes.
 * n must be divisible by 256. Packed input addresses may have ANY alignment;
 * base alignment cannot align the embedded fields or every packed block.
 * They are copied to bounded aligned scratch before converted IMCI loads.
 * out must point to a naturally aligned float. Scratch must be 64-aligned
 * and exclusive to this call. No allocation, OpenMP region, or TLS access.
 * n==0 writes zero and permits NULL x/y/workspace. All calls require out.
 * Writable ranges must not overlap inputs or each other. Inputs may overlap
 * one another. The caller must provide the complete indicated ranges.
 * On error out is unchanged; overflow is ESIZE, overlap/null is EINVAL.
 */
int ggml_knc_dot_q5_K_q8_K(size_t n, float *out, const void *x,
                         const void *y, ggml_knc_workspace *workspace);

/* One already-quantized Q8_K vector against independent Q5_K rows.
 * Call outside an OpenMP parallel region; do not nest this inside GGML's
 * own thread pool. Existing GGML worker threads should call the dot above.
 * row_stride >= (n/256)*176; checked ranges include intervening padding.
 * Allocate worker_count workspaces once before execution; the pool length
 * is a caller contract. Static scheduling uses one exclusive slot per worker.
 * worker_count must be 1..INT_MAX; non-OpenMP builds only accept 1.
 * rows==0 is a no-op (n must still be divisible by 256).
 * n==0 fills out with zeros and permits NULL x/y/workspaces.
 */
int ggml_knc_matvec_q5_K_q8_K(size_t n, size_t rows, const void *x,
                            size_t row_stride, const void *y, float *out,
                            ggml_knc_workspace *workspaces,
                            size_t worker_count);

/* F32 auxiliary kernel: x and y must be 64-byte aligned for n>0. Four
 * independent FMA chains cover 64-element batches, then 16-wide vectors,
 * then an exact-bound scalar tail. No overreads or allocation.
 * n==0 writes zero and permits NULL x/y. out cannot overlap x or y.
 */
int ggml_knc_dot_f32(size_t n, float *out, const float *x, const float *y);

#ifdef __cplusplus
}
#endif
#endif
