/* SPDX-License-Identifier: MIT */
#ifndef GGML_KNC_MQ_H
#define GGML_KNC_MQ_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_KNC_MQ_MAX_ROWS 8

typedef void (*ggml_knc_mq_tile_fn)(size_t          blocks,
                                    size_t          rows,
                                    const uint8_t * weights,
                                    size_t          row_stride,
                                    const uint8_t * activations,
                                    float *         out);

/* Internal leaves: 1..8 rows, complete disjoint buffers, blocks > 0.
 * Packed inputs may be byte-aligned. Output must be float-aligned. */
ggml_knc_mq_tile_fn ggml_knc_mq_kernel(int bits);

void ggml_knc_mq_init(void);
int  ggml_knc_mq_enabled(void);
int  ggml_knc_mq_rows(void);
void ggml_knc_mq_optimized(int enabled);
void ggml_knc_mq_trace(int enabled);
void ggml_knc_mq_count(int bits, uint64_t rows, uint64_t staged_blocks, int node);
void ggml_knc_mq_report(void);

/* Diagnostic controls: set only while graph workers are idle. */
void ggml_knc_trace_enable(int enabled);
void ggml_knc_optimized_enable(int enabled);
void ggml_knc_extended_enable(int enabled);
void ggml_knc_trace_report(void);

#ifdef __cplusplus
}
#endif
#endif
