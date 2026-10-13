/* SPDX-License-Identifier: MIT */
#include "knc-mq.h"
#include "candidate-config.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static pthread_once_t once    = PTHREAD_ONCE_INIT;
static int            enabled = 1, tile_rows = 4, optimized = 1, tracing;
static uint64_t       rows[7], nodes[7], staged[7];

static void initialize(void) {
    const char * option = getenv("GGML_KNC_MQ");
    if (option) {
        if (!strcmp(option, "0")) {
            enabled = 0;
        } else if (strcmp(option, "1")) {
            fputs("GGML_KNC_MQ must be 0 or 1\n", stderr);
            abort();
        }
    }
    option = getenv("GGML_KNC_MQ_ROWS");
    if (option) {
        if (strlen(option) != 1 || !strchr("1248", option[0])) {
            fputs("GGML_KNC_MQ_ROWS must be 1, 2, 4 or 8\n", stderr);
            abort();
        }
        tile_rows = option[0] - '0';
    }
}

void ggml_knc_mq_init(void) {
    if (pthread_once(&once, initialize)) {
        abort();
    }
}

int ggml_knc_mq_enabled(void) {
    return enabled && optimized;
}

int ggml_knc_mq_rows(void) {
    return tile_rows;
}

void ggml_knc_mq_optimized(int value) {
    optimized = !!value;
}

void ggml_knc_mq_trace(int value) {
    tracing = !!value;
}

void ggml_knc_mq_count(int bits, uint64_t nrows, uint64_t nstaged, int node) {
    if (!tracing) {
        return;
    }
    __atomic_fetch_add(&rows[bits], nrows, __ATOMIC_RELAXED);
    __atomic_fetch_add(&staged[bits], nstaged, __ATOMIC_RELAXED);
    if (node) {
        __atomic_fetch_add(&nodes[bits], 1, __ATOMIC_RELAXED);
    }
}

void ggml_knc_mq_report(void) {
    ggml_knc_mq_init();
    fprintf(stderr,
            "KNC_MQ {\"candidate_id\":\"%s\",\"profile\":\"%s\",\"enabled\":%d,\"rows\":%d,"
            "\"l1_iters\":%d,\"l2_iters\":%d,\"prefetch_step\":%d,\"prefetch_q8\":%d,\"native_qualified\":false}\n",
            GGML_KNC_MQ_ID, GGML_KNC_MQ_PROFILE, ggml_knc_mq_enabled(), tile_rows, GGML_KNC_MQ_L1, GGML_KNC_MQ_L2,
            GGML_KNC_MQ_STEP, GGML_KNC_MQ_Q8);
    if (tracing) {
        for (int q = 3; q <= 6; ++q) {
            fprintf(stderr, "KNC_MQ_TRACE {\"bits\":%d,\"nodes\":%llu,\"rows\":%llu,\"staged_q8_blocks\":%llu}\n", q,
                    (unsigned long long) nodes[q], (unsigned long long) rows[q], (unsigned long long) staged[q]);
        }
    }
}
