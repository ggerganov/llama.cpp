/* SPDX-License-Identifier: MIT
 * Exercise real GGML packing, barriers, direct dispatch and fallback paths.
 * Run on the card, or in an explicitly labeled host source-semantics build. */
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "knc-mq.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" void ggml_knc_optimized_enable(int);
extern "C" void ggml_knc_trace_enable(int);

static void compute(ggml_cgraph * graph, int workers, bool use_ref = false) {
    ggml_cplan                 plan = ggml_graph_plan(graph, workers, nullptr);
    plan.use_ref = use_ref;
    std::vector<unsigned char> work(plan.work_size + 64);
    plan.work_data = (uint8_t *) (((uintptr_t) work.data() + 63) & ~(uintptr_t) 63);
    if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
        std::abort();
    }
}

static void
check(ggml_type type, int k, int rows, int cols, int workers, int offset, bool strided, int batches, float scale = 3) {
    ggml_init_params ip  = { 2 * 1024 * 1024, nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        std::abort();
    }
    ggml_tensor * w = ggml_new_tensor_3d(ctx, type, k, rows, batches);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, cols, batches);
    if (strided) {
        x->nb[1] += 64;
        x->nb[2] = x->nb[1] * cols;
        x->nb[3] = x->nb[2] * batches;
    }
    ggml_tensor * out   = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    std::vector<unsigned char> weight(ggml_nbytes(w) + 128), shifted(weight.size() + 64);
    std::vector<float>         input((ggml_nbytes(x) + 3) / 4), values((size_t) k * rows * batches);
    std::vector<float>         result((size_t) rows * cols * batches), expected(result.size()), standard(result.size());
    unsigned char *            aligned = (unsigned char *) (((uintptr_t) weight.data() + 63) & ~(uintptr_t) 63);
    w->data                            = aligned;
    x->data                            = input.data();
    out->data                          = result.data();
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(i * 0.13f) * scale;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::cos(i * 0.07f) * 2;
    }
    if (type == GGML_TYPE_Q3_K) {
        quantize_row_q3_K_ref(values.data(), (block_q3_K *) aligned, values.size());
    } else if (type == GGML_TYPE_Q4_K) {
        quantize_row_q4_K_ref(values.data(), (block_q4_K *) aligned, values.size());
    } else if (type == GGML_TYPE_Q5_K) {
        quantize_row_q5_K_ref(values.data(), (block_q5_K *) aligned, values.size());
    } else if (type == GGML_TYPE_Q6_K) {
        quantize_row_q6_K_ref(values.data(), (block_q6_K *) aligned, values.size());
    } else {
        std::memcpy(aligned, values.data(), values.size() * sizeof(float));
    }
    ggml_knc_mq_optimized(0);
    compute(graph, workers, true);
    const std::vector<float> reference = result;
    ggml_knc_mq_optimized(1);
    compute(graph, workers, true);
    if (std::memcmp(result.data(), reference.data(), result.size() * sizeof(float)) != 0) {
        std::fputs("FAIL reference mode depends on KNC staging policy\n", stderr);
        std::exit(1);
    }
    ggml_knc_optimized_enable(0);
    compute(graph, workers);
    expected = result;
    std::memcpy(shifted.data() + offset, aligned, ggml_nbytes(w));
    w->data = shifted.data() + offset;
    ggml_knc_optimized_enable(1);
    ggml_knc_mq_optimized(0);
    compute(graph, workers);
    standard = result;
    ggml_knc_mq_optimized(1);
    compute(graph, workers);
    for (size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i]) ||
            std::fabs(result[i] - expected[i]) > 0.005f + 0.0005f * std::fabs(expected[i])) {
            std::fprintf(
                stderr,
                "FAIL type=%d k=%d rows=%d cols=%d workers=%d offset=%d strided=%d batches=%d i=%zu got=%g expected=%g\n",
                type, k, rows, cols, workers, offset, strided, batches, i, result[i], expected[i]);
            std::exit(1);
        }
#ifdef __k1om__
        if (std::memcmp(&result[i], &standard[i], sizeof(float))) {
            std::fprintf(stderr, "FAIL native rounding compatibility type=%d k=%d scale=%g i=%zu got=%g standard=%g\n",
                         type, k, scale, i, result[i], standard[i]);
            std::exit(1);
        }
#endif
    }
    ggml_free(ctx);
}

int main() {
    ggml_knc_mq_init();
    ggml_cpu_init();
    ggml_knc_trace_enable(1);
    for (ggml_type type : { GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
        for (int k : { 256, 512, 2048 }) {
            for (int workers : { 1, 3, 8 }) {
                check(type, k, 33, 1, workers, 0, false, 1);
                check(type, k, 17, 3, workers, 1, false, 1);
            }
        }
        for (int offset : { 0, 1, 15, 16, 31, 48, 63 }) {
            check(type, 512, 17, 3, 3, offset, false, 1);
        }
        for (int rows : { 1, 7, 8, 15, 16, 31, 32, 63, 64, 65 }) {
            check(type, 256, rows, 2, 8, 0, false, 1);
        }
        check(type, 512, 17, 3, 3, 0, true, 1);
        check(type, 512, 17, 3, 3, 0, false, 2);
    }
    check(GGML_TYPE_F32, 512, 17, 3, 3, 0, false, 1);
    for (ggml_type type : { GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
        for (float scale : { 0.003f, 0.02f, 1.0f, 13.5f }) {
            check(type, 2048, 33, 3, 8, 15, false, 1, scale);
        }
    }
    ggml_knc_mq_report();
    std::puts("PASS candidate graph/oracle comparisons; no model or throughput acceptance");
}
