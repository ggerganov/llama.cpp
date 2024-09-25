#pragma once

#include "ggml-impl.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// op profile data (per op / per thread)
enum ggml_profile_event {
    GGML_PROF_OP_START,
    GGML_PROF_OP_SYNC,
    GGML_PROF_OP_END
};

struct ggml_profile_data {
    uint64_t nsec[GGML_PROF_OP_END + 1]; // event times in nsec
};

#ifndef GGML_GRAPH_PROFILER

// Stub out all profiler functions

static inline void ggml_profile_graph_init(struct ggml_cgraph *cg, int n_threads)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(n_threads);
}

static inline void ggml_profile_graph_start(struct ggml_cgraph *cg, int n_threads)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(n_threads);
}

static inline void ggml_profile_graph_finish(struct ggml_cgraph *cg, int n_threads)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(n_threads);
}

static inline void ggml_profile_graph_free(struct ggml_cgraph *cg)
{
    GGML_UNUSED(cg);
}

static inline void ggml_profile_op_event(const struct ggml_cgraph *cg, enum ggml_profile_event e, int node_n, int ith)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(e);
    GGML_UNUSED(node_n);
    GGML_UNUSED(ith);
}

#else

void ggml_profile_graph_init(struct ggml_cgraph *cg, int n_threads);
void ggml_profile_graph_start(struct ggml_cgraph *cg, int n_threads);
void ggml_profile_graph_finish(struct ggml_cgraph *cg, int n_threads);
void ggml_profile_graph_free(struct ggml_cgraph *cg);
void ggml_profile_op_event(const struct ggml_cgraph *cg, enum ggml_profile_event e, int node_n, int ith);

#endif // GGML_GRAPH_PROFILER

#ifdef __cplusplus
}
#endif
