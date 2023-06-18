#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
typedef int ggml_thread_ret_t;
#else
typedef void *ggml_thread_ret_t;
#endif

struct ggml_threading_context;

// Optional (experimental) features.
enum ggml_threading_features {
    GGML_THREADING_FEATURE_NONE = 0,
    GGML_THREADING_FEATURE_WAIT_ON_DONE = 1 << 0,
    GGML_THREADING_FEATURE_PERF = 1 << 1,
};

// The thread runner to feed into OS threads.
typedef ggml_thread_ret_t(ggml_threading_thread_runner)(void *data);

// Init and start underlying workers if n_threads > 1.
//
// thread: optional OS thread runner, default value:
// `ggml_threading_graph_compute_thread`.
//
// task_runner: default task runner, nullable wheen tensor.runner is not NULL.
//              Overridden by tensor.runner.
// features: configure threading behaviour, optional.
// threading additional features. see `ggml_threading_feature`, default 0.
//
// stages_time: optional for collecting per-stage wall clock time.
struct ggml_threading_context *
ggml_threading_start(int n_threads, ggml_threading_thread_runner *thread,
                     ggml_task_runner *task_runner,
                     enum ggml_threading_features features,
                     int64_t stages_time[3]);

// Stop workers (if exist), free memories (including the ctx).
void ggml_threading_stop(struct ggml_threading_context *ctx);

// The default implementation of `ggml_threading_thread_runner`
ggml_thread_ret_t ggml_threading_graph_compute_thread(void *data);

// Compute a tensor. It computes the enabled task stages one by one.
// Caller should take care of the return error: retry for fallback error.
enum ggml_compute_error
ggml_threading_compute_tensor(struct ggml_threading_context *ctx,
                              struct ggml_tensor *node, void *wdata,
                              size_t wsize);
#ifdef __cplusplus
}
#endif
