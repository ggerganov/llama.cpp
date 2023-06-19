#include "ggml-threading.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Purposes:
// 1. general overview of the threading behaviors.
// 2. race (dead lock) detection.

// # build
// cd build
//
// # build release:
//   cmake .. && cmake --build . --config Release
//
// # build with sanitize:
//   cmake .. -DLLAMA_SANITIZE_THREAD=ON && cmake --build . --config Release
//
// # run:
// ./bin/test-ggml-threading

// How to turn off the warning on Apple: malloc: nano zone abandoned due to
// inability to reserve vm space?
// ==> export MallocNanoZone=0, no need to rebuild.
// See `nano_init()` from
// https://opensource.apple.com/source/libmalloc/libmalloc-140.40.1/src/nano_malloc.c.auto.html

// How to view the threading debug:
// ==> uncomment `#define GGML_THREADING_DEBUG 1` from file ggml-threading.c

#define UNUSED(x) (void)(x)

#define MAX_N_THREADS 16

static const int n_repeat = 10;

// It's frustrating to use atomic with c11 on Windows, let's replace atomic
// counter with array.
static int work_done_arr[MAX_N_THREADS];

static enum ggml_compute_error
mock_task_runner(const struct ggml_compute_params *params,
                 struct ggml_tensor *node) {
    int64_t loops = node->task_profile.dev_flags[1] * 1000 * 1000;
    if (node->task_profile.stages[params->type].parallel) {
        loops /= params->nth;
    }

    volatile int64_t j = 0;
    for (int i = 0; i < loops; i++) {
        j++;
    }

    UNUSED(j);

    work_done_arr[params->ith]++;
    return GGML_COMPUTE_OK;
}

static int test_driver(int id, struct ggml_tensor *node, int n_threads) {
    uint8_t loops = node->task_profile.dev_flags[1];
    printf(
        "\n[test-ggml-threading] #%02d, workload: %2d million(s), n_threads: "
        "%2d\n",
        id, loops, n_threads);

    for (int i = 0; i < n_threads; i++) {
        work_done_arr[i] = 0;
    }

    bool wait_on_done = (node->task_profile.dev_flags[0] > 0u);

    enum ggml_threading_features features = GGML_THREADING_FEATURE_PERF;
    if (wait_on_done) {
        features |= GGML_THREADING_FEATURE_WAIT_ON_DONE;
    }

    int t0 = (int)ggml_time_us();

    node->task_profile.runner = mock_task_runner;

    struct ggml_threading_context *ctx = ggml_threading_start(
        n_threads, NULL, NULL, features, /*stages_time*/ NULL);

    int t1 = (int)ggml_time_us();

    for (int i = 0; i < n_repeat; i++) {
        ggml_threading_compute_tensor(ctx, node, /*wdata*/ NULL, /*wsize*/ 0);
    }

    int t2 = (int)ggml_time_us();

    ggml_threading_stop(ctx);

    int t3 = (int)ggml_time_us();

    const struct ggml_task_stage *stages = node->task_profile.stages;

    int expect = 0;
    for (int i = 0; i < 3; i++) {
        const struct ggml_task_stage *ts = &stages[i];
        if (ts->valid) {
            if (ts->parallel) {
                expect += n_threads;
            } else {
                expect++;
            }
        }
    }
    expect *= n_repeat;

    int actual = 0;
    for (int i = 0; i < n_threads; i++) {
        actual += work_done_arr[i];
    }

    printf("\tstage-0: parallel: %d, wait: %d\n\tstage-1: parallel: %d, wait: "
           "%d, wait_on_done: %d %s\n",
           stages[0].parallel, stages[0].wait, stages[1].parallel,
           stages[1].wait, wait_on_done, stages[1].wait ? "<--------" : "");

    if (actual == expect) {
        printf("\tthreading: init %6.3f ms, compute %6.3f ms, cleanup %6.3f "
               "ms, total %6.3f ms\n",
               1.0 * (t1 - t0) / 1000, 1.0 * (t2 - t1) / 1000,
               1.0 * (t3 - t2) / 1000, 1.0 * (t3 - t0) / 1000);
        return 0;
    }

    printf("\t== failed. expect %d done, actual %d done\n\n", expect, actual);

    return 2;
}

static enum ggml_compute_error
mock_task_runner_fallback(const struct ggml_compute_params *params,
                          struct ggml_tensor *node) {
    UNUSED(params);

    // failed to run ...
    if (node->task_profile.id == 2) {
        return GGML_COMPUTE_FALLBACK;
    }
    return GGML_COMPUTE_OK;
}

// By design, fallback should happen when attempt computing tensor in GPU,
// thus it is not parallelled.
static int test_fallback(struct ggml_tensor *node) {
    struct ggml_threading_context *ctx = ggml_threading_start(
        1, NULL, mock_task_runner_fallback,
        /*features*/ GGML_THREADING_FEATURE_NONE, /*stages_time*/ NULL);

    enum ggml_compute_error err =
        ggml_threading_compute_tensor(ctx, node, /*wdata*/ NULL, /*wsize*/ 0);
    if (err == GGML_COMPUTE_FALLBACK) {
        // mock setup new profile ...
        node->task_profile.id = 1;

        err = ggml_threading_compute_tensor(ctx, node, /*wdata*/ NULL,
                                            /*wsize*/ 0);
    }

    ggml_threading_stop(ctx);
    if (err != GGML_COMPUTE_OK) {
        printf("ggml_threading_compute_tensor failed with error: %d.\n", err);
        return 1;
    }

    return 0;
}

static enum ggml_compute_error
customized_node_runner(const struct ggml_compute_params *params,
                       struct ggml_tensor *node) {
    UNUSED(params);
    // Reset runner thus caller will know it was called.
    node->task_profile.runner = NULL;
    return GGML_COMPUTE_OK;
}

// Test when node->task_profile.runner is not NULL.
static int test_customized_node_runner(struct ggml_tensor *node) {
    struct ggml_threading_context *ctx = ggml_threading_start(
        1, NULL, mock_task_runner,
        /*features*/ GGML_THREADING_FEATURE_NONE, /*stages_time*/ NULL);

    node->task_profile.runner = customized_node_runner;
    enum ggml_compute_error err =
        ggml_threading_compute_tensor(ctx, node, /*wdata*/ NULL, /*wsize*/ 0);

    ggml_threading_stop(ctx);
    if (err != GGML_COMPUTE_OK) {
        // should not happen.
        abort();
    }

    if (node->task_profile.runner != NULL) {
        return 2;
    }

    return 0;
}

static enum ggml_compute_error
lifecycle_runner(const struct ggml_compute_params *params,
                 struct ggml_tensor *node) {
    UNUSED(params);
    UNUSED(node);
    return GGML_COMPUTE_OK;
}

// Test thread lifecycle: start -> suspend -> resume -> stop
static int test_lifecycle(bool wait_on_done) {
    struct ggml_tensor node;
    memset(&node, 0, sizeof(struct ggml_tensor));

    struct ggml_task_stage *stages = node.task_profile.stages;

    stages[0].valid = true;
    stages[1].valid = true;
    stages[1].parallel = true;

    node.op = GGML_OP_MUL_MAT;
    struct ggml_tensor src0 = {
        .type = GGML_TYPE_Q4_0,
    };
    struct ggml_tensor src1 = {
        .type = GGML_TYPE_F32,
    };
    node.src0 = &src0;
    node.src1 = &src1;

    int t0 = (int)ggml_time_ms();
    // Suppose creating threading when entering session.

    // We have to try affable threads.
    struct ggml_threading_context *ctx = NULL;
    int threads_arr[] = {4, 2};
    int threads_arr_len = sizeof(threads_arr) / sizeof(threads_arr[0]);
    int n_threads = 1;

    enum ggml_threading_features features =
        wait_on_done ? GGML_THREADING_FEATURE_NONE
                     : GGML_THREADING_FEATURE_WAIT_ON_DONE;
    for (int i = 0; i < threads_arr_len; i++) {
        n_threads = threads_arr[i];
        int start_time = (int)ggml_time_ms();
        ctx = ggml_threading_start(n_threads, NULL, lifecycle_runner,
                                   features | GGML_THREADING_FEATURE_PERF,
                                   /*stages_time*/ NULL);
        int elapsed = (int)ggml_time_ms() - start_time;
        if (elapsed > 1 * n_threads) {
            printf("[test-ggml-threading] %s: it took %d ms to start %d worker "
                   "thread(s), skip\n",
                   __func__, elapsed, n_threads - 1);
            ggml_threading_stop(ctx);
        } else {
            break;
        }
    }

    if (n_threads == 1) {
        printf("[test-ggml-threading] %s: too slow to start at least 1 worker "
               "thread(s), skip\n",
               __func__);
        return 0;
    }

    // Suppose exiting from previous compute graph ...
    printf("[test-ggml-threading] %s: %d workers, suspending ...\n", __func__,
           n_threads - 1);
    ggml_threading_suspend(ctx);

    // Suppose entering new compute graph ...
    printf("[test-ggml-threading] test lifecycle: resuming ...\n");
    ggml_threading_resume(ctx);

    const int m = 2;
    const int n = 10;

    printf("[test-ggml-threading] %s: computing %d tensors (half wait)...\n",
           __func__, m * n);

    for (int i = 0; i < m; i++) {
        stages[0].wait = (i == 0);
        for (int j = 0; j < n; j++) {
            ggml_threading_compute_tensor(ctx, &node, /*wdata*/ NULL,
                                          /*wsize*/ 0);
        }
    }

    printf("[test-ggml-threading] %s: compute done, resuming...\n", __func__);
    ggml_threading_resume(ctx);

    const int loops = 10;
    printf("[test-ggml-threading] %s: try %d loops of suspend-resume ...\n",
           __func__, loops);

    for (int i = 0; i < loops; i++) {
        ggml_threading_suspend(ctx);
        if (!ggml_threading_is_suspending(ctx)) {
            abort();
        }

        ggml_threading_resume(ctx);
        if (ggml_threading_is_suspending(ctx)) {
            abort();
        }
    }

    printf("[test-ggml-threading] %s: stopping ...\n", __func__);
    ggml_threading_stop(ctx);

    int elapsed_ms = (int)ggml_time_ms() - t0;
    printf("[test-ggml-threading] %s: elapsed %d ms\n", __func__, elapsed_ms);

    return 0;
}

int main(void) {
    ggml_time_init();

    struct ggml_tensor node;
    memset(&node, 0, sizeof(struct ggml_tensor));
    node.task_profile.runner = mock_task_runner;

    struct ggml_task_stage *stages = node.task_profile.stages;

    stages[0].valid = true;
    stages[1].valid = true;

    int n_passed = 0;
    int n_tests = 0;

    // In github build actions (windows-latest-cmake and ubuntu-latest-cmake):
    // When n_threads >= 4, the thread init time and compute time suddenly goes
    // down to 100x ~ 1000x slow -- comparing to n_threads == 2.
    //
    // But the tests (n_threads 1, 2, 4, 6) looks sound on my devices:
    // - MacBook air 2013, ubuntu 22.04
    // - MacBook pro 2018, macOS 13.4
    //
    // So I assume the github build host has limited multi-cpu quota.
    // Will skip computing when threading init time is too slow.
    //
    // NOTE: it's observed that when workload is 0 and n_threads >= number of
    // physical cores:
    // - the wait/wakeup time varies much: can be up to tens or hundreds of the
    //   average time, thus greatly punishes those small workloads.
    // - wait_on_done is general faster than wait_now, can be 10x faster.

    int threads_arr[] = {1, 2, 4, 6, 8, 16};
    int threads_arr_len = sizeof(threads_arr) / sizeof(threads_arr[0]);

    // millions of loops.
    uint8_t workload_arr[] = {0u, 1u, 10u};
    int workload_arr_len = sizeof(workload_arr) / sizeof(workload_arr[0]);

    // skip slow/big n_threads.

    int n_slow = 0;

    for (int i = 0; i < threads_arr_len; i++) {
        int n_threads = threads_arr[i];

        // At github, Windows can take more than 20 seconds to start 15 threads.
        // Let's silently ignore when we saw two adjacent slowness.
        if (n_slow >= 2) {
            threads_arr[i] = 0;
            continue;
        }

        if (n_threads == 1) {
            continue;
        } else if (n_threads > MAX_N_THREADS) {
            printf("[test-ggml-threading] warning: the n_threads (%d) is too "
                   "big, allow at most %d, skip.\n",
                   n_threads, MAX_N_THREADS);
            threads_arr[i] = 0;
            continue;
        }

        // skip this n_threads when too slow.
        int t0 = (int)ggml_time_ms();

        struct ggml_threading_context *ctx =
            ggml_threading_start(n_threads, ggml_threading_graph_compute_thread,
                                 NULL, 0, /*stages_time*/ NULL);

        int t1 = (int)ggml_time_ms();

        ggml_threading_stop(ctx);

        int elapsed_ms = t1 - t0;
        if (elapsed_ms > 1 * n_threads) {
            printf("[test-ggml-threading] warning: it took took %7.3f "
                   "ms to start %2d worker thread(s). Too slow, skip.\n",
                   1.0 * elapsed_ms, n_threads - 1);
            threads_arr[i] = 0;
            ++n_slow;
        } else {
            // clear.
            n_slow = 0;
        }
    }

    // node.task_profile.dev_flags: byte 0 for wait_on_done, byte 1 for loops.

    for (int x = 0; x < workload_arr_len; x++) {
        node.task_profile.dev_flags[1] = workload_arr[x];

        for (int i = 0; i < threads_arr_len; i++) {
            int n_threads = threads_arr[i];
            if (n_threads <= 0) {
                continue;
            }

            printf("\n[test-ggml-threading] ==== workload: %2d million(s), "
                   "n_threads: %2d ====\n",
                   workload_arr[x], n_threads);

            // multi-threads: parallel + wait_now/wait_on_done

            if (n_threads == 1) {
                stages[0].parallel = false;
                stages[1].parallel = false;
                stages[0].wait = false;
                stages[1].wait = false;

                node.task_profile.dev_flags[0] = 0u;

                n_tests++;
                if (test_driver(n_tests, &node, n_threads) == 0) {
                    n_passed++;
                }
                continue;
            }

            { // no parallel, no wait
                stages[0].parallel = false;
                stages[1].parallel = false;
                stages[0].wait = false;
                stages[1].wait = false;

                node.task_profile.dev_flags[0] = 0u;

                n_tests++;
                if (test_driver(n_tests, &node, n_threads) == 0) {
                    n_passed++;
                }
            }

            { // both parallel, no wait
                stages[0].parallel = true;
                stages[1].parallel = true;
                stages[0].wait = false;
                stages[1].wait = false;

                node.task_profile.dev_flags[0] = 0u;

                n_tests++;
                if (test_driver(n_tests, &node, n_threads) == 0) {
                    n_passed++;
                }
            }

            { // stage 0 parallel, stage 1 may wait
                stages[0].parallel = true;
                stages[1].parallel = false;
                stages[0].wait = false;

                { // stage 1 no wait
                    stages[1].wait = false;
                    node.task_profile.dev_flags[0] = 0u;

                    n_tests++;
                    if (test_driver(n_tests, &node, n_threads) == 0) {
                        n_passed++;
                    }
                }

                { // stage 1 wait
                    stages[1].wait = true;
                    if (stages[1].parallel) {
                        abort();
                    }

                    { // disable wait_on_done
                        node.task_profile.dev_flags[0] = 0u; // wait now.

                        n_tests++;
                        if (test_driver(n_tests, &node, n_threads) == 0) {
                            n_passed++;
                        }
                    }

                    { // enable wait_on_done
                        node.task_profile.dev_flags[0] = 1u; // wait on done

                        n_tests++;
                        if (test_driver(n_tests, &node, n_threads) == 0) {
                            n_passed++;
                        }
                    }
                }
            }
        }
    }

    // fallback
    {
        printf("[test-ggml-threading] test fallback ...\n");

        ++n_tests;

        // required by getting task profiles.
        node.op = GGML_OP_MUL_MAT;
        struct ggml_tensor src0 = {
            .type = GGML_TYPE_Q4_0,
        };
        struct ggml_tensor src1 = {
            .type = GGML_TYPE_F32,
        };
        node.src0 = &src0;
        node.src1 = &src1;

        node.task_profile.id = 2;
        stages[1].valid = true;
        if (test_fallback(&node) == 0) {
            ++n_passed;
            printf("[test-ggml-threading] test fallback: ok\n\n");
        }
    }

    // customized node runner
    {
        printf("[test-ggml-threading] test customized node runner ...\n");
        ++n_tests;

        if (test_customized_node_runner(&node) == 0) {
            ++n_passed;
            printf("[test-ggml-threading] test customized node runner: ok\n\n");
        }
    }

    // lifecycle.
    for (int i = 0; i < 2; i++) {
        bool wait_on_done = (i == 1);
        printf("[test-ggml-threading] test lifecycle (want_on_done = %d) ...\n",
               wait_on_done);
        ++n_tests;

        if (test_lifecycle(wait_on_done) == 0) {
            ++n_passed;
            printf("[test-ggml-threading] test lifecycle (want_on_done = %d): "
                   "ok\n\n",
                   wait_on_done);
        }
    }

    printf("[test-ggml-threading] %d/%d passed.\n", n_passed, n_tests);

    return (n_passed == n_tests) ? 0 : 1;
}
