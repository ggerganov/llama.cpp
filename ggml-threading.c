
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-threading.h"
#include "ggml.h"

#define UNUSED(x) (void)(x)

// see https://github.com/ggerganov/llama.cpp/pull/1314
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#include <emmintrin.h>
static inline void ggml_spin_pause(void) { _mm_pause(); }
#else
static inline void ggml_spin_pause(void) {}
#endif

#if defined(_WIN32)

#include <windows.h>

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef LONG atomic_flag;

typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef void pthread_mutexattr_t;
typedef void pthread_condattr_t;

typedef HANDLE pthread_t;

static void atomic_store(atomic_int *ptr, LONG val) {
    InterlockedExchange(ptr, val);
}

static LONG atomic_load(atomic_int *ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}

static LONG atomic_fetch_add(atomic_int *ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}

static LONG atomic_fetch_sub(atomic_int *ptr, LONG dec) {
    return atomic_fetch_add(ptr, -(dec));
}

static inline LONG atomic_flag_test_and_set(volatile atomic_flag *ptr) {
    return InterlockedCompareExchange(ptr, 1, 0);
}
static inline LONG atomic_flag_clear(volatile atomic_flag *ptr) {
    return InterlockedExchange(ptr, 0);
}
static int pthread_create(pthread_t *out, void *unused,
                          ggml_thread_ret_t (*func)(void *), void *arg) {
    (void)unused;
    HANDLE handle =
        CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, arg, 0, NULL);
    if (handle == NULL) {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void *unused) {
    (void)unused;
    return (int)WaitForSingleObject(thread, INFINITE);
}

static int pthread_mutex_init(pthread_mutex_t *mutex,
                              pthread_mutexattr_t *attr) {
    (void)attr;
    InitializeCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_destroy(pthread_mutex_t *mutex) {
    DeleteCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_lock(pthread_mutex_t *mutex) {
    EnterCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    LeaveCriticalSection(mutex);
    return 0;
}

static int pthread_cond_init(pthread_cond_t *cond, pthread_condattr_t *attr) {
    (void)attr;
    InitializeConditionVariable(cond);
    return 0;
}

static int pthread_cond_destroy(pthread_cond_t *cond) {
    (void)cond;
    return 0;
}

static int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    SleepConditionVariableCS(cond, mutex, INFINITE);
    return 0;
}

static int pthread_cond_signal(pthread_cond_t *cond) {
    WakeConditionVariable(cond);
    return 0;
}

static int pthread_cond_broadcast(pthread_cond_t *cond) {
    WakeAllConditionVariable(cond);
    return 0;
}

static int sched_yield(void) {
    // https://learn.microsoft.com/en-us/windows/win32/api/winnt/nf-winnt-yieldprocessor
    YieldProcessor();
    return 0;
}

#else // ! _WIN32

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>

#endif

// #define GGML_THREADING_DEBUG 1

#ifdef GGML_THREADING_DEBUG
#define PRINT_DEBUG(...) fprintf(stdout, __VA_ARGS__)
#else
#define PRINT_DEBUG(...)
#endif

struct ggml_perf_stats {
    int runs;

    // total cycles
    atomic_int cycles;

    // total time in us.
    atomic_int time_us;
};

struct ggml_compute_state_shared {
    atomic_flag spin;
    pthread_mutex_t mutex;
    pthread_cond_t cond;

    // number of threads that has entered thread runner.
    atomic_int n_ready;

    // number of assigned but unfinished tasks, workers decrease it.
    atomic_int n_tasks;

    // number of waiting workers, workers increase it.
    atomic_int n_waiting;

    // commands.
    atomic_bool wait_now;
    atomic_bool wait_on_done;
    atomic_bool stop;

    ggml_threading_task_runner *task_runner;

    struct ggml_threading_context *ctx;
};
struct ggml_compute_state {
    pthread_t thrd;

    atomic_bool has_work;
    struct ggml_compute_params params;
    struct ggml_tensor *node;

    struct ggml_compute_state_shared *shared;
};
struct ggml_threading_context {
    int n_threads;
    struct ggml_compute_state_shared shared;
    struct ggml_compute_state *workers;

    enum ggml_threading_features features;

    struct ggml_perf_stats wait_perf;
    struct ggml_perf_stats wakeup_perf;

    int64_t *stages_time;
};

// NOTE: ggml_spin_lock and ggml_spin_unlock may can be noop if
// feature wait_on_done is off.
static inline void ggml_spin_lock(volatile atomic_flag *obj) {
    while (atomic_flag_test_and_set(obj)) {
        ggml_spin_pause();
    }
}

static inline void ggml_spin_unlock(volatile atomic_flag *obj) {
    atomic_flag_clear(obj);
}

static inline void ggml_perf_collect(struct ggml_perf_stats *st, int64_t c0,
                                     int64_t t0) {
    st->runs++;
    st->cycles += (ggml_cycles() - c0);
    st->time_us += (ggml_time_us() - t0);
}

// A worker thread goes cond waiting.
// NOTE: must be protected by shared->spin
static void ggml_threading_cond_wait(struct ggml_compute_state *state) {
    struct ggml_compute_state_shared *shared = state->shared;

    int64_t perf_cycles_0 = 0;
    int64_t perf_time_0 = 0;

    if (shared->ctx->features & GGML_THREADING_FEATURE_PERF) {
        perf_cycles_0 = ggml_cycles();
        perf_time_0 = ggml_time_us();
    }

    GGML_ASSERT(pthread_mutex_lock(&shared->mutex) == 0);

    if (!shared->wait_now) {
        GGML_ASSERT(pthread_mutex_unlock(&shared->mutex) == 0);
        ggml_spin_unlock(&shared->spin);
        return;
    }

    shared->n_waiting++;
    ggml_spin_unlock(&shared->spin);

    GGML_ASSERT(pthread_cond_wait(&shared->cond, &shared->mutex) == 0);
    GGML_ASSERT(pthread_mutex_unlock(&shared->mutex) == 0);

    ggml_spin_lock(&shared->spin);

    shared->n_waiting--;

    if (shared->ctx->features & GGML_THREADING_FEATURE_PERF) {
        ggml_perf_collect(&shared->ctx->wait_perf, perf_cycles_0, perf_time_0);
    }
}

// Wakeup all workers.
//
// Workers takes some time to wakeup, and has to lock spin after wakeup. Yield
// is used to avoid signal frequently. Current implementation is highly
// experimental. See tests/test-ggml-threading.c for details.
//
// NOTE: must be protected by shared->spin
static void
ggml_threading_wakeup_workers(struct ggml_compute_state_shared *shared) {
    int64_t perf_cycles_0 = 0;
    int64_t perf_time_0 = 0;

    if (shared->ctx->features & GGML_THREADING_FEATURE_PERF) {
        perf_cycles_0 = ggml_cycles();
        perf_time_0 = ggml_time_us();
    }

    shared->wait_now = false;

    int loop_counter = 0;
    int notify_counter = 0;
    int64_t last_signal_time = 0;

    while (shared->n_waiting != 0) {
        ggml_spin_unlock(&shared->spin);

        if (loop_counter > 0) {
            ggml_spin_pause();
            if (loop_counter > 3) {
                sched_yield();
            }
        }
        ++loop_counter;

        // TODO: should bench actual average wait/wakeup time.
        if (last_signal_time > 0 && (ggml_time_us() - last_signal_time) < 10) {
            continue;
        }

        GGML_ASSERT(pthread_mutex_lock(&shared->mutex) == 0);
        GGML_ASSERT(pthread_cond_broadcast(&shared->cond) == 0);
        GGML_ASSERT(pthread_mutex_unlock(&shared->mutex) == 0);
        ++notify_counter;
        last_signal_time = ggml_time_us();

        ggml_spin_lock(&shared->spin);
    }

    if (shared->ctx->features & GGML_THREADING_FEATURE_PERF) {
        ggml_perf_collect(&shared->ctx->wakeup_perf, perf_cycles_0,
                          perf_time_0);
    }

    // if (notify_counter > 1) {
    //     printf("%s: loop counter: %d, notify counter: %d\n", __func__,
    //            loop_counter, notify_counter);
    // }
    UNUSED(notify_counter);
}

// Setup workers for a task stage.
// NOTE: must be protected by shared->spin
static void ggml_threading_setup_workers(struct ggml_threading_context *ctx,
                                         struct ggml_task_profile *profile,
                                         enum ggml_task_type type) {
    PRINT_DEBUG("[main] setup workers for task ...\n");

#ifdef GGML_THREADING_DEBUG
    int64_t t0 = ggml_time_us();
#endif

    const int n_worker_threads = ctx->n_threads - 1;
    struct ggml_task_stage *current = &profile->stages[type];
    struct ggml_compute_state_shared *shared = &ctx->shared;

    if (current->parallel) {
        if (shared->n_waiting > 0) {
            ggml_threading_wakeup_workers(shared);
        }

        if ((ctx->features & GGML_THREADING_FEATURE_WAIT_ON_DONE) > 0) {
            // Optimize energy: wait_on_done. We MAY also check following nodes,
            // but that's a bit complicated.
            shared->wait_on_done = false;
            for (int i = type + 1; i <= GGML_TASK_FINALIZE; i++) {
                struct ggml_task_stage *next = &profile->stages[i];
                if (next->parallel) {
                    break;
                }
                if (next->wait) {
                    shared->wait_on_done = true;
                    PRINT_DEBUG("[main] wait_on_done is enabled for "
                                "current task stage\n");
                    break;
                }
            }
        }
    } else if (current->wait) {
        if (shared->n_waiting < n_worker_threads) {
            shared->wait_now = true;
            PRINT_DEBUG("[main] wait_now was set, expect %d workers wait\n",
                        n_worker_threads);
            ggml_spin_unlock(&shared->spin);

            while (shared->n_waiting != n_worker_threads) {
                ggml_spin_pause();
            }

            ggml_spin_lock(&shared->spin);
            PRINT_DEBUG("[main] saw %d workers waiting\n", n_worker_threads);
        }
    }

    PRINT_DEBUG("[main] setup workers for task took %d us\n",
                (int)(ggml_time_us() - t0));
}

ggml_thread_ret_t ggml_threading_graph_compute_thread(void *data) {
    GGML_ASSERT(data);
    struct ggml_compute_state *state = (struct ggml_compute_state *)data;
    GGML_ASSERT(state);

    struct ggml_compute_state_shared *shared = state->shared;
    GGML_ASSERT(shared);
    GGML_ASSERT(shared->task_runner);

    shared->n_ready++;

    PRINT_DEBUG("[%d-th] running\n", state->params.ith);

    while (!shared->stop) {
        if (shared->wait_now) {
            ggml_spin_lock(&shared->spin);
            if (!state->has_work) {
                ggml_threading_cond_wait(state);
            }
            ggml_spin_unlock(&shared->spin);
        }

        if (shared->n_tasks > 0 && state->has_work) {
            enum ggml_compute_error err =
                shared->task_runner(&state->params, state->node);

            GGML_ASSERT(err == GGML_COMPUTE_OK || err == GGML_COMPUTE_FALLBACK);

            ggml_spin_lock(&shared->spin);

            state->has_work = false;
            shared->n_tasks--;

            bool wait = shared->wait_on_done && !state->has_work;
            if (wait) {
                ggml_threading_cond_wait(state);
            }

            ggml_spin_unlock(&shared->spin);

            // no need to pause.
            if (wait) {
                continue;
            }
        }

        ggml_spin_pause();
    }

    PRINT_DEBUG("[%d-th] exited\n", state->params.ith);
    return 0;
}

enum ggml_compute_error
ggml_threading_compute_tensor(struct ggml_threading_context *ctx,
                              struct ggml_tensor *node, void *wdata,
                              size_t wsize) {
    GGML_ASSERT(ctx);
    GGML_ASSERT(node);

    GGML_ASSERT(ctx->shared.task_runner);
    struct ggml_compute_state_shared *state_shared = &ctx->shared;

    // This is the params for main thread.
    struct ggml_compute_params params;
    enum ggml_compute_error err;

    for (int type = GGML_TASK_INIT; type <= GGML_TASK_FINALIZE; type++) {
        if (node->task_profile.stages[type].backend == GGML_TASK_BACKEND_NONE) {
            continue;
        }

        PRINT_DEBUG("[main] stage: %d\n", type);

        int64_t t_stage = 0;
        if (ctx->stages_time) {
            t_stage = ggml_time_us();
        }

        // n_tasks is the total number of parallel computing tasks
        // (including main thread).
        int n_tasks =
            node->task_profile.stages[type].parallel ? ctx->n_threads : 1;

        ggml_spin_lock(&state_shared->spin);

        if (ctx->n_threads > 1) {
            ggml_threading_setup_workers(ctx, &node->task_profile, type);
        }

        if (n_tasks > 1) {
            // setup compute task parameters.
            for (int j = 0; j < n_tasks - 1; j++) {
                ctx->workers[j].params = (struct ggml_compute_params){
                    .type = type,
                    .ith = j + 1,
                    .nth = n_tasks,
                    .wsize = wsize,
                    .wdata = wdata,
                };
                ctx->workers[j].node = node;
                ctx->workers[j].has_work = true;
            }
            state_shared->n_tasks = n_tasks - 1;
            PRINT_DEBUG("[main] assigned %d tasks\n", state_shared->n_tasks);
        }

        ggml_spin_unlock(&state_shared->spin);

        // main thread always run the 0-th task.
        // TODO: assert(params->nth == 1) instead of
        // assert(params->ith == 0)
        {
            params.type = type;
            params.ith = 0;
            params.nth = n_tasks;
            params.wsize = wsize;
            params.wdata = wdata;

            err = state_shared->task_runner(&params, node);
        }

        // wait for tasks done.
        if (n_tasks > 1) {
            while (state_shared->n_tasks != 0) {
                ggml_spin_pause();
            }
        }

        PRINT_DEBUG("[main] all tasks finished\n\n");

        if (ctx->stages_time) {
            ctx->stages_time[type] = ggml_time_us() - t_stage;
        }

        if (err != GGML_COMPUTE_OK) {
            return err;
        }
    }

    return GGML_COMPUTE_OK;
}

struct ggml_threading_context *
ggml_threading_start(int n_threads, ggml_threading_thread_runner *thread_runner,
                     ggml_threading_task_runner *task_stage_runner,
                     enum ggml_threading_features features,
                     int64_t stages_time[3]) {
    GGML_ASSERT(n_threads > 0);
    GGML_ASSERT(thread_runner);
    GGML_ASSERT(task_stage_runner);

    size_t ctx_sz = sizeof(struct ggml_threading_context);
    struct ggml_threading_context *ctx = malloc(ctx_sz);
    GGML_ASSERT(ctx);
    memset(ctx, 0, ctx_sz);

    ctx->shared = (struct ggml_compute_state_shared){
        .spin = {0},
        .n_ready = 0,
        .n_tasks = 0,
        .n_waiting = 0,
        .wait_now = false,
        .wait_on_done = false,
        .stop = false,
        .task_runner = task_stage_runner,
        .ctx = ctx,
    };

    PRINT_DEBUG("[main] thread start, features: %d\n", features);

    ctx->n_threads = n_threads;
    ctx->features = features;
    ctx->stages_time = stages_time;

    int n_workers = n_threads - 1;
    if (n_workers > 0) {
        GGML_ASSERT(pthread_mutex_init(&ctx->shared.mutex, NULL) == 0);
        GGML_ASSERT(pthread_cond_init(&ctx->shared.cond, NULL) == 0);

        size_t workers_sz = sizeof(struct ggml_compute_state) * n_workers;
        struct ggml_compute_state *workers = malloc(workers_sz);
        GGML_ASSERT(workers);
        memset(workers, 0, workers_sz);

        for (int j = 0; j < n_workers; j++) {
            workers[j].shared = &ctx->shared;
            GGML_ASSERT(pthread_create(&workers[j].thrd, NULL, thread_runner,
                                       &workers[j]) == 0);
        }

        ctx->workers = workers;

        while (ctx->shared.n_ready != n_workers) {
            ggml_spin_pause();
        }
    }

    return ctx;
}

static void
ggml_threading_print_perf_stats(struct ggml_threading_context *ctx) {
    bool print_stats = ctx->features & GGML_THREADING_FEATURE_PERF;
#ifdef GGML_THREADING_DEBUG
    print_stats = true;
#endif

    if (!print_stats) {
        return;
    }

    const char *prefix_arr[2] = {"[threading wait  ]", "[threading wakeup]"};
    struct ggml_perf_stats *st_arr[2] = {&ctx->wait_perf, &ctx->wakeup_perf};
    for (int i = 0; i < 2; i++) {
        struct ggml_perf_stats *st = st_arr[i];
        if (st->runs == 0) {
            continue;
        }
        fprintf(stdout,
                "%s runs: %4d, avg cycles: %8.3f ms, avg time: "
                "%8.3f ms\n",
                prefix_arr[i], st->runs,
                1.0 * st->cycles / (st->runs * ggml_cycles_per_ms()),
                1.0 * st->time_us / (st->runs * 1000));
    }
}

void ggml_threading_stop(struct ggml_threading_context *ctx) {
    GGML_ASSERT(ctx);

    if (ctx->workers) {
        PRINT_DEBUG("[main] stopping thread pool ...\n");
        ctx->shared.stop = true;

        ggml_spin_lock(&ctx->shared.spin);
        ggml_threading_wakeup_workers(&ctx->shared);
        ggml_spin_unlock(&ctx->shared.spin);

        for (int j = 0; j < ctx->n_threads - 1; j++) {
            GGML_ASSERT(pthread_join(ctx->workers[j].thrd, NULL) == 0);
        }
        free(ctx->workers);
        PRINT_DEBUG("[main] thread pool stopped\n");
    }

    ggml_threading_print_perf_stats(ctx);

    free(ctx);
}
