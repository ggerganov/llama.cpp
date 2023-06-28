// https://github.com/ggerganov/ggml/issues/291
// https://github.com/ggerganov/llama.cpp/pull/1507

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#define GGML_ASSERT(x)                                                         \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__,    \
                    #x);                                                       \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define GGML_DEBUG 1
#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#define UNUSED(x) (void)(x)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#if defined(_WIN32)

#include <windows.h>

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;

typedef HANDLE pthread_t;
typedef int thread_ret_t;

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

static int pthread_create(pthread_t *out, void *unused,
                          thread_ret_t (*func)(void *), void *arg) {
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

static int sched_yield(void) {
    // https://learn.microsoft.com/en-us/windows/win32/api/winnt/nf-winnt-yieldprocessor
    YieldProcessor();
    return 0;
}

#else // ! _WIN32

typedef void *thread_ret_t;

#include <pthread.h>
#include <stdatomic.h>

#endif

typedef pthread_t ggml_thread_t;

//-----------------------------------------------------------------------------
/// Most of the above codes are taken from
/// https://github.com/ggerganov/llama.cpp/tree/master/ggml.c
/// Copyright original authors.
//-----------------------------------------------------------------------------

#define MAX_THREADS 16

struct task_allocator {
    int nth;

    int n_multiplier; // >= 1

    atomic_int lock; // 0 unlocked, 1 locked

    // total assigned.
    atomic_int global_counter;

    atomic_int thread_queue_heads[MAX_THREADS];
    atomic_int thread_queue_tails[MAX_THREADS];
};

static void task_allocator_reset(struct task_allocator *a) {
    for (int i = 0; i < a->nth; ++i) {
        atomic_store(&a->thread_queue_heads[i], 0);
        atomic_store(&a->thread_queue_tails[i], a->n_multiplier);
    }

    atomic_store(&a->lock, 0);
    atomic_store(&a->global_counter, 0);
}

// NOTE: when nth == 1, n_multiplier is actually useless.
static void task_allocator_init(struct task_allocator *a, int nth,
                                int n_multiplier) {
    GGML_ASSERT(nth > 0);
    GGML_ASSERT(nth <= MAX_THREADS);
    GGML_ASSERT(n_multiplier > 0);

    a->nth = nth;
    a->n_multiplier = nth == 1 ? 1 : n_multiplier;
    task_allocator_reset(a);
}

// ith: worker id (start from 0).
// chunk_idx and n_chunks will be updated.
// chunk_idx is set as -1 when nothing to do.
static void allocate_chunk(struct task_allocator *a, int ith, int *chunk_idx,
                           int *n_chunks) {
    GGML_ASSERT(ith >= 0 && ith < a->nth);

    int M = a->n_multiplier;
    int nth = a->nth;
    int total_chunks = M * nth;

    *chunk_idx = -1;
    *n_chunks = total_chunks;

    while (atomic_fetch_add(&a->lock, 1) != 0) { // lock
        atomic_fetch_sub(&a->lock, 1);
    }

    // all assigned?
    if (atomic_load(&a->global_counter) == total_chunks) {
        GGML_PRINT_DEBUG("[#_%d] %s(): nothing to do.\n", ith, __func__);
        atomic_fetch_sub(&a->lock, 1); // unlock
        return;
    }

    // try take its own, pop front.
    {
        int head = atomic_load(&a->thread_queue_heads[ith]);
        int tail = atomic_load(&a->thread_queue_tails[ith]);

        GGML_PRINT_DEBUG_5("[#_%d] %s(): head: %d, tail: %d.\n", ith, __func__,
                           head, tail);

        if (head < tail) {
            int idx = ith * M + head;

            atomic_fetch_add(&a->thread_queue_heads[ith], 1);
            atomic_fetch_add(&a->global_counter, 1);

            GGML_PRINT_DEBUG("[#_%d] %s(): take the %3d-th chunk of its own.\n",
                             ith, __func__, head + 1);

            *chunk_idx = idx;
            *n_chunks = total_chunks;
            atomic_fetch_sub(&a->lock, 1); // unlock
            return;
        }
    }

    // steal from others.
    // TODO: optimize: steal from the slowest one.
    for (int i = 0; i < nth; ++i) {
        if (i == ith) {
            continue;
        }

        int tail = atomic_load(&a->thread_queue_tails[i]);
        if (tail == atomic_load(&a->thread_queue_heads[i])) {
            continue;
        }

        // pop back
        int idx = i * M + tail;
        atomic_fetch_sub(&a->thread_queue_tails[i], 1);
        atomic_fetch_add(&a->global_counter, 1);

        GGML_PRINT_DEBUG("[#_%d] %s(): steal the %d-th chunk from #_%d\n", ith,
                         __func__, tail, i);

        *chunk_idx = idx;
        *n_chunks = total_chunks;
        atomic_fetch_sub(&a->lock, 1); // unlock
        return;
    }

    fprintf(stderr, "%s:%d should be unreachable!\n", __FILE__, __LINE__);
    abort();
}

struct state_shared {
    int n_threads;
    int n_multiplier;

    int n_nodes;
    struct ggml_tensor *nodes;

    // thread done counter for single node
    atomic_int done_counter;

    struct task_allocator task_allocator;
};

struct state {
    ggml_thread_t thrd;
    int ith;
    struct state_shared *shared;
};

// simulate tensor that can be compute in parallel
struct ggml_tensor {
    // simulate actual compute workload, e.g. src0 rows
    int n_compute_units;
};

struct params {
    int ith;
    int nth;

    // simulate performance jitters related to: OS workload, thread affinity,
    // economic cores, ...
    int jitter_percent;

    struct task_allocator *task_allocator;
};

void compute_tensor(struct params params, struct ggml_tensor *node) {
    GGML_PRINT_DEBUG_5("[#_%d] %s(): enter.\n", params.ith, __func__);

    const int ith = params.ith;
    int chunk_idx;
    int n_chunks;

    while (true) {
        allocate_chunk(params.task_allocator, ith, &chunk_idx, &n_chunks);
        if (chunk_idx < 0) {
            break;
        }

        const int nr = node->n_compute_units;
        const int dr = (nr + n_chunks - 1) / n_chunks;
        const int ir0 = dr * chunk_idx;
        const int ir1 = MIN(ir0 + dr, nr);
        const int n_loops = 10000 * (100 + params.jitter_percent);

        volatile int64_t x = 0;

        for (int i = ir0; i <= ir1; ++i) {
            for (int j = 0; j < n_loops; ++j) {
                ++x;
            }
        }
        UNUSED(x);
    }

    GGML_PRINT_DEBUG_5("[#_%d] %s(): exit.\n", ith, __func__);
}

static thread_ret_t demo_compute_thread(void *data) {
    struct state *state = (struct state *)data;
    GGML_ASSERT(state);

    struct state_shared *shared = state->shared;
    GGML_ASSERT(shared);

    struct task_allocator *allocator = &shared->task_allocator;
    GGML_ASSERT(allocator);

    int ith = state->ith;
    int n_threads = shared->n_threads;

    atomic_int *done_counter = &shared->done_counter;

    for (int i = 0; i < shared->n_nodes; ++i) {
        // Just slow down the last thread.
        struct params params = {
            .ith = state->ith,
            .nth = n_threads, // suppose parallel
            .task_allocator = allocator,
            .jitter_percent = ith + 1 < n_threads ? 0 : 50,
        };

        struct ggml_tensor *node = &shared->nodes[i];

        compute_tensor(params, node);
        atomic_fetch_add(done_counter, 1);

        GGML_PRINT_DEBUG_5("[#_%d] %s(): finished computing the node.\n", ith,
                           __func__);

        if (ith == 0) {
            while (atomic_load(done_counter) != n_threads) {
                sched_yield();
            }

            GGML_PRINT_DEBUG_5(
                "[#_%d] %s(): saw all threads finished computing the node.\n",
                ith, __func__);

            task_allocator_reset(allocator);
            atomic_store(done_counter, 0);
        } else {
            while (atomic_load(done_counter) != 0) {
                sched_yield();
            }
        }
    }

    GGML_PRINT_DEBUG_5("[#_%d] %s(): exited\n", ith, __func__);

    return 0;
}

static void test_task_allocator_init(void) {
    struct task_allocator a;

    task_allocator_init(&a, 1, 2);
    GGML_ASSERT(a.nth == 1);
    GGML_ASSERT(a.n_multiplier == 1); // when nth == 1, force n_multiplier as 1

    task_allocator_init(&a, 2, 2);
    GGML_ASSERT(a.nth == 2);
    GGML_ASSERT(a.n_multiplier == 2); // ok
}

static void task_allocator_unit_test_no_steal(void) {
    int chunk_idx; // out
    int n_chunks;  // out

    int n_threads = 2;
    int n_multiplier = 2;
    const int expected_n_slots = n_threads * n_multiplier;

    struct task_allocator a;
    task_allocator_init(&a, n_threads, n_multiplier);

    struct test_data_t {
        int ith;       // call by
        int chunk_idx; // expected
        int n_chunks;  // expected
    };

    struct test_data_t test_data[] = {
        //////////////////// clang format /////////////////////////
        {
            .ith = 0,
            .chunk_idx = 0,
        },
        {
            .ith = 1,
            .chunk_idx = 2,
        },
        {
            .ith = 0,
            .chunk_idx = 1,
        },
        {
            .ith = 1,
            .chunk_idx = 3,
        },
        {
            .ith = 0,
            .chunk_idx = -1,
        },
        {
            .ith = 1,
            .chunk_idx = -1,
        }};

    int t_len = sizeof(test_data) / sizeof(struct test_data_t);

    for (int i = 0; i < t_len; i++) {
        allocate_chunk(&a, test_data[i].ith, &chunk_idx, &n_chunks);
        if (chunk_idx != test_data[i].chunk_idx) {
            fprintf(stderr,
                    "%s(): chunk_idx mismatch. i: %d, actual: %d, expected: %d\n",
                    __func__, i, chunk_idx, test_data[i].chunk_idx);
            abort();
        }
        if (n_chunks != expected_n_slots) {
            fprintf(stderr,
                    "%s(): n_chunks mismatch. i: %d, actual: %d, expected: %d\n",
                    __func__, i, n_chunks, expected_n_slots);
            abort();
        }
    }
}

static void task_allocator_unit_test_steal(void) {
    int chunk_idx; // out
    int n_chunks;  // out

    int n_threads = 2;
    int n_multiplier = 2;
    const int expected_n_slots = n_threads * n_multiplier;

    struct task_allocator a;
    task_allocator_init(&a, n_threads, n_multiplier);

    struct test_data_t {
        int ith;       // call by
        int chunk_idx; // expected
    };

    struct test_data_t test_data[] = {
        //////////////////// clang format /////////////////////////
        {
            .ith = 0,
            .chunk_idx = 0,
        },
        {
            .ith = 0,
            .chunk_idx = 1,
        },
        {
            .ith = 1,
            .chunk_idx = 2,
        },
        {
            .ith = 0,
            .chunk_idx = 4, // steal from tail
        },
        {
            .ith = 0,
            .chunk_idx = -1,
        },
        {
            .ith = 1,
            .chunk_idx = -1,
        }};

    int t_len = sizeof(test_data) / sizeof(struct test_data_t);

    for (int i = 0; i < t_len; i++) {
        allocate_chunk(&a, test_data[i].ith, &chunk_idx, &n_chunks);
        if (chunk_idx != test_data[i].chunk_idx) {
            fprintf(stderr,
                    "%s(): chunk_idx mismatch. i: %d, actual: %d, expected: %d\n",
                    __func__, i, chunk_idx, test_data[i].chunk_idx);
            abort();
        }
        if (n_chunks != expected_n_slots) {
            fprintf(stderr,
                    "%s(): n_chunks mismatch. i: %d, actual: %d, expected: %d\n",
                    __func__, i, n_chunks, expected_n_slots);
            abort();
        }
    }
}

// Integration test.
static void test_task_allocator(int n_threads, int n_nodes, int n_compute_units,
                                int n_multiplier) {
    fprintf(stderr,
            "\n%s(): n_threads: %d, n_nodes: %d, n_compute_units: %d, "
            "n_multiplier: %d ===>\n\n",
            __func__, n_threads, n_nodes, n_compute_units, n_multiplier);

    struct ggml_tensor *nodes = alloca(n_nodes * sizeof(struct ggml_tensor));

    for (int i = 0; i < n_nodes; ++i) {
        nodes[i].n_compute_units = n_compute_units;
    }

    struct state_shared shared = {
        .n_threads = n_threads,
        .n_nodes = n_nodes,
        .nodes = nodes,
        .done_counter = 0,
    };

    task_allocator_init(&shared.task_allocator, n_threads, n_multiplier);

    struct state *workers = alloca(n_threads * sizeof(struct state));

    for (int i = 0; i < n_threads; ++i) {
        workers[i].ith = i;
        workers[i].shared = &shared;
        if (i > 0) {
            pthread_create(&workers[i].thrd, NULL, demo_compute_thread,
                           &workers[i]);
        }
    }

    demo_compute_thread(&workers[0]);

    for (int i = 1; i < n_threads; ++i) {
        pthread_join(workers[i].thrd, NULL);
    }
}

//
// Conclusions:
//
// - Given workers A and B, and the accumulated time T_a and T_b:
//   B can steal a chunk from A only if T_a > T_b + T_b_per_chunk.
// - Saw this situation: A steal B, B steal C.
// - n_chunks plays a key role, similar to choosing the best n_threads, it's
//   difficult to choose the ideal n_chunks value. Performance drops when
//   per-chunk compute time exceeds the scheduling overhead.
// - Work stealing chunked task allocator can save the response time
//   significantly when the majority threads runs fast but a few suddenly or
//   constantly slow.
//
int main(void) {
    test_task_allocator_init();
    task_allocator_unit_test_no_steal();
    task_allocator_unit_test_steal();

    // Integration tests
    const int n_compute_units = 64;

    if (false) {
        int n_threads = 1;
        int n_nodes = 1;
        int n_multiplier = 2; // equivalent to 1

        test_task_allocator(n_threads, n_nodes, n_compute_units, n_multiplier);
    }

    if (true) {
        int n_threads = 2;
        int n_nodes = 2;
        int n_multiplier = 1;

        test_task_allocator(n_threads, n_nodes, n_compute_units, n_multiplier);
    }

    if (true) {
        int n_threads = 2;
        int n_nodes = 2;
        int n_multiplier = 8;

        test_task_allocator(n_threads, n_nodes, n_compute_units, n_multiplier);
    }
}
