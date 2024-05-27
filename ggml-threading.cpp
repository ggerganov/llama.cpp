#include "ggml-threading.h"
#include <stdio.h>

#define GGML_UNUSED(x) (void)(x)

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

struct ggml_numa_nodes g_state_numa;

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_zero() {
    g_state_numa.n_nodes = 0;
    g_state_numa.total_cpus = 0;
}

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state_numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state_numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n", g_state_numa.numa_strategy);

    g_state_numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state_numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state_numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state_numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state_numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state_numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state_numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state_numa.n_nodes, g_state_numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 28) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state_numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state_numa.current_node);
#endif

    if (g_state_numa.n_nodes < 1 || g_state_numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state_numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state_numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state_numa.n_nodes; ++n) {
        struct ggml_numa_node* node = &g_state_numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state_numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE* fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_PRINT("WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    GGML_UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state_numa.n_nodes > 1;
}

////////////////////////////////////////////////////////////////////////////////

#if defined(_WIN32)

void atomic_store(atomic_int* ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
LONG atomic_load(atomic_int* ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
LONG atomic_fetch_add(atomic_int* ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
LONG atomic_fetch_sub(atomic_int* ptr, LONG dec) {
    return atomic_fetch_add(ptr, -(dec));
}

int pthread_create(pthread_t* out, void* unused, thread_ret_t(*func)(void*), void* arg) {
    (void)unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

int pthread_join(pthread_t thread, void* unused) {
    (void)unused;
    int ret = (int)WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

int sched_yield(void) {
    Sleep(0);
    return 0;
}

#endif

static atomic_int g_state_barrier = 0;

// barrier via spin lock for g_state
void ggml_critical_section_start(void) {
    int processing = atomic_fetch_add(&g_state_barrier, 1);

    while (processing > 0) {
        // wait for other threads to finish
        atomic_fetch_sub(&g_state_barrier, 1);
        sched_yield(); // TODO: reconsider this
        processing = atomic_fetch_add(&g_state_barrier, 1);
    }
}

// TODO: make this somehow automatically executed
//       some sort of "sentry" mechanism
void ggml_critical_section_end(void) {
    atomic_fetch_sub(&g_state_barrier, 1);
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state_numa.total_cpus);

    switch (g_state_numa.numa_strategy) {
    case GGML_NUMA_STRATEGY_DISTRIBUTE:
        // run thread on node_num thread_n / (threads per node)
        node_num = thread_n % g_state_numa.n_nodes;
        break;
    case GGML_NUMA_STRATEGY_ISOLATE:
        // run thread on current_node
        node_num = g_state_numa.current_node;
        break;
    case GGML_NUMA_STRATEGY_NUMACTL:
        // use the cpuset that numactl gave us
        rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state_numa.cpuset);
        if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
        }
        return;
    default:
        return;
    }

    struct ggml_numa_node* node = &g_state_numa.nodes[node_num];

    cpu_set_t* cpus = CPU_ALLOC(g_state_numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state_numa.total_cpus);

    cpu_set_t* cpus = CPU_ALLOC(g_state_numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state_numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
void set_numa_thread_affinity(int thread_n) { GGML_UNUSED(thread_n); }
void clear_numa_thread_affinity(void) {}
#endif
