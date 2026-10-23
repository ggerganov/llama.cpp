#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cstdio>
#include <vector>

#if defined(__linux__)
#include <sched.h>

static ggml_backend_t backend;
static decltype(ggml_threadpool_new) * make_pool;
static decltype(ggml_threadpool_free) * free_pool;
static decltype(ggml_backend_cpu_set_threadpool) * set_pool;

static void copy_input(ggml_tensor * dst, int ith, int nth, void *) {
    GGML_ASSERT(ith == 0 && nth == 1);
    static_cast<float *>(dst->data)[0] = static_cast<const float *>(dst->src[0]->data)[0];
}

static bool check_affinity(int target, const cpu_set_t & original) {
    auto params = ggml_threadpool_params_default(2);
    params.paused = true;
    params.strict_cpu = true;
    if (target >= 0) params.cpumask[target] = true;
    auto * pool = make_pool(&params);
    set_pool(backend, pool);
    auto * ctx = ggml_init({2*1024*1024, nullptr, false});
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    static_cast<float *>(input->data)[0] = 7.0f;
    ggml_tensor * args[] = {input};
    auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 1, copy_input, 1, nullptr);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    bool ok = true;
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto status = ggml_backend_graph_compute(backend, graph);
        cpu_set_t actual;
        CPU_ZERO(&actual);
        ok = sched_getaffinity(0, sizeof(actual), &actual) == 0 && status == GGML_STATUS_SUCCESS && ok;
        ok = ok && static_cast<const float *>(result->data)[0] == 7.0f;
        if (target >= 0) ok = ok && CPU_COUNT(&actual) == 1 && CPU_ISSET(target, &actual);
        else ok = ok && CPU_EQUAL(&actual, &original);
        ok = sched_setaffinity(0, sizeof(original), &original) == 0 && ok;
    }
    set_pool(backend, nullptr);
    free_pool(pool);
    ggml_free(ctx);
    ok = sched_setaffinity(0, sizeof(original), &original) == 0 && ok;
    std::printf("one-task graph, pool=2, target=%d: %s\n", target, ok ? "PASS" : "FAIL");
    return ok;
}
#endif

int main(int argc, char ** argv) {
#if defined(__linux__)
    auto * reg = argc > 1 ? ggml_backend_load(argv[1]) : ggml_backend_reg_by_name("CPU");
    if (!reg) return 1;
    backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    make_pool = reinterpret_cast<decltype(make_pool)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
    free_pool = reinterpret_cast<decltype(free_pool)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
    set_pool = reinterpret_cast<decltype(set_pool)>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
    auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (!backend || !make_pool || !free_pool || !set_pool || !set_threads) return 1;
    set_threads(backend, 2);
    cpu_set_t original;
    CPU_ZERO(&original);
    if (sched_getaffinity(0, sizeof(original), &original) != 0) return 1;
    std::vector<int> cpus;
    for (int i = 0; i < CPU_SETSIZE && i < GGML_MAX_N_THREADS; ++i) {
        if (CPU_ISSET(i, &original)) cpus.push_back(i);
    }
    if (cpus.size() < 2) {
        std::puts("SKIP: two allowed CPUs are needed");
        return 0;
    }
    bool ok = check_affinity(cpus[0], original);
    ok = check_affinity(cpus[1], original) && ok;
    ok = check_affinity(-1, original) && ok;
    ggml_backend_free(backend);
    return ok ? 0 : 1;
#else
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);
    std::puts("SKIP: Linux CPU affinity test");
    return 0;
#endif
}
