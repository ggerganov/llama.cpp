#include "common.h"
#include "../src/llama-ext.h"

#include <cstdio>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

static bool check_plan() {
    const std::vector<common_cpu_core> cores = {{{0,1},1024,5700}, {{2,3},1024,5700}, {{16},512,4400}, {{96,97},1024,5600}};
    const std::vector<common_cpu_core> sparse = {{{96,97},1024,5700}, {{128},1024,5700}};
    const std::vector<common_cpu_core> single = {{{96,97},1024,5700}};
    const std::vector<common_cpu_core> invalid = {{{-1},0,0}};
    auto p = common_cpu_plan_moe(cores, 3);
    bool ok = p.control == 0 && p.experts == std::vector<int32_t>({2,96});
    p = common_cpu_plan_moe(cores, 3, {2,3}, {0});
    ok = ok && p.control == 0 && p.experts == std::vector<int32_t>({2,3});
    p = common_cpu_plan_moe(sparse, 2);
    ok = ok && p.control == 96 && p.experts == std::vector<int32_t>({128});
    ok = ok && common_cpu_plan_moe(single, 2).control == -1;
    ok = ok && common_cpu_plan_moe(cores, 2, {97}, {96}).control == -1;
    ok = ok && common_cpu_plan_moe(cores, 2, {200}).control == -1;
    ok = ok && common_cpu_plan_moe(cores, 1).control == -1;
    ok = ok && common_cpu_plan_moe(invalid, 2).control == -1;
    ok = ok && common_cpu_plan_moe(cores, 2, {2,96,96}).control == -1;
    std::printf("CPU topology and explicit roles: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool check_guards() {
#if defined(__linux__)
    cpu_set_t original;
    CPU_ZERO(&original);
    if (sched_getaffinity(0, sizeof(original), &original) != 0) return false;
    std::vector<int> cpus;
    for (int i = 0; i < CPU_SETSIZE && i < GGML_MAX_N_THREADS; ++i) if (CPU_ISSET(i, &original)) cpus.push_back(i);
    if (cpus.size() < 2) {
        std::puts("SKIP affinity guards: two allowed CPUs are needed");
        return true;
    }
    bool ok = true;
    {
        llama_cpu_affinity_guard outer(true);
        ok = outer.bind(cpus[0]);
        {
            llama_cpu_affinity_guard inner(true);
            ok = !inner.allows(cpus[1]) && !inner.bind(cpus[1]) && inner.bind(cpus[0]) && ok;
        }
        cpu_set_t actual;
        CPU_ZERO(&actual);
        ok = sched_getaffinity(0, sizeof(actual), &actual) == 0 && CPU_COUNT(&actual) == 1 && CPU_ISSET(cpus[0], &actual) && ok;
        ok = outer.bind(cpus[1]) && ok;
    }
    cpu_set_t actual;
    CPU_ZERO(&actual);
    ok = sched_getaffinity(0, sizeof(actual), &actual) == 0 && CPU_EQUAL(&actual, &original) && ok;
    bool restricted_ok = false;
    std::thread restricted([&] {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpus[1], &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask) != 0) return;
        llama_cpu_affinity_guard guard(true);
        restricted_ok = !guard.allows(cpus[0]) && !guard.bind(cpus[0]) && guard.bind(cpus[1]);
    });
    restricted.join();
    {
        llama_cpu_affinity_guard restore(true);
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpus[0], &one);
        ok = sched_setaffinity(0, sizeof(one), &one) == 0 && ok;
    }
    ok = sched_getaffinity(0, sizeof(actual), &actual) == 0 && CPU_EQUAL(&actual, &original) && restricted_ok && ok;
    std::printf("nested, restricted-caller and backend-change restoration: %s\n", ok ? "PASS" : "FAIL");
    return ok;
#else
    std::puts("SKIP affinity guards: Linux test");
    return true;
#endif
}

int main() {
    const bool topology = check_plan();
    const bool guards = check_guards();
    return topology && guards ? 0 : 1;
}
