#pragma once

#include "common.cuh"
#include "ggml-moe-single.h"

#if defined(GGML_USE_HIP)

struct ggml_cuda_moe_single_publish {
    const float * input;
    int32_t * slots;
    const int32_t * map;
    ggml_moe_single_control * control;
    int32_t * host_ids;
    int32_t * trace_ids;
    float * host_input;
    uint32_t * abort;
    int zero_slot;
};

static __device__ uint32_t ggml_moe_single_load(const uint32_t * ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

static __device__ void ggml_moe_single_store(uint32_t * ptr, uint32_t value) {
    __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

static __device__ void ggml_moe_single_fail(ggml_moe_single_control * control, uint32_t * abort, uint32_t error) {
    ggml_moe_single_store(&control->error.value, error);
    ggml_moe_single_store(abort, 1);
}

template <bool parallel>
static __device__ __forceinline__ void ggml_moe_single_publish_device(const int32_t * ids, const float * input, int32_t * slots,
        const int32_t * map, ggml_moe_single_control * control, int32_t * host_ids, int32_t * trace_ids, float * host_input,
        int n_expert, int n_used, int n_embd, int zero_slot, uint32_t * abort,
        int thread_index, int thread_count) {
    __shared__ int misses;
    __shared__ bool failed;
    bool invalid = false;
    if constexpr (parallel && ggml_cuda_get_physical_warp_size() == 32) {
        if (thread_index < ggml_cuda_get_physical_warp_size()) {
            bool lane_miss = false;
            bool lane_invalid = false;
            if (thread_index < n_used) {
                const int id = ids[thread_index];
                trace_ids[thread_index] = id;
                const bool valid = id >= 0 && id < n_expert;
                const int slot = valid ? map[id] : -1;
                slots[thread_index] = slot < 0 ? zero_slot : slot;
                lane_miss = slot < 0;
                lane_invalid = !valid;
            }
            const uint32_t miss_mask = (uint32_t) __ballot(lane_miss);
            const uint32_t invalid_mask = (uint32_t) __ballot(lane_invalid);
            if (thread_index == 0) {
                misses = __popc(miss_mask);
                invalid = invalid_mask != 0;
            }
        }
    } else {
        if (thread_index == 0) {
            misses = 0;
            for (int i = 0; i < n_used; ++i) {
                const int id = ids[i];
                trace_ids[i] = id;
                const bool valid = id >= 0 && id < n_expert;
                const int slot = valid ? map[id] : -1;
                slots[i] = slot < 0 ? zero_slot : slot;
                misses += slot < 0;
                invalid |= !valid;
            }
        }
    }
    if (thread_index == 0) {
        control->calls++;
        control->misses += misses;
        failed = invalid || ggml_moe_single_load(&control->cancel.value) || ggml_moe_single_load(abort);
        if (invalid) ggml_moe_single_fail(control, abort, GGML_MOE_SINGLE_BAD_ID);
    }
    __syncthreads();
    if (failed || !misses) return;
    for (int i = thread_index; i < n_used; i += thread_count) host_ids[i] = ids[i];
    for (int i = thread_index; i < n_embd; i += thread_count) host_input[i] = input[i];
    __threadfence_system();
    __syncthreads();
    if (thread_index == 0) {
        const uint32_t previous = ggml_moe_single_load(&control->request.value);
        if (previous == UINT32_MAX) {
            ggml_moe_single_fail(control, abort, GGML_MOE_SINGLE_GENERATION);
        } else {
            ggml_moe_single_store(&control->request.value, previous + 1);
        }
    }
}

#endif
