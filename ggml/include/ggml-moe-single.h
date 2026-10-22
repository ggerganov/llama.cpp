#pragma once

#include "ggml-backend.h"

struct alignas(64) ggml_moe_single_flag {
    uint32_t value;
};

struct ggml_moe_single_control {
    ggml_moe_single_flag request;
    ggml_moe_single_flag completed;
    ggml_moe_single_flag error;
    ggml_moe_single_flag cancel;
    ggml_moe_single_flag abort;
    uint64_t calls;
    uint64_t misses;
};

enum ggml_moe_single_error {
    GGML_MOE_SINGLE_OK = 0,
    GGML_MOE_SINGLE_BAD_ID = 1,
    GGML_MOE_SINGLE_GENERATION = 2,
    GGML_MOE_SINGLE_TIMEOUT = 3,
    GGML_MOE_SINGLE_WORKER = 4,
};

struct ggml_moe_single_layer {
    ggml_moe_single_control * host;
    ggml_moe_single_control * device;
    int32_t * ids;
    int32_t * trace_ids;
    float * input;
    float * output;
    int32_t * device_ids;
    int32_t * device_trace_ids;
    float * device_input;
    float * device_output;
    int32_t * device_map;
    uint32_t * host_abort;
    uint32_t * device_abort;
    int n_expert;
    int n_used;
    int n_embd;
    int zero_slot;
    int device_id;
    uint64_t timeout_ticks;
    uint64_t creation_cookie;
};

struct ggml_moe_single_api {
    ggml_moe_single_layer * (*create)(ggml_backend_t backend, int n_expert, int n_used, int n_embd, int zero_slot, const int32_t * map);
    // Cancel and join the CPU service, then drain the backend before destroying a used layer.
    void (*destroy)(ggml_moe_single_layer * layer);
    ggml_tensor * (*publish)(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * input, ggml_moe_single_layer * layer);
    ggml_tensor * (*merge)(ggml_context * ctx, ggml_tensor * gpu, ggml_tensor * slots, ggml_moe_single_layer * layer);
};

using ggml_moe_single_get_api_t = const ggml_moe_single_api * (*)();

struct ggml_moe_single_bank_write {
    ggml_tensor * bank;
    const void * data;
    size_t offset;
    size_t size;
};

struct ggml_moe_single_map_write {
    ggml_moe_single_layer * layer;
    const int32_t * map;
    int32_t n_slots;
};

struct ggml_moe_single_update_api_v1 {
    bool (*apply)(ggml_backend_t backend, const ggml_moe_single_bank_write * banks, size_t n_banks,
            const ggml_moe_single_map_write * maps, size_t n_maps, bool * mutation_started);
};

using ggml_moe_single_get_update_api_v1_t = const ggml_moe_single_update_api_v1 * (*)();
