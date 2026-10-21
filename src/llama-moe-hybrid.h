#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <memory>

struct llama_model;
struct ggml_context;
struct ggml_tensor;

// Hybrid MoE decoding for experts kept in host memory (the scheme of Strata's hybrid mode):
//  - every tracked layer keeps the same number of its experts in device memory ("slots")
//  - the slot table of a layer lives on the device, so the device computes the resident experts without waiting for the host
//  - the other experts of a token are computed on the CPU at the same time, from the host copy, and the two results are summed
//  - experts are moved only between graph computes: after a prompt (many moves), then every swap_every generated tokens (few moves),
//    by routing counts that are halved at each move
struct llama_moe_hybrid_params {
    size_t  size        = 0;     // device memory for the slots in bytes, 0 = off
    int32_t max_tokens  = 64;    // largest batch that runs in the hybrid form (1..64), larger batches use the normal path
    int32_t swap_every  = 16;    // generated tokens between two rebalances (default: 16)
    int32_t swap_max    = 32;    // experts moved per rebalance while generating (default: 32)
    int32_t swap_prompt = 2048;  // experts moved by the rebalance after a prompt
    int32_t prompt_min  = 64;    // a batch of at least this many tokens is a prompt (it triggers the rebalance after a prompt)
    float   skip_weight = 0.05f; // skip CPU execution of experts whose router weight < skip_weight (default: 0.05)
    int32_t tail_skip   = 5;     // skip CPU execution of experts whose rank >= tail_skip (default: 5)
};

class llama_moe_hybrid {
public:
    // backend/buft: the device that holds the slots; layers whose attention is on another device are not tracked
    llama_moe_hybrid(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, const llama_moe_hybrid_params & params);
    ~llama_moe_hybrid();

    // layer il has host experts and is tracked (its routing is counted)
    bool tracked(int32_t il) const;

    // layer il runs in the hybrid form for a batch of n_tokens
    bool usable(int32_t il, int64_t n_tokens) const;

    ggml_backend_t backend() const;

    // device I32 [1, n_expert]: the slot of each expert of layer il, -1 if it is not resident
    ggml_tensor * slot_table(int32_t il) const;

    // the device slots of host expert tensor w, nullptr if w is not tracked
    ggml_tensor * bank(const ggml_tensor * w) const;

    // CPU op on the selected experts of layer il (I32 [n_expert_used, n_tokens], contiguous): counts every selected expert and returns
    // the ids for the CPU branch (the expert where it is not resident, -1 where it is or when tail-skipped)
    ggml_tensor * build_route(ggml_context * ctx, ggml_tensor * selected, ggml_tensor * weights, int32_t il) const;

    // prompt batches: count the experts host expert tensor w's layer selected (n ids), read by the scheduler's copy callback
    void count(const ggml_tensor * w, const int32_t * ids, size_t n) const;

    // call before every graph compute of the context: runs the pending rebalance
    void before_compute(int64_t n_tokens);

    // prompt batches (the scheduler copies host expert tensor w to the device as dst): copy the experts from e on (up to last) that
    // are resident, from device memory, so only the others cross PCIe. Returns the number copied, 0 if expert e is not resident
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_hybrid_ptr = std::unique_ptr<llama_moe_hybrid>;
