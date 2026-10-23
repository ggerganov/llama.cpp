#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct llama_moe_cpu_params;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const llama_moe_cpu_params * cpu = nullptr);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    ggml_tensor * get_cpu_result(int32_t il, ggml_tensor * slots, bool allowed) const;

    bool prepare_single_graph(bool allowed) noexcept;
    bool single_graph_active() const;
    ggml_tensor * build_single_slots(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * input, int32_t il, bool allowed) const;
    ggml_tensor * build_single_merge(ggml_context * ctx, ggml_tensor * gpu, ggml_tensor * slots, int32_t il) const;
    void check_single_graph(bool synchronized);
    void cancel_single_graph();
    void observe_completed(uint32_t decode_calls, bool only_decode) noexcept;
    bool maintenance_pending() const;
    ggml_status maintain(bool (*abort_callback)(void *) = nullptr, void * abort_data = nullptr) noexcept;
    bool failed() const;
    void poison(const char * reason) noexcept;

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    bool copy_cpu_result_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
