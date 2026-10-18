#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct llama_model;

// cache-aware MoE routing ("cache prior", arXiv:2412.00099)
// fills mask[e] with 1.0f for every expert of the layer that is currently in the cache (slots[e] >= 0)
// the caller applies the mask as a bias to the router scores, so that it only changes the ranking of the experts and
// never the weights they are combined with
void llama_moe_cache_prior_mask(const int32_t * slots, int32_t n_expert, float * mask);

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // prior is the lambda of the cache-aware routing, 0 keeps the original expert selection, see llama_moe_cache_prior_mask
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float prior = 0.0f);
    ~llama_moe_cache();

    // lambda of the cache-aware routing, 0 = disabled
    float prior() const;

    // true if the experts of the layer il are kept in the cache, so that a cache prior can be applied to its routing
    bool is_cached(int32_t il) const;

    // the slot of each expert of the layer il in the cache, -1 if it is not cached, nullptr if the layer is not cached
    const int32_t * get_cached_slots(int32_t il) const;

    // the number of experts of the layer il, 0 if the layer is not cached
    int32_t n_expert(int32_t il) const;

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
