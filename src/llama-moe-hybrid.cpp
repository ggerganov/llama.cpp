#include "llama-moe-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

bool llama_moe_hybrid_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

std::vector<ggml_tensor *> llama_moe_hybrid_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

} // namespace

struct llama_moe_hybrid::impl {
    struct layer {
        bool tracked = false;
        int32_t il = -1;
        std::vector<ggml_tensor *> experts;  // host expert tensors
        std::vector<ggml_tensor *> banks;    // device slots of each, [ne0, ne1, n_slots + 1]
        ggml_tensor * slot_tab = nullptr;    // device I32 [1, n_expert]
        std::vector<int32_t> slot_of;        // host mirror of slot_tab
        std::vector<int32_t> exp_in_slot;    // [n_slots]
        std::vector<double>  score;          // routing counts, halved at each rebalance
        impl * owner = nullptr;
    };

    llama_moe_hybrid_params params;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    bool no_alloc = false;

    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> bank_of;
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // host expert tensor -> layer
    ggml_context_ptr ctx_views; // views for copy_experts

    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    size_t buf_size = 0;

    // rebalance state
    bool prompt_pending = false;
    int64_t since = 0;
    size_t n_swaps = 0;
    size_t n_rebalance = 0;
    ggml_backend_event_t swap_event = nullptr;

    impl(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, const llama_moe_hybrid_params & p) :
            params(p), backend(backend), buft(buft) {
        params.max_tokens = std::clamp(params.max_tokens, 1, 64); // short prompts <= 64 tokens run via hybrid decode window
        params.swap_every = std::max(params.swap_every, 1);
        params.swap_max   = std::max(params.swap_max, 0);
        no_alloc = model.hparams.no_alloc;
        n_expert = (int32_t) model.hparams.n_expert;
        if (n_expert <= 0 || model.hparams.n_expert_used_max() <= 0) {
            throw std::runtime_error("MoE hybrid mode requires a MoE model");
        }

        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const size_t n_main = model.layers.size() - std::min<size_t>(model.layers.size(), model.hparams.n_layer_nextn);

        layers.resize(model.layers.size());
        size_t per_slot = 0; // bytes of one expert of every tracked layer
        for (size_t il = 0; il < model.layers.size(); ++il) {
            layer & l = layers[il];
            l.il = (int32_t) il;
            l.owner = this;
            if (il >= n_main) {
                continue; // the MTP block runs in the draft context
            }
            auto experts = llama_moe_hybrid_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_hybrid_is_host_weight)) {
                continue;
            }
            if (model.dev_layer((int) il) != dev) {
                continue;
            }
            bool ok = true;
            for (const ggml_tensor * w : experts) {
                ok = ok && w->ne[2] == n_expert && w->ne[3] == 1 && ggml_is_contiguous(w);
            }
            if (!ok) {
                continue;
            }
            l.tracked = true;
            l.experts = experts;
            for (const ggml_tensor * w : experts) {
                per_slot += w->nb[2];
            }
        }
        if (per_slot == 0) {
            throw std::runtime_error("MoE hybrid mode: no layer keeps its experts in plain host memory (use --cpu-moe or --n-cpu-moe; "
                                     "experts repacked for the CPU, the default for some types, need --no-repack)");
        }

        // one extra slot per bank: CUDA MMQ can read past the last expert
        n_slots = (int32_t) std::min<size_t>((size_t) n_expert, params.size / per_slot);
        n_slots = std::max(n_slots - 1, 0);
        // the CUDA kernels give each skipped entry of a token an unused slot: a token needs n_expert_used distinct slots
        if (n_slots + 1 < (int32_t) model.hparams.n_expert_used_max()) {
            throw std::runtime_error("MoE hybrid mode: the size holds " + std::to_string(n_slots) + " experts per layer, at least " +
                                     std::to_string(model.hparams.n_expert_used_max() - 1) + " are needed");
        }

        size_t n_tensors = 0;
        for (const layer & l : layers) {
            n_tensors += l.tracked ? l.experts.size() + 1 : 0;
        }
        ggml_init_params ip = {
            /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(ip));
        if (!ctx) {
            throw std::runtime_error("failed to create the MoE hybrid context");
        }

        size_t n_tracked = 0;
        for (layer & l : layers) {
            if (!l.tracked) {
                continue;
            }
            n_tracked++;
            for (ggml_tensor * w : l.experts) {
                ggml_tensor * b = ggml_new_tensor_3d(ctx.get(), w->type, w->ne[0], w->ne[1], n_slots + 1);
                GGML_ASSERT(b->nb[2] == w->nb[2]);
                ggml_format_name(b, "moe_hybrid.%s", w->name);
                l.banks.push_back(b);
                bank_of[w] = b;
                layer_of[w] = l.il;
            }
            l.slot_tab = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, n_expert);
            ggml_format_name(l.slot_tab, "moe_hybrid.slots-%d", l.il);
            // the first n_slots experts until the first rebalance
            l.slot_of.assign(n_expert, -1);
            l.exp_in_slot.resize(n_slots);
            for (int32_t s = 0; s < n_slots; ++s) {
                l.slot_of[s] = s;
                l.exp_in_slot[s] = s;
            }
            l.score.assign(n_expert, 0.0);
        }

        if (no_alloc) {
            // only measured, see llama_context::memory_breakdown
            const size_t align = ggml_backend_buft_get_alignment(buft);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                buf_size += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
            }
            buf.reset(ggml_backend_buft_alloc_buffer(buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf.get();
            }
        } else {
            buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
            if (!buf) {
                throw std::runtime_error("failed to allocate the MoE hybrid slots");
            }
            ggml_backend_buffer_clear(buf.get(), 0);
            buf_size = ggml_backend_buffer_get_size(buf.get());
        }
        // as weights: the ops that read them run on this device
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_init_params vp = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_views.reset(ggml_init(vp));

        if (!no_alloc) {
            for (layer & l : layers) {
                if (!l.tracked) {
                    continue;
                }
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    ggml_backend_tensor_set(l.banks[ip], w->data, 0, (size_t) n_slots*w->nb[2]);
                }
                ggml_backend_tensor_set(l.slot_tab, l.slot_of.data(), 0, l.slot_of.size()*sizeof(int32_t));
            }
        }

        LLAMA_LOG_INFO("%s: %d of %d experts per layer in %s for %zu layers, %.2f MiB, batches up to %d tokens (tail_skip %d, skip_weight %.3f)\n", __func__,
                n_slots, n_expert, ggml_backend_buft_name(buft), n_tracked, buf_size/1024.0/1024.0, params.max_tokens, params.tail_skip, params.skip_weight);
    }

    ~impl() {
        if (swap_event != nullptr) {
            ggml_backend_event_free(swap_event);
        }
        if (n_rebalance > 0) {
            LLAMA_LOG_INFO("llama_moe_hybrid: %zu rebalances moved %zu experts\n", n_rebalance, n_swaps);
        }
    }

    // move up to max_swaps experts, the largest score gains first, over all layers; then halve every score
    void rebalance(int32_t max_swaps) {
        if (no_alloc) {
            return;
        }
        struct swap {
            double  gain;
            int32_t il;
            int32_t x; // expert that gets a slot
            int32_t s; // slot it takes
        };
        std::vector<swap> cand;
        std::vector<int32_t> order(n_expert);
        std::vector<int32_t> victims(n_slots);
        for (layer & l : layers) {
            if (!l.tracked || max_swaps <= 0) {
                continue;
            }
            for (int32_t x = 0; x < n_expert; ++x) {
                order[x] = x;
            }
            // the n_slots best experts by score
            std::partial_sort(order.begin(), order.begin() + n_slots, order.end(), [&](int32_t a, int32_t b) {
                return l.score[a] > l.score[b];
            });
            // the resident experts, lowest score first
            for (int32_t s = 0; s < n_slots; ++s) {
                victims[s] = s;
            }
            std::sort(victims.begin(), victims.end(), [&](int32_t a, int32_t b) {
                return l.score[l.exp_in_slot[a]] < l.score[l.exp_in_slot[b]];
            });
            size_t vi = 0;
            for (int32_t i = 0; i < n_slots && vi < victims.size(); ++i) {
                const int32_t x = order[i];
                if (l.slot_of[x] >= 0) {
                    continue;
                }
                const int32_t s = victims[vi];
                // Strata Breakthrough #3: Candidate floor (2.0) and Additive Swap Margin (1.5)
                if (l.score[x] < 2.0) {
                    break; // candidate floor: do not swap for sporadic 1-time routed noise
                }
                const double gain = l.score[x] - l.score[l.exp_in_slot[s]];
                if (gain < 1.5) {
                    break; // additive margin: must beat victim by at least 1.5 routed weight
                }
                cand.push_back({ gain, l.il, x, s });
                vi++;
            }
        }
        std::sort(cand.begin(), cand.end(), [](const swap & a, const swap & b) { return a.gain > b.gain; });
        if (cand.size() > (size_t) std::max(max_swaps, 0)) {
            cand.resize(std::max(max_swaps, 0));
        }

        if (!cand.empty()) {
            if (swap_event == nullptr) {
                ggml_backend_dev_t dev = ggml_backend_get_device(backend);
                if (dev != nullptr) {
                    swap_event = ggml_backend_event_new(dev);
                }
            }
            if (swap_event != nullptr) {
                if (!ggml_backend_event_query(swap_event)) {
                    ggml_backend_event_synchronize(swap_event);
                }
            } else {
                ggml_backend_synchronize(backend);
            }
        }
        std::vector<char> dirty(layers.size(), 0);
        for (const swap & c : cand) {
            layer & l = layers[c.il];
            const int32_t y = l.exp_in_slot[c.s];
            l.slot_of[y] = -1;
            l.slot_of[c.x] = c.s;
            l.exp_in_slot[c.s] = c.x;
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                // queued on the device's stream, one wait at the end (async when the host experts are pinned)
                ggml_backend_tensor_set_async(backend, l.banks[ip], (const uint8_t *) w->data + (size_t) c.x*w->nb[2], (size_t) c.s*w->nb[2], w->nb[2]);
            }
            dirty[c.il] = 1;
        }
        for (layer & l : layers) {
            if (!l.tracked) {
                continue;
            }
            if (dirty[l.il]) {
                ggml_backend_tensor_set_async(backend, l.slot_tab, l.slot_of.data(), 0, l.slot_of.size()*sizeof(int32_t));
            }
            if (!cand.empty()) {
                // Strata Breakthrough #3: 0.7 exponential decay per round preserves routing stability
                for (double & v : l.score) {
                    v *= 0.7;
                }
            }
        }
        if (!cand.empty()) {
            // Strata Breakthrough #4: Async promotion with lag-1 event instead of blocking host thread
            if (swap_event != nullptr) {
                ggml_backend_event_record(swap_event, backend);
            } else {
                ggml_backend_synchronize(backend);
            }
        }
        n_swaps += cand.size();
        n_rebalance++;
    }

    // the CPU op of build_route: count the selected experts and, in the hybrid form, give the CPU branch the experts that are not resident
    // experts that are not resident but have low weight (< skip_weight) or high rank (>= tail_skip) are skipped on CPU (-1)
    static void route(ggml_tensor * dst, int ith, int nth, void * userdata) {
        GGML_UNUSED(nth);
        if (ith != 0) {
            return;
        }
        auto * l = (layer *) userdata;
        const ggml_tensor * ids = dst->src[0];
        const ggml_tensor * weights = dst->src[1];
        GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
        const int32_t * src = (const int32_t *) ids->data;
        const float * wdata = weights && weights->data ? (const float *) weights->data : nullptr;
        const int64_t n = ggml_nelements(ids);
        const int64_t n_expert_used = ids->ne[0];
        const int32_t n_expert = l->owner->n_expert;
        const float skip_weight = l->owner->params.skip_weight;
        const int32_t tail_skip = l->owner->params.tail_skip;
        int32_t * out = (int32_t *) dst->data;
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = src[i];
            const bool valid = e >= 0 && e < n_expert;
            if (valid) {
                l->score[e] += 1.0;
            }
            if (!valid || l->slot_of[e] >= 0) {
                out[i] = -1;
                continue;
            }
            // Host miss candidate for CPU: check tail skip and skip weight
            const int64_t rank = n_expert_used > 0 ? (i % n_expert_used) : 0;
            if (tail_skip > 0 && rank >= tail_skip) {
                out[i] = -1;
                continue;
            }
            if (skip_weight > 0.0f && wdata != nullptr && wdata[i] < skip_weight) {
                out[i] = -1;
                continue;
            }
            out[i] = e;
        }
    }

    void before_compute(int64_t n_tokens) {
        if (n_tokens > params.max_tokens) {
            // a prompt: its counts move many experts once generation starts (a batch just above max_tokens does not trigger that)
            prompt_pending = prompt_pending || n_tokens >= params.prompt_min;
            return;
        }
        if (prompt_pending) {
            prompt_pending = false;
            since = 0;
            rebalance(params.swap_prompt);
            return;
        }
        since += n_tokens;
        if (since >= params.swap_every) {
            since = 0;
            rebalance(params.swap_max);
        }
    }
};

llama_moe_hybrid::llama_moe_hybrid(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, const llama_moe_hybrid_params & params) :
    pimpl(new impl(model, backend, buft, params)) {
}

llama_moe_hybrid::~llama_moe_hybrid() = default;

bool llama_moe_hybrid::tracked(int32_t il) const {
    return il >= 0 && (size_t) il < pimpl->layers.size() && pimpl->layers[il].tracked;
}

bool llama_moe_hybrid::usable(int32_t il, int64_t n_tokens) const {
    return tracked(il) && n_tokens > 0 && n_tokens <= pimpl->params.max_tokens;
}

ggml_backend_t llama_moe_hybrid::backend() const {
    return pimpl->backend;
}

ggml_tensor * llama_moe_hybrid::slot_table(int32_t il) const {
    return tracked(il) ? pimpl->layers[il].slot_tab : nullptr;
}

ggml_tensor * llama_moe_hybrid::bank(const ggml_tensor * w) const {
    const auto it = pimpl->bank_of.find(w);
    return it != pimpl->bank_of.end() ? it->second : nullptr;
}

ggml_tensor * llama_moe_hybrid::build_route(ggml_context * ctx, ggml_tensor * selected, ggml_tensor * weights, int32_t il) const {
    GGML_ASSERT(tracked(il));
    ggml_tensor * args[2] = { selected, weights };
    void * ud = &pimpl->layers[il];
    return ggml_custom_4d(ctx, GGML_TYPE_I32, selected->ne[0], selected->ne[1], 1, 1, args, weights ? 2 : 1, impl::route, 1, ud);
}

void llama_moe_hybrid::count(const ggml_tensor * w, const int32_t * ids, size_t n) const {
    const auto it = pimpl->layer_of.find(w);
    if (it == pimpl->layer_of.end()) {
        return;
    }
    auto & l = pimpl->layers[it->second];
    for (size_t i = 0; i < n; ++i) {
        if (ids[i] >= 0 && ids[i] < pimpl->n_expert) {
            l.score[ids[i]] += 1.0;
        }
    }
}

void llama_moe_hybrid::before_compute(int64_t n_tokens) {
    pimpl->before_compute(n_tokens);
}

int64_t llama_moe_hybrid::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) const {
    impl & p = *pimpl;
    if (backend != p.backend || p.no_alloc) {
        return 0;
    }
    const auto it = p.layer_of.find(w);
    if (it == p.layer_of.end()) {
        return 0;
    }
    const impl::layer & l = p.layers[it->second];
    if (e < 0 || e >= p.n_expert || l.slot_of[e] < 0) {
        return 0;
    }
    // consecutive experts in consecutive slots are one copy
    int64_t n = 1;
    while (e + n <= last && e + n < p.n_expert && l.slot_of[e + n] == l.slot_of[e] + n) {
        n++;
    }
    ggml_tensor * bank = p.bank_of.at(w);
    ggml_reset(p.ctx_views.get());
    ggml_tensor * src_view = ggml_view_3d(p.ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], (size_t) l.slot_of[e]*bank->nb[2]);
    ggml_tensor * dst_view = ggml_view_3d(p.ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  (size_t) e*dst->nb[2]);
    ggml_backend_view_init(src_view);
    ggml_backend_view_init(dst_view);
    ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
    return n;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    if (pimpl->buf) {
        res[pimpl->buft] += pimpl->buf_size;
    }
    return res;
}
