#include "jarvis-sampling.h"

#include "jarvis-vocab.h"
#include "jarvis-grammar.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <numeric>
#include <random>
#include <unordered_map>

static int jarvis_sample_dist(jarvis_token_data_array * cur_p, std::mt19937 & rng) {
    // iterator for the probabilities
#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

    struct probs_iterator {
        typedef std::input_iterator_tag iterator_category;
        typedef float value_type;
        typedef float * pointer;
        typedef float & reference;
        typedef ptrdiff_t difference_type;

        const jarvis_token_data * data;

        bool operator==(const probs_iterator & other) const { return data == other.data; }
        bool operator!=(const probs_iterator & other) const { return data != other.data; }
        const float & operator*() const { return data->p; }
        probs_iterator & operator++() { ++data; return *this; }
        probs_iterator operator++(int) { probs_iterator tmp = *this; ++data; return tmp; }
    };

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

    std::discrete_distribution<int> dist(probs_iterator{cur_p->data}, probs_iterator{cur_p->data + cur_p->size});

    return dist(rng);
}

/*
static void jarvis_log_softmax(float * array, size_t size) {
    float max_l = *std::max_element(array, array + size);
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float p = expf(array[i] - max_l);
        sum += p;
        array[i] = p;
    }

    for (size_t i = 0; i < size; ++i) {
        array[i] = logf(array[i] / sum);
    }
}
*/

static void jarvis_sampler_temp_impl(jarvis_token_data_array * cur_p, float temp) {
    if (temp <= 0.0f) {
        // find the token with the highest logit and set the rest to -inf
        size_t max_i = 0;
        float  max_l = cur_p->data[0].logit;

        for (size_t i = 1; i < cur_p->size; ++i) {
            if (cur_p->data[i    ].logit > max_l) {
                cur_p->data[max_i].logit = -INFINITY;
                max_i = i;
                max_l = cur_p->data[i].logit;
            } else {
                cur_p->data[i].logit = -INFINITY;
            }
        }

        return;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= temp;
    }
}

static void jarvis_sampler_softmax_impl(jarvis_token_data_array * cur_p) {
    GGML_ASSERT(cur_p->size > 0);

    // Sort the logits in descending order
    if (!cur_p->sorted) {
        std::sort(cur_p->data, cur_p->data + cur_p->size, [](const jarvis_token_data & a, const jarvis_token_data & b) {
            return a.logit > b.logit;
        });
        cur_p->sorted = true;
    }

    float max_l = cur_p->data[0].logit;
    float cum_sum = 0.0f;

    for (size_t i = 0; i < cur_p->size; ++i) {
        float p = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        cum_sum += p;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= cum_sum;
    }
}

static void jarvis_sampler_top_k_impl(jarvis_token_data_array * cur_p, int32_t k) {
    // TODO: move bucket sort to separate function so that top_p/tail_free/typical/softmax first is equally fast
    // if (k >= (int32_t)cur_p->size) {
    //     return;
    // }

    if (k <= 0) {
        k = cur_p->size;
    }

    k = std::min(k, (int) cur_p->size);

    // Sort scores in descending order
    if (!cur_p->sorted) {
        auto comp = [](const jarvis_token_data & a, const jarvis_token_data & b) {
            return a.logit > b.logit;
        };
        if (k <= 128) {
            std::partial_sort(cur_p->data, cur_p->data + k, cur_p->data + cur_p->size, comp);
        } else {
            constexpr int   nbuckets     = 128;
            constexpr float bucket_low   = -10.0f;
            constexpr float bucket_high  =  10.0f;
            constexpr float bucket_scale = nbuckets/(bucket_high - bucket_low);
            constexpr float bucket_inter = -bucket_low * bucket_scale;

            std::vector<int> bucket_idx(cur_p->size);
            std::vector<int> histo(nbuckets, 0);

            for (int i = 0; i < (int)cur_p->size; ++i) {
                const float val = cur_p->data[i].logit;
                int ib = int(bucket_scale * val + bucket_inter); //nbuckets * (val - bucket_low) / (bucket_high - bucket_low);
                ib = std::max(0, std::min(nbuckets-1, ib));
                bucket_idx[i] = ib;
                ++histo[ib];
            }
            int nhave = 0;
            int ib = nbuckets - 1;
            for ( ; ib >= 0; --ib) {
                nhave += histo[ib];
                if (nhave >= k) {
                    break;
                }
            }
            std::vector<jarvis_token_data> tmp_tokens(nhave);
            auto * ptr = tmp_tokens.data();
            std::vector<jarvis_token_data*> bucket_ptrs;
            bucket_ptrs.reserve(nbuckets - ib);
            for (int j = nbuckets - 1; j >= ib; --j) {
                bucket_ptrs.push_back(ptr);
                ptr += histo[j];
            }
            for (int i = 0; i < (int)cur_p->size; ++i) {
                int j = bucket_idx[i];
                if (j >= ib) {
                    *bucket_ptrs[nbuckets-1-j]++ = cur_p->data[i];
                }
            }

            ptr = tmp_tokens.data();
            int ndone = 0;
            for (int j = nbuckets-1; j > ib; --j) {
                std::sort(ptr, ptr + histo[j], comp);
                ptr += histo[j];
                ndone += histo[j];
            }
            std::partial_sort(ptr, ptr + k - ndone, ptr + histo[ib], comp);

            std::memcpy(cur_p->data, tmp_tokens.data(), k*sizeof(jarvis_token_data));

        }
        cur_p->sorted = true;
    }
    cur_p->size = k;
}

static uint32_t get_rng_seed(uint32_t seed) {
    if (seed == JARVIS_DEFAULT_SEED) {
        // use system clock if std::random_device is not a true RNG
        static bool is_rd_prng = std::random_device().entropy() == 0;
        if (is_rd_prng) {
            return (uint32_t) std::chrono::system_clock::now().time_since_epoch().count();
        }
        std::random_device rd;
        return rd();
    }
    return seed;
}

// jarvis_sampler API

const char * jarvis_sampler_name(const struct jarvis_sampler * smpl) {
    if (!smpl->iface) {
        return "(null)";
    }

    return smpl->iface->name(smpl);
}

void jarvis_sampler_accept(struct jarvis_sampler * smpl, jarvis_token token) {
    if (smpl->iface->accept) {
        smpl->iface->accept(smpl, token);
    }
}

void jarvis_sampler_apply(struct jarvis_sampler * smpl, struct jarvis_token_data_array * cur_p) {
    GGML_ASSERT(smpl->iface->apply);
    smpl->iface->apply(smpl, cur_p);
}

void jarvis_sampler_reset(struct jarvis_sampler * smpl) {
    if (smpl->iface->reset) {
        smpl->iface->reset(smpl);
    }
}

struct jarvis_sampler * jarvis_sampler_clone(const struct jarvis_sampler * smpl) {
    if (smpl->iface->clone) {
        return smpl->iface->clone(smpl);
    }

    if (smpl->ctx == nullptr) {
        return new jarvis_sampler {
            /* .iface = */ smpl->iface,
            /* .ctx   = */ nullptr,
        };
    }

    GGML_ABORT("the sampler does not support cloning");
}

void jarvis_sampler_free(struct jarvis_sampler * smpl) {
    if (smpl == nullptr) {
        return;
    }

    if (smpl->iface->free) {
        smpl->iface->free(smpl);
    }

    delete smpl;
}

jarvis_token jarvis_sampler_sample(struct jarvis_sampler * smpl, struct jarvis_context * ctx, int32_t idx) {
    const auto * logits = jarvis_get_logits_ith(ctx, idx);

    const int n_vocab = jarvis_n_vocab(jarvis_get_model(ctx));

    // TODO: do not allocate each time
    std::vector<jarvis_token_data> cur;
    cur.reserve(n_vocab);
    for (jarvis_token token_id = 0; token_id < n_vocab; token_id++) {
        cur.emplace_back(jarvis_token_data{token_id, logits[token_id], 0.0f});
    }

    jarvis_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };

    jarvis_sampler_apply(smpl, &cur_p);

    GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size);

    auto token = cur_p.data[cur_p.selected].id;

    jarvis_sampler_accept(smpl, token);

    return token;
}

// sampler chain

static const char * jarvis_sampler_chain_name(const struct jarvis_sampler * /*smpl*/) {
    return "chain";
}

static void jarvis_sampler_chain_accept(struct jarvis_sampler * smpl, jarvis_token token) {
    auto * chain = (jarvis_sampler_chain *) smpl->ctx;

    time_meas tm(chain->t_sample_us, chain->params.no_perf);

    for (auto * smpl : chain->samplers) {
        jarvis_sampler_accept(smpl, token);
    }

    chain->n_sample++;
}

static void jarvis_sampler_chain_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * chain = (jarvis_sampler_chain *) smpl->ctx;

    time_meas tm(chain->t_sample_us, chain->params.no_perf);

    for (auto * smpl : chain->samplers) {
        jarvis_sampler_apply(smpl, cur_p);
    }
}

static void jarvis_sampler_chain_reset(struct jarvis_sampler * smpl) {
    auto * chain = (jarvis_sampler_chain *) smpl->ctx;

    for (auto * smpl : chain->samplers) {
        jarvis_sampler_reset(smpl);
    }

    chain->t_sample_us = 0;
    chain->n_sample    = 0;
}

static struct jarvis_sampler * jarvis_sampler_chain_clone(const struct jarvis_sampler * smpl) {
    const auto * chain_src = (const jarvis_sampler_chain *) smpl->ctx;

    auto * result = jarvis_sampler_chain_init(chain_src->params);

    for (auto * smpl : chain_src->samplers) {
        jarvis_sampler_chain_add(result, jarvis_sampler_clone(smpl));
    }

    return result;
}

static void jarvis_sampler_chain_free(struct jarvis_sampler * smpl) {
    auto * chain = (jarvis_sampler_chain *) smpl->ctx;

    for (auto * smpl : chain->samplers) {
        jarvis_sampler_free(smpl);
    }

    delete chain;
}

static struct jarvis_sampler_i jarvis_sampler_chain_i = {
    /* .name   = */ jarvis_sampler_chain_name,
    /* .accept = */ jarvis_sampler_chain_accept,
    /* .apply  = */ jarvis_sampler_chain_apply,
    /* .reset  = */ jarvis_sampler_chain_reset,
    /* .clone  = */ jarvis_sampler_chain_clone,
    /* .free   = */ jarvis_sampler_chain_free,
};

struct jarvis_sampler * jarvis_sampler_chain_init(struct jarvis_sampler_chain_params params) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_chain_i,
        /* .ctx   = */ new jarvis_sampler_chain {
            /* .params      = */ params,
            /* .samplers    = */ {},
            /* .t_sample_us = */ 0,
            /* .n_sample    = */ 0,
        },
    };
}

void jarvis_sampler_chain_add(struct jarvis_sampler * chain, struct jarvis_sampler * smpl) {
    auto * p = (jarvis_sampler_chain *) chain->ctx;
    p->samplers.push_back(smpl);
}

struct jarvis_sampler * jarvis_sampler_chain_get(const struct jarvis_sampler * chain, int32_t i) {
    const auto * p = (const jarvis_sampler_chain *) chain->ctx;

    if (i < 0 || (size_t) i >= p->samplers.size()) {
        return nullptr;
    }

    return p->samplers[i];
}

struct jarvis_sampler * jarvis_sampler_chain_remove(struct jarvis_sampler * chain, int32_t i) {
    auto * p = (jarvis_sampler_chain *) chain->ctx;

    if (i < 0 || (size_t) i >= p->samplers.size()) {
        return nullptr;
    }

    auto * result = p->samplers[i];
    p->samplers.erase(p->samplers.begin() + i);

    return result;
}

int jarvis_sampler_chain_n(const struct jarvis_sampler * chain) {
    const auto * p = (const jarvis_sampler_chain *) chain->ctx;

    return p->samplers.size();
}

//
// samplers
//

// greedy

static const char * jarvis_sampler_greedy_name(const struct jarvis_sampler * /*smpl*/) {
    return "greedy";
}

static void jarvis_sampler_greedy_apply(struct jarvis_sampler * /*smpl*/, jarvis_token_data_array * cur_p) {
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;
        }
    }
}

static struct jarvis_sampler_i jarvis_sampler_greedy_i = {
    /* .name   = */ jarvis_sampler_greedy_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_greedy_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ nullptr,
    /* .free   = */ nullptr,
};

struct jarvis_sampler * jarvis_sampler_init_greedy() {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_greedy_i,
        /* .ctx   = */ nullptr,
    };
}

// dist

struct jarvis_sampler_dist {
    const uint32_t seed;
          uint32_t seed_cur;

    std::mt19937 rng;
};

static const char * jarvis_sampler_dist_name(const struct jarvis_sampler * /*smpl*/) {
    return "dist";
}

static void jarvis_sampler_dist_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * ctx = (jarvis_sampler_dist *) smpl->ctx;

    jarvis_sampler_softmax_impl(cur_p);

    cur_p->selected = jarvis_sample_dist(cur_p, ctx->rng);
}

static struct jarvis_sampler * jarvis_sampler_dist_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_dist *) smpl->ctx;
    auto * result = jarvis_sampler_init_dist(ctx->seed);

    // copy the state
    {
        auto * result_ctx = (jarvis_sampler_dist *) result->ctx;

        result_ctx->rng = ctx->rng;
    }

    return result;
}

static void jarvis_sampler_dist_reset(struct jarvis_sampler * smpl) {
    auto * ctx = (jarvis_sampler_dist *) smpl->ctx;
    ctx->seed_cur = get_rng_seed(ctx->seed);
    ctx->rng.seed(ctx->seed_cur);
}

static void jarvis_sampler_dist_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_dist *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_dist_i = {
    /* .name   = */ jarvis_sampler_dist_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_dist_apply,
    /* .reset  = */ jarvis_sampler_dist_reset,
    /* .clone  = */ jarvis_sampler_dist_clone,
    /* .free   = */ jarvis_sampler_dist_free,
};

struct jarvis_sampler * jarvis_sampler_init_dist(uint32_t seed) {
    auto seed_cur = get_rng_seed(seed);
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_dist_i,
        /* .ctx   = */ new jarvis_sampler_dist {
            /* .seed     = */ seed,
            /* .seed_cur = */ seed_cur,
            /* .rng      = */ std::mt19937(seed_cur),
        },
    };
}

// softmax

static const char * jarvis_sampler_softmax_name(const struct jarvis_sampler * /*smpl*/) {
    return "softmax";
}

static void jarvis_sampler_softmax_apply(struct jarvis_sampler * /*smpl*/, jarvis_token_data_array * cur_p) {
    jarvis_sampler_softmax_impl(cur_p);
}

static struct jarvis_sampler_i jarvis_sampler_softmax_i = {
    /* .name   = */ jarvis_sampler_softmax_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_softmax_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ nullptr,
    /* .free   = */ nullptr,
};

struct jarvis_sampler * jarvis_sampler_init_softmax() {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_softmax_i,
        /* .ctx   = */ nullptr,
    };
}

// top-k

struct jarvis_sampler_top_k {
    const int32_t k;
};

static const char * jarvis_sampler_top_k_name(const struct jarvis_sampler * /*smpl*/) {
    return "top-k";
}

static void jarvis_sampler_top_k_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    const auto * ctx = (jarvis_sampler_top_k *) smpl->ctx;
    jarvis_sampler_top_k_impl(cur_p, ctx->k);
}

static struct jarvis_sampler * jarvis_sampler_top_k_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_top_k *) smpl->ctx;
    return jarvis_sampler_init_top_k(ctx->k);
}

static void jarvis_sampler_top_k_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_top_k *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_top_k_i = {
    /* .name   = */ jarvis_sampler_top_k_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_top_k_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ jarvis_sampler_top_k_clone,
    /* .free   = */ jarvis_sampler_top_k_free,
};

struct jarvis_sampler * jarvis_sampler_init_top_k(int32_t k) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_top_k_i,
        /* .ctx   = */ new jarvis_sampler_top_k {
            /* .k = */ k,
        },
    };
}

// top-p

struct jarvis_sampler_top_p {
    const float  p;
    const size_t min_keep;
};

static const char * jarvis_sampler_top_p_name(const struct jarvis_sampler * /*smpl*/) {
    return "top-p";
}

static void jarvis_sampler_top_p_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    const auto * ctx = (jarvis_sampler_top_p *) smpl->ctx;

    if (ctx->p >= 1.0f) {
        return;
    }

    jarvis_sampler_softmax_impl(cur_p);

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = cur_p->size;

    for (size_t i = 0; i < cur_p->size; ++i) {
        cum_sum += cur_p->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= ctx->p && i + 1 >= ctx->min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    cur_p->size = last_idx;
}

static struct jarvis_sampler * jarvis_sampler_top_p_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_top_p *) smpl->ctx;
    return jarvis_sampler_init_top_p(ctx->p, ctx->min_keep);
}

static void jarvis_sampler_top_p_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_top_p *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_top_p_i = {
    /* .name   = */ jarvis_sampler_top_p_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_top_p_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ jarvis_sampler_top_p_clone,
    /* .free   = */ jarvis_sampler_top_p_free,
};

struct jarvis_sampler * jarvis_sampler_init_top_p(float p, size_t min_keep) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_top_p_i,
        /* .ctx   = */ new jarvis_sampler_top_p {
            /* .p        = */ p,
            /* .min_keep = */ min_keep,
        },
    };
}

// min-p

struct jarvis_sampler_min_p {
    const float  p;
    const size_t min_keep;
};

static const char * jarvis_sampler_min_p_name(const struct jarvis_sampler * /*smpl*/) {
    return "min-p";
}

static void jarvis_sampler_min_p_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    const auto * ctx = (jarvis_sampler_min_p *) smpl->ctx;

    if (ctx->p <= 0.0f || !cur_p->size) {
        return;
    }

    bool min_p_applied = false;

    // if the cur_p aren't sorted, try the unsorted implementation first
    if (!cur_p->sorted) {
        std::vector<jarvis_token_data> filtered_tokens;

        float max_logit = -FLT_MAX;
        for (size_t i = 0; i < cur_p->size; ++i) {
            max_logit = std::max(max_logit, cur_p->data[i].logit);
        }
        const float min_logit = max_logit + logf(ctx->p); // min logit for p_i >= p * p_max

        for (size_t i = 0; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit >= min_logit) {
                filtered_tokens.push_back(cur_p->data[i]);
            }
        }

        // if we have enough values the operation was a success
        if (filtered_tokens.size() >= ctx->min_keep) {
            memcpy(cur_p->data, filtered_tokens.data(), filtered_tokens.size()*sizeof(jarvis_token_data));
            cur_p->size = filtered_tokens.size();
            min_p_applied = true;
        }
    }

    // if the cur_p are sorted or the unsorted implementation failed, use this implementation
    if (!min_p_applied) {
        // Sort the logits in descending order
        if (!cur_p->sorted) {
            std::sort(cur_p->data, cur_p->data + cur_p->size, [](const jarvis_token_data & a, const jarvis_token_data & b) {
                return a.logit > b.logit;
            });
            cur_p->sorted = true;
        }

        const float min_logit = cur_p->data[0].logit + logf(ctx->p); // min logit for p_i >= p * p_max
        size_t i = 1; // first token always matches

        for (; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit < min_logit && i >= ctx->min_keep) {
                break; // prob too small
            }
        }

        // Resize the output vector to keep only the matching tokens
        cur_p->size = i;
    }
}

static struct jarvis_sampler * jarvis_sampler_min_p_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_min_p *) smpl->ctx;
    return jarvis_sampler_init_min_p(ctx->p, ctx->min_keep);
}

static void jarvis_sampler_min_p_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_min_p *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_min_p_i = {
    /* .name   = */ jarvis_sampler_min_p_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_min_p_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ jarvis_sampler_min_p_clone,
    /* .free   = */ jarvis_sampler_min_p_free,
};

struct jarvis_sampler * jarvis_sampler_init_min_p(float p, size_t min_keep) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_min_p_i,
        /* .ctx   = */ new jarvis_sampler_min_p {
            /* .p        = */ p,
            /* .min_keep = */ min_keep,
        },
    };
}

// tail-free

struct jarvis_sampler_tail_free {
    const float  z;
    const size_t min_keep;
};

static const char * jarvis_sampler_tail_free_name(const struct jarvis_sampler * /*smpl*/) {
    return "tail-free";
}

static void jarvis_sampler_tail_free_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    const auto * ctx = (jarvis_sampler_tail_free *) smpl->ctx;

    if (ctx->z >= 1.0f || cur_p->size <= 2) {
        return;
    }

    jarvis_sampler_softmax_impl(cur_p);

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(cur_p->size - 1);
    std::vector<float> second_derivatives(cur_p->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = cur_p->data[i].p - cur_p->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = std::abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    {
        const float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);

        if (second_derivatives_sum > 1e-6f) {
            for (float & value : second_derivatives) {
                value /= second_derivatives_sum;
            }
        } else {
            for (float & value : second_derivatives) {
                value = 1.0f / second_derivatives.size();
            }
        }
    }

    float cum_sum = 0.0f;
    size_t last_idx = cur_p->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > ctx->z && i >= ctx->min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    cur_p->size = last_idx;
}

static struct jarvis_sampler * jarvis_sampler_tail_free_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_tail_free *) smpl->ctx;
    return jarvis_sampler_init_tail_free(ctx->z, ctx->min_keep);
}

static void jarvis_sampler_tail_free_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_tail_free *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_tail_free_i = {
    /* .name   = */ jarvis_sampler_tail_free_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_tail_free_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ jarvis_sampler_tail_free_clone,
    /* .free   = */ jarvis_sampler_tail_free_free,
};

struct jarvis_sampler * jarvis_sampler_init_tail_free(float z, size_t min_keep) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_tail_free_i,
        /* .ctx   = */ new jarvis_sampler_tail_free {
            /* .z        = */ z,
            /*. min_keep = */ min_keep,
        },
    };
}

// typical

struct jarvis_sampler_typical {
    const float  p;
    const size_t min_keep;
};

static const char * jarvis_sampler_typical_name(const struct jarvis_sampler * /*smpl*/) {
    return "typical";
}

static void jarvis_sampler_typical_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    const auto * ctx = (jarvis_sampler_typical *) smpl->ctx;

    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (ctx->p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    jarvis_sampler_softmax_impl(cur_p);

    float entropy = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        entropy += -cur_p->data[i].p * logf(cur_p->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < cur_p->size; ++i) {
        float shifted_score = fabsf(-logf(cur_p->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(cur_p->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += cur_p->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > ctx->p && i >= ctx->min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<jarvis_token_data> cur_p_new;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        cur_p_new.push_back(cur_p->data[idx]);
    }

    // Replace the data in cur_p with the cur_p_new data
    std::copy(cur_p_new.begin(), cur_p_new.end(), cur_p->data);
    cur_p->size = cur_p_new.size();
    cur_p->sorted = false;
}

static struct jarvis_sampler * jarvis_sampler_typical_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_typical *) smpl->ctx;
    return jarvis_sampler_init_typical(ctx->p, ctx->min_keep);
}

static void jarvis_sampler_typical_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_typical *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_typical_i = {
    /* .name   = */ jarvis_sampler_typical_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_typical_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ jarvis_sampler_typical_clone,
    /* .free   = */ jarvis_sampler_typical_free,
};

struct jarvis_sampler * jarvis_sampler_init_typical(float p, size_t min_keep) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_typical_i,
        /* .ctx   = */ new jarvis_sampler_typical {
            /* .p        = */ p,
            /* .min_keep = */ min_keep,
        },
    };
}

// temp

struct jarvis_sampler_temp {
    const float temp;
};

static const char * jarvis_sampler_temp_name(const struct jarvis_sampler * /*smpl*/) {
    return "temp";
}

static void jarvis_sampler_temp_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    const auto * ctx = (jarvis_sampler_temp *) smpl->ctx;

    jarvis_sampler_temp_impl(cur_p, ctx->temp);
}

static struct jarvis_sampler * jarvis_sampler_temp_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_temp *) smpl->ctx;
    return jarvis_sampler_init_temp(ctx->temp);
}

static void jarvis_sampler_temp_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_temp *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_temp_i = {
    /* .name   = */ jarvis_sampler_temp_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_temp_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ jarvis_sampler_temp_clone,
    /* .free   = */ jarvis_sampler_temp_free,
};

struct jarvis_sampler * jarvis_sampler_init_temp(float temp) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_temp_i,
        /* .ctx   = */ new jarvis_sampler_temp {
            /*.temp = */ temp,
        },
    };
}

// temp-ext

struct jarvis_sampler_temp_ext {
    const float temp;
    const float delta;
    const float exponent;
};

static const char * jarvis_sampler_temp_ext_name(const struct jarvis_sampler * /*smpl*/) {
    return "temp-ext";
}

static void jarvis_sampler_temp_ext_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    const auto * ctx = (jarvis_sampler_temp_ext *) smpl->ctx;
    if (ctx->delta > 0) {
        const float min_temp = std::max(0.0f, ctx->temp - ctx->delta);
        const float max_temp = ctx->temp + ctx->delta;

        float exponent_val = ctx->exponent;

        // no need to do anything if there is only one (or zero) candidates
        if (cur_p->size <= 1) {
            return;
        }

        // Calculate maximum possible entropy
        float max_entropy = -logf(1.0f / cur_p->size);

        jarvis_sampler_softmax_impl(cur_p);

        // Calculate entropy of the softmax probabilities
        float entropy = 0.0f;
        for (size_t i = 0; i < cur_p->size; ++i) {
            float prob = cur_p->data[i].p;
            if (prob > 0.0f) { // Ensure no log(0)
                entropy -= prob * logf(prob);
            }
        }

        // Normalize the entropy (max_entropy cannot be 0 here because we checked cur_p->size != 1 above)
        float normalized_entropy = entropy / max_entropy;

        // Map the normalized entropy to the desired temperature range using the power function
        float dyn_temp = min_temp + (max_temp - min_temp) * powf(normalized_entropy, exponent_val);

    #ifdef DEBUG
        JARVIS_LOG_INFO("Your text maxtemp value is: %f\n", max_temp);
        JARVIS_LOG_INFO("Entropy: %f\n", entropy);
        JARVIS_LOG_INFO("Max Possible Entropy: %f\n", max_entropy);
        JARVIS_LOG_INFO("Normalized Entropy: %f\n", normalized_entropy);
        JARVIS_LOG_INFO("Exponent: %f\n", exponent_val);
        JARVIS_LOG_INFO("Dynamic Temperature (dyn_temp): %f\n", dyn_temp);
    #endif

        // Apply the dynamically calculated temperature scaling
        jarvis_sampler_temp_impl(cur_p, dyn_temp);

        // Re-compute softmax probabilities after scaling logits with dynamic temperature
        const double max_l_double = cur_p->data[0].logit;

        double cum_sum_double = 0.0;
        for (size_t i = 0; i < cur_p->size; ++i) {
            double p = exp(cur_p->data[i].logit - max_l_double);
            cur_p->data[i].p = p; // Store the scaled probability
            cum_sum_double += p;
        }

        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].p /= cum_sum_double; // Re-normalize the probabilities
        }

    #ifdef DEBUG
        // Print the updated top 25 probabilities after temperature scaling
        JARVIS_LOG_INFO("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):\n");
        for (size_t i = 0; i < 25 && i < cur_p->size; ++i) {
            JARVIS_LOG_INFO("Token %zu: %f%%\n", i + 1, cur_p->data[i].p * 100.0f);
        }
    #endif
    } else {
        jarvis_sampler_temp_impl(cur_p, ctx->temp);
    }
}

static struct jarvis_sampler * jarvis_sampler_temp_ext_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_temp_ext *) smpl->ctx;
    return jarvis_sampler_init_temp_ext(ctx->temp, ctx->delta, ctx->exponent);
}

static void jarvis_sampler_temp_ext_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_temp_ext *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_temp_ext_i = {
    /* .name   = */ jarvis_sampler_temp_ext_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_temp_ext_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ jarvis_sampler_temp_ext_clone,
    /* .free   = */ jarvis_sampler_temp_ext_free,
};

struct jarvis_sampler * jarvis_sampler_init_temp_ext(float temp, float delta, float exponent) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_temp_ext_i,
        /* .ctx   = */ new jarvis_sampler_temp_ext {
            /* .temp     = */ temp,
            /* .delta    = */ delta,
            /* .exponent = */ exponent,
        },
    };
}

// xtc

struct jarvis_sampler_xtc {
    const float    probability;
    const float    threshold;
    const size_t   min_keep;

    const uint32_t seed;
    uint32_t       seed_cur;

    std::mt19937   rng;
};

static const char * jarvis_sampler_xtc_name(const struct jarvis_sampler * /*smpl*/) {
    return "xtc";
}

static void jarvis_sample_xtc_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * ctx = (jarvis_sampler_xtc *) smpl->ctx;

    if (ctx->probability <= 0.0f
        || ctx->threshold > 0.5f
        || cur_p->size < 2) {
        return;
    }

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    float chance = distribution(ctx->rng);
    if (chance > ctx->probability) return;

    // in case it's not sorted/recalculated yet
    jarvis_sampler_softmax_impl(cur_p);

    int pos_last = 0;

    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p >= ctx->threshold) {
            pos_last = i;
        } else break;
    }

    if (cur_p->size - pos_last >= ctx->min_keep && pos_last > 0) {
        cur_p->data += pos_last;
        cur_p->size -= pos_last;
    }
}

static struct jarvis_sampler * jarvis_sampler_xtc_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_xtc *) smpl->ctx;
    auto * result = jarvis_sampler_init_xtc(ctx->probability, ctx->threshold, ctx->min_keep, ctx->seed);

    // copy the state
    {
        auto * result_ctx = (jarvis_sampler_xtc *) result->ctx;

        result_ctx->rng = ctx->rng;
    }

    return result;
}

static void jarvis_sampler_xtc_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_xtc *) smpl->ctx;
}

static void jarvis_sampler_xtc_reset(struct jarvis_sampler * smpl) {
    auto * ctx = (jarvis_sampler_xtc *) smpl->ctx;
    ctx->seed_cur = get_rng_seed(ctx->seed);
    ctx->rng.seed(ctx->seed_cur);
}

static struct jarvis_sampler_i jarvis_sampler_xtc_i = {
    /* .name   = */ jarvis_sampler_xtc_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sample_xtc_apply,
    /* .reset  = */ jarvis_sampler_xtc_reset,
    /* .clone  = */ jarvis_sampler_xtc_clone,
    /* .free   = */ jarvis_sampler_xtc_free,
};

struct jarvis_sampler * jarvis_sampler_init_xtc(float p, float t, size_t min_keep, uint32_t seed) {
    auto seed_cur = get_rng_seed(seed);
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_xtc_i,
        /* .ctx   = */ new jarvis_sampler_xtc {
            /* .probability   = */ p,
            /* .threshold     = */ t,
            /* .min_keep      = */ min_keep,
            /* .seed          = */ seed,
            /* .seed_cur      = */ seed_cur,
            /* .rng           = */ std::mt19937(seed_cur),
        },
    };
}

// mirostat

struct jarvis_sampler_mirostat {
    const int32_t n_vocab;

    const uint32_t seed;
          uint32_t seed_cur;

    const float tau;
    const float eta;

    const int32_t m;

    float mu;

    std::mt19937 rng;
};

static const char * jarvis_sampler_mirostat_name(const struct jarvis_sampler * /*smpl*/) {
    return "mirostat";
}

static void jarvis_sampler_mirostat_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * ctx = (jarvis_sampler_mirostat *) smpl->ctx;

    jarvis_sampler_softmax_impl(cur_p);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(ctx->m - 1) && i < cur_p->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(cur_p->data[i].p / cur_p->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, ctx->mu)) / (1 - powf(ctx->n_vocab, -epsilon_hat)), 1 / s_hat);

    jarvis_sampler_top_k_impl(cur_p, std::max(int(k), 1));
    jarvis_sampler_softmax_impl(cur_p);

    const int idx = jarvis_sample_dist(cur_p, ctx->rng);

    cur_p->selected = idx;

    float observed_surprise = -log2f(cur_p->data[idx].p);
    float e = observed_surprise - ctx->tau;

    // Update mu using the learning rate and error
    ctx->mu = ctx->mu - ctx->eta * e;
}

static struct jarvis_sampler * jarvis_sampler_mirostat_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_mirostat *) smpl->ctx;
    auto * result = jarvis_sampler_init_mirostat(ctx->n_vocab, ctx->seed, ctx->tau, ctx->eta, ctx->m);

    // copy the state
    {
        auto * result_ctx = (jarvis_sampler_mirostat *) smpl->ctx;

        result_ctx->mu  = ctx->mu;
        result_ctx->rng = ctx->rng;
    }

    return result;
}

static void jarvis_sampler_mirostat_reset(struct jarvis_sampler * smpl) {
    auto * ctx = (jarvis_sampler_mirostat *) smpl->ctx;
    ctx->mu = 2.0f*ctx->tau;
    ctx->seed_cur = get_rng_seed(ctx->seed);
    ctx->rng.seed(ctx->seed_cur);
}

static void jarvis_sampler_mirostat_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_mirostat *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_mirostat_i = {
    /* .name   = */ jarvis_sampler_mirostat_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_mirostat_apply,
    /* .reset  = */ jarvis_sampler_mirostat_reset,
    /* .clone  = */ jarvis_sampler_mirostat_clone,
    /* .free   = */ jarvis_sampler_mirostat_free,
};

struct jarvis_sampler * jarvis_sampler_init_mirostat(int32_t n_vocab, uint32_t seed, float tau, float eta, int32_t m) {
    auto seed_cur = get_rng_seed(seed);
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_mirostat_i,
        /* .ctx   = */ new jarvis_sampler_mirostat {
            /* .n_vocab  = */ n_vocab,
            /* .seed     = */ seed,
            /* .seed_cur = */ seed_cur,
            /* .tau      = */ tau,
            /* .eta      = */ eta,
            /* .m        = */ m,
            /* .mu       = */ 2.0f*tau,
            /* .rng      = */ std::mt19937(seed_cur),
        },
    };
}

// mirostat v2

struct jarvis_sampler_mirostat_v2 {
    const uint32_t seed;
          uint32_t seed_cur;

    const float tau;
    const float eta;

    float mu;

    std::mt19937 rng;
};

static const char * jarvis_sampler_mirostat_v2_name(const struct jarvis_sampler * /*smpl*/) {
    return "mirostat-v2";
}

static void jarvis_sampler_mirostat_v2_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * ctx = (jarvis_sampler_mirostat_v2 *) smpl->ctx;

    jarvis_sampler_softmax_impl(cur_p);

    // Truncate the words with surprise values greater than mu
    cur_p->size = std::distance(cur_p->data, std::find_if(cur_p->data, cur_p->data + cur_p->size, [&](const jarvis_token_data & candidate) {
        return -log2f(candidate.p) > ctx->mu;
    }));

    if (cur_p->size == 0) {
        cur_p->size = 1;
    }

    // Normalize the probabilities of the remaining words
    jarvis_sampler_softmax_impl(cur_p);

    const int idx = jarvis_sample_dist(cur_p, ctx->rng);

    cur_p->selected = idx;

    float observed_surprise = -log2f(cur_p->data[idx].p);
    float e = observed_surprise - ctx->tau;

    // Update mu using the learning rate and error
    ctx->mu = ctx->mu - ctx->eta * e;
}

static void jarvis_sampler_mirostat_v2_reset(struct jarvis_sampler * smpl) {
    auto * ctx = (jarvis_sampler_mirostat_v2 *) smpl->ctx;
    ctx->mu = 2.0f*ctx->tau;
    ctx->seed_cur = get_rng_seed(ctx->seed);
    ctx->rng.seed(ctx->seed_cur);
}

static struct jarvis_sampler * jarvis_sampler_mirostat_v2_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_mirostat_v2 *) smpl->ctx;

    auto * result = jarvis_sampler_init_mirostat_v2(ctx->seed, ctx->tau, ctx->eta);

    // copy the state
    {
        auto * result_ctx = (jarvis_sampler_mirostat_v2 *) result->ctx;

        result_ctx->mu  = ctx->mu;
        result_ctx->rng = ctx->rng;
    }

    return result;
}

static void jarvis_sampler_mirostat_v2_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_mirostat_v2 *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_mirostat_v2_i = {
    /* .name   = */ jarvis_sampler_mirostat_v2_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_mirostat_v2_apply,
    /* .reset  = */ jarvis_sampler_mirostat_v2_reset,
    /* .clone  = */ jarvis_sampler_mirostat_v2_clone,
    /* .free   = */ jarvis_sampler_mirostat_v2_free,
};

struct jarvis_sampler * jarvis_sampler_init_mirostat_v2(uint32_t seed, float tau, float eta) {
    auto seed_cur = get_rng_seed(seed);
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_mirostat_v2_i,
        /* .ctx   = */ new jarvis_sampler_mirostat_v2 {
            /* .seed     = */ seed,
            /* .seed_cur = */ seed_cur,
            /* .tau      = */ tau,
            /* .eta      = */ eta,
            /* .mu       = */ 2.0f*tau,
            /* .rng      = */ std::mt19937(seed_cur),
        },
    };
}

// grammar

struct jarvis_sampler_grammar {
    const struct jarvis_vocab * vocab;

    std::string grammar_str;
    std::string grammar_root;

    struct jarvis_grammar * grammar;
};

static const char * jarvis_sampler_grammar_name(const struct jarvis_sampler * /*smpl*/) {
    return "grammar";
}

static void jarvis_sampler_grammar_accept_impl(struct jarvis_sampler * smpl, jarvis_token token) {
    auto * ctx = (jarvis_sampler_grammar *) smpl->ctx;
    if (ctx->grammar) {
        jarvis_grammar_accept_impl(*ctx->grammar, token);
    }
}

static void jarvis_sampler_grammar_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * ctx = (jarvis_sampler_grammar *) smpl->ctx;
    if (ctx->grammar) {
        jarvis_grammar_apply_impl(*ctx->grammar, cur_p);
    }
}

static void jarvis_sampler_grammar_reset(struct jarvis_sampler * smpl) {
    auto * ctx = (jarvis_sampler_grammar *) smpl->ctx;
    if (!ctx->grammar) {
        return;
    }

    auto * grammar_new = jarvis_grammar_init_impl(ctx->grammar->vocab, ctx->grammar_str.c_str(), ctx->grammar_root.c_str());

    jarvis_grammar_free_impl(ctx->grammar);
    ctx->grammar = grammar_new;
}

static struct jarvis_sampler * jarvis_sampler_grammar_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_grammar *) smpl->ctx;

    auto * result = jarvis_sampler_init_grammar_impl(*ctx->vocab, nullptr, nullptr);

    // copy the state
    {
        auto * result_ctx = (jarvis_sampler_grammar *) result->ctx;

        if (ctx->grammar) {
            result_ctx->grammar_str  = ctx->grammar_str;
            result_ctx->grammar_root = ctx->grammar_root;

            result_ctx->grammar = jarvis_grammar_clone_impl(*ctx->grammar);
        }
    }

    return result;
}

static void jarvis_sampler_grammar_free(struct jarvis_sampler * smpl) {
    const auto * ctx = (jarvis_sampler_grammar *) smpl->ctx;

    if (ctx->grammar) {
        jarvis_grammar_free_impl(ctx->grammar);
    }

    delete ctx;
}

static struct jarvis_sampler_i jarvis_sampler_grammar_i = {
    /* .name   = */ jarvis_sampler_grammar_name,
    /* .accept = */ jarvis_sampler_grammar_accept_impl,
    /* .apply  = */ jarvis_sampler_grammar_apply,
    /* .reset  = */ jarvis_sampler_grammar_reset,
    /* .clone  = */ jarvis_sampler_grammar_clone,
    /* .free   = */ jarvis_sampler_grammar_free,
};

struct jarvis_sampler * jarvis_sampler_init_grammar_impl(const struct jarvis_vocab & vocab, const char * grammar_str, const char * grammar_root) {
    auto * ctx = new jarvis_sampler_grammar;

    if (grammar_str != nullptr && grammar_str[0] != '\0') {
        *ctx = {
            /* .vocab        = */ &vocab,
            /* .grammar_str  = */ grammar_str,
            /* .grammar_root = */ grammar_root,
            /* .grammar      = */ jarvis_grammar_init_impl(&vocab, grammar_str, grammar_root),
        };
    } else {
        *ctx = {
            /* .vocab        = */ &vocab,
            /* .grammar_str  = */ {},
            /* .grammar_root = */ {},
            /* .grammar      = */ nullptr,
        };
    }

    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_grammar_i,
        /* .ctx   = */ ctx,
    };
}

// penalties

struct jarvis_sampler_penalties {
    const int32_t     n_vocab;
    const jarvis_token special_eos_id;
    const jarvis_token linefeed_id;

    const int32_t penalty_last_n;
    const float   penalty_repeat;
    const float   penalty_freq;
    const float   penalty_present;

    const bool    penalize_nl;
    const bool    ignore_eos;

    ring_buffer<jarvis_token> prev;
};

static const char * jarvis_sampler_penalties_name(const struct jarvis_sampler * /*smpl*/) {
    return "penalties";
}

static void jarvis_sampler_penalties_accept(struct jarvis_sampler * smpl, jarvis_token token) {
    auto * ctx = (jarvis_sampler_penalties *) smpl->ctx;
    if (ctx->penalty_last_n == 0) {
        return;
    }

    ctx->prev.push_back(token);
}

static void jarvis_sampler_penalties_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * ctx = (jarvis_sampler_penalties *) smpl->ctx;

    if (ctx->ignore_eos) {
        assert(ctx->special_eos_id >= 0);

        // optimistically check if the candidates are not yet sorted/shuffled/truncated
        if (cur_p->size > (size_t) ctx->special_eos_id && cur_p->data[ctx->special_eos_id].id == ctx->special_eos_id) {
            cur_p->data[ctx->special_eos_id].logit = -INFINITY;
        } else {
            // else, search for the special EOS token
            for (size_t i = 0; i < cur_p->size; ++i) {
                if (cur_p->data[i].id == ctx->special_eos_id) {
                    cur_p->data[i].logit = -INFINITY;
                    break;
                }
            }
        }
    }

    if ((ctx->penalty_last_n == 0) ||
        (ctx->penalty_repeat == 1.0f && ctx->penalty_freq == 0.0f && ctx->penalty_present == 0.0f)) {
        return;
    }

    bool nl_found = false;
    size_t nl_idx = 0;
    float nl_logit = -INFINITY;
    if (!ctx->penalize_nl) {
        assert(ctx->linefeed_id >= 0);

        // optimistically check if the candidates are not yet sorted/shuffled/truncated
        if (cur_p->size > (size_t) ctx->linefeed_id && cur_p->data[ctx->linefeed_id].id == ctx->linefeed_id) {
            nl_found = true;
            nl_idx = ctx->linefeed_id;
            nl_logit = cur_p->data[ctx->linefeed_id].logit;
        } else {
            // else, search for the linefeed token
            for (size_t i = 0; i < cur_p->size; ++i) {
                if (cur_p->data[i].id == ctx->linefeed_id) {
                    nl_found = true;
                    nl_idx = i;
                    nl_logit = cur_p->data[i].logit;
                    break;
                }
            }
        }
    }

    // Create a frequency map to count occurrences of each token in last_tokens
    // TODO: optimize this by maintaining the token count in the sampler context
    using jarvis_token_cnt = std::unordered_map<jarvis_token, int>;
    jarvis_token_cnt token_count;

    for (int i = 0; i < std::min<int>(ctx->penalty_last_n, ctx->prev.size()); ++i) {
        token_count[ctx->prev.rat(i)]++;
    }

    // Apply frequency and presence penalties to the cur_p
    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto token_iter = token_count.find(cur_p->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (cur_p->data[i].logit <= 0) {
            cur_p->data[i].logit *= ctx->penalty_repeat;
        } else {
            cur_p->data[i].logit /= ctx->penalty_repeat;
        }

        cur_p->data[i].logit -= float(count) * ctx->penalty_freq + float(count > 0) * ctx->penalty_present;
    }

    cur_p->sorted = false;

    if (!ctx->penalize_nl && nl_found) {
        // restore the logit of the newline token if it was penalized
        cur_p->data[nl_idx].logit = nl_logit;
    }
}

static void jarvis_sampler_penalties_reset(struct jarvis_sampler * smpl) {
    auto * ctx = (jarvis_sampler_penalties *) smpl->ctx;
    ctx->prev.clear();
}

static struct jarvis_sampler * jarvis_sampler_penalties_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_penalties *) smpl->ctx;
    auto * result = jarvis_sampler_init_penalties(
            ctx->n_vocab,
            ctx->special_eos_id,
            ctx->linefeed_id,
            ctx->penalty_last_n,
            ctx->penalty_repeat,
            ctx->penalty_freq,
            ctx->penalty_present,
            ctx->penalize_nl,
            ctx->ignore_eos);

    // copy the state
    {
        auto * result_ctx = (jarvis_sampler_penalties *) result->ctx;

        result_ctx->prev = ctx->prev;
    }

    return result;
}

static void jarvis_sampler_penalties_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_penalties *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_penalties_i = {
    /* .name   = */ jarvis_sampler_penalties_name,
    /* .accept = */ jarvis_sampler_penalties_accept,
    /* .apply  = */ jarvis_sampler_penalties_apply,
    /* .reset  = */ jarvis_sampler_penalties_reset,
    /* .clone  = */ jarvis_sampler_penalties_clone,
    /* .free   = */ jarvis_sampler_penalties_free,
};

struct jarvis_sampler * jarvis_sampler_init_penalties(
        int32_t n_vocab,
        jarvis_token special_eos_id,
        jarvis_token linefeed_id,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        bool penalize_nl,
        bool ignore_eos) {
    if (linefeed_id == JARVIS_TOKEN_NULL) {
        penalize_nl = true;
    }

    if (special_eos_id == JARVIS_TOKEN_NULL) {
        ignore_eos = false;
    }

    penalty_last_n = std::max(penalty_last_n, 0);

    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_penalties_i,
        /* .ctx   = */ new jarvis_sampler_penalties {
            /* .n_vocab         = */ n_vocab,
            /* .special_eos_id  = */ special_eos_id,
            /* .linefeed_id     = */ linefeed_id,
            /* .penalty_last_n  = */ penalty_last_n,
            /* .penalty_repeat  = */ penalty_repeat,
            /* .penalty_freq    = */ penalty_freq,
            /* .penalty_present = */ penalty_present,
            /* .penalize_nl     = */ penalize_nl,
            /* .ignore_eos      = */ ignore_eos,
            /* .prev            = */ ring_buffer<jarvis_token>(penalty_last_n),
        },
    };
}

// DRY

struct jarvis_sampler_dry {
    int32_t total_context_size;

    const float   dry_multiplier;
    const float   dry_base;
    const int32_t dry_allowed_length;
    const int32_t dry_penalty_last_n;

    std::unordered_multimap<jarvis_token, std::vector<jarvis_token>> dry_processed_breakers;
    std::vector<int> dry_repeat_count;
    std::unordered_map<jarvis_token, int> dry_max_token_repeat;
    ring_buffer<jarvis_token> last_tokens;
};

// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
static void get_overlapping_token_sequences(const jarvis_vocab & vocab, const std::string& str, std::unordered_multimap<jarvis_token, std::vector<jarvis_token>>& token_sequences, int max_tail_len = -1) {
    for (jarvis_token token_id = 0; token_id < (jarvis_token)vocab.n_vocab; token_id++) {
        std::string word = jarvis_detokenize(vocab, {token_id}, true);
        if (word.find(str) != std::string::npos) {
            token_sequences.emplace(token_id, std::vector<jarvis_token>());
        } else {
            size_t word_len = word.size(), str_len = str.size();
            size_t pos = -1;
            while ((pos = word.find(str[0], pos + 1)) != std::string::npos) {
                bool match = true;
                size_t i;
                for (i = 1; i < str_len && i + pos < word_len; ++i) {
                    if (word[pos + i] != str[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    std::vector<jarvis_token> tokenization = jarvis_tokenize_internal(vocab, str.substr(i), false, false);
                    if (max_tail_len >= 0 && tokenization.size() > (size_t)max_tail_len) {
                        tokenization.resize(max_tail_len);
                    }

                    // Ensure we don't already have a duplicate matching tokenization
                    auto its = token_sequences.equal_range(token_id);
                    bool found = false;
                    for (auto it = its.first; it != its.second; ++it) {
                        if (tokenization == it->second) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        token_sequences.emplace(token_id, tokenization);
                    }
                }
            }
        }
    }
}

static const char * jarvis_sampler_dry_name(const struct jarvis_sampler * /*smpl*/) {
    return "dry";
}

static void jarvis_sampler_dry_accept(struct jarvis_sampler * smpl, jarvis_token token) {
    auto * ctx = (jarvis_sampler_dry *) smpl->ctx;
    if (ctx->dry_multiplier == 0.0f || ctx->dry_base < 1.0f || ctx->dry_penalty_last_n == 0) {
        return;
    }

    ctx->last_tokens.push_back(token);
}

// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
static void jarvis_sampler_dry_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * ctx = (jarvis_sampler_dry *) smpl->ctx;

    if (ctx->dry_multiplier == 0.0f || ctx->dry_base < 1.0f || ctx->dry_penalty_last_n == 0) {
        return;
    }

    int32_t effective_dry_penalty_last_n = (ctx->dry_penalty_last_n == -1) ? ctx->total_context_size : std::max(ctx->dry_penalty_last_n, 0);
    int last_n_repeat = std::min(std::min((int)ctx->last_tokens.size(), effective_dry_penalty_last_n), ctx->total_context_size);

    if (last_n_repeat <= ctx->dry_allowed_length) {
        return;
    }

    ctx->dry_repeat_count.assign(last_n_repeat, 0);
    ctx->dry_max_token_repeat.clear();

    // Step 1: Look for restart sequences to limit the maximum repetition length.
    // Work backwards through the context looking for any token that begins a restart sequence.
    //
    // The collection `restart_sequences` is a mapping from a "head" token to all "tail"
    // sequences that together comprise a restart sequence. This allows us to quickly check
    // whether each token is the head of a complete sequence. Most restart sequences are actually
    // a single token, and for these the "tail" is an empty vector.
    //
    // If the token is a "head", test all restart sequences that begin with this token
    // (there will often only be one sequence for each token, but if sequences like 'aaaq1' and
    // 'aaa1' are used as restart strings, both could start with 'aaa' when tokenized). The
    // longest matching sequence (if any) is used to limit the maximum repetition length.
    //
    // Note that in the case case of a short sequence contained in a longer one, this might fail to
    // find the smallest value for `rep_limit`. For example, if 'amniotic' and 'ni' are both used as
    // restart sequences, 'ni' will be found first, and since it's shorter it will fail to suppress
    // 'otic'. This is a minor issue since fully contained restart sequences are likely to be rare.
    //
    // This is theoretically worst-case O(N^2) for arbitrary restart sequences, which is why we
    // have already clamped the maximum tail sequence length when generating `restart_sequences`.
    // With clamping, this scan is O(N) in the context length.

    int rep_limit = last_n_repeat;
    for (int i = 0; i < last_n_repeat; ++i) {
        jarvis_token token = ctx->last_tokens.rat(i);
        auto its = ctx->dry_processed_breakers.equal_range(token);
        if (its.first == ctx->dry_processed_breakers.end()) {
            continue;
        }
        int longest_match = -1;
        for (auto it = its.first; it != its.second; ++it) {
            // Note that (*it) does not contain the head character, so seq_len will be
            // the restart sequence length minus 1.
            // In the common case of a single-token restart sequence, (*it) will be empty
            // and we will trivially match.
            int seq_len = (int)it->second.size();
            if (seq_len > longest_match && seq_len <= (int)i) {
                bool match = true;
                for (int offset = 0; offset < seq_len; ++offset) {
                    // The -1 when indexing `last_tokens` is because we already matched the head.
                    if (it->second[offset] != ctx->last_tokens.rat(i - offset - 1)) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    longest_match = seq_len;
                }
            }
        }
        if (longest_match >= 0) {
            // We found a restart sequence starting `i` tokens from the end and continuing for
            // `longest_match` tokens.
            rep_limit = i - longest_match;
            break;
        }
    }
    if (rep_limit < ctx->dry_allowed_length) {
        return;
    }

    // Step 2: Iterate in reverse over the last N tokens of the context, using the "Z-algorithm" (in
    // the reverse direction) to efficiently compute the positions and lengths of suffixes appearing
    // elsewhere in the context. We limit the suffix length to `rep_limit` to respect restart sequences.
    //
    // This algorithm is not currently documented on Wikipedia, but there is a clear description here:
    // https://ivanyu.me/blog/2014/10/15/z-algorithm/
    //
    // The code below is adapted from the public domain implementation by the same author here:
    // https://github.com/ivanyu/string-algorithms/blob/master/z_algorithm.py
    //
    // Example:
    // Last N tokens: a b c c b c y a b c
    // Repeat counts: 0 0 3 1 0 2 0 0 0 0
    //                    ^
    //   This `3` means that the last three tokens of the context (a b c) also appear here.
    //
    // This step is worst case O(N) since the Z-algorithm is linear, despite the appearance of nested
    // for/while loops. This can be seen by observing that the `lt` and `rt` bounds are set after each
    // repeated suffix is detected (i.e. after each while loop when n > 0). These bound variables
    // ensure that the inner while loops only examine each token in the context once as the outer
    // for loop iterates over the context.

    {
        const int last = last_n_repeat - 1;
        int rt = 0, lt = 0;

        for (int k = 1; k < last_n_repeat; ++k) {
            if (k > rt) {
                // If k is outside the current Z-box, do naive computation.
                int n = 0;
                while (n + k < last_n_repeat && ctx->last_tokens.rat(n) == ctx->last_tokens.rat(n+k)) {
                    ++n;
                }
                ctx->dry_repeat_count[last - k] = std::min(n, rep_limit);
                if (n > 0) {
                    lt = k;
                    rt = k+n-1;
                }
            } else {
                // If k is inside the current Z-box, consider two cases.

                int p = k - lt; // Pair index.
                int right_part_len = rt - k + 1;

                if (ctx->dry_repeat_count[last - p] < right_part_len) {
                    int n = std::min(ctx->dry_repeat_count[last - p], rep_limit);
                    ctx->dry_repeat_count[last - k] = n;
                } else {
                    int i = rt + 1;
                    while (i < last_n_repeat && ctx->last_tokens.rat(i) == ctx->last_tokens.rat(i - k)) {
                        i += 1;
                    }

                    int n = std::min(i - k, rep_limit);
                    ctx->dry_repeat_count[last - k] = n;
                    lt = k;
                    rt = i - 1;
                }
            }
        }
    }

    // Step 3: Iterate over dry_repeat_count and last_tokens, examining the maximum repeat length
    // that would be generated by emitting each new token that would extend a sequence.
    //
    // Following the same example as above:
    // Last N tokens: a b c c b c y a b c
    // Repeat counts: 0 0 3 1 0 2 0 0 0 0
    //
    // For each non-zero, look ahead one token. This token, if emitted, would extend the repetition.
    // c: 3 -> 4 (from `a b c` to `a b c c`)
    // b: 1 -> 2 (from `c` to `c b`)
    // y: 2 -> 3 (from `b c` to `b c y`)

    for (int i = 0; i < last_n_repeat - 1; ++i) {
        int repeat_len = ctx->dry_repeat_count[i];
        if (repeat_len >= ctx->dry_allowed_length) {
            // This token ends a repeat, so the next token would continue one.
            // By convention, the value of `repeat_len` only includes the tokens currently
            // in the context, not the new token that would be added.
            jarvis_token token = ctx->last_tokens.rat(last_n_repeat - 2 - i);
            // Track the maximum sequence ending in this token.
            const auto& it = ctx->dry_max_token_repeat.find(token);
            if (it == ctx->dry_max_token_repeat.end() || it->second < repeat_len) {
                ctx->dry_max_token_repeat[token] = repeat_len;
            }
        }
    }

    // Step 4: Apply logit penalties based on the maximum repeat length for relevant tokens.

    // Prevent floating point overflow in `pow(penalty_base, exponent)` by clamping to `max_exponent`.
    // Compute it from `penalty_base` and the approximate log of `std::numeric_limits<float>::max()`
    const float FLOAT_MAX_LOG = 88.7228391f;
    int max_exponent = 0;
    if (ctx->dry_base > 1.000001f) {
        max_exponent = FLOAT_MAX_LOG / std::log(ctx->dry_base);
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto& af_kvp = ctx->dry_max_token_repeat.find(cur_p->data[i].id);
        if (af_kvp != ctx->dry_max_token_repeat.end()) {
            // Check all sequence breakers starting with this token
            auto range = ctx->dry_processed_breakers.equal_range(cur_p->data[i].id);
            bool is_single_token_breaker = false;

            for (auto it = range.first; it != range.second; ++it) {
                if (it->second.empty()) {
                    is_single_token_breaker = true;
                    break;
                }
            }

            // Apply penalty only if it's not a single-token sequence breaker
            if (!is_single_token_breaker) {
                int repeat_exp = af_kvp->second - ctx->dry_allowed_length;
                if (max_exponent > 0 && repeat_exp > max_exponent) {
                    repeat_exp = max_exponent;
                }
                float penalty = ctx->dry_multiplier * std::pow(ctx->dry_base, repeat_exp);
                cur_p->data[i].logit -= penalty;
            }
        }
    }

    cur_p->sorted = false;
}

static void jarvis_sampler_dry_reset(struct jarvis_sampler * smpl) {
    auto * ctx = (jarvis_sampler_dry *) smpl->ctx;
    ctx->last_tokens.clear();
    ctx->dry_repeat_count.clear();
    ctx->dry_max_token_repeat.clear();
}

static struct jarvis_sampler * jarvis_sampler_dry_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (jarvis_sampler_dry *) smpl->ctx;

    // nullptr is passed as vocab because it is only needed for raw sequence breaker processing, which we have already done and will be copying
    auto * result = jarvis_sampler_init_dry(nullptr, ctx->dry_multiplier, ctx->dry_base, ctx->dry_allowed_length, ctx->dry_penalty_last_n, NULL, 0);
    // Copy the state, including the processed breakers
    {
        auto * result_ctx = (jarvis_sampler_dry *) result->ctx;
        result_ctx->dry_processed_breakers = ctx->dry_processed_breakers;
        result_ctx->dry_repeat_count = ctx->dry_repeat_count;
        result_ctx->dry_max_token_repeat = ctx->dry_max_token_repeat;
        result_ctx->last_tokens = ctx->last_tokens;
    }

    return result;
}

static void jarvis_sampler_dry_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_dry *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_dry_i = {
    /* .name   = */ jarvis_sampler_dry_name,
    /* .accept = */ jarvis_sampler_dry_accept,
    /* .apply  = */ jarvis_sampler_dry_apply,
    /* .reset  = */ jarvis_sampler_dry_reset,
    /* .clone  = */ jarvis_sampler_dry_clone,
    /* .free   = */ jarvis_sampler_dry_free,
};

struct jarvis_sampler * jarvis_sampler_init_dry_impl(const struct jarvis_vocab & vocab, int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const char** seq_breakers, size_t num_breakers) {
    int32_t effective_dry_penalty_last_n = (dry_penalty_last_n == -1) ? context_size : std::max(dry_penalty_last_n, 0);
    std::unordered_multimap<jarvis_token, std::vector<jarvis_token>> processed_breakers;
    const int MAX_CHAR_LEN = 40;
    const int MAX_SEQ_LEN = 20;

    const bool dry_enabled = (dry_multiplier != 0.0f && dry_base >= 1.0f && dry_penalty_last_n != 0);

    if (dry_enabled && seq_breakers != nullptr && num_breakers > 0) {
        // Process sequence breakers
        for (size_t i = 0; i < num_breakers; ++i) {
            if (seq_breakers[i] == nullptr || std::strlen(seq_breakers[i]) == 0) {
                JARVIS_LOG_WARN("skipping null or empty DRY sequence breaker at index %zu\n", i);
                continue;
            }

            std::string sequence_break(seq_breakers[i]);
            if (sequence_break.empty()) {
                JARVIS_LOG_WARN("skipping empty DRY sequence breaker\n");
                continue;
            }

            if (sequence_break.size() > MAX_CHAR_LEN) {
                JARVIS_LOG_WARN("truncating DRY sequence breaker to %d characters\n", MAX_CHAR_LEN);
                sequence_break.resize(MAX_CHAR_LEN);
            }

            get_overlapping_token_sequences(vocab, sequence_break, processed_breakers, MAX_SEQ_LEN);
        }
    }

    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_dry_i,
        /* .ctx   = */ new jarvis_sampler_dry {
            /* .total_context_size     = */ context_size,
            /* .dry_multiplier         = */ dry_multiplier,
            /* .dry_base               = */ dry_base,
            /* .dry_allowed_length     = */ dry_allowed_length,
            /* .dry_penalty_last_n     = */ dry_penalty_last_n,
            /* .dry_processed_breakers = */ std::move(processed_breakers),
            /* .dry_repeat_count       = */ dry_enabled ? std::vector<int>(effective_dry_penalty_last_n, 0) : std::vector<int>{},
            /* .dry_max_token_repeat   = */ {},
            /* .last_tokens            = */ dry_enabled ? ring_buffer<jarvis_token>(effective_dry_penalty_last_n) : ring_buffer<jarvis_token>(0),
        },
    };
}

// wrapper for test-sampling.cpp
struct jarvis_sampler * jarvis_sampler_init_dry_testing(int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<jarvis_token>>& seq_breakers) {
    jarvis_vocab dummy_vocab;
    auto * result = jarvis_sampler_init_dry_impl(dummy_vocab, context_size, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, NULL, 0);
    auto * ctx = (jarvis_sampler_dry *) result->ctx;

    // Process the token-based sequence breakers
    ctx->dry_processed_breakers.clear();
    if (seq_breakers.empty()) {
        JARVIS_LOG_WARN("empty DRY sequence breakers list in jarvis_sampler_init_dry_testing\n");
    } else {
        for (const auto& breaker : seq_breakers) {
            if (breaker.empty()) {
                JARVIS_LOG_WARN("skipping DRY empty sequence breaker\n");
                continue;
            }
            jarvis_token head_token = breaker[0];
            std::vector<jarvis_token> tail_tokens(breaker.begin() + 1, breaker.end());
            ctx->dry_processed_breakers.emplace(head_token, std::move(tail_tokens));
        }

        if (ctx->dry_processed_breakers.empty()) {
            JARVIS_LOG_WARN("no valid DRY sequence breakers processed in jarvis_sampler_init_dry_testing\n");
        }
    }

    return result;
}

// logit-bias

struct jarvis_sampler_logit_bias {
    const int32_t n_vocab;

    const std::vector<jarvis_logit_bias> logit_bias;

    std::vector<jarvis_logit_bias> to_search;
};

static const char * jarvis_sampler_logit_bias_name(const struct jarvis_sampler * /*smpl*/) {
    return "logit-bias";
}

static void jarvis_sampler_logit_bias_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * ctx = (jarvis_sampler_logit_bias *) smpl->ctx;

    if (ctx->logit_bias.empty()) {
        return;
    }

    ctx->to_search.clear();

    // update the candidates that have not been shuffled in the vocabulary (i.e. idx == id)
    for (const auto & lb : ctx->logit_bias) {
        if (lb.token >= 0 && cur_p->size > (size_t) lb.token && cur_p->data[lb.token].id == lb.token) {
            cur_p->data[lb.token].logit += lb.bias;
        } else {
            ctx->to_search.push_back(lb);
        }
    }

    if (ctx->to_search.empty()) {
        return;
    }

    // search for the remaining candidates that were not found in the previous step
    for (size_t i = 0; i < cur_p->size; ++i) {
        for (const auto & lb : ctx->to_search) {
            if (cur_p->data[i].id == lb.token) {
                cur_p->data[i].logit += lb.bias;
                break;
            }
        }
    }
}

static struct jarvis_sampler * jarvis_sampler_logit_bias_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_logit_bias *) smpl->ctx;
    return jarvis_sampler_init_logit_bias(ctx->n_vocab, ctx->logit_bias.size(), ctx->logit_bias.data());
}

static void jarvis_sampler_logit_bias_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_logit_bias *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_logit_bias_i = {
    /* .name   = */ jarvis_sampler_logit_bias_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_logit_bias_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ jarvis_sampler_logit_bias_clone,
    /* .free   = */ jarvis_sampler_logit_bias_free,
};

struct jarvis_sampler * jarvis_sampler_init_logit_bias(
                         int32_t   n_vocab,
                         int32_t   n_logit_bias,
          const jarvis_logit_bias * logit_bias) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_logit_bias_i,
        /* .ctx   = */ new jarvis_sampler_logit_bias {
            /* .n_vocab    = */ n_vocab,
            /* .logit_bias = */ std::vector<jarvis_logit_bias>(logit_bias, logit_bias + n_logit_bias),
            /* .to_search  = */ {},
        },
    };
}

// infill

//#define GGML_DEBUG_SAMPLER_INFILL

struct jarvis_sampler_infill {
    const struct jarvis_vocab * vocab;

    std::vector<char> buf0;
    std::vector<char> buf1;
};

static const char * jarvis_sampler_infill_name(const struct jarvis_sampler * /*smpl*/) {
    return "infill";
}

static void jarvis_sampler_infill_apply(struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p) {
    auto * ctx = (jarvis_sampler_infill *) smpl->ctx;

    jarvis_sampler_softmax_impl(cur_p);

#if defined(GGML_DEBUG_SAMPLER_INFILL)
#define LOG_DBG_CUR JARVIS_LOG_DEBUG
#else
#define LOG_DBG_CUR(...)
#endif

    for (size_t i = 0; i < cur_p->size; ++i) {
        LOG_DBG_CUR("%s: cur_p[%3zu] = { id: %6d, p: %.6f, logit: %6.3f }\n", __func__, i, cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }

    float p_txt_sum = 0.0f;
    float p_eog_sum = 0.0f;

    for (size_t i = 0; i < cur_p->size; ++i) {
        if (jarvis_token_is_eog_impl(*ctx->vocab, cur_p->data[i].id)) {
            p_eog_sum += cur_p->data[i].p;
        } else {
            p_txt_sum += cur_p->data[i].p;
        }
    }

    const float rat = p_eog_sum == 0.0 ? INFINITY : p_txt_sum / p_eog_sum; GGML_UNUSED(rat);

    LOG_DBG_CUR("%s: p_txt_sum = %.2f, p_eog_sum = %.2f, rat = %.2f, n = %zu\n", __func__, p_txt_sum, p_eog_sum, rat, cur_p->size);

    if (3*p_eog_sum*cur_p->size > p_txt_sum) {
        LOG_DBG_CUR("%s: the ratio p_txt/p_eog = %.2f is too low -> sampling EOG\n", __func__, p_txt_sum/p_eog_sum);

        // keep just the EOG tokens
        const auto size_org = cur_p->size;

        cur_p->size = 0;

        float p_sum = 0.0f;

        for (size_t i = 0; i < size_org; ++i) {
            if (jarvis_token_is_eog_impl(*ctx->vocab, cur_p->data[i].id)) {
                p_sum += cur_p->data[i].p;

                cur_p->data[cur_p->size++] = cur_p->data[i];
            }
        }

        // normalize probs
        for (size_t i = 0; i < cur_p->size; ++i) {
            cur_p->data[i].p /= p_sum;
        }

        return;
    }

    size_t n_combined = 0; GGML_UNUSED(n_combined);

    // combine tokens with common prefix
    for (size_t i0 = 0; i0 < cur_p->size; ++i0) {
        for (size_t i1 = 0; i1 < cur_p->size; ++i1) {
            if (cur_p->data[i0].logit == -INFINITY) {
                break;
            }

            if (i0 == i1 || cur_p->data[i1].logit == -INFINITY) {
                continue;
            }

            int len0 = jarvis_token_to_piece_impl(*ctx->vocab, cur_p->data[i0].id, ctx->buf0.data(), ctx->buf0.size(), 0, false);
            if (len0 < 0) {
                ctx->buf0.resize(len0);
                len0 = jarvis_token_to_piece_impl(*ctx->vocab, cur_p->data[i0].id, ctx->buf0.data(), ctx->buf0.size(), 0, false);
                assert(len0 > 0);
            }

            int len1 = jarvis_token_to_piece_impl(*ctx->vocab, cur_p->data[i1].id, ctx->buf1.data(), ctx->buf1.size(), 0, false);
            if (len1 < 0) {
                ctx->buf1.resize(len1);
                len1 = jarvis_token_to_piece_impl(*ctx->vocab, cur_p->data[i1].id, ctx->buf1.data(), ctx->buf1.size(), 0, false);
                assert(len1 > 0);
            }

            // token i0 is a prefix of token i1
            if (len0 > 0 && len0 <= len1 && memcmp(ctx->buf0.data(), ctx->buf1.data(), len0) == 0) {
                int dst = i0;
                int src = i1;

                // merge into the token with higher probability
                if (cur_p->data[i1].p > cur_p->data[i0].p) {
                    std::swap(dst, src);
                }

                cur_p->data[dst].p += cur_p->data[src].p;
                cur_p->data[src].logit = -INFINITY;
                cur_p->data[src].p     = 0.0f;

                n_combined++;
            }
        }
    }

    size_t n_non_eog = 0;

    size_t size_org = cur_p->size;

    float p_sum = 0.0f;
    float thold = 0.2f;

    cur_p->size = 0;

    LOG_DBG_CUR("%s: n_combined = %zu, applying thold = %.3f\n", __func__, n_combined, thold);

    for (size_t i = 0; i < size_org; ++i) {
        const bool is_eog = jarvis_token_is_eog_impl(*ctx->vocab, cur_p->data[i].id);

        if (cur_p->data[i].p < thold && !is_eog) {
            continue;
        }

        if (!is_eog) {
            ++n_non_eog;
        }

        p_sum += cur_p->data[i].p;

        // keep this token
        cur_p->data[cur_p->size++] = cur_p->data[i];
    }

    LOG_DBG_CUR("%s: n_non_eog = %zu\n", __func__, n_non_eog);

    // if no non-EOG tokens are left -> reduce cur_p to single EOT token
    if (n_non_eog == 0) {
        cur_p->size = 1;
        cur_p->data[0].id = jarvis_token_eot_impl(*ctx->vocab);
        cur_p->data[0].logit = 1.0f;

        return;
    }

    // normalize probs
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= p_sum;

        LOG_DBG_CUR("%s: cur_p[%3zu] = { id: %6d, p: %.6f, logit: %6.3f }\n", __func__, i, cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }

    size_org = cur_p->size;
    p_sum = 0.0f;
    thold = 1.0/(n_non_eog + 1);

    cur_p->size = 0;

    LOG_DBG_CUR("%s: applying thold = %.3f\n", __func__, thold);

    for (size_t i = 0; i < size_org; ++i) {
        const bool is_eog = jarvis_token_is_eog_impl(*ctx->vocab, cur_p->data[i].id);

        if (cur_p->data[i].p < thold && !is_eog) {
            continue;
        }

        p_sum += cur_p->data[i].p;

        cur_p->data[cur_p->size++] = cur_p->data[i];
    }

    // normalize probs
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].p /= p_sum;

        LOG_DBG_CUR("%s: cur_p[%3zu] = { id: %6d, p: %.6f, logit: %6.3f }\n", __func__, i, cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }

#undef LOG_DBG_CUR
}

static struct jarvis_sampler * jarvis_sampler_infill_clone(const struct jarvis_sampler * smpl) {
    const auto * ctx = (const jarvis_sampler_infill *) smpl->ctx;
    return jarvis_sampler_init_infill_impl(*ctx->vocab);
}

static void jarvis_sampler_infill_free(struct jarvis_sampler * smpl) {
    delete (jarvis_sampler_infill *) smpl->ctx;
}

static struct jarvis_sampler_i jarvis_sampler_infill_i = {
    /* .name   = */ jarvis_sampler_infill_name,
    /* .accept = */ nullptr,
    /* .apply  = */ jarvis_sampler_infill_apply,
    /* .reset  = */ nullptr,
    /* .clone  = */ jarvis_sampler_infill_clone,
    /* .free   = */ jarvis_sampler_infill_free,
};

struct jarvis_sampler * jarvis_sampler_init_infill_impl(
        const struct jarvis_vocab & vocab) {
    return new jarvis_sampler {
        /* .iface = */ &jarvis_sampler_infill_i,
        /* .ctx   = */ new jarvis_sampler_infill {
            /* .vocab = */ &vocab,
            /* .buf0 = */ std::vector<char>(512),
            /* .buf1 = */ std::vector<char>(512),
        },
    };
}

// utils

uint32_t jarvis_sampler_get_seed(const struct jarvis_sampler * smpl) {
    if (smpl->iface == &jarvis_sampler_dist_i) {
        return ((const jarvis_sampler_dist *) smpl->ctx)->seed_cur;
    }

    if (smpl->iface == &jarvis_sampler_mirostat_i) {
        return ((const jarvis_sampler_mirostat *) smpl->ctx)->seed_cur;
    }

    if (smpl->iface == &jarvis_sampler_mirostat_v2_i) {
        return ((const jarvis_sampler_mirostat_v2 *) smpl->ctx)->seed_cur;
    }

    if (smpl->iface == &jarvis_sampler_chain_i) {
        const auto * ctx = (const jarvis_sampler_chain *) smpl->ctx;
        for (auto it = ctx->samplers.rbegin(); it != ctx->samplers.rend(); ++it) {
            const uint32_t seed = jarvis_sampler_get_seed(*it);
            if (seed != JARVIS_DEFAULT_SEED) {
                return seed;
            }
        }
    }

    return JARVIS_DEFAULT_SEED;
}

// perf

struct jarvis_perf_sampler_data jarvis_perf_sampler(const struct jarvis_sampler * chain) {
    struct jarvis_perf_sampler_data data = {};

    if (chain == nullptr || chain->iface != &jarvis_sampler_chain_i) {
        GGML_ABORT("%s: invalid sampler passed - requires a sampler created with jarvis_sampler_chain_init()\n", __func__);
    }

    const auto * ctx = (const struct jarvis_sampler_chain *) chain->ctx;

    data.t_sample_ms = 1e-3 * ctx->t_sample_us;
    data.n_sample    = std::max(0, ctx->n_sample);

    return data;
}

void jarvis_perf_sampler_print(const struct jarvis_sampler * chain) {
    const auto data = jarvis_perf_sampler(chain);

    JARVIS_LOG_INFO("%s:    sampling time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_sample_ms, data.n_sample, data.t_sample_ms / data.n_sample, 1e3 / data.t_sample_ms * data.n_sample);
}

void jarvis_perf_sampler_reset(struct jarvis_sampler * chain) {
    if (chain == nullptr || chain->iface != &jarvis_sampler_chain_i) {
        GGML_ABORT("%s: invalid sampler passed - requires a sampler created with jarvis_sampler_chain_init()\n", __func__);
    }

    auto * ctx = (struct jarvis_sampler_chain *) chain->ctx;

    ctx->t_sample_us = ctx->n_sample = 0;
}
