#include "llama-sampling.h"

#include "llama-vocab.h"
#include "llama-grammar.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <cfloat>
#include <numeric>
#include <unordered_map>

static void llama_log_softmax(float * array, size_t size) {
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

llama_sampling::llama_sampling(const struct llama_vocab & vocab) : vocab(vocab) {
}

llama_sampling::~llama_sampling() {
    if (grammar) {
        llama_grammar_free_impl(grammar);
    }
}

struct llama_sampling * llama_sampling_init_impl(const struct llama_vocab & vocab, struct llama_sampling_params params) {
    auto * result = new llama_sampling(vocab);

    result->params = params;

    result->prev = ring_buffer<llama_token>(params.n_prev);

    for (int i = 0; i < params.n_samplers; ++i) {
        result->samplers.push_back(params.samplers[i]);
    }

    llama_sampling_set_rng_seed_impl(*result, params.seed);

    return result;
}

void llama_sampling_free_impl(struct llama_sampling * sampling) {
    if (sampling == nullptr) {
        return;
    }

    delete sampling;
}

struct llama_sampling * llama_sampling_cp_impl(const struct llama_sampling & smpl) {
    auto * result = new llama_sampling(smpl.vocab);

    result->params = smpl.params;

    result->grammar_str  = smpl.grammar_str;
    result->grammar_root = smpl.grammar_root;

    result->logit_bias = smpl.logit_bias;

    if (smpl.grammar) {
        result->grammar = llama_grammar_cp_impl(*smpl.grammar);
    }

    result->rng  = smpl.rng;
    result->prev = smpl.prev;

    return result;
}

void llama_sampling_reset_impl(struct llama_sampling & smpl) {
    if (smpl.grammar) {
        llama_grammar_free_impl(smpl.grammar);
        smpl.grammar = nullptr;
    }

    if (!smpl.grammar_str.empty()) {
        smpl.grammar = llama_grammar_init_impl(&smpl.vocab, smpl.grammar_str.data(), smpl.grammar_root.data());
    }

    smpl.prev.clear();
}

void llama_sampling_set_rng_seed_impl(struct llama_sampling & smpl, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = time(NULL);
    }

    smpl.rng.seed(seed);
}

void llama_sampling_set_grammar_impl(struct llama_sampling & smpl, const char * grammar_str, const char * grammar_root) {
    if (smpl.grammar) {
        llama_grammar_free_impl(smpl.grammar);
        smpl.grammar = nullptr;
    }

    if (grammar_str != nullptr && grammar_str[0] != '\0') {
        smpl.grammar_str  = grammar_str;
        smpl.grammar_root = grammar_root;

        smpl.grammar = llama_grammar_init_impl(&smpl.vocab, grammar_str, grammar_root);
    } else {
        smpl.grammar_str.clear();
        smpl.grammar_root.clear();
    }
}

void llama_sampling_set_logit_bias_impl(struct llama_sampling & smpl, int32_t n_logit_bias, const llama_logit_bias * logit_bias) {
    smpl.logit_bias.clear();
    smpl.logit_bias.reserve(n_logit_bias);

    for (int32_t i = 0; i < n_logit_bias; ++i) {
        smpl.logit_bias.push_back(logit_bias[i]);
    }
}

void llama_sampling_softmax_impl(llama_token_data_array * candidates) {
    GGML_ASSERT(candidates->size > 0);

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;

    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }

    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }
}

void llama_sampling_top_k_impl(llama_token_data_array * candidates, int32_t k, size_t min_keep) {
    // TODO: move bucket sort to separate function so that top_p/tail_free/typical/softmax first is equally fast
    // if (k >= (int32_t)candidates->size) {
    //     return;
    // }

    if (k <= 0) {
        k = candidates->size;
    }

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit;
        };
        if (k <= 128) {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        } else {
            constexpr int   nbuckets     = 128;
            constexpr float bucket_low   = -10.0f;
            constexpr float bucket_high  =  10.0f;
            constexpr float bucket_scale = nbuckets/(bucket_high - bucket_low);
            constexpr float bucket_inter = -bucket_low * bucket_scale;

            std::vector<int> bucket_idx(candidates->size);
            std::vector<int> histo(nbuckets, 0);

            for (int i = 0; i < (int)candidates->size; ++i) {
                const float val = candidates->data[i].logit;
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
            std::vector<llama_token_data> tmp_tokens(nhave);
            auto * ptr = tmp_tokens.data();
            std::vector<llama_token_data*> bucket_ptrs;
            bucket_ptrs.reserve(nbuckets - ib);
            for (int j = nbuckets - 1; j >= ib; --j) {
                bucket_ptrs.push_back(ptr);
                ptr += histo[j];
            }
            for (int i = 0; i < (int)candidates->size; ++i) {
                int j = bucket_idx[i];
                if (j >= ib) {
                    *bucket_ptrs[nbuckets-1-j]++ = candidates->data[i];
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

            std::memcpy(candidates->data, tmp_tokens.data(), k*sizeof(llama_token_data));

        }
        candidates->sorted = true;
    }
    candidates->size = k;
}

void llama_sampling_top_p_impl(llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    llama_sampling_softmax_impl(candidates);

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;
}

void llama_sampling_min_p_impl(llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p <= 0.0f || !candidates->size) {
        return;
    }

    bool min_p_applied = false;

    // if the candidates aren't sorted, try the unsorted implementation first
    if (!candidates->sorted) {
        std::vector<llama_token_data> filtered_tokens;

        float max_logit = -FLT_MAX;
        for (size_t i = 0; i < candidates->size; ++i) {
            max_logit = std::max(max_logit, candidates->data[i].logit);
        }
        const float min_logit = max_logit + logf(p); // min logit for p_i >= p * p_max

        for (size_t i = 0; i < candidates->size; ++i) {
            if (candidates->data[i].logit >= min_logit) {
                filtered_tokens.push_back(candidates->data[i]);
            }
        }

        // if we have enough values the operation was a success
        if (filtered_tokens.size() >= min_keep) {
            memcpy(candidates->data, filtered_tokens.data(), filtered_tokens.size()*sizeof(llama_token_data));
            candidates->size = filtered_tokens.size();
            min_p_applied = true;
        }
    }

    // if the candidates are sorted or the unsorted implementation failed, use this implementation
    if (!min_p_applied) {
        // Sort the logits in descending order
        if (!candidates->sorted) {
            std::sort(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
                return a.logit > b.logit;
            });
            candidates->sorted = true;
        }

        const float min_logit = candidates->data[0].logit + logf(p); // min logit for p_i >= p * p_max
        size_t i = 1; // first token always matches

        for (; i < candidates->size; ++i) {
            if (candidates->data[i].logit < min_logit && i >= min_keep) {
                break; // prob too small
            }
        }

        // Resize the output vector to keep only the matching tokens
        candidates->size = i;
    }
}

void llama_sampling_tail_free_impl(llama_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    llama_sampling_softmax_impl(candidates);

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
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
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;
}

void llama_sampling_typical_impl(llama_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_sampling_softmax_impl(candidates);

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();
    candidates->sorted = false;
}

void llama_sampling_entropy_impl(llama_token_data_array * candidates, float min_temp, float max_temp, float exponent_val) {
    // no need to do anything if there is only one (or zero) candidates
    if(candidates->size <= 1) {
        return;
    }

    // Calculate maximum possible entropy
    float max_entropy = -logf(1.0f / candidates->size);

    llama_sampling_softmax_impl(candidates);

    // Calculate entropy of the softmax probabilities
    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float prob = candidates->data[i].p;
        if (prob > 0.0f) { // Ensure no log(0)
            entropy -= prob * logf(prob);
        }
    }

    // Normalize the entropy (max_entropy cannot be 0 here because we checked candidates->size != 1 above)
    float normalized_entropy = entropy / max_entropy;

    // Map the normalized entropy to the desired temperature range using the power function
    float dyn_temp = min_temp + (max_temp - min_temp) * powf(normalized_entropy, exponent_val);

#ifdef DEBUG
    LLAMA_LOG_INFO("Your text maxtemp value is: %f\n", max_temp);
    LLAMA_LOG_INFO("Entropy: %f\n", entropy);
    LLAMA_LOG_INFO("Max Possible Entropy: %f\n", max_entropy);
    LLAMA_LOG_INFO("Normalized Entropy: %f\n", normalized_entropy);
    LLAMA_LOG_INFO("Exponent: %f\n", exponent_val);
    LLAMA_LOG_INFO("Dynamic Temperature (dyn_temp): %f\n", dyn_temp);
#endif

    // Apply the dynamically calculated temperature scaling
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].logit /= dyn_temp;
    }

    // Re-compute softmax probabilities after scaling logits with dynamic temperature
    const double max_l_double = candidates->data[0].logit;

    double cum_sum_double = 0.0;
    for (size_t i = 0; i < candidates->size; ++i) {
        double p = exp(candidates->data[i].logit - max_l_double);
        candidates->data[i].p = p; // Store the scaled probability
        cum_sum_double += p;
    }

    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum_double; // Re-normalize the probabilities
    }

#ifdef DEBUG
    // Print the updated top 25 probabilities after temperature scaling
    LLAMA_LOG_INFO("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):\n");
    for (size_t i = 0; i < 25 && i < candidates->size; ++i) {
        LLAMA_LOG_INFO("Token %zu: %f%%\n", i + 1, candidates->data[i].p * 100.0f);
    }
#endif
}

void llama_sampling_temp_impl(llama_token_data_array * candidates, float temp) {
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].logit /= temp;
    }
}

void llama_sampling_grammar_impl(llama_token_data_array * candidates, const struct llama_grammar & grammar) {
    llama_grammar_apply_impl(grammar, candidates);
}

void llama_sampling_penalties_impl(
       llama_token_data_array * candidates,
        const llama_token_cnt & token_count,
                        float   penalty_repeat,
                        float   penalty_freq,
                        float   penalty_present) {
    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        const auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty_repeat;
        } else {
            candidates->data[i].logit /= penalty_repeat;
        }

        candidates->data[i].logit -= float(count) * penalty_freq + float(count > 0) * penalty_present;
    }

    candidates->sorted = false;
}

llama_token llama_sampling_sample_mirostat_impl(struct llama_token_data_array * candidates, std::mt19937 & rng, float tau, float eta, int32_t m, int32_t n_vocab, float & mu) {
    llama_sampling_softmax_impl(candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, mu)) / (1 - powf(n_vocab, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_sampling_top_k_impl(candidates, int(k), 1);
    llama_token X = llama_sampling_sample_dist_impl(candidates, rng);

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    mu = mu - eta * e;

    return X;
}

llama_token llama_sampling_sample_mirostat_v2_impl(struct llama_token_data_array * candidates, std::mt19937 & rng, float tau, float eta, float & mu) {
    llama_sampling_softmax_impl(candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return -log2f(candidate.p) > mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    // Normalize the probabilities of the remaining words
    llama_sampling_softmax_impl(candidates);

    // Sample the next word X from the remaining words
    llama_token X = llama_sampling_sample_dist_impl(candidates, rng);

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));

    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    mu = mu - eta * e;

    return X;
}

llama_token llama_sampling_sample_greedy_impl(llama_token_data_array * candidates) {
    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;

    return result;
}

llama_token llama_sampling_sample_dist_impl(struct llama_token_data_array * candidates, std::mt19937 & rng) {
    llama_sampling_softmax_impl(candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);

    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());

    const int idx = dist(rng);
    llama_token result = candidates->data[idx].id;

    return result;
}

void llama_sampling_accept_impl(struct llama_sampling & smpl, llama_token token, bool apply_grammar) {
    smpl.prev.push_back(token);

    if (apply_grammar && smpl.grammar) {
        llama_grammar_accept_impl(*smpl.grammar, token);
    }
}

llama_token llama_sampling_prev_impl(const struct llama_sampling & smpl, int ith) {
    if (ith < 0 || ith >= (int) smpl.prev.size()) {
        return LLAMA_TOKEN_NULL;
    }

    return smpl.prev.rat(ith);
}

int llama_sampling_n_prev_impl(const struct llama_sampling & smpl) {
    return smpl.prev.size();
}

//
// sampling v2
//

// constraints

// top-k

struct llama_constraint_context_top_k {
    int32_t k;
    size_t min_keep;
};

static struct llama_constraint_i llama_constraint_top_k_i = {
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_top_k *) cnstr->ctx;
        llama_sampling_top_k_impl(candidates, ctx->k, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_top_k;
        const auto * ctx_src = (const llama_constraint_context_top_k *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_top_k *) cnstr->ctx;
        *ctx_dst = *ctx_src;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            delete (llama_constraint_context_top_k *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_top_k_impl(int32_t k, size_t min_keep) {
    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_top_k_i;
    result->ctx = new llama_constraint_context_top_k;

    auto * ctx = (llama_constraint_context_top_k *) result->ctx;

    *ctx = {
        /*.k        =*/ k,
        /*.min_keep =*/ min_keep,
    };

    return result;
}

// top-p

struct llama_constraint_context_top_p {
    float p;
    size_t min_keep;
};

static struct llama_constraint_i llama_constraint_top_p_i = {
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_top_p *) cnstr->ctx;
        llama_sampling_top_p_impl(candidates, ctx->p, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_top_p;
        const auto * ctx_src = (const llama_constraint_context_top_p *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_top_p *) cnstr->ctx;
        *ctx_dst = *ctx_src;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            delete (llama_constraint_context_top_p *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_top_p_impl(float p, size_t min_keep) {
    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_top_p_i;
    result->ctx = new llama_constraint_context_top_p;

    auto * ctx = (llama_constraint_context_top_p *) result->ctx;

    *ctx = {
        /*.p        =*/ p,
        /*.min_keep =*/ min_keep,
    };

    return result;
}

// min-p

struct llama_constraint_context_min_p {
    float p;
    size_t min_keep;
};

static struct llama_constraint_i llama_constraint_min_p_i = {
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_min_p *) cnstr->ctx;
        llama_sampling_min_p_impl(candidates, ctx->p, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_min_p;
        const auto * ctx_src = (const llama_constraint_context_min_p *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_min_p *) cnstr->ctx;
        *ctx_dst = *ctx_src;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            delete (llama_constraint_context_min_p *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_min_p_impl(float p, size_t min_keep) {
    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_min_p_i;
    result->ctx = new llama_constraint_context_min_p;

    auto * ctx = (llama_constraint_context_min_p *) result->ctx;

    *ctx = {
        /*.p        =*/ p,
        /*.min_keep =*/ min_keep,
    };

    return result;
}

// tail-free

struct llama_constraint_context_tail_free {
    float z;
    size_t min_keep;
};

static struct llama_constraint_i llama_constraint_tail_free_i = {
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_tail_free *) cnstr->ctx;
        llama_sampling_tail_free_impl(candidates, ctx->z, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_tail_free;
        const auto * ctx_src = (const llama_constraint_context_tail_free *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_tail_free *) cnstr->ctx;
        *ctx_dst = *ctx_src;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            delete (llama_constraint_context_tail_free *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_tail_free_impl(float z, size_t min_keep) {
    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_tail_free_i;
    result->ctx = new llama_constraint_context_tail_free;

    auto * ctx = (llama_constraint_context_tail_free *) result->ctx;

    *ctx = {
        /*.z        =*/ z,
        /*.min_keep =*/ min_keep,
    };

    return result;
}

// typical

struct llama_constraint_context_typical {
    float p;
    size_t min_keep;
};

static struct llama_constraint_i llama_constraint_typical_i = {
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_typical *) cnstr->ctx;
        llama_sampling_typical_impl(candidates, ctx->p, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_typical;
        const auto * ctx_src = (const llama_constraint_context_typical *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_typical *) cnstr->ctx;
        *ctx_dst = *ctx_src;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            delete (llama_constraint_context_typical *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_typical_impl(float p, size_t min_keep) {
    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_typical_i;
    result->ctx = new llama_constraint_context_typical;

    auto * ctx = (llama_constraint_context_typical *) result->ctx;

    *ctx = {
        /*.p        =*/ p,
        /*.min_keep =*/ min_keep,
    };

    return result;
}

// temp

struct llama_constraint_context_temp {
    float temp;
};

static struct llama_constraint_i llama_constraint_temp_i = {
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_temp *) cnstr->ctx;
        llama_sampling_temp_impl(candidates, ctx->temp);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_temp;
        const auto * ctx_src = (const llama_constraint_context_temp *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_temp *) cnstr->ctx;
        *ctx_dst = *ctx_src;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            delete (llama_constraint_context_temp *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_temp_impl(float temp) {
    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_temp_i;
    result->ctx = new llama_constraint_context_temp;

    auto * ctx = (llama_constraint_context_temp *) result->ctx;

    *ctx = {
        /*.temp =*/ temp,
    };

    return result;
}

// temp-ext

struct llama_constraint_context_temp_ext {
    float temp;
    float delta;
    float exponent;
};

static struct llama_constraint_i llama_constraint_temp_ext_i = {
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_temp_ext *) cnstr->ctx;
        if (ctx->delta > 0) {
            const float temp_min = std::max(0.0f, ctx->temp - ctx->delta);
            const float temp_max = ctx->temp + ctx->delta;

            llama_sampling_entropy_impl(candidates, temp_min, temp_max, ctx->exponent);
        } else {
            llama_sampling_temp_impl(candidates, ctx->temp);
        }
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_temp_ext;
        const auto * ctx_src = (const llama_constraint_context_temp_ext *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_temp_ext *) cnstr->ctx;
        *ctx_dst = *ctx_src;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            delete (llama_constraint_context_temp_ext *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_temp_ext_impl(float temp, float delta, float exponent) {
    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_temp_ext_i;
    result->ctx = new llama_constraint_context_temp_ext;

    auto * ctx = (llama_constraint_context_temp_ext *) result->ctx;

    *ctx = {
        /*.temp     =*/ temp,
        /*.delta    =*/ delta,
        /*.exponent =*/ exponent,
    };

    return result;
}

// grammar

struct llama_constraint_context_grammar {
    std::string grammar_str;
    std::string grammar_root;

    struct llama_grammar * grammar;
};

static struct llama_constraint_i llama_constraint_grammar_i = {
    /* .accept = */ [](struct llama_constraint * cnstr, llama_token token) {
        auto * ctx = (llama_constraint_context_grammar *) cnstr->ctx;
        if (ctx->grammar) {
            llama_grammar_accept_impl(*ctx->grammar, token);
        }
    },
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_grammar *) cnstr->ctx;
        if (ctx->grammar) {
            llama_sampling_grammar_impl(candidates, *ctx->grammar);
        }
    },
    /* .reset  = */ [](struct llama_constraint * cnstr) {
        auto * ctx = (llama_constraint_context_grammar *) cnstr->ctx;
        if (ctx->grammar) {
            llama_grammar_free_impl(ctx->grammar);
            ctx->grammar = nullptr;
        }

        if (!ctx->grammar_str.empty()) {
            ctx->grammar = llama_grammar_init_impl(nullptr, ctx->grammar_str.c_str(), ctx->grammar_root.c_str());
        }
    },
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_grammar;
        const auto * ctx_src = (const llama_constraint_context_grammar *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_grammar *) cnstr->ctx;

        *ctx_dst = *ctx_src;

        if (ctx_src->grammar) {
            ctx_dst->grammar = llama_grammar_cp_impl(*ctx_src->grammar);
        } else {
            ctx_dst->grammar = nullptr;
        }
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            {
                auto * ctx = (llama_constraint_context_grammar *) cnstr->ctx;
                llama_grammar_free_impl(ctx->grammar);
            }

            delete (llama_constraint_context_grammar *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_grammar_impl(const struct llama_vocab & vocab, const char * grammar_str, const char * grammar_root) {
    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_grammar_i;
    result->ctx = new llama_constraint_context_grammar;

    auto * ctx = (llama_constraint_context_grammar *) result->ctx;

    if (grammar_str != nullptr && grammar_str[0] != '\0') {
        *ctx = {
            /*.grammar_str  = */ grammar_str,
            /*.grammar_root = */ grammar_root,
            /*.grammar      = */ llama_grammar_init_impl(&vocab, grammar_str, grammar_root),
        };
    } else {
        *ctx = {
            /*.grammar_str  = */ {},
            /*.grammar_root = */ {},
            /*.grammar      = */ nullptr,
        };
    }

    return result;
}

// penalties

struct llama_constraint_context_penalties {
    const struct llama_vocab * vocab;

    int32_t penalty_last_n;
    float   penalty_repeat;
    float   penalty_freq;
    float   penalty_present;

    bool    penalize_nl;
    bool    ignore_eos;

    ring_buffer<llama_token> prev;
};

static struct llama_constraint_i llama_constraint_penalties_i = {
    /* .accept = */ [](struct llama_constraint * cnstr, llama_token token) {
        auto * ctx = (llama_constraint_context_penalties *) cnstr->ctx;
        ctx->prev.push_back(token);
    },
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_penalties *) cnstr->ctx;

        GGML_ASSERT(candidates->size == ctx->vocab->n_vocab && candidates->sorted == false && "the 'penalties' constraint must be applied on the full vocabulary");

        if (ctx->ignore_eos) {
            candidates->data[ctx->vocab->special_eos_id].logit = -INFINITY;
        }

        if ((ctx->penalty_last_n == 0) ||
            (ctx->penalty_repeat == 1.0f && ctx->penalty_freq == 0.0f && ctx->penalty_present == 0.0f)) {
            return;
        }

        const float nl_logit = !ctx->penalize_nl ? candidates->data[ctx->vocab->linefeed_id].logit : -INFINITY;

        // Create a frequency map to count occurrences of each token in last_tokens
        // TODO: optimize this by maintaining the token count in the constraint context
        llama_token_cnt token_count;
        for (int i = 0; i < ctx->penalty_last_n; ++i) {
            token_count[ctx->prev.rat(i)]++;
        }

        llama_sampling_penalties_impl(candidates, token_count, ctx->penalty_repeat, ctx->penalty_freq, ctx->penalty_present);

        if (!ctx->penalize_nl) {
            // restore the logit of the newline token if it was penalized
            candidates->data[ctx->vocab->linefeed_id].logit = nl_logit;
        }
    },
    /* .reset  = */ [](struct llama_constraint * cnstr) {
        auto * ctx = (llama_constraint_context_penalties *) cnstr->ctx;
        ctx->prev.clear();
    },
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_penalties;
        const auto * ctx_src = (const llama_constraint_context_penalties *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_penalties *) cnstr->ctx;

        *ctx_dst = *ctx_src;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            delete (llama_constraint_context_penalties *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_penalties_impl(const struct llama_vocab & vocab, int32_t penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present, bool penalize_nl, bool ignore_eos) {
    GGML_ASSERT(penalize_nl || vocab.linefeed_id != LLAMA_TOKEN_NULL);
    GGML_ASSERT(!ignore_eos || vocab.special_eos_id != LLAMA_TOKEN_NULL);

    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_penalties_i;
    result->ctx = new llama_constraint_context_penalties;

    auto * ctx = (llama_constraint_context_penalties *) result->ctx;

    *ctx = {
        /*.vocab            = */ &vocab,
        /*.penalty_last_n   = */ penalty_last_n,
        /*.penalty_repeat   = */ penalty_repeat,
        /*.penalty_freq     = */ penalty_freq,
        /*.penalty_present  = */ penalty_present,
        /*.penalize_nl      = */ penalize_nl,
        /*.ignore_eos       = */ ignore_eos,
        /*.prev             = */ {},
    };

    return result;
}

// logit-bias

struct llama_constraint_context_logit_bias {
    const struct llama_vocab * vocab;

    std::vector<llama_logit_bias> logit_bias;
};

static struct llama_constraint_i llama_constraint_logit_bias_i = {
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * candidates) {
        auto * ctx = (llama_constraint_context_logit_bias *) cnstr->ctx;

        GGML_ASSERT(candidates->size == ctx->vocab->n_vocab && candidates->sorted == false && "the 'logit_bias' constraint must be applied on the full vocabulary");

        for (const auto & lb : ctx->logit_bias) {
            candidates->data[lb.token].logit += lb.bias;
        }
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](struct llama_constraint * cnstr, const struct llama_constraint * cnstr_src) {
        cnstr->ctx = new llama_constraint_context_logit_bias;
        const auto * ctx_src = (const llama_constraint_context_logit_bias *) cnstr_src->ctx;
              auto * ctx_dst = (      llama_constraint_context_logit_bias *) cnstr->ctx;

        *ctx_dst = *ctx_src;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        if (cnstr->ctx) {
            delete (llama_constraint_context_logit_bias *) cnstr->ctx;
        }
        delete cnstr;
    }
};

struct llama_constraint * llama_constraint_init_logit_bias_impl(
        const struct llama_vocab & vocab,
                         int32_t   n_logit_bias,
          const llama_logit_bias * logit_bias) {
    struct llama_constraint * result = new llama_constraint;

    result->iface = &llama_constraint_logit_bias_i;
    result->ctx = new llama_constraint_context_logit_bias;

    auto * ctx = (llama_constraint_context_logit_bias *) result->ctx;

    *ctx = {
        /*.vocab      = */ &vocab,
        /*.logit_bias = */ std::vector<llama_logit_bias>(logit_bias, logit_bias + n_logit_bias),
    };

    return result;
}

////////////////////////////////////////

void llama_constraint_free_impl(struct llama_constraint * cnstr) {
    if (cnstr->iface->free && cnstr) {
        cnstr->iface->free(cnstr);
    }
}

void llama_constraint_accept_impl(struct llama_constraint & cnstr, llama_token token) {
    if (cnstr.iface->accept) {
        cnstr.iface->accept(&cnstr, token);
    }
}

void llama_constraint_apply_impl(struct llama_constraint & cnstr, struct llama_token_data_array * candidates) {
    GGML_ASSERT(cnstr.iface->apply);
    cnstr.iface->apply(&cnstr, candidates);
}

void llama_constraint_reset_impl(struct llama_constraint & cnstr) {
    if (cnstr.iface->reset) {
        cnstr.iface->reset(&cnstr);
    }
}

// samplers

struct llama_sampler * llama_sampler_init_impl(const struct llama_vocab & vocab, struct llama_sampler_params params) {
    auto * result = new llama_sampler;

    result->params = params;
    result->vocab = &vocab;

    result->rng.seed(params.seed);

    return result;
}

void llama_sampler_free_impl(struct llama_sampler * smpl) {
    if (smpl == nullptr) {
        return;
    }

    for (auto * cnstr : smpl->constraints) {
        llama_constraint_free_impl(cnstr);
    }

    delete smpl;
}

struct llama_sampler * llama_sampler_cp_impl(const struct llama_sampler & smpl) {
    auto * result = new llama_sampler;

    *result = smpl;

    // copy the constraints objects
    result->constraints.clear();
    for (const auto & cnstr : smpl.constraints) {
        result->constraints.push_back(new llama_constraint);
        result->constraints.back()->iface = cnstr->iface;

        if (cnstr->ctx) {
            GGML_ASSERT(cnstr->iface->copy);
            result->constraints.back()->iface->copy(result->constraints.back(), cnstr);
        }
    }

    return result;
}

void llama_sampler_reset_impl(struct llama_sampler & smpl) {
    smpl.prev.clear();

    for (auto * cnstr : smpl.constraints) {
        llama_constraint_reset_impl(*cnstr);
    }

    // TODO: should we reset the timings?
}

void llama_sampler_add_constraint_impl(struct llama_sampler & smpl, struct llama_constraint * cnstr) {
    smpl.constraints.push_back(cnstr);
}

void llama_sampler_accept_impl(struct llama_sampler & smpl, llama_token token) {
    smpl.prev.push_back(token);

    for (auto * cnstr : smpl.constraints) {
        llama_constraint_accept_impl(*cnstr, token);
    }
}

void llama_sampler_apply_impl(struct llama_sampler & smpl, struct llama_token_data_array * candidates) {
    for (auto * cnstr : smpl.constraints) {
        llama_constraint_apply_impl(*cnstr, candidates);
    }
}

llama_token llama_sampler_prev_impl(const struct llama_sampler & smpl, int ith) {
    if (ith < 0 || ith >= (int) smpl.prev.size()) {
        return LLAMA_TOKEN_NULL;
    }

    return smpl.prev.rat(ith);
}

int llama_sampler_n_prev_impl(const struct llama_sampler & smpl) {
    return smpl.prev.size();
}

