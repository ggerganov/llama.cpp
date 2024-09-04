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

static void llama_constraint_softmax_impl(llama_token_data_array * cur_p) {
    GGML_ASSERT(cur_p->size > 0);

    // Sort the logits in descending order
    if (!cur_p->sorted) {
        std::sort(cur_p->data, cur_p->data + cur_p->size, [](const llama_token_data & a, const llama_token_data & b) {
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

static void llama_constraint_top_k_impl(llama_token_data_array * cur_p, int32_t k, size_t min_keep) {
    // TODO: move bucket sort to separate function so that top_p/tail_free/typical/softmax first is equally fast
    // if (k >= (int32_t)cur_p->size) {
    //     return;
    // }

    if (k <= 0) {
        k = cur_p->size;
    }

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) cur_p->size);

    // Sort scores in descending order
    if (!cur_p->sorted) {
        auto comp = [](const llama_token_data & a, const llama_token_data & b) {
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
            std::vector<llama_token_data> tmp_tokens(nhave);
            auto * ptr = tmp_tokens.data();
            std::vector<llama_token_data*> bucket_ptrs;
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

            std::memcpy(cur_p->data, tmp_tokens.data(), k*sizeof(llama_token_data));

        }
        cur_p->sorted = true;
    }
    cur_p->size = k;
}

static void llama_constraint_top_p_impl(llama_token_data_array * cur_p, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    llama_constraint_softmax_impl(cur_p);

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = cur_p->size;

    for (size_t i = 0; i < cur_p->size; ++i) {
        cum_sum += cur_p->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    cur_p->size = last_idx;
}

static void llama_constraint_min_p_impl(llama_token_data_array * cur_p, float p, size_t min_keep) {
    if (p <= 0.0f || !cur_p->size) {
        return;
    }

    bool min_p_applied = false;

    // if the cur_p aren't sorted, try the unsorted implementation first
    if (!cur_p->sorted) {
        std::vector<llama_token_data> filtered_tokens;

        float max_logit = -FLT_MAX;
        for (size_t i = 0; i < cur_p->size; ++i) {
            max_logit = std::max(max_logit, cur_p->data[i].logit);
        }
        const float min_logit = max_logit + logf(p); // min logit for p_i >= p * p_max

        for (size_t i = 0; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit >= min_logit) {
                filtered_tokens.push_back(cur_p->data[i]);
            }
        }

        // if we have enough values the operation was a success
        if (filtered_tokens.size() >= min_keep) {
            memcpy(cur_p->data, filtered_tokens.data(), filtered_tokens.size()*sizeof(llama_token_data));
            cur_p->size = filtered_tokens.size();
            min_p_applied = true;
        }
    }

    // if the cur_p are sorted or the unsorted implementation failed, use this implementation
    if (!min_p_applied) {
        // Sort the logits in descending order
        if (!cur_p->sorted) {
            std::sort(cur_p->data, cur_p->data + cur_p->size, [](const llama_token_data & a, const llama_token_data & b) {
                return a.logit > b.logit;
            });
            cur_p->sorted = true;
        }

        const float min_logit = cur_p->data[0].logit + logf(p); // min logit for p_i >= p * p_max
        size_t i = 1; // first token always matches

        for (; i < cur_p->size; ++i) {
            if (cur_p->data[i].logit < min_logit && i >= min_keep) {
                break; // prob too small
            }
        }

        // Resize the output vector to keep only the matching tokens
        cur_p->size = i;
    }
}

static void llama_constraint_tail_free_impl(llama_token_data_array * cur_p, float z, size_t min_keep) {
    if (z >= 1.0f || cur_p->size <= 2) {
        return;
    }

    llama_constraint_softmax_impl(cur_p);

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
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    cur_p->size = last_idx;
}

static void llama_constraint_typical_impl(llama_token_data_array * cur_p, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_constraint_softmax_impl(cur_p);

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
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> cur_p_new;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        cur_p_new.push_back(cur_p->data[idx]);
    }

    // Replace the data in cur_p with the cur_p_new data
    std::copy(cur_p_new.begin(), cur_p_new.end(), cur_p->data);
    cur_p->size = cur_p_new.size();
    cur_p->sorted = false;
}

static void llama_constraint_entropy_impl(llama_token_data_array * cur_p, float min_temp, float max_temp, float exponent_val) {
    // no need to do anything if there is only one (or zero) candidates
    if (cur_p->size <= 1) {
        return;
    }

    // Calculate maximum possible entropy
    float max_entropy = -logf(1.0f / cur_p->size);

    llama_constraint_softmax_impl(cur_p);

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
    LLAMA_LOG_INFO("Your text maxtemp value is: %f\n", max_temp);
    LLAMA_LOG_INFO("Entropy: %f\n", entropy);
    LLAMA_LOG_INFO("Max Possible Entropy: %f\n", max_entropy);
    LLAMA_LOG_INFO("Normalized Entropy: %f\n", normalized_entropy);
    LLAMA_LOG_INFO("Exponent: %f\n", exponent_val);
    LLAMA_LOG_INFO("Dynamic Temperature (dyn_temp): %f\n", dyn_temp);
#endif

    // Apply the dynamically calculated temperature scaling
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= dyn_temp;
    }

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
    LLAMA_LOG_INFO("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):\n");
    for (size_t i = 0; i < 25 && i < cur_p->size; ++i) {
        LLAMA_LOG_INFO("Token %zu: %f%%\n", i + 1, cur_p->data[i].p * 100.0f);
    }
#endif
}

static void llama_constraint_temp_impl(llama_token_data_array * cur_p, float temp) {
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit /= temp;
    }
}

static void llama_constraint_grammar_impl(llama_token_data_array * cur_p, const struct llama_grammar & grammar) {
    llama_grammar_apply_impl(grammar, cur_p);
}

void llama_constraint_penalties_impl(
       llama_token_data_array * cur_p,
        const llama_token_cnt & token_count,
                        float   penalty_repeat,
                        float   penalty_freq,
                        float   penalty_present) {
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
            cur_p->data[i].logit *= penalty_repeat;
        } else {
            cur_p->data[i].logit /= penalty_repeat;
        }

        cur_p->data[i].logit -= float(count) * penalty_freq + float(count > 0) * penalty_present;
    }

    cur_p->sorted = false;
}

//
// constraints
//

// softmax

static struct llama_constraint_i llama_constraint_softmax_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "softmax"; },
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * /*cnstr*/, llama_token_data_array * cur_p) {
        llama_constraint_softmax_impl(cur_p);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ nullptr,
    /* .free   = */ nullptr,
};

struct llama_constraint * llama_constraint_init_softmax_impl() {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_softmax_i,
        /* .ctx   = */ nullptr,
    };
}

// top-k

struct llama_constraint_context_top_k {
    const int32_t k;
    const size_t  min_keep;
};

static struct llama_constraint_i llama_constraint_top_k_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "top-k"; },
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        const auto * ctx = (llama_constraint_context_top_k *) cnstr->ctx;
        llama_constraint_top_k_impl(cur_p, ctx->k, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx = (const llama_constraint_context_top_k *) cnstr->ctx;
        return llama_constraint_init_top_k_impl(ctx->k, ctx->min_keep);
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_top_k *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_top_k_impl(int32_t k, size_t min_keep) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_top_k_i,
        /* .ctx   = */ new llama_constraint_context_top_k {
            /*.k        =*/ k,
            /*.min_keep =*/ min_keep,
        },
    };
}

// top-p

struct llama_constraint_context_top_p {
    const float  p;
    const size_t min_keep;
};

static struct llama_constraint_i llama_constraint_top_p_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "top-p"; },
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        const auto * ctx = (llama_constraint_context_top_p *) cnstr->ctx;
        llama_constraint_top_p_impl(cur_p, ctx->p, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx = (const llama_constraint_context_top_p *) cnstr->ctx;
        return llama_constraint_init_top_p_impl(ctx->p, ctx->min_keep);
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_top_p *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_top_p_impl(float p, size_t min_keep) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_top_p_i,
        /* .ctx   = */ new llama_constraint_context_top_p {
            /*.p        =*/ p,
            /*.min_keep =*/ min_keep,
        },
    };
}

// min-p

struct llama_constraint_context_min_p {
    const float  p;
    const size_t min_keep;
};

static struct llama_constraint_i llama_constraint_min_p_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "min-p"; },
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        const auto * ctx = (llama_constraint_context_min_p *) cnstr->ctx;
        llama_constraint_min_p_impl(cur_p, ctx->p, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx = (const llama_constraint_context_min_p *) cnstr->ctx;
        return llama_constraint_init_min_p_impl(ctx->p, ctx->min_keep);
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_min_p *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_min_p_impl(float p, size_t min_keep) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_min_p_i,
        /* .ctx   = */ new llama_constraint_context_min_p {
            /*.p        =*/ p,
            /*.min_keep =*/ min_keep,
        },
    };
}

// tail-free

struct llama_constraint_context_tail_free {
    const float  z;
    const size_t min_keep;
};

static struct llama_constraint_i llama_constraint_tail_free_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "tail-free"; },
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        const auto * ctx = (llama_constraint_context_tail_free *) cnstr->ctx;
        llama_constraint_tail_free_impl(cur_p, ctx->z, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx = (const llama_constraint_context_tail_free *) cnstr->ctx;
        return llama_constraint_init_tail_free_impl(ctx->z, ctx->min_keep);
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_tail_free *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_tail_free_impl(float z, size_t min_keep) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_tail_free_i,
        /* .ctx   = */ new llama_constraint_context_tail_free {
            /*.z        =*/ z,
            /*.min_keep =*/ min_keep,
        },
    };
}

// typical

struct llama_constraint_context_typical {
    const float  p;
    const size_t min_keep;
};

static struct llama_constraint_i llama_constraint_typical_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "typical"; },
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        const auto * ctx = (llama_constraint_context_typical *) cnstr->ctx;
        llama_constraint_typical_impl(cur_p, ctx->p, ctx->min_keep);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx = (const llama_constraint_context_typical *) cnstr->ctx;
        return llama_constraint_init_typical_impl(ctx->p, ctx->min_keep);
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_typical *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_typical_impl(float p, size_t min_keep) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_typical_i,
        /* .ctx   = */ new llama_constraint_context_typical {
            /*.p        =*/ p,
            /*.min_keep =*/ min_keep,
        },
    };
}

// temp

struct llama_constraint_context_temp {
    const float temp;
};

static struct llama_constraint_i llama_constraint_temp_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "temp"; },
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        const auto * ctx = (llama_constraint_context_temp *) cnstr->ctx;
        llama_constraint_temp_impl(cur_p, ctx->temp);
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx = (const llama_constraint_context_temp *) cnstr->ctx;
        return llama_constraint_init_temp_impl(ctx->temp);
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_temp *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_temp_impl(float temp) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_temp_i,
        /* .ctx   = */ new llama_constraint_context_temp {
            /*.temp =*/ temp,
        },
    };
}

// temp-ext

struct llama_constraint_context_temp_ext {
    const float temp;
    const float delta;
    const float exponent;
};

static struct llama_constraint_i llama_constraint_temp_ext_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "temp-ext"; },
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        const auto * ctx = (llama_constraint_context_temp_ext *) cnstr->ctx;
        if (ctx->delta > 0) {
            const float temp_min = std::max(0.0f, ctx->temp - ctx->delta);
            const float temp_max = ctx->temp + ctx->delta;

            llama_constraint_entropy_impl(cur_p, temp_min, temp_max, ctx->exponent);
        } else {
            llama_constraint_temp_impl(cur_p, ctx->temp);
        }
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx = (const llama_constraint_context_temp_ext *) cnstr->ctx;
        return llama_constraint_init_temp_ext_impl(ctx->temp, ctx->delta, ctx->exponent);
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_temp_ext *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_temp_ext_impl(float temp, float delta, float exponent) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_temp_ext_i,
        /* .ctx   = */ new llama_constraint_context_temp_ext {
            /*.temp     =*/ temp,
            /*.delta    =*/ delta,
            /*.exponent =*/ exponent,
        },
    };
}

// mirostat

struct llama_constraint_context_mirostat {
    const struct llama_vocab * vocab;

    const float tau;
    const float eta;

    const int32_t m;

    float mu;

    std::vector<llama_token_data> cur;
};

static struct llama_constraint_i llama_constraint_mirostat_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "mirostat"; },
    /* .accept = */ [](struct llama_constraint * cnstr, llama_token token) {
        auto * ctx = (llama_constraint_context_mirostat *) cnstr->ctx;

        int32_t idx = -1;
        for (size_t i = 0; i < ctx->cur.size(); ++i) {
            if (ctx->cur[i].id == token) {
                idx = i;
                break;
            }
        }

        float observed_surprise = -log2f(ctx->cur[idx].p);
        float e = observed_surprise - ctx->tau;

        // Update mu using the learning rate and error
        ctx->mu = ctx->mu - ctx->eta * e;
    },
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        auto * ctx = (llama_constraint_context_mirostat *) cnstr->ctx;

        llama_constraint_softmax_impl(cur_p);

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
        float k = powf((epsilon_hat * powf(2, ctx->mu)) / (1 - powf(ctx->vocab->n_vocab, -epsilon_hat)), 1 / s_hat);

        llama_constraint_top_k_impl(cur_p, int(k), 1);

        // remember the order to be able to compute the distance later when accepting the token
        ctx->cur.resize(cur_p->size);
        for (size_t i = 0; i < cur_p->size; ++i) {
            ctx->cur[i] = cur_p->data[i];
        }
    },
    /* .reset  = */ [](struct llama_constraint * cnstr) {
        auto * ctx = (llama_constraint_context_mirostat *) cnstr->ctx;
        ctx->mu = 0.0f;
    },
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx = (const llama_constraint_context_mirostat *) cnstr->ctx;
        return llama_constraint_init_mirostat_impl(*ctx->vocab, ctx->tau, ctx->eta, ctx->m);
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_mirostat *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_mirostat_impl(const struct llama_vocab & vocab, float tau, float eta, int32_t m) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_mirostat_i,
        /* .ctx   = */ new llama_constraint_context_mirostat {
            /*.vocab =*/ &vocab,
            /*.tau   =*/ tau,
            /*.eta   =*/ eta,
            /*.m     =*/ m,
            /*.mu    =*/ 0.0f,
            /*.cur   =*/ {},
        },
    };
}

// mirostat v2

struct llama_constraint_context_mirostat_v2 {
    const float tau;
    const float eta;

    float mu;

    std::vector<llama_token_data> cur;
};

static struct llama_constraint_i llama_constraint_mirostat_v2_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "mirostat-v2"; },
    /* .accept = */ [](struct llama_constraint * cnstr, llama_token token) {
        auto * ctx = (llama_constraint_context_mirostat_v2 *) cnstr->ctx;

        int32_t idx = -1;
        for (size_t i = 0; i < ctx->cur.size(); ++i) {
            if (ctx->cur[i].id == token) {
                idx = i;
                break;
            }
        }

        float observed_surprise = -log2f(ctx->cur[idx].p);
        float e = observed_surprise - ctx->tau;

        // Update mu using the learning rate and error
        ctx->mu = ctx->mu - ctx->eta * e;
    },
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        auto * ctx = (llama_constraint_context_mirostat_v2 *) cnstr->ctx;

        llama_constraint_softmax_impl(cur_p);

        // Truncate the words with surprise values greater than mu
        cur_p->size = std::distance(cur_p->data, std::find_if(cur_p->data, cur_p->data + cur_p->size, [&](const llama_token_data & candidate) {
            return -log2f(candidate.p) > ctx->mu;
        }));

        if (cur_p->size == 0) {
            cur_p->size = 1;
        }

        // Normalize the probabilities of the remaining words
        llama_constraint_softmax_impl(cur_p);
    },
    /* .reset  = */ [](struct llama_constraint * cnstr) {
        auto * ctx = (llama_constraint_context_mirostat_v2 *) cnstr->ctx;
        ctx->mu = 0.0f;
    },
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx = (const llama_constraint_context_mirostat_v2 *) cnstr->ctx;
        return llama_constraint_init_mirostat_v2_impl(ctx->tau, ctx->eta);
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_mirostat_v2 *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_mirostat_v2_impl(float tau, float eta) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_mirostat_v2_i,
        /* .ctx   = */ new llama_constraint_context_mirostat_v2 {
            /*.tau =*/ tau,
            /*.eta =*/ eta,
            /*.mu  =*/ 0.0f,
            /*.cur =*/ {},
        },
    };
}

// grammar

struct llama_constraint_context_grammar {
    std::string grammar_str;
    std::string grammar_root;

    struct llama_grammar * grammar;
};

static struct llama_constraint_i llama_constraint_grammar_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "grammar"; },
    /* .accept = */ [](struct llama_constraint * cnstr, llama_token token) {
        const auto * ctx = (llama_constraint_context_grammar *) cnstr->ctx;
        if (ctx->grammar) {
            llama_grammar_accept_impl(*ctx->grammar, token);
        }
    },
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        const auto * ctx = (llama_constraint_context_grammar *) cnstr->ctx;
        if (ctx->grammar) {
            llama_constraint_grammar_impl(cur_p, *ctx->grammar);
        }
    },
    /* .reset  = */ [](struct llama_constraint * cnstr) {
        auto * ctx = (llama_constraint_context_grammar *) cnstr->ctx;
        if (!ctx->grammar) {
            return;
        }

        auto * grammar_new = llama_grammar_init_impl(ctx->grammar->vocab, ctx->grammar_str.c_str(), ctx->grammar_root.c_str());

        llama_grammar_free_impl(ctx->grammar);
        ctx->grammar = grammar_new;
    },
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx_src = (const llama_constraint_context_grammar *) cnstr->ctx;

        auto * result = llama_constraint_init_grammar_impl(*ctx_src->grammar->vocab, nullptr, nullptr);

        auto * ctx_dst = (llama_constraint_context_grammar *) result->ctx;
        if (ctx_src->grammar) {
            ctx_dst->grammar_str  = ctx_src->grammar_str;
            ctx_dst->grammar_root = ctx_src->grammar_root;

            ctx_dst->grammar = llama_grammar_cp_impl(*ctx_src->grammar);
        }

        return result;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        const auto * ctx = (llama_constraint_context_grammar *) cnstr->ctx;

        if (ctx->grammar) {
            llama_grammar_free_impl(ctx->grammar);
        }

        delete ctx;
    },
};

struct llama_constraint * llama_constraint_init_grammar_impl(const struct llama_vocab & vocab, const char * grammar_str, const char * grammar_root) {
    auto * ctx = new llama_constraint_context_grammar;

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

    return new llama_constraint {
        /* .iface = */ &llama_constraint_grammar_i,
        /* .ctx   = */ ctx,
    };
}

// penalties

struct llama_constraint_context_penalties {
    const struct llama_vocab * vocab;

    const int32_t penalty_last_n;
    const float   penalty_repeat;
    const float   penalty_freq;
    const float   penalty_present;

    const bool    penalize_nl;
    const bool    ignore_eos;

    ring_buffer<llama_token> prev;
};

static struct llama_constraint_i llama_constraint_penalties_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "penalties"; },
    /* .accept = */ [](struct llama_constraint * cnstr, llama_token token) {
        auto * ctx = (llama_constraint_context_penalties *) cnstr->ctx;
        ctx->prev.push_back(token);
    },
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        auto * ctx = (llama_constraint_context_penalties *) cnstr->ctx;

        GGML_ASSERT(cur_p->size == ctx->vocab->n_vocab && cur_p->sorted == false && "the 'penalties' constraint must be applied on the full vocabulary");

        if (ctx->ignore_eos) {
            cur_p->data[ctx->vocab->special_eos_id].logit = -INFINITY;
        }

        if ((ctx->penalty_last_n == 0) ||
            (ctx->penalty_repeat == 1.0f && ctx->penalty_freq == 0.0f && ctx->penalty_present == 0.0f)) {
            return;
        }

        const float nl_logit = !ctx->penalize_nl ? cur_p->data[ctx->vocab->linefeed_id].logit : -INFINITY;

        // Create a frequency map to count occurrences of each token in last_tokens
        // TODO: optimize this by maintaining the token count in the constraint context
        llama_token_cnt token_count;
        for (int i = 0; i < ctx->penalty_last_n; ++i) {
            token_count[ctx->prev.rat(i)]++;
        }

        llama_constraint_penalties_impl(cur_p, token_count, ctx->penalty_repeat, ctx->penalty_freq, ctx->penalty_present);

        if (!ctx->penalize_nl) {
            // restore the logit of the newline token if it was penalized
            cur_p->data[ctx->vocab->linefeed_id].logit = nl_logit;
        }
    },
    /* .reset  = */ [](struct llama_constraint * cnstr) {
        auto * ctx = (llama_constraint_context_penalties *) cnstr->ctx;
        ctx->prev.clear();
    },
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx_src = (const llama_constraint_context_penalties *) cnstr->ctx;
        auto * result = llama_constraint_init_penalties_impl(
               *ctx_src->vocab,
                ctx_src->penalty_last_n,
                ctx_src->penalty_repeat,
                ctx_src->penalty_freq,
                ctx_src->penalty_present,
                ctx_src->penalize_nl,
                ctx_src->ignore_eos);

        auto * ctx_dst = (llama_constraint_context_penalties *) result->ctx;
        ctx_dst->prev = ctx_src->prev;

        return result;
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_penalties *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_penalties_impl(const struct llama_vocab & vocab, int32_t penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present, bool penalize_nl, bool ignore_eos) {
    GGML_ASSERT(penalize_nl || vocab.linefeed_id    != LLAMA_TOKEN_NULL);
    GGML_ASSERT(!ignore_eos || vocab.special_eos_id != LLAMA_TOKEN_NULL);

    return new llama_constraint {
        /* .iface = */ &llama_constraint_penalties_i,
        /* .ctx   = */ new llama_constraint_context_penalties {
            /*.vocab           =*/ &vocab,
            /*.penalty_last_n  =*/ penalty_last_n,
            /*.penalty_repeat  =*/ penalty_repeat,
            /*.penalty_freq    =*/ penalty_freq,
            /*.penalty_present =*/ penalty_present,
            /*.penalize_nl     =*/ penalize_nl,
            /*.ignore_eos      =*/ ignore_eos,
            /*.prev            =*/ ring_buffer<llama_token>(penalty_last_n),
        },
    };
}

// logit-bias

struct llama_constraint_context_logit_bias {
    const struct llama_vocab * vocab;

    std::vector<llama_logit_bias> logit_bias;
};

static struct llama_constraint_i llama_constraint_logit_bias_i = {
    /* .name   = */ [](const struct llama_constraint * /*cnstr*/) { return "logit-bias"; },
    /* .accept = */ nullptr,
    /* .apply  = */ [](struct llama_constraint * cnstr, llama_token_data_array * cur_p) {
        auto * ctx = (llama_constraint_context_logit_bias *) cnstr->ctx;

        GGML_ASSERT(cur_p->size == ctx->vocab->n_vocab && cur_p->sorted == false && "the 'logit_bias' constraint must be applied on the full vocabulary");

        for (const auto & lb : ctx->logit_bias) {
            cur_p->data[lb.token].logit += lb.bias;
        }
    },
    /* .reset  = */ nullptr,
    /* .copy   = */ [](const struct llama_constraint * cnstr) {
        const auto * ctx_src = (const llama_constraint_context_logit_bias *) cnstr->ctx;
        return llama_constraint_init_logit_bias_impl(*ctx_src->vocab, ctx_src->logit_bias.size(), ctx_src->logit_bias.data());
    },
    /* .free   = */ [](struct llama_constraint * cnstr) {
        delete (llama_constraint_context_logit_bias *) cnstr->ctx;
    },
};

struct llama_constraint * llama_constraint_init_logit_bias_impl(
        const struct llama_vocab & vocab,
                         int32_t   n_logit_bias,
          const llama_logit_bias * logit_bias) {
    return new llama_constraint {
        /* .iface = */ &llama_constraint_logit_bias_i,
        /* .ctx   = */ new llama_constraint_context_logit_bias {
            /*.vocab     =*/ &vocab,
            /*.logit_bias=*/ std::vector<llama_logit_bias>(logit_bias, logit_bias + n_logit_bias),
        },
    };
}

////////////////////////////////////////

struct llama_constraint * llama_constraint_cp_impl(const struct llama_constraint & cnstr) {
    return cnstr.iface->copy ? cnstr.iface->copy(&cnstr) : nullptr;
}

void llama_constraint_free_impl(struct llama_constraint * cnstr) {
    if (cnstr == nullptr) {
        return;
    }

    if (cnstr->iface->free) {
        cnstr->iface->free(cnstr);
    }

    delete cnstr;
}

void llama_constraint_accept_impl(struct llama_constraint & cnstr, llama_token token) {
    if (cnstr.iface->accept) {
        cnstr.iface->accept(&cnstr, token);
    }
}

void llama_constraint_apply_impl(struct llama_constraint & cnstr, struct llama_token_data_array * cur_p) {
    GGML_ASSERT(cnstr.iface->apply);
    cnstr.iface->apply(&cnstr, cur_p);
}

void llama_constraint_reset_impl(struct llama_constraint & cnstr) {
    if (cnstr.iface->reset) {
        cnstr.iface->reset(&cnstr);
    }
}

//
// samplers
//

struct llama_sampler * llama_sampler_init_impl(const struct llama_vocab & vocab, struct llama_sampler_params params) {
    return new llama_sampler {
        /* .params = */ params,
        /* .vocab  = */ &vocab,

        /* .rng = */ std::mt19937(params.seed),

        /* .prev        = */ { (size_t) params.n_prev },
        /* .constraints = */ {},
        /* .cur         = */ {},
        /* .cur_p       = */ {},
        /* .t_sample_us = */ 0,
        /* .n_sample    = */ 0,
    };
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
    auto * result = new llama_sampler {
        /* .params = */ smpl.params,
        /* .vocab  = */ smpl.vocab,

        /* .rng = */ smpl.rng,

        /* .prev        = */ smpl.prev,
        /* .constraints = */ {},
        /* .cur         = */ {},
        /* .cur_p       = */ {},
        /* .t_sample_us = */ 0,
        /* .n_sample    = */ 0,
    };

    // copy the constraints objects
    result->constraints.clear();
    for (const auto & cnstr : smpl.constraints) {
        if (cnstr->ctx == nullptr) {
            result->constraints.push_back(new llama_constraint {
                /* .iface = */ cnstr->iface,
                /* .ctx   = */ nullptr,
            });
        } else {
            GGML_ASSERT(cnstr->iface->copy);
            result->constraints.push_back(cnstr->iface->copy(cnstr));
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

void llama_sampler_constraint_add_impl(struct llama_sampler & smpl, struct llama_constraint * cnstr) {
    smpl.constraints.push_back(cnstr);
}

int llama_sampler_n_constraints_impl (const struct llama_sampler & smpl) {
    return smpl.constraints.size();
}

struct llama_constraint * llama_sampler_constraint_get_impl(const struct llama_sampler & smpl, int ith) {
    if (ith < 0 || ith >= (int) smpl.constraints.size()) {
        return nullptr;
    }

    return smpl.constraints[ith];
}

void llama_sampler_accept_impl(struct llama_sampler & smpl, llama_token token) {
    smpl.prev.push_back(token);

    for (auto * cnstr : smpl.constraints) {
        llama_constraint_accept_impl(*cnstr, token);
    }
}

void llama_sampler_apply_impl(struct llama_sampler & smpl, struct llama_token_data_array * cur_p) {
    for (auto * cnstr : smpl.constraints) {
        llama_constraint_apply_impl(*cnstr, cur_p);
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

llama_token llama_sampler_sample_greedy_impl(llama_token_data_array * cur_p, bool probs) {
    if (probs) {
        // if probs are needed, we apply softmax to get the probabilities
        llama_constraint_softmax_impl(cur_p);

        // the cur_p are sorted, so we can just return the first one
        return cur_p->data[0].id;
    }

    // return the token with the highest logit
    auto * max_iter = std::max_element(cur_p->data, cur_p->data + cur_p->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;

    return result;
}

llama_token llama_sampler_sample_dist_impl(struct llama_token_data_array * cur_p, std::mt19937 & rng) {
    llama_constraint_softmax_impl(cur_p);

    std::vector<float> probs;
    probs.reserve(cur_p->size);

    for (size_t i = 0; i < cur_p->size; ++i) {
        probs.push_back(cur_p->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());

    const int idx = dist(rng);
    llama_token result = cur_p->data[idx].id;

    return result;
}
