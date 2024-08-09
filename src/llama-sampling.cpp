#include "llama-sampling.h"

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

void llama_set_rng_seed_impl(struct llama_sampling * smpl, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = time(NULL);
    }

    smpl->rng.seed(seed);
}

void llama_sample_softmax_impl(struct llama_sampling * smpl, llama_token_data_array * candidates) {
    GGML_ASSERT(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_time_us();

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

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_k_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, int32_t k, size_t min_keep) {
    // TODO: move bucket sort to separate function so that top_p/tail_free/typical/softmax first is equally fast
    // if (k >= (int32_t)candidates->size) {
    //     return;
    // }

    const int64_t t_start_sample_us = ggml_time_us();

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
            constexpr float bucker_inter = -bucket_low * bucket_scale;

            std::vector<int> bucket_idx(candidates->size);
            std::vector<int> histo(nbuckets, 0);

            for (int i = 0; i < (int)candidates->size; ++i) {
                const float val = candidates->data[i].logit;
                int ib = int(bucket_scale * val + bucker_inter); //nbuckets * (val - bucket_low) / (bucket_high - bucket_low);
                ib = std::max(0, std::min(nbuckets-1, ib));
                bucket_idx[i] = ib;
                ++histo[ib];
            }
            int nhave = 0;
            int ib = nbuckets - 1;
            for ( ; ib >= 0; --ib) {
                nhave += histo[ib];
                if (nhave >= k) break;
            }
            std::vector<llama_token_data> tmp_tokens(nhave);
            auto ptr = tmp_tokens.data();
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

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_p_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    llama_sample_softmax_impl(smpl, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

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

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_min_p_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p <= 0.0f || !candidates->size) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

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

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    return llama_tokenize(llama_get_model(ctx), text, add_special, parse_special);
}

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_detokenize(llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(llama_get_model(ctx), tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

std::string llama_detokenize_single(llama_context * ctx, llama_token token, bool special) {
    std::vector<llama_token> tokens = {token};
    return llama_detokenize(ctx, tokens, special);
}

// Constants for preventing overflow
const float FLOAT_MAX_LOG = 88.7228391f;
const int MAX_CHAR_LEN = 40;
const int MAX_SEQ_LEN = 20;


void llama_sample_dry_impl(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float dry_base, float dry_multiplier, int dry_allowed_length, const std::vector<std::string> & dry_seq_breakers) {
    if (last_tokens_size < 1) {
        return;
    }

    // Cache for token-to-string conversions
    std::unordered_map<llama_token, std::string> token_to_string_cache;
    // Store sequence breakers for more efficient lookup
    std::unordered_multimap<std::string, std::vector<std::string>> restart_sequences;

    auto detokenize_with_cache = [&](llama_token token) -> std::string {
        auto it = token_to_string_cache.find(token);
        if (it != token_to_string_cache.end()) {
            return it->second;
        }
        std::string token_str = llama_detokenize_single(ctx, token, false);
        token_to_string_cache[token] = token_str;
        return token_str;
    };

    // Pre-process dry_seq_breakers
    for (const auto& breaker : dry_seq_breakers) {
        std::string breaker_trimmed = breaker.substr(0, MAX_CHAR_LEN);
        std::vector<llama_token> tokens = llama_tokenize(ctx, breaker_trimmed, false, false);

        if (!tokens.empty()) {
            std::string head = detokenize_with_cache(tokens[0]);
            std::vector<std::string> tail;

            for (size_t i = 1; i < tokens.size() && i <= MAX_SEQ_LEN; ++i) {
                tail.push_back(detokenize_with_cache(tokens[i]));
            }
            restart_sequences.emplace(head, tail);
        }
    }

    // Find max repetition length considering restart sequences
    int rep_limit = last_tokens_size;

    for (size_t i = 0; i < last_tokens_size; ++i) {
        size_t ix = last_tokens_size - 1 - i;
        std::string token_str = detokenize_with_cache(last_tokens[ix]);

        // Check if the token is a potential sequence breaker
        auto its = restart_sequences.equal_range(token_str);
        if (its.first == restart_sequences.end()) continue;

        int longest_match = -1;
        // Check all potential sequence breakers starting with this token
        for (auto it = its.first; it != its.second; ++it) {
            int seq_len = (int)it->second.size();
            if (seq_len > longest_match && seq_len <= i) {
                bool match = true;
                // Check if the following tokens match the sequence breaker
                for (size_t offset = 0; offset < seq_len; ++offset) {
                    if (it->second[offset] != detokenize_with_cache(last_tokens[ix + 1 + offset])) {
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
            rep_limit = static_cast<int>(i) - longest_match;
            break;
        }
    }

    if (rep_limit <= dry_allowed_length) {
        return;
    }

    // Store max match length for each token
    std::unordered_map<llama_token, size_t> match_lengths;

    // Find repeated sequences
    for (size_t i = 0; i < last_tokens_size - 1; ++i) {
        if (last_tokens[i] != last_tokens[last_tokens_size - 1]) {
            continue;
        }

        auto next_token = last_tokens[i + 1];
        std::string next_token_str = detokenize_with_cache(next_token);

        // Skip if next token is a sequence breaker
        auto its = restart_sequences.equal_range(next_token_str);
        if (its.first != restart_sequences.end()) {
            continue;
        }

        size_t match_length = 1;

        // Extend match as far as possible
        for (;; match_length++) {
            if (i < match_length || match_length > rep_limit) {
                break;
            }

            auto compare_token = last_tokens[i - match_length];
            std::string compare_token_str = detokenize_with_cache(compare_token);

            auto head_token = last_tokens[last_tokens_size - 1 - match_length];
            std::string head_token_str = detokenize_with_cache(head_token);

            if (compare_token_str != head_token_str) {
                break;
            }

            // Check if we've hit a sequence breaker
            its = restart_sequences.equal_range(compare_token_str);
            if (its.first != restart_sequences.end()) {
                break;
            }
        }

        // Update max match length for this token
        auto it = match_lengths.find(next_token);
        if (it == match_lengths.end()) {
            match_lengths[next_token] = match_length;
        } else {
            it->second = std::max(it->second, match_length);
        }
    }

    // Calculate max safe exponent
    int max_exponent = 0;
    if (dry_base > 1.000001f) {
        max_exponent = static_cast<int>(FLOAT_MAX_LOG / log(dry_base));
    }

#ifdef DEBUG
    LLAMA_LOG_INFO("DRY Sampling parameters:\n");
    LLAMA_LOG_INFO("  dry_base: %f\n", dry_base);
    LLAMA_LOG_INFO("  dry_multiplier: %f\n", dry_multiplier);
    LLAMA_LOG_INFO("  dry_allowed_length: %d\n", dry_allowed_length);
    LLAMA_LOG_INFO("  max_exponent: %d\n", max_exponent);
    LLAMA_LOG_INFO("DRY penalties [");
#endif

    // Apply penalties
    for (const auto& pair : match_lengths) {
        auto next_token = pair.first;
        auto match_length = pair.second;

        if (match_length >= static_cast<size_t>(dry_allowed_length)) {
            for (size_t i = 0; i < candidates->size; ++i) {
                if (candidates->data[i].id == next_token) {
                    int repeat_exp = static_cast<int>(match_length - dry_allowed_length);
                    if (max_exponent > 0 && repeat_exp > max_exponent) {
                        repeat_exp = max_exponent;
                    }
                    float penalty = dry_multiplier * pow(dry_base, static_cast<float>(repeat_exp));
                    candidates->data[i].logit -= penalty;

#ifdef DEBUG
                    LLAMA_LOG_INFO("  Token %d: %s (Penalty: %.2f)", next_token, detokenize_with_cache(next_token).c_str(), penalty);
#endif
                    break;
                }
            }
        }
    }

#ifdef DEBUG
    LLAMA_LOG_INFO("]\n");
#endif
}

void llama_sample_tail_free_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);
    const int64_t t_start_sample_us = ggml_time_us();

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

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_typical_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

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

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_entropy_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float min_temp, float max_temp, float exponent_val) {
    const int64_t t_start_sample_us = ggml_time_us();

    // no need to do anything if there is only one (or zero) candidates
    if(candidates->size <= 1) {
        return;
    }

    // Calculate maximum possible entropy
    float max_entropy = -logf(1.0f / candidates->size);

    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

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
    double max_l_double = candidates->data[0].logit;
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

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_temp_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float temp) {
    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].logit /= temp;
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_repetition_penalties_impl(
        struct llama_sampling * smpl,
       llama_token_data_array * candidates,
            const llama_token * last_tokens,
                       size_t   penalty_last_n,
                       float   penalty_repeat,
                       float   penalty_freq,
                       float   penalty_present) {
    if (penalty_last_n == 0 || (penalty_repeat == 1.0f && penalty_freq == 0.0f && penalty_present == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<llama_token, int> token_count;
    for (size_t i = 0; i < penalty_last_n; ++i) {
        token_count[last_tokens[i]]++;
    }

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

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_apply_guidance_impl(
        struct llama_sampling * smpl,
                        float * logits,
                        float * logits_guidance,
                        float   scale) {
    GGML_ASSERT(smpl);

    const auto t_start_sample_us = ggml_time_us();
    const auto n_vocab = smpl->n_vocab;

    llama_log_softmax(logits, n_vocab);
    llama_log_softmax(logits_guidance, n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
              auto & l = logits[i];
        const auto & g = logits_guidance[i];

        l = scale * (l - g) + g;
    }

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
}

llama_token llama_sample_token_mirostat_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float tau, float eta, int32_t m, float * mu) {
    GGML_ASSERT(smpl);

    const int32_t n_vocab = float(smpl->n_vocab);

    int64_t t_start_sample_us = ggml_time_us();

    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

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
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(n_vocab, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_sample_top_k_impl((struct llama_sampling *) nullptr, candidates, int(k), 1);
    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    llama_token X = llama_sample_token_impl(smpl, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    return X;
}

llama_token llama_sample_token_mirostat_v2_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float tau, float eta, float * mu) {
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax_impl(smpl, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }

    // Normalize the probabilities of the remaining words
    llama_sample_softmax_impl(smpl, candidates);

    // Sample the next word X from the remaining words
    llama_token X = llama_sample_token_impl(smpl, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_greedy_impl(struct llama_sampling * smpl, llama_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_time_us();

    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;
    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
        smpl->n_sample++;
    }
    return result;
}

llama_token llama_sample_token_with_rng_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, std::mt19937 & rng) {
    GGML_ASSERT(smpl);

    const int64_t t_start_sample_us = ggml_time_us();
    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    llama_token result = candidates->data[idx].id;

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    smpl->n_sample++;

    return result;
}

llama_token llama_sample_token_impl(struct llama_sampling * smpl, llama_token_data_array * candidates) {
    return llama_sample_token_with_rng_impl(smpl, candidates, smpl->rng);
}
