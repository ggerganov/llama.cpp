#include "sampling.h"

#include "common.h"

struct llama_sampling_context * llama_sampling_init(const struct gpt_sampling_params & params, const struct llama_model * model) {
    struct llama_sampling_context * result = new llama_sampling_context();

    result->params = params;

    {
        auto lp = llama_sampling_default_params();

        lp.seed              = params.seed;
        lp.n_prev            = params.n_prev;
        lp.n_probs           = params.n_probs;
        lp.min_keep          = params.min_keep;
        lp.top_k             = params.top_k;
        lp.top_p             = params.top_p;
        lp.min_p             = params.min_p;
        lp.tfs_z             = params.tfs_z;
        lp.typical_p         = params.typical_p;
        lp.temp              = params.temp;
        lp.dynatemp_range    = params.dynatemp_range;
        lp.dynatemp_exponent = params.dynatemp_exponent;
        lp.penalty_last_n    = params.penalty_last_n;
        lp.penalty_repeat    = params.penalty_repeat;
        lp.penalty_freq      = params.penalty_freq;
        lp.penalty_present   = params.penalty_present;
        lp.mirostat          = params.mirostat;
        lp.mirostat_tau      = params.mirostat_tau;
        lp.mirostat_eta      = params.mirostat_eta;
        lp.penalize_nl       = params.penalize_nl;
        lp.ignore_eos        = params.ignore_eos;

        result->smpl = llama_sampling_init(model, lp);

        llama_sampling_set_rng_seed  (result->smpl, params.seed);
        llama_sampling_set_grammar   (result->smpl, params.grammar.c_str(), "root");
        llama_sampling_set_cfg       (result->smpl, params.cfg_negative_prompt.c_str(), params.cfg_scale);
        llama_sampling_set_logit_bias(result->smpl, params.logit_bias.size(), params.logit_bias.data());
    }

    result->prev = ring_buffer<llama_token>(params.n_prev);

    result->n_valid = 0;

    return result;
}

void llama_sampling_free(struct llama_sampling_context * ctx) {
    llama_sampling_free(ctx->smpl);

    delete ctx;
}

void llama_sampling_reset(llama_sampling_context * ctx) {
    llama_sampling_reset(ctx->smpl);

    ctx->prev.clear();
    ctx->cur.clear();
    ctx->n_valid = 0;
}

void llama_sampling_cp(llama_sampling_context * src, llama_sampling_context * dst) {
    if (dst->smpl) {
        llama_sampling_free(dst->smpl);
    }

    dst->smpl = llama_sampling_cp(src->smpl);
    dst->prev = src->prev;
}

llama_token llama_sampling_last(llama_sampling_context * ctx) {
    return ctx->prev.back();
}

std::string llama_sampling_prev_str(llama_sampling_context * ctx_sampling, llama_context * ctx_main, int n) {
    const int size = ctx_sampling->prev.size();

    n = std::min(n, size);

    std::string result;

    for (int i = size - n; i < size; i++) {
        result += llama_token_to_piece(ctx_main, ctx_sampling->prev[i]);
    }

    return result;
}

std::string llama_sampling_print(const gpt_sampling_params & params) {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\ttop_k = %d, tfs_z = %.3f, top_p = %.3f, min_p = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present,
            params.top_k, params.tfs_z, params.top_p, params.min_p, params.typical_p, params.temp,
            params.mirostat, params.mirostat_eta, params.mirostat_tau);

    return std::string(result);
}

std::string llama_sampling_order_print(const gpt_sampling_params & params) {
    std::string result = "CFG -> Penalties ";
    if (params.mirostat == 0) {
        for (auto sampler_type : params.samplers_sequence) {
            const auto sampler_type_name = llama_sampling_type_to_str(sampler_type);
            if (!sampler_type_name.empty()) {
                result += "-> " + sampler_type_name + " ";
            }
        }
    } else {
        result += "-> mirostat ";
    }

    return result;
}

std::string llama_sampling_type_to_str(llama_sampler_type sampler_type) {
    switch (sampler_type) {
        case llama_sampler_type::TOP_K:       return "top_k";
        case llama_sampler_type::TFS_Z:       return "tfs_z";
        case llama_sampler_type::TYPICAL_P:   return "typical_p";
        case llama_sampler_type::TOP_P:       return "top_p";
        case llama_sampler_type::MIN_P:       return "min_p";
        case llama_sampler_type::TEMPERATURE: return "temperature";
        default : return "";
    }
}

std::vector<llama_sampler_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, llama_sampler_type> sampler_canonical_name_map {
        {"top_k",       llama_sampler_type::TOP_K},
        {"top_p",       llama_sampler_type::TOP_P},
        {"typical_p",   llama_sampler_type::TYPICAL_P},
        {"min_p",       llama_sampler_type::MIN_P},
        {"tfs_z",       llama_sampler_type::TFS_Z},
        {"temperature", llama_sampler_type::TEMPERATURE}
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, llama_sampler_type> sampler_alt_name_map {
        {"top-k",       llama_sampler_type::TOP_K},
        {"top-p",       llama_sampler_type::TOP_P},
        {"nucleus",     llama_sampler_type::TOP_P},
        {"typical-p",   llama_sampler_type::TYPICAL_P},
        {"typical",     llama_sampler_type::TYPICAL_P},
        {"min-p",       llama_sampler_type::MIN_P},
        {"tfs-z",       llama_sampler_type::TFS_Z},
        {"tfs",         llama_sampler_type::TFS_Z},
        {"temp",        llama_sampler_type::TEMPERATURE}
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names.size());
    for (const auto & name : names)
    {
        auto sampler_item = sampler_canonical_name_map.find(name);
        if (sampler_item != sampler_canonical_name_map.end())
        {
            sampler_types.push_back(sampler_item->second);
        }
        else
        {
            if (allow_alt_names)
            {
                sampler_item = sampler_alt_name_map.find(name);
                if (sampler_item != sampler_alt_name_map.end())
                {
                    sampler_types.push_back(sampler_item->second);
                }
            }
        }
    }
    return sampler_types;
}

std::vector<llama_sampler_type> llama_sampling_types_from_chars(const std::string & names_string) {
    std::unordered_map<char, llama_sampler_type> sampler_name_map {
        {'k', llama_sampler_type::TOP_K},
        {'p', llama_sampler_type::TOP_P},
        {'y', llama_sampler_type::TYPICAL_P},
        {'m', llama_sampler_type::MIN_P},
        {'f', llama_sampler_type::TFS_Z},
        {'t', llama_sampler_type::TEMPERATURE}
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names_string.size());
    for (const auto & c : names_string) {
        const auto sampler_item = sampler_name_map.find(c);
        if (sampler_item != sampler_name_map.end()) {
            sampler_types.push_back(sampler_item->second);
        }
    }
    return sampler_types;
}

// no reasons to expose this function in header
static void sampler_queue(
          struct llama_sampling_context * ctx_sampling,
                 llama_token_data_array & cur_p,
                                 size_t   min_keep) {
    llama_sampling * smpl = ctx_sampling->smpl;

    const gpt_sampling_params & params = ctx_sampling->params;

    const float         temp              = params.temp;
    const float         dynatemp_range    = params.dynatemp_range;
    const float         dynatemp_exponent = params.dynatemp_exponent;
    const int32_t       top_k             = params.top_k;
    const float         top_p             = params.top_p;
    const float         min_p             = params.min_p;
    const float         tfs_z             = params.tfs_z;
    const float         typical_p         = params.typical_p;
    const std::vector<llama_sampler_type> & samplers_sequence = params.samplers_sequence;

    for (auto sampler_type : samplers_sequence) {
        switch (sampler_type) {
            case llama_sampler_type::TOP_K    : llama_sampling_top_k    (smpl, &cur_p, top_k,     min_keep); break;
            case llama_sampler_type::TFS_Z    : llama_sampling_tail_free(smpl, &cur_p, tfs_z,     min_keep); break;
            case llama_sampler_type::TYPICAL_P: llama_sampling_typical  (smpl, &cur_p, typical_p, min_keep); break;
            case llama_sampler_type::TOP_P    : llama_sampling_top_p    (smpl, &cur_p, top_p,     min_keep); break;
            case llama_sampler_type::MIN_P    : llama_sampling_min_p    (smpl, &cur_p, min_p,     min_keep); break;
            case llama_sampler_type::TEMPERATURE:
                if (dynatemp_range > 0) {
                    float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                    float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                    llama_sampling_entropy(smpl, &cur_p, dynatemp_min, dynatemp_max, dynatemp_exponent);
                } else {
                    llama_sampling_temp(smpl, &cur_p, temp);
                }
                break;
            default : break;
        }
    }
}

static llama_token llama_sampling_sample_impl(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool is_resampling) {
    llama_sampling * smpl = ctx_sampling->smpl;

    const gpt_sampling_params & params = ctx_sampling->params;

    const float temp         = params.temp;
    const int   mirostat     = params.mirostat;
    const float mirostat_tau = params.mirostat_tau;
    const float mirostat_eta = params.mirostat_eta;

    std::vector<float> original_logits;
    auto cur_p = llama_sampling_prepare(ctx_sampling, ctx_main, ctx_cfg, idx, /* apply_grammar= */ is_resampling, &original_logits);
    if (!is_resampling) {
        GGML_ASSERT(!original_logits.empty());
    }
    llama_token id = 0;

    if (temp < 0.0) {
        // greedy sampling, with probs
        llama_sampling_softmax(smpl, &cur_p);
        id = cur_p.data[0].id;
    } else if (temp == 0.0) {
        // greedy sampling, no probs
        id = llama_sampling_sample_greedy(smpl, &cur_p);
    } else {
        if (mirostat == 1) {
            const int mirostat_m = 100;
            llama_sampling_temp(smpl, &cur_p, temp);
            id = llama_sampling_sample_mirostat(smpl, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &ctx_sampling->mirostat_mu);
        } else if (mirostat == 2) {
            llama_sampling_temp(smpl, &cur_p, temp);
            id = llama_sampling_sample_mirostat_v2(smpl, &cur_p, mirostat_tau, mirostat_eta, &ctx_sampling->mirostat_mu);
        } else {
            // temperature sampling
            size_t min_keep = std::max(1, params.min_keep);

            sampler_queue(ctx_sampling, cur_p, min_keep);

            id = llama_sampling_sample(smpl, &cur_p);

            //{
            //    const int n_top = 10;
            //    LOG("top %d candidates:\n", n_top);

            //    for (int i = 0; i < n_top; i++) {
            //        const llama_token id = cur_p.data[i].id;
            //        (void)id; // To avoid a warning that id is unused when logging is disabled.
            //        LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(smpl, id).c_str(), cur_p.data[i].p);
            //    }
            //}

            //LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(smpl, id).c_str());
        }
    }

    if (!is_resampling) {
        // Get a pointer to the logits
        float * logits = llama_get_logits_ith(ctx_main, idx);

        // Create an array with a single token data element for the sampled id
        llama_token_data single_token_data = {id, logits[id], 0.0f};
        llama_token_data_array single_token_data_array = { &single_token_data, 1, false };

        // Apply grammar constraints to the single token
        llama_sampling_grammar(ctx_sampling->smpl, &single_token_data_array);

        // Check if the token is valid according to the grammar by seeing if its logit has been set to -INFINITY
        bool is_valid = single_token_data_array.data[0].logit != -INFINITY;

        // If the token is not valid according to the grammar, perform resampling
        if (!is_valid) {
            LOG("Resampling because token %d: '%s' does not meet grammar rules\n", id, llama_token_to_piece(ctx_main, id).c_str());

            // Restore logits from the copy
            std::copy(original_logits.begin(), original_logits.end(), logits);

            return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ true);
        }
    }

    ctx_sampling->n_valid = temp == 0.0f ? 0 : cur_p.size;

    return id;
}

static llama_token_data_array llama_sampling_prepare_impl(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool apply_grammar,
                  std::vector<float> * original_logits) {
    llama_sampling * smpl = ctx_sampling->smpl;

    const gpt_sampling_params & params = ctx_sampling->params;

    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));

    const int32_t penalty_last_n  = params.penalty_last_n < 0 ? params.n_prev : params.penalty_last_n;
    const float   penalty_repeat  = params.penalty_repeat;
    const float   penalty_freq    = params.penalty_freq;
    const float   penalty_present = params.penalty_present;

    const bool    penalize_nl     = params.penalize_nl;

    auto & prev = ctx_sampling->prev;
    auto & cur  = ctx_sampling->cur;

    // Get a pointer to the logits
    float * logits = llama_get_logits_ith(ctx_main, idx);

    if (!apply_grammar) {
        GGML_ASSERT(original_logits != NULL);
        // Only make a copy of the original logits if we are not applying grammar checks, not sure if I actually have to do this.
        *original_logits = {logits, logits + n_vocab};
    }

    // apply params.logit_bias map
    for (const auto & logit_bias : params.logit_bias) {
        logits[logit_bias.token] += logit_bias.bias;
    }

    if (params.ignore_eos) {
        logits[llama_token_eos(llama_get_model(ctx_main))] = -INFINITY;
    }

    if (ctx_cfg) {
        float * logits_guidance = llama_get_logits_ith(ctx_cfg, idx);
        llama_sampling_cfg(smpl, logits, logits_guidance, params.cfg_scale);
    }

    cur.resize(n_vocab);

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), false };

    // apply penalties
    const auto & penalty_tokens = prev.to_vector();
    const int penalty_tokens_used_size = std::min((int)penalty_tokens.size(), penalty_last_n);
    if (penalty_tokens_used_size) {
        const float nl_logit = logits[llama_token_nl(llama_get_model(ctx_main))];

        llama_sampling_repetition_penalties(smpl, &cur_p,
                penalty_tokens.data() + penalty_tokens.size() - penalty_tokens_used_size,
                penalty_tokens_used_size, penalty_repeat, penalty_freq, penalty_present);

        if (!penalize_nl) {
            for (size_t idx = 0; idx < cur_p.size; idx++) {
                if (cur_p.data[idx].id == llama_token_nl(llama_get_model(ctx_main))) {
                    cur_p.data[idx].logit = nl_logit;
                    break;
                }
            }
        }
    }

    // apply grammar checks before sampling logic
    if (apply_grammar) {
        llama_sampling_grammar(ctx_sampling->smpl, &cur_p);
    }

    return cur_p;
}

llama_token llama_sampling_sample(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx) {
    // Call the implementation function with is_resampling set to false by default
    return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ false);
}

llama_token_data_array llama_sampling_prepare(
                  struct llama_sampling_context * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool apply_grammar,
                  std::vector<float> * original_logits) {
    return llama_sampling_prepare_impl(ctx_sampling, ctx_main, ctx_cfg, idx, apply_grammar, original_logits);
}

void llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        llama_token id,
        bool apply_grammar) {
    if (!ctx_sampling->prev.empty()) {
        ctx_sampling->prev.pop_front();
    }
    ctx_sampling->prev.push_back(id);

    if (apply_grammar) {
        llama_sampling_accept(ctx_sampling->smpl, id);
    }
}
