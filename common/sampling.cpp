#define LLAMA_API_INTERNAL
#include "sampling.h"
#include <random>

//
// Token healing (internal)
//

static bool startswith(const std::string & str, const std::string & prefix) {
    return str.rfind(prefix, 0) != std::string::npos;
}

static bool token_healing_prefix_exists(const llama_context * ctx_main, const std::string & prefix) {
    const int32_t n_vocab = llama_n_vocab(llama_get_model(ctx_main));
    for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
        std::string token = llama_token_to_piece(ctx_main, token_id);
        if (startswith(token, prefix)) {
            return true;
        }
    }
    return false;
}

static std::vector<llama_token> token_healing_get_candidates(
                                const llama_context * ctx_main,
                                const std::string & prefix,
                                const bool include_partial_prefix) {
    // Example: prefix=" world" -> " world", " worldwide", ...
    // If `include_partial_prefix`, include also: " w", " wo", ...
    std::vector<llama_token> candidates;
    const int32_t n_vocab = llama_n_vocab(llama_get_model(ctx_main));
    for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
        std::string token = llama_token_to_piece(ctx_main, token_id);
        if (startswith(token, prefix) ||
                (include_partial_prefix && startswith(prefix, token))) {
            candidates.push_back(token_id);
        }
    }
    return candidates;
}

static size_t get_max_token_length(const llama_context * ctx_main) {
    const int32_t n_vocab = llama_n_vocab(llama_get_model(ctx_main));
    size_t len = 0;
    for (llama_token token_id = 0; token_id < n_vocab; ++token_id) {
        std::string token = llama_token_to_piece(ctx_main, token_id);
        len = std::max(len, token.size());
    }
    return len;
}

static llama_token_healing_output llama_token_healing_get_prefix(
                                  const llama_context * ctx_main,
                                  const llama_token_healing_type th_type,
                                  const std::vector<llama_token> & tokens,
                                  int max_to_remove) {
    if (tokens.size() <= 1) {
        return {};
    }

    const int n_ctx = tokens.size();
    max_to_remove = th_type == llama_token_healing_type::ROLLBACK_LAST ? 1 : max_to_remove;
    max_to_remove = max_to_remove < 0 ? n_ctx - 1 : std::min(max_to_remove, n_ctx - 1);  // 1 token must remain

    int removed = 0;
    std::string prefix;

    const llama_model * model = llama_get_model(ctx_main);
    auto is_special_token = [&](const llama_token token_id) {
        return llama_token_is_control(model,    token_id)
            || llama_token_bos       (model) == token_id
            || llama_token_eos       (model) == token_id
            || llama_token_cls       (model) == token_id
            || llama_token_sep       (model) == token_id
            || llama_token_pad       (model) == token_id
            || llama_token_prefix    (model) == token_id
            || llama_token_middle    (model) == token_id
            || llama_token_suffix    (model) == token_id
            || llama_token_eot       (model) == token_id;
    };

    if (th_type == llama_token_healing_type::DYNAMIC_ONCE || th_type == llama_token_healing_type::DYNAMIC_MULTI) {
        // The number of bytes to roll back cannot exceed the length of the longest token.
        const size_t n_longest_token = get_max_token_length(ctx_main);
        size_t len = 0;
        while (removed < max_to_remove) {
            const llama_token next_token_id = tokens[n_ctx - removed - 1];
            if (is_special_token(next_token_id)) {
                break;
            }
            const size_t next_token_size = llama_token_to_piece(ctx_main, next_token_id).size();
            if (len + next_token_size > n_longest_token) {
                break;
            }
            len += next_token_size;
            removed += 1;
        }

        while (removed > 0) {
            prefix.clear();
            for (int i = n_ctx - removed; i < n_ctx; i++) {
                prefix += llama_token_to_piece(ctx_main, tokens[i]);
            }
            if (token_healing_prefix_exists(ctx_main, prefix)) {
                break;  // Stop on longest valid prefix
            }
            removed -= 1;
        }
    } else {
        // Roll back tokens a fixed amount and stop early if a special token is encountered.
        while (removed < max_to_remove) {
            const llama_token next_token_id = tokens[n_ctx - removed - 1];
            if (is_special_token(next_token_id)) {
                break;
            }
            removed += 1;
        }
        for (int i = n_ctx - removed; i < n_ctx; i++) {
            prefix += llama_token_to_piece(ctx_main, tokens[i]);
        }
    }
    return {prefix, removed};
}

//
// Token healing (external)
//

llama_token_healing_output llama_token_healing_rollback(
                           const llama_context * ctx_main,
                           std::vector<llama_token> & tokens,
                           llama_token_healing_type th_type,
                           int max_to_remove) {
    // NB. To avoid returning empty `tokens`, at least 1 token will remain in `tokens` after rolling back.
    //     It is the caller's responsibility to add BOS to the start of the prompt if they want to roll back the whole prompt.
    llama_token_healing_output out = llama_token_healing_get_prefix(ctx_main, th_type, tokens, max_to_remove);

    // If constrained decoding would give back the original prompt, there is no need to modify the prompt.
    const bool is_multi_step = th_type == llama_token_healing_type::ROLLBACK_MULTI ||
                               th_type == llama_token_healing_type::DYNAMIC_MULTI;
    const std::vector<llama_token> candidates = token_healing_get_candidates(ctx_main, out.prefix, is_multi_step);
    LOG("token_healing: prefix = '%s' (%d tokens)\n", out.prefix.c_str(), out.n_tokens_removed);
    if (out.n_tokens_removed == 1 && candidates.size() == 1) {
        LOG("token_healing: nothing to heal\n");
        return {};
    }

    // Finally, trim prompt tokens
    tokens.resize(tokens.size() - out.n_tokens_removed);
    return out;
}

void llama_token_healing_set_prefix(llama_sampling_context * ctx_sampling, const std::string & prefix) {
    ctx_sampling->token_healing_prefix = prefix;
}

bool llama_token_healing_parse_params(const std::string & params, llama_token_healing_params & th_params) {
    th_params.enabled = true;
    th_params.n_rollback = -1;
    /**/ if (params    == "0" ) { th_params.enabled = false; }
    else if (params    == "1" ) { th_params.type = llama_token_healing_type::ROLLBACK_LAST; }
    else if (params    == "d1") { th_params.type = llama_token_healing_type::DYNAMIC_ONCE;  }
    else if (params    == "d" ) { th_params.type = llama_token_healing_type::DYNAMIC_MULTI; }
    else if (params[0] == 'r' ) {
        th_params.type = llama_token_healing_type::ROLLBACK_MULTI;
        th_params.n_rollback = std::stoi(params.substr(1));
        if (th_params.n_rollback <= 0) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

//
// Sampling
//

struct llama_sampling_context * llama_sampling_init(const struct llama_sampling_params & params) {
    struct llama_sampling_context * result = new llama_sampling_context();

    result->params  = params;
    result->grammar = nullptr;

    // if there is a grammar, parse it
    if (!params.grammar.empty()) {
        result->parsed_grammar = grammar_parser::parse(params.grammar.c_str());

        // will be empty (default) if there are parse errors
        if (result->parsed_grammar.rules.empty()) {
            fprintf(stderr, "%s: failed to parse grammar\n", __func__);
            delete result;
            return nullptr;
        }

        // Ensure that there is a "root" node.
        if (result->parsed_grammar.symbol_ids.find("root") == result->parsed_grammar.symbol_ids.end()) {
            fprintf(stderr, "%s: grammar does not contain a 'root' symbol\n", __func__);
            delete result;
            return nullptr;
        }

        std::vector<const llama_grammar_element *> grammar_rules(result->parsed_grammar.c_rules());

        struct llama_grammar * grammar = llama_grammar_init(
                grammar_rules.data(),
                grammar_rules.size(), result->parsed_grammar.symbol_ids.at("root"));
        if (grammar == nullptr) {
            throw std::runtime_error("Failed to initialize llama_grammar");
        }
        result->grammar = grammar;
    }

    result->prev.resize(params.n_prev);

    result->n_valid = 0;

    llama_sampling_set_rng_seed(result, params.seed);

    return result;
}

void llama_sampling_free(struct llama_sampling_context * ctx) {
    if (ctx->grammar != NULL) {
        llama_grammar_free(ctx->grammar);
    }

    delete ctx;
}

void llama_sampling_reset(llama_sampling_context * ctx) {
    if (ctx->grammar != NULL) {
        llama_grammar_free(ctx->grammar);
        ctx->grammar = NULL;
    }

    if (!ctx->parsed_grammar.rules.empty()) {
        std::vector<const llama_grammar_element *> grammar_rules(ctx->parsed_grammar.c_rules());

        struct llama_grammar * grammar = llama_grammar_init(
                grammar_rules.data(),
                grammar_rules.size(), ctx->parsed_grammar.symbol_ids.at("root"));
        if (grammar == nullptr) {
            throw std::runtime_error("Failed to initialize llama_grammar");
        }
        ctx->grammar = grammar;
    }

    ctx->token_healing_prefix.clear();

    std::fill(ctx->prev.begin(), ctx->prev.end(), 0);
    ctx->cur.clear();
    ctx->n_valid = 0;
}

void llama_sampling_set_rng_seed(struct llama_sampling_context * ctx, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = std::random_device{}();
    }
    ctx->rng.seed(seed);
}

void llama_sampling_cp(llama_sampling_context * src, llama_sampling_context * dst) {
    if (dst->grammar) {
        llama_grammar_free(dst->grammar);
        dst->grammar = nullptr;
    }

    if (src->grammar) {
        dst->grammar = llama_grammar_copy(src->grammar);
    }

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

std::string llama_sampling_print(const llama_sampling_params & params) {
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

std::string llama_sampling_order_print(const llama_sampling_params & params) {
    std::string result = "(Token healing) -> CFG -> Penalties ";
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
                   struct llama_context * ctx_main,
            const llama_sampling_params & params,
                 llama_token_data_array & cur_p,
                                 size_t   min_keep) {
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
            case llama_sampler_type::TOP_K    : llama_sample_top_k    (ctx_main, &cur_p, top_k,     min_keep); break;
            case llama_sampler_type::TFS_Z    : llama_sample_tail_free(ctx_main, &cur_p, tfs_z,     min_keep); break;
            case llama_sampler_type::TYPICAL_P: llama_sample_typical  (ctx_main, &cur_p, typical_p, min_keep); break;
            case llama_sampler_type::TOP_P    : llama_sample_top_p    (ctx_main, &cur_p, top_p,     min_keep); break;
            case llama_sampler_type::MIN_P    : llama_sample_min_p    (ctx_main, &cur_p, min_p,     min_keep); break;
            case llama_sampler_type::TEMPERATURE:
                if (dynatemp_range > 0) {
                    float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                    float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                    llama_sample_entropy(ctx_main, &cur_p, dynatemp_min, dynatemp_max, dynatemp_exponent);
                } else {
                    llama_sample_temp(ctx_main, &cur_p, temp);
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
    const llama_sampling_params & params = ctx_sampling->params;

    const float   temp            = params.temp;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;

    std::vector<float> original_logits;
    auto cur_p = llama_sampling_prepare(ctx_sampling, ctx_main, ctx_cfg, idx, /* apply_grammar= */ is_resampling, &original_logits);
    if (ctx_sampling->grammar != NULL && !is_resampling) {
        GGML_ASSERT(!original_logits.empty());
    }
    llama_token id = 0;

    if (temp < 0.0) {
        // greedy sampling, with probs
        llama_sample_softmax(ctx_main, &cur_p);
        id = cur_p.data[0].id;
    } else if (temp == 0.0) {
        // greedy sampling, no probs
        id = llama_sample_token_greedy(ctx_main, &cur_p);
    } else {
        if (mirostat == 1) {
            const int mirostat_m = 100;
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat(ctx_main, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &ctx_sampling->mirostat_mu);
        } else if (mirostat == 2) {
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat_v2(ctx_main, &cur_p, mirostat_tau, mirostat_eta, &ctx_sampling->mirostat_mu);
        } else {
            // temperature sampling
            size_t min_keep = std::max(1, params.min_keep);

            sampler_queue(ctx_main, params, cur_p, min_keep);

            id = llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);

            //{
            //    const int n_top = 10;
            //    LOG("top %d candidates:\n", n_top);

            //    for (int i = 0; i < n_top; i++) {
            //        const llama_token id = cur_p.data[i].id;
            //        (void)id; // To avoid a warning that id is unused when logging is disabled.
            //        LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(ctx_main, id).c_str(), cur_p.data[i].p);
            //    }
            //}

            //LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx_main, id).c_str());
        }
    }

    if (ctx_sampling->grammar != NULL && !is_resampling) {
        // Get a pointer to the logits
        float * logits = llama_get_logits_ith(ctx_main, idx);

        // Create an array with a single token data element for the sampled id
        llama_token_data single_token_data = {id, logits[id], 0.0f};
        llama_token_data_array single_token_data_array = { &single_token_data, 1, false };

        // Apply grammar constraints to the single token
        llama_sample_grammar(ctx_main, &single_token_data_array, ctx_sampling->grammar);

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
    const llama_sampling_params & params = ctx_sampling->params;

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

    if (ctx_sampling->grammar != NULL && !apply_grammar) {
        GGML_ASSERT(original_logits != NULL);
        // Only make a copy of the original logits if we are not applying grammar checks, not sure if I actually have to do this.
        *original_logits = {logits, logits + n_vocab};
    }

    // apply params.logit_bias map
    for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
        logits[it->first] += it->second;
    }

    if (ctx_cfg) {
        float * logits_guidance = llama_get_logits_ith(ctx_cfg, idx);
        llama_sample_apply_guidance(ctx_main, logits, logits_guidance, params.cfg_scale);
    }

    // Constrain tokens based on the remaining token healing prefix (if any)
    const auto & th_prefix = ctx_sampling->token_healing_prefix;
    if (params.token_healing.enabled && !th_prefix.empty()) {
        const bool is_multi_step = params.token_healing.type == llama_token_healing_type::ROLLBACK_MULTI ||
                                   params.token_healing.type == llama_token_healing_type::DYNAMIC_MULTI;
        std::vector<llama_token> th_candidates = token_healing_get_candidates(ctx_main, th_prefix, is_multi_step);

        LOG("token_healing: prefix = '%s'\n", th_prefix.c_str());
        for (const llama_token token_id : th_candidates) {
            LOG(" [%6d] '%s'\n", token_id, llama_token_to_piece(ctx_main, token_id).c_str());
        }

        // N.B. We could also set token constraints by setting rejected tokens' logits to -inf
        cur.clear();
        for (const llama_token token_id : th_candidates) {
            cur.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }
    } else {
        cur.resize(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), false };

    // apply penalties
    const auto& penalty_tokens = params.use_penalty_prompt_tokens ? params.penalty_prompt_tokens : prev;
    const int penalty_tokens_used_size = std::min((int)penalty_tokens.size(), penalty_last_n);
    if (penalty_tokens_used_size) {
        const float nl_logit = logits[llama_token_nl(llama_get_model(ctx_main))];

        llama_sample_repetition_penalties(ctx_main, &cur_p,
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
    if (apply_grammar && ctx_sampling->grammar != NULL) {
        llama_sample_grammar(ctx_main, &cur_p, ctx_sampling->grammar);
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
    return llama_sampling_prepare_impl(ctx_sampling,ctx_main, ctx_cfg, idx, apply_grammar, original_logits);
}

void llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar) {
    ctx_sampling->prev.erase(ctx_sampling->prev.begin());
    ctx_sampling->prev.push_back(id);

    if (ctx_sampling->grammar != NULL && apply_grammar) {
        llama_grammar_accept_token(ctx_main, ctx_sampling->grammar, id);
    }

    if (ctx_sampling->params.token_healing.enabled && apply_grammar) {
        std::string & th_prefix = ctx_sampling->token_healing_prefix;
        if (!th_prefix.empty()) {
            const std::string new_token_piece = llama_token_to_piece(ctx_main, id);
            if (new_token_piece.size() < th_prefix.size()) {
                // Shift prefix constraint (for multi step token healing)
                th_prefix = th_prefix.substr(new_token_piece.size());
            } else {
                // Prefix has been generated => no more constrained generation
                th_prefix.clear();
                LOG("token_healing: done\n");
            }
        }
    }
}
