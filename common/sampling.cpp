#include "sampling.h"

#include "common.h"

std::string gpt_sampling_params::print_all() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\ttop_k = %d, tfs_z = %.3f, top_p = %.3f, min_p = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            top_k, tfs_z, top_p, min_p, typ_p, temp,
            mirostat, mirostat_eta, mirostat_tau);

    return std::string(result);
}

std::string gpt_sampling_params::print_samplers() const {
    std::string result = "CFG -> Penalties ";
    if (mirostat == 0) {
        for (const auto & sampler : samplers) {
            const auto name = llama_sampling_type_to_str(sampler);
            if (!name.empty()) {
                result += "-> " + name + " ";
            }
        }
    } else {
        result += "-> mirostat ";
    }

    return result;
}

struct llama_sampling * llama_sampling_init(const struct llama_model * model, const struct gpt_sampling_params & params) {
    llama_sampling_params lparams = llama_sampling_default_params();

    lparams.seed              = params.seed;
    lparams.n_prev            = params.n_prev;
    lparams.n_probs           = params.n_probs;
    lparams.min_keep          = params.min_keep;
    lparams.top_k             = params.top_k;
    lparams.top_p             = params.top_p;
    lparams.min_p             = params.min_p;
    lparams.tfs_z             = params.tfs_z;
    lparams.typ_p             = params.typ_p;
    lparams.temp              = params.temp;
    lparams.dynatemp_range    = params.dynatemp_range;
    lparams.dynatemp_exponent = params.dynatemp_exponent;
    lparams.penalty_last_n    = params.penalty_last_n;
    lparams.penalty_repeat    = params.penalty_repeat;
    lparams.penalty_freq      = params.penalty_freq;
    lparams.penalty_present   = params.penalty_present;
    lparams.mirostat          = params.mirostat;
    lparams.mirostat_tau      = params.mirostat_tau;
    lparams.mirostat_eta      = params.mirostat_eta;
    lparams.penalize_nl       = params.penalize_nl;
    lparams.ignore_eos        = params.ignore_eos;

    lparams.n_samplers = params.samplers.size();
    for (int i = 0; i < lparams.n_samplers; i++) {
        lparams.samplers[i] = params.samplers[i];
    }

    struct llama_sampling * result = llama_sampling_init(model, lparams);

    llama_sampling_set_grammar   (result, params.grammar.c_str(), "root");
    llama_sampling_set_logit_bias(result, params.logit_bias.size(), params.logit_bias.data());

    return result;
}

void llama_sampling_cp(llama_sampling * src, llama_sampling *& dst) {
    if (dst) {
        llama_sampling_free(dst);
    }

    dst = llama_sampling_cp(src);
}

llama_token llama_sampling_sample(
        struct llama_sampling * smpl,
        struct llama_context * ctx,
        int idx) {
    llama_sampling_set_logits(smpl, llama_get_logits_ith(ctx, idx));

    // first, sample the token without any grammar constraints
    const llama_token id = llama_sampling_sample(smpl, nullptr);

    // create an array with a single token data element for the sampled id
    llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
    llama_token_data_array single_token_data_array = { &single_token_data, 1, false };

    llama_sampling_grammar(smpl, &single_token_data_array);

    // check if the token is valid according to the grammar by seeing if its logit has been set to -INFINITY
    const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
    if (is_valid) {
        return id;
    }

    // if the token is not valid, sample again, after applying the grammar constraints
    llama_sampling_set_logits(smpl, llama_get_logits_ith(ctx, idx));

    llama_sampling_grammar(smpl, nullptr);

    return llama_sampling_sample(smpl, nullptr);
}

std::string llama_sampling_prev_str(llama_sampling * smpl, llama_context * ctx_main, int n) {
    n = std::min(n, llama_sampling_n_prev(smpl));

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = llama_sampling_prev(smpl, i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += llama_token_to_piece(ctx_main, id);
    }

    return result;
}

char llama_sampling_type_to_chr(llama_constraint_type sampler) {
    switch (sampler) {
        case LLAMA_CONSTRAINT_TYPE_TOP_K:       return 'k';
        case LLAMA_CONSTRAINT_TYPE_TFS_Z:       return 'f';
        case LLAMA_CONSTRAINT_TYPE_TYPICAL_P:   return 'y';
        case LLAMA_CONSTRAINT_TYPE_TOP_P:       return 'p';
        case LLAMA_CONSTRAINT_TYPE_MIN_P:       return 'm';
        case LLAMA_CONSTRAINT_TYPE_TEMPERATURE: return 't';
        default : return '?';
    }
}

std::string llama_sampling_type_to_str(llama_constraint_type sampler) {
    switch (sampler) {
        case LLAMA_CONSTRAINT_TYPE_TOP_K:       return "top_k";
        case LLAMA_CONSTRAINT_TYPE_TFS_Z:       return "tfs_z";
        case LLAMA_CONSTRAINT_TYPE_TYPICAL_P:   return "typ_p";
        case LLAMA_CONSTRAINT_TYPE_TOP_P:       return "top_p";
        case LLAMA_CONSTRAINT_TYPE_MIN_P:       return "min_p";
        case LLAMA_CONSTRAINT_TYPE_TEMPERATURE: return "temperature";
        default : return "";
    }
}

std::vector<llama_constraint_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, llama_constraint_type> sampler_canonical_name_map {
        { "top_k",       LLAMA_CONSTRAINT_TYPE_TOP_K },
        { "top_p",       LLAMA_CONSTRAINT_TYPE_TOP_P },
        { "typ_p",       LLAMA_CONSTRAINT_TYPE_TYPICAL_P },
        { "min_p",       LLAMA_CONSTRAINT_TYPE_MIN_P },
        { "tfs_z",       LLAMA_CONSTRAINT_TYPE_TFS_Z },
        { "temperature", LLAMA_CONSTRAINT_TYPE_TEMPERATURE },
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, llama_constraint_type> sampler_alt_name_map {
        { "top-k",       LLAMA_CONSTRAINT_TYPE_TOP_K },
        { "top-p",       LLAMA_CONSTRAINT_TYPE_TOP_P },
        { "nucleus",     LLAMA_CONSTRAINT_TYPE_TOP_P },
        { "typical-p",   LLAMA_CONSTRAINT_TYPE_TYPICAL_P },
        { "typical",     LLAMA_CONSTRAINT_TYPE_TYPICAL_P },
        { "typ-p",       LLAMA_CONSTRAINT_TYPE_TYPICAL_P },
        { "typ",         LLAMA_CONSTRAINT_TYPE_TYPICAL_P },
        { "min-p",       LLAMA_CONSTRAINT_TYPE_MIN_P },
        { "tfs-z",       LLAMA_CONSTRAINT_TYPE_TFS_Z },
        { "tfs",         LLAMA_CONSTRAINT_TYPE_TFS_Z },
        { "temp",        LLAMA_CONSTRAINT_TYPE_TEMPERATURE },
    };

    std::vector<llama_constraint_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        auto sampler = sampler_canonical_name_map.find(name);
        if (sampler != sampler_canonical_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            if (allow_alt_names) {
                sampler = sampler_alt_name_map.find(name);
                if (sampler != sampler_alt_name_map.end()) {
                    samplers.push_back(sampler->second);
                }
            }
        }
    }

    return samplers;
}

std::vector<llama_constraint_type> llama_sampling_types_from_chars(const std::string & chars) {
    std::unordered_map<char, llama_constraint_type> sampler_name_map {
        { llama_sampling_type_to_chr(LLAMA_CONSTRAINT_TYPE_TOP_K),       LLAMA_CONSTRAINT_TYPE_TOP_K },
        { llama_sampling_type_to_chr(LLAMA_CONSTRAINT_TYPE_TFS_Z),       LLAMA_CONSTRAINT_TYPE_TFS_Z },
        { llama_sampling_type_to_chr(LLAMA_CONSTRAINT_TYPE_TYPICAL_P),   LLAMA_CONSTRAINT_TYPE_TYPICAL_P },
        { llama_sampling_type_to_chr(LLAMA_CONSTRAINT_TYPE_TOP_P),       LLAMA_CONSTRAINT_TYPE_TOP_P },
        { llama_sampling_type_to_chr(LLAMA_CONSTRAINT_TYPE_MIN_P),       LLAMA_CONSTRAINT_TYPE_MIN_P },
        { llama_sampling_type_to_chr(LLAMA_CONSTRAINT_TYPE_TEMPERATURE), LLAMA_CONSTRAINT_TYPE_TEMPERATURE }
    };

    std::vector<llama_constraint_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        }
    }

    return samplers;
}
