#include "sampling.h"

#include "common.h"

#include <cmath>
#include <unordered_map>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with jarvis-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_sampler_params params;

    struct jarvis_sampler * grmr;
    struct jarvis_sampler * chain;

    ring_buffer<jarvis_token> prev;

    std::vector<jarvis_token_data> cur;

    jarvis_token_data_array cur_p;

    void set_logits(struct jarvis_context * ctx, int idx) {
        const auto * logits = jarvis_get_logits_ith(ctx, idx);

        const int n_vocab = jarvis_n_vocab(jarvis_get_model(ctx));

        cur.resize(n_vocab);

        for (jarvis_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = jarvis_token_data{token_id, logits[token_id], 0.0f};
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }
};

std::string common_sampler_params::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, tfs_z = %.3f, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, tfs_z, top_p, min_p, xtc_probability, xtc_threshold, typ_p, temp,
            mirostat, mirostat_eta, mirostat_tau);

    return std::string(result);
}

struct common_sampler * common_sampler_init(const struct jarvis_model * model, const struct common_sampler_params & params) {
    jarvis_sampler_chain_params lparams = jarvis_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    auto * result = new common_sampler {
        /* .params = */ params,
        /* .grmr   = */ jarvis_sampler_init_grammar(model, params.grammar.c_str(), "root"),
        /* .chain  = */ jarvis_sampler_chain_init(lparams),
        /* .prev   = */ ring_buffer<jarvis_token>(std::max(32, params.n_prev)),
        /* .cur    = */ {},
        /* .cur_p  = */ {},
    };

    jarvis_sampler_chain_add(result->chain,
            jarvis_sampler_init_logit_bias(
                jarvis_n_vocab(model),
                params.logit_bias.size(),
                params.logit_bias.data()));

    jarvis_sampler_chain_add(result->chain,
            jarvis_sampler_init_penalties(
                jarvis_n_vocab  (model),
                jarvis_token_eos(model),
                jarvis_token_nl (model),
                params.penalty_last_n,
                params.penalty_repeat,
                params.penalty_freq,
                params.penalty_present,
                params.penalize_nl,
                params.ignore_eos));

    if (params.mirostat == 0) {
        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                    case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char*> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto& str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }

                        jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_dry      (model, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                        break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_top_k    (params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_top_p    (params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_min_p    (params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_xtc      (params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TFS_Z:
                    jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_tail_free(params.tfs_z, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_typical  (params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_temp_ext (params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_infill   (model));
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_dist(params.seed));
    } else if (params.mirostat == 1) {
        jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_temp(params.temp));
        jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_mirostat(jarvis_n_vocab(model), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_temp(params.temp));
        jarvis_sampler_chain_add(result->chain, jarvis_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (gsmpl) {
        jarvis_sampler_free(gsmpl->grmr);

        jarvis_sampler_free(gsmpl->chain);

        delete gsmpl;
    }
}

void common_sampler_accept(struct common_sampler * gsmpl, jarvis_token token, bool accept_grammar) {
    if (accept_grammar) {
        jarvis_sampler_accept(gsmpl->grmr, token);
    }

    jarvis_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    jarvis_sampler_reset(gsmpl->grmr);

    jarvis_sampler_reset(gsmpl->chain);
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params = */ gsmpl->params,
        /* .grmr   = */ jarvis_sampler_clone(gsmpl->grmr),
        /* .chain  = */ jarvis_sampler_clone(gsmpl->chain),
        /* .prev   = */ gsmpl->prev,
        /* .cur    = */ gsmpl->cur,
        /* .cur_p  = */ gsmpl->cur_p,
    };
}

void common_perf_print(const struct jarvis_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    if (gsmpl) {
        jarvis_perf_sampler_print(gsmpl->chain);
    }
    if (ctx) {
        jarvis_perf_context_print(ctx);
    }
}

jarvis_token common_sampler_sample(struct common_sampler * gsmpl, struct jarvis_context * ctx, int idx, bool grammar_first) {
    gsmpl->set_logits(ctx, idx);

    auto & grmr  = gsmpl->grmr;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    if (grammar_first) {
        jarvis_sampler_apply(grmr, &cur_p);
    }

    jarvis_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    const jarvis_token id = cur_p.data[cur_p.selected].id;

    if (grammar_first) {
        return id;
    }

    // check if it the sampled token fits the grammar
    {
        jarvis_token_data       single_token_data       = { id, 1.0f, 0.0f };
        jarvis_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        jarvis_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    jarvis_sampler_apply(grmr,  &cur_p);
    jarvis_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during re-sampling - check your sampling configuration");

    return cur_p.data[cur_p.selected].id;
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return jarvis_sampler_get_seed(gsmpl->chain);
}

// helpers

jarvis_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl) {
    return &gsmpl->cur_p;
}

jarvis_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < jarvis_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = jarvis_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ") + jarvis_sampler_name(smpl) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, jarvis_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const jarvis_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != JARVIS_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TFS_Z:       return 'f';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TFS_Z:       return "tfs_z";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, common_sampler_type> sampler_canonical_name_map {
        { "dry",         COMMON_SAMPLER_TYPE_DRY },
        { "top_k",       COMMON_SAMPLER_TYPE_TOP_K },
        { "top_p",       COMMON_SAMPLER_TYPE_TOP_P },
        { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "min_p",       COMMON_SAMPLER_TYPE_MIN_P },
        { "tfs_z",       COMMON_SAMPLER_TYPE_TFS_Z },
        { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
        { "xtc",         COMMON_SAMPLER_TYPE_XTC },
        { "infill",      COMMON_SAMPLER_TYPE_INFILL },
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, common_sampler_type> sampler_alt_name_map {
        { "top-k",       COMMON_SAMPLER_TYPE_TOP_K },
        { "top-p",       COMMON_SAMPLER_TYPE_TOP_P },
        { "nucleus",     COMMON_SAMPLER_TYPE_TOP_P },
        { "typical-p",   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typical",     COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typ-p",       COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typ",         COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "min-p",       COMMON_SAMPLER_TYPE_MIN_P },
        { "tfs-z",       COMMON_SAMPLER_TYPE_TFS_Z },
        { "tfs",         COMMON_SAMPLER_TYPE_TFS_Z },
        { "temp",        COMMON_SAMPLER_TYPE_TEMPERATURE },
    };

    std::vector<common_sampler_type> samplers;
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

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TFS_Z),       COMMON_SAMPLER_TYPE_TFS_Z },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        }
    }

    return samplers;
}
