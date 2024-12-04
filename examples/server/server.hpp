#pragma once

#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

#include <string>
#include <memory>
#include <unordered_set>

using json = nlohmann::ordered_json;

// cast a shared_ptr to a specific type using copy constructor
#define copy_cast_ptr(TYPEOUT, ptr) *(static_cast<TYPEOUT*>(ptr.get()));

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

// state diagram: https://github.com/ggerganov/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED, // TODO: this state is only used for setting up the initial prompt processing; maybe merge it with launch_slot_with_task in the future
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

enum server_task_type {
    SERVER_TASK_TYPE_INFERENCE,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
};

enum server_task_inf_type {
    SERVER_TASK_INF_TYPE_COMPLETION,
    SERVER_TASK_INF_TYPE_EMBEDDING,
    SERVER_TASK_INF_TYPE_RERANK,
    SERVER_TASK_INF_TYPE_INFILL,
};

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
};

enum result_type {
    RESULT_TYPE_CMPL_FINAL,
    RESULT_TYPE_CMPL_PARTIAL,
    RESULT_TYPE_EMBD,
    RESULT_TYPE_RERANK,
    RESULT_TYPE_METRICS,
    RESULT_TYPE_SLOT_SAVE_LOAD,
    RESULT_TYPE_SLOT_ERASE,
    RESULT_TYPE_APPLY_LORA,
    RESULT_TYPE_ERROR,
    RESULT_TYPE_UNKNOWN, // will throw an error
};

struct server_task {
    int id        = -1; // to be filled by server_queue
    int id_target = -1; // used by SERVER_TASK_TYPE_CANCEL

    llama_tokens prompt_tokens;
    server_task_type type;

    // TODO @ngxson : we should get rid of json type here
    json data;

    server_task_inf_type inf_type = SERVER_TASK_INF_TYPE_COMPLETION;

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
        }
        return ids;
    }
};

struct slot_params {
    bool stream       = true;
    bool cache_prompt = true; // remember the prompt to avoid reprocessing all prompt

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // mininum line indentation for the generated text in number of whitespace characters

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::vector<std::string> antiprompt;
    bool timings_per_token = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // params only used in to_json()
    int32_t n_ctx;
    uint32_t seed_cur;
    bool can_speculative;

    json to_json() {
        std::vector<std::string> samplers;
        samplers.reserve(sampling.samplers.size());
        for (const auto & sampler : sampling.samplers) {
            samplers.emplace_back(common_sampler_type_to_str(sampler));
        }

        return json {
            {"n_ctx",                     n_ctx},
            {"n_predict",                 n_predict},     // Server configured n_predict
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"penalize_nl",               sampling.penalize_nl},
            {"stop",                      antiprompt},
            {"max_tokens",                n_predict}, // User configured n_predict
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            //{"logit_bias",                sampling.logit_bias},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"grammar",                   sampling.grammar},
            {"samplers",                  samplers},
            {"speculative",               can_speculative},
            {"speculative.n_max",         speculative.n_max},
            {"speculative.n_min",         speculative.n_min},
            {"speculative.p_min",         speculative.p_min},
            {"timings_per_token",         timings_per_token},
        };
    }
};

struct result_timings {
    int32_t prompt_n = -1;
    double prompt_ms;
    double prompt_per_token_ms;
    double prompt_per_second;

    int32_t predicted_n = -1;
    double predicted_ms;
    double predicted_per_token_ms;
    double predicted_per_second;

    json to_json() {
        return {
            {"prompt_n",               prompt_n},
            {"prompt_ms",              prompt_ms},
            {"prompt_per_token_ms",    prompt_per_token_ms},
            {"prompt_per_second",      prompt_per_second},

            {"predicted_n",            predicted_n},
            {"predicted_ms",           predicted_ms},
            {"predicted_per_token_ms", predicted_per_token_ms},
            {"predicted_per_second",   predicted_per_second},
        };
    }
};

struct server_task_result {
    result_type type = RESULT_TYPE_UNKNOWN;
    int id           = -1;
    int id_slot      = -1;
    server_task_result() = default;
    server_task_result(result_type type) : type(type) {}
    virtual ~server_task_result() = default;
};

inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

struct completion_token_output {
    llama_token tok;
    std::string text_to_send;
    struct token_prob {
        llama_token tok;
        float prob;
    };
    std::vector<token_prob> probs;
};

struct server_task_result_cmpl_final : server_task_result {
    server_task_result_cmpl_final() : server_task_result(RESULT_TYPE_CMPL_FINAL) {}
    int index = 0;
    std::string content;
    bool stream;
    result_timings timings;
    std::string model_alias;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_tokens_cached;
    int32_t has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    std::vector<completion_token_output> probs_output;

    slot_params generation_params;

    json to_json() {
        // non-OAI-compat JSON
        return json {
            {"index",               index},
            {"content",             content},
            {"id_slot",             id_slot},
            {"stop",                true},
            {"model",               model_alias},
            {"tokens_predicted",    n_decoded},
            {"tokens_evaluated",    n_prompt_tokens},
            {"generation_settings", generation_params.to_json()},
            {"prompt",              prompt},
            {"has_new_line",        has_new_line},
            {"truncated",           truncated},
            {"stop_type",           stop_type_to_str(stop)},
            {"stopping_word",       stopping_word},
            {"tokens_cached",       n_tokens_cached},
            {"timings",             timings.to_json()},
        };
    }

    static server_task_result_cmpl_final from_ptr(std::unique_ptr<server_task_result> & result_ptr) {
        return copy_cast_ptr(server_task_result_cmpl_final, result_ptr);
    }

    virtual ~server_task_result_cmpl_final() = default;
};

struct server_task_result_cmpl_partial : server_task_result {
    server_task_result_cmpl_partial() : server_task_result(RESULT_TYPE_CMPL_PARTIAL) {}
    int index = 0;
    std::string content;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;

    stop_type stop = STOP_TYPE_NONE;

    std::vector<completion_token_output> probs_output;
    result_timings timings;

    json to_json() {
        bool is_stop = stop != STOP_TYPE_NONE;
        // non-OAI-compat JSON
        json res = json {
            {"index",            index},
            {"content",          content},
            {"stop_type",        stop_type_to_str(stop)},
            {"stop",             is_stop},
            {"id_slot",          id_slot},
            {"tokens_predicted", n_decoded},
            {"tokens_evaluated", n_prompt_tokens},
        };
        // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
        if (timings.prompt_n > 0) {
            res.push_back({"timings", timings.to_json()});
        }
        if (is_stop) {
            res.push_back({"truncated", truncated});
        }
        return res;
    }

    static server_task_result_cmpl_partial from_ptr(std::unique_ptr<server_task_result> & result_ptr) {
        return copy_cast_ptr(server_task_result_cmpl_partial, result_ptr);
    }

    virtual ~server_task_result_cmpl_partial() = default;
};

struct server_task_result_embd : server_task_result {
    server_task_result_embd() : server_task_result(RESULT_TYPE_EMBD) {}
    result_type type = RESULT_TYPE_EMBD;
    int index = 0;
    std::vector<float> embedding;

    json to_json() {
        return json {
            {"index",     index},
            {"embedding", embedding},
        };
    }

    static server_task_result_embd from_ptr(std::unique_ptr<server_task_result> & result_ptr) {
        return copy_cast_ptr(server_task_result_embd, result_ptr);
    }

    virtual ~server_task_result_embd() = default;
};

struct server_task_result_rerank : server_task_result {
    server_task_result_rerank() : server_task_result(RESULT_TYPE_RERANK) {}
    int index = 0;
    float score = -1e6;

    json to_json() {
        return json {
            {"index", index},
            {"score", score},
        };
    }

    static server_task_result_rerank from_ptr(std::unique_ptr<server_task_result> & result_ptr) {
        return copy_cast_ptr(server_task_result_rerank, result_ptr);
    }

    virtual ~server_task_result_rerank() = default;
};

struct server_task_result_error : server_task_result {
    server_task_result_error() : server_task_result(RESULT_TYPE_ERROR) {}
    int index = 0;
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    static server_task_result_error from_ptr(std::unique_ptr<server_task_result> & result_ptr) {
        return copy_cast_ptr(server_task_result_error, result_ptr);
    }

    virtual ~server_task_result_error() = default;
};

struct server_task_result_metrics : server_task_result {
    server_task_result_metrics() : server_task_result(RESULT_TYPE_METRICS) {}
    int n_idle_slots;
    int n_processing_slots;
    int n_tasks_deferred;
    int64_t t_start;

    int32_t kv_cache_tokens_count;
    int32_t kv_cache_used_cells;

    // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    // TODO: get rid of this json object and use to_json() instead
    json slots_data = json::array();

    json to_json() {
        return json {
            { "idle",                            n_idle_slots },
            { "processing",                      n_processing_slots },
            { "deferred",                        n_tasks_deferred },
            { "t_start",                         t_start },

            { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
            { "t_tokens_generation_total",       t_tokens_generation_total },
            { "n_tokens_predicted_total",        n_tokens_predicted_total },
            { "t_prompt_processing_total",       t_prompt_processing_total },

            { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
            { "t_prompt_processing",             t_prompt_processing },
            { "n_tokens_predicted",              n_tokens_predicted },
            { "t_tokens_generation",             t_tokens_generation },

            { "n_decode_total",                  n_decode_total },
            { "n_busy_slots_total",              n_busy_slots_total },

            { "kv_cache_tokens_count",           kv_cache_tokens_count },
            { "kv_cache_used_cells",             kv_cache_used_cells },

            { "slots",                           slots_data },
        };
    }

    static server_task_result_metrics from_ptr(std::unique_ptr<server_task_result> & result_ptr) {
        return copy_cast_ptr(server_task_result_metrics, result_ptr);
    }

    virtual ~server_task_result_metrics() = default;
};

struct server_task_result_slot_save_load : server_task_result {
    server_task_result_slot_save_load() : server_task_result(RESULT_TYPE_SLOT_SAVE_LOAD) {}
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    json to_json() {
        if (is_save) {
            return json {
                { "id_slot",   id_slot },
                { "filename",  filename },
                { "n_saved",   n_tokens },
                { "n_written", n_bytes },
                { "timings", {
                    { "save_ms", t_ms }
                }},
            };
        } else {
            return json {
                { "id_slot",    id_slot },
                { "filename",   filename },
                { "n_restored", n_tokens },
                { "n_read",     n_bytes },
                { "timings", {
                    { "restore_ms", t_ms }
                }},
            };
        }
    }

    static server_task_result_slot_save_load from_ptr(std::unique_ptr<server_task_result> & result_ptr) {
        return copy_cast_ptr(server_task_result_slot_save_load, result_ptr);
    }

    virtual ~server_task_result_slot_save_load() = default;
};

struct server_task_result_slot_erase : server_task_result {
    server_task_result_slot_erase() : server_task_result(RESULT_TYPE_SLOT_ERASE) {}
    size_t n_erased;

    json to_json() {
        return json {
            { "id_slot",  id_slot },
            { "n_erased", n_erased },
        };
    }

    static server_task_result_slot_erase from_ptr(std::unique_ptr<server_task_result> & result_ptr) {
        return copy_cast_ptr(server_task_result_slot_erase, result_ptr);
    }

    virtual ~server_task_result_slot_erase() = default;
};

struct server_task_result_apply_lora : server_task_result {
    server_task_result_apply_lora() : server_task_result(RESULT_TYPE_APPLY_LORA) {}
    json to_json() {
        return json {{ "success", true }};
    }

    static server_task_result_apply_lora from_ptr(std::unique_ptr<server_task_result> & result_ptr) {
        return copy_cast_ptr(server_task_result_apply_lora, result_ptr);
    }
};
