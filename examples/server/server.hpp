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

struct result_timings {
    int32_t prompt_n;
    double prompt_ms;
    double prompt_per_token_ms;
    double prompt_per_second;

    int32_t predicted_n;
    double predicted_ms;
    double predicted_per_token_ms;
    double predicted_per_second;
};

enum result_type {
    RESULT_TYPE_CMPL_FINAL,
    RESULT_TYPE_CMPL_PARTIAL,
    RESULT_TYPE_EMBD,
    RESULT_TYPE_RERANK,
    RESULT_TYPE_ERROR,
    RESULT_TYPE_UNKNOWN, // will throw an error
};

struct server_task_result {
    result_type type = RESULT_TYPE_UNKNOWN;
    int id           = -1;
    int id_slot      = -1;
};

struct server_task_result_cmpl_final : server_task_result {
    result_type type = RESULT_TYPE_CMPL_FINAL;
    int index = 0;
    std::string content;
    bool stream;
    bool timings_per_token;
    result_timings timings;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t has_new_line;
    int32_t stopping_word;
    int32_t n_tokens_cached;
    stop_type stop = STOP_TYPE_NONE;
    std::vector<completion_token_output> probs_output;

    slot_params params;
};

struct completion_token_output {
    llama_token tok;
    std::string text_to_send;
    struct token_prob {
        llama_token tok;
        float prob;
    };
    std::vector<token_prob> probs;
};

struct server_task_result_cmpl_partial : server_task_result {
    result_type type = RESULT_TYPE_CMPL_PARTIAL;
    int index = 0;
    std::string content;
    stop_type stop = STOP_TYPE_NONE;
    std::vector<completion_token_output> probs_output;
    result_timings timings;
};

struct server_task_result_embd : server_task_result {
    result_type type = RESULT_TYPE_EMBD;
    int index = 0;
    std::vector<float> embedding;
};

struct server_task_result_rerank : server_task_result {
    result_type type = RESULT_TYPE_RERANK;
    int index = 0;
    float score;
};

struct server_task_result_error : server_task_result {
    result_type type = RESULT_TYPE_ERROR;
    int index = 0;
    error_type err_type;
    std::string err_msg;
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
};
