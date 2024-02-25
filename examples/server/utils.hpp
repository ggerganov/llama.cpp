#pragma once

#include <string>
#include <vector>
#include <set>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

#include "json.hpp"

#include "../llava/clip.h"

using json = nlohmann::json;

extern bool server_verbose;
extern bool server_log_json;

#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
#endif

#if SERVER_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                            \
    do                                                                   \
    {                                                                    \
        if (server_verbose)                                              \
        {                                                                \
            server_log("VERB", __func__, __LINE__, MSG, __VA_ARGS__); \
        }                                                                \
    } while (0)
#endif

#define LOG_ERROR(  MSG, ...) server_log("ERR",  __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log("WARN", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_INFO(   MSG, ...) server_log("INFO", __func__, __LINE__, MSG, __VA_ARGS__)

static inline void server_log(const char *level, const char *function, int line, const char *message, const nlohmann::ordered_json &extra)
{
    std::stringstream ss_tid;
    ss_tid << std::this_thread::get_id();
    json log = nlohmann::ordered_json{
        {"tid", ss_tid.str()},
        {"timestamp", time(nullptr)},
    };

    if (server_log_json) {
        log.merge_patch(
                {
                        {"level",     level},
                        {"function",  function},
                        {"line",      line},
                        {"msg",       message},
                });
        if (!extra.empty()) {
            log.merge_patch(extra);
        }

        std::cout << log.dump(-1, ' ', false, json::error_handler_t::replace) << "\n" << std::flush;
    } else {
        char buf[1024];
        snprintf(buf, 1024, "%4s [%24s] %s", level, function, message);

        if (!extra.empty()) {
            log.merge_patch(extra);
        }
        std::stringstream ss;
        ss << buf << " |";
        for (const auto& el : log.items())
        {
            const std::string value = el.value().dump(-1, ' ', false, json::error_handler_t::replace);
            snprintf(buf, 1024, " %s=%s", el.key().c_str(), value.c_str());
            ss << buf;
        }

        const std::string str = ss.str();
        printf("%.*s\n", (int)str.size(), str.data());
        fflush(stdout);
    }
}

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
    SERVER_STATE_ERROR           // An error occurred, load_model failed
};

enum task_type {
    TASK_TYPE_COMPLETION,
    TASK_TYPE_CANCEL,
    TASK_TYPE_NEXT_RESPONSE,
    TASK_TYPE_METRICS
};

struct task_server {
    int id = -1; // to be filled by llama_server_queue
    int target_id;
    task_type type;
    json data;
    bool infill_mode = false;
    bool embedding_mode = false;
    int multitask_id = -1;
};

struct task_result {
    int id;
    int multitask_id = -1;
    bool stop;
    bool error;
    json result_json;
};

struct task_multi {
    int id;
    std::set<int> subtasks_remaining{};
    std::vector<task_result> results{};
};

// TODO: can become bool if we can't find use of more states
enum slot_state
{
    IDLE,
    PROCESSING,
};

enum slot_command
{
    NONE,
    LOAD_PROMPT,
    RELEASE,
};

struct slot_params
{
    bool stream       = true;
    bool cache_prompt = false; // remember the prompt to avoid reprocessing all prompt

    uint32_t seed      = -1; // RNG seed
    int32_t  n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t  n_predict = -1; // new tokens to predict

    std::vector<std::string> antiprompt;

    json input_prefix;
    json input_suffix;
};

struct slot_image
{
    int32_t id;

    bool request_encode_image = false;
    float * image_embedding = nullptr;
    int32_t image_tokens = 0;

    clip_image_u8 * img_data;

    std::string prefix_prompt; // before of this image
};

// completion token output with probabilities
struct completion_token_output
{
    struct token_prob
    {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
    llama_token tok;
    std::string text_to_send;
};

struct llama_client_slot
{
    int id;
    int task_id = -1;

    struct slot_params params;

    slot_state state = IDLE;
    slot_command command = NONE;

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_past      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;
    int32_t n_predict   = -1;

    int32_t num_prompt_tokens           = 0;
    int32_t num_prompt_tokens_processed = 0;

    json prompt;
    std::string generated_text;
    llama_token sampled;
    std::vector<llama_token> cache_tokens;
    std::vector<completion_token_output> generated_token_probs;

    bool infill = false;
    bool embedding = false;
    bool has_next_token = true;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;

    bool oaicompat = false;
    std::string oaicompat_model;

    std::string stopping_word;

    // sampling
    struct llama_sampling_params sparams;
    llama_sampling_context *ctx_sampling = nullptr;

    int32_t ga_i = 0;   // group-attention state
    int32_t ga_n = 1;   // group-attention factor
    int32_t ga_w = 512; // group-attention width

    int32_t n_past_se = 0; // self-extend

    // multimodal
    std::vector<slot_image> images;

    // stats
    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    int64_t t_start_process_prompt;
    int64_t t_start_genereration;

    double t_prompt_processing; // ms
    double t_token_generation; // ms

    // multitasks
    int multitask_id = -1;

    void reset() {
        num_prompt_tokens      = 0;
        generated_text         = "";
        truncated              = false;
        stopped_eos            = false;
        stopped_word           = false;
        stopped_limit          = false;
        stopping_word          = "";
        n_past                 = 0;
        sent_count             = 0;
        sent_token_probs_index = 0;
        infill                 = false;
        ga_i                   = 0;
        n_past_se              = 0;

        generated_token_probs.clear();

        for (slot_image & img : images)
        {
            free(img.image_embedding);
            if (img.img_data) {
                clip_image_u8_free(img.img_data);
            }
            img.prefix_prompt = "";
        }

        images.clear();
    }

    bool has_budget(gpt_params &global_params) {
        if (params.n_predict == -1 && global_params.n_predict == -1)
        {
            return true; // limitless
        }

        n_remaining = -1;

        if (params.n_predict != -1)
        {
            n_remaining = params.n_predict - n_decoded;
        }
        else if (global_params.n_predict != -1)
        {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool available() const {
        return state == IDLE && command == NONE;
    }

    bool is_processing() const {
        return (state == IDLE && command == LOAD_PROMPT) || state == PROCESSING;
    }

    void add_token_string(const completion_token_output &token) {
        if (command == RELEASE)
        {
            return;
        }
        cache_tokens.push_back(token.tok);
        generated_token_probs.push_back(token);
    }

    void release() {
        if (state == PROCESSING)
        {
            t_token_generation = (ggml_time_us() - t_start_genereration) / 1e3;
            command = RELEASE;
        }
    }

    json get_formated_timings() {
        return json
        {
            {"prompt_n",               num_prompt_tokens_processed},
            {"prompt_ms",              t_prompt_processing},
            {"prompt_per_token_ms",    t_prompt_processing / num_prompt_tokens_processed},
            {"prompt_per_second",      1e3 / t_prompt_processing * num_prompt_tokens_processed},

            {"predicted_n",            n_decoded},
            {"predicted_ms",           t_token_generation},
            {"predicted_per_token_ms", t_token_generation / n_decoded},
            {"predicted_per_second",   1e3 / t_token_generation * n_decoded},
        };
    }

    void print_timings() const {
       char buffer[512];
        double t_token = t_prompt_processing / num_prompt_tokens_processed;
        double n_tokens_second = 1e3 / t_prompt_processing * num_prompt_tokens_processed;
        sprintf(buffer, "prompt eval time     = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)",
                t_prompt_processing, num_prompt_tokens_processed,
                t_token, n_tokens_second);
        LOG_INFO(buffer, {
            {"slot_id",                     id},
            {"task_id",                     task_id},
            {"t_prompt_processing",         t_prompt_processing},
            {"num_prompt_tokens_processed", num_prompt_tokens_processed},
            {"t_token",                     t_token},
            {"n_tokens_second",             n_tokens_second},
        });

        t_token = t_token_generation / n_decoded;
        n_tokens_second = 1e3 / t_token_generation * n_decoded;
        sprintf(buffer, "generation eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)",
                t_token_generation, n_decoded,
                t_token, n_tokens_second);
        LOG_INFO(buffer, {
            {"slot_id",            id},
            {"task_id",            task_id},
            {"t_token_generation", t_token_generation},
            {"n_decoded",          n_decoded},
            {"t_token",            t_token},
            {"n_tokens_second",    n_tokens_second},
        });

        sprintf(buffer, "          total time = %10.2f ms", t_prompt_processing + t_token_generation);
        LOG_INFO(buffer, {
            {"slot_id",             id},
            {"task_id",             task_id},
            {"t_prompt_processing", t_prompt_processing},
            {"t_token_generation",  t_token_generation},
            {"t_total",             t_prompt_processing + t_token_generation},
        });
    }

    // context extension via Self-Extend
    void grp_attn_update_params() {
        int grpa_i = 0;
        // copy to local variables
        int32_t grpa_n = ga_n;
        int32_t grpa_w = ga_w;
        int32_t slot_npast = 0;
        for (int k = 0; k < n_past; ++k)
        {
            while (slot_npast >= grpa_i + grpa_w) {
                const int bd = (grpa_w/grpa_n)*(grpa_n - 1);
                slot_npast -= bd;
                grpa_i += grpa_w/grpa_n;
            }
            slot_npast++;
        }
        n_past_se = slot_npast;
        ga_i = grpa_i;
    }

    int32_t grp_attn_calc_npast() {
        int32_t slot_npast = n_past_se > 0 ? n_past_se : n_past;
        // copy to local variables
        int32_t grpa_i = ga_i;
        int32_t grpa_n = ga_n;
        int32_t grpa_w = ga_w;
        while (slot_npast >= grpa_i + grpa_w) {
            const int bd = (grpa_w/grpa_n)*(grpa_n - 1);
            slot_npast -= bd;
            grpa_i += grpa_w/grpa_n;
        }
        return slot_npast;
    }

    void grp_attn_shift(llama_context * ctx, const int32_t n_tokens) {
        while (n_past_se >= ga_i + ga_w)
        {
            const int ib = (ga_n * ga_i) / ga_w;
            const int bd = (ga_w / ga_n) * (ga_n - 1);
            const int dd = (ga_w / ga_n) - ib * bd - ga_w;

            LOG_TEE("\n");
            LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past_se, ib * bd, ga_i + ib * bd, n_past_se + ib * bd);
            LOG_TEE("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n, (ga_i + ib * bd) / ga_n, (ga_i + ib * bd + ga_w) / ga_n);
            LOG_TEE("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib * bd + ga_w, n_past_se + ib * bd, dd, ga_i + ib * bd + ga_w + dd, n_past_se + ib * bd + dd);

            llama_kv_cache_seq_shift(ctx, id, ga_i, n_past_se, ib * bd);
            llama_kv_cache_seq_div(ctx, id, ga_i + ib * bd, ga_i + ib * bd + ga_w,ga_n);
            llama_kv_cache_seq_shift(ctx, id, ga_i + ib * bd + ga_w,n_past_se + ib * bd, dd);

            n_past_se -= bd;

            ga_i += ga_w / ga_n;

            LOG_TEE("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past_se + bd, n_past_se, ga_i);
        }
        n_past_se += n_tokens;
    }
};

struct llama_metrics {
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t n_tokens_predicted_total        = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted       = 0;
    uint64_t t_tokens_generation      = 0;


    void on_prompt_eval(const llama_client_slot &slot) {
        n_prompt_tokens_processed_total += slot.num_prompt_tokens_processed;

        n_prompt_tokens_processed += slot.num_prompt_tokens_processed;
        t_prompt_processing       += slot.t_prompt_processing;
    }

    void on_prediction(const llama_client_slot &slot) {
        n_tokens_predicted_total += slot.n_decoded;

        n_tokens_predicted  += slot.n_decoded;
        t_tokens_generation += slot.t_token_generation;
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};

//
// server utils
//

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value)
{
    // Fallback null to default value
    return body.contains(key) && !body.at(key).is_null()
        ? body.value(key, default_value)
        : default_value;
}

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
inline bool verify_custom_template(const std::string & tmpl) {
    llama_chat_message chat[] = {{"user", "test"}};
    std::vector<char> buf(1);
    int res = llama_chat_apply_template(nullptr, tmpl.c_str(), chat, 1, true, buf.data(), buf.size());
    return res >= 0;
}

// Format given chat. If tmpl is empty, we take the template from model metadata
inline std::string format_chat(const struct llama_model * model, const std::string & tmpl, const std::vector<json> & messages)
{
    size_t alloc_size = 0;
    // vector holding all allocated string to be passed to llama_chat_apply_template
    std::vector<std::string> str(messages.size() * 2);
    std::vector<llama_chat_message> chat(messages.size());

    for (size_t i = 0; i < messages.size(); ++i) {
        auto &curr_msg = messages[i];
        str[i*2 + 0]    = json_value(curr_msg, "role",    std::string(""));
        str[i*2 + 1]    = json_value(curr_msg, "content", std::string(""));
        alloc_size     += str[i*2 + 1].length();
        chat[i].role    = str[i*2 + 0].c_str();
        chat[i].content = str[i*2 + 1].c_str();
    }

    const char * ptr_tmpl = tmpl.empty() ? nullptr : tmpl.c_str();
    std::vector<char> buf(alloc_size * 2);

    // run the first time to get the total output length
    int32_t res = llama_chat_apply_template(model, ptr_tmpl, chat.data(), chat.size(), true, buf.data(), buf.size());

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(model, ptr_tmpl, chat.data(), chat.size(), true, buf.data(), buf.size());
    }

    std::string formatted_chat(buf.data(), res);
    LOG_VERBOSE("formatted_chat", {{"text", formatted_chat.c_str()}});

    return formatted_chat;
}

//
// work queue utils
//

struct llama_server_queue {
    int id = 0;
    std::mutex mutex_tasks;
    bool running;
    // queues
    std::vector<task_server> queue_tasks;
    std::vector<task_server> queue_tasks_deferred;
    std::vector<task_multi> queue_multitasks;
    std::condition_variable condition_tasks;
    // callback functions
    std::function<void(task_server&)> callback_new_task;
    std::function<void(task_multi&)> callback_finish_multitask;
    std::function<void(void)> callback_run_slots;

    // Add a new task to the end of the queue
    int post(task_server task) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        if (task.id == -1) {
            task.id = id++;
            LOG_VERBOSE("new task id", {{"new_id", task.id}});
        }
        queue_tasks.push_back(std::move(task));
        condition_tasks.notify_one();
        return task.id;
    }

    // Add a new task, but defer until one slot is available
    void defer(task_server task) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        queue_tasks_deferred.push_back(std::move(task));
    }

    // Get the next id for creating anew task
    int get_new_id() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        int new_id = id++;
        LOG_VERBOSE("new task id", {{"new_id", new_id}});
        return new_id;
    }

    // Register function to process a new task
    void on_new_task(std::function<void(task_server&)> callback) {
        callback_new_task = callback;
    }

    // Register function to process a multitask when it is finished
    void on_finish_multitask(std::function<void(task_multi&)> callback) {
        callback_finish_multitask = callback;
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_run_slots(std::function<void(void)> callback) {
        callback_run_slots = callback;
    }

    // Call when the state of one slot is changed
    void notify_slot_changed() {
        // move deferred tasks back to main loop
        std::unique_lock<std::mutex> lock(mutex_tasks);
        for (auto & task : queue_tasks_deferred) {
            queue_tasks.push_back(std::move(task));
        }
        queue_tasks_deferred.clear();
    }

    // end the start_loop routine
    void terminate() {
        {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            running = false;
        }
        condition_tasks.notify_all();
    }

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Run all slots
     */
    void start_loop() {
        running = true;
        while (true) {
            LOG_VERBOSE("new task may arrive", {});
            {
                while (true)
                {
                    std::unique_lock<std::mutex> lock(mutex_tasks);
                    if (queue_tasks.empty()) {
                        lock.unlock();
                        break;
                    }
                    task_server task = queue_tasks.front();
                    queue_tasks.erase(queue_tasks.begin());
                    lock.unlock();
                    LOG_VERBOSE("callback_new_task", {{"task_id", task.id}});
                    callback_new_task(task);
                }
                LOG_VERBOSE("update_multitasks", {});
                // check if we have any finished multitasks
                auto queue_iterator = queue_multitasks.begin();
                while (queue_iterator != queue_multitasks.end())
                {
                    if (queue_iterator->subtasks_remaining.empty())
                    {
                        // all subtasks done == multitask is done
                        task_multi current_multitask = *queue_iterator;
                        callback_finish_multitask(current_multitask);
                        // remove this multitask
                        queue_iterator = queue_multitasks.erase(queue_iterator);
                    }
                    else
                    {
                        ++queue_iterator;
                    }
                }
                // all tasks in the current loop is processed, slots data is now ready
                LOG_VERBOSE("callback_run_slots", {});
                callback_run_slots();
            }
            LOG_VERBOSE("wait for new task", {});
            // wait for new task
            {
                std::unique_lock<std::mutex> lock(mutex_tasks);
                if (queue_tasks.empty()) {
                    if (!running) {
                        LOG_VERBOSE("ending start_loop", {});
                        return;
                    }
                    condition_tasks.wait(lock, [&]{
                        return (!queue_tasks.empty() || !running);
                    });
                }
            }
        }
    }

    //
    // functions to manage multitasks
    //

    // add a multitask by specifying the id of all subtask (subtask is a task_server)
    void add_multitask(int multitask_id, std::vector<int>& sub_ids)
    {
        std::lock_guard<std::mutex> lock(mutex_tasks);
        task_multi multi;
        multi.id = multitask_id;
        std::copy(sub_ids.begin(), sub_ids.end(), std::inserter(multi.subtasks_remaining, multi.subtasks_remaining.end()));
        queue_multitasks.push_back(multi);
    }

    // updatethe remaining subtasks, while appending results to multitask
    void update_multitask(int multitask_id, int subtask_id, task_result& result)
    {
        std::lock_guard<std::mutex> lock(mutex_tasks);
        for (auto& multitask : queue_multitasks)
        {
            if (multitask.id == multitask_id)
            {
                multitask.subtasks_remaining.erase(subtask_id);
                multitask.results.push_back(result);
            }
        }
    }
};

struct llama_server_response {
    typedef std::function<void(int, int, task_result&)> callback_multitask_t;
    callback_multitask_t callback_update_multitask;
    // for keeping track of all tasks waiting for the result
    std::set<int> waiting_task_ids;
    // the main result queue
    std::vector<task_result> queue_results;
    std::mutex mutex_results;
    std::condition_variable condition_results;

    // add the task_id to the list of tasks waiting for response
    void add_waiting_task_id(int task_id) {
        LOG_VERBOSE("waiting for task id", {{"task_id", task_id}});
        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.insert(task_id);
    }

    // when thr request is finished, we can remove task associated with it
    void remove_waiting_task_id(int task_id) {
        LOG_VERBOSE("remove waiting for task id", {{"task_id", task_id}});
        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.erase(task_id);
        // also clear pending results, just in case
        for (int i = 0; i < (int) queue_results.size(); i++)
        {
            if (queue_results[i].id == task_id)
            {
                queue_results.erase(queue_results.begin() + i);
            }
        }
    }

    // This function blocks the thread until there is a response for this task_id
    task_result recv(int task_id) {
        while (true)
        {
            std::unique_lock<std::mutex> lock(mutex_results);
            condition_results.wait(lock, [&]{
                return !queue_results.empty();
            });

            for (int i = 0; i < (int) queue_results.size(); i++)
            {
                if (queue_results[i].id == task_id)
                {
                    assert(queue_results[i].multitask_id == -1);
                    task_result res = queue_results[i];
                    LOG_VERBOSE("got task result", {{"task_id", res.id}});
                    queue_results.erase(queue_results.begin() + i);
                    return res;
                }
            }
        }

        // should never reach here
    }

    // Register the function to update multitask
    void on_multitask_update(callback_multitask_t callback) {
        callback_update_multitask = callback;
    }

    // Send a new result to a waiting task_id
    void send(task_result result) {
        std::unique_lock<std::mutex> lock(mutex_results);
        LOG_VERBOSE("send new result", {{"task_id", result.id}});
        for (auto& task_id : waiting_task_ids) {
            // LOG_TEE("waiting task id %i \n", task_id);
            // for now, tasks that have associated parent multitasks just get erased once multitask picks up the result
            if (result.multitask_id == task_id)
            {
                LOG_VERBOSE("callback_update_multitask", {{"task_id", task_id}});
                callback_update_multitask(task_id, result.id, result);
                continue;
            }

            if (result.id == task_id)
            {
                LOG_VERBOSE("queue_results.push_back", {{"task_id", task_id}});
                queue_results.push_back(result);
                condition_results.notify_all();
                return;
            }
        }
    }
};

//
// base64 utils (TODO: move to common in the future)
//

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c)
{
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline std::vector<uint8_t> base64_decode(const std::string & encoded_string)
{
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_]))
    {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4)
        {
            for (i = 0; i <4; i++)
            {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++)
            {
                ret.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }

    if (i)
    {
        for (j = i; j <4; j++)
        {
            char_array_4[j] = 0;
        }

        for (j = 0; j <4; j++)
        {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; (j < i - 1); j++)
        {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// random string / id
//

static std::string random_string()
{
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

static std::string gen_chatcmplid()
{
    std::stringstream chatcmplid;
    chatcmplid << "chatcmpl-" << random_string();
    return chatcmplid.str();
}
