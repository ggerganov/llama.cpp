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

#ifndef SERVER_VERBOSE          // if not defined externally above, make it true and enable verbose logging
#define SERVER_VERBOSE 1
#endif

/*
// Example usage (WIP):
LogRedirection default_settings;  // Use defaults but not necessary to say so
LogRedirection custom_settings;

// Modify values for custom settings if needed
log_settings.stdout_target = "/tmp/custom.log";

LOG_ERROR("Default error", "Details", default_settings);    // can omit default_settings because it will default to log_settings unmodified
LOG_ERROR("Custom error", "More info", custom_settings);


Yes, using the LogRedirection struct approach eliminates the need to explicitly declare the extra variables for redirection targets and reset strings each time you call LOG_ERROR. Here's how it works:

1. Redirection Settings Encapsulated: The LogRedirection struct holds these settings, making them reusable and adaptable.
2. Default Values: The struct's members have default values defined, serving as fallbacks.
3. Macro Handles Settings: The LOG_ERROR (etc.) macros take a LogRedirection object and passes its members to the server_log function.

Example:

LOG_ERROR("Default error", {});  // Uses defaults from an empty LogRedirection object

This compact usage is possible because:

{} creates a temporary LogRedirection object with its members implicitly initialized to the default values.
The macro passes those defaults to server_log, achieving the desired behaviour without requiring explicit variable declarations at every call.
Customization:

When needed, you can create a LogRedirection object with specific values and pass it to LOG_ERROR (etc.) for tailored logging behaviour:

LogRedirection custom_settings = {.stdout_target = "/tmp/my_log.out"};
LOG_ERROR("Custom error", "Details", custom_settings);
*/

// ATTEMPT TO REFACTOR THE LOGGING BEHAVIOUR AND ALLOW REDIRECTION OF STDOUT, STDERR

struct LogRedirection {
  // Set default values for redirection targets and reset strings
  std::string stdout_target = "stdout.log"; // will be in ./build and eventually overwritten
  std::string stdout_reset = "/dev/stdout";
  std::string stderr_target = "stderr.log"; // will be in ./build and eventually overwritten
  std::string stderr_reset = "/dev/stderr";
};

LogRedirection log_settings;    // TODO: avoid global declaration

#if SERVER_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)   // if not verbose logging just return empty
#else
#define LOG_VERBOSE(MSG, ...)                                            \
    do                                                                   \
    {                                                                    \
        if (server_verbose)                                              \
        {                                                                \
            server_log("VERB", __func__, __LINE__, MSG, __VA_ARGS__,     \
               log_settings.stdout_target, log_settings.stderr_target, \
               log_settings.stdout_reset, log_settings.stderr_reset);   \
        }                                                                \
    } while (0)     // this is always false so the loop only compiles once but is treated as a single statement
#endif

#define LOG_ERROR(MSG, ...) \
    server_log("ERR", __func__, __LINE__, MSG, __VA_ARGS__, \
               log_settings.stdout_target, log_settings.stderr_target, \
               log_settings.stdout_reset, log_settings.stderr_reset)

#define LOG_WARNING(MSG, ...) \
    server_log("WARN", __func__, __LINE__, MSG, __VA_ARGS__, \
               log_settings.stdout_target, log_settings.stderr_target, \
               log_settings.stdout_reset, log_settings.stderr_reset)

#define LOG_INFO(MSG, ...) \
    server_log("INFO", __func__, __LINE__, MSG, __VA_ARGS__, \
               log_settings.stdout_target, log_settings.stderr_target, \
               log_settings.stdout_reset, log_settings.stderr_reset)


//
// parallel
//

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
    int id = -1; // for any instance, task id is not assigned yet; to be filled by llama_server_queue
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

// completion token output with probabilities
struct completion_token_output {
    struct token_prob
    {
        llama_token tok;
        float prob;
    };

    std::vector<token_prob> probs;
    llama_token tok;
    std::string text_to_send;
};

struct token_translator {
    llama_context * ctx;
    std::string operator()(llama_token tok)                    const { return llama_token_to_piece(ctx, tok); }
    std::string operator()(const completion_token_output &cto) const { return (*this)(cto.tok); }
};

static inline void server_log(
    const char *level,
    const char *function,
    int line,
    const char *message,
    const nlohmann::ordered_json &extra,
    // targets and resets for stdout and stderr are specified in LogRedirection above;
    // this allows different behaviour for LOG_ERROR, LOG_WARNING and LOG_INFO
    std::string stdout_target,
    std::string stderr_target,
    std::string stdout_reset,
    std::string stderr_reset
    )
    {
    std::stringstream ss_tid;
    ss_tid << std::this_thread::get_id();
    json log = nlohmann::ordered_json{
        {"tid", ss_tid.str()},
        {"timestamp", time(nullptr)},
    };

    /*
    // to allow the graphics to print to stdout we redirect non-graphical stdout to stderr
    // to silence the graphics we direct stdout to dev/null or just don't invoke them
    std::cerr << stdout_target.c_str() << std::endl;
    FILE* new_stdout = freopen(stdout_target.c_str(), "a", stdout);
    if (new_stdout == nullptr) {
        std::cerr << "Error on redirecting stdout to " << stdout_target.c_str() << std::endl;
    }
    */

    std::cerr << stderr_target.c_str() << std::endl;
    FILE* new_stderr = freopen(stderr_target.c_str(), "a", stderr);
    if (new_stderr == nullptr) {
        std::cerr << "Error on redirecting stderr to " << stderr_target.c_str() << std::endl;
    }
    //freopen(stderr_target.c_str(), "a", stderr);      // we assign stderr to dev/null effectively 'blackholing' the output because log.dump below is redirected too

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
        std::cerr << log.dump(-1, ' ', false, json::error_handler_t::replace) << "\n" << std::flush;    // was originally std:cout

    } else {                // store the logs in (because not json) text format
        char buf[1024];
        snprintf(buf, 1024, "\033[85;0H%4s [%24s] %s", level, function, message);

        if (!extra.empty()) {
            log.merge_patch(extra);
        }
        std::stringstream ss;
        ss << buf << " |";
        for (const auto& el : log.items())
        {
            const std::string value = el.value().dump(-1, ' ', false, json::error_handler_t::replace);
            ss << " " << el.key() << "=" << value;
        }

        const std::string str = ss.str();
        printf("\033[85;0H%.*s\n", (int)str.size(), str.data());
        fflush(stderr);                                                // was originally fflush(stdout)

        /*
        freopen(stdout_reset.c_str(), "a", stdout);                    // decide whether to restore stdout
        freopen(stderr_reset.c_str(), "a", stderr);                    // decide whether to restore stderr (both need automating)
        */
    }
}

//
// server utils
//

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value) {
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
inline std::string format_chat(const struct llama_model * model, const std::string & tmpl, const std::vector<json> & messages) {
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
    LOG_VERBOSE("formatted_chat", {"text", formatted_chat.c_str()});

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
            LOG_VERBOSE("new task id", {"new_id", task.id});
        }
        queue_tasks.push_back(std::move(task));
        //LOG("Queue now has %2zu members.\n", queue_tasks.size());
        condition_tasks.notify_one();
        return task.id;
    }

    // Add a new task, but defer until one slot is available
    void defer(task_server task) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        queue_tasks_deferred.push_back(std::move(task));
        LOG("Deferred queue now has %3zu members.\n", queue_tasks_deferred.size());
    }

    // Get the next id for creating a new task
    int get_new_id() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        int new_id = id++;
        LOG_VERBOSE("new task id", {"new_id", new_id});
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
        // does this mean when ONE slot finished we move ALL deferred tasks back to the main queue? Why?
        // it seems that we move everything back to the main queue but we don't allocate a task to the slot just released
        // lock so nothing gets added while we are clearing the deferred queue
        std::unique_lock<std::mutex> lock(mutex_tasks);
        for (auto & task : queue_tasks_deferred) {
            queue_tasks.push_back(std::move(task));
        }
        queue_tasks_deferred.clear();   // and clear the deferred tasks completely?
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
        //LOG("In start_loop have new task number %d.\n", id);
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
                    LOG_VERBOSE("callback_new_task", {"task_id", task.id});
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
    std::set<int> waiting_task_ids;     // this stores waiting tasks with no obvious limit
    // the main result queue
    std::vector<task_result> queue_results;
    std::mutex mutex_results;
    std::condition_variable condition_results;

    // add the task_id to the list of tasks waiting for response
    void add_waiting_task_id(int task_id) {
        LOG_VERBOSE("waiting for task id", {"task_id", task_id});
        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.insert(task_id);
        LOG("Waiting task list size after addition: %2zu.\n", waiting_task_ids.size());
    }

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int task_id) {
        LOG_VERBOSE("remove waiting for task id", {"task_id", task_id});
        std::unique_lock<std::mutex> lock(mutex_results);
        waiting_task_ids.erase(task_id);
        LOG("Waiting task list size after removal: %zu.\n", waiting_task_ids.size());
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
        LOG_VERBOSE("send new result", {"task_id", result.id});
        for (auto& task_id : waiting_task_ids) {
            // LOG("waiting task id %i \n", task_id);
            // for now, tasks that have associated parent multitasks just get erased once multitask picks up the result
            if (result.multitask_id == task_id)
            {
                LOG_VERBOSE("callback_update_multitask", {"task_id", task_id});
                callback_update_multitask(task_id, result.id, result);
                continue;
            }

            if (result.id == task_id)
            {
                LOG_VERBOSE("queue_results.push_back", {"task_id", task_id});
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

//
// other common utils
//

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }
    return i;
}

static bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string &stop,
                                       const std::string &text)
{
    if (!text.empty() && !stop.empty())
    {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--)
        {
            if (stop[char_index] == text_last_char)
            {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial))
                {
                    return text.size() - char_index - 1;
                }
            }
        }
    }
    return std::string::npos;
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(llama_context *ctx, Iter begin, Iter end)
{
    std::string ret;
    for (; begin != end; ++begin)
    {
        ret += llama_token_to_piece(ctx, *begin);
    }
    return ret;
}

// format incomplete utf-8 multibyte character for output
static std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);
    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }
    return out;
}

// convert a vector of completion_token_output to json
static json probs_vector_to_json(const llama_context *ctx, const std::vector<completion_token_output> &probs)
{
    json out = json::array();
    for (const auto &prob : probs)
    {
        json probs_for_token = json::array();
        for (const auto &p : prob.probs)
        {
            std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
            probs_for_token.push_back(json
            {
                {"tok_str", tok_str},
                {"prob",    p.prob},
            });
        }
        std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
        out.push_back(json{
            {"content", tok_str},
            {"probs",   probs_for_token},
        });
    }
    return out;
}
