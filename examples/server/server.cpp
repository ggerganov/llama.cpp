#include "common.h"
#include "llama.h"
#include "build-info.h"
#include "grammar-parser.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif

#include "httplib.h"
#include "json.hpp"

// auto generated files (update with ./deps.sh)
#include "index.html.hpp"
#include "index.js.hpp"
#include "completion.js.hpp"
#include "json-schema-to-grammar.mjs.hpp"

#include <cstddef>
#include <thread>
#include <chrono>

#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
#endif

using namespace httplib;
using json = nlohmann::json;

struct server_params
{
    std::string hostname = "127.0.0.1";
    std::string public_path = "examples/server/public";
    int32_t port = 8040;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
};

static bool server_verbose = false;

#if SERVER_VERBOSE != 1
#define LOG_VERBOSE(MSG, ...)
#else
#define LOG_VERBOSE(MSG, ...)                                            \
    do                                                                   \
    {                                                                    \
        if (server_verbose)                                              \
        {                                                                \
            server_log("VERBOSE", __func__, __LINE__, MSG, __VA_ARGS__); \
        }                                                                \
    } while (0)
#endif

#define LOG_ERROR(MSG, ...) server_log("ERROR", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log("WARNING", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_INFO(MSG, ...) server_log("INFO", __func__, __LINE__, MSG, __VA_ARGS__)


// parallel
enum slot_state
{
    IDLE,
    SLEEPING,
    PROCESSING
};

enum slot_command {
    NONE,
    LOAD_PROMPT,
    RELEASE
};

struct slot_params {
    bool stream = true;
    uint32_t seed                           = -1;   // RNG seed
    int32_t n_predict                       = 128;   // new tokens to predict

    // sampler params
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typical_p         = 1.00f; // 1.0 = disabled
    float   temp              = 0.80f; // 1.0 = disabled
    float   repeat_penalty    = 1.10f; // 1.0 = disabled
    int32_t repeat_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   frequency_penalty = 0.00f; // 0.0 = disabled
    float   presence_penalty  = 0.00f; // 0.0 = disabled
    int32_t mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate
    int n_probs                   = 0;
    bool penalize_nl          = false;

    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens

    std::string grammar           = "";  // optional BNF-like grammar to constrain sampling
    bool remember_generation =   false;  // remember a part of the prompt to avoid reprocessing all prompt
    std::vector<std::string> antiprompt;
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
};

static size_t common_part(const std::vector<llama_token> &a, const std::vector<llama_token> &b)
{
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++)
    {
    }
    return i;
}

enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

static bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static void slot_params_to_gpt_params(const slot_params &src, gpt_params & dst)
{
    dst.frequency_penalty = src.frequency_penalty;
    dst.temp = src.temp;
    dst.top_k = src.top_k;
    dst.top_p = src.top_p;
    dst.grammar = src.grammar;
    dst.logit_bias = src.logit_bias;
    dst.mirostat = src.mirostat;
    dst.mirostat_eta = src.mirostat_eta;
    dst.mirostat_tau = src.mirostat_tau;
    dst.typical_p = src.typical_p;
    dst.repeat_penalty = src.repeat_penalty;
    dst.repeat_last_n = src.repeat_last_n;
    dst.presence_penalty = src.presence_penalty;
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

static void server_log(const char *level, const char *function, int line,
                       const char *message, const nlohmann::ordered_json &extra)
{
    nlohmann::ordered_json log{
        {"timestamp", time(nullptr)},
        {"level", level},
        {"function", function},
        {"line", line},
        {"message", message},
    };

    if (!extra.empty())
    {
        log.merge_patch(extra);
    }

    const std::string str = log.dump(-1, ' ', false, json::error_handler_t::replace);
    printf("%.*s\n", (int)str.size(), str.data());
    fflush(stdout);
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
static json probs_vector_to_json(const llama_context *ctx, const std::vector<completion_token_output> & probs)
{
    json out = json::array();
    for (const auto &prob : probs)
    {
        json probs_for_token = json::array();
        for (const auto &p : prob.probs)
        {
            std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
            probs_for_token.push_back(json{
                {"tok_str", tok_str},
                {"prob", p.prob},
            });
        }
        std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
        out.push_back(json{
            {"content", tok_str},
            {"probs", probs_for_token},
        });
    }
    return out;
}

struct llama_client_slot
{
    int id;
    // generation props
    int32_t num_prompt_tokens  = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;
    size_t n_past = 0;
    json prompt;
    std::string generated_text = "";
    int num_tokens_predicted = 0;
    llama_token sampled;
    std::vector<llama_token> context_tokens;
    std::vector<llama_token> last_n_tokens;
    std::vector<completion_token_output> generated_token_probs;
    int sent_tokens = 0;
    slot_state state = IDLE;
    slot_command command = NONE;
    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    int32_t multibyte_pending = 0;

    slot_params params;

    // grammar props
    grammar_parser::parse_state parsed_grammar;
    llama_grammar *grammar = nullptr;

    void reset() {
        state = IDLE;
        command = NONE;
        num_prompt_tokens = 0;
        num_tokens_predicted = 0;
        generated_text = "";
        generated_token_probs.clear();
        truncated = false;
        stopped_eos = false;
        stopped_word = false;
        stopped_limit = false;
        stopping_word = "";
        multibyte_pending = 0;
        n_past = 0;

        if (grammar != nullptr) {
            llama_grammar_free(grammar);
            grammar = nullptr;
        }

        // llama_set_rng_seed(ctx, params.seed); in batched the seed matter???????
    }

    bool loadGrammar()
    {
        if (!params.grammar.empty()) {
            parsed_grammar = grammar_parser::parse(params.grammar.c_str());
            // will be empty (default) if there are parse errors
            if (parsed_grammar.rules.empty()) {
                LOG_ERROR("grammar parse error", {{"grammar", params.grammar}});
                return false;
            }
            grammar_parser::print_grammar(stderr, parsed_grammar);

            // TODO: fix this comment
            // {
            //     auto it = params.logit_bias.find(llama_token_eos(ctx));
            //     if (it != params.logit_bias.end() && it->second == -INFINITY) {
            //         LOG_WARNING("EOS token is disabled, which will cause most grammars to fail", {});
            //     }
            // }

            std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
            grammar = llama_grammar_init(
                grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
        }
        return true;
    }

    bool hasNewToken() {
        return generated_token_probs.size() > sent_tokens;
    }

    bool available() {
        return state == IDLE &&
            command == NONE && !params.remember_generation;
    }

    bool isProcessing() {
        return (state == IDLE && command == LOAD_PROMPT) || state == PROCESSING;
    }

    completion_token_output next() {
        completion_token_output tkn = generated_token_probs.at(sent_tokens);
        sent_tokens++;
        return tkn;
    }

    void addTokenString(completion_token_output token) {
        if(command == RELEASE) {
            generated_token_probs.clear();
            sent_tokens = 0;
            return;
        }
        context_tokens.push_back(token.tok);
        generated_token_probs.push_back(token);
        num_tokens_predicted++;
    }

    void release() {
        if(state == PROCESSING) {
            command = RELEASE;
        }
    }
};

struct llama_server_context
{
    std::vector<llama_client_slot> slots;

    // system prompt
    std::string system_prompt = "";
    bool update_system_prompt = false;
    std::vector<llama_token> tokens_system;
    int32_t n_tokens_system = 0;

    // broadcast to all clients to keep the same prompt format
    std::string user_name = ""; // this should be the anti prompt
    std::string assistant_name = ""; // this is for generate the prompt

    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_batch batch;
    std::vector<llama_token_data> candidates;
    bool all_slots_are_idle = false;
    gpt_params params;
    int n_ctx;
    int n_vocab;

    std::mutex mutex;

    std::unique_lock<std::mutex> lock()
    {
        return std::unique_lock<std::mutex>(mutex);
    }

    ~llama_server_context()
    {
        if (ctx)
        {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model)
        {
            llama_free_model(model);
            model = nullptr;
        }

        for(auto &slot : slots) {
            if(slot.grammar) {
                llama_grammar_free(slot.grammar);
            }
        }
    }

    bool loadModel(const gpt_params &params_)
    {
        params = params_;
        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        if (model == nullptr)
        {
            LOG_ERROR("unable to load model", {{"model", params_.model}});
            return false;
        }
        n_ctx = llama_n_ctx(ctx);
        n_vocab = llama_n_vocab(model);
        candidates.reserve(n_vocab);
        return true;
    }

    void initialize() {
        // create slots
        LOG_TEE("Available slots:\n");
        for (int i = 0; i < params.n_parallel; i++)
        {
            llama_client_slot slot;
            slot.id = i;
            slot.last_n_tokens.resize(params.n_predict); // max prediction per slot
            slot.reset();
            std::fill(slot.last_n_tokens.begin(), slot.last_n_tokens.end(), 0);
            LOG_TEE(" - slot %i\n", slot.id);
            slots.push_back(slot);
        }
        batch = llama_batch_init(params.n_ctx, 0);

        // empty system prompt
        system_prompt = "";
        all_slots_are_idle = true;
    }

    std::vector<llama_token> tokenize(const json & json_prompt, bool add_bos) const
    {
        // If `add_bos` is true, we only add BOS, when json_prompt is a string,
        // or the first element of the json_prompt array is a string.
        std::vector<llama_token> prompt_tokens;

        if (json_prompt.is_array())
        {
            bool first = true;
            for (const auto& p : json_prompt)
            {
                if (p.is_string())
                {
                    auto s = p.template get<std::string>();
                    std::vector<llama_token> p;
                    if (first)
                    {
                        p = ::llama_tokenize(ctx, s, add_bos);
                        first = false;
                    }
                    else
                    {
                        p = ::llama_tokenize(ctx, s, false);
                    }
                    prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
                }
                else
                {
                    if (first)
                    {
                        first = false;
                    }
                    prompt_tokens.push_back(p.template get<llama_token>());
                }
            }
        }
        else
        {
            auto s = json_prompt.template get<std::string>();
            prompt_tokens = ::llama_tokenize(ctx, s, add_bos);
        }

        return prompt_tokens;
    }

    void processPrompt() {


        //params.n_keep = std::min(n_ctx - 4, params.n_keep);

        // if input prompt is too big, truncate like normal
        // if (num_prompt_tokens >= (size_t)n_ctx)
        // {
        //     const int n_left = (n_ctx - params.n_keep) / 2;
        //     std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
        //     const int erased_blocks = (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
        //     new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
        //     std::copy(prompt_tokens.end() - n_ctx, prompt_tokens.end(), last_n_tokens.begin());

        //     LOG_VERBOSE("input truncated", {
        //                                        {"n_ctx", n_ctx},
        //                                        {"n_keep", params.n_keep},
        //                                        {"n_left", n_left},
        //                                        {"new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend())},
        //                                    });

        //     truncated = true;
        //     prompt_tokens = new_tokens;
        // }
        // else
        // {
        //     const size_t ps = num_prompt_tokens;
        //     std::fill(last_n_tokens.begin(), last_n_tokens.end() - ps, 0);
        //     std::copy(prompt_tokens.begin(), prompt_tokens.end(), last_n_tokens.end() - ps);
        // }

        // compare the evaluated prompt with the new prompt
    }


    llama_client_slot* getSlot(int id) {
        for (llama_client_slot & slot : slots)
        {
            if ((id == -1 && slot.available()) || slot.id == id)
            {
                return &slot;
            }
        }
        return nullptr;
    }

    bool launchSlot(llama_client_slot* &slot) {
        if(!slot->loadGrammar()) {
            return false;
        }
        all_slots_are_idle = false;
        slot->command = LOAD_PROMPT;
        LOG_TEE("slot %i is processing\n", slot->id);
        return true;
    }

    void loadInfill()
    {
        // bool suff_rm_leading_spc = true;
        // if (params.input_suffix.find_first_of(" ") == 0 && params.input_suffix.size() > 1) {
        //     params.input_suffix.erase(0, 1);
        //     suff_rm_leading_spc = false;
        // }

        // auto prefix_tokens = tokenize(params.input_prefix, false);
        // auto suffix_tokens = tokenize(params.input_suffix, false);
        // const int space_token = 29871;
        // if (suff_rm_leading_spc  && suffix_tokens[0] == space_token) {
        //     suffix_tokens.erase(suffix_tokens.begin());
        // }
        // prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(ctx));
        // prefix_tokens.insert(prefix_tokens.begin(), llama_token_bos(ctx)); // always add BOS
        // prefix_tokens.insert(prefix_tokens.end(), llama_token_suffix(ctx));
        // prefix_tokens.insert(prefix_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
        // prefix_tokens.push_back(llama_token_middle(ctx));
        // auto prompt_tokens = prefix_tokens;

        // num_prompt_tokens = prompt_tokens.size();

        // if (params.n_keep < 0)
        // {
        //     params.n_keep = (int)num_prompt_tokens;
        // }
        // params.n_keep = std::min(params.n_ctx - 4, params.n_keep);

        // // if input prompt is too big, truncate like normal
        // if (num_prompt_tokens >= (size_t)params.n_ctx)
        // {
        //     printf("Input prompt is too big, truncating. Can only take %d tokens but got %zu\n", params.n_ctx, num_prompt_tokens);
        //     // todo we probably want to cut from both sides
        //     const int n_left = (params.n_ctx - params.n_keep) / 2;
        //     std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
        //     const int erased_blocks = (num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
        //     new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
        //     std::copy(prompt_tokens.end() - params.n_ctx, prompt_tokens.end(), last_n_tokens.begin());

        //     LOG_VERBOSE("input truncated", {
        //                                        {"n_ctx", params.n_ctx},
        //                                        {"n_keep", params.n_keep},
        //                                        {"n_left", n_left},
        //                                        {"new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend())},
        //                                    });

        //     truncated = true;
        //     prompt_tokens = new_tokens;
        // }
        // else
        // {
        //     const size_t ps = num_prompt_tokens;
        //     std::fill(last_n_tokens.begin(), last_n_tokens.end() - ps, 0);
        //     std::copy(prompt_tokens.begin(), prompt_tokens.end(), last_n_tokens.end() - ps);
        // }

        // // compare the evaluated prompt with the new prompt
        // n_past = common_part(embd, prompt_tokens);
        // embd = prompt_tokens;
        // if (n_past == num_prompt_tokens)
        // {
        //     // we have to evaluate at least 1 token to generate logits.
        //     printf("we have to evaluate at least 1 token to generate logits\n");
        //     n_past--;
        // }

        // LOG_VERBOSE("prompt ingested", {
        //                                    {"n_past", n_past},
        //                                    {"cached", tokens_to_str(ctx, embd.cbegin(), embd.cbegin() + n_past)},
        //                                    {"to_eval", tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend())},
        //                                });

        // has_next_token = true;
    }

    void updateSystemPrompt() {
        tokens_system = ::llama_tokenize(ctx, system_prompt, true);
        n_tokens_system = tokens_system.size();

        batch.n_tokens = n_tokens_system;

        // clear the entire KV cache
        for (int i = 0; i < params.n_parallel; ++i)
        {
            llama_kv_cache_seq_rm(ctx, i, 0, -1);
        }

        for (int32_t i = 0; i < batch.n_tokens; ++i)
        {
            batch.token[i] = tokens_system[i];
            batch.pos[i] = i;
            batch.seq_id[i] = 0;
            batch.logits[i] = false;
        }

        if (llama_decode(ctx, batch) != 0)
        {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return;
        }

        // assign the system KV cache to all parallel sequences
        for (int32_t i = 1; i < params.n_parallel; ++i)
        {
            llama_kv_cache_seq_cp(ctx, 0, i, 0, n_tokens_system);
        }

        LOG_TEE("system prompt updated\n");
        update_system_prompt = false;
    }

    void notifySystemPromptChanged() {
        // release all slots
        for (llama_client_slot &slot : slots)
        {
            slot.release();
        }
        waitAllAreIdle();
        all_slots_are_idle = true;
        // wait until system prompt load
        update_system_prompt = true;
        while(update_system_prompt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        // system prompt loaded, continue
    }

    void processSystemPromptData(json sys_props) {
        system_prompt = sys_props.value("system_prompt", "");
        user_name = sys_props.value("anti_prompt", "");
        assistant_name = sys_props.value("assistant_name", "");
        notifySystemPromptChanged();
    }

    void waitAllAreIdle() {
        bool wait = true;
        while(wait) {
            wait = false;
            for (auto &slot : slots)
            {
                if (!slot.available())
                {
                    wait = true;
                    break;
                }
            }
        }
    }

    size_t findStoppingStrings(const size_t last_token_size,
                               const stop_type type, llama_client_slot & slot)
    {
        size_t stop_pos = std::string::npos;
        for (const std::string &word : slot.params.antiprompt)
        {
            size_t pos;
            if (type == STOP_FULL)
            {
                const size_t tmp = word.size() + last_token_size;
                const size_t from_pos = slot.generated_text.size() > tmp ? slot.generated_text.size() - tmp : 0;
                pos = slot.generated_text.find(word, from_pos);
            }
            else
            {
                pos = find_partial_stop_string(word, slot.generated_text);
            }
            if (pos != std::string::npos &&
                (stop_pos == std::string::npos || pos < stop_pos))
            {
                if (type == STOP_FULL)
                {
                    slot.stopping_word = word;
                    slot.stopped_word = true;
                }
                stop_pos = pos;
            }
        }
        return stop_pos;
    }

    bool processToken(completion_token_output & result, llama_client_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        slot.last_n_tokens.erase(slot.last_n_tokens.begin());
        slot.last_n_tokens.push_back(result.tok);
        const std::string token_str = llama_token_to_piece(ctx, result.tok);
        slot.sampled = result.tok;

        size_t stop_pos =
                findStoppingStrings(token_str.size(), STOP_FULL, slot);

        slot.addTokenString(result);

        slot.generated_text += token_str;

        bool has_next_token = !(slot.n_decoded > 2 &&
                    (result.tok == llama_token_eos(ctx) ||
                     (slot.n_decoded + slot.n_past >=
                      params.n_predict) ||
                     stop_pos != std::string::npos));

        if (slot.params.n_probs > 0)
        {
            slot.generated_token_probs.push_back(result);
        }

        if (slot.multibyte_pending > 0)
        {
            slot.multibyte_pending -= token_str.size();
        }
        else if (token_str.size() == 1)
        {
            const char c = token_str[0];
            // 2-byte characters: 110xxxxx 10xxxxxx
            if ((c & 0xE0) == 0xC0)
            {
                slot.multibyte_pending = 1;
                // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
            }
            else if ((c & 0xF0) == 0xE0)
            {
                slot.multibyte_pending = 2;
                // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            }
            else if ((c & 0xF8) == 0xF0)
            {
                slot.multibyte_pending = 3;
            }
            else
            {
                slot.multibyte_pending = 0;
            }
        }

        if (slot.multibyte_pending > 0 && !has_next_token)
        {
            has_next_token = true;
        }

        if (!has_next_token && (slot.n_decoded + slot.n_past >= params.n_predict))
        {
            slot.stopped_limit = true;
        }

        if (!slot.context_tokens.empty() && result.tok == llama_token_eos(ctx)){
                slot.stopped_eos = true;
                LOG_VERBOSE("eos token found", {});
        }

        LOG_VERBOSE("next token", {
                                      {"token", result.tok},
                                      {"token_text", tokens_to_output_formatted_string(ctx, result.tok)},
                                      {"has_next_token", has_next_token},
                                      {"n_remain", (params.n_predict - slot.n_decoded + slot.n_past)},
                                      {"num_tokens_predicted", slot.num_tokens_predicted},
                                      {"stopped_eos", slot.stopped_eos},
                                      {"stopped_word", slot.stopped_word},
                                      {"stopped_limit", slot.stopped_limit},
                                      {"stopping_word", slot.stopping_word},
                                  });
        return has_next_token; // continue
    }

    bool updateSlots() {
        // update the system prompt wait until all slots are idle state
        if(update_system_prompt) {
            updateSystemPrompt();
        }

        batch.n_tokens = 0;
        int kv_cache_free = (n_ctx - n_tokens_system);

        if(all_slots_are_idle) {
            // avoid 100% usage of cpu all time
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // decode any currently ongoing sequences
        for (auto & slot : slots) {
            // release the slot
            if (slot.state == PROCESSING && slot.command == RELEASE && !slot.hasNewToken())
            {
                LOG_TEE("slot %i released\n", slot.id);
                if(!slot.params.remember_generation) {
                    llama_kv_cache_seq_rm(ctx, slot.id, n_tokens_system, n_ctx);
                    slot.state = IDLE;
                    slot.command = NONE;
                    slot.num_prompt_tokens = 0;
                    slot.num_tokens_predicted = 0;
                } else {
                    slot.state = SLEEPING;
                    slot.command = NONE;
                }
                continue;
            }

            kv_cache_free -= slot.num_prompt_tokens;

            if (slot.state == IDLE || slot.command == RELEASE) {
                continue;
            }

            batch.token [batch.n_tokens] = slot.sampled;
            batch.pos   [batch.n_tokens] = n_tokens_system + slot.n_past + slot.n_decoded;
            batch.seq_id[batch.n_tokens] = slot.id;
            batch.logits[batch.n_tokens] = true;

            slot.n_decoded += 1;
            slot.i_batch = batch.n_tokens;

            batch.n_tokens += 1;
        }

        // assign workload to the slots
        if (params.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                // need process the prompt
                bool keep_gen = slot.state == SLEEPING; // remember generation
                if ((slot.state == IDLE || keep_gen) && slot.command == LOAD_PROMPT) {
                    slot.state = PROCESSING;
                    slot.command = NONE;

                    auto prompt_tokens = tokenize(slot.prompt, system_prompt.empty());  // add BOS if there isn't system prompt
                    slot.num_prompt_tokens = prompt_tokens.size();

                    slot.n_past = keep_gen ? common_part(slot.context_tokens, prompt_tokens) : 0;

                    slot.context_tokens = prompt_tokens;

                    LOG_VERBOSE("prompt ingested", {
                                                    {"n_past", slot.n_past},
                                                    {"cached", tokens_to_str(ctx, slot.context_tokens.cbegin(), slot.context_tokens.cbegin() + slot.n_past)},
                                                    {"to_eval", tokens_to_str(ctx, slot.context_tokens.cbegin() + slot.n_past, slot.context_tokens.cend())},
                                                });

                    std::fill(slot.last_n_tokens.begin(), slot.last_n_tokens.end(), 0);

                    for (size_t i = slot.n_past; i < slot.context_tokens.size(); ++i) {
                        batch.token [batch.n_tokens] = slot.context_tokens[i];
                        batch.pos   [batch.n_tokens] = i + n_tokens_system;
                        batch.seq_id[batch.n_tokens] = slot.id;
                        batch.logits[batch.n_tokens] = false;
                        batch.n_tokens += 1;
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0) {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    slot.n_decoded = 0;
                    slot.i_batch   = batch.n_tokens - 1;
                }
            }
        }

        if (batch.n_tokens == 0) {
            all_slots_are_idle = true;
            return true;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token  + i,
                nullptr,
                batch.pos    + i,
                batch.seq_id + i,
                batch.logits + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_TEE("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return false;
                }

                LOG("%s : failed to decode the batch, retrying with n_batch = %d\n", __func__, n_batch / 2);

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;
                continue;
            }

            for (auto & slot : slots) {
                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue;
                }

                slot_params_to_gpt_params(slot.params, params);
                completion_token_output result;
                const llama_token id = llama_sample_token(ctx, NULL, NULL, params, slot.last_n_tokens, candidates, slot.i_batch - i);
                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
                result.tok = id;
                const int32_t n_probs = params.n_probs;
                if (params.temp <= 0 && n_probs > 0)
                {
                    // For llama_sample_token_greedy we need to sort candidates
                    llama_sample_softmax(ctx, &candidates_p);
                }

                for (size_t i = 0; i < std::min(candidates_p.size, (size_t)n_probs); ++i)
                {
                    result.probs.push_back({candidates_p.data[i].id, candidates_p.data[i].p});
                }

                if (!processToken(result, slot)) {
                    slot.generated_text.clear();
                    slot.release();
                }
                kv_cache_free -= slot.num_tokens_predicted;
                slot.i_batch = -1;
            }
        }

        if(kv_cache_free < 0) {
            LOG_TEE("\nError: kv cache is full, increase context size.");
            return false;
        }
        return true;
    }

    std::vector<float> getEmbedding()
    {
        static const int n_embd = llama_n_embd(model);
        if (!params.embedding)
        {
            LOG_WARNING("embedding disabled", {
                                                  {"params.embedding", params.embedding},
                                              });
            return std::vector<float>(n_embd, 0.0f);
        }
        const float *data = llama_get_embeddings(ctx);
        std::vector<float> embedding(data, data + n_embd);
        return embedding;
    }
};

static void server_print_usage(const char *argv0, const gpt_params &params,
                               const server_params &sparams)
{
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  -v, --verbose         verbose output (default: %s)\n", server_verbose ? "enabled" : "disabled");
    printf("  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    printf("  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
    printf("  --rope-freq-base N    RoPE base frequency (default: loaded from model)\n");
    printf("  --rope-freq-scale N   RoPE frequency scaling factor (default: loaded from model)\n");
    printf("  -b N, --batch-size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    printf("  --memory-f32          use f32 instead of f16 for memory key+value (default: disabled)\n");
    printf("                        not recommended: doubles context memory required and no measurable increase in quality\n");
    if (llama_mlock_supported())
    {
        printf("  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_mmap_supported())
    {
        printf("  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
    printf("  --numa                attempt optimizations that help on some NUMA systems\n");
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
    printf("  -ngl N, --n-gpu-layers N\n");
    printf("                        number of layers to store in VRAM\n");
    printf("  -ts SPLIT --tensor-split SPLIT\n");
    printf("                        how to split tensors across multiple GPUs, comma-separated list of proportions, e.g. 3,1\n");
    printf("  -mg i, --main-gpu i   the GPU to use for scratch and small tensors\n");
    printf("  -nommq, --no-mul-mat-q\n");
    printf("                        use cuBLAS instead of custom mul_mat_q CUDA kernels.\n");
    printf("                        Not recommended since this is both slower and uses more VRAM.\n");
#endif
    printf("  -m FNAME, --model FNAME\n");
    printf("                        model path (default: %s)\n", params.model.c_str());
    printf("  -a ALIAS, --alias ALIAS\n");
    printf("                        set an alias for the model, will be added as `model` field in completion response\n");
    printf("  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
    printf("  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
    printf("  --host                ip address to listen (default  (default: %s)\n", sparams.hostname.c_str());
    printf("  --port PORT           port to listen (default  (default: %d)\n", sparams.port);
    printf("  --path PUBLIC_PATH    path from which to serve static files (default %s)\n", sparams.public_path.c_str());
    printf("  -to N, --timeout N    server read/write timeout in seconds (default: %d)\n", sparams.read_timeout);
    printf("  --embedding           enable embedding vector output (default: %s)\n", params.embedding ? "enabled" : "disabled");
    printf("\n");
}

static void server_params_parse(int argc, char **argv, server_params &sparams,
                                gpt_params &params)
{
    gpt_params default_params;
    server_params default_sparams;
    std::string arg;
    bool invalid_param = false;

    for (int i = 1; i < argc; i++)
    {
        arg = argv[i];
        if (arg == "--port")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.port = std::stoi(argv[i]);
        }
        else if (arg == "--host")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.hostname = argv[i];
        }
        else if (arg == "--path")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.public_path = argv[i];
        }
        else if (arg == "--timeout" || arg == "-to")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            sparams.read_timeout = std::stoi(argv[i]);
            sparams.write_timeout = std::stoi(argv[i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        }
        else if (arg == "-a" || arg == "--alias")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.model_alias = argv[i];
        }
        else if (arg == "-h" || arg == "--help")
        {
            server_print_usage(argv[0], default_params, default_sparams);
            exit(0);
        }
        else if (arg == "-c" || arg == "--ctx-size" || arg == "--ctx_size")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_ctx = std::stoi(argv[i]);
        }
        else if (arg == "--rope-freq-base")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.rope_freq_base = std::stof(argv[i]);
        }
        else if (arg == "--rope-freq-scale")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.rope_freq_scale = std::stof(argv[i]);
        }
        else if (arg == "--memory-f32" || arg == "--memory_f32")
        {
            params.memory_f16 = false;
        }
        else if (arg == "--threads" || arg == "-t")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        }
        else if (arg == "-b" || arg == "--batch-size")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(argv[i]);
            params.n_batch = std::min(512, params.n_batch);
        }
        else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
            params.n_gpu_layers = std::stoi(argv[i]);
#else
            LOG_WARNING("Not compiled with GPU offload support, --n-gpu-layers option will be ignored. "
                        "See main README.md for information on enabling GPU BLAS support",
                        {{"n_gpu_layers", params.n_gpu_layers}});
#endif
        }
        else if (arg == "--tensor-split" || arg == "-ts")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
#ifdef GGML_USE_CUBLAS
            std::string arg_next = argv[i];

            // split string by , and /
            const std::regex regex{R"([,/]+)"};
            std::sregex_token_iterator it{arg_next.begin(), arg_next.end(), regex, -1};
            std::vector<std::string> split_arg{it, {}};
            GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

            for (size_t i_device = 0; i_device < LLAMA_MAX_DEVICES; ++i_device)
            {
                if (i_device < split_arg.size())
                {
                    params.tensor_split[i_device] = std::stof(split_arg[i_device]);
                }
                else
                {
                    params.tensor_split[i_device] = 0.0f;
                }
            }
#else
            LOG_WARNING("llama.cpp was compiled without cuBLAS. It is not possible to set a tensor split.\n", {});
#endif // GGML_USE_CUBLAS
        }
        else if (arg == "--no-mul-mat-q" || arg == "-nommq")
        {
#ifdef GGML_USE_CUBLAS
            params.mul_mat_q = false;
#else
            LOG_WARNING("warning: llama.cpp was compiled without cuBLAS. Disabling mul_mat_q kernels has no effect.\n", {});
#endif // GGML_USE_CUBLAS
        }
        else if (arg == "--main-gpu" || arg == "-mg")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
#ifdef GGML_USE_CUBLAS
            params.main_gpu = std::stoi(argv[i]);
#else
            LOG_WARNING("llama.cpp was compiled without cuBLAS. It is not possible to set a main GPU.", {});
#endif
        }
        else if (arg == "--lora")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_adapter.push_back(std::make_tuple(argv[i], 1.0f));
            params.use_mmap = false;
        }
        else if (arg == "--lora-scaled")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            const char * lora_adapter = argv[i];
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_adapter.push_back(std::make_tuple(lora_adapter, std::stof(argv[i])));
            params.use_mmap = false;
        }
        else if (arg == "--lora-base")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_base = argv[i];
        }
        else if (arg == "-v" || arg == "--verbose")
        {
#if SERVER_VERBOSE != 1
            LOG_WARNING("server.cpp is not built with verbose logging.", {});
#else
            server_verbose = true;
#endif
        }
        else if (arg == "--mlock")
        {
            params.use_mlock = true;
        }
        else if (arg == "--no-mmap")
        {
            params.use_mmap = false;
        }
        else if (arg == "--numa")
        {
            params.numa = true;
        }
        else if (arg == "--embedding")
        {
            params.embedding = true;
        } else if (arg == "-cb" || arg == "--cont-batching")
        {
            params.cont_batching = true;
        }
        else if (arg == "-np" || arg == "--parallel")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_parallel = std::stoi(argv[i]);
        } else if (arg == "-n" || arg == "--n-predict")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.n_predict = std::stoi(argv[i]);
            if(params.n_predict <= 128) { // this example don't support long prompts
                params.n_predict = 128;
            }
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            server_print_usage(argv[0], default_params, default_sparams);
            exit(1);
        }
    }

    if (invalid_param)
    {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        server_print_usage(argv[0], default_params, default_sparams);
        exit(1);
    }
}

static json format_generation_settings(llama_server_context &llama, llama_client_slot* &slot)
{
    const auto eos_bias = slot->params.logit_bias.find(llama_token_eos(llama.ctx));
    const bool ignore_eos = eos_bias != slot->params.logit_bias.end() &&
                            eos_bias->second < 0.0f && std::isinf(eos_bias->second);

    return json{
        {"n_ctx", llama.n_ctx},
        {"model", llama.params.model_alias},
        {"seed", slot->params.seed},
        {"temp", slot->params.temp},
        {"top_k", slot->params.top_k},
        {"top_p", slot->params.top_p},
        {"tfs_z", slot->params.tfs_z},
        {"typical_p", slot->params.typical_p},
        {"repeat_last_n", slot->params.repeat_last_n},
        {"repeat_penalty", slot->params.repeat_penalty},
        {"presence_penalty",slot->params.presence_penalty},
        {"frequency_penalty", slot->params.frequency_penalty},
        {"mirostat", slot->params.mirostat},
        {"mirostat_tau", slot->params.mirostat_tau},
        {"mirostat_eta", slot->params.mirostat_eta},
        {"penalize_nl", slot->params.penalize_nl},
        {"stop", slot->params.antiprompt},
        {"n_predict", slot->params.n_predict},
        // {"n_keep", slot.params.n_keep},
        {"ignore_eos", ignore_eos},
        {"stream", slot->params.stream},
        {"logit_bias", slot->params.logit_bias},
        {"n_probs", slot->params.n_probs},
        {"grammar", slot->params.grammar},
    };
}

static json format_embedding_response(llama_server_context &llama)
{
    return json{
        {"embedding", llama.getEmbedding()},
    };
}

static json format_timings(llama_server_context &llama)
{
    const auto timings = llama_get_timings(llama.ctx);

    return json{
        {"prompt_n", timings.n_p_eval},
        {"prompt_ms", timings.t_p_eval_ms},
        {"prompt_per_token_ms", timings.t_p_eval_ms / timings.n_p_eval},
        {"prompt_per_second", 1e3 / timings.t_p_eval_ms * timings.n_p_eval},

        {"predicted_n", timings.n_eval},
        {"predicted_ms", timings.t_eval_ms},
        {"predicted_per_token_ms", timings.t_eval_ms / timings.n_eval},
        {"predicted_per_second", 1e3 / timings.t_eval_ms * timings.n_eval},
    };
}

static json format_final_response(llama_server_context &llama, llama_client_slot* &slot, const std::string &content, const std::vector<completion_token_output> &probs)
{

    json res = json{
        {"content", content},
        {"stop", true},
        {"model", llama.params.model_alias},
        {"tokens_predicted", slot->num_tokens_predicted},
        {"tokens_evaluated", slot->num_prompt_tokens},
        {"generation_settings", format_generation_settings(llama, slot)},
        {"prompt", slot->prompt},
        {"truncated", slot->truncated},
        {"stopped_eos", slot->stopped_eos},
        {"stopped_word", slot->stopped_word},
        {"stopped_limit", slot->stopped_limit},
        {"stopping_word", slot->stopping_word},
        {"tokens_cached", slot->n_past},
        // {"timings", format_timings(llama)},
    };

    if (slot->params.n_probs > 0)
    {
        res["completion_probabilities"] = probs_vector_to_json(llama.ctx, probs);
    }

    return res;
}

static json format_partial_response(
    llama_server_context &llama, llama_client_slot* &slot, const std::string &content, const std::vector<completion_token_output> &probs
) {
    json res = json{
        {"content", content},
        {"stop", false},
        { "slot_id", slot->id }
    };

    if (slot->params.n_probs > 0)
    {
        res["completion_probabilities"] = probs_vector_to_json(llama.ctx, probs);
    }

    return res;
}

static json format_tokenizer_response(const std::vector<llama_token> &tokens)
{
    return json{
        {"tokens", tokens}};
}

static json format_detokenized_response(std::string content)
{
    return json{
        {"content", content}};
}

template <typename T>
static T json_value(const json &body, const std::string &key, const T &default_value)
{
    // Fallback null to default value
    return body.contains(key) && !body.at(key).is_null()
        ? body.value(key, default_value)
        : default_value;
}

static void parse_options_completion(const json &body, llama_client_slot* &slot, llama_server_context &llama)
{
    slot_params default_params;

    slot->params.stream = json_value(body, "stream", false);
    slot->params.n_predict = json_value(body, "n_predict", default_params.n_predict);
    slot->params.top_k = json_value(body, "top_k", default_params.top_k);
    slot->params.top_p = json_value(body, "top_p", default_params.top_p);
    slot->params.tfs_z = json_value(body, "tfs_z", default_params.tfs_z);
    slot->params.typical_p = json_value(body, "typical_p", default_params.typical_p);
    slot->params.repeat_last_n = json_value(body, "repeat_last_n", default_params.repeat_last_n);
    slot->params.temp = json_value(body, "temperature", default_params.temp);
    slot->params.repeat_penalty = json_value(body, "repeat_penalty", default_params.repeat_penalty);
    slot->params.presence_penalty = json_value(body, "presence_penalty", default_params.presence_penalty);
    slot->params.frequency_penalty = json_value(body, "frequency_penalty", default_params.frequency_penalty);
    slot->params.mirostat = json_value(body, "mirostat", default_params.mirostat);
    slot->params.mirostat_tau = json_value(body, "mirostat_tau", default_params.mirostat_tau);
    slot->params.mirostat_eta = json_value(body, "mirostat_eta", default_params.mirostat_eta);
    slot->params.penalize_nl = json_value(body, "penalize_nl", default_params.penalize_nl);
    //llama.params.n_keep = json_value(body, "n_keep", default_params.n_keep);
    slot->params.seed = json_value(body, "seed", default_params.seed);
    slot->params.grammar = json_value(body, "grammar", default_params.grammar);
    slot->params.n_probs = json_value(body, "n_probs", default_params.n_probs);

    if (body.count("prompt") != 0)
    {
        slot->prompt = body["prompt"];
    }
    else
    {
        slot->prompt = "";
    }

    slot->params.logit_bias.clear();
    if (json_value(body, "ignore_eos", false))
    {
        slot->params.logit_bias[llama_token_eos(llama.ctx)] = -INFINITY;
    }

    const auto &logit_bias = body.find("logit_bias");
    if (logit_bias != body.end() && logit_bias->is_array())
    {
        const int n_vocab = llama_n_vocab(llama.model);
        for (const auto &el : *logit_bias)
        {
            if (el.is_array() && el.size() == 2 && el[0].is_number_integer())
            {
                llama_token tok = el[0].get<llama_token>();
                if (tok >= 0 && tok < n_vocab)
                {
                    if (el[1].is_number())
                    {
                        slot->params.logit_bias[tok] = el[1].get<float>();
                    }
                    else if (el[1].is_boolean() && !el[1].get<bool>())
                    {
                        slot->params.logit_bias[tok] = -INFINITY;
                    }
                }
            }
        }
    }

    slot->params.antiprompt.clear();
    const auto &stop = body.find("stop");
    if (stop != body.end() && stop->is_array())
    {
        for (const auto &word : *stop)
        {
            if (!word.empty())
            {
                slot->params.antiprompt.push_back(word);
            }
        }
    }

    LOG_VERBOSE("completion parameters parsed", format_generation_settings(llama, slot));
}

// static void parse_options_infill(const json &body, llama_server_context &llama)
// {
//     if (body.count("input_prefix") != 0)
//     {
//         llama.params.input_prefix = body["input_prefix"];
//     }
//     else
//     {
//         llama.params.input_prefix = "";
//     }
//     if (body.count("input_suffix") != 0)
//     {
//         llama.params.input_suffix = body["input_suffix"];
//     }
//     else
//     {
//         llama.params.input_suffix = "";
//     }
//     parse_options_completion(body, slot, llama);
// }

static void log_server_request(const Request &req, const Response &res)
{
    LOG_INFO("request", {
                            {"remote_addr", req.remote_addr},
                            {"remote_port", req.remote_port},
                            {"status", res.status},
                            {"method", req.method},
                            {"path", req.path},
                            {"params", req.params},
                        });

    LOG_VERBOSE("request", {
                               {"request", req.body},
                               {"response", res.body},
                           });
}

static bool is_at_eob(llama_server_context &server_context, const llama_token *tokens, const size_t n_tokens) {
    return n_tokens && tokens[n_tokens-1] == llama_token_eos(server_context.ctx);
}

// Function matching type llama_beam_search_callback_fn_t.
// Custom callback example is called each time the beams lengths increase:
//  * Show progress by printing ',' following by number of convergent beam tokens if any.
//  * When all beams converge to a common prefix, they are made available in beams_state.beams[0].
//    This is also called when the stop condition is met.
//    Collect tokens into std::vector<llama_token> response which is pointed to by callback_data.

// AVOID HEADACHES unnecessaries

// static void beam_search_callback(void *callback_data, llama_beams_state beams_state) {
//     auto & llama = *static_cast<llama_server_context*>(callback_data);
//     // Mark beams as EOS as needed.
//     for (size_t i = 0 ; i < beams_state.n_beams ; ++i) {
//         llama_beam_view& beam_view = beams_state.beam_views[i];
//         if (!beam_view.eob && is_at_eob(llama, beam_view.tokens, beam_view.n_tokens)) {
//             beam_view.eob = true;
//         }
//     }
//     printf(",");  // Show progress
//     if (const size_t n = beams_state.common_prefix_length) {
//         llama.generated_token_probs.resize(llama.generated_token_probs.size() + n);
//         assert(0u < beams_state.n_beams);
//         const llama_token * tokens = beams_state.beam_views[0].tokens;
//         const auto map = [](llama_token tok) { return completion_token_output{{},tok}; };
//         std::transform(tokens, tokens + n, llama.generated_token_probs.end() - n, map);
//         printf("%zu", n);
//     }
//     fflush(stdout);
// #if 0 // DEBUG: print current beams for this iteration
//     std::cout << "\n\nCurrent beams:\n";
//     for (size_t i=0 ; i < beams_state.n_beams ; ++i) {
//         std::cout << "beams["<<i<<"]: " << ostream_beam_view{state.ctx,beams_state.beam_views[i]} << std::endl;
//     }
// #endif
// }

struct token_translator {
    llama_context * ctx;
    std::string operator()(llama_token tok) const { return llama_token_to_piece(ctx, tok); }
    std::string operator()(const completion_token_output & cto) const { return (*this)(cto.tok); }
};

static void append_to_generated_text_from_generated_token_probs(llama_server_context &llama, llama_client_slot & slot)
{
    auto & gtps = slot.generated_token_probs;
    auto translator = token_translator{llama.ctx};
    auto add_strlen = [=](size_t sum, const completion_token_output & cto) { return sum + translator(cto).size(); };
    const size_t len = std::accumulate(gtps.begin(), gtps.end(), size_t(0), add_strlen);
    if (slot.generated_text.capacity() < slot.generated_text.size() + len) {
        slot.generated_text.reserve(slot.generated_text.size() + len);
    }
    for (const completion_token_output & cto : gtps) {
        slot.generated_text += translator(cto);
    }
}

int main(int argc, char **argv)
{
    // own arguments required by this example
    gpt_params params;
    server_params sparams;

    // struct that contains llama context and inference
    llama_server_context llama;

    server_params_parse(argc, argv, sparams, params);

    if (params.model_alias == "unknown")
    {
        params.model_alias = params.model;
    }

    llama_backend_init(params.numa);

    LOG_INFO("build info", {{"build", BUILD_NUMBER},
                            {"commit", BUILD_COMMIT}});
    LOG_INFO("system info", {
                                {"n_threads", params.n_threads},
                                {"n_threads_batch", params.n_threads_batch},
                                {"total_threads", std::thread::hardware_concurrency()},
                                {"system_info", llama_print_system_info()},
                            });

    // load the model
    if (!llama.loadModel(params))
    {
        return 1;
    }

    llama.initialize();

    Server svr;

    svr.set_default_headers({{"Server", "llama.cpp"},
                             {"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "content-type"}});

    // this is only called if no index.html is found in the public --path
    svr.Get("/", [](const Request &, Response &res)
            {
        res.set_content(reinterpret_cast<const char*>(&index_html), index_html_len, "text/html");
        return false; });

    // this is only called if no index.js is found in the public --path
    svr.Get("/index.js", [](const Request &, Response &res)
            {
        res.set_content(reinterpret_cast<const char *>(&index_js), index_js_len, "text/javascript");
        return false; });

    // this is only called if no index.html is found in the public --path
    svr.Get("/completion.js", [](const Request &, Response &res)
            {
        res.set_content(reinterpret_cast<const char*>(&completion_js), completion_js_len, "application/javascript");
        return false; });

    // this is only called if no index.html is found in the public --path
    svr.Get("/json-schema-to-grammar.mjs", [](const Request &, Response &res)
            {
        res.set_content(reinterpret_cast<const char*>(&json_schema_to_grammar_mjs), json_schema_to_grammar_mjs_len, "application/javascript");
        return false; });

     svr.Get("/props", [&llama](const Request & /*req*/, Response &res)
            {
                res.set_header("Access-Control-Allow-Origin", "*");
                json data = {
                    { "user_name", llama.user_name.c_str() },
                    { "assistant_name", llama.assistant_name.c_str() }
                };
                res.set_content(data.dump(), "application/json"); });

    svr.Post("/completion", [&llama](const Request &req, Response &res)
             {
        //auto lock = llama.lock();

        json data = json::parse(req.body);

        llama_client_slot* slot = llama.getSlot(json_value(data, "slot_id", -1));

        if(slot == nullptr) {
            LOG_TEE("slot unavailable\n");
            res.status = 404;
            res.set_content("slot_error", "text/plain");
            return;
        }

        if(data.contains("system_prompt")) {
            llama.processSystemPromptData(data["system_prompt"]);
        }

        // llama_reset_timings(llama.ctx);

        slot->reset();

        parse_options_completion(json::parse(req.body), slot, llama);

        if (!llama.launchSlot(slot))
        {
            res.status = 400;
            return;
        }

        if (!slot->params.stream) {
            // if (llama.params.n_beams) {
            //     // Fill llama.generated_token_probs vector with final beam.
            //     llama_beam_search(llama.ctx, beam_search_callback, &llama, llama.params.n_beams,
            //                         llama.n_past, llama.n_remain);
            //     // Translate llama.generated_token_probs to llama.generated_text.
            //     append_to_generated_text_from_generated_token_probs(llama);
            // } else {
            //     size_t stop_pos = std::string::npos;

            //     while (llama.has_next_token) {
            //         const completion_token_output token_with_probs = llama.doCompletion();
            //         const std::string token_text = token_with_probs.tok == -1 ? "" : llama_token_to_piece(llama.ctx, token_with_probs.tok);

            //         stop_pos = llama.findStoppingStrings(llama.generated_text,
            //             token_text.size(), STOP_FULL);
            //     }

            //     if (stop_pos == std::string::npos) {
            //         stop_pos = llama.findStoppingStrings(llama.generated_text, 0, STOP_PARTIAL);
            //     }
            //     if (stop_pos != std::string::npos) {
            //         llama.generated_text.erase(llama.generated_text.begin() + stop_pos,
            //             llama.generated_text.end());
            //     }
            // }

            // auto probs = llama.generated_token_probs;
            // if (llama.params.n_probs > 0 && llama.stopped_word) {
            //     const std::vector<llama_token> stop_word_toks = llama_tokenize(llama.ctx, llama.stopping_word, false);
            //     probs = std::vector<completion_token_output>(llama.generated_token_probs.begin(), llama.generated_token_probs.end() - stop_word_toks.size());
            // }

            // const json data = format_final_response(llama, llama.generated_text, probs);

            // llama_print_timings(llama.ctx);

            // res.set_content(data.dump(-1, ' ', false, json::error_handler_t::replace),
            //                 "application/json");
        } else {
            auto chunked_content_provider = [&](size_t /*offset*/, DataSink &sink) {
                    size_t sent_count = 0;
                    size_t sent_token_probs_index = 0;
                    while(slot->isProcessing()) {
                        if(slot->hasNewToken()) { // new token notification
                            const completion_token_output token = slot->next();
                            std::string token_str = llama_token_to_piece(llama.ctx, token.tok);

                            std::vector<completion_token_output> probs_output = {};

                            const json data = format_partial_response(llama, slot, token_str, probs_output);
                            const std::string str =
                                "data: " +
                                data.dump(-1, ' ', false, json::error_handler_t::replace) +
                                "\n\n";

                            LOG_VERBOSE("data stream", {
                                { "to_send", str }
                            });
                            if(!sink.write(str.c_str(), str.size())) {
                                slot->release();
                                return false;
                            }
                        } else {
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        }
                    }
                    const json data = format_final_response(
                            llama, slot,
                            "",
                            std::vector<completion_token_output>(
                                slot->generated_token_probs.begin(),
                                slot->generated_token_probs.begin() + sent_token_probs_index)
                        );

                        const std::string str =
                            "data: " +
                            data.dump(-1, ' ', false, json::error_handler_t::replace) +
                            "\n\n";

                        LOG_VERBOSE("data stream", {
                            { "to_send", str }
                        });

                        if (!sink.write(str.data(), str.size())) {
                            LOG_VERBOSE("stream closed", {});
                            llama_print_timings(llama.ctx);
                            return false;
                        }
                    sink.done();
                    return true;
                };
            auto on_complete = [&] (bool) {
                //llama.mutex.unlock();
                slot->sent_tokens = 0;
                slot->generated_token_probs.clear();
                slot->release();
            };
            //lock.release();
            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        } });

    // svr.Post("/infill", [&llama](const Request &req, Response &res)
    //          {
    //     auto lock = llama.lock();

    //     llama.rewind();

    //     llama_reset_timings(llama.ctx);

    //     parse_options_infill(json::parse(req.body), llama);

    //     if (!llama.loadGrammar())
    //     {
    //         res.status = 400;
    //         return;
    //     }
    //     llama.loadInfill();
    //     llama.beginCompletion();
    //     const auto chunked_content_provider = [&](size_t, DataSink & sink) {
    //         size_t sent_count = 0;
    //         size_t sent_token_probs_index = 0;

    //         while (llama.has_next_token) {
    //             const completion_token_output token_with_probs = llama.doCompletion();
    //             if (token_with_probs.tok == -1 || llama.multibyte_pending > 0) {
    //                 continue;
    //             }
    //             const std::string token_text = llama_token_to_piece(llama.ctx, token_with_probs.tok);

    //             size_t pos = std::min(sent_count, llama.generated_text.size());

    //             const std::string str_test = llama.generated_text.substr(pos);
    //             bool is_stop_full = false;
    //             size_t stop_pos =
    //                 llama.findStoppingStrings(str_test, token_text.size(), STOP_FULL);
    //             if (stop_pos != std::string::npos) {
    //                 is_stop_full = true;
    //                 llama.generated_text.erase(
    //                     llama.generated_text.begin() + pos + stop_pos,
    //                     llama.generated_text.end());
    //                 pos = std::min(sent_count, llama.generated_text.size());
    //             } else {
    //                 is_stop_full = false;
    //                 stop_pos = llama.findStoppingStrings(str_test, token_text.size(),
    //                     STOP_PARTIAL);
    //             }

    //             if (
    //                 stop_pos == std::string::npos ||
    //                 // Send rest of the text if we are at the end of the generation
    //                 (!llama.has_next_token && !is_stop_full && stop_pos > 0)
    //             ) {
    //                 const std::string to_send = llama.generated_text.substr(pos, std::string::npos);

    //                 sent_count += to_send.size();

    //                 std::vector<completion_token_output> probs_output = {};

    //                 if (llama.params.n_probs > 0) {
    //                     const std::vector<llama_token> to_send_toks = llama_tokenize(llama.ctx, to_send, false);
    //                     size_t probs_pos = std::min(sent_token_probs_index, llama.generated_token_probs.size());
    //                     size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), llama.generated_token_probs.size());
    //                     if (probs_pos < probs_stop_pos) {
    //                         probs_output = std::vector<completion_token_output>(llama.generated_token_probs.begin() + probs_pos, llama.generated_token_probs.begin() + probs_stop_pos);
    //                     }
    //                     sent_token_probs_index = probs_stop_pos;
    //                 }

    //                 const json data = format_partial_response(llama, to_send, probs_output);

    //                 const std::string str =
    //                     "data: " +
    //                     data.dump(-1, ' ', false, json::error_handler_t::replace) +
    //                     "\n\n";

    //                 LOG_VERBOSE("data stream", {
    //                     { "to_send", str }
    //                 });

    //                 if (!sink.write(str.data(), str.size())) {
    //                     LOG_VERBOSE("stream closed", {});
    //                     llama_print_timings(llama.ctx);
    //                     return false;
    //                 }
    //             }

    //             if (!llama.has_next_token) {
    //                 // Generation is done, send extra information.
    //                 const json data = format_final_response(
    //                     llama,
    //                     "",
    //                     std::vector<completion_token_output>(llama.generated_token_probs.begin(), llama.generated_token_probs.begin() + sent_token_probs_index)
    //                 );

    //                 const std::string str =
    //                     "data: " +
    //                     data.dump(-1, ' ', false, json::error_handler_t::replace) +
    //                     "\n\n";

    //                 LOG_VERBOSE("data stream", {
    //                     { "to_send", str }
    //                 });

    //                 if (!sink.write(str.data(), str.size())) {
    //                     LOG_VERBOSE("stream closed", {});
    //                     llama_print_timings(llama.ctx);
    //                     return false;
    //                 }
    //             }
    //         }

    //         llama_print_timings(llama.ctx);
    //         sink.done();
    //         return true;
    //     };
    //     const auto on_complete = [&](bool) {
    //         llama.mutex.unlock();
    //     };
    //     lock.release();
    //     res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
    //     });

    // svr.Get("/model.json", [&llama](const Request &, Response &res)
    //         {
    //     const json data = format_generation_settings(llama);
    //     return res.set_content(data.dump(), "application/json"); });

    svr.Options(R"(/.*)", [](const Request &, Response &res)
                { return res.set_content("", "application/json"); });

    svr.Post("/tokenize", [&llama](const Request &req, Response &res)
             {
        auto lock = llama.lock();

        const json body = json::parse(req.body);
        std::vector<llama_token> tokens;
        if (body.count("content") != 0)
        {
            tokens = llama.tokenize(body["content"], false);
        }
        const json data = format_tokenizer_response(tokens);
        return res.set_content(data.dump(), "application/json"); });

    svr.Post("/detokenize", [&llama](const Request &req, Response &res)
             {
        auto lock = llama.lock();

        const json body = json::parse(req.body);
        std::string content;
        if (body.count("tokens") != 0)
        {
            const std::vector<llama_token> tokens = body["tokens"];
            content = tokens_to_str(llama.ctx, tokens.cbegin(), tokens.cend());
        }

        const json data = format_detokenized_response(content);
        return res.set_content(data.dump(), "application/json"); });

    // svr.Post("/embedding", [&llama](const Request &req, Response &res)
    //          {
    //     auto lock = llama.lock();

    //     const json body = json::parse(req.body);

    //     llama.rewind();
    //     llama_reset_timings(llama.ctx);
    //     if (body.count("content") != 0)
    //     {
    //         llama.prompt = body["content"];
    //     }
    //     else
    //     {
    //         llama.prompt = "";
    //     }
    //     llama.params.n_predict = 0;
    //     llama.loadPrompt();
    //     llama.beginCompletion();
    //     llama.doCompletion();

    //     const json data = format_embedding_response(llama);
    //     return res.set_content(data.dump(), "application/json"); });

    svr.set_logger(log_server_request);

    svr.set_exception_handler([](const Request &, Response &res, std::exception_ptr ep)
                              {
        const char fmt[] = "500 Internal Server Error\n%s";
        char buf[BUFSIZ];
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception & e) {
            snprintf(buf, sizeof(buf), fmt, e.what());
        } catch (...) {
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/plain");
        res.status = 500; });

    svr.set_error_handler([](const Request &, Response &res)
                          {
        if (res.status == 400) {
            res.set_content("Invalid request", "text/plain");
        } else if (res.status != 500) {
            res.set_content("File Not Found", "text/plain");
            res.status = 404;
        } });

    // set timeouts and change hostname and port
    svr.set_read_timeout(sparams.read_timeout);
    svr.set_write_timeout(sparams.write_timeout);

    if (!svr.bind_to_port(sparams.hostname, sparams.port))
    {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n", sparams.hostname.c_str(), sparams.port);
        return 1;
    }

    // Set the base directory for serving static files
    svr.set_base_dir(sparams.public_path);

    // to make it ctrl+clickable:
    printf("\nllama server listening at http://%s:%d\n\n", sparams.hostname.c_str(), sparams.port);

    LOG_INFO("HTTP server listening", {
                                          {"hostname", sparams.hostname},
                                          {"port", sparams.port},
                                      });
    std::thread t([&llama]()
             {
            bool running = true;
            while (running)
            {
                running = llama.updateSlots();
            } });
    if (!svr.listen_after_bind())
    {
        return 1;
    }
    llama_backend_free();
    return 0;
}
