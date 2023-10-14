#include "common.h"
#include "llama.h"
#include "build-info.h"
#include "grammar-parser.h"

// #define SERVER_MULTIMODAL_SUPPORT

#ifdef SERVER_MULTIMODAL_SUPPORT
#include "../llava/clip.h"
#include "stb_image.h"
#endif

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
    int32_t port = 8080;
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

#ifdef SERVER_MULTIMODAL_SUPPORT
static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static inline bool is_base64(uint8_t c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::vector<uint8_t> base64_decode(std::string const& encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  BYTE char_array_4[4], char_array_3[3];
  std::vector<uint8_t> ret;
  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64_chars.find(char_array_4[i]);

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++)
          ret.push_back(char_array_3[i]);
      i = 0;
    }
  }

  if (i) {
    for (j = i; j <4; j++)
      char_array_4[j] = 0;

    for (j = 0; j <4; j++)
      char_array_4[j] = base64_chars.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
  }

  return ret;
}
#endif

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
    uint32_t seed                 = -1;   // RNG seed
    int32_t n_predict             = -1;   // new tokens to predict
    std::string grammar           = "";  // optional BNF-like grammar to constrain sampling
    bool cache_prompt =           false;  // remember a the prompt to avoid reprocessing all prompt
    std::vector<std::string> antiprompt;
    json                    input_prefix;
    json                    input_suffix;
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

#ifdef SERVER_MULTIMODAL_SUPPORT
struct slot_image {
    clip_image_u8 img_data;
    bool request_encode_image = false;
    float* image_embedding = nullptr;
    int image_tokens = 0;
    int id;
    std::string prefix_prompt = ""; // before of this image
};
#endif

struct llama_client_slot
{
    int id;
    // generation props
    int32_t n_past = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;
    int32_t num_prompt_tokens = 0;
    int32_t num_prompt_tokens_processed = 0;
    int32_t n_remaining = -1;

    json prompt;
    std::string generated_text = "";
    int num_tokens_predicted = 0;
    llama_token sampled;
    std::vector<llama_token> cache_tokens;
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
    size_t sent_count = 0;
    bool infill = false;
    int64_t t_start_process_prompt;
    int64_t t_start_genereration;

    double t_prompt_processing; // ms
    double t_token_generation; // ms

    struct slot_params params;
    struct llama_sampling_params sparams;
    llama_sampling_context ctx_sampling;

    // grammar props
    grammar_parser::parse_state parsed_grammar;
    llama_grammar *grammar = nullptr;

#ifdef SERVER_MULTIMODAL_SUPPORT
    std::vector<slot_image> images;
#endif

    void reset() {
        num_prompt_tokens = 0;
        generated_text = "";
        truncated = false;
        stopped_eos = false;
        stopped_word = false;
        stopped_limit = false;
        stopping_word = "";
        multibyte_pending = 0;
        n_past = 0;
        sent_count = 0;
        infill = false;
        clean_tokens();

        if (grammar != nullptr) {
            llama_grammar_free(grammar);
            grammar = nullptr;
            ctx_sampling.params = sparams;
            ctx_sampling.grammar = NULL;
        }

        // llama_set_rng_seed(ctx, params.seed); in batched the seed matter???????
    }

    bool loadGrammar(llama_token eos)
    {
        if (!params.grammar.empty()) {
            parsed_grammar = grammar_parser::parse(params.grammar.c_str());
            // will be empty (default) if there are parse errors
            if (parsed_grammar.rules.empty()) {
                LOG_ERROR("grammar parse error", {{"grammar", params.grammar}});
                return false;
            }
            grammar_parser::print_grammar(stderr, parsed_grammar);

            {
                auto it = sparams.logit_bias.find(eos);
                if (it != sparams.logit_bias.end() && it->second == -INFINITY) {
                    LOG_WARNING("EOS token is disabled, which will cause most grammars to fail", {});
                }
            }

            std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
            grammar = llama_grammar_init(
                grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
        }
        ctx_sampling.params = sparams;
        ctx_sampling.grammar = grammar;
        return true;
    }

    bool hasBudget(gpt_params &global_params) {
        n_remaining = -1;
        if(params.n_predict != -1) {
            n_remaining = params.n_predict - n_decoded;
        } else if(global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }
        return n_remaining > 0 || n_remaining == -1; // no budget || limitless
    }

    bool hasNewToken() {
        return num_tokens_predicted > sent_tokens;
    }

    bool available() {
        return state == IDLE && command == NONE;
    }

    bool isProcessing() {
        return (state == IDLE || state == SLEEPING) && command == LOAD_PROMPT || state == PROCESSING;
    }

    completion_token_output next() {
        completion_token_output tkn = generated_token_probs.at(sent_tokens);
        sent_tokens++;
        return tkn;
    }

    void addTokenString(completion_token_output token) {
        if(command == RELEASE) {
            num_tokens_predicted = 0;
            return;
        }
        cache_tokens.push_back(token.tok);
        generated_token_probs.push_back(token);
        num_tokens_predicted++;
    }

    void release() {
        if(state == PROCESSING) {
            t_token_generation = (ggml_time_us() - t_start_genereration) / 1e3;
            command = RELEASE;
        }
    }

    void clean_tokens() {
        sent_tokens = 0;
        generated_token_probs.clear();
        num_tokens_predicted = 0;
    }
};

struct llama_server_context
{
    std::vector<llama_client_slot> slots;

    // system prompt
    std::string system_prompt = "";
    bool update_system_prompt = false;
    std::vector<llama_token> tokens_system;
    int32_t num_tokens_system;

    // broadcast to all clients to keep the same prompt format
    std::string user_name = ""; // this should be the anti prompt
    std::string assistant_name = ""; // this is for generate the prompt

#ifdef SERVER_MULTIMODAL_SUPPORT
    bool multimodal = false;
    clip_ctx *clp_ctx = nullptr;
    int n_embd;
#endif

    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_batch batch;
    std::vector<llama_token_data> candidates;
    bool all_slots_are_idle = false;
    gpt_params params;
    int n_ctx;
    int n_vocab;
    bool clean_kv_cache = true;

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
#ifdef SERVER_MULTIMODAL_SUPPORT
        if(!params.mmproj.empty()) {
            multimodal = true;
            LOG_TEE("Multi Modal Mode Enabled");
            clp_ctx = clip_model_load(params.mmproj.c_str(), /*verbosity=*/ 1);
            if(clp_ctx == nullptr) {
                LOG_ERROR("unable to load clip model", {{"model", params.mmproj}});
                return false;
            }

             if(params.n_ctx < 2048) { // request larger context for the image embedding
                params.n_ctx = 2048;
            }
        }
#endif
        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        if (model == nullptr)
        {
            LOG_ERROR("unable to load model", {{"model", params.model}});
            return false;
        }

#ifdef SERVER_MULTIMODAL_SUPPORT
        if(multimodal) {
            int n_img_embd = clip_n_mmproj_embd(clp_ctx);
            n_embd = llama_n_embd(model);
            if (n_img_embd != n_embd) {
                LOG_TEE("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_img_embd, n_embd);
                llama_free(ctx);
                llama_free_model(model);
                return false;
            }
        }
#endif
        n_ctx = llama_n_ctx(ctx);
        n_vocab = llama_n_vocab(model);
        candidates.reserve(n_vocab);
        return true;
    }

    void initialize() {
        // create slots
        LOG_TEE("Available slots:\n");
        all_slots_are_idle = true;
        for (int i = 0; i < params.n_parallel; i++)
        {
            llama_client_slot slot;
            slot.id = i;
            slot.last_n_tokens.resize(n_ctx); // a slot can fill context size
            std::fill(slot.last_n_tokens.begin(), slot.last_n_tokens.end(), 0);
            slot.reset();
            LOG_TEE(" -> Slot %i\n", slot.id);
            slots.push_back(slot);
        }
        batch = llama_batch_init(n_ctx, 0);
        // empty system prompt
        system_prompt = "";
        num_tokens_system = 0;
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
        if(!slot->loadGrammar(llama_token_eos(ctx))) {
            return false;
        }
        all_slots_are_idle = false;
        slot->command = LOAD_PROMPT;
        LOG_TEE("slot %i is processing\n", slot->id);
        return true;
    }

    void cleanKVCache() {
        // clear the entire KV cache
        for (int i = 0; i < params.n_parallel; ++i)
        {
            llama_kv_cache_seq_rm(ctx, i, 0, -1);
        }
        clean_kv_cache = false;
    }

    void updateSystemPrompt() {
        tokens_system = ::llama_tokenize(ctx, system_prompt, true);
        num_tokens_system = tokens_system.size();

        batch.n_tokens = num_tokens_system;

        cleanKVCache();

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
            llama_kv_cache_seq_cp(ctx, 0, i, 0, num_tokens_system);
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
        system_prompt = sys_props.value("prompt", "");
        user_name = sys_props.value("anti_prompt", "");
        assistant_name = sys_props.value("assistant_name", "");
        if(slots.size() > 0) {
            notifySystemPromptChanged();
        } else {
            update_system_prompt = true;
        }
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

    size_t findStoppingStrings(const std::string &text, const size_t last_token_size,
                               const stop_type type, llama_client_slot &slot)
    {
        size_t stop_pos = std::string::npos;
        for (const std::string &word : slot.params.antiprompt)
        {
            size_t pos;
            if (type == STOP_FULL)
            {
                const size_t tmp = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
                pos = text.find(word, from_pos);
            }
            else
            {
                pos = find_partial_stop_string(word, text);
            }
            if (pos != std::string::npos &&
                (stop_pos == std::string::npos || pos < stop_pos))
            {
                stop_pos = pos;
                slot.stopped_word = true;
                slot.stopping_word = word;
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

        // search stop word and delete it
        slot.generated_text += token_str;
        size_t pos = std::min(slot.sent_count, slot.generated_text.size());
        const std::string str_test = slot.generated_text.substr(pos);
        bool is_stop_full = false;
        size_t stop_pos = findStoppingStrings(str_test, token_str.size(), STOP_FULL, slot);
        if (stop_pos != std::string::npos) {
            is_stop_full = true;
            slot.generated_text.erase(
                slot.generated_text.begin() + pos + stop_pos,
                slot.generated_text.end());
            pos = std::min(slot.sent_count, slot.generated_text.size());
        } else {
            is_stop_full = false;
            stop_pos = findStoppingStrings(str_test, token_str.size(),
                STOP_PARTIAL, slot);
        }

        // check if there is any token to predict
        bool has_next_token = !is_stop_full && stop_pos > 0;
        if(stop_pos == std::string::npos) {
            // no send the stop word in the response
            result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
            slot.sent_count += result.text_to_send.size();
            has_next_token = true;
        }
        // add the token to slot queue and cache
        slot.addTokenString(result);
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

        // check the limits
        if (
            slot.n_decoded > 2 && has_next_token && !slot.hasBudget(params))
        {
            slot.stopped_limit = true;
            has_next_token = false;
        }

        if (!slot.cache_tokens.empty() && result.tok == llama_token_eos(ctx)){
                slot.stopped_eos = true;
                has_next_token = false;
                LOG_VERBOSE("eos token found", {});
        }

        LOG_VERBOSE("next token", {
                                      {"token", result.tok},
                                      {"token_text", tokens_to_output_formatted_string(ctx, result.tok)},
                                      {"has_next_token", has_next_token},
                                      {"n_remain", slot.n_remaining},
                                      {"num_tokens_predicted", slot.num_tokens_predicted},
                                      {"stopped_eos", slot.stopped_eos},
                                      {"stopped_word", slot.stopped_word},
                                      {"stopped_limit", slot.stopped_limit},
                                      {"stopping_word", slot.stopping_word},
                                  });
        return has_next_token; // continue
    }

#ifdef SERVER_MULTIMODAL_SUPPORT
    bool processImages(llama_client_slot &slot) {
        for(slot_image &img : slot.images) {
            if(!img.request_encode_image) {
                continue;
            }
            clip_image_f32 img_res;
            if (!clip_image_preprocess(clp_ctx, &img.img_data, &img_res, /*pad2square =*/ true)) {
                LOG_TEE("Error processing the given image");
                clip_free(clp_ctx);
                return false;
            }
            img.image_tokens = clip_n_patches(clp_ctx);
            img.image_embedding = (float *)malloc(clip_embd_nbytes(clp_ctx));
            if (!img.image_embedding) {
                LOG_TEE("Unable to allocate memory for image embeddings\n");
                clip_free(clp_ctx);
                return false;
            }
            LOG_TEE("slot %i - encoding image %i\n", slot.id, img.id);
            if (!clip_image_encode(clp_ctx, params.n_threads, &img_res, img.image_embedding)) {
                LOG_TEE("Unable to encode image\n");
                return false;
            }
            img.request_encode_image = false;
        }
        return slot.images.size() > 0;
    }

    // for multiple images processing
    bool ingestImages(llama_client_slot &slot, int n_batch) {
        int image_idx = 0;
        while(image_idx < slot.images.size()) {
            slot_image img = slot.images[image_idx];

            // process prefix prompt
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
                if (llama_decode(ctx, batch_view)) {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return false;
                }
            }

            // process image with llm
            for (int i = 0; i < img.image_tokens; i += n_batch) {
                int n_eval = img.image_tokens - i;
                if (n_eval > n_batch) {
                    n_eval = n_batch;
                }
                llama_batch batch = {int32_t(n_eval), nullptr, (img.image_embedding + i * n_embd), nullptr, nullptr, nullptr, slot.n_past, 1, 0, };
                if (llama_decode(ctx, batch)) {
                    LOG_TEE("%s : failed to eval image\n", __func__);
                    return false;
                }
                slot.n_past += n_eval;
            }
            image_idx++;

            // append prefix of next image
            batch.n_tokens = 0;
            std::vector<llama_token> append_tokens = tokenize(
                    image_idx >= slot.images.size() ? slot.params.input_suffix : // no more images, then process suffix prompt
                    slot.images[image_idx].prefix_prompt, true); // has next image
            for (int i = 0; i < append_tokens.size(); ++i) {
                batch.token [batch.n_tokens] = append_tokens[i];
                batch.pos   [batch.n_tokens] = slot.n_past;
                batch.seq_id[batch.n_tokens] = slot.id;
                batch.logits[batch.n_tokens] = false;
                slot.n_past += 1;
                batch.n_tokens += 1;
            }
        }
        return true;
    }
#endif

    bool updateSlots() {
        // update the system prompt wait until all slots are idle state
        if(update_system_prompt) {
            updateSystemPrompt();
        }

        batch.n_tokens = 0;
        int kv_cache_free = (n_ctx - num_tokens_system);

        if(all_slots_are_idle) {
            if(system_prompt.empty() && clean_kv_cache) {
                cleanKVCache();
            }
            // avoid 100% usage of cpu all time
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // context shift takes effect only when there is a single slot
        if(slots.size() == 1) {
            llama_client_slot slot = slots[0];
            if (slot.isProcessing() && slot.cache_tokens.size() >= (size_t)n_ctx)
            {
                // Shift context
                const int n_left    = slot.n_past - params.n_keep - 1;
                const int n_discard = n_left / 2;

                llama_kv_cache_seq_rm   (ctx, 0, params.n_keep + 1            , params.n_keep + n_discard + 1);
                llama_kv_cache_seq_shift(ctx, 0, params.n_keep + 1 + n_discard, slot.n_past, -n_discard);

                for (size_t i = params.n_keep + 1 + n_discard; i < slot.cache_tokens.size(); i++)
                {
                    slot.cache_tokens[i - n_discard] = slot.cache_tokens[i];
                }

                slot.cache_tokens.resize(slot.cache_tokens.size() - n_discard);

                slot.n_past -= n_discard;

                slot.truncated = true;

                LOG_VERBOSE("input truncated", {
                                                {"n_ctx", n_ctx},
                                                {"n_keep", params.n_keep},
                                                {"n_left", n_left},
                                            });
            }
        }

        // decode any currently ongoing sequences
        for (auto & slot : slots) {
            // release the slot
            if (slot.state == PROCESSING && slot.command == RELEASE && !slot.hasNewToken())
            {
                slot.state = slot.params.cache_prompt ? SLEEPING : IDLE;
                if(slot.state == SLEEPING) {
                    LOG_TEE("slot %i has %i tokens in cache.\n", slot.id, slot.cache_tokens.size());
                } else {
                    LOG_TEE("slot %i released\n", slot.id);
                }
                slot.command = NONE;
                continue;
            }

            kv_cache_free -= slot.num_prompt_tokens;

            if (
                slot.state == IDLE ||
                slot.state == SLEEPING ||
                slot.command == RELEASE) {
                continue;
            }

            batch.token [batch.n_tokens] = slot.sampled;
            batch.pos   [batch.n_tokens] = num_tokens_system + slot.n_past;
            batch.seq_id[batch.n_tokens] = slot.id;
            batch.logits[batch.n_tokens] = true;

            slot.n_decoded += 1;
            slot.i_batch = batch.n_tokens;
            slot.n_past += 1;

            batch.n_tokens += 1;
        }
        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        // assign workload to the slots
        if (params.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                // need process the prompt
                if ((slot.state == IDLE || slot.state == SLEEPING) && slot.command == LOAD_PROMPT) {
                    slot.state = PROCESSING;
                    slot.command = NONE;
                    std::vector<llama_token> prompt_tokens;
                    slot.t_start_process_prompt = ggml_time_us();
                    slot.t_start_genereration = 0;

                    if(slot.infill) {
                        bool suff_rm_leading_spc = true;
                        if (params.input_suffix.find_first_of(" ") == 0 && params.input_suffix.size() > 1) {
                            params.input_suffix.erase(0, 1);
                            suff_rm_leading_spc = false;
                        }
                        auto prefix_tokens = tokenize(slot.params.input_prefix, false);
                        auto suffix_tokens = tokenize(slot.params.input_suffix, false);
                        const int space_token = 29871;
                        if (suff_rm_leading_spc  && suffix_tokens[0] == space_token) {
                            suffix_tokens.erase(suffix_tokens.begin());
                        }
                        prefix_tokens.insert(prefix_tokens.begin(), llama_token_prefix(ctx));
                        prefix_tokens.insert(prefix_tokens.begin(), llama_token_bos(ctx)); // always add BOS
                        prefix_tokens.insert(prefix_tokens.end(), llama_token_suffix(ctx));
                        prefix_tokens.insert(prefix_tokens.end(), suffix_tokens.begin(), suffix_tokens.end());
                        prefix_tokens.push_back(llama_token_middle(ctx));
                        prompt_tokens = prefix_tokens;
                    } else {
                        prompt_tokens = tokenize(slot.prompt, system_prompt.empty());  // add BOS if there isn't system prompt
                    }

                    slot.num_prompt_tokens = prompt_tokens.size();

                    slot.n_past = slot.params.cache_prompt ? common_part(slot.cache_tokens, prompt_tokens) : 0;

                    slot.cache_tokens = prompt_tokens;

                    if (slot.n_past == slot.num_prompt_tokens) {
                        // we have to evaluate at least 1 token to generate logits.
                        printf("we have to evaluate at least 1 token to generate logits\n");
                        slot.n_past--;
                    }

                    slot.num_prompt_tokens_processed = slot.num_prompt_tokens - slot.n_past;

                    if(!slot.params.cache_prompt) {
                        std::fill(slot.last_n_tokens.begin(), slot.last_n_tokens.end(), 0);
                    } else {
                        LOG_TEE("slot %i - in cache: %i tokens | to process: %i tokens\n", slot.id, slot.n_past, slot.num_prompt_tokens_processed);
                        //if input prompt is too big, truncate like normal
                        if (slot.num_prompt_tokens >= (size_t)n_ctx)
                        {
                            const int n_left = (n_ctx - params.n_keep) / 2;
                            std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
                            const int erased_blocks = (slot.num_prompt_tokens - params.n_keep - n_left - 1) / n_left;
                            new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
                            std::copy(prompt_tokens.end() - n_ctx, prompt_tokens.end(), slot.last_n_tokens.begin());

                            LOG_VERBOSE("input truncated", {
                                                            {"n_ctx", n_ctx},
                                                            {"n_keep", params.n_keep},
                                                            {"n_left", n_left},
                                                            {"new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend())},
                                                        });

                            slot.truncated = true;
                            prompt_tokens = new_tokens;
                        } else {
                            const size_t ps = slot.num_prompt_tokens;
                            std::fill(slot.last_n_tokens.begin(), slot.last_n_tokens.end()  - ps, 0);
                            std::copy(prompt_tokens.begin(), prompt_tokens.end(), slot.last_n_tokens.end() - ps);
                        }
                    }

                    llama_kv_cache_seq_rm(ctx, slot.id, num_tokens_system + slot.n_past, -1);

                    LOG_VERBOSE("prompt ingested", {
                                                    {"n_past", slot.n_past},
                                                    {"cached", tokens_to_str(ctx, slot.cache_tokens.cbegin(), slot.cache_tokens.cbegin() + slot.n_past)},
                                                    {"to_eval", tokens_to_str(ctx, slot.cache_tokens.cbegin() + slot.n_past, slot.cache_tokens.cend())},
                                                });

#ifdef SERVER_MULTIMODAL_SUPPORT
                    bool ingest_images = processImages(slot); // has images?

                    // process the prefix of first image
                    std::vector<llama_token> prefix_tokens = ingest_images ? tokenize(slot.images[0].prefix_prompt, true) : prompt_tokens;
                    for (; slot.n_past < prefix_tokens.size(); ++slot.n_past) {
                        batch.token [batch.n_tokens] = prefix_tokens[slot.n_past];
                        batch.pos   [batch.n_tokens] = slot.n_past + num_tokens_system;
                        batch.seq_id[batch.n_tokens] = slot.id;
                        batch.logits[batch.n_tokens] = false;
                        batch.n_tokens += 1;
                    }

                    if(ingest_images && !ingestImages(slot, n_batch)) {
                        LOG_TEE("failed processing images\n");
                        return false;
                    }
#else
                    for (; slot.n_past < prompt_tokens.size(); ++slot.n_past) {
                        batch.token [batch.n_tokens] = prompt_tokens[slot.n_past];
                        batch.pos   [batch.n_tokens] = slot.n_past + num_tokens_system;
                        batch.seq_id[batch.n_tokens] = slot.id;
                        batch.logits[batch.n_tokens] = false;
                        batch.n_tokens += 1;
                    }
#endif
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

                // prompt evaluated for embedding
                if(params.embedding) {
                    slot.release();
                    slot.i_batch = -1;
                    return true;
                }

                completion_token_output result;
                const llama_token id = llama_sampling_sample(ctx, NULL, slot.ctx_sampling, slot.last_n_tokens, candidates, slot.i_batch - i);

                if (slot.n_decoded == 1) {
                    slot.t_start_genereration = ggml_time_us();
                    slot.t_prompt_processing = (slot.t_start_genereration - slot.t_start_process_prompt) / 1e3;
                }

                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
                result.tok = id;
                const int32_t n_probs = slot.sparams.n_probs;
                if (slot.sparams.temp <= 0 && n_probs > 0)
                {
                    // For llama_sample_token_greedy we need to sort candidates
                    llama_sample_softmax(ctx, &candidates_p);
                }

                for (size_t i = 0; i < std::min(candidates_p.size, (size_t)n_probs); ++i)
                {
                    result.probs.push_back({candidates_p.data[i].id, candidates_p.data[i].p});
                }

                if (!processToken(result, slot)) {
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

struct server_beam_search_callback_data {
    llama_context * ctx;
    llama_client_slot * slot;
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
    printf("  -np N, --parallel N   number of slots for process requests (default: %d)\n", params.n_parallel);
    printf("  -cb, --cont-batching  enable continuous batching (a.k.a dynamic batching) (default: disabled)\n");
    printf("    -spf FNAME, --system-prompt-file FNAME\n");
    printf("                        Set a file to load a system prompt (initial prompt of all slots), this is useful for chat applications.\n");
#ifdef SERVER_MULTIMODAL_SUPPORT
    printf("  --mmproj MMPROJ_FILE  path to a multimodal projector file for LLaVA.\n");
#endif
    printf("\n");
}

static void server_params_parse(int argc, char **argv, server_params &sparams,
                                gpt_params &params, llama_server_context& llama)
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
        } else if (arg == "-spf" || arg == "--system-prompt-file")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            std::ifstream file(argv[i]);
            if (!file) {
                fprintf(stderr, "error: failed to open file '%s'\n", argv[i]);
                invalid_param = true;
                break;
            }
            std::string systm_content = "";
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(systm_content)
            );
            llama.processSystemPromptData(json::parse(systm_content));
        }
#ifdef SERVER_MULTIMODAL_SUPPORT
        else if(arg == "--mmproj") {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.mmproj = argv[i];
        }
#endif
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

static json format_generation_settings(llama_server_context &llama, llama_client_slot* slot)
{
    const auto eos_bias = slot->sparams.logit_bias.find(llama_token_eos(llama.ctx));
    const bool ignore_eos = eos_bias != slot->sparams.logit_bias.end() &&
                            eos_bias->second < 0.0f && std::isinf(eos_bias->second);

    return json{
        {"n_ctx", llama.n_ctx},
        {"model", llama.params.model_alias},
        {"seed", slot->params.seed},
        {"temp", slot->sparams.temp},
        {"top_k", slot->sparams.top_k},
        {"top_p", slot->sparams.top_p},
        {"tfs_z", slot->sparams.tfs_z},
        {"typical_p", slot->sparams.typical_p},
        {"repeat_last_n", slot->sparams.repeat_last_n},
        {"repeat_penalty", slot->sparams.repeat_penalty},
        {"presence_penalty",slot->sparams.presence_penalty},
        {"frequency_penalty", slot->sparams.frequency_penalty},
        {"mirostat", slot->sparams.mirostat},
        {"mirostat_tau", slot->sparams.mirostat_tau},
        {"mirostat_eta", slot->sparams.mirostat_eta},
        {"penalize_nl", slot->sparams.penalize_nl},
        {"stop", slot->params.antiprompt},
        {"n_predict", slot->params.n_predict},
        // {"n_keep", slot.params.n_keep},
        {"ignore_eos", ignore_eos},
        {"stream", slot->params.stream},
        {"logit_bias", slot->sparams.logit_bias},
        {"n_probs", slot->sparams.n_probs},
        {"grammar", slot->params.grammar},
    };
}

static json format_embedding_response(llama_server_context &llama)
{
    return json{
        {"embedding", llama.getEmbedding()},
    };
}

static json format_timings(llama_client_slot* slot)
{
    return json{
        {"prompt_n", slot->num_prompt_tokens_processed},
        {"prompt_ms", slot->t_prompt_processing},
        {"prompt_per_token_ms",slot->t_prompt_processing / slot->num_prompt_tokens_processed},
        {"prompt_per_second", 1e3 / slot->t_prompt_processing * slot->num_prompt_tokens_processed},

        {"predicted_n", slot->n_decoded},
        {"predicted_ms", slot->t_token_generation},
        {"predicted_per_token_ms",slot->t_token_generation / slot->n_decoded},
        {"predicted_per_second", 1e3 / slot->t_token_generation * slot->n_decoded},
    };
}

static json format_final_response(llama_server_context &llama, llama_client_slot* slot, const std::string &content, const std::vector<completion_token_output> &probs)
{

    json res = json{
        {"content", content},
        {"slot_id", slot->id},
        {"stop", true},
        {"model", llama.params.model_alias},
        {"tokens_predicted", slot->n_decoded},
        {"tokens_evaluated", slot->num_prompt_tokens},
        {"generation_settings", format_generation_settings(llama, slot)},
        {"prompt", slot->prompt},
        {"truncated", slot->truncated},
        {"stopped_eos", slot->stopped_eos},
        {"stopped_word", slot->stopped_word},
        {"stopped_limit", slot->stopped_limit},
        {"stopping_word", slot->stopping_word},
        {"tokens_cached", slot->n_past},
        {"timings", format_timings(slot)}
    };

    if (slot->sparams.n_probs > 0)
    {
        res["completion_probabilities"] = probs_vector_to_json(llama.ctx, probs);
    }

    return res;
}

static json format_partial_response(
    llama_server_context &llama, llama_client_slot* slot, const std::string &content, const std::vector<completion_token_output> &probs
) {
    json res = json{
        {"content", content },
        {"stop", false},
        { "slot_id", slot->id },
#ifdef SERVER_MULTIMODAL_SUPPORT
        {"multimodal", llama.multimodal }
#else
        {"multimodal", false }
#endif
    };

    if (slot->sparams.n_probs > 0)
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

static void parse_options_completion(const json &body, llama_client_slot* slot, llama_server_context &llama)
{
    slot_params default_params;
    llama_sampling_params default_sparams;

    slot->params.stream = json_value(body, "stream", false);
    slot->params.cache_prompt = json_value(body, "cache_prompt", false);
    slot->params.n_predict = json_value(body, "n_predict", default_params.n_predict);
    slot->sparams.top_k = json_value(body, "top_k", default_sparams.top_k);
    slot->sparams.top_p = json_value(body, "top_p", default_sparams.top_p);
    slot->sparams.tfs_z = json_value(body, "tfs_z", default_sparams.tfs_z);
    slot->sparams.typical_p = json_value(body, "typical_p", default_sparams.typical_p);
    slot->sparams.repeat_last_n = json_value(body, "repeat_last_n", default_sparams.repeat_last_n);
    slot->sparams.temp = json_value(body, "temperature", default_sparams.temp);
    slot->sparams.repeat_penalty = json_value(body, "repeat_penalty", default_sparams.repeat_penalty);
    slot->sparams.presence_penalty = json_value(body, "presence_penalty", default_sparams.presence_penalty);
    slot->sparams.frequency_penalty = json_value(body, "frequency_penalty", default_sparams.frequency_penalty);
    slot->sparams.mirostat = json_value(body, "mirostat", default_sparams.mirostat);
    slot->sparams.mirostat_tau = json_value(body, "mirostat_tau", default_sparams.mirostat_tau);
    slot->sparams.mirostat_eta = json_value(body, "mirostat_eta", default_sparams.mirostat_eta);
    slot->sparams.penalize_nl = json_value(body, "penalize_nl", default_sparams.penalize_nl);
    llama.params.n_keep = json_value(body, "n_keep", -1);
    slot->params.seed = json_value(body, "seed", default_params.seed);
    slot->params.grammar = json_value(body, "grammar", default_params.grammar);
    slot->sparams.n_probs = json_value(body, "n_probs", default_sparams.n_probs);

    if (body.count("prompt") != 0)
    {
        slot->prompt = body["prompt"];
    }
    else
    {
        slot->prompt = "";
    }

    slot->sparams.logit_bias.clear();
    if (json_value(body, "ignore_eos", false))
    {
        slot->sparams.logit_bias[llama_token_eos(llama.ctx)] = -INFINITY;
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
                        slot->sparams.logit_bias[tok] = el[1].get<float>();
                    }
                    else if (el[1].is_boolean() && !el[1].get<bool>())
                    {
                        slot->sparams.logit_bias[tok] = -INFINITY;
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
#ifdef SERVER_MULTIMODAL_SUPPORT
    if(!llama.multimodal) {
        return;
    }

    const auto &images_data = body.find("image_data");
    if (images_data != body.end() && images_data->is_array())
    {
        slot->images.clear();
        for (const auto &img : *images_data)
        {
            slot_image img_sl;
            std::string data_b64 = img["data"].get<std::string>();
            img_sl.id = img.count("id") != 0 ? img["id"].get<int>() : slot->images.size();
            int width, height, channels;
            std::vector<uint8_t> image_buffer = base64_decode(data_b64);
            data_b64.clear();
            auto data = stbi_load_from_memory(image_buffer.data(), image_buffer.size(), &width, &height, &channels, 3);
            if(!data) {
                LOG_TEE("slot %i - failed to load image\n", slot->id);
                return;
            }
            LOG_TEE("slot %i - RGB image %i loaded (%i x %i)\n", slot->id, img_sl.id, width, height);
            img_sl.img_data.nx = width;
            img_sl.img_data.ny = height;
            img_sl.img_data.size = width * height * 3;
            img_sl.img_data.data = new uint8_t[width * height * 3]();
            memcpy(img_sl.img_data.data, data, width * height * 3);
            stbi_image_free(data);
            img_sl.request_encode_image = true;
            slot->images.push_back(img_sl);
        }
        // process prompt
        // example: system prompt <img-102> user <img-103> describe <img-134> -> [{id: 102, prefix: 'system prompt '}, {id: 103, prefix: ' user '}, {id: 134, prefix: ' describe '}]}
        if(slot->images.size() > 0 && !slot->prompt.is_array()) {
            std::string prompt = slot->prompt.get<std::string>();
            size_t pos = 0, begin_prefix = 0;
            std::string pattern = "[img-";
            while ((pos = prompt.find(pattern, pos)) != std::string::npos) {
                size_t end_prefix = pos;
                pos += pattern.length();
                size_t end_pos = prompt.find("]", pos);
                if (end_pos != std::string::npos) {
                    std::string image_id = prompt.substr(pos, end_pos - pos);
                    try {
                        int img_id = std::stoi(image_id);
                        bool found = false;
                        for(slot_image &img : slot->images) {
                            if(img.id == img_id) {
                                found = true;
                                img.prefix_prompt = prompt.substr(begin_prefix, end_prefix - begin_prefix);
                                begin_prefix = end_pos + 1;
                                break;
                            }
                        }
                        if(!found) {
                            LOG_TEE("ERROR: Image with id %i not found.\n", img_id);
                            slot->images.clear();
                            return;
                        }
                    } catch (const std::invalid_argument& e) {
                        LOG_TEE("Invalid image number id in prompt\n");
                        slot->images.clear();
                        return;
                    }
                }
            }
            slot->prompt = "";
            slot->params.input_suffix = prompt.substr(begin_prefix);
            slot->params.cache_prompt = false; // multimodal doesn't support cache prompt
        }
    }
#endif
}

static void parse_options_infill(const json &body, llama_server_context &llama, llama_client_slot *slot)
{
    if (body.count("input_prefix") != 0)
    {
        slot->params.input_prefix = body["input_prefix"];
    }
    else
    {
        slot->params.input_prefix = "";
    }
    if (body.count("input_suffix") != 0)
    {
        slot->params.input_suffix = body["input_suffix"];
    }
    else
    {
        slot->params.input_suffix = "";
    }
    parse_options_completion(body, slot, llama);
}

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

static bool is_at_eob(const server_beam_search_callback_data & server_context, const llama_token *tokens, const size_t n_tokens) {
    return n_tokens && tokens[n_tokens - 1] == llama_token_eos(server_context.ctx);
}

// Function matching type llama_beam_search_callback_fn_t.
// Custom callback example is called each time the beams lengths increase:
//  * Show progress by printing ',' following by number of convergent beam tokens if any.
//  * When all beams converge to a common prefix, they are made available in beams_state.beams[0].
//    This is also called when the stop condition is met.
//    Collect tokens into std::vector<llama_token> response which is pointed to by callback_data.

// NO TESTED after PR #3589

static void beam_search_callback(void *callback_data, llama_beams_state beams_state) {
    auto & llama = *static_cast<server_beam_search_callback_data*>(callback_data);
    // Mark beams as EOS as needed.
    for (size_t i = 0 ; i < beams_state.n_beams ; ++i) {
        llama_beam_view& beam_view = beams_state.beam_views[i];
        if (!beam_view.eob && is_at_eob(llama, beam_view.tokens, beam_view.n_tokens)) {
            beam_view.eob = true;
        }
    }
    printf(",");  // Show progress
    if (const size_t n = beams_state.common_prefix_length) {
        llama.slot->generated_token_probs.resize(llama.slot->generated_token_probs.size() + n);
        assert(0u < beams_state.n_beams);
        const llama_token * tokens = beams_state.beam_views[0].tokens;
        const auto map = [](llama_token tok) { return completion_token_output{{},tok}; };
        std::transform(tokens, tokens + n, llama.slot->generated_token_probs.end() - n, map);
        printf("%zu", n);
    }
    fflush(stdout);
#if 0 // DEBUG: print current beams for this iteration
    std::cout << "\n\nCurrent beams:\n";
    for (size_t i=0 ; i < beams_state.n_beams ; ++i) {
        std::cout << "beams["<<i<<"]: " << ostream_beam_view{state.ctx,beams_state.beam_views[i]} << std::endl;
    }
#endif
}

struct token_translator {
    llama_context * ctx;
    std::string operator()(llama_token tok) const { return llama_token_to_piece(ctx, tok); }
    std::string operator()(const completion_token_output & cto) const { return (*this)(cto.tok); }
};

static void append_to_generated_text_from_generated_token_probs(llama_server_context &llama, llama_client_slot* slot)
{
    auto & gtps = slot->generated_token_probs;
    auto translator = token_translator{llama.ctx};
    auto add_strlen = [=](size_t sum, const completion_token_output & cto) { return sum + translator(cto).size(); };
    const size_t len = std::accumulate(gtps.begin(), gtps.end(), size_t(0), add_strlen);
    if (slot->generated_text.capacity() < slot->generated_text.size() + len) {
        slot->generated_text.reserve(slot->generated_text.size() + len);
    }
    for (const completion_token_output & cto : gtps) {
        slot->generated_text += translator(cto);
    }
}

int main(int argc, char **argv)
{
    // own arguments required by this example
    gpt_params params;
    server_params sparams;

    // struct that contains llama context and inference
    llama_server_context llama;

    server_params_parse(argc, argv, sparams, params, llama);

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

        slot->reset();

        parse_options_completion(data, slot, llama);

        if (!llama.launchSlot(slot))
        {
            res.status = 400;
            return;
        }

        if (!slot->params.stream) {
            std::string completion_text = "";
            if (llama.params.n_beams) {
                // Fill llama.generated_token_probs vector with final beam.
                server_beam_search_callback_data data_beam;
                data_beam.slot = slot;
                data_beam.ctx = llama.ctx;
                llama_beam_search(llama.ctx, beam_search_callback, &data_beam, llama.params.n_beams,
                                     slot->n_past, llama.params.n_predict);
                // Translate llama.generated_token_probs to llama.generated_text.
                append_to_generated_text_from_generated_token_probs(llama, slot);
            } else {
                while (slot->isProcessing()) {
                    if(slot->hasNewToken()) {
                        completion_text += slot->next().text_to_send;
                    } else {
                        std::this_thread::sleep_for(std::chrono::microseconds(5));
                    }
                }
            }

            auto probs = slot->generated_token_probs;
            if (slot->sparams.n_probs > 0 && slot->stopped_word) {
                const std::vector<llama_token> stop_word_toks = llama_tokenize(llama.ctx, slot->stopping_word, false);
                probs = std::vector<completion_token_output>(slot->generated_token_probs.begin(), slot->generated_token_probs.end() - stop_word_toks.size());
            }

            const json data = format_final_response(llama, slot, completion_text, probs);
            slot->release();
            res.set_content(data.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        } else {
            const auto chunked_content_provider = [slot, &llama](size_t, DataSink & sink) {
                    size_t sent_token_probs_index = 0;
                    while(slot->isProcessing()) {
                        if(slot->hasNewToken()) { // new token notification
                            const completion_token_output token = slot->next();
                            std::vector<completion_token_output> probs_output = {};
                            if (slot->sparams.n_probs > 0) {
                                const std::vector<llama_token> to_send_toks = llama_tokenize(llama.ctx, token.text_to_send, false);
                                size_t probs_pos = std::min(sent_token_probs_index, slot->generated_token_probs.size());
                                size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), slot->generated_token_probs.size());
                                if (probs_pos < probs_stop_pos) {
                                    probs_output = std::vector<completion_token_output>(slot->generated_token_probs.begin() + probs_pos, slot->generated_token_probs.begin() + probs_stop_pos);
                                }
                                sent_token_probs_index = probs_stop_pos;
                            }
                            const json data = format_partial_response(llama, slot, token.text_to_send, probs_output);
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
                            std::this_thread::sleep_for(std::chrono::microseconds(5));
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
                        slot->release();
                        return false;
                    }
                    sink.done();
                    return true;
            };
            auto on_complete = [slot, &llama] (bool) {
                slot->release();
                slot->clean_tokens();
            };
            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        } });

    svr.Post("/infill", [&llama](const Request &req, Response &res)
             {

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

        slot->reset();
        slot->infill = true;

        parse_options_infill(data, llama, slot);

        if (!llama.launchSlot(slot))
        {
            res.status = 400;
            return;
        }

        if(!slot->params.stream) {
            std::string completion_text = "";
            while (slot->isProcessing()) {
                if(slot->hasNewToken()) {
                    completion_text += slot->next().text_to_send;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(5));
                }
            }
            auto probs = slot->generated_token_probs;
            if (slot->sparams.n_probs > 0 && slot->stopped_word) {
                const std::vector<llama_token> stop_word_toks = llama_tokenize(llama.ctx, slot->stopping_word, false);
                probs = std::vector<completion_token_output>(slot->generated_token_probs.begin(), slot->generated_token_probs.end() - stop_word_toks.size());
            }

            const json data = format_final_response(llama, slot, completion_text, probs);
            res.set_content(data.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        } else {
            const auto chunked_content_provider = [slot, &llama](size_t, DataSink & sink) {
                    size_t sent_token_probs_index = 0;
                    while(slot->isProcessing()) {
                        if(slot->hasNewToken()) { // new token notification
                            const completion_token_output token = slot->next();
                            std::vector<completion_token_output> probs_output = {};
                            if (slot->sparams.n_probs > 0) {
                                const std::vector<llama_token> to_send_toks = llama_tokenize(llama.ctx, token.text_to_send, false);
                                size_t probs_pos = std::min(sent_token_probs_index, slot->generated_token_probs.size());
                                size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), slot->generated_token_probs.size());
                                if (probs_pos < probs_stop_pos) {
                                    probs_output = std::vector<completion_token_output>(slot->generated_token_probs.begin() + probs_pos, slot->generated_token_probs.begin() + probs_stop_pos);
                                }
                                sent_token_probs_index = probs_stop_pos;
                            }
                            const json data = format_partial_response(llama, slot, token.text_to_send, probs_output);
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
                        slot->release();
                        return false;
                    }
                    sink.done();
                    return true;
            };
            auto on_complete = [slot, &llama] (bool) {
                slot->clean_tokens();
                slot->release();
            };
            res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
        }
        });

    svr.Get("/model.json", [&llama](const Request &, Response &res)
            {
        const json data = format_generation_settings(llama, llama.getSlot(0));
        return res.set_content(data.dump(), "application/json"); });

    svr.Options(R"(/.*)", [](const Request &, Response &res)
                { return res.set_content("", "application/json"); });

    svr.Post("/tokenize", [&llama](const Request &req, Response &res)
             {

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

        const json body = json::parse(req.body);
        std::string content;
        if (body.count("tokens") != 0)
        {
            const std::vector<llama_token> tokens = body["tokens"];
            content = tokens_to_str(llama.ctx, tokens.cbegin(), tokens.cend());
        }

        const json data = format_detokenized_response(content);
        return res.set_content(data.dump(), "application/json"); });

    svr.Post("/embedding", [&llama](const Request &req, Response &res)
             {
        const json body = json::parse(req.body);
        llama_client_slot* slot = llama.getSlot(-1);
        slot->reset();
        //llama_reset_timings(llama.ctx);
        if (body.count("content") != 0)
        {
            slot->prompt = body["content"];
        }
        else
        {
            slot->prompt = "";
        }
        llama.params.n_predict = 0;
        llama.launchSlot(slot);
        while(slot->isProcessing()) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        const json data = format_embedding_response(llama);
        return res.set_content(data.dump(), "application/json"); });

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
