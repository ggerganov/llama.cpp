#include "common.h"
#include "llama.h"
#include "build-info.h"

// single thread
#define CPPHTTPLIB_THREAD_POOL_COUNT 1
#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif

#include "httplib.h"
#include "json.hpp"

#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
#endif

using namespace httplib;
using json = nlohmann::json;

struct server_params {
    std::string hostname = "127.0.0.1";
    int32_t port = 8080;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
};

static size_t common_part(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}
    return i;
}

enum stop_type {
    STOP_FULL,
    STOP_PARTIAL,
};

static bool ends_with(const std::string & str, const std::string & suffix) {
    return str.size() >= suffix.size() &&
        0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static size_t find_partial_stop_string(const std::string & stop,
                                       const std::string & text) {
    if (!text.empty() && !stop.empty()) {
        const char text_last_char = text.back();
        for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
            if (stop[char_index] == text_last_char) {
                const std::string current_partial = stop.substr(0, char_index + 1);
                if (ends_with(text, current_partial)) {
                    return text.size() - char_index - 1;
                }
            }
        }
    }
    return std::string::npos;
}

template<class Iter>
static std::string tokens_to_str(llama_context * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += llama_token_to_str(ctx, *begin);
    }
    return ret;
}

static void server_log(const char * level, const char * function, int line,
                       const char * message, const nlohmann::ordered_json & extra) {
    nlohmann::ordered_json log {
        { "timestamp", time(nullptr) },
        { "level", level },
        { "function", function },
        { "line", line },
        { "message", message },
    };

    if (!extra.empty()) {
        log.merge_patch(extra);
    }

    const std::string str = log.dump(-1, ' ', false, json::error_handler_t::replace);
    fprintf(stdout, "%.*s\n", (int)str.size(), str.data());
    fflush(stdout);
}

static bool server_verbose = false;

#if SERVER_VERBOSE != 1
#  define LOG_VERBOSE(MSG, ...)
#else
#  define LOG_VERBOSE(MSG, ...)                                          \
    do {                                                                 \
        if (server_verbose) {                                            \
            server_log("VERBOSE", __func__, __LINE__, MSG, __VA_ARGS__); \
        }                                                                \
    } while(0)
#endif

#define LOG_ERROR(MSG, ...) server_log("ERROR", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_WARNING(MSG, ...) server_log("WARNING", __func__, __LINE__, MSG, __VA_ARGS__)
#define LOG_INFO(MSG, ...) server_log("INFO", __func__, __LINE__, MSG, __VA_ARGS__)

struct llama_server_context {
    bool stream = false;
    bool has_next_token = false;
    std::string generated_text;

    size_t num_tokens_predicted = 0;
    size_t n_past = 0;
    size_t n_remain = 0;

    std::vector<llama_token> embd;
    std::vector<llama_token> last_n_tokens;

    llama_context * ctx = nullptr;
    gpt_params params;

    bool truncated = false;
    bool stopped_eos = false;
    bool stopped_word = false;
    bool stopped_limit = false;
    std::string stopping_word;
    int32_t multibyte_pending = 0;

    ~llama_server_context() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
    }

    void rewind() {
        params.antiprompt.clear();
        num_tokens_predicted = 0;
        generated_text = "";
        generated_text.reserve(params.n_ctx);
        truncated = false;
        stopped_eos = false;
        stopped_word = false;
        stopped_limit = false;
        stopping_word = "";
        multibyte_pending = 0;

        n_remain = 0;
        n_past = 0;
    }

    bool loadModel(const gpt_params & params_) {
        params = params_;
        ctx = llama_init_from_gpt_params(params);
        if (ctx == nullptr) {
            LOG_ERROR("unable to load model", { { "model", params_.model } });
            return false;
        }

        last_n_tokens.resize(params.n_ctx);
        std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
        return true;
    }

    void loadPrompt() {
        params.prompt.insert(0, 1, ' '); // always add a first space
        std::vector<llama_token> prompt_tokens = ::llama_tokenize(ctx, params.prompt, true);

        if (params.n_keep < 0) {
            params.n_keep = (int)prompt_tokens.size();
        }
        params.n_keep = std::min(params.n_ctx - 4, params.n_keep);

        // if input prompt is too big, truncate like normal
        if (prompt_tokens.size() >= (size_t)params.n_ctx) {
            const int n_left = (params.n_ctx - params.n_keep) / 2;
            std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + params.n_keep);
            const int erased_blocks = (prompt_tokens.size() - params.n_keep - n_left - 1) / n_left;
            new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + params.n_keep + erased_blocks * n_left, prompt_tokens.end());
            std::copy(prompt_tokens.end() - params.n_ctx, prompt_tokens.end(), last_n_tokens.begin());

            LOG_VERBOSE("input truncated", {
                { "n_ctx", params.n_ctx },
                { "n_keep", params.n_keep },
                { "n_left", n_left },
                { "new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend()) },
            });

            truncated = true;
            prompt_tokens = new_tokens;
        } else {
            const size_t ps = prompt_tokens.size();
            std::fill(last_n_tokens.begin(), last_n_tokens.end() - ps, 0);
            std::copy(prompt_tokens.begin(), prompt_tokens.end(), last_n_tokens.end() - ps);
        }

        // compare the evaluated prompt with the new prompt
        n_past = common_part(embd, prompt_tokens);
        embd = prompt_tokens;
        if (n_past == prompt_tokens.size()) {
            // we have to evaluate at least 1 token to generate logits.
            n_past--;
        }

        LOG_VERBOSE("prompt ingested", {
            { "n_past", n_past },
            { "cached", tokens_to_str(ctx, embd.cbegin(), embd.cbegin() + n_past) },
            { "to_eval", tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()) },
        });

        has_next_token = true;
    }

    void beginCompletion() {
        // number of tokens to keep when resetting context
        n_remain = params.n_predict;
        llama_set_rng_seed(ctx, params.seed);
    }

    llama_token nextToken() {
        llama_token result = -1;

        if (embd.size() >= (size_t)params.n_ctx) {
            // Reset context
            const int n_left = (params.n_ctx - params.n_keep) / 2;

            std::vector<llama_token> new_tokens(embd.begin(), embd.begin() + params.n_keep);
            new_tokens.insert(new_tokens.end(), embd.end() - n_left, embd.end());
            embd = new_tokens;
            n_past = params.n_keep;
            truncated = true;
            LOG_VERBOSE("input truncated", {
                { "n_ctx", params.n_ctx },
                { "n_keep", params.n_keep },
                { "n_left", n_left },
                { "new_tokens", tokens_to_str(ctx, new_tokens.cbegin(), new_tokens.cend()) },
            });
        }

        while (n_past < embd.size()) {
            int n_eval = (int)embd.size() - n_past;
            if (n_eval > params.n_batch) {
                n_eval = params.n_batch;
            }
            if (llama_eval(ctx, &embd[n_past], n_eval, n_past, params.n_threads)) {
                LOG_ERROR("failed to eval", {
                    { "n_eval", n_eval },
                    { "n_past", n_past },
                    { "n_threads", params.n_threads },
                    { "embd", tokens_to_str(ctx, embd.cbegin() + n_past, embd.cend()) },
                });
                has_next_token = false;
                return result;
            }
            n_past += n_eval;
        }

        // out of user input, sample next token
        const float temp = params.temp;
        const int32_t top_k = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
        const float top_p = params.top_p;
        const float tfs_z = params.tfs_z;
        const float typical_p = params.typical_p;
        const int32_t repeat_last_n = params.repeat_last_n < 0 ? params.n_ctx : params.repeat_last_n;
        const float repeat_penalty = params.repeat_penalty;
        const float alpha_presence = params.presence_penalty;
        const float alpha_frequency = params.frequency_penalty;
        const int mirostat = params.mirostat;
        const float mirostat_tau = params.mirostat_tau;
        const float mirostat_eta = params.mirostat_eta;
        const bool penalize_nl = params.penalize_nl;
        llama_token id = 0;

        {
            auto * logits = llama_get_logits(ctx);
            auto n_vocab = llama_n_vocab(ctx);

            // Apply params.logit_bias map
            for (const auto & it : params.logit_bias) {
                logits[it.first] += it.second;
            }

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // Apply penalties
            float nl_logit = logits[llama_token_nl()];
            auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), params.n_ctx);
            llama_sample_repetition_penalty(ctx, &candidates_p,
                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                last_n_repeat, repeat_penalty);
            llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                last_n_repeat, alpha_frequency, alpha_presence);
            if (!penalize_nl) {
                logits[llama_token_nl()] = nl_logit;
            }

            if (temp <= 0) {
                // Greedy sampling
                id = llama_sample_token_greedy(ctx, &candidates_p);
            } else {
                if (mirostat == 1) {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    const int mirostat_m = 100;
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                } else if (mirostat == 2) {
                    static float mirostat_mu = 2.0f * mirostat_tau;
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                } else {
                    // Temperature sampling
                    llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                    llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                    llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                    llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                    llama_sample_temperature(ctx, &candidates_p, temp);
                    id = llama_sample_token(ctx, &candidates_p);
                }
            }
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);
            num_tokens_predicted++;
        }

        // add it to the context
        embd.push_back(id);
        result = id;
        // decrement remaining sampling budget
        --n_remain;

        if (!embd.empty() && embd.back() == llama_token_eos()) {
            //stopping_word = llama_token_to_str(ctx, embd.back());
            has_next_token = false;
            stopped_eos = true;
            LOG_VERBOSE("eos token found", {});
            return result;
        }

        has_next_token = params.n_predict == -1 || n_remain != 0;
        return result;
    }

    size_t findStoppingStrings(const std::string & text, const size_t last_token_size,
                               const stop_type type) {
        size_t stop_pos = std::string::npos;
        for (const std::string & word : params.antiprompt) {
            size_t pos;
            if (type == STOP_FULL) {
                const size_t tmp = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
                pos = text.find(word, from_pos);
            }
            else {
                pos = find_partial_stop_string(word, text);
            }
            if (pos != std::string::npos &&
                (stop_pos == std::string::npos || pos < stop_pos)) {
                if (type == STOP_FULL) {
                    stopping_word = word;
                    stopped_word = true;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }
        return stop_pos;
    }

    std::string doCompletion() {
        const llama_token token = nextToken();

        const std::string token_text = token == -1 ? "" : llama_token_to_str(ctx, token);
        generated_text += token_text;

        if (multibyte_pending > 0) {
            multibyte_pending -= token_text.size();
        } else if (token_text.size() == 1) {
            const char c = token_text[0];
            // 2-byte characters: 110xxxxx 10xxxxxx
            if ((c & 0xE0) == 0xC0) {
                multibyte_pending = 1;
            // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
            } else if ((c & 0xF0) == 0xE0) {
                multibyte_pending = 2;
            // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
            } else if ((c & 0xF8) == 0xF0) {
                multibyte_pending = 3;
            } else {
                multibyte_pending = 0;
            }
        }

        if (multibyte_pending > 0 && !has_next_token) {
            has_next_token = true;
            n_remain++;
        }

        if (!has_next_token && n_remain == 0) {
            stopped_limit = true;
        }

        LOG_VERBOSE("next token", {
            { "token", token },
            { "token_text", llama_token_to_str(ctx, token) },
            { "has_next_token", has_next_token },
            { "n_remain", n_remain },
            { "num_tokens_predicted", num_tokens_predicted },
            { "stopped_eos", stopped_eos },
            { "stopped_word", stopped_word },
            { "stopped_limit", stopped_limit },
            { "stopping_word", stopping_word },
        });

        return token_text;
    }
};

static void server_print_usage(const char * argv0, const gpt_params & params,
                               const server_params & sparams) {
    fprintf(stderr, "usage: %s [options]\n", argv0);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -v, --verbose         verbose output (default: %s)\n", server_verbose ? "enabled" : "disabled");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -c N, --ctx-size N    size of the prompt context (default: %d)\n", params.n_ctx);
    fprintf(stderr, "  -b N, --batch-size N  batch size for prompt processing (default: %d)\n", params.n_batch);
    fprintf(stderr, "  --memory-f32          use f32 instead of f16 for memory key+value (default: disabled)\n");
    fprintf(stderr, "                        not recommended: doubles context memory required and no measurable increase in quality\n");
    if (llama_mlock_supported()) {
        fprintf(stderr, "  --mlock               force system to keep model in RAM rather than swapping or compressing\n");
    }
    if (llama_mmap_supported()) {
        fprintf(stderr, "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)\n");
    }
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
    fprintf(stderr, "  -ngl N, --n-gpu-layers N\n");
    fprintf(stderr, "                        number of layers to store in VRAM\n");
    fprintf(stderr, "  -ts SPLIT --tensor-split SPLIT\n");
    fprintf(stderr, "                        how to split tensors across multiple GPUs, comma-separated list of proportions, e.g. 3,1\n");
    fprintf(stderr, "                        how to split tensors across multiple GPUs, comma-separated list of proportions, e.g. 3,1\n");
    fprintf(stderr, "  -mg i, --main-gpu i   the GPU to use for scratch and small tensors\n");
    fprintf(stderr, "  -lv, --low-vram don't allocate VRAM scratch buffer\n");
#endif
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -a ALIAS, --alias ALIAS\n");
    fprintf(stderr, "                        set an alias for the model, will be added as `model` field in completion response\n");
    fprintf(stderr, "  --lora FNAME          apply LoRA adapter (implies --no-mmap)\n");
    fprintf(stderr, "  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter\n");
    fprintf(stderr, "  --host                ip address to listen (default  (default: %s)\n", sparams.hostname.c_str());
    fprintf(stderr, "  --port PORT           port to listen (default  (default: %d)\n", sparams.port);
    fprintf(stderr, "  -to N, --timeout N    server read/write timeout in seconds (default: %d)\n", sparams.read_timeout);
    fprintf(stderr, "\n");
}

static void server_params_parse(int argc, char ** argv, server_params & sparams,
                                gpt_params & params) {
    gpt_params default_params;
    server_params default_sparams;
    std::string arg;
    bool invalid_param = false;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "--port") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.port = std::stoi(argv[i]);
        } else if (arg == "--host") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.hostname = argv[i];
        } else if (arg == "--timeout" || arg == "-to") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            sparams.read_timeout = std::stoi(argv[i]);
            sparams.write_timeout = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-a" || arg == "--alias") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model_alias = argv[i];
        } else if (arg == "-h" || arg == "--help") {
            server_print_usage(argv[0], default_params, default_sparams);
            exit(0);
        } else if (arg == "-c" || arg == "--ctx-size" || arg == "--ctx_size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_ctx = std::stoi(argv[i]);
        } else if (arg == "--memory-f32" || arg == "--memory_f32") {
            params.memory_f16 = false;
        } else if (arg == "--threads" || arg == "-t") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(argv[i]);
            params.n_batch = std::min(512, params.n_batch);
        } else if (arg == "--gpu-layers" || arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
#ifdef LLAMA_SUPPORTS_GPU_OFFLOAD
            params.n_gpu_layers = std::stoi(argv[i]);
#else
            LOG_WARNING("Not compiled with GPU offload support, --n-gpu-layers option will be ignored. "
                        "See main README.md for information on enabling GPU BLAS support", { { "n_gpu_layers", params.n_gpu_layers } });
#endif
        }
        else if (arg == "--tensor-split" || arg == "-ts") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
#ifdef GGML_USE_CUBLAS
            std::string arg_next = argv[i];

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            GGML_ASSERT(split_arg.size() <= LLAMA_MAX_DEVICES);

            for (size_t i_device = 0; i_device < LLAMA_MAX_DEVICES; ++i_device) {
                if (i_device < split_arg.size()) {
                    params.tensor_split[i_device] = std::stof(split_arg[i_device]);
                }
                else {
                    params.tensor_split[i_device] = 0.0f;
                }
            }
#else
            LOG_WARNING("llama.cpp was compiled without cuBLAS. It is not possible to set a tensor split.", {});
#endif // GGML_USE_CUBLAS
        }
        else if (arg == "--low-vram" || arg == "-lv")
        {
#ifdef GGML_USE_CUBLAS
            params.low_vram = true;
#else
            fprintf(stderr, "warning: llama.cpp was compiled without cuBLAS. It is not possible to set lower vram usage.\n");
#endif // GGML_USE_CUBLAS
        }
        else if (arg == "--main-gpu" || arg == "-mg") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
#ifdef GGML_USE_CUBLAS
            params.main_gpu = std::stoi(argv[i]);
#else
            LOG_WARNING("llama.cpp was compiled without cuBLAS. It is not possible to set a main GPU.", {});
#endif
        } else if (arg == "--lora") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.lora_adapter = argv[i];
            params.use_mmap = false;
        } else if (arg == "--lora-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.lora_base = argv[i];
        } else if (arg == "-v" || arg == "--verbose") {
#if SERVER_VERBOSE != 1
            LOG_WARNING("server.cpp is not built with verbose logging.", {});
#else
            server_verbose = true;
#endif
        } else if (arg == "--mlock") {
            params.use_mlock = true;
        } else if (arg == "--no-mmap") {
            params.use_mmap = false;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            server_print_usage(argv[0], default_params, default_sparams);
            exit(1);
        }
    }

    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        server_print_usage(argv[0], default_params, default_sparams);
        exit(1);
    }
}

static json format_generation_settings(llama_server_context & llama) {
    const auto eos_bias = llama.params.logit_bias.find(llama_token_eos());
    const bool ignore_eos = eos_bias != llama.params.logit_bias.end() &&
        eos_bias->second < 0.0f && std::isinf(eos_bias->second);

    return json {
        { "seed", llama.params.seed },
        { "temp", llama.params.temp },
        { "top_k", llama.params.top_k },
        { "top_p", llama.params.top_p },
        { "tfs_z", llama.params.tfs_z },
        { "typical_p", llama.params.typical_p },
        { "repeat_last_n", llama.params.repeat_last_n },
        { "repeat_penalty", llama.params.repeat_penalty },
        { "presence_penalty", llama.params.presence_penalty },
        { "frequency_penalty", llama.params.frequency_penalty },
        { "mirostat", llama.params.mirostat },
        { "mirostat_tau", llama.params.mirostat_tau },
        { "mirostat_eta", llama.params.mirostat_eta },
        { "penalize_nl", llama.params.penalize_nl },
        { "stop", llama.params.antiprompt },
        { "n_predict", llama.params.n_predict },
        { "n_keep", llama.params.n_keep },
        { "ignore_eos", ignore_eos },
        { "stream", llama.stream },
        { "logit_bias", llama.params.logit_bias },
    };
}

static json format_final_response(llama_server_context & llama, const std::string & content) {
    return json {
        { "content", content },
        { "stop", true },
        { "model", llama.params.model_alias },
        { "tokens_predicted", llama.num_tokens_predicted },
        { "generation_settings", format_generation_settings(llama) },
        { "prompt", llama.params.prompt },
        { "truncated", llama.truncated },
        { "stopped_eos", llama.stopped_eos },
        { "stopped_word", llama.stopped_word },
        { "stopped_limit", llama.stopped_limit },
        { "stopping_word", llama.stopping_word },
    };
}

static json format_partial_response(const std::string & content) {
    return json {
        { "content", content },
        { "stop", false },
    };
}

static json format_tokenizer_response(const std::vector<llama_token> & tokens) {
    return json {
        { "tokens", tokens }
    };
}

static void parse_options_completion(const json & body, llama_server_context & llama) {
    gpt_params default_params;

    llama.stream = body.value("stream", false);
    llama.params.n_predict = body.value("n_predict", default_params.n_predict);
    llama.params.top_k = body.value("top_k", default_params.top_k);
    llama.params.top_p = body.value("top_p", default_params.top_p);
    llama.params.tfs_z = body.value("tfs_z", default_params.tfs_z);
    llama.params.typical_p = body.value("typical_p", default_params.typical_p);
    llama.params.repeat_last_n = body.value("repeat_last_n", default_params.repeat_last_n);
    llama.params.temp = body.value("temperature", default_params.temp);
    llama.params.repeat_penalty = body.value("repeat_penalty", default_params.repeat_penalty);
    llama.params.presence_penalty = body.value("presence_penalty", default_params.presence_penalty);
    llama.params.frequency_penalty = body.value("frequency_penalty", default_params.frequency_penalty);
    llama.params.mirostat = body.value("mirostat", default_params.mirostat);
    llama.params.mirostat_tau = body.value("mirostat_tau", default_params.mirostat_tau);
    llama.params.mirostat_eta = body.value("mirostat_eta", default_params.mirostat_eta);
    llama.params.penalize_nl = body.value("penalize_nl", default_params.penalize_nl);
    llama.params.n_keep = body.value("n_keep", default_params.n_keep);
    llama.params.seed = body.value("seed", default_params.seed);
    llama.params.prompt = body.value("prompt", default_params.prompt);

    llama.params.logit_bias.clear();
    if (body.value("ignore_eos", false)) {
        llama.params.logit_bias[llama_token_eos()] = -INFINITY;
    }

    const auto & logit_bias = body.find("logit_bias");
    if (logit_bias != body.end() && logit_bias->is_array()) {
        const int n_vocab = llama_n_vocab(llama.ctx);
        for (const auto & el : *logit_bias) {
            if (el.is_array() && el.size() == 2 && el[0].is_number_integer()) {
                llama_token tok = el[0].get<llama_token>();
                if (tok >= 0 && tok < n_vocab) {
                    if (el[1].is_number()) {
                        llama.params.logit_bias[tok] = el[1].get<float>();
                    } else if (el[1].is_boolean() && !el[1].get<bool>()) {
                        llama.params.logit_bias[tok] = -INFINITY;
                    }
                }
            }
        }
    }

    llama.params.antiprompt.clear();
    const auto & stop = body.find("stop");
    if (stop != body.end() && stop->is_array()) {
        for (const auto & word : *stop) {
            if (!word.empty()) {
                llama.params.antiprompt.push_back(word);
            }
        }
    }

    LOG_VERBOSE("completion parameters parsed", format_generation_settings(llama));
}

static void log_server_request(const Request & req, const Response & res) {
    LOG_INFO("request", {
        { "remote_addr", req.remote_addr },
        { "remote_port", req.remote_port },
        { "status", res.status },
        { "path", req.path },
        { "request", req.body },
        { "response", res.body },
    });
}

int main(int argc, char ** argv) {
    // own arguments required by this example
    gpt_params params;
    server_params sparams;

    // struct that contains llama context and inference
    llama_server_context llama;

    server_params_parse(argc, argv, sparams, params);

    if (params.model_alias == "unknown") {
        params.model_alias = params.model;
    }

    llama_init_backend();

    LOG_INFO("build info", {
        { "build", BUILD_NUMBER },
        { "commit", BUILD_COMMIT }
    });
    LOG_INFO("system info", {
        { "n_threads", params.n_threads },
        { "total_threads", std::thread::hardware_concurrency() },
        { "system_info", llama_print_system_info() },
    });

    // load the model
    if (!llama.loadModel(params)) {
        return 1;
    }

    Server svr;

    svr.set_default_headers({
        { "Access-Control-Allow-Origin", "*" },
        { "Access-Control-Allow-Headers", "content-type" }
    });

    svr.Get("/", [](const Request &, Response & res) {
        res.set_content("<h1>llama.cpp server works</h1>", "text/html");
    });

    svr.Post("/completion", [&llama](const Request & req, Response & res) {
        llama.rewind();
        llama_reset_timings(llama.ctx);

        parse_options_completion(json::parse(req.body), llama);

        llama.loadPrompt();
        llama.beginCompletion();

        if (!llama.stream) {
            size_t stop_pos = std::string::npos;

            while (llama.has_next_token) {
                const std::string token_text = llama.doCompletion();

                stop_pos = llama.findStoppingStrings(llama.generated_text,
                    token_text.size(), STOP_FULL);
            }

            if (stop_pos == std::string::npos) {
                stop_pos = llama.findStoppingStrings(llama.generated_text, 0, STOP_PARTIAL);
            }
            if (stop_pos != std::string::npos) {
                llama.generated_text.erase(llama.generated_text.begin() + stop_pos,
                    llama.generated_text.end());
            }

            const json data = format_final_response(llama, llama.generated_text);

            llama_print_timings(llama.ctx);

            res.set_content(data.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        } else {
            const auto chunked_content_provider = [&](size_t, DataSink & sink) {
                size_t sent_count = 0;

                while (llama.has_next_token) {
                    const std::string token_text = llama.doCompletion();
                    if (llama.multibyte_pending > 0) {
                        continue;
                    }

                    size_t pos = std::min(sent_count, llama.generated_text.size());

                    const std::string str_test = llama.generated_text.substr(pos);
                    size_t stop_pos =
                        llama.findStoppingStrings(str_test, token_text.size(), STOP_FULL);
                    if (stop_pos != std::string::npos) {
                        llama.generated_text.erase(
                            llama.generated_text.begin() + pos + stop_pos,
                            llama.generated_text.end());
                        pos = std::min(sent_count, llama.generated_text.size());
                    } else {
                        stop_pos = llama.findStoppingStrings(str_test, token_text.size(),
                            STOP_PARTIAL);
                    }

                    const std::string to_send = llama.generated_text.substr(pos, stop_pos);
                    sent_count += to_send.size();

                    const json data = llama.has_next_token
                                          ? format_partial_response(to_send)
                                          // Generation is done, send extra information.
                                          : format_final_response(llama, to_send);

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
                }

                llama_print_timings(llama.ctx);
                sink.done();
                return true;
            };
            res.set_chunked_content_provider("text/event-stream", chunked_content_provider);
        }
    });

    svr.Options(R"(/.*)", [](const Request &, Response & res) {
        return res.set_content("", "application/json");
    });

    svr.Post("/tokenize", [&llama](const Request & req, Response & res) {
        const json body = json::parse(req.body);
        const std::string content = body["content"].get<std::string>();
        const std::vector<llama_token> tokens = llama_tokenize(llama.ctx, content, false);
        const json data = format_tokenizer_response(tokens);
        return res.set_content(data.dump(), "application/json");
    });

    svr.set_logger(log_server_request);

    svr.set_exception_handler([](const Request &, Response & res, std::exception_ptr ep) {
        const auto * fmt = "500 Internal Server Error\n%s";
        char buf[BUFSIZ];
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception & e) {
            snprintf(buf, sizeof(buf), fmt, e.what());
        } catch (...) {
            snprintf(buf, sizeof(buf), fmt, "Unknown Exception");
        }
        res.set_content(buf, "text/plain");
        res.status = 500;
    });

    // set timeouts and change hostname and port
    svr.set_read_timeout(sparams.read_timeout);
    svr.set_write_timeout(sparams.write_timeout);

    if (!svr.bind_to_port(sparams.hostname, sparams.port)) {
        LOG_ERROR("couldn't bind to server socket", {
            { "hostname", sparams.hostname },
            { "port", sparams.port },
        });
        return 1;
    }

    LOG_INFO("HTTP server listening", {
        { "hostname", sparams.hostname },
        { "port", sparams.port },
    });

    if (!svr.listen_after_bind()) {
        return 1;
    }

    return 0;
}
