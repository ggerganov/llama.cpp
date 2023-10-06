#include "frontend.h"
#include "common.h"
#include "llama.h"

#include "httplib.h"
#include "json.hpp"

#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <chrono>

using namespace httplib;
using namespace std;
using namespace nlohmann;

struct server_params
{
    std::string hostname = "127.0.0.1";
    std::string public_path = "examples/server/public";
    int32_t port = 8080;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
};

// utils functions taken of examples/server

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

enum stop_type
{
    STOP_FULL,
    STOP_PARTIAL,
};

enum slot_state
{
    IDLE,
    PROCESSING
};

enum slot_command {
    NONE,
    LOAD_PROMPT,
    RELEASE
};


struct llama_client_slot
{
    int id;
    int32_t n_prompt  = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;
    string prompt = "";
    string sampled_token_str;
    string generated_text = "";
    llama_token sampled;
    std::vector<llama_token> tokens_prev;
    slot_state state = IDLE;
    slot_command command = NONE;
    bool newToken = false;
    float temperature = 0.1f;

    void start(string prompt_, float temp_) {
        prompt = prompt_;
        command = LOAD_PROMPT;
        temperature = temp_;
        newToken = false;
    }

    bool hasNewToken() {
        if(newToken) {
            newToken = false;
            return true;
        }
        return false;
    }

    bool available() {
        return state == IDLE && command == NONE;
    }

    void nofity() {
        newToken = !newToken;
    }

    void release() {
        if(state == PROCESSING) {
            command = RELEASE;
        }
    }
};

struct server_parallel_context {
    // example props
    vector<llama_client_slot> slots;
    std::string system_prompt = "";
    bool update_system_prompt = true;

    // broadcast to all clients to keep the same prompt format
    std::string user_name = ""; // this should be the anti prompt
    std::string assistant_name = ""; // this is for generate the prompt

    // llama native props
    gpt_params params;
    llama_model *model = NULL;
    llama_context *ctx = NULL;
    int n_ctx;
    int n_vocab;
    std::vector<llama_token_data> candidates;
    std::vector<llama_token> tokens_system;
    int32_t n_tokens_system = 0;
    llama_batch batch;

    bool loadModel(gpt_params params_) {
        params = params_;
        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        if (model == nullptr)
        {
            LOG_TEE("unable to load model: %s", params.model.c_str());
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
            slot.prompt = "default";
            slot.state = IDLE;
            slot.tokens_prev.resize(std::max(256, params.n_predict));
            std::fill(slot.tokens_prev.begin(), slot.tokens_prev.end(), 0);
            LOG_TEE(" - slot %i\n", slot.id);
            slots.push_back(slot);
        }
        batch = llama_batch_init(params.n_ctx, 0);

        // always assign a default system prompt
        system_prompt = system_prompt_default;
        user_name = "User:";
        assistant_name = "Assistant:";
        params.antiprompt.push_back(user_name);
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
        // wait until system prompt load
        update_system_prompt = true;
        while(update_system_prompt) {
            this_thread::sleep_for(chrono::milliseconds(5));
        }
        // system prompt loaded, continue
    }

    llama_client_slot* requestCompletion(json data) {
        if(data.contains("system_prompt") &&
            data.contains("anti_prompt") &&
            data.contains("assistant_name")) {
            system_prompt = data.value("system_prompt", "");
            user_name = data.value("anti_prompt", "");
            assistant_name = data.value("assistant_name", "");
            params.antiprompt.clear();
            params.antiprompt.push_back(user_name);
            notifySystemPromptChanged();
        }
        int slot_id = data.value("slot_id", -1);
        float temperature = data.value("temperature", 0.1f);
        string prompt = data.value("prompt", "");
        for (llama_client_slot & slot : slots)
        {
            if ((slot_id == -1 && slot.available()) || slot.id == slot_id)
            {
                slot.start(prompt, temperature);
                LOG_TEE("slot %i is processing\n", slot.id);
                return &slot; // return a pointer to slot (thread safe?)
            }
        }
        return nullptr;
    }

    size_t findStoppingStrings(const std::string &text, const size_t last_token_size,
                               const stop_type type)
    {
        size_t stop_pos = std::string::npos;
        for (const std::string &word : params.antiprompt)
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
            }
        }
        return stop_pos;
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

    bool updateSlots() {
        // update the system prompt wait until all slots are idle state
        if(update_system_prompt) {
            updateSystemPrompt();
        }

        batch.n_tokens = 0;

        // decode any currently ongoing sequences
        for (auto & slot : slots) {
            if (slot.state == PROCESSING && slot.command == RELEASE)
            {
                LOG_TEE("slot %i released\n", slot.id);
                llama_kv_cache_seq_rm(ctx, slot.id, n_tokens_system, n_ctx);
                slot.state = IDLE;
                slot.command = NONE;
                continue;
            }

            // no decode wait until the token had been send to client
            // improves performance and avoid decoherence?

            if (slot.state == IDLE || slot.newToken) {
                continue;
            }

            batch.token [batch.n_tokens] = slot.sampled;
            batch.pos   [batch.n_tokens] = n_tokens_system + slot.n_prompt + slot.n_decoded;
            batch.seq_id[batch.n_tokens] = slot.id;
            batch.logits[batch.n_tokens] = true;

            slot.n_decoded += 1;
            slot.i_batch = batch.n_tokens;

            batch.n_tokens += 1;
        }

        // assign workload to the slots
        if (params.cont_batching || batch.n_tokens == 0) {
            for (llama_client_slot & slot : slots) {
                // need process the prompt
                if (slot.state == IDLE && slot.command == LOAD_PROMPT) {
                    slot.state = PROCESSING;
                    slot.command = NONE;
                    //LOG_TEE("slot %i process prompt:\n%s%s'------------------------------\n", slot.id, system_prompt.c_str(), slot.prompt.c_str());
                    std::fill(slot.tokens_prev.begin(), slot.tokens_prev.end(), 0);

                    // do not prepend BOS because we have a system prompt!
                    std::vector<llama_token> tokens_prompt;
                    tokens_prompt = ::llama_tokenize(ctx, slot.prompt, false);

                    for (size_t i = 0; i < tokens_prompt.size(); ++i) {
                        batch.token [batch.n_tokens] = tokens_prompt[i];
                        batch.pos   [batch.n_tokens] = i + n_tokens_system;
                        batch.seq_id[batch.n_tokens] = slot.id;
                        batch.logits[batch.n_tokens] = false;
                        batch.n_tokens += 1;
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0) {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    slot.n_prompt  = tokens_prompt.size();
                    slot.n_decoded = 0;
                    slot.i_batch   = batch.n_tokens - 1;

                    // insert new requests one-by-one
                    //if (cont_batching) {
                    //    break;
                    //}
                }
            }
        }

        if (batch.n_tokens == 0) {
            return true;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            // experiment: process in powers of 2
            //if (i + n_batch > (int32_t) batch.n_tokens && n_batch > 32) {
            //    n_batch /= 2;
            //    i -= n_batch;
            //    continue;
            //}

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

                params.temp = slot.temperature;
                const llama_token id = llama_sample_token(ctx, NULL, NULL, params, slot.tokens_prev, candidates, slot.i_batch - i);

                // remember which tokens were sampled - used for repetition penalties during sampling
                slot.tokens_prev.erase(slot.tokens_prev.begin());
                slot.tokens_prev.push_back(id);

                const std::string token_str = llama_token_to_piece(ctx, id);
                slot.generated_text += token_str;
                slot.sampled = id;

                size_t stop_pos =
                        findStoppingStrings(slot.generated_text, token_str.size(), STOP_FULL);

                slot.sampled_token_str = token_str;
                // notify new token
                slot.nofity();

                if (slot.n_decoded > 2 &&
                        (id == llama_token_eos(ctx) ||
                         (params.n_predict > 0 &&
                        slot.n_decoded + slot.n_prompt >=
                        params.n_predict) ||
                         stop_pos != std::string::npos)) {
                    //LOG_TEE("slot %i generated text:\n%s'------------------------------\n", slot.id, slot.generated_text.c_str());
                    slot.generated_text.clear();
                    slot.release();
                }
                slot.i_batch = -1;
            }
        }
        return true;
    }
};

static void server_print_usage(const char *argv0, const gpt_params &params,
                               const server_params &sparams)
{
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help            show this help message and exit\n");
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

    // new arguments
    printf("  -np N, --parallel N   number of parallel sequences to decode (default: %d)\n", params.n_parallel);
    printf("  -cb, --cont-batching  enable continuous batching (a.k.a dynamic batching) (default: disabled)\n");
    printf("  -f FNAME, --file FNAME\n");
    printf("                        load a system prompt from a file.\n");
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
            LOG_TEE("Not compiled with GPU offload support, --n-gpu-layers option will be ignored. "
                        "See main README.md for information on enabling GPU BLAS support\n");
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
            LOG_TEE("llama.cpp was compiled without cuBLAS. It is not possible to set a tensor split.\n");
#endif // GGML_USE_CUBLAS
        }
        else if (arg == "--no-mul-mat-q" || arg == "-nommq")
        {
#ifdef GGML_USE_CUBLAS
            params.mul_mat_q = false;
#else
            LOG_TEE("warning: llama.cpp was compiled without cuBLAS. Disabling mul_mat_q kernels has no effect.\n");
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
            LOG_TEE("llama.cpp was compiled without cuBLAS. It is not possible to set a main GPU.");
#endif
        }
        else if (arg == "--lora")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.lora_adapter.push_back({argv[i], 1.0f});
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
            params.lora_adapter.push_back(make_tuple(lora_adapter, std::stof(argv[i])));
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
        } else if (arg == "-r" || arg == "--reverse-prompt")
        {
            if (++i >= argc)
            {
                invalid_param = true;
                break;
            }
            params.antiprompt.push_back(argv[i]);
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

int main(int argc, char **argv)
{
    gpt_params params;

    server_params sparams;

    server_params_parse(argc, argv, sparams, params);

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("server-parallel", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    llama_backend_init(params.numa);

    // load the target model
    params.logits_all = true;
    server_parallel_context llama;

    if(!llama.loadModel(params)) {
        return 1;
    }

    llama.initialize();

    Server svr;

    svr.set_default_headers({{"Server", "llama.cpp"},
                             {"Access-Control-Allow-Origin", "*"},
                             {"Access-Control-Allow-Headers", "content-type"}});

    svr.Get("/", [&](const Request & /*req*/, Response &res)
            { res.set_content(index_html_, "text/html"); });

    svr.Get("/index.js", [&](const Request & /*req*/, Response &res)
            { res.set_content(index_js_, "text/html"); });

    svr.Get("/props", [&llama](const Request & /*req*/, Response &res)
            {
                json data = {
                    { "user_name", llama.user_name.c_str() },
                    { "assistant_name", llama.assistant_name.c_str() }
                };
                res.set_content(data.dump(), "application/json"); });

    svr.Post("/completion", [&llama](const Request &req, Response &res)
             {
        llama_client_slot* slot = llama.requestCompletion(json::parse(req.body));
        // Verify if the slot exist
        if (slot) {
            res.set_chunked_content_provider("text/event-stream",
                [slot](size_t /*offset*/, DataSink &sink) {
                    if(slot->available()) { // slot has been released
                        sink.done();
                        return false;
                    }
                    if(slot->hasNewToken()) { // new token notification
                        stringstream ss;
                        json res_d = {{ "content", slot->sampled_token_str }};
                        ss << "data: " << res_d.dump() << "\n\n";
                        string result = ss.str();
                        if(!sink.write(result.c_str(), result.size())) {
                            slot->release();
                            return false;
                        }
                    }
                    return true;
                });
        } else {
            LOG_TEE("slot unavailable\n");
            res.status = 404;
            res.set_content("slot_error", "text/plain");
        } });

    thread t([&llama]()
             {
            bool running = true;
            while (running)
            {
                running = llama.updateSlots();
            } });

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

    if (!svr.listen_after_bind())
    {
        return 1;
    }
}
