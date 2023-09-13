// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"

#include "console.h"
#include "llama.h"
#include "build-info.h"
#include "grammar-parser.h"

#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static std::atomic<bool> interrupted {false};

void write_logfile(
    const llama_context * ctx, const gpt_params & params, const llama_model * model,
    const std::vector<llama_token> input_tokens, const std::string output, const std::vector<llama_token> output_tokens) {

    if (params.logdir.empty()) {
        return;
    }

    const std::string timestamp = get_sortable_timestamp();

    const bool success = create_directory_with_parents(params.logdir);
    if (!success) {
        fprintf(stderr, "%s: warning: failed to create logdir %s, cannot write logfile\n",
                __func__, params.logdir.c_str());
        return;
    }

    const std::string logfile_path = params.logdir + timestamp + ".yml";
    FILE * logfile = fopen(logfile_path.c_str(), "w");

    if (logfile == NULL) {
        fprintf(stderr, "%s: failed to open logfile %s\n", __func__, logfile_path.c_str());
        return;
    }

    fprintf(logfile, "binary: simple-inference\n");
    char model_desc[128];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    dump_non_result_info_yaml(logfile, params, ctx, timestamp, input_tokens, model_desc);

    fprintf(logfile, "\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "# Generation Results #\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "\n");

    dump_string_yaml_multiline(logfile, "output", output.c_str());
    dump_vector_int_yaml(logfile, "output_tokens", output_tokens);

    llama_dump_timing_info_yaml(logfile, ctx);
    fclose(logfile);
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        interrupted.store(true);
    }
}
#endif


bool check_unsupported(const gpt_params * params) {
    std::string nope;
    if (params->perplexity)
        nope = "perplexity";
    else if (params->embedding)
        nope = "embedding";
    else if (params->cfg_scale != 1.0f)
        nope = "cfg_scale";
    else if (!params->cfg_negative_prompt.empty())
        nope = "cfg_negative_prompt";
    else if (params->mem_test)
        nope = "mem test";
    else if (params->export_cgraph)
        nope = "export cgraph";
    else if (!params->path_prompt_cache.empty())
        nope = "prompt cache";
    else if (params->escape)
        nope = "prompt escaping";
    else if (params->interactive || params->interactive_first || params->instruct)
        nope = "interactive mode";
    else if (!params->input_prefix.empty() || !params->input_suffix.empty() || params->input_prefix_bos)
        nope = "input prefix or suffix";
    else if (params->hellaswag)
        nope = "hellaswag";
    else if (params->n_keep != 0)
        nope = "keep";
    else if (!params->antiprompt.empty())
        nope = "reverse prompt";
    if (!nope.empty()) {
        LOG_TEE("%s: error: We don't support %s here.\n", __func__, nope.c_str());
        return false;
    }
    return true;
}


bool initialize(llama_context **ctx_p, llama_model **model_p, gpt_params & params, std::vector<llama_token> & embd_inp, llama_grammar ** grammar_p) {
    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });


    if (params.rope_freq_base != 10000.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g (default 10000.0)\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g (default 1.0)\n", __func__, params.rope_freq_scale);
    }

    if (params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    LOG_TEE("%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    LOG("%s: llama backend init\n", __func__);
    llama_backend_init(params.numa);

    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(*model_p, *ctx_p) = llama_init_from_gpt_params(params);
    llama_model * model = *model_p;
    llama_context * ctx = *ctx_p;

    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return false;
    }

    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    const bool add_bos = llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM;
    LOG("add_bos: %d\n", add_bos);

    if (!params.prompt.empty()) {
        LOG("tokenize the prompt\n");
        embd_inp = ::llama_tokenize(ctx, params.prompt, add_bos);
    }

    LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp));

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(ctx));
        LOG("input was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp));
    }

    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    if (params.verbose_prompt) {
        LOG_TEE("\n");
        LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        LOG_TEE("\n");
    }

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    LOG_TEE("sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
            params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    LOG_TEE("\n\n");

    grammar_parser::parse_state parsed_grammar;
    if (!params.grammar.empty()) {
        parsed_grammar = grammar_parser::parse(params.grammar.c_str());
        // will be empty (default) if there are parse errors
        if (parsed_grammar.rules.empty()) {
            return false;
        }
        LOG_TEE("%s: grammar:\n", __func__);
        grammar_parser::print_grammar(stderr, parsed_grammar);
        LOG_TEE("\n");

        {
            auto it = params.logit_bias.find(llama_token_eos(ctx));
            if (it != params.logit_bias.end() && it->second == -INFINITY) {
                LOG_TEE("%s: warning: EOS token is disabled, which will cause most grammars to fail\n", __func__);
            }
        }

        std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
        *grammar_p = llama_grammar_init(
            grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    }
    return true;
}

bool feed_prompt(llama_context *ctx, const gpt_params * params, llama_token * tokens, int tokens_len, int n_past) {
    console::set_display(console::prompt);
    while (tokens_len > 0 && !interrupted) {
        const int this_chunk_size = std::min(tokens_len, params->n_batch);

        if (llama_eval(ctx, tokens, this_chunk_size, n_past, params->n_threads)) {
            console::set_display(console::reset);
            LOG_TEE("%s : failed to eval\n", __func__);
            return false;
        }

        // display text
        for (int i = 0; i < this_chunk_size; i++) {
            const std::string token_str = llama_token_to_piece(ctx, tokens[i]);
            fputs(token_str.c_str(), stdout);
        }
        fflush(stdout);

        tokens += this_chunk_size;
        tokens_len -= this_chunk_size;
        n_past += this_chunk_size;
    }
    console::set_display(console::reset);
    return true;
}


int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (!check_unsupported(&params)) {
        return 1;
    }

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("simple-inference", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc,argv);
#endif // LOG_DISABLE_LOGS

    llama_context *     ctx = NULL;
    llama_model *     model = NULL;
    llama_grammar * grammar = NULL;
    std::vector<llama_token> prompt_tokens;

    if (!initialize(&ctx, &model, params, prompt_tokens, &grammar)) {
        return 1;
    }

    const int n_ctx   = llama_n_ctx(ctx);
    int n_remain      = params.n_predict;
    std::vector<int>   input_tokens;

    {
        LOG("warming up the model with an empty run\n");

        const std::vector<llama_token> tmp = { llama_token_bos(ctx), };
        llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        llama_reset_timings(ctx);
    }

    if (!feed_prompt(ctx, &params, prompt_tokens.data(), prompt_tokens.size(), 0)) {
        return 1;
    }

    if (n_remain < 0 || n_remain + int(prompt_tokens.size()) > n_ctx) {
        n_remain = n_ctx - prompt_tokens.size();
    }

    std::vector<llama_token> last_tokens = prompt_tokens;
    last_tokens.reserve(params.n_ctx);

    std::vector<llama_token_data> candidates;
    candidates.reserve(llama_n_vocab(ctx));

    while (n_remain > 0 && !interrupted) {
        const llama_token id = llama_sample_token(ctx, NULL, grammar, params, last_tokens, candidates);

        last_tokens.push_back(id);
        --n_remain;

        LOG("n_remain: %d\n", n_remain);

        // end of text token
        if (id == llama_token_eos(ctx)) {
            LOG_TEE(" [end of text]\n");
            break;
        }

        const std::string token_str = llama_token_to_piece(ctx, id);
        fputs(token_str.c_str(), stdout);
        fflush(stdout);

        // predict
        if (n_remain > 0 && llama_eval(ctx, &id, 1, last_tokens.size(), params.n_threads)) {
            LOG_TEE("%s : failed to eval\n", __func__);
            return 1;
        }
    }

    std::vector<int>   output_tokens;
    std::ostringstream output_ss;
    const size_t prompt_size = prompt_tokens.size();
    output_tokens.reserve(last_tokens.size() - prompt_size);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        const std::string token_str = llama_token_to_piece(ctx, last_tokens[i]);
        if (i >= prompt_size) {
            output_ss << token_str;
            output_tokens.push_back(last_tokens[i]);
        }

    }

    console::cleanup();
    printf("\n");

    llama_print_timings(ctx);
    write_logfile(ctx, params, model, prompt_tokens, output_ss.str(), output_tokens);

    llama_free(ctx);
    llama_free_model(model);

    if (grammar != NULL) {
        llama_grammar_free(grammar);
    }
    llama_backend_free();

#ifndef LOG_DISABLE_LOGS
    LOG_TEE("Log end\n")
#endif // LOG_DISABLE_LOGS

    return interrupted ? 130 : 0;
}
