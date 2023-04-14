// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

static console_state con_st;

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
    fflush(stdout);
    fflush(stderr);
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    con_st.use_color = params.use_color;

#if defined (_WIN32)
    win32_console_init(params.use_color);
#endif

    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    bool instruct_mode = !params.instruct_prefix.empty() || !params.instruct_suffix.empty();

//    params.prompt = R"(// this function checks if the number n is prime
//bool is_prime(int n) {)";

    llama_context * ctx;

    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.use_mmap   = params.use_mmap;
        lparams.use_mlock  = params.use_mlock;

        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        {
            const std::vector<llama_token> tmp(params.n_batch, 0);
            llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        }

        {
            const std::vector<llama_token> tmp = { 0, };
            llama_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
        }

        llama_print_timings(ctx);
        llama_free(ctx);

        return 0;
    }

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    }

    // prefix & suffix for instruct mode
    const auto inp_pfx = ::llama_tokenize(ctx, params.instruct_prefix, params.instruct_prefix_bos);
    std::string instruct_suffix = params.instruct_suffix;
    if (params.rm_trailing_space_workaround) {
        if (instruct_suffix.back() == ' ') { instruct_suffix.pop_back(); }
    }
    const auto inp_sfx = ::llama_tokenize(ctx, instruct_suffix, params.instruct_suffix_bos);

    // enable interactive mode if reverse prompt or interactive start is specified
    if (params.antiprompt.size() != 0 || params.stopprompt.size() != 0 || params.interactive_start) {
        params.interactive = true;
    }

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }
        if (params.n_keep > 0) {
        fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]));
            }
            fprintf(stderr, "'\n");
        }
        fprintf(stderr, "\n");
    }

    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        signal(SIGINT, sigint_handler);
#endif

        fprintf(stderr, "%s: interactive mode on.\n", __func__);

        if (params.antiprompt.size()) {
            for (auto antiprompt : params.antiprompt) {
                fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }
        if (params.stopprompt.size()) {
            for (auto stopprompt : params.stopprompt) {
                fprintf(stderr, "Stop prompt: '%s'\n", stopprompt.c_str());
            }
        }

        if (!params.input_prefix.empty()) {
            fprintf(stderr, "Input prefix: '%s'\n", params.input_prefix.c_str());
        }
        if (!params.instruct_prefix.empty()) {
            fprintf(stderr, "Instruct prefix %s: '%s'\n", params.instruct_prefix_bos ? "(with bos token)" : "", params.instruct_prefix.c_str());
        }
        if (!params.instruct_suffix.empty()) {
            fprintf(stderr, "Instruct suffix %s: '%s'\n", params.instruct_suffix_bos ? "(with bos token)" : "", params.instruct_suffix.c_str());
        }
    }
    fprintf(stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n",
        params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    fprintf(stderr, "\n\n");

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
        );
        if (params.multiline_mode) {
            fprintf(stderr, " - Press Return to return control to LLaMa.\n"
#if defined (_WIN32)
                            " - [MULTILINE MODE] Press Ctrl+Z then Return (EOF) to toggle.\n\n");
#else
                            " - [MULTILINE MODE] Press Ctrl+D (EOF) to toggle.\n\n");
#endif
        }
        else {
            fprintf(stderr, " - Press Return to return control to LLaMa.\n"
                            " - If you want to submit another line, end your input in '\\'.\n\n");
        }
        is_interacting = params.interactive_start;
    }

    struct Antiprompt {
        bool any = false;
        bool trailing_space = false;
        size_t len;
        bool is_stop_prompt = false;
    } antiprompt;

    bool input_noecho  = false;

    int n_past     = 0;
    int n_remain   = params.n_predict;
    int n_consumed = 0;

    // the first thing we will do is to output the prompt, so set color accordingly
    set_console_color(con_st, CONSOLE_COLOR_PROMPT);

    std::vector<llama_token> embd;

    while (n_remain != 0 || params.interactive) {
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - params.n_keep;

                n_past = params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                //printf("\n---\n");
                //printf("resetting: '");
                //for (int i = 0; i < (int) embd.size(); i++) {
                //    printf("%s", llama_token_to_str(ctx, embd[i]));
                //}
                //printf("'\n");
                //printf("\n---\n");
            }

            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // out of user input, sample next token
            const int32_t top_k          = params.top_k;
            const float   top_p          = params.top_p;
            const float   temp           = params.temp;
            const float   repeat_penalty = params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);

                if (params.ignore_eos) {
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(ctx,
                        last_n_tokens.data() + n_ctx - params.repeat_last_n,
                        params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && params.interactive && !instruct_mode) {
                id = llama_token_newline.front();
                if (params.antiprompt.size() != 0) {
                    // tokenize and inject first reverse prompt
                    const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                }
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_noecho = false;

            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (!input_noecho) {
            for (auto id : embd) {
                printf("%s", llama_token_to_str(ctx, id));
            }
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
        if (!input_noecho && (int)embd_inp.size() == n_consumed) {
            set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
        }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && (int) embd_inp.size() <= n_consumed) {

            // check for reverse prompt or stop prompt
            if (params.antiprompt.size() || params.stopprompt.size()) {
                std::string last_output;
                for (auto id : last_n_tokens) {
                    last_output += llama_token_to_str(ctx, id);
                }

                antiprompt.any = false;
                antiprompt.is_stop_prompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                for (std::string & prompt : params.antiprompt) {
                    if (params.rm_trailing_space_workaround) {
                        antiprompt.trailing_space = prompt.back() == ' ';
                        antiprompt.len = prompt.length() - (antiprompt.trailing_space ? 1 : 0);
                    }
                    if (last_output.find(prompt.c_str(), last_output.length() - antiprompt.len, antiprompt.len) != std::string::npos) {
                        is_interacting = true;
                        antiprompt.any = true;
                        set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);
                        fflush(stdout);
                        break;
                    }
                }
                if (!antiprompt.any) {
                    for (std::string & prompt : params.stopprompt) {
                        if (params.rm_trailing_space_workaround) {
                            antiprompt.trailing_space = prompt.back() == ' ';
                            antiprompt.len = prompt.length() - (antiprompt.trailing_space ? 1 : 0);
                        }
                        if (last_output.find(prompt.c_str(), last_output.length() - antiprompt.len, antiprompt.len) != std::string::npos) {
                            is_interacting = true;
                            antiprompt.any = true;
                            antiprompt.is_stop_prompt = true;
                            set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);
                            fflush(stdout);
                            break;
                        }
                    }
                }
            }

            if (n_past > 0 && is_interacting)
            {
                std::string buffer;
                if (!params.clean_interface && !params.instruct_prefix.empty() && !antiprompt.any) {
                    // avoid printing again user's new line (TODO: try to revert enter press and print newline)
                    int i = params.instruct_prefix.front() == '\n' ? 1 : 0;
                    for (; i < inp_pfx.size(); i++) {
                        printf("%s", llama_token_to_str(ctx, inp_pfx[i]));
                    }
                    fflush(stdout);
                }
                if (params.rm_trailing_space_workaround) {
                    // add only if not stopprompt (as stopprompt could be used to pause
                        //     assistant and then continue without input - adding back trailing
                        //     space may mess it up.)
                    if (!antiprompt.is_stop_prompt && antiprompt.any && antiprompt.trailing_space) {
                        // add back removed trailing space to buffer(workaround)
                        buffer += ' ';
                        if (!params.clean_interface) {
                            printf("%s", buffer.c_str());
                        }
                        fflush(stdout);
                    }
                }

                // potentially set color to indicate we are taking user input
                set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);

#if defined (_WIN32)
                // Windows: must reactivate sigint handler after each signal
                signal(SIGINT, sigint_handler);
#endif

                if (params.clean_interface) {
                    printf("\n> ");
                }

                if (!params.input_prefix.empty()) {
                    buffer += params.input_prefix;
                    printf("%s", buffer.c_str());
                }

                if (!get_input_text(buffer, params.multiline_mode)) {
                    // input stream is bad
                    return 1;
                }
                if (!antiprompt.is_stop_prompt) {
                    buffer += "\n";
                }

                // done taking input, reset color
                set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

                if (!params.clean_interface && !params.instruct_suffix.empty() && !antiprompt.is_stop_prompt) {
                    // avoid printing again user's new line (TODO: try to revert enter press and print newline)
                    int i = params.instruct_suffix.front() == '\n' ? 1 : 0;
                    for (; i < inp_sfx.size(); i++) {
                        printf("%s", llama_token_to_str(ctx, inp_sfx[i]));
                    }
                    // if (remove trailing space workaround) {
                    //     We won't add back removed trailing space here, because assistant continues here,
                    //         and it may mess up it's output (remove trailing space workaround).
                    // }
                    fflush(stdout);
                }

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {

                    // insert input prefix
                    if (!params.instruct_prefix.empty() && !antiprompt.any) {
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }

                    auto line_inp = ::llama_tokenize(ctx, buffer, false);
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                    // insert response suffix
                    if (!params.instruct_suffix.empty() && !antiprompt.is_stop_prompt) {
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }

                    n_remain -= line_inp.size();
                }

                input_noecho = true; // do not echo this again
            }

            if (n_past > 0) {
                is_interacting = false;
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos()) {
            if (instruct_mode) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

#if defined (_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    llama_print_timings(ctx);
    llama_free(ctx);

    set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

    return 0;
}
