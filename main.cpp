#include "ggml.h"
#include "llama.h"
#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"


// determine number of model parts based on the dimension
static const std::map<int, int> LLAMA_N_PARTS = {
    { 4096, 1 },
    { 5120, 2 },
    { 6656, 4 },
    { 8192, 8 },
};


static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    printf(ANSI_COLOR_RESET);
    printf("\n"); // this also force flush stdout.
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            _exit(130);
        }
    }
}
#endif


void process_interactive_input(llama_context& ctx, const gpt_params& params);

int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/7B/ggml-model-q4_0.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

//    params.prompt = R"(// this function checks if the number n is prime
//bool is_prime(int n) {)";

    // load the model
    llama_context* ctx_ptr = nullptr;
    {
        ctx_ptr = llama_init_from_params(params);
        if (!ctx_ptr) {  
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }
    
    llama_context & ctx = *ctx_ptr;
    const gpt_vocab & vocab = llama_context_get_vocab(ctx);

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // prefix & suffix for instruct mode
    const std::vector<gpt_vocab::id> inp_pfx = ::llama_tokenize(vocab, "\n\n### Instruction:\n\n", true);
    const std::vector<gpt_vocab::id> inp_sfx = ::llama_tokenize(vocab, "\n\n### Response:\n\n", false);

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive = true;
        params.antiprompt.push_back("### Instruction:\n\n");
    }

    // tokenize the reverse prompt
    std::vector<std::vector<gpt_vocab::id>> antipromptv_inp;
    
    for (auto antiprompt : params.antiprompt) {
        antipromptv_inp.push_back(::llama_tokenize(vocab, antiprompt, false));
    }

    // enable interactive mode if reverse prompt is specified
    if (!antipromptv_inp.size()) {
        params.interactive = true;
    }

    // Setup interactive mode
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

        if(antipromptv_inp.size()) {
            for (size_t apindex = 0; apindex < antipromptv_inp.size(); ++apindex) {
                auto antiprompt_inp = antipromptv_inp.at(apindex);
                fprintf(stderr, "%s: reverse prompt: '%s'\n", __func__, params.antiprompt.at(apindex).c_str());
                fprintf(stderr, "%s: number of tokens in reverse prompt = %zu\n", __func__, antiprompt_inp.size());
                for (int i = 0; i < (int) antiprompt_inp.size(); i++) {
                    fprintf(stderr, "%6d -> '%s'\n", antiprompt_inp[i], vocab.id_to_token.at(antiprompt_inp[i]).c_str());
                }
                fprintf(stderr, "\n");
            }
        }
    }
    fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "\n\n");

    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
               " - Press Return to return control to LLaMa.\n"
               " - If you want to submit another line, end your input in '\\'.\n\n");
        is_interacting = true;
    }

    // set the color for the prompt which will be output initially
    if (params.use_color) {
        printf(ANSI_COLOR_YELLOW);
    }

    // Prepare the context with input
    // Send "beginning of string"
    llama_add_bos(ctx);

    // load the input
    llama_update_input(ctx, params.prompt);

    llama_print_startup_stats(ctx);

    if(!llama_prepare_context(ctx))
    {
        fprintf(stderr, "%s: failed to prepare context\n", __func__);
        return 1;
    }

    bool input_noecho = false;
    bool is_end_of_text = false;
    while (llama_context_is_finished(ctx) == false) {
        std::string model_output{};

        if (llama_has_unconsumed_input(ctx)) {
            llama_ingest_all_pending_input(ctx, !input_noecho);
        }else{
            // Run inference if we don't have any pending input
            llama_infer(ctx, model_output, is_end_of_text);
            // print the single token output
            printf("%s", model_output.c_str());
            input_noecho = false;
        }
        // reset color to default (all input will be ingested already at this point)
        if (!input_noecho && params.use_color) {
            printf(ANSI_COLOR_RESET);
        }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && !llama_has_unconsumed_input(ctx)) {
            // check for reverse prompt
            for (auto antiprompt_inp : antipromptv_inp) {
                if (antiprompt_inp.size() && llama_is_anti_prompt_present(ctx, antiprompt_inp)) {
                    // reverse prompt found
                    is_interacting = true;
                    break;
                }
            }
            if (is_interacting) {
                if (params.instruct) {
                    llama_update_input(ctx, inp_pfx);
                    printf("\n> ");
                }

                // currently being interactive
                process_interactive_input(ctx, params);

                if (params.instruct) {
                    llama_update_input(ctx, inp_sfx);
                }
                input_noecho = true; // do not echo this input again
                is_interacting = false;
            }
            is_interacting = false;
        }

        // end of text token
        if (is_end_of_text) {
            if (params.interactive) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && llama_context_is_finished(ctx)) {
            llama_reset_remaining_tokens(ctx);
            is_interacting = true;
        }
    }
    

#if defined (_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();
        llama_print_end_stats(ctx);
        fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    llama_free_context(ctx_ptr);

    if (params.use_color) {
        printf(ANSI_COLOR_RESET);
    }
    return 0;
}

void process_interactive_input(llama_context& ctx, const gpt_params& params)
{
    bool another_line = true;
    while (another_line) {
        fflush(stdout);
        char buf[256] = {0};
        int n_read;
        if (params.use_color) printf(ANSI_BOLD ANSI_COLOR_GREEN);
        if (scanf("%255[^\n]%n%*c", buf, &n_read) <= 0) {
            // presumable empty line, consume the newline
            std::ignore = scanf("%*c");
            n_read=0;
        }
        if (params.use_color) printf(ANSI_COLOR_RESET);

        if (n_read > 0 && buf[n_read-1]=='\\') {
            another_line = true;
            buf[n_read-1] = '\n';
            buf[n_read] = 0;
        } else {
            another_line = false;
            buf[n_read] = '\n';
            buf[n_read+1] = 0;
        }

        // Do not clear existing context in interactive mode
        llama_update_input(ctx, buf);
    }
}
