#include "utils.h"
#include "ggml.h"
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

#if defined (_WIN32)
#pragma comment(lib,"kernel32.lib")
extern "C" __declspec(dllimport) void* __stdcall GetStdHandle(unsigned long nStdHandle);
extern "C" __declspec(dllimport) int __stdcall GetConsoleMode(void* hConsoleHandle, unsigned long* lpMode);
extern "C" __declspec(dllimport) int __stdcall SetConsoleMode(void* hConsoleHandle, unsigned long dwMode);
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

/* Keep track of current color of output, and emit ANSI code if it changes. */
enum console_state {
    CONSOLE_STATE_DEFAULT=0,
    CONSOLE_STATE_PROMPT,
    CONSOLE_STATE_USER_INPUT
};

static console_state con_st = CONSOLE_STATE_DEFAULT;
static bool con_use_color = false;

void set_console_state(console_state new_st)
{
    if (!con_use_color) return;
    // only emit color code if state changed
    if (new_st != con_st) {
        con_st = new_st;
        switch(con_st) {
        case CONSOLE_STATE_DEFAULT:
            printf(ANSI_COLOR_RESET);
            return;
        case CONSOLE_STATE_PROMPT:
            printf(ANSI_COLOR_YELLOW);
            return;
        case CONSOLE_STATE_USER_INPUT:
            printf(ANSI_BOLD ANSI_COLOR_GREEN);
            return;
        }
    }
}

std::vector<double> softmax(const std::vector<float>& logits) {
    std::vector<double> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) max_logit = std::max(max_logit, v);
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        float logit = logits[i] - max_logit;
        double exp_logit = std::exp(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) probs[i] /= sum_exp;
    return probs;
}

void perplexity(llama_context * ctx, const gpt_params & params) {
    // Download: https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
    // Run `./main --perplexity -m models/7B/ggml-model-q4_0.bin -f wiki.test.raw`
    // Output: `perplexity: 13.5106 [114/114]`
    auto tokens = ::llama_tokenize(ctx, params.prompt, true);

    int count = 0;
    double nll = 0.0;
    int seq_count = tokens.size() / params.n_ctx;

    fprintf(stderr, "%s : calculating perplexity over %d chunks\n", __func__, seq_count);

    for (int i = 0; i < seq_count; ++i) {
        int start = i * params.n_ctx;
        int end = start + params.n_ctx - 1;
        std::vector<llama_token> embd(tokens.begin() + start, tokens.begin() + end);
        auto start_t = std::chrono::high_resolution_clock::now();
        if (llama_eval(ctx, embd.data(), embd.size(), 0, params.n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return;
        }
        auto end_t = std::chrono::high_resolution_clock::now();
        if (i == 0) {
            double seconds = std::chrono::duration<double>(end_t - start_t).count();
            printf("%.2f seconds per pass - ETA %.2f hours\n", seconds, (seconds * seq_count) / (60.0*60.0));
        }
        // We get the logits for all the tokens in the context window (params.n_ctx)
        // from llama_eval above.  Now, based on https://huggingface.co/docs/transformers/perplexity,
        // calculate the perplexity over the last half the window (so the model always has
        // some context to predict the token).
        //
        // We rely on the fact that attention in the forward pass only looks at previous
        // tokens here, so the logits returned for each token are an accurate representation
        // of what the model would have predicted at that point.
        //
        // Example, we have a context window of 512, we will compute perplexity for each of the
        // last 256 tokens.  Then, we split the input up into context window size chunks to
        // process the entire prompt.

        auto logits = llama_get_logits(ctx);
        for (int j = params.n_ctx / 2; j < params.n_ctx - 1; ++j) {
            // Calculate probability of next token, given the previous ones.
            int n_vocab = llama_n_vocab(ctx);
            std::vector<float> tok_logits(
                logits + j * n_vocab,
                logits + (j + 1) * n_vocab);
            double prob = softmax(tok_logits)[tokens[start + j + 1]];
            nll += -std::log(prob);
            ++count;
        }
        // perplexity is e^(average negative log-likelihood)
        printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
        fflush(stdout);
    }
    printf("\n");
}

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    set_console_state(CONSOLE_STATE_DEFAULT);
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

int main(int argc, char ** argv) {
    // has to be called once at the start of the program to init ggml stuff
    ggml_time_init();

    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
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

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    con_use_color = params.use_color;

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
        lparams.logits_all = params.perplexity;
        lparams.use_mlock  = params.use_mlock;
        lparams.embedding  = params.embedding;

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

    if (params.perplexity) {
        perplexity(ctx, params);
        exit(0);
    }

    int n_past = 0;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);

    params.n_predict = std::min(params.n_predict, n_ctx - (int) embd_inp.size());

    // prefix & suffix for instruct mode
    const auto inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", true);
    const auto inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n", false);

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive = true;
        params.antiprompt.push_back("### Instruction:\n\n");
    }

    // enable interactive mode if reverse prompt is specified
    if (params.antiprompt.size() != 0) {
        params.interactive = true;
    }

    if (params.interactive_start) {
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

        if(params.antiprompt.size()) {
            for (auto antiprompt : params.antiprompt) {
                fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }

        if (!params.input_prefix.empty()) {
            fprintf(stderr, "Input prefix: '%s'\n", params.input_prefix.c_str());
        }
    }
    fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "\n\n");

    std::vector<llama_token> embd;


    int last_n_size = params.repeat_last_n;
    std::vector<llama_token> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
               " - Press Return to return control to LLaMa.\n"
               " - If you want to submit another line, end your input in '\\'.\n\n");
        is_interacting = params.interactive_start || params.instruct;
    }

    int input_consumed = 0;
    bool input_noecho = false;

    int remaining_tokens = params.n_predict;

#if defined (_WIN32)
  if (params.use_color) {
        // Enable ANSI colors on Windows 10+
        unsigned long dwMode = 0;
        void* hConOut = GetStdHandle((unsigned long)-11); // STD_OUTPUT_HANDLE (-11)
        if (hConOut && hConOut != (void*)-1 && GetConsoleMode(hConOut, &dwMode) && !(dwMode & 0x4)) {
            SetConsoleMode(hConOut, dwMode | 0x4); // ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x4)
        }
    }
#endif
    // the first thing we will do is to output the prompt, so set color accordingly
    set_console_state(CONSOLE_STATE_PROMPT);

    if (params.embedding){
        embd = embd_inp;

        if (embd.size() > 0) {
            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        const auto embeddings = llama_get_embeddings(ctx);

        // TODO: print / use the embeddings

        if (params.use_color) {
            printf(ANSI_COLOR_RESET);
        }

        return 0;
    }

    while (remaining_tokens > 0 || params.interactive) {
        // predict
        if (embd.size() > 0) {
            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int) embd_inp.size() <= input_consumed && !is_interacting) {
            // out of user input, sample next token
            const float top_k          = params.top_k;
            const float top_p          = params.top_p;
            const float temp           = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(ctx);

                if (params.ignore_eos) {
                    // set the logit of the eos token to zero to avoid sampling it
                    //logits[logits.size() - n_vocab + EOS_TOKEN_ID] = 0;
                    // TODO: this does not work of params.logits_all == true
                    assert(params.perplexity == false);
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(ctx, last_n_tokens.data(), last_n_tokens.size(), top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && params.interactive && !params.instruct) {
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
            --remaining_tokens;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > input_consumed) {
                embd.push_back(embd_inp[input_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[input_consumed]);
                ++input_consumed;
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
        if (!input_noecho && (int)embd_inp.size() == input_consumed) {
            set_console_state(CONSOLE_STATE_DEFAULT);
        }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && (int) embd_inp.size() <= input_consumed) {
            // check for reverse prompt
            std::string last_output;
            for (auto id : last_n_tokens) {
                last_output += llama_token_to_str(ctx, id);
            }

            // Check if each of the reverse prompts appears at the end of the output.
            for (std::string & antiprompt : params.antiprompt) {
                if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                    is_interacting = true;
                    set_console_state(CONSOLE_STATE_USER_INPUT);
                    fflush(stdout);
                    break;
                }
            }

            if (n_past > 0 && is_interacting) {
                // potentially set color to indicate we are taking user input
                set_console_state(CONSOLE_STATE_USER_INPUT);

                if (params.instruct) {
                    input_consumed = embd_inp.size();
                    embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());

                    printf("\n> ");
                }

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    buffer += params.input_prefix;
                    printf("%s", buffer.c_str());
                }

                std::string line;
                bool another_line = true;
                do {
                    std::getline(std::cin, line);
                    if (line.empty() || line.back() != '\\') {
                        another_line = false;
                    } else {
                        line.pop_back(); // Remove the continue character
                    }
                    buffer += line + '\n'; // Append the line to the result
                } while (another_line);

                // done taking input, reset color
                set_console_state(CONSOLE_STATE_DEFAULT);

                auto line_inp = ::llama_tokenize(ctx, buffer, false);
                embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                if (params.instruct) {
                    embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                }

                remaining_tokens -= line_inp.size();

                input_noecho = true; // do not echo this again
            }

            if (n_past > 0) {
                is_interacting = false;
            }
        }

        // end of text token
        if (embd.back() == llama_token_eos()) {
            if (params.instruct) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && remaining_tokens <= 0) {
            remaining_tokens = params.n_predict;
            is_interacting = true;
        }
    }

#if defined (_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    llama_print_timings(ctx);
    llama_free(ctx);

    set_console_state(CONSOLE_STATE_DEFAULT);

    return 0;
}
