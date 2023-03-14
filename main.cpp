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

static const int EOS_TOKEN_ID = 2;

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

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";

    return s.c_str();
}

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

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    llama_model model;

    // load the model
    {
        const ggml_type memory_type = params.memory_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        const int64_t t_start_us = ggml_time_us();
        if (!llama_model_load(params.model, model, vocab, params.n_ctx, memory_type)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');
    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::llama_tokenize(vocab, params.prompt, true);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

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

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    for (int i = 0; i < (int) embd_inp.size(); i++) {
        fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    }
    fprintf(stderr, "\n");
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

    std::vector<gpt_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    llama_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    int last_n_size = params.repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
               " - Press Return to return control to LLaMa.\n"
               " - If you want to submit another line, end your input in '\\'.\n\n");
        is_interacting = true;
    }

    int input_consumed = 0;
    bool input_noecho = false;

    int remaining_tokens = params.n_predict;

    // set the color for the prompt which will be output initially
    if (params.use_color) {
        printf(ANSI_COLOR_YELLOW);
    }

    while (remaining_tokens > 0 || params.interactive) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!llama_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) {
                fprintf(stderr, "Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (embd_inp.size() <= input_consumed) {
            // out of user input, sample next token
            const float top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                if (params.ignore_eos) {
                    // set the logit of the eos token to zero to avoid sampling it
                    logits[logits.size() - n_vocab + EOS_TOKEN_ID] = 0;
                }

                id = llama_sample_top_p_top_k(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, rng);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);

            // echo this to console
            input_noecho = false;

            // decrement remaining sampling budget
            --remaining_tokens;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while (embd_inp.size() > input_consumed) {
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
                printf("%s", vocab.id_to_token[id].c_str());
            }
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
        if (!input_noecho && params.use_color && (int)embd_inp.size() == input_consumed) {
            printf(ANSI_COLOR_RESET);
        }

        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive && embd_inp.size() <= input_consumed) {
            // check for reverse prompt
            for (auto antiprompt_inp : antipromptv_inp) {
                if (antiprompt_inp.size() && std::equal(antiprompt_inp.rbegin(), antiprompt_inp.rend(), last_n_tokens.rbegin())) {
                    // reverse prompt found
                    is_interacting = true;
                    break;
                }
            }
            if (is_interacting) {
                if (params.instruct) {
                    input_consumed = embd_inp.size();
                    embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());

                    printf("\n> ");
                }

                // currently being interactive
                if (params.use_color) printf(ANSI_BOLD ANSI_COLOR_GREEN);
                std::string buffer;
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
                if (params.use_color) printf(ANSI_COLOR_RESET);

                std::vector<gpt_vocab::id> line_inp = ::llama_tokenize(vocab, buffer, false);
                embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                if (params.instruct) {
                    embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                }

                remaining_tokens -= line_inp.size();

                input_noecho = true; // do not echo this again
            }
            is_interacting = false;
        }

        // end of text token
        if (embd.back() == EOS_TOKEN_ID) {
            if (params.interactive) {
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

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        fprintf(stderr, "\n\n");
        fprintf(stderr, "%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        fprintf(stderr, "%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        fprintf(stderr, "%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        fprintf(stderr, "%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    if (params.use_color) {
        printf(ANSI_COLOR_RESET);
    }

    return 0;
}
