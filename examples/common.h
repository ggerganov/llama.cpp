// Various helper functions and utilities

#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <random>
#include <thread>

//
// CLI argument parsing
//

struct gpt_params {
    int32_t seed          = -1;    // RNG seed
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency()); // max 4 threads (default)
    int32_t n_predict     = 128;   // new tokens to predict
    int32_t repeat_last_n = 64;    // last n tokens to penalize
    int32_t n_parts       = -1;    // amount of model parts (-1 = determine from model dimensions)
    int32_t n_ctx         = 512;   // context size
    int32_t n_batch       = 8;     // batch size for prompt processing
    int32_t n_keep        = 0;     // number of tokens to keep from initial prompt (-1 for all)

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    float   repeat_penalty  = 1.10f;

    std::string model  = "models/lamma-7B/ggml-model.bin"; // model path
    std::string prompt = "";
    std::string input_prefix = ""; // string to prefix user inputs with

    std::string instruct_prefix = ""; // prefix user inputs with tokenized string
    bool instruct_prefix_bos = false; // prepend bos token to instruct prefix
    std::string instruct_suffix = ""; // suffix user inputs with tokenized string
    bool instruct_suffix_bos = false; // prepend bos token to instruct suffix

    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted
    std::vector<std::string> stopprompt; // string upon seeing which more user input is prompted (without adding instruct prefixes and suffixes)

    bool rm_trailing_space_workaround = false; // workaround for removing trailing space from reverse/stop prompts

    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode

    bool embedding         = false; // get only sentence embedding
    bool interactive_start = false; // wait for user input immediately

    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool ignore_eos        = false; // do not stop generating after eos
    bool perplexity        = false; // compute perplexity over the prompt
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool mem_test          = false; // compute maximum memory usage
    bool verbose_prompt    = false; // print prompt tokens before generation

    bool clean_interface   = false; // hides input prefix & suffix and displays '>'
    bool multiline_mode    = true; // enables multi-line mode, to send input press CTRL+D on Linux/Max, Ctrl+Z then Return on Windows
};

bool gpt_params_parse(int argc, char ** argv, gpt_params & params);

void gpt_print_usage(char * argv_0, const gpt_params & params);

std::string gpt_random_prompt(std::mt19937 & rng);

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos);

//
// Console utils
//

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

enum console_color_t {
    CONSOLE_COLOR_DEFAULT=0,
    CONSOLE_COLOR_PROMPT,
    CONSOLE_COLOR_USER_INPUT
};

struct console_state {
    bool use_color = false;
    console_color_t color = CONSOLE_COLOR_DEFAULT;
};

void set_console_color(console_state & con_st, console_color_t color);

#if defined (_WIN32)
void win32_console_init(bool enable_color);
void win32_utf8_encode(const std::wstring & wstr, std::string & str);
#endif

bool get_input_text(std::string & input_text, bool escape_newline_mode);
