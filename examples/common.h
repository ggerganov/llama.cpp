// Various helper functions and utilities

#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>

//
// CLI argument parsing
//
int32_t get_num_physical_cores();

struct gpt_params {
    int32_t seed          = -1;   // RNG seed
    int32_t n_threads     = get_num_physical_cores();
    int32_t n_predict     = -1;  // new tokens to predict
    int32_t n_parts       = -1;   // amount of model parts (-1 = determine from model dimensions)
    int32_t n_ctx         = 512;  // context size
    int32_t n_batch       = 512;  // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep        = 0;    // number of tokens to keep from initial prompt

    // sampling parameters
    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typical_p         = 1.00f; // 1.0 = disabled
    float   temp              = 0.80f; // 1.0 = disabled
    float   repeat_penalty    = 1.10f; // 1.0 = disabled
    int32_t repeat_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   frequency_penalty = 0.00f; // 0.0 = disabled
    float   presence_penalty  = 0.00f; // 0.0 = disabled
    int     mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate

    std::string model  = "models/lamma-7B/ggml-model.bin"; // model path
    std::string prompt = "";
    std::string path_session = "";       // path to file for saving/loading model eval state
    std::string input_prefix = "";       // string to prefix user inputs with
    std::string input_suffix = "";       // string to suffix user inputs with
    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted

    std::string lora_adapter = "";  // lora adapter path
    std::string lora_base = "";     // base model path for the lora adapter

    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode

    bool embedding         = false; // get only sentence embedding
    bool interactive_first = false; // wait for user input immediately

    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool penalize_nl       = true;  // consider newlines as a repeatable token
    bool perplexity        = false; // compute perplexity over the prompt
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool mem_test          = false; // compute maximum memory usage
    bool verbose_prompt    = false; // print prompt tokens before generation
};

bool gpt_params_parse(int argc, char ** argv, gpt_params & params);

void gpt_print_usage(int argc, char ** argv, const gpt_params & params);

std::string gpt_random_prompt(std::mt19937 & rng);

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos);

//
// Model utils
//

struct llama_context * llama_init_from_gpt_params(const gpt_params & params);

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
