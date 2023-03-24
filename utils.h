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
    int32_t seed          = -1;  // RNG seed
    int32_t n_threads     = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict     = 128; // new tokens to predict
    int32_t repeat_last_n = 64;  // last n tokens to penalize
    int32_t n_parts       = -1;  // amount of model parts (-1 = determine from model dimensions)
    int32_t n_ctx         = 512; //context size

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.95f;
    float   temp  = 0.80f;
    float   repeat_penalty  = 1.10f;

    int32_t n_batch = 8; // batch size for prompt processing

    std::string model  = "models/lamma-7B/ggml-model.bin"; // model path
    std::string prompt = "";


    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted

    bool memory_f16        = false; // use f16 instead of f32 for memory kv
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode

    bool embedding         = false; // get only sentence embedding
    bool interactive_start = false; // wait for user input immediately

    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool ignore_eos        = false; // do not stop generating after eos
    bool perplexity        = false; // compute perplexity over the prompt
    bool use_mlock         = false; // use mlock to keep model in memory
};

bool gpt_params_parse(int argc, char ** argv, gpt_params & params);

void gpt_print_usage(int argc, char ** argv, const gpt_params & params);

std::string gpt_random_prompt(std::mt19937 & rng);

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos);
