// Various helper functions and utilities

#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>
#include <tuple>

//
// CLI argument parsing
//
int32_t get_num_physical_cores();

struct gpt_params {
    uint32_t seed                           = -1;   // RNG seed
    int32_t n_threads                       = get_num_physical_cores();
    int32_t n_predict                       = -1;   // new tokens to predict
    int32_t n_ctx                           = 512;  // context size
    int32_t n_batch                         = 512;  // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                          = 0;    // number of tokens to keep from initial prompt
    int32_t n_chunks                        = -1;   // max number of chunks to process (-1 = unlimited)
    int32_t n_gpu_layers                    = 0;    // number of layers to store in VRAM
    int32_t main_gpu                        = 0;    // the GPU that is used for scratch and small tensors
    float   tensor_split[LLAMA_MAX_DEVICES] = {0};  // how split tensors should be distributed across GPUs
    int32_t n_probs                         = 0;    // if greater than 0, output the probabilities of top n_probs tokens.
    float   rope_freq_base                  = 10000.0f; // RoPE base frequency
    float   rope_freq_scale                 = 1.0f;     // RoPE frequency scaling factor

    // sampling parameters
    int32_t top_k             = 40;    // <= 0 to use vocab size
    float   top_p             = 0.95f; // 1.0 = disabled
    float   tfs_z             = 1.00f; // 1.0 = disabled
    float   typical_p         = 1.00f; // 1.0 = disabled
    float   temp              = 0.80f; // 1.0 = disabled
    float   repeat_penalty    = 1.10f; // 1.0 = disabled
    int32_t repeat_last_n     = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   frequency_penalty = 0.00f; // 0.0 = disabled
    float   presence_penalty  = 0.00f; // 0.0 = disabled
    int32_t mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau      = 5.00f; // target entropy
    float   mirostat_eta      = 0.10f; // learning rate

    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    std::string cfg_negative_prompt;       // string to help guidance
    float       cfg_scale         = 1.f;   // How strong is guidance

    std::string model             = "models/7B/ggml-model-f16.gguf"; // model path
    std::string model_alias       = "unknown"; // model alias
    std::string prompt            = "";
    std::string path_prompt_cache = "";  // path to file for saving/loading prompt eval state
    std::string input_prefix      = "";  // string to prefix user inputs with
    std::string input_suffix      = "";  // string to suffix user inputs with
    std::string grammar           = "";  // optional BNF-like grammar to constrain sampling
    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted

    std::string lora_adapter = "";  // lora adapter path
    std::string lora_base    = "";  // base model path for the lora adapter

    bool hellaswag         = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks = 400;   // number of tasks to use when computing the HellaSwag score

    bool low_vram          = false; // if true, reduce VRAM usage at the cost of performance
    bool mul_mat_q         = false; // if true, use experimental mul_mat_q kernels
    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool embedding         = false; // get only sentence embedding
    bool interactive_first = false; // wait for user input immediately
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool ignore_eos        = false; // ignore generated EOS tokens
    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool penalize_nl       = true;  // consider newlines as a repeatable token
    bool perplexity        = false; // compute perplexity over the prompt
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool mem_test          = false; // compute maximum memory usage
    bool numa              = false; // attempt optimizations that help on some NUMA systems
    bool export_cgraph     = false; // export the computation graph
    bool verbose_prompt    = false; // print prompt tokens before generation
};

bool gpt_params_parse(int argc, char ** argv, gpt_params & params);

void gpt_print_usage(int argc, char ** argv, const gpt_params & params);

std::string gpt_random_prompt(std::mt19937 & rng);

//
// Model utils
//

std::tuple<struct llama_model *, struct llama_context *> llama_init_from_gpt_params(gpt_params & params);
struct llama_context_params llama_context_params_from_gpt_params(const gpt_params & params);

//
// Vocab utils
//

std::vector<llama_token> llama_tokenize(
        struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos);

std::vector<llama_token> llama_tokenize_bpe(
        struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos);

std::string llama_token_to_str(
        const struct llama_context * ctx,
                       llama_token   token);

std::string llama_token_to_str_bpe(
    const struct llama_context * ctx,
                   llama_token   token);
