// Various helper functions and utilities

#pragma once

#include "llama.h"

#include "sampling.h"

#define LOG_NO_FILE_LINE_FUNCTION
#include "log.h"

#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <unordered_map>
#include <tuple>

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

#define print_build_info() do {                                                                     \
    fprintf(stderr, "%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);           \
    fprintf(stderr, "%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);    \
} while(0)

// build info
extern int LLAMA_BUILD_NUMBER;
extern char const *LLAMA_COMMIT;
extern char const *LLAMA_COMPILER;
extern char const *LLAMA_BUILD_TARGET;

//
// CLI argument parsing
//
int32_t get_num_physical_cores();

struct gpt_params {
    uint32_t seed                           = -1;    // RNG seed

    int32_t n_threads                       = get_num_physical_cores();
    int32_t n_threads_batch                 = -1;    // number of threads to use for batch processing (-1 = use n_threads)
    int32_t n_predict                       = -1;    // new tokens to predict
    int32_t n_ctx                           = 512;   // context size
    int32_t n_batch                         = 512;   // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                          = 0;     // number of tokens to keep from initial prompt
    int32_t n_draft                         = 16;    // number of tokens to draft during speculative decoding
    int32_t n_chunks                        = -1;    // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel                      = 1;     // number of parallel sequences to decode
    int32_t n_sequences                     = 1;     // number of sequences to decode
    float   p_accept                        = 0.5f;  // speculative decoding accept probability
    float   p_split                         = 0.1f;  // speculative decoding split probability
    int32_t n_gpu_layers                    = -1;    // number of layers to store in VRAM (-1 - use default)
    int32_t n_gpu_layers_draft              = -1;    // number of layers to store in VRAM for the draft model (-1 - use default)
    int32_t main_gpu                        = 0;     // the GPU that is used for scratch and small tensors
    float   tensor_split[LLAMA_MAX_DEVICES] = {0};   // how split tensors should be distributed across GPUs
    int32_t n_beams                         = 0;     // if non-zero then use beam search of given width.
    float   rope_freq_base                  = 0.0f;  // RoPE base frequency
    float   rope_freq_scale                 = 0.0f;  // RoPE frequency scaling factor
    float   yarn_ext_factor                 = -1.0f; // YaRN extrapolation mix factor
    float   yarn_attn_factor                = 1.0f;  // YaRN magnitude scaling factor
    float   yarn_beta_fast                  = 32.0f; // YaRN low correction dim
    float   yarn_beta_slow                  = 1.0f;  // YaRN high correction dim
    int32_t yarn_orig_ctx                   = 0;     // YaRN original context length
    int8_t  rope_scaling_type               = LLAMA_ROPE_SCALING_UNSPECIFIED; // TODO: better to be int32_t for alignment
                                                                              //       pinging @cebtenzzre

    // // sampling parameters
    struct llama_sampling_params sparams;

    std::string model             = "models/7B/ggml-model-f16.gguf"; // model path
    std::string model_draft       = "";                              // draft model for speculative decoding
    std::string model_alias       = "unknown"; // model alias
    std::string prompt            = "";
    std::string prompt_file       = "";  // store the external prompt file name
    std::string path_prompt_cache = "";  // path to file for saving/loading prompt eval state
    std::string input_prefix      = "";  // string to prefix user inputs with
    std::string input_suffix      = "";  // string to suffix user inputs with
    std::vector<std::string> antiprompt; // string upon seeing which more user input is prompted
    std::string logdir            = "";  // directory in which to save YAML log files

    std::vector<llama_model_kv_override> kv_overrides;

    // TODO: avoid tuple, use struct
    std::vector<std::tuple<std::string, float>> lora_adapter; // lora adapter path with user defined scale
    std::string lora_base  = "";                              // base model path for the lora adapter

    int  ppl_stride        = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int  ppl_output_type   = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                    //                                       (which is more convenient to use for plotting)
                                    //
    bool   hellaswag       = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks = 400;   // number of tasks to use when computing the HellaSwag score

    bool mul_mat_q         = true;  // if true, use mul_mat_q kernels instead of cuBLAS
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode
    bool chatml            = false; // chatml mode (used for models trained on chatml syntax)
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool embedding         = false; // get only sentence embedding
    bool escape            = false; // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool interactive_first = false; // wait for user input immediately
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles
    bool cont_batching     = false; // insert new sequences for decoding on-the-fly

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool ignore_eos        = false; // ignore generated EOS tokens
    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool logits_all        = false; // return logits for all tokens in the batch
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool numa              = false; // attempt optimizations that help on some NUMA systems
    bool verbose_prompt    = false; // print prompt tokens before generation
    bool infill            = false; // use infill mode
    bool dump_kv_cache     = false; // dump the KV cache contents for debugging purposes
    bool no_kv_offload     = false; // disable KV offloading

    std::string cache_type_k = "f16"; // KV cache data type for the K
    std::string cache_type_v = "f16"; // KV cache data type for the V

    // multimodal models (see examples/llava)
    std::string mmproj = ""; // path to multimodal projector
    std::string image  = ""; // path to an image file
};

bool gpt_params_parse_ex(int argc, char ** argv, gpt_params & params);

bool gpt_params_parse(int argc, char ** argv, gpt_params & params);

void gpt_print_usage(int argc, char ** argv, const gpt_params & params);

std::string get_system_info(const gpt_params & params);

std::string gpt_random_prompt(std::mt19937 & rng);

void process_escapes(std::string& input);

//
// String parsing
//

std::string parse_samplers_input(std::string input);

//
// Model utils
//

// TODO: avoid tuplue, use struct
std::tuple<struct llama_model *, struct llama_context *> llama_init_from_gpt_params(gpt_params & params);

struct llama_model_params   llama_model_params_from_gpt_params  (const gpt_params & params);
struct llama_context_params llama_context_params_from_gpt_params(const gpt_params & params);

// Batch utils

void llama_batch_clear(struct llama_batch & batch);

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> llama_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_bos,
                        bool   special = false);

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_bos,
                        bool   special = false);

// tokenizes a token into a piece
// should work similar to Python's `tokenizer.id_to_piece`
std::string llama_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token);

// TODO: these should be moved in llama.h C-style API under single `llama_detokenize` function
//       that takes into account the tokenizer type and decides how to handle the leading space
//
// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// removes the leading space from the first non-BOS token
std::string llama_detokenize_spm(
                         llama_context * ctx,
        const std::vector<llama_token> & tokens);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
std::string llama_detokenize_bpe(
                         llama_context * ctx,
        const std::vector<llama_token> & tokens);

// Uses the value from the model metadata if possible, otherwise
// defaults to true when model type is SPM, otherwise false.
bool llama_should_add_bos_token(const llama_model * model);

//
// YAML utils
//

bool create_directory_with_parents(const std::string & path);
void dump_vector_float_yaml(FILE * stream, const char * prop_name, const std::vector<float> & data);
void dump_vector_int_yaml(FILE * stream, const char * prop_name, const std::vector<int> & data);
void dump_string_yaml_multiline(FILE * stream, const char * prop_name, const char * data);
std::string get_sortable_timestamp();

void dump_non_result_info_yaml(
    FILE * stream, const gpt_params & params, const llama_context * lctx,
    const std::string & timestamp, const std::vector<int> & prompt_tokens, const char * model_desc);

//
// KV cache utils
//

// Dump the KV cache view with the number of sequences per cell.
void dump_kv_cache_view(const llama_kv_cache_view & view, int row_size = 80);

// Dump the KV cache view showing individual sequences in each cell (long output).
void dump_kv_cache_view_seqs(const llama_kv_cache_view & view, int row_size = 40);
