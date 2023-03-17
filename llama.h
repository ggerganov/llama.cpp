#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "utils.h"
#include "ggml.h"

#ifdef LLAMA_SHARED
#    ifdef _WIN32
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

static const int EOS_TOKEN_ID = 2;


// default hparams (LLaMA 7B)
struct llama_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

struct llama_context;

// Startup
llama_context* llama_init_from_params(const gpt_params& params);
bool llama_prepare_context(llama_context& ctx);

// Input processing and inference
// Tokenize text (never adds BOS)
const std::vector<gpt_vocab::id> llama_tokenize_text(const llama_context& ctx, const std::string& text);
// Queues up a BOS token to the model input
void llama_add_bos(llama_context& ctx);
// Queues up input text to the model input
void llama_update_input(llama_context& ctx, const std::string& text);
// Ingests input previously added using llama_update_input()
void llama_ingest_input_batch(llama_context& ctx);
// Ingests all input previously added using llama_update_input() in multiple batches
// Batch size is determined by gpt_params::n_predict
bool llama_ingest_all_pending_input(llama_context& ctx, bool print_tokens = false);
// Checks if the model has unconsumed input to be ingested using llama_ingest_input_batch()
bool llama_has_unconsumed_input(llama_context& ctx);
// Checks if the model has an anti-prompt present its most recent output
bool llama_is_anti_prompt_present(llama_context& ctx, const std::vector<gpt_vocab::id>& antiprompt_inp);

// Evaluate the model on a batch of input. Must call llama_ingest_input_batch() first.
bool llama_eval_model(llama_context& ctx);
// Checks if the model has finished generating output (i.e. has generated an EOS token or remaining_tokens == 0)
bool llama_context_is_finished(const llama_context& ctx);
// Resets the remaining_tokens counter to the value specified in the gpt_params
void llama_reset_remaining_tokens(const llama_context& ctx);

// Overloaded functions to run inference and return either the model output or the decoded text
bool llama_infer(llama_context& ctx, gpt_vocab::id& model_output);
bool llama_infer(llama_context& ctx, std::string& output, bool& is_end_of_text);

// Teardown
void llama_free_context(llama_context* ctx);

// Getters and setters
gpt_vocab& llama_context_get_vocab(llama_context& ctx);
const std::vector<gpt_vocab::id>& llama_context_get_embedding(const llama_context& ctx);
const std::vector<gpt_vocab::id>& llama_context_get_last_n_tokens(const llama_context& ctx);

// Other
bool llama_model_quantize(const std::string & fname_inp, const std::string & fname_out, int itype);

// Stats
void llama_print_startup_stats(const llama_context& ctx);
void llama_print_end_stats(const llama_context& ctx);
