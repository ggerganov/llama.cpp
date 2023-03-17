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

// Input processing and inference
bool llama_ingest_input(llama_context& ctx, const std::string& text, bool clear_existing = true);
bool llama_context_is_finished(const llama_context& ctx);
bool llama_update_context_with_prompt(llama_context& ctx, const std::string& text, bool clear_existing = true);
const std::vector<gpt_vocab::id> llama_tokenize_text(const llama_context& ctx, const std::string& text);
bool llama_infer(llama_context& ctx, gpt_vocab::id& model_output);

// Teardown
void llama_free_context(llama_context* ctx);

// Getters and setters
gpt_vocab& llama_context_get_vocab(llama_context& ctx);
const std::vector<gpt_vocab::id>& llama_context_get_embedding(const llama_context& ctx);
const std::vector<gpt_vocab::id>& llama_context_get_last_n_tokens(const llama_context& ctx);

// Other
bool llama_model_quantize(const std::string & fname_inp, const std::string & fname_out, int itype);

// Stats
void llama_print_context_info(const llama_context& ctx);
void llama_print_end_stats(const llama_context& ctx);
