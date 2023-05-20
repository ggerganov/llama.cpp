#ifndef LLAMA_V2_H
#define LLAMA_V2_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef LLAMA_V2_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_V2_BUILD
#            define LLAMA_V2_API __declspec(dllexport)
#        else
#            define LLAMA_V2_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_V2_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_V2_API
#endif

#define LLAMA_V2_FILE_VERSION           3
#define LLAMA_V2_FILE_MAGIC             'ggjt'
#define LLAMA_V2_FILE_MAGIC_UNVERSIONED 'ggml'
#define LLAMA_V2_SESSION_MAGIC          'ggsn'
#define LLAMA_V2_SESSION_VERSION        1

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_v2_context;

    typedef int llama_v2_token;

    typedef struct llama_v2_token_data {
        llama_v2_token id;  // token id
        float logit; // log-odds of the token
        float p;     // probability of the token
    } llama_v2_token_data;

    typedef struct llama_v2_token_data_array {
        llama_v2_token_data * data;
        size_t size;
        bool sorted;
    } llama_v2_token_data_array;

    typedef void (*llama_v2_progress_callback)(float progress, void *ctx);

    struct llama_v2_context_params {
        int n_ctx;        // text context
        int n_gpu_layers; // number of layers to store in VRAM
        int seed;         // RNG seed, -1 for random

        bool f16_kv;     // use fp16 for KV cache
        bool logits_all; // the llama_v2_eval() call computes all logits, not just the last one
        bool vocab_only; // only load the vocabulary, no weights
        bool use_mmap;   // use mmap if possible
        bool use_mlock;  // force system to keep model in RAM
        bool embedding;  // embedding mode only

        // called with a progress value between 0 and 1, pass NULL to disable
        llama_v2_progress_callback progress_callback;
        // context pointer passed to the progress callback
        void * progress_callback_user_data;
    };

    // model file types
    enum llama_v2_ftype {
        LLAMA_V2_FTYPE_ALL_F32     = 0,
        LLAMA_V2_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        LLAMA_V2_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        LLAMA_V2_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        LLAMA_V2_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        LLAMA_V2_FTYPE_MOSTLY_Q4_2 = 5,  // except 1d tensors
        LLAMA_V2_FTYPE_MOSTLY_Q4_3 = 6,  // except 1d tensors
        LLAMA_V2_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        LLAMA_V2_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        LLAMA_V2_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
    };

    LLAMA_V2_API struct llama_v2_context_params llama_v2_context_default_params();

    LLAMA_V2_API bool llama_v2_mmap_supported();
    LLAMA_V2_API bool llama_v2_mlock_supported();

    // Various functions for loading a ggml llama model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    LLAMA_V2_API struct llama_v2_context * llama_v2_init_from_file(
                             const char * path_model,
            struct llama_v2_context_params   params);

    // Frees all allocated memory
    LLAMA_V2_API void llama_v2_free(struct llama_v2_context * ctx);

    // TODO: not great API - very likely to change
    // Returns 0 on success
    // nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given
    LLAMA_V2_API int llama_v2_model_quantize(
            const char * fname_inp,
            const char * fname_out,
      enum llama_v2_ftype   ftype,
            int          nthread);

    // Apply a LoRA adapter to a loaded model
    // path_base_model is the path to a higher quality model to use as a base for
    // the layers modified by the adapter. Can be NULL to use the current loaded model.
    // The model needs to be reloaded before applying a new adapter, otherwise the adapter
    // will be applied on top of the previous one
    // Returns 0 on success
    LLAMA_V2_API int llama_v2_apply_lora_from_file(
            struct llama_v2_context * ctx,
                      const char * path_lora,
                      const char * path_base_model,
                             int   n_threads);

    // Returns the number of tokens in the KV cache
    LLAMA_V2_API int llama_v2_get_kv_cache_token_count(const struct llama_v2_context * ctx);

    // Sets the current rng seed.
    LLAMA_V2_API void llama_v2_set_rng_seed(struct llama_v2_context * ctx, int seed);

    // Returns the maximum size in bytes of the state (rng, logits, embedding
    // and kv_cache) - will often be smaller after compacting tokens
    LLAMA_V2_API size_t llama_v2_get_state_size(const struct llama_v2_context * ctx);

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLAMA_V2_API size_t llama_v2_copy_state_data(struct llama_v2_context * ctx, uint8_t * dst);

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLAMA_V2_API size_t llama_v2_set_state_data(struct llama_v2_context * ctx, const uint8_t * src);

    // Save/load session file
    LLAMA_V2_API bool llama_v2_load_session_file(struct llama_v2_context * ctx, const char * path_session, llama_v2_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
    LLAMA_V2_API bool llama_v2_save_session_file(struct llama_v2_context * ctx, const char * path_session, const llama_v2_token * tokens, size_t n_token_count);

    // Run the llama inference to obtain the logits and probabilities for the next token.
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    LLAMA_V2_API int llama_v2_eval(
            struct llama_v2_context * ctx,
               const llama_v2_token * tokens,
                             int   n_tokens,
                             int   n_past,
                             int   n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    LLAMA_V2_API int llama_v2_tokenize(
            struct llama_v2_context * ctx,
                      const char * text,
                     llama_v2_token * tokens,
                             int   n_max_tokens,
                            bool   add_bos);

    
    std::vector<llama_v2_token> legacy_llama_v2_tokenize(struct llama_v2_context * ctx, const std::string & text, bool add_bos);

    LLAMA_V2_API int llama_v2_n_vocab(const struct llama_v2_context * ctx);
    LLAMA_V2_API int llama_v2_n_ctx  (const struct llama_v2_context * ctx);
    LLAMA_V2_API int llama_v2_n_embd (const struct llama_v2_context * ctx);

    // Token logits obtained from the last call to llama_v2_eval()
    // The logits for the last token are stored in the last row
    // Can be mutated in order to change the probabilities of the next token
    // Rows: n_tokens
    // Cols: n_vocab
    LLAMA_V2_API float * llama_v2_get_logits(struct llama_v2_context * ctx);

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    LLAMA_V2_API float * llama_v2_get_embeddings(struct llama_v2_context * ctx);

    // Token Id -> String. Uses the vocabulary in the provided context
    LLAMA_V2_API const char * llama_v2_token_to_str(const struct llama_v2_context * ctx, llama_v2_token token);

    // Special tokens
    LLAMA_V2_API llama_v2_token llama_v2_token_bos();
    LLAMA_V2_API llama_v2_token llama_v2_token_eos();
    LLAMA_V2_API llama_v2_token llama_v2_token_nl();

    // Sampling functions

    /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    LLAMA_V2_API void llama_v2_sample_repetition_penalty(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, const llama_v2_token * last_tokens, size_t last_tokens_size, float penalty);

    /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    LLAMA_V2_API void llama_v2_sample_frequency_and_presence_penalties(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, const llama_v2_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    LLAMA_V2_API void llama_v2_sample_softmax(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_V2_API void llama_v2_sample_top_k(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, int k, size_t min_keep);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_V2_API void llama_v2_sample_top_p(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float p, size_t min_keep);

    /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    LLAMA_V2_API void llama_v2_sample_tail_free(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float z, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_V2_API void llama_v2_sample_typical(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float p, size_t min_keep);
    LLAMA_V2_API void llama_v2_sample_temperature(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float temp);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_v2_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_V2_API llama_v2_token llama_v2_sample_token_mirostat(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float tau, float eta, int m, float * mu);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_v2_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_V2_API llama_v2_token llama_v2_sample_token_mirostat_v2(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates, float tau, float eta, float * mu);

    /// @details Selects the token with the highest probability.
    LLAMA_V2_API llama_v2_token llama_v2_sample_token_greedy(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates);

    /// @details Randomly selects a token from the candidates based on their probabilities.
    LLAMA_V2_API llama_v2_token llama_v2_sample_token(struct llama_v2_context * ctx, llama_v2_token_data_array * candidates);

    // Performance information
    LLAMA_V2_API void llama_v2_print_timings(struct llama_v2_context * ctx);
    LLAMA_V2_API void llama_v2_reset_timings(struct llama_v2_context * ctx);

    // Print system information
    LLAMA_V2_API const char * llama_v2_print_system_info(void);

#ifdef __cplusplus
}
#endif

// Internal API to be implemented by llama.cpp and used by tests/benchmarks only
#ifdef LLAMA_V2_API_INTERNAL

#include <vector>
#include <string>
struct ggml_v2_tensor;

std::vector<std::pair<std::string, struct ggml_v2_tensor *>>& llama_v2_internal_get_tensor_map(struct llama_v2_context * ctx);

#endif

#endif // LLAMA_V2_H
