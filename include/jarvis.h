#ifndef JARVIS_H
#define JARVIS_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef JARVIS_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef JARVIS_BUILD
#            define JARVIS_API __declspec(dllexport)
#        else
#            define JARVIS_API __declspec(dllimport)
#        endif
#    else
#        define JARVIS_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define JARVIS_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define JARVIS_DEFAULT_SEED 0xFFFFFFFF

// TODO: use everywhere in the implementation
#define JARVIS_TOKEN_NULL -1

#define JARVIS_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define JARVIS_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define JARVIS_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define JARVIS_SESSION_MAGIC   JARVIS_FILE_MAGIC_GGSN
#define JARVIS_SESSION_VERSION 9

#define JARVIS_STATE_SEQ_MAGIC   JARVIS_FILE_MAGIC_GGSQ
#define JARVIS_STATE_SEQ_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    // struct jarvis_vocab; // TODO: add in the future
    struct jarvis_model;
    struct jarvis_context;
    struct jarvis_sampler;

    typedef int32_t jarvis_pos;
    typedef int32_t jarvis_token;
    typedef int32_t jarvis_seq_id;

    enum jarvis_vocab_type {
        JARVIS_VOCAB_TYPE_NONE = 0, // For models without vocab
        JARVIS_VOCAB_TYPE_SPM  = 1, // JARVIS tokenizer based on byte-level BPE with byte fallback
        JARVIS_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
        JARVIS_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
        JARVIS_VOCAB_TYPE_UGM  = 4, // T5 tokenizer based on Unigram
        JARVIS_VOCAB_TYPE_RWKV = 5, // RWKV tokenizer based on greedy tokenization
    };

    // pre-tokenization types
    enum jarvis_vocab_pre_type {
        JARVIS_VOCAB_PRE_TYPE_DEFAULT        = 0,
        JARVIS_VOCAB_PRE_TYPE_JARVIS3         = 1,
        JARVIS_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,
        JARVIS_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
        JARVIS_VOCAB_PRE_TYPE_FALCON         = 4,
        JARVIS_VOCAB_PRE_TYPE_MPT            = 5,
        JARVIS_VOCAB_PRE_TYPE_STARCODER      = 6,
        JARVIS_VOCAB_PRE_TYPE_GPT2           = 7,
        JARVIS_VOCAB_PRE_TYPE_REFACT         = 8,
        JARVIS_VOCAB_PRE_TYPE_COMMAND_R      = 9,
        JARVIS_VOCAB_PRE_TYPE_STABLELM2      = 10,
        JARVIS_VOCAB_PRE_TYPE_QWEN2          = 11,
        JARVIS_VOCAB_PRE_TYPE_OLMO           = 12,
        JARVIS_VOCAB_PRE_TYPE_DBRX           = 13,
        JARVIS_VOCAB_PRE_TYPE_SMAUG          = 14,
        JARVIS_VOCAB_PRE_TYPE_PORO           = 15,
        JARVIS_VOCAB_PRE_TYPE_CHATGLM3       = 16,
        JARVIS_VOCAB_PRE_TYPE_CHATGLM4       = 17,
        JARVIS_VOCAB_PRE_TYPE_VIKING         = 18,
        JARVIS_VOCAB_PRE_TYPE_JAIS           = 19,
        JARVIS_VOCAB_PRE_TYPE_TEKKEN         = 20,
        JARVIS_VOCAB_PRE_TYPE_SMOLLM         = 21,
        JARVIS_VOCAB_PRE_TYPE_CODESHELL      = 22,
        JARVIS_VOCAB_PRE_TYPE_BLOOM          = 23,
        JARVIS_VOCAB_PRE_TYPE_GPT3_FINNISH   = 24,
        JARVIS_VOCAB_PRE_TYPE_EXAONE         = 25,
        JARVIS_VOCAB_PRE_TYPE_CHAMELEON      = 26,
    };

    enum jarvis_rope_type {
        JARVIS_ROPE_TYPE_NONE = -1,
        JARVIS_ROPE_TYPE_NORM = 0,
        JARVIS_ROPE_TYPE_NEOX = GGML_ROPE_TYPE_NEOX,
    };

    enum jarvis_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        JARVIS_TOKEN_TYPE_UNDEFINED    = 0,
        JARVIS_TOKEN_TYPE_NORMAL       = 1,
        JARVIS_TOKEN_TYPE_UNKNOWN      = 2,
        JARVIS_TOKEN_TYPE_CONTROL      = 3,
        JARVIS_TOKEN_TYPE_USER_DEFINED = 4,
        JARVIS_TOKEN_TYPE_UNUSED       = 5,
        JARVIS_TOKEN_TYPE_BYTE         = 6,
    };

    enum jarvis_token_attr {
        JARVIS_TOKEN_ATTR_UNDEFINED    = 0,
        JARVIS_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        JARVIS_TOKEN_ATTR_UNUSED       = 1 << 1,
        JARVIS_TOKEN_ATTR_NORMAL       = 1 << 2,
        JARVIS_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        JARVIS_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        JARVIS_TOKEN_ATTR_BYTE         = 1 << 5,
        JARVIS_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        JARVIS_TOKEN_ATTR_LSTRIP       = 1 << 7,
        JARVIS_TOKEN_ATTR_RSTRIP       = 1 << 8,
        JARVIS_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum jarvis_ftype {
        JARVIS_FTYPE_ALL_F32              = 0,
        JARVIS_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // JARVIS_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // JARVIS_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // JARVIS_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        JARVIS_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q4_0_4_4      = 33, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q4_0_4_8      = 34, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_Q4_0_8_8      = 35, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        JARVIS_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors

        JARVIS_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum jarvis_rope_scaling_type {
        JARVIS_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        JARVIS_ROPE_SCALING_TYPE_NONE        = 0,
        JARVIS_ROPE_SCALING_TYPE_LINEAR      = 1,
        JARVIS_ROPE_SCALING_TYPE_YARN        = 2,
        JARVIS_ROPE_SCALING_TYPE_MAX_VALUE   = JARVIS_ROPE_SCALING_TYPE_YARN,
    };

    enum jarvis_pooling_type {
        JARVIS_POOLING_TYPE_UNSPECIFIED = -1,
        JARVIS_POOLING_TYPE_NONE = 0,
        JARVIS_POOLING_TYPE_MEAN = 1,
        JARVIS_POOLING_TYPE_CLS  = 2,
        JARVIS_POOLING_TYPE_LAST = 3,
        JARVIS_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum jarvis_attention_type {
        JARVIS_ATTENTION_TYPE_UNSPECIFIED = -1,
        JARVIS_ATTENTION_TYPE_CAUSAL      = 0,
        JARVIS_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum jarvis_split_mode {
        JARVIS_SPLIT_MODE_NONE  = 0, // single GPU
        JARVIS_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
        JARVIS_SPLIT_MODE_ROW   = 2, // split rows across GPUs
    };

    // TODO: simplify (https://github.com/ggerganov/jarvis.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct jarvis_token_data {
        jarvis_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } jarvis_token_data;

    typedef struct jarvis_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        jarvis_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;
    } jarvis_token_data_array;

    typedef bool (*jarvis_progress_callback)(float progress, void * user_data);

    // Input data for jarvis_decode
    // A jarvis_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by jarvis_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL, only the logits for last token will be returned)
    //
    typedef struct jarvis_batch {
        int32_t n_tokens;

        jarvis_token  *  token;
        float        *  embd;
        jarvis_pos    *  pos;
        int32_t      *  n_seq_id;
        jarvis_seq_id ** seq_id;
        int8_t       *  logits; // TODO: rename this to "output"
    } jarvis_batch;

    enum jarvis_model_kv_override_type {
        JARVIS_KV_OVERRIDE_TYPE_INT,
        JARVIS_KV_OVERRIDE_TYPE_FLOAT,
        JARVIS_KV_OVERRIDE_TYPE_BOOL,
        JARVIS_KV_OVERRIDE_TYPE_STR,
    };

    struct jarvis_model_kv_override {
        enum jarvis_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct jarvis_model_params {
        int32_t n_gpu_layers; // number of layers to store in VRAM
        enum jarvis_split_mode split_mode; // how to split the model across multiple GPUs

        // main_gpu interpretation depends on split_mode:
        // JARVIS_SPLIT_MODE_NONE: the GPU that is used for the entire model
        // JARVIS_SPLIT_MODE_ROW: the GPU that is used for small tensors and intermediate results
        // JARVIS_SPLIT_MODE_LAYER: ignored
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: jarvis_max_devices()
        const float * tensor_split;

        // comma separated list of RPC servers to use for offloading
        const char * rpc_servers;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        jarvis_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct jarvis_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;    // only load the vocabulary, no weights
        bool use_mmap;      // use mmap if possible
        bool use_mlock;     // force system to keep model in RAM
        bool check_tensors; // validate model tensor data
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggerganov/jarvis.cpp/pull/7544
    struct jarvis_context_params {
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to jarvis_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        int32_t  n_threads;         // number of threads to use for generation
        int32_t  n_threads_batch;   // number of threads to use for batch processing

        enum jarvis_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum jarvis_rope_scaling_type`
        enum jarvis_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum jarvis_attention_type    attention_type;    // attention type to use for embeddings

        // ref: https://github.com/ggerganov/jarvis.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // defragment the KV cache if holes/size > thold, < 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        // TODO: move at the end of the struct
        bool logits_all;  // the jarvis_decode() call computes all logits, not just the last one (DEPRECATED - set jarvis_batch.logits instead)
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
        bool flash_attn;  // whether to use flash attention [EXPERIMENTAL]
        bool no_perf;     // whether to measure performance timings

        // Abort callback
        // if it returns true, execution of jarvis_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;
    };

    // model quantization parameters
    typedef struct jarvis_model_quantize_params {
        int32_t nthread;                     // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum jarvis_ftype ftype;              // quantize to this jarvis_ftype
        enum ggml_type output_tensor_type;   // output tensor type
        enum ggml_type token_embedding_type; // token embeddings tensor type
        bool allow_requantize;               // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;         // quantize output.weight
        bool only_copy;                      // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                           // quantize all tensors to the default type
        bool keep_split;                     // quantize to the same number of shards
        void * imatrix;                      // pointer to importance matrix data
        void * kv_overrides;                 // pointer to vector containing overrides
    } jarvis_model_quantize_params;

    typedef struct jarvis_logit_bias {
        jarvis_token token;
        float bias;
    } jarvis_logit_bias;

    typedef struct jarvis_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } jarvis_sampler_chain_params;

    // used in chat template
    typedef struct jarvis_chat_message {
        const char * role;
        const char * content;
    } jarvis_chat_message;

    // lora adapter
    struct jarvis_lora_adapter;

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggerganov/jarvis.cpp/discussions/9172)
    JARVIS_API struct jarvis_model_params          jarvis_model_default_params(void);
    JARVIS_API struct jarvis_context_params        jarvis_context_default_params(void);
    JARVIS_API struct jarvis_sampler_chain_params  jarvis_sampler_chain_default_params(void);
    JARVIS_API struct jarvis_model_quantize_params jarvis_model_quantize_default_params(void);

    // Initialize the jarvis + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    JARVIS_API void jarvis_backend_init(void);

    //optional:
    JARVIS_API void jarvis_numa_init(enum ggml_numa_strategy numa);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    JARVIS_API void jarvis_attach_threadpool(
               struct   jarvis_context * ctx,
            ggml_threadpool_t   threadpool,
            ggml_threadpool_t   threadpool_batch);
    JARVIS_API void jarvis_detach_threadpool(struct jarvis_context * ctx);

    // Call once at the end of the program - currently only used for MPI
    JARVIS_API void jarvis_backend_free(void);

    JARVIS_API struct jarvis_model * jarvis_load_model_from_file(
                             const char * path_model,
              struct jarvis_model_params   params);

    JARVIS_API void jarvis_free_model(struct jarvis_model * model);

    // TODO: rename to jarvis_init_from_model
    JARVIS_API struct jarvis_context * jarvis_new_context_with_model(
                     struct jarvis_model * model,
            struct jarvis_context_params   params);

    // Frees all allocated memory
    JARVIS_API void jarvis_free(struct jarvis_context * ctx);

    JARVIS_API int64_t jarvis_time_us(void);

    JARVIS_API size_t jarvis_max_devices(void);

    JARVIS_API bool jarvis_supports_mmap       (void);
    JARVIS_API bool jarvis_supports_mlock      (void);
    JARVIS_API bool jarvis_supports_gpu_offload(void);
    JARVIS_API bool jarvis_supports_rpc        (void);

    JARVIS_API uint32_t jarvis_n_ctx      (const struct jarvis_context * ctx);
    JARVIS_API uint32_t jarvis_n_batch    (const struct jarvis_context * ctx);
    JARVIS_API uint32_t jarvis_n_ubatch   (const struct jarvis_context * ctx);
    JARVIS_API uint32_t jarvis_n_seq_max  (const struct jarvis_context * ctx);

    JARVIS_API int32_t jarvis_n_vocab    (const struct jarvis_model * model);
    JARVIS_API int32_t jarvis_n_ctx_train(const struct jarvis_model * model);
    JARVIS_API int32_t jarvis_n_embd     (const struct jarvis_model * model);
    JARVIS_API int32_t jarvis_n_layer    (const struct jarvis_model * model);
    JARVIS_API int32_t jarvis_n_head     (const struct jarvis_model * model);

    JARVIS_API const struct jarvis_model * jarvis_get_model(const struct jarvis_context * ctx);

    JARVIS_API enum jarvis_pooling_type jarvis_pooling_type(const struct jarvis_context * ctx);
    JARVIS_API enum jarvis_vocab_type   jarvis_vocab_type  (const struct jarvis_model * model);
    JARVIS_API enum jarvis_rope_type    jarvis_rope_type   (const struct jarvis_model * model);

    // Get the model's RoPE frequency scaling factor
    JARVIS_API float jarvis_rope_freq_scale_train(const struct jarvis_model * model);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    JARVIS_API int32_t jarvis_model_meta_val_str(const struct jarvis_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    JARVIS_API int32_t jarvis_model_meta_count(const struct jarvis_model * model);

    // Get metadata key name by index
    JARVIS_API int32_t jarvis_model_meta_key_by_index(const struct jarvis_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    JARVIS_API int32_t jarvis_model_meta_val_str_by_index(const struct jarvis_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    JARVIS_API int32_t jarvis_model_desc(const struct jarvis_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    JARVIS_API uint64_t jarvis_model_size(const struct jarvis_model * model);

    // Returns the total number of parameters in the model
    JARVIS_API uint64_t jarvis_model_n_params(const struct jarvis_model * model);

    // Get a jarvis model tensor
    JARVIS_API struct ggml_tensor * jarvis_get_model_tensor(struct jarvis_model * model, const char * name);

    // Returns true if the model contains an encoder that requires jarvis_encode() call
    JARVIS_API bool jarvis_model_has_encoder(const struct jarvis_model * model);

    // Returns true if the model contains a decoder that requires jarvis_decode() call
    JARVIS_API bool jarvis_model_has_decoder(const struct jarvis_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    JARVIS_API jarvis_token jarvis_model_decoder_start_token(const struct jarvis_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    JARVIS_API bool jarvis_model_is_recurrent(const struct jarvis_model * model);

    // Returns 0 on success
    JARVIS_API uint32_t jarvis_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const jarvis_model_quantize_params * params);

    // Load a LoRA adapter from file
    // The loaded adapter will be associated to the given model, and will be free when the model is deleted
    JARVIS_API struct jarvis_lora_adapter * jarvis_lora_adapter_init(
            struct jarvis_model * model,
            const char * path_lora);

    // Add a loaded LoRA adapter to given context
    // This will not modify model's weight
    JARVIS_API int32_t jarvis_lora_adapter_set(
            struct jarvis_context * ctx,
            struct jarvis_lora_adapter * adapter,
            float scale);

    // Remove a specific LoRA adapter from given context
    // Return -1 if the adapter is not present in the context
    JARVIS_API int32_t jarvis_lora_adapter_remove(
            struct jarvis_context * ctx,
            struct jarvis_lora_adapter * adapter);

    // Remove all LoRA adapters from given context
    JARVIS_API void jarvis_lora_adapter_clear(
            struct jarvis_context * ctx);

    // Manually free a LoRA adapter
    // Note: loaded adapters will be free when the associated model is deleted
    JARVIS_API void jarvis_lora_adapter_free(struct jarvis_lora_adapter * adapter);

    // Apply a loaded control vector to a jarvis_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See jarvis_control_vector_load in common to load a control vector.
    JARVIS_API int32_t jarvis_control_vector_apply(
            struct jarvis_context * lctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // KV cache
    //

    // Information associated with an individual cell in the KV cache view.
    struct jarvis_kv_cache_view_cell {
        // The position for this cell. Takes KV cache shifts into account.
        // May be negative if the cell is not populated.
        jarvis_pos pos;
    };

    // An updateable view of the KV cache.
    struct jarvis_kv_cache_view {
        // Number of KV cache cells. This will be the same as the context size.
        int32_t n_cells;

        // Maximum number of sequences that can exist in a cell. It's not an error
        // if there are more sequences in a cell than this value, however they will
        // not be visible in the view cells_sequences.
        int32_t n_seq_max;

        // Number of tokens in the cache. For example, if there are two populated
        // cells, the first with 1 sequence id in it and the second with 2 sequence
        // ids then you'll have 3 tokens.
        int32_t token_count;

        // Number of populated cache cells.
        int32_t used_cells;

        // Maximum contiguous empty slots in the cache.
        int32_t max_contiguous;

        // Index to the start of the max_contiguous slot range. Can be negative
        // when cache is full.
        int32_t max_contiguous_idx;

        // Information for an individual cell.
        struct jarvis_kv_cache_view_cell * cells;

        // The sequences for each cell. There will be n_seq_max items per cell.
        jarvis_seq_id * cells_sequences;
    };

    // Create an empty KV cache view. (use only for debugging purposes)
    JARVIS_API struct jarvis_kv_cache_view jarvis_kv_cache_view_init(const struct jarvis_context * ctx, int32_t n_seq_max);

    // Free a KV cache view. (use only for debugging purposes)
    JARVIS_API void jarvis_kv_cache_view_free(struct jarvis_kv_cache_view * view);

    // Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
    JARVIS_API void jarvis_kv_cache_view_update(const struct jarvis_context * ctx, struct jarvis_kv_cache_view * view);

    // Returns the number of tokens in the KV cache (slow, use only for debug)
    // If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    JARVIS_API int32_t jarvis_get_kv_cache_token_count(const struct jarvis_context * ctx);

    // Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
    JARVIS_API int32_t jarvis_get_kv_cache_used_cells(const struct jarvis_context * ctx);

    // Clear the KV cache - both cell info is erased and KV data is zeroed
    JARVIS_API void jarvis_kv_cache_clear(
            struct jarvis_context * ctx);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    JARVIS_API bool jarvis_kv_cache_seq_rm(
            struct jarvis_context * ctx,
                    jarvis_seq_id   seq_id,
                       jarvis_pos   p0,
                       jarvis_pos   p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    JARVIS_API void jarvis_kv_cache_seq_cp(
            struct jarvis_context * ctx,
                    jarvis_seq_id   seq_id_src,
                    jarvis_seq_id   seq_id_dst,
                       jarvis_pos   p0,
                       jarvis_pos   p1);

    // Removes all tokens that do not belong to the specified sequence
    JARVIS_API void jarvis_kv_cache_seq_keep(
            struct jarvis_context * ctx,
                    jarvis_seq_id   seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next jarvis_decode()
    //   - explicitly with jarvis_kv_cache_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    JARVIS_API void jarvis_kv_cache_seq_add(
            struct jarvis_context * ctx,
                    jarvis_seq_id   seq_id,
                       jarvis_pos   p0,
                       jarvis_pos   p1,
                       jarvis_pos   delta);

    // Integer division of the positions by factor of `d > 1`
    // If the KV cache is RoPEd, the KV data is updated accordingly:
    //   - lazily on next jarvis_decode()
    //   - explicitly with jarvis_kv_cache_update()
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    JARVIS_API void jarvis_kv_cache_seq_div(
            struct jarvis_context * ctx,
                    jarvis_seq_id   seq_id,
                       jarvis_pos   p0,
                       jarvis_pos   p1,
                             int   d);

    // Returns the largest position present in the KV cache for the specified sequence
    JARVIS_API jarvis_pos jarvis_kv_cache_seq_pos_max(
            struct jarvis_context * ctx,
                    jarvis_seq_id   seq_id);

    // Defragment the KV cache
    // This will be applied:
    //   - lazily on next jarvis_decode()
    //   - explicitly with jarvis_kv_cache_update()
    JARVIS_API void jarvis_kv_cache_defrag(struct jarvis_context * ctx);

    // Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
    JARVIS_API void jarvis_kv_cache_update(struct jarvis_context * ctx);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and kv_cache)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    JARVIS_API size_t jarvis_state_get_size(struct jarvis_context * ctx);
    JARVIS_API DEPRECATED(size_t jarvis_get_state_size(struct jarvis_context * ctx),
        "use jarvis_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    JARVIS_API size_t jarvis_state_get_data(
            struct jarvis_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    JARVIS_API DEPRECATED(size_t jarvis_copy_state_data(
            struct jarvis_context * ctx,
                         uint8_t * dst),
        "use jarvis_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    JARVIS_API size_t jarvis_state_set_data(
            struct jarvis_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    JARVIS_API DEPRECATED(size_t jarvis_set_state_data(
            struct jarvis_context * ctx,
                   const uint8_t * src),
        "use jarvis_state_set_data instead");

    // Save/load session file
    JARVIS_API bool jarvis_state_load_file(
            struct jarvis_context * ctx,
                      const char * path_session,
                     jarvis_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    JARVIS_API DEPRECATED(bool jarvis_load_session_file(
            struct jarvis_context * ctx,
                      const char * path_session,
                     jarvis_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use jarvis_state_load_file instead");

    JARVIS_API bool jarvis_state_save_file(
            struct jarvis_context * ctx,
                      const char * path_session,
               const jarvis_token * tokens,
                          size_t   n_token_count);
    JARVIS_API DEPRECATED(bool jarvis_save_session_file(
            struct jarvis_context * ctx,
                      const char * path_session,
               const jarvis_token * tokens,
                          size_t   n_token_count),
        "use jarvis_state_save_file instead");

    // Get the exact size needed to copy the KV cache of a single sequence
    JARVIS_API size_t jarvis_state_seq_get_size(
            struct jarvis_context * ctx,
                    jarvis_seq_id   seq_id);

    // Copy the KV cache of a single sequence into the specified buffer
    JARVIS_API size_t jarvis_state_seq_get_data(
            struct jarvis_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    jarvis_seq_id   seq_id);

    // Copy the sequence data (originally copied with `jarvis_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    JARVIS_API size_t jarvis_state_seq_set_data(
            struct jarvis_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    jarvis_seq_id   dest_seq_id);

    JARVIS_API size_t jarvis_state_seq_save_file(
            struct jarvis_context * ctx,
                      const char * filepath,
                    jarvis_seq_id   seq_id,
               const jarvis_token * tokens,
                          size_t   n_token_count);

    JARVIS_API size_t jarvis_state_seq_load_file(
            struct jarvis_context * ctx,
                      const char * filepath,
                    jarvis_seq_id   dest_seq_id,
                     jarvis_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by jarvis_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    JARVIS_API struct jarvis_batch jarvis_batch_get_one(
                  jarvis_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with jarvis_batch_free()
    // If embd != 0, jarvis_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, jarvis_batch.token will be allocated to store n_tokens jarvis_token
    // The rest of the jarvis_batch members are allocated with size n_tokens
    // All members are left uninitialized
    JARVIS_API struct jarvis_batch jarvis_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with jarvis_batch_init()
    JARVIS_API void jarvis_batch_free(struct jarvis_batch batch);

    // Processes a batch of tokens with the ecoder part of the encoder-decoder model.
    // Stores the encoder output internally for later use by the decoder cross-attention layers.
    //   0 - success
    // < 0 - error
    JARVIS_API int32_t jarvis_encode(
            struct jarvis_context * ctx,
              struct jarvis_batch   batch);

    // Positive return values does not mean a fatal error, but rather a warning.
    //   0 - success
    //   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    // < 0 - error
    JARVIS_API int32_t jarvis_decode(
            struct jarvis_context * ctx,
              struct jarvis_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    JARVIS_API void jarvis_set_n_threads(struct jarvis_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    JARVIS_API int32_t jarvis_n_threads(struct jarvis_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    JARVIS_API int32_t jarvis_n_threads_batch(struct jarvis_context * ctx);

    // Set whether the model is in embeddings mode or not
    // If true, embeddings will be returned but logits will not
    JARVIS_API void jarvis_set_embeddings(struct jarvis_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    JARVIS_API void jarvis_set_causal_attn(struct jarvis_context * ctx, bool causal_attn);

    // Set abort callback
    JARVIS_API void jarvis_set_abort_callback(struct jarvis_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    JARVIS_API void jarvis_synchronize(struct jarvis_context * ctx);

    // Token logits obtained from the last call to jarvis_decode()
    // The logits for which jarvis_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which jarvis_batch.logits[i] != 0
    // Cols: n_vocab
    JARVIS_API float * jarvis_get_logits(struct jarvis_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // jarvis_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    JARVIS_API float * jarvis_get_logits_ith(struct jarvis_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == JARVIS_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which jarvis_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    JARVIS_API float * jarvis_get_embeddings(struct jarvis_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // jarvis_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    JARVIS_API float * jarvis_get_embeddings_ith(struct jarvis_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is JARVIS_POOLING_TYPE_NONE
    // when pooling_type == JARVIS_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    JARVIS_API float * jarvis_get_embeddings_seq(struct jarvis_context * ctx, jarvis_seq_id seq_id);

    //
    // Vocab
    //

    JARVIS_API const char * jarvis_token_get_text(const struct jarvis_model * model, jarvis_token token);

    JARVIS_API float jarvis_token_get_score(const struct jarvis_model * model, jarvis_token token);

    JARVIS_API enum jarvis_token_attr jarvis_token_get_attr(const struct jarvis_model * model, jarvis_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    JARVIS_API bool jarvis_token_is_eog(const struct jarvis_model * model, jarvis_token token);

    // Identify if Token Id is a control token or a render-able token
    JARVIS_API bool jarvis_token_is_control(const struct jarvis_model * model, jarvis_token token);

    // Special tokens
    JARVIS_API jarvis_token jarvis_token_bos(const struct jarvis_model * model); // beginning-of-sentence
    JARVIS_API jarvis_token jarvis_token_eos(const struct jarvis_model * model); // end-of-sentence
    JARVIS_API jarvis_token jarvis_token_eot(const struct jarvis_model * model); // end-of-turn
    JARVIS_API jarvis_token jarvis_token_cls(const struct jarvis_model * model); // classification
    JARVIS_API jarvis_token jarvis_token_sep(const struct jarvis_model * model); // sentence separator
    JARVIS_API jarvis_token jarvis_token_nl (const struct jarvis_model * model); // next-line
    JARVIS_API jarvis_token jarvis_token_pad(const struct jarvis_model * model); // padding

    JARVIS_API bool jarvis_add_bos_token(const struct jarvis_model * model);
    JARVIS_API bool jarvis_add_eos_token(const struct jarvis_model * model);

    // infill tokens
    DEPRECATED(JARVIS_API jarvis_token jarvis_token_prefix(const struct jarvis_model * model), "use jarvis_token_fim_pre instead");
    DEPRECATED(JARVIS_API jarvis_token jarvis_token_middle(const struct jarvis_model * model), "use jarvis_token_fim_mid instead");
    DEPRECATED(JARVIS_API jarvis_token jarvis_token_suffix(const struct jarvis_model * model), "use jarvis_token_fim_suf instead");

    JARVIS_API jarvis_token jarvis_token_fim_pre(const struct jarvis_model * model);
    JARVIS_API jarvis_token jarvis_token_fim_suf(const struct jarvis_model * model);
    JARVIS_API jarvis_token jarvis_token_fim_mid(const struct jarvis_model * model);
    JARVIS_API jarvis_token jarvis_token_fim_pad(const struct jarvis_model * model);
    JARVIS_API jarvis_token jarvis_token_fim_rep(const struct jarvis_model * model);
    JARVIS_API jarvis_token jarvis_token_fim_sep(const struct jarvis_model * model);

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    JARVIS_API int32_t jarvis_tokenize(
        const struct jarvis_model * model,
                      const char * text,
                         int32_t   text_len,
                     jarvis_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    JARVIS_API int32_t jarvis_token_to_piece(
              const struct jarvis_model * model,
                           jarvis_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of jarvis_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    JARVIS_API int32_t jarvis_detokenize(
        const struct jarvis_model * model,
               const jarvis_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    /// Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/jarvis.cpp/wiki/Templates-supported-by-jarvis_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
    /// @param chat Pointer to a list of multiple jarvis_chat_message
    /// @param n_msg Number of jarvis_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    JARVIS_API int32_t jarvis_chat_apply_template(
              const struct jarvis_model * model,
                            const char * tmpl,
       const struct jarvis_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = jarvis_sampler_chain_default_params();
    //
    //    jarvis_sampler * smpl = jarvis_sampler_chain_init(sparams);
    //
    //    jarvis_sampler_chain_add(smpl, jarvis_sampler_init_top_k(50));
    //    jarvis_sampler_chain_add(smpl, jarvis_sampler_init_top_p(0.9, 1));
    //    jarvis_sampler_chain_add(smpl, jarvis_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    jarvis_sampler_chain_add(smpl, jarvis_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        jarvis_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const jarvis_token id = jarvis_sampler_sample(smpl, ctx, -1);
    //
    //        // accepting the token updates the internal state of certain samplers (e.g. grammar, repetition, etc.)
    //        jarvis_sampler_accept(smpl, id);
    //        ...
    //    }
    //
    //    jarvis_sampler_free(smpl);
    //
    // TODO: In the future, jarvis_sampler will be utilized to offload the sampling to the backends (e.g. GPU).
    // TODO: in the future, the entire sampling API that uses jarvis_model should start using jarvis_vocab
    //

    typedef void * jarvis_sampler_context_t;

    // user code can implement the interface below in order to create custom jarvis_sampler
    struct jarvis_sampler_i {
        const char *           (*name)  (const struct jarvis_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct jarvis_sampler * smpl, jarvis_token token);              // can be NULL
        void                   (*apply) (      struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p); // required
        void                   (*reset) (      struct jarvis_sampler * smpl);                                 // can be NULL
        struct jarvis_sampler * (*clone) (const struct jarvis_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct jarvis_sampler * smpl);                                 // can be NULL if ctx is NULL

        // TODO: API for internal libjarvis usage for appending the sampling to an existing ggml_cgraph
        //void (*apply_ggml) (struct jarvis_sampler * smpl, ...);
    };

    struct jarvis_sampler {
        struct jarvis_sampler_i  * iface;
        jarvis_sampler_context_t   ctx;
    };

    // mirror of jarvis_sampler_i:
    JARVIS_API const char *           jarvis_sampler_name  (const struct jarvis_sampler * smpl);
    JARVIS_API void                   jarvis_sampler_accept(      struct jarvis_sampler * smpl, jarvis_token token);
    JARVIS_API void                   jarvis_sampler_apply (      struct jarvis_sampler * smpl, jarvis_token_data_array * cur_p);
    JARVIS_API void                   jarvis_sampler_reset (      struct jarvis_sampler * smpl);
    JARVIS_API struct jarvis_sampler * jarvis_sampler_clone (const struct jarvis_sampler * smpl);
    // important: do not free if the sampler has been added to a jarvis_sampler_chain (via jarvis_sampler_chain_add)
    JARVIS_API void                   jarvis_sampler_free  (      struct jarvis_sampler * smpl);

    // jarvis_sampler_chain
    // a type of jarvis_sampler that can chain multiple samplers one after another

    JARVIS_API struct jarvis_sampler * jarvis_sampler_chain_init(struct jarvis_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when jarvis_sampler_free is called
    JARVIS_API void                   jarvis_sampler_chain_add(      struct jarvis_sampler * chain, struct jarvis_sampler * smpl);
    JARVIS_API struct jarvis_sampler * jarvis_sampler_chain_get(const struct jarvis_sampler * chain, int32_t i);
    JARVIS_API int                    jarvis_sampler_chain_n  (const struct jarvis_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    JARVIS_API struct jarvis_sampler * jarvis_sampler_chain_remove(   struct jarvis_sampler * chain, int32_t i);

    // available samplers:

    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_greedy(void);
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_dist  (uint32_t seed);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    /// NOTE: Avoid using on the full vocabulary as the sorting can become slow. For example, apply top-k or top-p sampling first.
    DEPRECATED(JARVIS_API struct jarvis_sampler * jarvis_sampler_init_softmax    (void),
        "will be removed in the future (see https://github.com/ggerganov/jarvis.cpp/pull/9896#discussion_r1800920915)");

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggerganov/jarvis.cpp/pull/3841
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_tail_free  (float   z, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `jarvis_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `jarvis_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_grammar(
            const struct jarvis_model * model,
                          const char * grammar_str,
                          const char * grammar_root);

    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_penalties(
                             int32_t   n_vocab,         // jarvis_n_vocab()
                         jarvis_token   special_eos_id,  // jarvis_token_eos()
                         jarvis_token   linefeed_id,     // jarvis_token_nl()
                             int32_t   penalty_last_n,  // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,  // 1.0 = disabled
                               float   penalty_freq,    // 0.0 = disabled
                               float   penalty_present, // 0.0 = disabled
                                bool   penalize_nl,     // consider newlines as a repeatable token
                                bool   ignore_eos);     // ignore the end-of-sequence token

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    JARVIS_API struct jarvis_sampler *    jarvis_sampler_init_dry(
            const struct jarvis_model *  model,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const jarvis_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    JARVIS_API struct jarvis_sampler * jarvis_sampler_init_infill(const struct jarvis_model * model);

    // Returns the seed used by the sampler if applicable, JARVIS_DEFAULT_SEED otherwise
    JARVIS_API uint32_t jarvis_sampler_get_seed(const struct jarvis_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = jarvis_get_logits_ith(ctx, idx);
    //    jarvis_token_data_array cur_p = { ... init from logits ... };
    //    jarvis_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    jarvis_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    JARVIS_API jarvis_token jarvis_sampler_sample(struct jarvis_sampler * smpl, struct jarvis_context * ctx, int32_t idx);

    // TODO: extend in the future
    //JARVIS_API void jarvis_decode_with_sampler(struct jarvis_context * ctx, struct jarvis_sampler * smpl, struct jarvis_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          jarvis_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    JARVIS_API int jarvis_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          jarvis_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    JARVIS_API int jarvis_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count);

    // Print system information
    JARVIS_API const char * jarvis_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    JARVIS_API void jarvis_log_set(ggml_log_callback log_callback, void * user_data);

    //
    // Performance utils
    //
    // NOTE: Used by jarvis.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct jarvis_perf_context_data {
        double t_start_ms;
        double t_load_ms;
        double t_p_eval_ms;
        double t_eval_ms;

        int32_t n_p_eval;
        int32_t n_eval;
    };

    struct jarvis_perf_sampler_data {
        double t_sample_ms;

        int32_t n_sample;
    };

    JARVIS_API struct jarvis_perf_context_data jarvis_perf_context      (const struct jarvis_context * ctx);
    JARVIS_API void                           jarvis_perf_context_print(const struct jarvis_context * ctx);
    JARVIS_API void                           jarvis_perf_context_reset(      struct jarvis_context * ctx);

    // NOTE: the following work only with samplers constructed via jarvis_sampler_chain_init
    JARVIS_API struct jarvis_perf_sampler_data jarvis_perf_sampler      (const struct jarvis_sampler * chain);
    JARVIS_API void                           jarvis_perf_sampler_print(const struct jarvis_sampler * chain);
    JARVIS_API void                           jarvis_perf_sampler_reset(      struct jarvis_sampler * chain);

    JARVIS_API void jarvis_perf_dump_yaml(FILE * stream, const struct jarvis_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // JARVIS_H
