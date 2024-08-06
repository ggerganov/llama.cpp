#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <stdexcept>

// sampler types
enum class llama_sampler_type : char {
    TOP_K       = 'k',
    TOP_P       = 'p',
    MIN_P       = 'm',
    TFS_Z       = 'f',
    TYPICAL_P   = 'y',
    TEMPERATURE = 't'
};

// sampling parameters
typedef struct gpt_sampling_params {
    uint32_t    seed                  = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampling_context
    int32_t     n_prev                = 64;                 // number of previous tokens to remember
    int32_t     n_probs               = 0;                  // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t     min_keep              = 0;                  // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t     top_k                 = 40;                 // <= 0 to use vocab size
    float       top_p                 = 0.95f;              // 1.0 = disabled
    float       min_p                 = 0.05f;              // 0.0 = disabled
    float       tfs_z                 = 1.00f;              // 1.0 = disabled
    float       typical_p             = 1.00f;              // 1.0 = disabled
    float       temp                  = 0.80f;              // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float       dynatemp_range        = 0.00f;              // 0.0 = disabled
    float       dynatemp_exponent     = 1.00f;              // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t     penalty_last_n        = 64;                 // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float       penalty_repeat        = 1.00f;              // 1.0 = disabled
    float       penalty_freq          = 0.00f;              // 0.0 = disabled
    float       penalty_present       = 0.00f;              // 0.0 = disabled
    int32_t     mirostat              = 0;                  // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float       mirostat_tau          = 5.00f;              // target entropy
    float       mirostat_eta          = 0.10f;              // learning rate
    bool        penalize_nl           = false;              // consider newlines as a repeatable token
    bool        ignore_eos            = false;

    std::vector<llama_sampler_type> samplers_sequence = {
        llama_sampler_type::TOP_K,
        llama_sampler_type::TFS_Z,
        llama_sampler_type::TYPICAL_P,
        llama_sampler_type::TOP_P,
        llama_sampler_type::MIN_P,
        llama_sampler_type::TEMPERATURE
    };

    std::string grammar;  // optional BNF-like grammar to constrain sampling

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    std::string cfg_negative_prompt; // string to help guidance
    float       cfg_scale     = 1.f; // how strong is guidance

    std::vector<llama_logit_bias> logit_bias; // logit biases to apply
} gpt_sampling_params;

// the ring buffer works similarly to std::deque, but with a fixed capacity
template<typename T>
struct ring_buffer {
    ring_buffer() {}
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    T & operator[](size_t i) {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + i) % capacity];
    }

    const T & operator[](size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + i) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

// general sampler context
// TODO: move to llama.h
struct llama_sampling_context {
    // parameters that will be used for sampling
    gpt_sampling_params params;

    // mirostat sampler state
    float mirostat_mu;

    llama_sampling * smpl;

    ring_buffer<llama_token>      prev;
    std::vector<llama_token_data> cur;

    size_t n_valid; // Number of correct top tokens with correct probabilities.
};

// Create a new sampling context instance.
struct llama_sampling_context * llama_sampling_init(const struct gpt_sampling_params & params, const struct llama_model * model);

void llama_sampling_free(struct llama_sampling_context * ctx);

// Reset the sampler context
// - clear prev tokens
// - reset grammar
void llama_sampling_reset(llama_sampling_context * ctx);

// Copy the sampler context
void llama_sampling_cp(llama_sampling_context * src, llama_sampling_context * dst);

// Get the last sampled token
llama_token llama_sampling_last(llama_sampling_context * ctx);

// Get a string representation of the last sampled tokens
std::string llama_sampling_prev_str(llama_sampling_context * ctx_sampling, llama_context * ctx_main, int n);

// Print sampling parameters into a string
std::string llama_sampling_print(const gpt_sampling_params & params);

// Print sampling order into a string
std::string llama_sampling_order_print(const gpt_sampling_params & params);

std::string llama_sampling_type_to_str(llama_sampler_type sampler_type);

std::vector<llama_sampler_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names);
std::vector<llama_sampler_type> llama_sampling_types_from_chars(const std::string & names_string);

// this is a common sampling function used across the examples for convenience
// it can serve as a starting point for implementing your own sampling function
// Note: When using multiple sequences, it is the caller's responsibility to call
//       llama_sampling_reset when a sequence ends
//
// required:
//  - ctx_main:     context to use for sampling
//  - ctx_sampling: sampling-specific context
//
// optional:
//  - ctx_cfg:      context to use for classifier-free guidance
//  - idx:          sample from llama_get_logits_ith(ctx, idx)
//
// returns:
//  - token:      sampled token
//  - candidates: vector of candidate tokens
//
llama_token llama_sampling_sample(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        struct llama_context * ctx_cfg,
        int idx = -1);

// Prepares and adjusts the set of token candidates for sampling based on penalties, biases, and sampling parameters.
llama_token_data_array llama_sampling_prepare(
        struct llama_sampling_context * ctx_sampling,
        struct llama_context * ctx_main,
        struct llama_context * ctx_cfg,
        int idx = 0,
        bool apply_grammar = true,
        std::vector<float> * original_logits = nullptr);

void llama_sampling_accept(
        struct llama_sampling_context * ctx_sampling,
        llama_token id,
        bool apply_grammar);
