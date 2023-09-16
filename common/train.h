// Various helper functions and utilities for training

#pragma once

#include <string>
#include <random>
#include <vector>

#include "ggml.h"
#include "llama.h"

typedef std::string mt19937_state;

struct train_state {
    struct ggml_opt_context * opt;

    uint64_t train_its;
    uint64_t train_samples;
    uint64_t train_tokens;
    uint64_t train_epochs;

    size_t        shuffle_samples_hash; // fn, sample_count, *zip(sample_begins, sample_sizes)
    mt19937_state shuffle_rng_state_current;
    mt19937_state shuffle_rng_state_next;
    size_t        shuffle_sample_count;
    size_t        shuffle_next_sample;
};

struct train_state * init_train_state(int seed);
void free_train_state(struct train_state  * state);

struct random_normal_distribution;
struct random_uniform_distribution;

struct random_normal_distribution  * init_random_normal_distribution (int seed, float mean, float std, float min, float max);
struct random_uniform_distribution * init_random_uniform_distribution(int seed, float min, float max);

void free_random_normal_distribution (struct random_normal_distribution  * rnd);
void free_random_uniform_distribution(struct random_uniform_distribution * rnd);

struct ggml_tensor * randomize_tensor_normal (struct ggml_tensor * tensor, struct random_normal_distribution * rnd);
struct ggml_tensor * randomize_tensor_uniform(struct ggml_tensor * tensor, struct random_uniform_distribution * rnd);

float frand();
float frand_normal (struct random_normal_distribution * rnd);
float frand_uniform(struct random_uniform_distribution * rnd);

int   clamp (const int v, const int min, const int max);
float fclamp(const float v, const float min, const float max);

void assert_shape_1d(struct ggml_tensor * tensor, int64_t ne0);
void assert_shape_2d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1);
void assert_shape_3d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2);
void assert_shape_4d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

size_t tokenize_file(
        struct llama_context     * lctx,
        const char               * filename,
        const std::string        & sample_start,
        bool                       include_sample_start,
        bool                       overlapping_samples,
        unsigned                   context_length,
        std::vector<llama_token> & out_tokens,
        std::vector<size_t>      & out_samples_begin,
        std::vector<size_t>      & out_samples_size);

int64_t get_example_targets_batch(
        struct llama_context * lctx,
        struct ggml_tensor   * tokens_input,
        struct ggml_tensor   * target_probs,
        int64_t                example_id,
        const size_t         * samples_begin,
        const size_t         * samples_size,
              size_t           samples_count,
        const llama_token    * train_data,
        size_t                 n_train_data,
        bool                   separate_with_eos,
        bool                   separate_with_bos,
        bool                   fill_with_next_samples);


void          mt19937_set_state(std::mt19937& rng, const mt19937_state& rng_state);
mt19937_state mt19937_get_state(const std::mt19937& rng);
mt19937_state mt19937_seed_to_state(unsigned seed);

mt19937_state shuffle_samples(
        const mt19937_state & rng_state,
        size_t              * shuffled_begins,
        size_t              * shuffled_sizes,
        const size_t        * begins,
        const size_t        * sizes,
        size_t                count);

size_t hash_combine(size_t h1, size_t h2);

size_t compute_samples_hash(
    const char* fn,
    const size_t* samples_begin,
    const size_t* samples_size,
    size_t sample_count);


std::string replace_str(const char * s, const char * needle, const char * replacement);

void print_duration(double milliseconds);

float cosine_decay(
    int64_t step,
    int64_t decay_steps,
    float   minimum);

float cosine_decay_restart(
    int64_t step,
    int64_t decay_steps,
    float   minimum,
    float   restart_step_mult);

float learning_schedule(
    int64_t step,
    int64_t warmup_steps,
    int64_t decay_steps,
    float   learning_rate,
    float   overall_minimum,
    float   cos_decay_minimum,
    float   cos_decay_restart_step_mult,
    bool    enable_restart);

void copy_tensor_by_name(struct ggml_tensor * dst, struct ggml_context * ctx, const char * name);

void load_opt_context_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct ggml_opt_context * opt);
void save_opt_context_gguf(struct gguf_context * fctx, struct ggml_opt_context * opt);

bool load_train_state_gguf(struct gguf_context * fctx, struct ggml_context * f_ggml_ctx, struct train_state * train);
void save_train_state_gguf(struct gguf_context * fctx, struct train_state * train);

std::string get_train_filename(const char * filename, const char * pattern_it, const char * latest, int64_t iteration);

typedef void (*save_train_files_callback)(void * data, struct train_state * train);
