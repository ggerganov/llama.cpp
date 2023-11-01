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

struct train_params_common {
    const char * fn_train_data;
    const char * fn_checkpoint_in;
    const char * fn_checkpoint_out;
    const char * pattern_fn_it;
    const char * fn_latest;

    bool print_usage;

    int save_every;

    uint32_t seed;

    int n_ctx;
    int n_threads;
    int n_batch;
    int n_gradient_accumulation;
    int n_epochs;
    int n_gpu_layers;

    bool custom_n_ctx;

    bool use_flash;
    bool use_checkpointing;

    std::string sample_start;
    bool include_sample_start;
    bool escape;
    bool overlapping_samples;
    bool fill_with_next_samples;
    bool separate_with_eos;
    bool separate_with_bos;
    bool sample_random_offsets;

    bool force_reshuffle;

    int   warmup;
    int   cos_decay_steps;
    float cos_decay_restart;
    float cos_decay_min;
    bool  enable_restart;

    int   opt_past;
    float opt_delta;
    int   opt_max_no_improvement;

    int   adam_n_iter;
    float adam_alpha;
    float adam_min_alpha;
    float adam_decay;
    int   adam_decay_min_ndim;
    float adam_beta1;
    float adam_beta2;
    float adam_gclip;
    float adam_eps_f;
};

typedef void (*save_train_files_callback)(void * data, struct train_state * train);

struct train_opt_callback_data {
    struct train_params_common * params;
    struct train_state         * train;
    save_train_files_callback    save_cb;
    void                       * save_data;
    struct llama_context       * lctx;
    int                          last_save_iter;
    llama_token                * tokens_data;
    size_t                       tokens_size;
    size_t                     * samples_begin;
    size_t                     * samples_size;
    size_t                     * shuffled_samples_offs;
    size_t                     * shuffled_samples_begin;
    size_t                     * shuffled_samples_size;
    size_t                       samples_count;
    struct ggml_tensor         * tokens_input;
    struct ggml_tensor         * target_probs;
    int                          first_iter;
    int                          first_epoch;
    int                          iter_at_last_epoch;
    int64_t                      last_time;
    double                       millis_per_iter;
};

struct train_state * init_train_state();
void free_train_state(struct train_state  * state);

struct train_params_common get_default_train_params_common();
void print_common_train_usage(int /*argc*/, char ** argv, const struct train_params_common * params);

bool consume_common_train_arg(int argc, char ** argv, int * idx, struct train_params_common * params, bool * invalid_param);
void finish_processing_train_args(struct train_params_common * params);

struct random_normal_distribution;
struct random_uniform_distribution;

struct random_normal_distribution  * init_random_normal_distribution (int seed, float mean, float std, float min, float max);
struct random_uniform_distribution * init_random_uniform_distribution(int seed, float min, float max);

void free_random_normal_distribution (struct random_normal_distribution  * rnd);
void free_random_uniform_distribution(struct random_uniform_distribution * rnd);

struct ggml_tensor * randomize_tensor_normal (struct ggml_tensor * tensor, struct random_normal_distribution * rnd);
struct ggml_tensor * randomize_tensor_uniform(struct ggml_tensor * tensor, struct random_uniform_distribution * rnd);

// generate random float in interval [0,1)
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
        const size_t         * samples_offs,
        const size_t         * samples_begin,
        const size_t         * samples_size,
              size_t           samples_count,
        const llama_token    * train_data,
        size_t                 n_train_data,
        bool                   separate_with_eos,
        bool                   separate_with_bos,
        bool                   fill_with_next_samples,
        bool                   sample_random_offsets);


void          mt19937_set_state(std::mt19937& rng, const mt19937_state& rng_state);
mt19937_state mt19937_get_state(const std::mt19937& rng);
mt19937_state mt19937_seed_to_state(unsigned seed);

mt19937_state shuffle_samples(
        const mt19937_state & rng_state,
        size_t              * shuffled_offs,
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

void train_opt_callback(void * vdata, int accum_step, float * sched, bool * cancel);
