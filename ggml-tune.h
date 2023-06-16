#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_MULMAT_TUNE_VERSION 9
#define GGML_MULMAT_N_SHAPES 4
#define GGML_MULMAT_CACHE_LEN 16

#define GGML_MULMAT_MAX_PASS 3

struct ggml_mulmat_tune_m {
    int M;

    int stages_time[3];
};

struct ggml_mulmat_tune_model {
    const char *name;

    enum ggml_ftype ftype;

    int n_vocab;

    int n_embd;

    // n_ff = ((2*(4*n_embd)/3 + n_mult - 1)/n_mult)*n_mult
    int n_ff;

    // n_rot = n_embd/n_head;
    int n_rot;
};

struct ggml_mulmat_tune_shape {
    // For RoPE, one of N / K is 0.
    int N;
    int K;

    enum ggml_type src0_type;
    enum ggml_type src1_type;

    int n_profiles;
    struct ggml_task_profile profiles[GGML_MAX_TASK_PROFILES];

    int m_num;
    int *arr_m;

    struct ggml_mulmat_tune_m *items;
};

struct ggml_mulmat_tune_cache_ele {
    int M;
    int N;
    int K;
    const struct ggml_task_profile *profile;
    int stages_time[3];
};

struct ggml_mulmat_tune {
    int version;

    char model[16];

    enum ggml_ftype ftype;

    int n_shapes;
    // Given N/K, we bench for mul_mat [M,K] x [K,N].
    struct ggml_mulmat_tune_shape shapes[GGML_MULMAT_N_SHAPES];

    int n_threads;

    // Cache for time estimating.
    struct ggml_mulmat_tune_cache_ele cache[GGML_MULMAT_CACHE_LEN];
};

struct ggml_mulmat_tune_time {
    const struct ggml_task_profile *profile;
    int stage_time[3];
    int total_time;
};

// params for tune/bench.
struct ggml_mulmat_tune_params {
    struct ggml_mulmat_tune_model model;
    int m_num;
    int n_pass;
    int n_threads;
    bool progress;       // print and clear '.'
    bool output_console; // also print result to console
    const char *fname;
};

// NOTE: stages_time is filled if not null.
// Return profile id.
int ggml_mulmat_tune_select_task_profile(struct ggml_mulmat_tune *tune, int M,
                                         int N, int K, enum ggml_type src0_t,
                                         enum ggml_type src1_t,
                                         int stages_time[3]);

bool ggml_mulmat_tune_validate(const struct ggml_mulmat_tune *tune,
                               const char *model_name, int ftype,
                               int n_threads);

void ggml_mulmat_tune_model_init(struct ggml_mulmat_tune_model *model,
                                 const char *name, enum ggml_ftype ftype);

bool ggml_mulmat_tune_init(struct ggml_mulmat_tune *tune,
                           struct ggml_mulmat_tune_params *params,
                           ggml_task_profiles_provider *profiles_provider);

void ggml_mulmat_tune_free(struct ggml_mulmat_tune *tune);

bool ggml_mulmat_tune_write_data(const struct ggml_mulmat_tune *tune, FILE *fp);

bool ggml_mulmat_tune_read_data(struct ggml_mulmat_tune *tune, FILE *fp);

const struct ggml_mulmat_tune_shape *
ggml_mulmat_tune_get_shape(const struct ggml_mulmat_tune *tune, int N, int K,
                           enum ggml_type src0_type, enum ggml_type src1_type);

void ggml_mulmat_tune_estimate_time(const struct ggml_mulmat_tune_shape *shape,
                                    int M,
                                    struct ggml_mulmat_tune_time *profile_time);

const char *ggml_task_backend_name(enum ggml_task_backend backend);

int ggml_mulmat_tune_get_builtin_task_backends(
    enum ggml_task_backend *backends);

bool ggml_mulmat_tune_bench(struct ggml_mulmat_tune *tune,
                            struct ggml_mulmat_tune_params *params);

#ifdef __cplusplus
}
#endif
