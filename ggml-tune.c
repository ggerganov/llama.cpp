#include <string.h>

#include "ggml-threading.h"
#include "ggml-tune.h"
#include "ggml.h"

// MUL_MAT fine tunning for non-GPU-offloading cases.

#define GGML_MULMAT_CACHE_LEN 16
static struct mm_cache_element default_mm_cache[GGML_MULMAT_CACHE_LEN] = {0};

#define FNV_OFFSET 14695981039346656037UL
#define FNV_PRIME 1099511628211UL
static uint64_t ggml_mulmat_tune_cache_hash(int M, int N, int K) {
    char buf[30];
    snprintf(buf, 30, "%d%d%d", M, N, K);

    uint64_t hash = FNV_OFFSET;
    for (const char *p = buf; *p; p++) {
        hash ^= (uint64_t)(unsigned char)(*p);
        hash *= FNV_PRIME;
    }
    return hash;
}

static const char *
ggml_mulmat_tune_task_backend_name(enum ggml_task_backend backend) {
    switch (backend) {
    case GGML_TASK_BACKEND_NONE:
        return "";
    case GGML_TASK_BACKEND_CPU:
        return "CPU";
    case GGML_TASK_BACKEND_CPU_BLAS:
        return "BLAS";
    case GGML_TASK_BACKEND_GPU:
        return "GPU";
    case GGML_TASK_BACKEND_GPU_CUDA:
        return "CUDA";
    case GGML_TASK_BACKEND_GPU_CL:
        return "CL";
    default:
        GGML_ASSERT(false);
    }
}

const struct ggml_task_profile *ggml_mulmat_tune_select_task_profile(
    struct ggml_mulmat_tune *tune, int M, int N, int K, enum ggml_type src0_t,
    enum ggml_type src1_t, int stages_time[3]) {
    GGML_ASSERT(tune);

    // TODO: default_mm_cache is thread-unsafe.
    struct mm_cache_element *mm_cache = default_mm_cache;
    int slot = ggml_mulmat_tune_cache_hash(M, N, K) % GGML_MULMAT_CACHE_LEN;
    struct mm_cache_element *e = &mm_cache[slot];

    struct ggml_mulmat_tune_time profiles_time[GGML_MAX_TASK_PROFILES] = {0};

    const struct ggml_task_profile *prof = NULL;

    if (e->M == M && e->N == N && e->K == K) {
        prof = e->profile;
        if (stages_time) {
            for (int i = 0; i < 3; i++) {
                stages_time[i] = e->stages_time[i];
            }
        }
    } else {
        const struct ggml_mulmat_tune_shape *shape = NULL;
        shape = ggml_mulmat_tune_get_shape(tune, N, K, src0_t, src1_t);
        if (shape) {
            ggml_mulmat_tune_estimate_time(shape, M, profiles_time);

            int min = INT32_MAX;
            int index = -1;
            for (int i = 0; i < shape->n_profiles; i++) {
                int total = profiles_time[i].total_time;
                if (total < min) {
                    min = total;
                    index = i;
                }
            }

            if (index >= 0) {
                prof = profiles_time[index].profile;
                for (int i = 0; i < 3; i++) {
                    int t = profiles_time[index].stage_time[i];
                    if (stages_time) {
                        stages_time[i] = t;
                    }
                    e->stages_time[i] = t;
                }

                GGML_ASSERT(prof);

                e->profile = prof;
                e->M = M;
                e->N = N;
                e->K = K;

#ifndef GGML_TUNE_NDEBUG
                const char *names[3];
                for (int i = 0; i < 3; i++) {
                    names[i] = ggml_mulmat_tune_task_backend_name(
                        prof->stages[i].backend);
                }
                printf(
                    "\n[tune] M: %3d, N: %5d, K: %5d, backends of the "
                    "fastest profile: %s %s %s\n",
                    M, N, K, names[0], names[1], names[2]);
#endif
            }
        }
    }

    return prof;
}

void ggml_mulmat_tune_model_init(struct ggml_mulmat_tune_model *model,
                                 const char *name, enum ggml_ftype ftype) {
    const int n_vocab = 32000;
    int n_embd;
    // n_ff = ((2*(4*n_embd)/3 + n_mult - 1)/n_mult)*n_mult
    int n_ff;
    // n_rot = n_embd/n_head;
    int n_rot;

    if (strcmp(name, "3B") == 0) {
        // n_head=32, n_mult=216, n_layer=26
        // https://github.com/ggerganov/llama.cpp/pull/1588
        n_embd = 3200;
        n_ff = 8640;
        n_rot = 100;
    } else if (strcmp(name, "7B") == 0) {
        n_embd = 4096;
        n_ff = 11008;
        n_rot = 128;
    } else if (strcmp(name, "13B") == 0) {
        n_embd = 5120;
        n_ff = 13824;
        n_rot = 128;
    } else if (strcmp(name, "30B") == 0) {
        n_embd = 6656;
        n_ff = 17920;
        n_rot = 128;
    } else if (strcmp(name, "65B") == 0) {
        n_embd = 8192;
        n_ff = 22016;
        n_rot = 128;
    } else {
        GGML_ASSERT(false);
    }

    model->name = name;
    model->ftype = ftype;
    model->n_vocab = n_vocab;
    model->n_embd = n_embd;
    model->n_ff = n_ff;
    model->n_rot = n_rot;
}

bool ggml_mulmat_tune_init(struct ggml_mulmat_tune *tune,
                           struct ggml_mulmat_tune_params *params,
                           ggml_task_profiles_provider *profiles_provider) {
    GGML_ASSERT(profiles_provider);
    struct ggml_mulmat_tune_model *model = &params->model;

    memset(tune, 0, sizeof(struct ggml_mulmat_tune));

    tune->version = GGML_MULMAT_TUNE_VERSION;
    tune->n_threads = params->n_threads;
    tune->ftype = model->ftype;

    size_t name_len = strlen(model->name);
    GGML_ASSERT(name_len > 0);
    strncpy(tune->model, model->name, sizeof(tune->model) - 1);

    const enum ggml_type rot_src0_type = GGML_TYPE_F16;
    const enum ggml_type src1_type = GGML_TYPE_F32;

    int n_vocab = model->n_vocab;
    int n_embd = model->n_embd;
    int n_ff = model->n_ff;
    int n_rot = model->n_rot;

    enum ggml_type type = ggml_ftype_to_ggml_type(model->ftype);

    GGML_ASSERT(GGML_MULMAT_N_SHAPES >= 6);
    tune->n_shapes = GGML_MULMAT_N_SHAPES;

    // Attention layers
    tune->shapes[0] = (struct ggml_mulmat_tune_shape){
        .N = n_embd, .K = n_embd, .src0_type = type, .src1_type = src1_type};
    // Feed forward layers
    tune->shapes[1] = (struct ggml_mulmat_tune_shape){
        .N = n_embd, .K = n_ff, .src0_type = type, .src1_type = src1_type};
    tune->shapes[2] = (struct ggml_mulmat_tune_shape){
        .N = n_ff, .K = n_embd, .src0_type = type, .src1_type = src1_type};
    tune->shapes[3] = (struct ggml_mulmat_tune_shape){
        .N = n_vocab, .K = n_embd, .src0_type = type, .src1_type = src1_type};
    // RoPE
    tune->shapes[4] = (struct ggml_mulmat_tune_shape){
        .N = n_rot, .K = 0, .src0_type = rot_src0_type, .src1_type = src1_type};
    tune->shapes[5] = (struct ggml_mulmat_tune_shape){
        .N = 0, .K = n_rot, .src0_type = rot_src0_type, .src1_type = src1_type};

    for (int i = 0; i < tune->n_shapes; i++) {
        struct ggml_mulmat_tune_shape *shape = &tune->shapes[i];

        struct ggml_tensor src0 = {
            .type = shape->src0_type,
        };
        struct ggml_tensor src1 = {
            .type = shape->src1_type,
        };
        struct ggml_tensor node = {
            .op = GGML_OP_MUL_MAT,
            .src0 = &src0,
            .src1 = &src1,
        };

        shape->n_profiles = profiles_provider(&node, shape->profiles);
        if (shape->n_profiles == 0) {
            // allowed for testing.
            continue;
        }

        shape->m_num = params->m_num;
        shape->arr_m = malloc(shape->m_num * sizeof(int));
        for (int j = 0; j < shape->m_num; j++) {
            shape->arr_m[j] = 1 << j;
        }

        size_t sz = sizeof(struct ggml_mulmat_tune_m) *
                    (shape->n_profiles * shape->m_num);
        shape->items = malloc(sz);
        GGML_ASSERT(shape->items);
        memset(shape->items, 0, sz);
    }

    return true;
}

void ggml_mulmat_tune_free(struct ggml_mulmat_tune *tune) {
    for (int i = 0; i < tune->n_shapes; i++) {
        struct ggml_mulmat_tune_shape *shape = &tune->shapes[i];
        GGML_ASSERT(shape);

        // arr_m and items can be NULL only when testing.
        if (shape->arr_m) {
            free(shape->arr_m);
        }
        if (shape->items) {
            free(shape->items);
        }
    }
}

static bool ggml_mulmat_tune_write_profiles(
    FILE *fp, const struct ggml_task_profile *profiles, int n_profiles) {
    int rc;
    for (int i = 0; i < n_profiles; i++) {
        const struct ggml_task_profile *profile = &profiles[i];
        for (int j = 0; j < 3; j++) {
            const struct ggml_task_stage *ts = &profile->stages[j];
            rc = fprintf(fp, "%2d %d %d", ts->backend, ts->parallel ? 1 : 0,
                         ts->wait ? 1 : 0);
            if (rc <= 0) {
                return false;
            }
            if (j < 2) {
                rc = fprintf(fp, "  ");
                if (rc <= 0) {
                    return false;
                }
            }
        }
        rc = fprintf(fp, "\n");
        if (rc <= 0) {
            return false;
        }
    }

    return true;
}

static bool
ggml_mulmat_tune_validate_internal(const struct ggml_mulmat_tune *tune,
                                   const char *model, int ftype, int n_threads,
                                   char *errbuf, int errbuf_len) {

    if (tune->version != GGML_MULMAT_TUNE_VERSION) {
        snprintf(errbuf, errbuf_len - 1,
                 "version mismatch, built-in: %d, "
                 "yours: %d",
                 GGML_MULMAT_TUNE_VERSION, tune->version);
        return false;
    } else if (strcmp(model, tune->model) != 0) {
        snprintf(errbuf, errbuf_len - 1,
                 "model mismatch. built-in: %s, yours: %s", model, tune->model);
        return false;
    } else if (ftype != tune->ftype) {
        snprintf(errbuf, errbuf_len - 1,
                 "ftype mismatch. built-in: %d, yours: %d\n", ftype,
                 tune->ftype);
        return false;
    } else if (n_threads != tune->n_threads) {
        snprintf(errbuf, errbuf_len - 1,
                 "n_threads mismatch. run-time: %d, yours: %d\n", n_threads,
                 tune->n_threads);
        return false;
    }

    for (int i = 0; i < tune->n_shapes; i++) {
        const struct ggml_mulmat_tune_shape *shape = &tune->shapes[i];

        struct ggml_tensor src0 = {
            .type = shape->src0_type,
        };
        struct ggml_tensor src1 = {
            .type = shape->src1_type,
        };
        struct ggml_tensor node = {
            .op = GGML_OP_MUL_MAT,
            .src0 = &src0,
            .src1 = &src1,
        };

        struct ggml_task_profile builtin_profiles[GGML_MAX_TASK_PROFILES];
        int n_profiles = ggml_get_task_profiles(&node, builtin_profiles);

        if (n_profiles != shape->n_profiles) {
            snprintf(errbuf, errbuf_len - 1, "task profiles mismatch");
            return false;
        }

        // TODO: profiles order is relevant, too strict.
        size_t sz = sizeof(struct ggml_task_profile) * n_profiles;
        if (memcmp(builtin_profiles, shape->profiles, sz) != 0) {
            snprintf(errbuf, errbuf_len - 1, "task profiles mismatch");

            printf("=== built-in profiles:\n");
            ggml_mulmat_tune_write_profiles(stderr, builtin_profiles,
                                            n_profiles);

            printf("=== incoming profiles:\n");
            ggml_mulmat_tune_write_profiles(stderr, shape->profiles,
                                            shape->n_profiles);
            return false;
        }
    }

    return true;
}

bool ggml_mulmat_tune_validate(const struct ggml_mulmat_tune *tune,
                               const char *model, int ftype, int n_threads) {
    char errbuf[128];
    bool ok = ggml_mulmat_tune_validate_internal(tune, model, ftype, n_threads,
                                                 errbuf, sizeof(errbuf));
    if (!ok) {
        fprintf(stderr, "[tune] error: %s. run bench again.\n", errbuf);
    }

    return ok;
}

bool ggml_mulmat_tune_read_data(struct ggml_mulmat_tune *tune, FILE *fp) {
    int rc = fscanf(fp, "%d", &tune->version);
    if (rc <= 0) {
        return false;
    }

    if (tune->version != GGML_MULMAT_TUNE_VERSION) {
        fprintf(stderr, "[tune] version mismatch, run bench again\n");
        return false;
    }

    rc = fscanf(fp, "%s %d %d %d", tune->model, (int *)&tune->ftype,
                &tune->n_shapes, &tune->n_threads);
    if (rc <= 0) {
        return false;
    }

    for (int i_shape = 0; i_shape < tune->n_shapes; i_shape++) {
        struct ggml_mulmat_tune_shape *shape = &tune->shapes[i_shape];

        rc = fscanf(fp, "%d %d %d %d %d %d", &shape->N, &shape->K,
                    (int *)&shape->src0_type, (int *)&shape->src1_type,
                    &shape->n_profiles, &shape->m_num);
        if (rc <= 0) {
            return false;
        }

        {
            size_t item_size = sizeof(struct ggml_mulmat_tune_m) *
                               (shape->n_profiles * shape->m_num);
            shape->items = malloc(item_size);
            if (shape->items == NULL) {
                fprintf(stderr, "[tune] failed to allocate memory\n");
                return false;
            }
            memset(shape->items, 0, item_size);
        }

        for (int ip = 0; ip < shape->n_profiles; ip++) {
            struct ggml_task_profile *profile = &shape->profiles[ip];
            for (int j = 0; j < 3; j++) {
                struct ggml_task_stage *ts = &profile->stages[j];
                int backend;
                int parallel;
                int wait;
                rc = fscanf(fp, "%d %d %d", &backend, &parallel, &wait);
                if (rc <= 0) {
                    return false;
                }
                ts->backend = (enum ggml_task_backend)backend;
                ts->parallel = parallel ? true : false;
                ts->wait = wait ? true : false;
            }
        }

        for (int i_m = 0; i_m < shape->m_num; i_m++) {
            int M;
            for (int ip = 0; ip < shape->n_profiles; ip++) {
                if (ip == 0) {
                    rc = fscanf(fp, "%d", &M);
                    if (rc <= 0) {
                        return false;
                    }
                }
                struct ggml_mulmat_tune_m *item =
                    &shape->items[ip * shape->m_num + i_m];
                item->M = M;
                rc = fscanf(fp, "%d %d %d", &item->stages_time[0],
                            &item->stages_time[1], &item->stages_time[2]);
                if (rc <= 0) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool ggml_mulmat_tune_write_data(const struct ggml_mulmat_tune *tune,
                                 FILE *fp) {
    int rc;
    rc = fprintf(fp, "%d %s %d %d %d\n\n", tune->version, tune->model,
                 tune->ftype, tune->n_shapes, tune->n_threads);
    if (rc <= 0) {
        return false;
    }

    for (int i_shape = 0; i_shape < tune->n_shapes; i_shape++) {
        if (i_shape > 0) {
            fprintf(fp, "\n");
        }
        const struct ggml_mulmat_tune_shape *shape = &tune->shapes[i_shape];
        rc = fprintf(fp, "%d %d %d %d %d %d\n", shape->N, shape->K,
                     shape->src0_type, shape->src1_type, shape->n_profiles,
                     shape->m_num);
        if (rc <= 0) {
            return false;
        }

        rc = ggml_mulmat_tune_write_profiles(fp, shape->profiles,
                                             shape->n_profiles);
        if (rc <= 0) {
            return false;
        }

        for (int i_m = 0; i_m < shape->m_num; i_m++) {
            for (int ip = 0; ip < shape->n_profiles; ip++) {
                struct ggml_mulmat_tune_m *item =
                    &shape->items[ip * shape->m_num + i_m];
                if (ip == 0) {
                    rc = fprintf(fp, "%4d", item->M);
                    if (rc <= 0) {
                        return false;
                    }
                }

                const struct ggml_task_profile *profile = &shape->profiles[ip];
                for (int k = 0; k < 3; k++) {
                    if (profile->stages[k].backend != GGML_TASK_BACKEND_NONE) {
                        rc = fprintf(fp, "%9d", item->stages_time[k]);
                        if (rc <= 0) {
                            return false;
                        }
                    } else {
                        rc = fprintf(fp, " 0");
                        if (rc <= 0) {
                            return false;
                        }
                    }
                }
            }
            rc = fprintf(fp, "\n");
            if (rc <= 0) {
                return false;
            }
        }
    }

    return true;
}

const struct ggml_mulmat_tune_shape *
ggml_mulmat_tune_get_shape(const struct ggml_mulmat_tune *tune, const int N,
                           const int K, enum ggml_type src0_type,
                           enum ggml_type src1_type) {
    GGML_ASSERT(N > 0 && K > 0);

    for (int i = 0; i < tune->n_shapes; i++) {
        const struct ggml_mulmat_tune_shape *s = &tune->shapes[i];
        if (s->src0_type != src0_type || s->src1_type != src1_type) {
            continue;
        }

        if (s->N > 0 && s->K > 0) {
            if (s->N == N && s->K == K) {
                return s;
            }
        } else if (s->N > 0 && s->K == 0) {
            if (s->N == N) {
                return s;
            }
        } else if (s->N == 0 && s->K > 0) {
            if (s->K == K) {
                return s;
            }
        }
    }

    return NULL;
}

// This is the experimental reference implementation.
// Requires both n_threads are same at bench time and runtime.
void ggml_mulmat_tune_estimate_time(
    const struct ggml_mulmat_tune_shape *shape, int M,
    struct ggml_mulmat_tune_time *profile_time) {

    GGML_ASSERT(shape);
    GGML_ASSERT(profile_time);

    const int m_num = shape->m_num;
    const int min_m = shape->items[0].M;
    const int max_m = shape->items[m_num - 1].M;

    for (int ip = 0; ip < shape->n_profiles; ip++) {
        const struct ggml_task_profile *profile = &shape->profiles[ip];
        profile_time[ip].total_time = 0;
        profile_time[ip].profile = profile;

        const int items_offset = ip * m_num;

        struct ggml_mulmat_tune_m *p0 = NULL;
        struct ggml_mulmat_tune_m *p1 = NULL;
        if (M < min_m) {
            // first two.
            p0 = &shape->items[items_offset];
            p1 = &shape->items[items_offset + 1];
        } else if (M > max_m) {
            // last two
            p0 = &shape->items[items_offset + m_num - 2];
            p1 = &shape->items[items_offset + m_num - 1];
        } else {
            for (int i = 0; i < m_num; i++) {
                p1 = &shape->items[items_offset + i];
                if (p1->M == M) {
                    p0 = p1;
                    break;
                }

                if (i > 0) {
                    p0 = (struct ggml_mulmat_tune_m *)(p1 - 1);
                    if (M > p0->M && M < p1->M) {
                        break;
                    }
                }
            }
        }

        GGML_ASSERT(p0 && p1);

        for (int i_stage = 0; i_stage < 3; i_stage++) {
            const struct ggml_task_stage *stage = &profile->stages[i_stage];
            if (stage->backend == GGML_TASK_BACKEND_NONE) {
                continue;
            }

            int p0_v = p0->stages_time[i_stage];
            int p1_v = p1->stages_time[i_stage];

            GGML_ASSERT(p0_v >= 0);
            GGML_ASSERT(p1_v >= 0);

            // t = aM + b
            double a;
            double b;

            if (p0 == p1) {
                a = 0.0;
                b = p1_v;
            } else {
                a = 1.0 * (p1_v - p0_v) / (p1->M - p0->M);
                b = p1_v - a * p1->M;
            }
            int t = (int)(a * M + b);

            profile_time[ip].stage_time[i_stage] = t;
            profile_time[ip].total_time += t;
        }
    }
}

// Experimental: create mul_mat tensor.
static struct ggml_tensor *ggml_mulmat_new_tensor(int M, int N, int K,
                                                  enum ggml_type src0_type,
                                                  struct ggml_context **ctx) {
    // At most 256, because in `ggml_quantize_qx_x`, the index type of hist is
    // either int8_t or uint8_t.
    // Use 1024 to avoid suddenly broken.
    int64_t hist[1024];

    bool src0_is_quantized = ggml_is_quantized(src0_type);

    size_t ctx_size = 0;
    ctx_size += (size_t)(M * N * ggml_type_sizef(GGML_TYPE_F32)); // src1
    ctx_size += (size_t)(N * K * ggml_type_sizef(src0_type));     // src0
    ctx_size += (size_t)(1024 * 1024 * 64); // experimental

    if (src0_is_quantized) {
        // quantize F32 to Qx_x
        ctx_size += (size_t)(N * K * ggml_type_sizef(GGML_TYPE_F32));
    }

    struct ggml_init_params init_params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = 0,
    };

    *ctx = ggml_init(init_params);
    GGML_ASSERT(*ctx);

    // src0: N x K
    struct ggml_tensor *src0 =
        ggml_new_tensor_2d(*ctx, src0_type, (int64_t)K, (int64_t)N);

    // src1: M x K
    struct ggml_tensor *src1 =
        ggml_new_tensor_2d(*ctx, GGML_TYPE_F32, (int64_t)K, (int64_t)M);
    ggml_set_f32(src1, 0.5f);

    if (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16) {
        ggml_set_f32(src0, 0.1f);
    } else if (src0_is_quantized) {
        struct ggml_tensor *src0_f32 =
            ggml_new_tensor_2d(*ctx, GGML_TYPE_F32, (int64_t)K, (int64_t)N);
        ggml_set_f32(src0_f32, 0.1f);

        switch (src0_type) {
        case GGML_TYPE_Q4_0:
            ggml_quantize_q4_0((const float *)src0_f32->data, src0->data, N * K,
                               K, hist);
            break;
        case GGML_TYPE_Q4_1:
            ggml_quantize_q4_1((const float *)src0_f32->data, src0->data, N * K,
                               K, hist);
            break;
        case GGML_TYPE_Q5_0:
            ggml_quantize_q5_0((const float *)src0_f32->data, src0->data, N * K,
                               K, hist);
            break;
        case GGML_TYPE_Q5_1:
            ggml_quantize_q5_1((const float *)src0_f32->data, src0->data, N * K,
                               K, hist);
            break;
        case GGML_TYPE_Q8_0:
            ggml_quantize_q8_0((const float *)src0_f32->data, src0->data, N * K,
                               K, hist);
            break;
        default:
            GGML_ASSERT(false);
        }
    } else {
        GGML_ASSERT(false);
    }

    // node: M x N
    // Will compute z = y * xT, z: node, y: src1, x: src0
    return ggml_mul_mat(*ctx, src0, src1);
}

// Experimental: allocate memory for wdata with max possible size.
// This part of code is actually belongs to ggml compute graph.
static size_t ggml_mulmat_allocate_wdata(int N, int K, char **wdata) {
    // The size is actually determined by cgraph before computing.
    // Apart from the src0_type, wsize is affected by backend, cache line size,
    // n_threads etc.

    const size_t extra = 1024 * 1024;
    size_t sz = (size_t)(N * K * ggml_type_sizef(GGML_TYPE_F32)) + extra;
    void *buf = malloc(sz);

    if (!buf) {
        fprintf(stderr,
                "[tune] error: failed to allocate %zu MiB memory",
                sz / 1024 / 1024);
        return 0;
    }

    memset(buf, 0, sz);
    *wdata = buf;
    return sz;
}

int ggml_mulmat_tune_get_builtin_task_backends(
    enum ggml_task_backend *backends) {
    int i = 0;
    backends[i++] = GGML_TASK_BACKEND_CPU;

    if (ggml_cpu_has_cpublas()) {
        backends[i++] = GGML_TASK_BACKEND_CPU_BLAS;
    }

    if (ggml_cpu_has_cublas()) {
        backends[i++] = GGML_TASK_BACKEND_GPU_CUDA;
    } else if (ggml_cpu_has_clblast()) {
        backends[i++] = GGML_TASK_BACKEND_GPU_CL;
    }
    return i;
}

bool ggml_mulmat_tune_bench(struct ggml_mulmat_tune *tune,
                            struct ggml_mulmat_tune_params *params) {
    GGML_ASSERT(tune);
    GGML_ASSERT(params);
    GGML_ASSERT(params->model.name);

    enum ggml_task_backend backends[16];
    int n_backends = ggml_mulmat_tune_get_builtin_task_backends(backends);
    if (n_backends < 2) {
        fprintf(stderr,
                "[tune] error: this program was not built with BLAS.\n");
        return false;
    }

    bool ok = ggml_mulmat_tune_init(tune, params, ggml_get_task_profiles);
    if (!ok) {
        return false;
    }

    {
        char buf[128] = {0};
        int offset = 0;

        for (int i = 0; i < n_backends; i++) {
            if (i > 0) {
                buf[offset++] = ',';
                buf[offset++] = ' ';
            }
            const char *name = ggml_mulmat_tune_task_backend_name(backends[i]);
            size_t len = strlen(name);
            memcpy(&buf[offset], name, len);
            offset += (int)len;
        }

        fprintf(stdout,
                "[tune] model: %s, ggml ftype: %d, "
                "n_pass: %d, n_threads: %d, n_shapes: %d, backends: %s\n",
                params->model.name, params->model.ftype, params->n_pass,
                params->n_threads, tune->n_shapes, buf);
    }

    int64_t stages_time[3];
    int64_t t0 = ggml_time_ms();

    struct ggml_threading_context *thrd_ctx = ggml_threading_start(
        tune->n_threads, ggml_threading_graph_compute_thread,
        ggml_compute_forward_wrapper, GGML_THREADING_FEATURE_WAIT_ON_DONE,
        stages_time);

    for (int i_shape = 0; i_shape < tune->n_shapes; i_shape++) {
        const struct ggml_mulmat_tune_shape *shape = &tune->shapes[i_shape];
        int M;
        int N = shape->N;
        int K = shape->K;

        char buf[20] = {0};
        int buf_len = sizeof(buf) - 1;
        int line_len = 0;

        for (int i_m = 0; i_m < shape->m_num; i_m++) {
            M = shape->arr_m[i_m];
            if (shape->N == 0) {
                N = M;
            } else if (shape->K == 0) {
                K = M;
            }

            if (params->progress) {
                line_len = snprintf(buf, buf_len, "%d %d %d ", N, K, M);
                fprintf(stdout, "%s", buf);
                fflush(stdout);
            }

            char *wdata = NULL;
            size_t wsize = ggml_mulmat_allocate_wdata(N, K, &wdata);
            if (wsize == 0) {
                return false;
            }

            struct ggml_context *ctx = NULL;
            struct ggml_tensor *node =
                ggml_mulmat_new_tensor(M, N, K, shape->src0_type, &ctx);

            for (int ip = 0; ip < shape->n_profiles; ip++) {
                const struct ggml_task_profile *profile = &shape->profiles[ip];

                memcpy(&node->task_profile, profile,
                       sizeof(struct ggml_task_profile));

                struct ggml_mulmat_tune_m *item =
                    &shape->items[ip * shape->m_num + i_m];
                item->M = M;

                int min[3] = {INT32_MAX, INT32_MAX, INT32_MAX};

                for (int k = 0; k < params->n_pass; k++) {
                    for (int j = 0; j < 3; j++) {
                        stages_time[j] = 0;
                    }

                    /*enum ggml_compute_error err = */
                    ggml_threading_compute_tensor(thrd_ctx, node, wdata, wsize);

                    for (int i = 0; i < 3; i++) {
                        int v = (int)stages_time[i];
                        if (v < min[i]) {
                            min[i] = v;
                        }
                    }

                    if (params->progress) {
                        fprintf(stdout, ".");
                        fflush(stdout);
                        line_len++;
                    }
                }
                for (int i = 0; i < 3; i++) {
                    item->stages_time[i] = min[i];
                }
            }

            ggml_free(ctx);
            free(wdata);

            if (params->progress) {
                line_len += 10;
                for (int j = 0; j < line_len; j++) {
                    fprintf(stdout, "\b \b");
                }
                fflush(stdout);
            }
        }
    }

    ggml_threading_stop(thrd_ctx);

    fprintf(stdout, "[tune] done, elapsed time: %d seconds.\n",
            (int)(ggml_time_ms() - t0) / 1000);

    // output

    if (params->fname && strcmp(params->fname, "") != 0) {
        FILE *fp = fopen(params->fname, "w");
        if (!fp) {
            fprintf(stderr,
                    "[tune] warn: failed to open file `%s`, print to "
                    "console instead\n\n",
                    params->fname);
            params->output_console = 1;
        } else {
            ok = ggml_mulmat_tune_write_data(tune, fp);
            fclose(fp);

            if (ok) {
                fprintf(stdout, "[tune] data was written to `%s`\n",
                        params->fname);
            } else {
                fprintf(
                    stderr,
                    "[tune] warn: failed to write file `%s`, print to "
                    "console instead\n\n",
                    params->fname);
                params->output_console = 1;
            }
        }
    }

    if (params->output_console) {
        return ggml_mulmat_tune_write_data(tune, stdout);
    }

    return true;
}
