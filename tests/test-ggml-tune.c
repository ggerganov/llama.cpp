#include "ggml-tune.h"
#include "ggml.h"

#include <string.h>

static int bench(void);
static int estimate_time_non_zero_NK(void);

static void init_params(struct ggml_mulmat_tune_params *params, int m_num) {
    *params = (struct ggml_mulmat_tune_params){
        .model =
            (struct ggml_mulmat_tune_model){
                .name = "3B", // fake
                .ftype = GGML_FTYPE_MOSTLY_Q4_0,
                .n_vocab = 4096,
                .n_embd = 1024,
                .n_ff = 2048,
                .n_rot = 128,
            },
        .m_num = m_num,
        .n_pass = 1,
        .n_threads = 1,
        .progress = false,
        .output_console = true,
        .fname = NULL};
}

int main(void) {
    int rv = bench();
    if (rv != 0) {
        return rv;
    }

    printf("\n");

    rv = estimate_time_non_zero_NK();
    if (rv != 0) {
        return rv;
    }
    printf("\n");

    return 0;
}

static int bench(void) {
    printf("test: %s\n", __func__);

    {
        enum ggml_task_backend backends[16];
        int n_backends = ggml_mulmat_tune_get_builtin_task_backends(backends);
        if (n_backends < 2) {
            printf("test: %s, skipped because no BLAS\n", __func__);
            return 0;
        }
    }

    {
        struct ggml_init_params init_params = {
            /*.mem_size   =*/1,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/0,
        };
        struct ggml_context *ctx = ggml_init(init_params);
        GGML_ASSERT(ctx);
        ggml_free(ctx);
    }

    struct ggml_mulmat_tune tune;

    struct ggml_mulmat_tune_params params;

    init_params(&params, /*m_num*/ 4);

    bool ok = ggml_mulmat_tune_bench(&tune, &params);
    ggml_mulmat_tune_free(&tune);

    return ok ? 0 : 1;
}

int estimate_time_non_zero_NK(void) {
    printf("test: %s\n", __func__);

    struct test_data_t {
        int M;
        int time[3]; // 3 profiles.
    };

    struct ggml_mulmat_tune tune = {
        .version = 1,
        .ftype = GGML_FTYPE_MOSTLY_Q4_0,
    };

    const int m_num = 2;

    struct ggml_task_profile_factory pf;
    memset(&pf, 0, sizeof(struct ggml_task_profile_factory));

    {
        pf.n_qxx_f32 = 2;
        pf.qxx_f32[0].stages[0].backend = GGML_TASK_BACKEND_CPU;
        pf.qxx_f32[0].stages[1].backend = GGML_TASK_BACKEND_CPU;

        pf.qxx_f32[1].stages[0].backend = GGML_TASK_BACKEND_CPU;
        pf.qxx_f32[1].stages[1].backend = GGML_TASK_BACKEND_CPU_BLAS;
    }

    struct ggml_mulmat_tune_params params;
    init_params(&params, m_num);

    ggml_mulmat_tune_init(&tune, &params, &pf);

    struct ggml_mulmat_tune_shape *shape = NULL;
    for (int i = 0; i < tune.n_shapes; i++) {
        if (tune.shapes[i].N > 0 && tune.shapes[i].K > 0) {
            shape = &tune.shapes[i];
            break;
        }
    }
    GGML_ASSERT(shape);
    GGML_ASSERT(shape->n_profiles == 2);
    GGML_ASSERT(ggml_is_quantized(shape->src0_type));

    printf("shape: N: %d, K: %d, n_profiles: %d\n", shape->N, shape->K,
           shape->n_profiles);

    {
        shape->items[0] =
            (struct ggml_mulmat_tune_m){.M = 2, .stages_time = {2, 4, 0}};
        shape->items[1] =
            (struct ggml_mulmat_tune_m){.M = 4, .stages_time = {4, 8, 0}};

        shape->items[2] =
            (struct ggml_mulmat_tune_m){.M = 2, .stages_time = {4, 4, 0}};
        shape->items[3] =
            (struct ggml_mulmat_tune_m){.M = 4, .stages_time = {4, 4, 0}};
    }

    const struct test_data_t test_data[] = {
        {
            .M = 1, // out of range
            .time = {3, 8},
        },
        {
            .M = 2,
            .time = {6, 8},
        },
        {
            .M = 3,
            .time = {9, 8},
        },
        {
            .M = 4,
            .time = {12, 8},
        },
        {
            .M = 5, // out of range
            .time = {15, 8},
        },
    };

    int n_tests = (int)(sizeof(test_data) / sizeof(struct test_data_t));

    struct ggml_mulmat_tune_time profile_time[GGML_MAX_TASK_PROFILES];
    size_t profile_time_sz =
        sizeof(struct ggml_mulmat_tune_time) * GGML_MAX_TASK_PROFILES;

    int n_passed = 0;
    for (int i = 0; i < n_tests; i++) {
        memset(profile_time, 0, profile_time_sz);
        const struct test_data_t *e = &test_data[i];

        const struct ggml_mulmat_tune_shape *matched_shape =
            ggml_mulmat_tune_get_shape(&tune, shape->N, shape->K,
                                       shape->src0_type, shape->src1_type);
        GGML_ASSERT(matched_shape);
        GGML_ASSERT(matched_shape == shape);

        ggml_mulmat_tune_estimate_time(matched_shape, e->M, profile_time);

        for (int j = 0; j < shape->n_profiles; j++) {
            int actual = profile_time[j].total_time;
            int expect = e->time[j];
            if (expect != actual) {
                fprintf(stderr,
                        "test fail. i: %d, j: %d, M: %d, expect: "
                        "%d, actual: %d\n",
                        i, j, e->M, expect, actual);
            } else {
                ++n_passed;
            }
        }
    }

    n_tests *= shape->n_profiles;
    printf("%2d of %2d pass\n", n_passed, n_tests);

    ggml_mulmat_tune_free(&tune);

    return n_passed == n_tests ? 0 : 1;
}
