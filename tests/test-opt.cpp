#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#define MAX_NARGS 2

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

//
// logging
//
#define GGML_DEBUG 0
#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)


static float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

static struct ggml_tensor * get_random_tensor(
    struct ggml_context * ctx0, int ndims, int64_t ne[], float fmin, float fmax
) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)result->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)result->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    }

    return result;
}

int main(void) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    int64_t ne1[4] = {4, 128, 1, 1};
    int64_t ne2[4] = {4, 256, 1, 1};
    int64_t ne3[4] = {128, 256, 1, 1};

    struct ggml_tensor * a = get_random_tensor(ctx, 2, ne1, -1, +1);
    struct ggml_tensor * b = get_random_tensor(ctx, 2, ne2, -1, +1);
    ggml_set_param(ctx, a);
    ggml_set_param(ctx, b);

    struct ggml_tensor * c = get_random_tensor(ctx, 2, ne3, -1, +1);

    struct ggml_tensor * ab = ggml_mul_mat(ctx, a, b);
    struct ggml_tensor * d  = ggml_sub(ctx, c, ab);
    struct ggml_tensor * e  = ggml_sum(ctx, ggml_sqr(ctx, d));

    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(ge, e);
    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    const float fe = ggml_get_f32_1d(e, 0);
    printf("%s: e = %.4f\n", __func__, fe);

    struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_ADAM);

    ggml_opt(ctx, opt_params, e);

    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    const float fe_opt = ggml_get_f32_1d(e, 0);
    printf("%s: original  e = %.4f\n", __func__, fe);
    printf("%s: optimized e = %.4f\n", __func__, fe_opt);

    const bool success = (fe_opt <= fe);
    assert(success);

    ggml_free(ctx);
    return success ? 0 : -1;
}
// int64_t ne1[4] = {4, 128, 1, 1};
// int64_t ne2[4] = {4, 256, 1, 1};;
// int64_t ne3[4] = {128, 256, 1, 1};
// main: original  e = 25890.9375
// main: optimized e = 10094.7031

// int64_t ne1[4] = {8, 128, 1, 1};
// int64_t ne2[4] = {8, 256, 1, 1};;
// int64_t ne3[4] = {128, 256, 1, 1};
// main: original  e = 39429.5078
// main: optimized e = 9275.8936

// int64_t ne1[4] = {16, 128, 1, 1};
// int64_t ne2[4] = {16, 256, 1, 1};;
// int64_t ne3[4] = {128, 256, 1, 1};
// main: original  e = 68371.1328
// main: optimized e = 7854.4502


// int64_t ne1[4] = {32, 128, 1, 1};
// int64_t ne2[4] = {32, 256, 1, 1};;
// int64_t ne3[4] = {128, 256, 1, 1};
// main: original  e = 126061.1953
// main: optimized e = 5451.0166

// int64_t ne1[4] = {4, 1024, 1, 1};
// int64_t ne2[4] = {4, 2048, 1, 1};;
// int64_t ne3[4] = {1024, 2048, 1, 1};
// main: original  e = 1620817.8750
// main: optimized e = 698387.6875

// another run on M1
// int64_t ne1[4] = {4, 1024, 1, 1};
// int64_t ne2[4] = {4, 2048, 1, 1};;
// int64_t ne3[4] = {1024, 2048, 1, 1};
// main: original  e = 1629595.6250
// main: optimized e = 698169.1250

// int64_t ne1[4] = {32, 1024, 1, 1};
// int64_t ne2[4] = {32, 2048, 1, 1};;
// int64_t ne3[4] = {1024, 2048, 1, 1};
// main: original  e = 8146770.5000
// main: optimized e = 651119.1250
