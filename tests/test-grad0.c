#include "ggml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAX_NARGS 2

//
// logging
//

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



float frand() {
    return (float)rand()/(float)RAND_MAX;
}

int irand(int n) {
    return rand()%n;
}

void get_random_dims(int64_t * dims, int ndims) {
    dims[0] = dims[1] = dims[2] = dims[3] = 1;

    for (int i = 0; i < ndims; i++) {
        dims[i] = 1 + irand(4);
    }
}

struct ggml_tensor * get_random_tensor(
        struct ggml_context * ctx0,
        int ndims,
        int64_t ne[],
        float fmin,
        float fmax) {
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
    };

    return result;
}

float get_element(const struct ggml_tensor * t, int idx) {
    return ((float *)t->data)[idx];
}

void set_element(struct ggml_tensor * t, int idx, float value) {
    ((float *)t->data)[idx] = value;
}

bool check_gradient(
        const char * op_name,
        struct ggml_context * ctx0,
        struct ggml_tensor * x[],
        struct ggml_tensor * f,
        int ndims,
        int nargs,
        float eps,
        float max_error_abs,
        float max_error_rel) {

    struct ggml_cgraph gf = ggml_build_forward (f);
    struct ggml_cgraph gb = ggml_build_backward(ctx0, &gf, false);

    ggml_graph_compute(ctx0, &gf);
    ggml_graph_reset  (&gf);
    ggml_set_f32      (f->grad, 1.0f);
    ggml_graph_compute(ctx0, &gb);

    ggml_graph_dump_dot(&gf, NULL, "test-grad0-forward.dot");
    ggml_graph_dump_dot(&gb, &gf,  "test-grad0-backward.dot");

    for (int i = 0; i < nargs; ++i) {
        const int nelements = ggml_nelements(x[i]);
        for (int k = 0; k < nelements; ++k) {
            // compute gradient using finite differences
            const float x0 = get_element(x[i], k);
            const float xm = x0 - eps;
            const float xp = x0 + eps;
            set_element(x[i], k, xp);
            ggml_graph_compute(ctx0, &gf);

            const float f0 = ggml_get_f32_1d(f, 0);

            set_element(x[i], k, xm);
            ggml_graph_compute(ctx0, &gf);

            const float f1 = ggml_get_f32_1d(f, 0);

            const float g0 = (f0 - f1)/(2.0f*eps);

            set_element(x[i], k, x0);

            // compute gradient using backward graph
            ggml_graph_reset  (&gf);
            ggml_set_f32      (f->grad, 1.0f);
            ggml_graph_compute(ctx0, &gb);

            const float g1 = get_element(x[i]->grad, k);

            const float error_abs = fabsf(g0 - g1);
            const float error_rel = g0 != 0 ? fabsf(g0 - g1)/fabs(g0) : 0;

            if (error_abs > max_error_abs || error_rel > max_error_rel) {
                printf("%s: ndims=%d, i=%d, k=%d, x0=%f, xm=%f, xp=%f, f0=%f, f1=%f, g0=%f, g1=%f, eps=%f, error_abs=%f, error_rel=%f\n",
                            op_name, ndims, i, k, x0, xm, xp, f0, f1, g0, g1, eps, error_abs, error_rel);
                assert(false);
            }
        }
    }

    return true;
}

// TODO: clean-up this ..
bool check_mat_mul(
        const struct ggml_tensor * y,
        const struct ggml_tensor * x0,
        const struct ggml_tensor * x1) {
    float * dst  = (float *) y->data;
    float * src0 = (float *) x0->data;
    float * src1 = (float *) x1->data;

    const int nc = x0->ne[1];
    const int nr = x1->ne[1];
    const int nk = x0->ne[0];

    GGML_PRINT_DEBUG("check_mat_mul: nc=%d, nr=%d, nk=%d\n", nc, nr, nk);

    GGML_PRINT_DEBUG("x0:\n");
    for (int j = 0; j < x0->ne[1]; ++j) {
        for (int i = 0; i < x0->ne[0]; ++i) {
            GGML_PRINT_DEBUG("%6.3f ", src0[j*nk + i]);
        }
        GGML_PRINT_DEBUG("\n");
    }
    GGML_PRINT_DEBUG("\n");

    GGML_PRINT_DEBUG("x1:\n");
    for (int j = 0; j < x1->ne[1]; ++j) {
        for (int i = 0; i < x1->ne[0]; ++i) {
            GGML_PRINT_DEBUG("%6.3f ", src1[j*nk + i]);
        }
        GGML_PRINT_DEBUG("\n");
    }
    GGML_PRINT_DEBUG("\n");

    GGML_PRINT_DEBUG("y: n_dims = %d, (%lld, %lld)\n", y->n_dims, y->ne[0], y->ne[1]);
    for (int j = 0; j < y->ne[1]; ++j) {
        for (int i = 0; i < y->ne[0]; ++i) {
            GGML_PRINT_DEBUG("%6.3f ", dst[j*nr + i]);
        }
        GGML_PRINT_DEBUG("\n");
    }

    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < nc; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < nk; ++k) {
                sum += src0[j*nk + k]*src1[i*nk + k];
            }

            if (fabsf(dst[i*nc + j] - sum) > 1e-5f) {
                fprintf(stderr, "check_mat_mul: dst[%d] = %f, sum = %f\n", i*nc + j, dst[i*nc + j], sum);
                assert(false);
                return false;
            }
        }
    }

    return true;
}

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    int64_t ne[4];

    // original loop: 1000
    int niter = 1000;
    const char *env = getenv("GGML_NLOOP");
    if (env != NULL) {
        niter = atoi(env);
    }
    if (argc > 1) {
        niter = atoi(argv[1]);
    }
    for (int iter = 0; iter < niter; ++iter) {
        printf("test-grad0: iter:%d/%d\n", iter, niter);
        struct ggml_context * ctx0 = ggml_init(params);

        get_random_dims(ne, 4);

        struct ggml_tensor * x[MAX_NARGS];

        // add
        {
            const int nargs = 2;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_add(ctx0, x[0], x[1]));

                check_gradient("add", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f);
            }
        }

        // sub
        {
            const int nargs = 2;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_sub(ctx0, x[0], x[1]));

                check_gradient("sub", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f);
            }
        }

        // mul
        {
            const int nargs = 2;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_mul(ctx0, x[0], x[1]));

                check_gradient("mul", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
            }
        }

        // div
        {
            const int nargs = 2;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor(ctx0, ndims, ne, 0.5f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_div(ctx0, x[0], x[1]));

                check_gradient("div", ctx0, x, f, ndims, nargs, 1e-3f, INFINITY, 1e-2f);
            }
        }

        // sqr
        {
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_sqr(ctx0, x[0]));

                check_gradient("sqr", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
            }
        }

        // sqrt
        {
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor(ctx0, ndims, ne, 2.0f*1e-3f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, ggml_sqrt(ctx0, x[0]));

                check_gradient("sqrt", ctx0, x, f, ndims, nargs, 1e-3f, INFINITY, 1e-1f);
            }
        }

        // sum
        {
            const int nargs = 1;

            for (int ndims = 1; ndims <= 2; ++ndims) {
                for (int i = 0; i < nargs; ++i) {
                    x[i] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);
                    ggml_set_param(ctx0, x[i]);
                }

                struct ggml_tensor * f = ggml_sum(ctx0, x[0]);

                check_gradient("sum", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, 1e-3f);
            }
        }

        // abs (finite differences do not work)
        //{
        //    const int nargs = 1;

        //    for (int ndims = 1; ndims <= 2; ++ndims) {
        //        for (int i = 0; i < nargs; ++i) {
        //            x[i] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);
        //            ggml_set_param(ctx0, x[i]);
        //        }

        //        struct ggml_tensor * f = ggml_sum(ctx0, ggml_abs(ctx0, x[0]));

        //        check_gradient("abs", ctx0, x, f, ndims, nargs, 1e-3f, INFINITY, 1e-3f);
        //    }
        //}

        // mul_mat
        {
            const int nargs = 1;

            for (int ndims = 2; ndims <= 2; ++ndims) {
                x[0] = get_random_tensor(ctx0, ndims, ne, -1.0f, 1.0f);
                {
                    int64_t ne2[4];
                    get_random_dims(ne2, 4);
                    ne2[0] = ne[0];
                    x[1] = get_random_tensor(ctx0, ndims, ne2, -1.0f, 1.0f);
                }

                ggml_set_param(ctx0, x[0]);

                struct ggml_tensor * m = ggml_mul_mat(ctx0, x[1], x[0]);
                struct ggml_tensor * f = ggml_sum(ctx0, m);

                GGML_PRINT_DEBUG("testing: mul_mat, [%lld, %lld] (%d) * [%lld, %lld] (%d)\n", x[1]->ne[0], x[1]->ne[1], x[1]->n_dims, x[0]->ne[0], x[0]->ne[1], x[0]->n_dims);

                check_gradient("mul_mat", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
                check_mat_mul(m, x[1], x[0]);
            }
        }

        ggml_free(ctx0);
    }

    return 0;
}
