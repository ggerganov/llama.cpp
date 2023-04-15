/*
    License: MIT License

    Changelog:
    - 2023-03-31 Initial version by Sebastian Apel (https://github.com/SebastianApel)

*/

#include <locale.h>
#include "ggml.h"
#include <assert.h>
#include <math.h>
#include <cstring>
#include <cstdio>
#include <cinttypes>
#include <unordered_map>
#include <queue>
#include <string.h>
#include <cassert>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>

float tensor_sum_elements(struct ggml_tensor * tensor) {
    float sum = 0;
    if (tensor->type==GGML_TYPE_F32) {
        for (int j = 0; j < tensor->ne[1]; j++) {
            for (int k = 0; k < tensor->ne[0]; k++) {
                sum +=  ((float *) tensor->data)[j*tensor->ne[0]+k];
            }
        }
    }
    return sum;
}


/*
    These are mapping to unknown
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
*/

#define TENSOR_TYPE_AS_STR(TYPE) TYPE == GGML_TYPE_F32 ? "FP32" : TYPE == GGML_TYPE_F16 ? "FP16" : TYPE == GGML_TYPE_Q4_0 ? "Q4_0" : TYPE == GGML_TYPE_Q4_1 ? "Q4_1" : "UNKNOWN"

#define TENSOR_DUMP(TENSOR) printf("%15s: type = %i (%5s) ne = %5d x %5d x %5d, nb = (%5li, %5li, %5li) - ", #TENSOR, \
        TENSOR->type,TENSOR_TYPE_AS_STR(TENSOR->type),\
        TENSOR->ne[0], TENSOR->ne[1], TENSOR->ne[2], TENSOR->nb[0], TENSOR->nb[1], TENSOR->nb[2]); \
    { float sum = tensor_sum_elements(TENSOR); printf("Sum of tensor %s is %6.2f\n",#TENSOR, sum); }

struct benchmark_params_struct {
    int32_t n_threads     = 1;
    int32_t n_iterations  = 10;
};

void print_usage(int /*argc*/, char ** argv, struct benchmark_params_struct params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -i N, --iter N     number of iterations to use during computation (default: %d)\n", params.n_iterations);
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv)  {


    struct benchmark_params_struct benchmark_params;

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-i" || arg == "--iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_iterations = std::stoi(argv[i]);
        }  else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, benchmark_params);
            exit(0);
        }
        if (invalid_param) {
            fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
            print_usage(argc, argv, benchmark_params);
            exit(1);
        }
    }


    // create the ggml context
    printf("Starting Test\n");



    struct ggml_context * ctx;
    //const int sizex = 4096;
    //const int sizey = 11008;

#undef VERBOSE_DEBUGGING
#ifndef VERBOSE_DEBUGGING
    const int sizey = 4096;
    const int sizex = 11008;
    const int sizez = 128;
#else
    /* Working - let's increase size */
    const int sizey = 1;
    const int sizex = (8*32);
    const int sizez = 1;

    /*const int sizey = 1;
    const int sizex = 3*(8*32);
    const int sizez = 1;*/
#endif

    //printf("Memsize required = %i\n", sizex*sizex);
    ggml_type wtype = GGML_TYPE_F32;

    size_t ctx_size = 0;
    ctx_size += sizex*sizey*ggml_type_sizef(wtype);
    ctx_size += sizex*sizey*ggml_type_sizef(wtype);
    ctx_size += sizex*sizey*ggml_type_sizef(GGML_TYPE_F32);
    ctx_size += sizex*sizeof(float);
    ctx_size += 1024*1024*100;

    printf("Allocating Memory of size %li byes, %li MB\n",ctx_size, (ctx_size/1024/1024));

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /* no_alloc   =*/ 0
    };

    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return false;
    }


    printf("Creating new tensors\n");
    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m11 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
    ggml_set_f32(m11, 1.0f);

    // printf("Creating new tensor m1\n");
    struct ggml_tensor * m12 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizey);
    ggml_set_f32(m12, 1.5f);

    // printf("Creating new tensor m2\n");
    struct ggml_tensor * m2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizez);
    ggml_set_f32(m2, 2.0f);

    printf("\n------ Test 1 - Matrix Mult via F32 code ------------------------------------------------------------------------------\n");
    // printf("Creating new tensor m11xm2\n");
    struct ggml_tensor * m11xm2 = ggml_mul_mat(ctx, m11, m2);

    // printf("Creating compute graph\n");
    struct ggml_cgraph gf = ggml_build_forward(m11xm2);

    gf.n_threads=benchmark_params.n_threads;
    printf("cgraph->n_threads=%i\n",gf.n_threads);

    TENSOR_DUMP(m11);
    TENSOR_DUMP(m2);

    ggml_graph_compute(ctx, &gf);

    TENSOR_DUMP(gf.nodes[0]);

    printf("\n------ Test 2 - Matrix Mult via Q4_0 code ------------------------------------------------------------------------------\n");

    int32_t nelements = sizex*sizey;
    int32_t ne[2] = { sizex, sizey };

    std::vector<int64_t> hist_cur(1 << 4, 0);

    // Set up a the benchmark matrices
    // printf("Creating new tensor q11 & Running quantize\n");
    struct ggml_tensor * q11 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, sizex, sizey);
    ggml_quantize_q4_0((const float *) m11->data, q11->data, nelements, ne[0], hist_cur.data());

    // Set up a the compute graph
    // printf("Creating new tensor q31\n");
    struct ggml_tensor * q31 = ggml_mul_mat(ctx, q11, m2);

    // printf("Creating compute graph\n");
    struct ggml_cgraph gf31 = ggml_build_forward(q31);
    gf31.n_threads=benchmark_params.n_threads;

    // Set up a second graph computation to make sure we override the CPU cache lines
    // printf("Creating new tensor q12 & Running quantize\n");
    struct ggml_tensor * q12 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, sizex, sizey);
    ggml_quantize_q4_0((const float *) m12->data, q12->data, nelements, ne[0], hist_cur.data());

    // printf("Creating new tensor q32\n");
    struct ggml_tensor * q32 = ggml_mul_mat(ctx, q12, m2);

    //printf("Creating compute graph\n");
    struct ggml_cgraph gf32 = ggml_build_forward(q32);
    gf32.n_threads=benchmark_params.n_threads;
    printf("cgraph->n_threads=%i\n",gf31.n_threads);

    const int dimx = sizex;
    const int dimy = sizey;
    const int dimz = sizez;
    long long int flops_per_dot_product = dimy + dimy;
    long long int flops_per_matrix = flops_per_dot_product * dimx * dimz; ;
    printf("Matrix Multiplication of (%i,%i,%i) x (%i,%i,%i) - aboout %6.2f gFLOPS\n\n", sizex, sizey, 1, sizex, sizez, 1, 1.0f*flops_per_matrix / 1000 / 1000 / 1000);


    // Let's use the F32 result from above as a reference for the q4_0 multiplication
    float sum_of_F32_reference = tensor_sum_elements(gf.nodes[0]);


    printf("Iteration;NThreads; SizeX; SizeY; SizeZ; Required_FLOPS; Elapsed_u_Seconds; FLOPS_per_u_Second\n");
    printf("==============================================================================================\n");

    for (int i=0;i<benchmark_params.n_iterations ;i++) {

        long long int start = ggml_time_us();
        //printf("Running ggml_graph_compute\n");
        ggml_graph_compute(ctx, &gf31);
        long long int stop = ggml_time_us();
        long long int usec = stop-start;
        float sec = usec/1000000;
        float flops_per_usec = (1.0f*flops_per_matrix)/usec;
        printf("%9i;%8i;%6i;%6i;%6i;%15lli;%18lli;%19.2f\n",
            i,
            gf31.n_threads,
            sizex, sizey, sizez, flops_per_matrix,
            usec,flops_per_usec);

#ifdef VERBOSE_DEBUGGING
        TENSOR_DUMP("res",gf31.nodes[0])
#endif

        // Check that the matrix multiplication result is in the right ballpark
        // We cannot use the exact value from the F32 multiplication because the quantizuation will be slightly different
        float sum_of_Q4_result = tensor_sum_elements(gf31.nodes[0]);
        float delta = abs(sum_of_Q4_result - sum_of_F32_reference);
        float allowed_delta = (sum_of_F32_reference) / 1000 / 1000; //  Let's accept an epsilon of 10^-6

        if (delta > allowed_delta)  {
            printf("\nABORT - ERROR in Matrix Multiplication result - expected %6.2f, got %6.2f (delta %6.2f > allowed_delta %6.2f)\n",
                sum_of_F32_reference,
                sum_of_Q4_result,
                delta,
                allowed_delta
            );
            exit(0);
        }

        // Running a different graph computation to make sure we override the CPU cache lines
        ggml_graph_compute(ctx, &gf32);

    }

}
