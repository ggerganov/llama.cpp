#include "common.h"
#include "llama.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdio>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#define DEBUG_POS 5

static void print_debug_tensor(struct ggml_tensor * t) {
    printf("%s: %s (%s): [%ld, %ld]\n", __func__, t->name, ggml_type_name(t->type), t->ne[0], t->ne[1]);
    printf("%s: %s[0] = [", __func__, t->name);
    for (size_t i = 0; i <= DEBUG_POS; i++) {
        printf(" %f,", ggml_get_f32_nd(t, i, 0, 0, 0));
    }
    printf(" ... ]\n");
}

namespace PCA {

struct pca_model {
    struct ggml_tensor * v_diff_original;
    struct ggml_tensor * square;
    struct ggml_tensor * eigenvector;

    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_pca_model(pca_model & model, struct ggml_tensor * input) {
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
    model.backend = ggml_backend_metal_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }

    const int num_tensors = 4;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    model.ctx = ggml_init(params);

    auto n_embd    = input->ne[0];
    auto n_samples = input->ne[1];

    model.v_diff_original = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_embd, n_samples);
    model.square          = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_embd, n_embd);
    model.eigenvector     = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, n_embd);

    ggml_set_name(model.v_diff_original, "v_diff_original");
    ggml_set_name(model.square,          "square");
    ggml_set_name(model.eigenvector,     "eigenvector");

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    ggml_backend_tensor_set(model.v_diff_original, input->data, 0, ggml_nbytes(input));

    // initialize model.eigenvector to random vector
    std::vector<float> random_vec;
    std::default_random_engine generator(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < ggml_nelements(model.eigenvector); ++i) {
        random_vec.push_back(distribution(generator));
    }

    // we don't normalize it at first but that shouldn't be a problem
    ggml_backend_tensor_set(model.eigenvector, random_vec.data(), 0, ggml_nbytes(model.eigenvector));
}

static struct ggml_cgraph * build_graph_piter(
        const pca_model & model,
        bool calc_square = false,
        int nb_iterations = 1) {
    GGML_ASSERT(nb_iterations > 0);
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };
    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // turn v_diff_original into square matrix if needed
    if (calc_square) {
        //struct ggml_tensor * v_diff_transposed = ggml_transpose(ctx0, model.v_diff_original);
        struct ggml_tensor * square = ggml_mul_mat(ctx0, model.v_diff_original, model.v_diff_original);
        ggml_set_name(square, "square");
        //model.square = ggml_scale_inplace(ctx0, model.square, 0.0);
    }

    struct ggml_tensor * b_tensor;

    for (int i = 0; i < nb_iterations; ++i) {
        // b_tensor = square * eigenvector^T
        b_tensor = ggml_mul_mat(ctx0, model.square, model.eigenvector);
        ggml_set_name(b_tensor, "b_tensor");

        // normalize
        b_tensor = ggml_div_inplace(ctx0,
            b_tensor,
            ggml_sqrt_inplace(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, b_tensor)))
        );
    }

    // calculate distance
    struct ggml_tensor * distance;
    {
        distance = ggml_sub(ctx0, model.eigenvector, b_tensor);
        ggml_set_name(distance, "distance");
        distance = ggml_sqrt_inplace(ctx0,
            ggml_sum_rows(ctx0, ggml_sqr_inplace(ctx0, distance)));
    }

    // build operations nodes
    ggml_build_forward_expand(gf, distance);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor * compute_piter(
        const pca_model & model,
        struct ggml_cgraph * gf,
        ggml_gallocr_t allocr,
        int n_threads) {
    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

    ggml_backend_graph_compute(model.backend, gf);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}

static void power_iteration(
        struct ggml_tensor * input,
        struct ggml_tensor * output,
        int n_threads,
        int maxIterations = 1000,
        float tolerance = 1e-7) {
    printf("in power iteration\n");
    int n_embd = input->ne[0]; // shape of input: [n_embd, m]

    pca_model model;
    load_pca_model(model, input);

    ggml_gallocr_t allocr = NULL;

    struct ggml_init_params host_params = {
        /*.mem_size   =*/ (n_embd * sizeof(float) + ggml_tensor_overhead()) * 4u,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * host_ctx = ggml_init(host_params);

    struct ggml_tensor * host_old_eigenvector = ggml_new_tensor_1d(host_ctx, GGML_TYPE_F32, n_embd);
    struct ggml_tensor * host_new_eigenvector = ggml_new_tensor_1d(host_ctx, GGML_TYPE_F32, n_embd);

    for (int iter = 0; iter < maxIterations; ++iter) {
        if (allocr) {
            ggml_gallocr_free(allocr);
        }
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        struct ggml_cgraph * gf = build_graph_piter(model, iter == 0);
        printf("kkk\n");
        ggml_graph_dump_dot(gf, nullptr, "/tmp/_cgraph.dot");
        struct ggml_tensor * distance = compute_piter(model, gf, allocr, n_threads);

        ggml_backend_tensor_get(distance, host_new_eigenvector->data, 0, ggml_nbytes(distance));
        print_debug_tensor(host_new_eigenvector);
        
        break; // FIXME
    }

    ggml_backend_tensor_get(model.eigenvector, output->data, 0, ggml_nbytes(model.eigenvector));

    ggml_gallocr_free(allocr);
    ggml_free(host_ctx);
    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    exit(0);
}

static void run_pca(
        const std::vector<struct ggml_tensor *> & v_input,
        const std::vector<struct ggml_tensor *> & v_output) {
    printf("Running PCA...\n");
    int n_embd = v_input[0]->ne[0]; // shape of v_input[0]: [n_embd, m]
    int n_threads = 8; // TODO: change me
    for (size_t il = 0; il < v_input.size(); ++il) {
        // prepare output vector
        struct ggml_tensor * ctrl_out = v_output[il];
        auto name = std::string("direction.") + std::to_string(il + 1);
        ggml_set_name(ctrl_out, name.c_str());
        // run power_iteration
        power_iteration(v_input[il], ctrl_out, n_threads);
        printf("Done with layer %d\n", il);
        print_debug_tensor(ctrl_out);
    }
    printf("Done with PCA.\n");
}

}
