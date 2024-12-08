#include "common.h"
#include "llama.h"
#include "ggml.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#define DEBUG_POS 5

static void print_debug_tensor(struct ggml_tensor * t, bool with_data = true) {
    printf("%s: %s (%s): [%d, %d]\n", __func__, t->name, ggml_type_name(t->type), (int) t->ne[0], (int) t->ne[1]);
    if (!with_data) return;
    printf("%s: %s[0] = [", __func__, t->name);
    for (size_t i = 0; i <= DEBUG_POS; i++) {
        printf(" %f,", ggml_get_f32_nd(t, i, 0, 0, 0));
    }
    printf(" ... ]\n");
}

// begin vanilla pca namespace
namespace PCA {

// input params for PCA computations
struct pca_params {
    int n_threads    = 1;
    int n_batch      = 20; // number of iterations do to in one batch. larger the batch, more memory is used
    int n_iterations = 1000;
    float tolerance  = 1e-7;
};

// result from each iteration
struct pca_result {
    float * principal_component; // eigenvectors of the covariance matrix
    float explained_variance;                 // eigenvalues of the covariance matrix
};

static void compute_covariance(struct pca_params &pca_params,
                        struct ggml_tensor * X,
                        float * covariance,
                        struct ggml_backend * backend) {

    size_t ctx_size = 0;
    ctx_size += 7 * ggml_tensor_overhead();
    ctx_size += ggml_graph_overhead();
    ctx_size += 1024;

    // Memory allocation
    struct ggml_cgraph  * gf  = NULL;
    struct ggml_context * ctx = NULL;
    struct ggml_init_params ctx_params = {
        ctx_size,
        NULL,
        true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    ctx = ggml_init(ctx_params);
    gf  = ggml_new_graph(ctx);

    // Step 0: Transpose the input because of row-major
    X = ggml_cont(ctx, ggml_transpose(ctx, X));

    // Step 1: Compute the mean for each feature
    struct ggml_tensor * mean           = ggml_repeat(ctx, ggml_mean(ctx, X), X); // mean with trick to make it easier to sub
    struct ggml_tensor * centered_data  = ggml_sub(ctx, X, mean);

    // Step 2: Compute the covariance matrix
    struct ggml_tensor * cov            = ggml_mul_mat(ctx, centered_data, centered_data); // C = X * X^T
    cov                                 = ggml_scale(ctx, cov, 1.0/(X->ne[0]-1));
    ggml_build_forward_expand(gf, cov);

    // Step 3: Create ggml_gallocr for graph computation
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Step 4: Check if CPU and compute the result of the graph
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, pca_params.n_threads);
    }
    ggml_backend_graph_compute(backend, gf);

    // Step 5: Store covariance matrix in the data pointer
    struct ggml_tensor * result = ggml_graph_node(gf, ggml_graph_n_nodes(gf)-1);
    ggml_backend_tensor_get(result, covariance, 0, ggml_nbytes(result));

    // Step 6: Free memory
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}

static void compute_cross_covariance(struct pca_params &pca_params,
                              struct ggml_tensor * A,
                              struct ggml_tensor * B,
                              float * cross_covariance,
                              struct ggml_backend * backend) {

    size_t ctx_size = 0;
    ctx_size += 9 * ggml_tensor_overhead();
    ctx_size += ggml_graph_overhead();
    ctx_size += 1024;

    // Memory allocation
    struct ggml_cgraph  * gf  = NULL;
    struct ggml_context * ctx = NULL;
    struct ggml_init_params ctx_params = {
        ctx_size,
        NULL,
        true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    ctx = ggml_init(ctx_params);
    gf  = ggml_new_graph(ctx);

    // Step 1: Compute matrices of cross_covariance
    struct ggml_tensor * AT     = ggml_cont(ctx, ggml_transpose(ctx, A));
    struct ggml_tensor * BT     = ggml_cont(ctx, ggml_transpose(ctx, B));
    struct ggml_tensor * AT_B   = ggml_mul_mat(ctx, AT, BT);
    struct ggml_tensor * BT_A   = ggml_cont(ctx, ggml_transpose(ctx, AT_B));

    // Step 2: Compute the covariance matrix
    struct ggml_tensor * cross_cov      = ggml_add(ctx, AT_B, BT_A);
    cross_cov                           = ggml_scale(ctx, cross_cov, 0.5);
    ggml_build_forward_expand(gf, cross_cov);

    // Step 3: Create ggml_gallocr for graph computation
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // Step 4: Check if CPU and compute the result of the graph
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, pca_params.n_threads);
    }
    ggml_backend_graph_compute(backend, gf);

    // Step 5: Store covariance matrix in the data pointer
    struct ggml_tensor * result = ggml_graph_node(gf, ggml_graph_n_nodes(gf)-1);
    ggml_backend_tensor_get(result, cross_covariance, 0, ggml_nbytes(result));

    // Step 6: Free memory
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
}

// Find the dominant eigenvector of tensor M
static void power_iteration(struct pca_params &pca_params,
                     struct ggml_tensor * M,
                     struct pca_result &result,
                     struct ggml_backend * backend) {

    int m = M->ne[1];

    // Initialize random vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float * b = result.principal_component;
    for (int i = 0; i < m; i++) {
        b[i] = dist(gen);
    };
    float eigenvalue = 0;

    // Iterate
    int n_rounds = pca_params.n_iterations / pca_params.n_batch;
    for(int i = 0; i < n_rounds; i++) {

        // Memory allocation
        struct ggml_cgraph  * gf  = NULL;
        struct ggml_context * ctx = NULL;
        struct ggml_init_params params = {
            ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
            NULL,
            true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx = ggml_init(params);
        gf  = ggml_new_graph(ctx);

        // Fill current eigen vector
        struct ggml_tensor * e_curr = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m);
        struct ggml_tensor * e_prev = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, m);

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

        ggml_backend_tensor_set(e_curr, b, 0, ggml_nbytes(e_curr));
        ggml_backend_tensor_set(e_prev, b, 0, ggml_nbytes(e_curr));

        struct ggml_tensor * e_next   = NULL;
        struct ggml_tensor * e_norm   = NULL;
        for(int j = 0; j < pca_params.n_batch; j++) {
            // Compute next candidate vector multiplying M with the current vector
            e_next = ggml_mul_mat(ctx, M, e_curr);

            // Compute the norm of the new vector (and normalize it)
            // this will give us the next eigenvector and eigenvalue
            e_norm = ggml_sqrt_inplace(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, e_next)));
            e_curr = ggml_div_inplace(ctx, e_next, e_norm);
            ggml_format_name(e_norm, "eigenvalue_%d", j);
            ggml_format_name(e_curr, "eigenvector_%d", j);

            // Update graph
            ggml_build_forward_expand(gf, e_curr);
        }

        // Compute the similarity between the current eigenvector and the previous (dot product)
        struct ggml_tensor * similarity = ggml_mul_mat(ctx, e_curr, e_prev);
        ggml_build_forward_expand(gf, similarity);

        // Create ggml_gallocr for graph computation
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(allocr, gf);

        // Check if CPU and compute the result of the graph
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, pca_params.n_threads);
        }
        ggml_status graph_status = ggml_backend_graph_compute(backend, gf);

        // Get graph results (eigenvector and eigenvalue) and store it in b and eigenvalue
        if(graph_status == GGML_STATUS_SUCCESS){

            // Similarity is the last node in the graph
            struct ggml_tensor * similarity_tensor = ggml_graph_node(gf, ggml_graph_n_nodes(gf)-1);
            float similarity = (float)((float*) similarity_tensor->data)[0];

            // Eigenvector is the second last node in the graph
            // struct ggml_tensor * eigenvector_tensor = gf->nodes[gf->n_nodes-2];
            struct ggml_tensor * eigenvector_tensor = ggml_graph_node(gf,ggml_graph_n_nodes(gf)-2);
            ggml_backend_tensor_get(eigenvector_tensor, b, 0, ggml_nbytes(eigenvector_tensor));

            // Eigenvalue computation is 1 operation before eigenvector computation
            // struct ggml_tensor * eigenvalue_tensor = gf->nodes[gf->n_nodes-3];
            struct ggml_tensor * eigenvalue_tensor = ggml_graph_node(gf, ggml_graph_n_nodes(gf)-3);
            eigenvalue = (float)((float*) eigenvalue_tensor->data)[0];

            // Check if the similarity is close enough to 1, if so we converged and should break
            if(1 - similarity < pca_params.tolerance)
                break;
        }

        // Free memory
        ggml_backend_buffer_free(buffer);
        ggml_gallocr_free(allocr);
        ggml_free(ctx);
    }

    // Store result
    result.principal_component = b;
    result.explained_variance = eigenvalue;
    return;
}

static void run_single_pca(struct pca_params &pca_params,
             struct ggml_tensor * X,
             struct pca_result &result
             ) {

    ggml_set_name(X, "input_tensor");

    int m = X->ne[1]; // Number of features

    // Step 1. Initialize GGML Backend
    ggml_backend_t backend = NULL;
    #ifdef GGML_USE_CUDA
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        backend = ggml_backend_cuda_init(0); // init device 0
        if (!backend) { fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__); }
    #endif

    // If there aren't GPU Backends fallback to CPU backend
    if (!backend) { backend = ggml_backend_cpu_init(); }

    // Compute the context size needed
    size_t ctx_size = 0;
    ctx_size += 1 * ggml_tensor_overhead();

    // Step 2. Initialize GGML Context
    struct ggml_init_params ctx_params {
        ctx_size,  // mem_size
        NULL,      // mem_buffer
        true,      // no_alloc
    };
    struct ggml_context * ctx = ggml_init(ctx_params);

    // Step 3. Compute the data covariance matrix
    // Using a CPU buffer to copy data from the backend
    float * covariance = (float *) malloc(m * m * sizeof(float));
    compute_covariance(pca_params, X, covariance, backend);

    // Create covariance tensor on backend
    struct ggml_tensor * covariance_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, m);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(covariance_tensor, covariance, 0, ggml_nbytes(covariance_tensor));

    // Step 4. Power iteration
    result.principal_component = (float *) malloc(m * sizeof(float));
    power_iteration(pca_params, covariance_tensor, result, backend);

    // Step 5. Free ggml ctx and backend
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}

static void run_pca(
        struct pca_params & params,
        const std::vector<struct ggml_tensor *> & v_input, // shape of v_input[0]: [n_samples, n_embd]
        const std::vector<struct ggml_tensor *> & v_output) {

    for (size_t i = 0; i < v_input.size(); i++) {
        // Check shape of tensor inside v_output
        GGML_ASSERT(v_output[i]->ne[0] == v_input[i]->ne[1]);
        struct pca_result result = {NULL, 0};
        run_single_pca(params, v_input[i], result);
        ggml_backend_tensor_set(v_output[i], result.principal_component, 0, ggml_nbytes(v_output[i]));
        free(result.principal_component);
    }
}

// end namesace
}
