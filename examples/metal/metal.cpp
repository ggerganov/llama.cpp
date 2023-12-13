// Evaluate a statically exported ggml computation graph with Metal
//
// - First, export a LLaMA graph:
//
//  $ ./bin/main -m ../models/7B/ggml-model-q4_0.gguf --export
//
// - Run this tool to evaluate the exported graph:
//
//  $ ./bin/metal llama.ggml
//
// The purpose of this tool is mostly for debugging and demonstration purposes.
// The main limitation of exporting computation graphs is that their sizes are static which often
// can be a problem for real-world applications.
//

#include "ggml.h"
#include "ggml-metal.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

int main(int argc, char ** argv) {
    ggml_time_init();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s llama.ggml\n", argv[0]);
        return -1;
    }

    const char * fname_cgraph = argv[1];

    // load the compute graph
    struct ggml_context * ctx_data = NULL;
    struct ggml_context * ctx_eval = NULL;

    struct ggml_cgraph * gf = ggml_graph_import(fname_cgraph, &ctx_data, &ctx_eval);

    // this allocates all Metal resources and memory buffers
    auto * ctx_metal = ggml_metal_init(1);

    const size_t max_size_data = ggml_get_max_tensor_size(ctx_data);
    const size_t max_size_eval = ggml_get_max_tensor_size(ctx_eval);
    ggml_metal_add_buffer(ctx_metal, "data", ggml_get_mem_buffer(ctx_data), ggml_get_mem_size(ctx_data), max_size_data);
    ggml_metal_add_buffer(ctx_metal, "eval", ggml_get_mem_buffer(ctx_eval), ggml_get_mem_size(ctx_eval), max_size_eval);

    // main
    {
        struct ggml_tensor * input = ggml_graph_get_tensor(gf, "embd");
        *(int32_t *) input->data = 1; // BOS

        ggml_metal_set_tensor(ctx_metal, input);

        // warmup
        ggml_metal_graph_compute(ctx_metal, gf);

        const int n_iter = 16;

        const int64_t t0 = ggml_time_us();

        // the actual inference happens here
        for (int i = 0; i < n_iter; ++i) {
            ggml_metal_graph_compute(ctx_metal, gf);
        }

        const int64_t t1 = ggml_time_us();

        printf("time: %.2f ms, %.2f ms/tok\n", (t1 - t0) / 1000.0, (t1 - t0) / 1000.0 / n_iter);
    }

    // debug output
    {
        struct ggml_tensor * logits = gf->nodes[gf->n_nodes - 1];
        ggml_metal_get_tensor(ctx_metal, logits);

        float * ptr = (float *) ggml_get_data(logits);

        printf("logits: ");
        for (int i = 0; i < 10; i++) {
            printf("%8.4f ", ptr[i]);
        }
        printf("\n");
        int imax = 0;
        double sum = 0.0;
        double vmax = -1e9;
        for (int i = 0; i < 32000; i++) {
            sum += (double) ptr[i];
            if (ptr[i] > vmax) {
                vmax = ptr[i];
                imax = i;
            }
        }
        printf("sum: %f, imax = %d, vmax = %f\n", sum, imax, vmax);
    }

    ggml_metal_free(ctx_metal);

    ggml_free(ctx_data);
    ggml_free(ctx_eval);

    return 0;
}

