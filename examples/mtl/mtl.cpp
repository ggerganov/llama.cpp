#include "ggml.h"
#include "mtl.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <vector> // tmp

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

    struct ggml_cgraph gf = ggml_graph_import(fname_cgraph, &ctx_data, &ctx_eval);
    gf.n_threads = 1;

    // allocate work context
    static size_t buf_size = gf.work_size; // TODO
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_work = ggml_init(params);

    // this allocates all Metal resources and memory buffers
    auto * ctx_mtl = llama_mtl_init(ctx_data, ctx_eval, ctx_work, &gf);

    // TODO: tmp to match the input used when creating the cgraph
    {
        const int n_ctx   = 128;
        const int n_batch = 32;

        const std::vector<int> tmp(n_batch, 1); // BOS

        struct ggml_tensor * input = ggml_graph_get_tensor(&gf, "embd");
        memcpy(input->data, tmp.data(), tmp.size() * sizeof(int));
    }

    // the actual inference happens here
    llama_mtl_eval(ctx_mtl, &gf);

    llama_mtl_free(ctx_mtl);

    ggml_free(ctx_work);
    ggml_free(ctx_data);
    ggml_free(ctx_eval);

    return 0;
}

