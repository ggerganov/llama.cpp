#include "ggml.h"
#include "ggml-mtl.h"

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

    int32_t n_vocab = 0;

    {
        struct ggml_tensor * t_vocab = ggml_graph_get_tensor(&gf, "vocab");
        if (t_vocab == NULL) {
            fprintf(stderr, "%s: vocab tensor not found\n", __func__);
            return -1;
        }

        const char * ptr = (const char *) t_vocab->data;

        memcpy(&n_vocab, ptr, sizeof(n_vocab)); ptr += sizeof(n_vocab);

        printf("%s: n_vocab = %d\n", __func__, n_vocab);

        for (int i = 0; i < 512; ++i) {
            char text[32];
            float score;

            memcpy(text,   ptr, sizeof(text));  ptr += sizeof(text);
            memcpy(&score, ptr, sizeof(score)); ptr += sizeof(score);

            printf("%s: token[%4d] = %16.*s, score = %6.2f\n", __func__, i, (int) sizeof(text), text, score);
        }
    }

    // this allocates all Metal resources and memory buffers
    auto * ctx_mtl = ggml_mtl_init(
            ggml_get_mem_buffer(ctx_data),
            ggml_get_mem_size  (ctx_data),
            ggml_get_mem_buffer(ctx_eval),
            ggml_get_mem_size  (ctx_eval),
            NULL, 0, // cache
            32*n_vocab*sizeof(float));

    // TODO: tmp to match the input used when creating the cgraph
    {
        const int n_batch = 1;
        const int n_past  = 512 - n_batch;

        const std::vector<int> tmp(n_batch, 1); // BOS

        // warmup
        ggml_mtl_graph_compute(ctx_mtl, &gf, tmp.data(), tmp.size(), n_past);

        const int n_iter = 16;

        const int64_t t0 = ggml_time_us();

        // the actual inference happens here
        for (int i = 0; i < n_iter; ++i) {
            ggml_mtl_graph_compute(ctx_mtl, &gf, tmp.data(), tmp.size(), n_past);
        }

        const int64_t t1 = ggml_time_us();

        printf("time: %.2f ms, %.2f ms/tok\n", (t1 - t0) / 1000.0, (t1 - t0) / 1000.0 / n_iter);
    }

    ggml_mtl_free(ctx_mtl);

    ggml_free(ctx_data);
    ggml_free(ctx_eval);

    return 0;
}

