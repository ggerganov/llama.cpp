#include "common.h"
#include "llama.h"

int main(int argc, char ** argv) {
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    llama_init_backend();

    llama_context * ctx = llama_init_from_gpt_params(params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    llama_eval_export(ctx, "llama.ggml");

    llama_print_timings(ctx);
    llama_free(ctx);

    return 0;
}
