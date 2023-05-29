#pragma once

struct ggml_context;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_mtl_context;

struct ggml_mtl_context * llama_mtl_init(
        struct ggml_context * ctx_data,
        struct ggml_context * ctx_eval,
        struct ggml_context * ctx_work,
        struct ggml_cgraph  * gf);

void llama_mtl_free(struct ggml_mtl_context * ctx);

// return 0 on success
int llama_mtl_eval(
        struct ggml_mtl_context * ctx,
        struct ggml_cgraph      * gf);

#ifdef __cplusplus
}
#endif

