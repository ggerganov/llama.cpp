#pragma once

#include <stddef.h>

struct ggml_context;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_mtl_context;

struct ggml_mtl_context * llama_mtl_init(
        void   * data_buf,
        size_t   data_size,
        void   * eval_buf,
        size_t   eval_size,
        void   * cach_buf,
        size_t   cach_size,
        size_t   outp_size);

void llama_mtl_free(struct ggml_mtl_context * ctx);

// return 0 on success
int llama_mtl_eval(
        struct ggml_mtl_context * ctx,
             struct ggml_cgraph * gf,
                      const int * tokens,
                            int   n_tokens,
                            int   n_past);

float * llama_mtl_get_logits(struct ggml_mtl_context * ctx);

#ifdef __cplusplus
}
#endif

