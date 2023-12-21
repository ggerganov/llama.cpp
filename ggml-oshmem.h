#pragma once
#ifndef __LLAMA_CPP_GGML_OSHMEM_H__
#define __LLAMA_CPP_GGML_OSHMEM_H__

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_openshmem_context;

void ggml_openshmem_backend_init(void);
void ggml_openshmem_backend_free(void);

struct ggml_openshmem_context * ggml_openshmem_init(void);
void ggml_openshmem_free(struct ggml_openshmem_context * ctx);

int ggml_openshmem_rank(struct ggml_openshmem_context * ctx);

void ggml_openshmem_eval_init(
        struct ggml_openshmem_context * ctx_openshmem,
                            int * n_tokens,
                            int * n_past,
                            int * n_threads);

void ggml_openshmem_graph_compute_pre(
        struct ggml_openshmem_context * ctx_openshmem,
             struct ggml_cgraph * gf,
                            int   n_layers);

void ggml_openshmem_graph_compute_post(
        struct ggml_openshmem_context * ctx_openshmem,
             struct ggml_cgraph * gf,
                            int   n_layers);

#ifdef __cplusplus
}
#endif

#endif
