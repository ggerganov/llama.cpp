#pragma once

struct ggml_context;
struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor * ggml_mpi_send_tensor(
        struct ggml_context * ctx,
         struct ggml_tensor * src,
                        int   dst_rank);
struct ggml_tensor * ggml_mpi_recv_tensor(
        struct ggml_context * ctx,
         struct ggml_tensor * parent,
         struct ggml_tensor * dst,
                        int   src_rank);

struct ggml_mpi_context;

void ggml_mpi_backend_init(void);
void ggml_mpi_backend_free(void);

struct ggml_mpi_context * ggml_mpi_init(void);
void ggml_mpi_free(struct ggml_mpi_context * ctx);

int ggml_mpi_rank(struct ggml_mpi_context * ctx);

struct ggml_tensor * ggml_mpi_eval_init(
        struct ggml_mpi_context * ctx_mpi,
        struct ggml_context     * ctx,
                            int   n_embd,
                            int * n_tokens,
                            int * n_past,
                            int * n_threads);

#ifdef __cplusplus
}
#endif
