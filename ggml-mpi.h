#pragma once

struct ggml_context;
struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor * ggml_mpi_send_tensor(
        struct ggml_context * ctx,
        struct ggml_tensor *src,
        int dst_rank);
struct ggml_tensor * ggml_mpi_recv_tensor(
        struct ggml_context * ctx,
        struct ggml_tensor *parent,
        struct ggml_tensor *dst,
        int src_rank);

#ifdef __cplusplus
}
#endif
