#include "ggml-mpi.h"

#include "ggml.h"

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

#define UNUSED GGML_UNUSED

struct ggml_mpi_tensor_info {
    int rank;
};

// ggml_compute_forward_send

static void ggml_mpi_compute_forward_send(
        struct ggml_tensor * src,
        const struct ggml_tensor * orig) {
    UNUSED(orig);
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int dst_rank = ((struct ggml_mpi_tensor_info *)src->extra)->rank;
    // fprintf(stderr, "(%d) Sending to (%d)\n", my_rank, (int)dst->extra);
    int retval = MPI_Send(src->data, ggml_nelements(src), MPI_FLOAT, dst_rank, 0, MPI_COMM_WORLD);
    // fprintf(stderr, "(%d) Sent to (%d)\n", my_rank, (int)dst->extra);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

// ggml_compute_forward_recv

static void ggml_mpi_compute_forward_recv(
        struct ggml_tensor * dst,
        const struct ggml_tensor * orig,
        const struct ggml_tensor * parent) {
    UNUSED(parent);
    UNUSED(orig);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    MPI_Status status;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int src_rank = ((struct ggml_mpi_tensor_info *)dst->extra)->rank;
    // fprintf(stderr, "(%d) Receiving from (%d)\n", my_rank, src_extra);
    int retval = MPI_Recv(dst->data, ggml_nelements(dst), MPI_FLOAT, src_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    // fprintf(stderr, "(%d) Received from (%d)\n", my_rank, src_extra);
    GGML_ASSERT(retval == MPI_SUCCESS);
}

struct ggml_tensor * ggml_mpi_send_tensor(
        struct ggml_context * ctx,
         struct ggml_tensor * src,
                        int   dst_rank) {
    struct ggml_tensor * result = ggml_map_custom1_inplace_f32(ctx, src, ggml_mpi_compute_forward_send);

    // TODO how/when to free this struct?
    struct ggml_mpi_tensor_info *info = calloc(1, sizeof(struct ggml_mpi_tensor_info));
    info->rank = dst_rank;
    result->extra = info;

    return result;
}

struct ggml_tensor * ggml_mpi_recv_tensor(
        struct ggml_context * ctx,
         struct ggml_tensor * parent,
         struct ggml_tensor * dst,
                        int   src_rank) {
    struct ggml_tensor * result = ggml_map_custom2_inplace_f32(ctx, dst, parent, ggml_mpi_compute_forward_recv);

    // TODO how/when to free this struct?
    struct ggml_mpi_tensor_info *info = calloc(1, sizeof(struct ggml_mpi_tensor_info));
    info->rank = src_rank;
    result->extra = info;

    return result;
}

struct ggml_mpi_context {
    int mpi_rank;
    int mpi_size;
};

void ggml_mpi_backend_init(void) {
    MPI_Init(NULL, NULL);
}

void ggml_mpi_backend_free(void) {
    MPI_Finalize();
}

struct ggml_mpi_context * ggml_mpi_init(void) {
    struct ggml_mpi_context * ctx = calloc(1, sizeof(struct ggml_mpi_context));

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->mpi_size);

    return ctx;
}

void ggml_mpi_free(struct ggml_mpi_context * ctx) {
    free(ctx);
}

int ggml_mpi_rank(struct ggml_mpi_context * ctx) {
    return ctx->mpi_rank;
}

struct ggml_tensor * ggml_mpi_eval_init(
        struct ggml_mpi_context * ctx_mpi,
        struct ggml_context     * ctx,
                            int   n_embd,
                            int * n_tokens,
                            int * n_past,
                            int * n_threads) {
    struct ggml_tensor * res = NULL;

    // synchronize the worker node parameters with the root node
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(n_tokens,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_past,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (ctx_mpi->mpi_rank > 0) {
        res = ggml_mpi_recv_tensor(ctx, NULL,
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, *n_tokens), ctx_mpi->mpi_rank - 1);
        ggml_set_name(res, "mpi_recv");
    }

    return res;
}

void ggml_mpi_graph_compute(
        struct ggml_mpi_context * ctx_mpi,
             struct ggml_cgraph * gf,
                            int   n_layers,
                            int   n_embd,
                            int   n_tokens) {

}
