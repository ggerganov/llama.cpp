#include "ggml-mpi.h"

#include "ggml.h"

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED GGML_UNUSED

struct ggml_mpi_context {
    int rank;
    int size;
};

void ggml_mpi_backend_init(void) {
    MPI_Init(NULL, NULL);
}

void ggml_mpi_backend_free(void) {
    MPI_Finalize();
}

struct ggml_mpi_context * ggml_mpi_init(void) {
    struct ggml_mpi_context * ctx = calloc(1, sizeof(struct ggml_mpi_context));

    MPI_Comm_rank(MPI_COMM_WORLD, &ctx->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx->size);

    return ctx;
}

void ggml_mpi_free(struct ggml_mpi_context * ctx) {
    free(ctx);
}

int ggml_mpi_rank(struct ggml_mpi_context * ctx) {
    return ctx->rank;
}

void ggml_mpi_eval_init(
        struct ggml_mpi_context * ctx_mpi,
                            int * n_tokens,
                            int * n_past,
                            int * n_threads) {
    UNUSED(ctx_mpi);

    // synchronize the worker node parameters with the root node
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(n_tokens,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_past,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int ggml_graph_get_node_idx( struct ggml_cgraph * gf, const char * name) {
    struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (t == NULL) {
        fprintf(stderr, "%s: tensor %s not found\n", __func__, name);
        return -1;
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i] == t) {
            return i;
        }
    }

    fprintf(stderr, "%s: tensor %s not found in graph (should not happen)\n", __func__, name);
    return -1;
}

void ggml_mpi_graph_compute(
        struct ggml_mpi_context * ctx_mpi,
        struct ggml_context     * ctx,
             struct ggml_cgraph * gf,
                            int   n_layers) {
    const int mpi_rank = ctx_mpi->rank;
    const int mpi_size = ctx_mpi->size;

    struct ggml_tensor * embd = ggml_graph_get_tensor(gf, "layer_inp_0");
    if (embd == NULL) {
        fprintf(stderr, "%s: tensor 'embd' not found\n", __func__);
        return;
    }

    GGML_ASSERT(embd == gf->nodes[0]);

    // distribute the compute graph into slices across the MPI nodes
    //
    // the main node (0) processes the last layers + the remainder of the compute graph
    // and is responsible to pass the input embeddings to the first node (1)
    //
    // node 1:   [(  0) * n_per_node, (  1) * n_per_node)
    // node 2:   [(  1) * n_per_node, (  2) * n_per_node)
    // ...
    // node n-1: [(n-2) * n_per_node, (n-1) * n_per_node)
    // node 0:   [(n-1) * n_per_node,            n_nodes)
    //
    if (mpi_rank > 0) {
        // recv input data for each node into the "embd" tensor (i.e. the first node in the compute graph)
        {
            MPI_Status status; UNUSED(status);

            const int mpi_rank_src = mpi_rank - 1;

            //printf("%s: node %d: waiting for %d elements from %d\n", __func__, mpi_rank, (int) ggml_nelements(embd), mpi_rank_src);
            const int retval = MPI_Recv(embd->data, ggml_nelements(embd), MPI_FLOAT, mpi_rank_src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            GGML_ASSERT(retval == MPI_SUCCESS);
        }
    } else {
        // node 0 sends the input data to node 1
        {
            const int mpi_rank_dst = mpi_rank + 1;

            const int retval = MPI_Send(embd->data, ggml_nelements(embd), MPI_FLOAT, mpi_rank_dst, 0, MPI_COMM_WORLD);
            GGML_ASSERT(retval == MPI_SUCCESS);
        }

        // recv the output data from the last node
        {
            MPI_Status status; UNUSED(status);

            const int mpi_rank_src = mpi_size - 1;

            //fprintf(stderr, "%s: node %d: waiting for %d elements from %d\n", __func__, mpi_rank, (int) ggml_nelements(embd), mpi_rank_src);
            const int retval = MPI_Recv(embd->data, ggml_nelements(embd), MPI_FLOAT, mpi_rank_src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            GGML_ASSERT(retval == MPI_SUCCESS);
        }
    }

    {
        const int n_per_node = (n_layers + (mpi_size - 1)) / mpi_size;

        const int mpi_idx = mpi_rank > 0 ? mpi_rank - 1 : mpi_size - 1;

        const int il0 =               (mpi_idx + 0) * n_per_node;
        const int il1 = MIN(n_layers, (mpi_idx + 1) * n_per_node);

        char name_l0[64];
        char name_l1[64];

        snprintf(name_l0, sizeof(name_l0), "layer_inp_%d", il0);
        snprintf(name_l1, sizeof(name_l1), "layer_inp_%d", il1);

        const int idx_l0 =                ggml_graph_get_node_idx(gf, name_l0);
        const int idx_l1 = mpi_rank > 0 ? ggml_graph_get_node_idx(gf, name_l1) : gf->n_nodes;

        if (idx_l0 < 0 || idx_l1 < 0) {
            fprintf(stderr, "%s: layer input nodes not found\n", __func__);
            return;
        }

        // attach the input data to the first layer for this node
        gf->nodes[idx_l0 + 1]->src0 = gf->nodes[1]->src0;
        gf->nodes[idx_l0 + 1]->src1 = gf->nodes[1]->src1;

        memcpy(gf->nodes[idx_l0 + 1]->opt, gf->nodes[1]->opt, sizeof(gf->nodes[idx_l0 + 1]->opt));

        for (int i = 1; i < idx_l1 - idx_l0; i++) {
            gf->nodes[i] = gf->nodes[idx_l0 + i];
            gf->grads[i] = gf->grads[idx_l0 + i];

            //fprintf(stderr, "%s: node %d: %d -> %d\n", __func__, mpi_rank, idx_l0 + i, i);
        }

        gf->n_nodes = idx_l1 - idx_l0;

        //fprintf(stderr, "%s: node %d: processing %d nodes [%d, %d)\n", __func__, mpi_rank, gf->n_nodes, il0, il1);
    }

    ggml_graph_compute(ctx, gf);

    //fprintf(stderr, "%s: node %d: done\n", __func__, mpi_rank);

    // send the output data to the next node
    if (mpi_rank > 0) {
        struct ggml_tensor * output = gf->nodes[gf->n_nodes - 1];

        const int mpi_rank_dst = (mpi_rank + 1) % mpi_size;

        //fprintf(stderr, "%s: node %d: sending %d elements to node %d\n", __func__, mpi_rank, ggml_nelements(output), mpi_rank_dst);

        const int retval = MPI_Send(output->data, ggml_nelements(output), MPI_FLOAT, mpi_rank_dst, 0, MPI_COMM_WORLD);
        GGML_ASSERT(retval == MPI_SUCCESS);
    }
}
