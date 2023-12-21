#include "ggml-oshmem.h"

#include "ggml.h"

#include <shmem.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define UNUSED GGML_UNUSED

#define OPENSHMEM_SYMMETRIC_BUFFER_SIZE 4096

struct ggml_openshmem_context {
    int pe;
    int n_pes;
    int64_t symmetric_buffer_size;
    int64_t symmetric_comm_structure_size;
    uint8_t * symmetric_comm_structure;
    uint64_t * recv_signal;
};

void ggml_openshmem_backend_init(void) {
    shmem_init();
}

void ggml_openshmem_backend_free(void) {
    shmem_finalize();
}

struct ggml_openshmem_context * ggml_openshmem_init(void) {
    struct ggml_openshmem_context * ctx = calloc(1, sizeof(struct ggml_openshmem_context));

    ctx->pe = shmem_my_pe(); 
    ctx->n_pes = shmem_n_pes();

    /*
     * makes a symmetric heap allocation on all processing elements (processes running this SPMD program)
     *
     * below is a struct representing the layout of the symmetric allocation:
     *
     * {
     *     int64_t offset_in_buffer,
     *     int64_t length_in_buffer,
     *     uint8_t buffer[shmem_npes()][OPENSHMEM_SYMMETRIC_BUFFER_SIZE]
     * }
     *
     */
    ctx->symmetric_buffer_size = OPENSHMEM_SYMMETRIC_BUFFER_SIZE;
    ctx->symmetric_comm_structure_size = OPENSHMEM_SYMMETRIC_BUFFER_SIZE + sizeof(int64_t) + sizeof(int64_t);
    ctx->symmetric_comm_structure = (uint8_t*)shmem_calloc(1, ctx->n_pes*ctx->symmetric_comm_structure_size);

    /*
     * uint8_t signal_byte[shmem_npes()];
     */
    ctx->recv_signal = (uint64_t*)shmem_calloc(1, ctx->n_pes*sizeof(uint64_t));

    return ctx;
}

void ggml_openshmem_free(struct ggml_openshmem_context * ctx) {
    free(ctx);
}

int ggml_openshmem_pe(struct ggml_openshmem_context * ctx) {
    return ctx->pe;
}

void ggml_openshmem_eval_init(
        struct ggml_openshmem_context * ctx_openshmem,
        int * n_tokens,
        int * n_past,
        int * n_threads) {
    UNUSED(ctx_openshmem);

    // synchronize the worker node parameters with the root node
    shmem_barrier_all();

    shmem_broadcast(SHMEM_TEAM_WORLD, n_tokens, n_tokens, 1, 0);
    shmem_broadcast(SHMEM_TEAM_WORLD, n_past, n_tokens, 1, 0);
    shmem_broadcast(SHMEM_TEAM_WORLD, n_threads, n_tokens, 1, 0);

    shmem_quiet();
}

static int ggml_graph_get_node_idx(struct ggml_cgraph * gf, const char * name) {
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

/*
 * The OpenSHMEM mechanism used in this application reflects a message passing model; this is a byproduct of OpenSHMEM's symmetric memory requirements.
 * Care has been taken to limit the number of branches made in send/recv and the amount of two-sided communication. Memory consistency maybe an issue
 * which is why a `shmem_fence` is placed at the end of both send/recv.
 *
 */
static void ggml_openshmem_tensor_send(struct ggml_openshmem_context * ctx, struct ggml_tensor * t, int dst_pe) {

    const int64_t symmetric_comm_structure_size =
        ctx->symmetric_comm_structure_size;
    uint8_t * dst_symmetric_comm_structure =
        ((uint8_t*)ctx->symmetric_comm_structure)+(ctx->symmetric_comm_structure_size*ctx->pe);
    int64_t * dst_symmetric_comm_offset =
        (int64_t*)(dst_symmetric_comm_structure);
    int64_t * dst_symmetric_comm_length =
        ((int64_t*)dst_symmetric_comm_offset)+sizeof(int64_t);
    uint8_t * dst_symmetric_comm_buffer =
        ((uint8_t*)dst_symmetric_comm_length)+sizeof(int64_t);
    uint64_t * dst_recv_signal =
        ctx->recv_signal+dst_pe;
    uint64_t * my_recv_signal =
        ctx->recv_signal+ctx->pe;

    const int64_t nelements = ggml_nelements(t);
    int64_t xmt_size = 0;

    switch (t->type) {
        case GGML_TYPE_I32:
            xmt_size = nelements * sizeof(int32_t);
        break;
        case GGML_TYPE_F32:
            xmt_size = nelements * sizeof(int32_t);
        break;
        default: GGML_ASSERT(false && "not implemented");
    }

    int64_t count[2] = { (xmt_size / OPENSHMEM_SYMMETRIC_BUFFER_SIZE), 1 };
    const int64_t total_loop_count = count[ count[0] == 0 ];

    int64_t xmt_amount [2] = { OPENSHMEM_SYMMETRIC_BUFFER_SIZE, xmt_size - (OPENSHMEM_SYMMETRIC_BUFFER_SIZE * count[0]) };
    int64_t xmt_byte_offset = 0;
    int64_t xmt_byte_amount = 0;
 
    memcpy(dst_symmetric_comm_offset, &total_loop_count, sizeof(int64_t));

    shmem_put_signal(
        dst_symmetric_comm_offset,
        dst_symmetric_comm_offset,
        sizeof(int64_t),
        dst_recv_signal,
        1,
        SHMEM_SIGNAL_SET,
        dst_pe
    );

    shmem_wait_until(
        my_recv_signal,
        SHMEM_CMP_EQ,
        1
    );

    (*my_recv_signal) = 0;

    xmt_byte_amount = xmt_amount[0 == (total_loop_count-1)];

    for(int32_t i = 0; i < total_loop_count; ++i) {
        memcpy(dst_symmetric_comm_offset, &xmt_byte_offset, sizeof(int64_t)); 
        memcpy(dst_symmetric_comm_length, &xmt_byte_amount, sizeof(int64_t)); 
        memcpy(dst_symmetric_comm_buffer, ((uint8_t*)t->data)+xmt_byte_offset, xmt_byte_amount); 

        shmem_put_signal(
            dst_symmetric_comm_structure,
            dst_symmetric_comm_structure,
            symmetric_comm_structure_size,
            dst_recv_signal,
            1,
            SHMEM_SIGNAL_SET,
            dst_pe
        );

        shmem_wait_until(
            my_recv_signal,
            SHMEM_CMP_EQ,
            1
        );

        (*my_recv_signal) = 0;
       
        xmt_byte_offset += xmt_byte_amount;
        xmt_amount[1] -= xmt_byte_amount;
        xmt_byte_amount = xmt_amount[i == (total_loop_count-1)];
    }

    shmem_fence();
}

static void ggml_openshmem_tensor_recv(struct ggml_openshmem_context * ctx, struct ggml_tensor * t, int src_pe) {

    uint8_t * src_symmetric_comm_structure =
        ((uint8_t*)ctx->symmetric_comm_structure)+(ctx->symmetric_comm_structure_size*src_pe);
    int64_t * src_symmetric_comm_offset =
        (int64_t*)(src_symmetric_comm_structure);
    int64_t * src_symmetric_comm_length =
        ((int64_t*)src_symmetric_comm_offset)+sizeof(int64_t);
    uint8_t * src_symmetric_comm_buffer =
        ((uint8_t*)src_symmetric_comm_length)+sizeof(int64_t);
    uint64_t * src_recv_signal =
        ctx->recv_signal+src_pe;
    uint64_t * my_recv_signal =
        ctx->recv_signal+ctx->pe;

    int64_t total_loop_count = 0;

    shmem_wait_until(my_recv_signal, SHMEM_CMP_EQ, 1);
    (*my_recv_signal) = 0;

    memcpy(src_symmetric_comm_offset, &total_loop_count, sizeof(int64_t));
    shmem_put_signal(src_symmetric_comm_structure, src_symmetric_comm_structure, 0, src_recv_signal, 1, SHMEM_SIGNAL_SET, src_pe);

    for(int32_t i = 0; i < total_loop_count; ++i) {
        shmem_wait_until(my_recv_signal, SHMEM_CMP_EQ, 1);
        (*my_recv_signal) = 0;

        memcpy(
            ((uint8_t*)t->data)+(*src_symmetric_comm_offset),
            src_symmetric_comm_buffer+(*src_symmetric_comm_offset),
            (*src_symmetric_comm_length)
        );

        shmem_put_signal(src_symmetric_comm_structure, src_symmetric_comm_structure, 0, src_recv_signal, 1, SHMEM_SIGNAL_SET, src_pe);
    }

    shmem_fence();
}

// TODO: there are many improvements that can be done to this implementation
void ggml_openshmem_graph_compute_pre(
        struct ggml_openshmem_context * ctx_openshmem,
             struct ggml_cgraph * gf,
                            int   n_layers) {
    const int openshmem_rank = ctx_openshmem->pe;
    const int openshmem_size = ctx_openshmem->n_pes;

    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    if (inp_tokens == NULL) {
        fprintf(stderr, "%s: tensor 'inp_tokens' not found\n", __func__);
        return;
    }

    struct ggml_tensor * inp0 = ggml_graph_get_tensor(gf, "layer_inp_0");
    if (inp0 == NULL) {
        fprintf(stderr, "%s: tensor 'inp0' not found\n", __func__);
        return;
    }

    GGML_ASSERT(inp0 == gf->nodes[0]);

    // distribute the compute graph into slices across the MPI nodes
    //
    // the main node (0) processes the last layers + the remainder of the compute graph
    // and is responsible to pass the input tokens to the first node (1)
    //
    // node 1:   [(  0) * n_per_node, (  1) * n_per_node)
    // node 2:   [(  1) * n_per_node, (  2) * n_per_node)
    // ...
    // node n-1: [(n-2) * n_per_node, (n-1) * n_per_node)
    // node 0:   [(n-1) * n_per_node,            n_nodes)
    //
    {
        struct ggml_tensor * input_tokens[2] = { inp_tokens, inp0 };

        if (openshmem_rank > 0) {
            ggml_openshmem_tensor_recv(ctx_openshmem, input_tokens[openshmem_rank == 1], openshmem_rank-1);
        }
        else if (openshmem_size > 1) {
            // node 0 sends the input tokens to node 1
            ggml_openshmem_tensor_send(ctx_openshmem, input_tokens[0], 1);

            // recv the output data from the last node
            ggml_openshmem_tensor_recv(ctx_openshmem, input_tokens[1], openshmem_size - 1);
        }
    }

    {
        const int n_per_node = (n_layers + (openshmem_size - 1)) / openshmem_size;

        const int openshmem_idx = openshmem_rank > 0 ? openshmem_rank - 1 : openshmem_size - 1;

        const int il0 =               (openshmem_idx + 0) * n_per_node;
        const int il1 = MIN(n_layers, (openshmem_idx + 1) * n_per_node);

        char name_l0[GGML_MAX_NAME];
        char name_l1[GGML_MAX_NAME];

        snprintf(name_l0, sizeof(name_l0), "layer_inp_%d", il0);
        snprintf(name_l1, sizeof(name_l1), "layer_inp_%d", il1);

        const int idx_l0 =                ggml_graph_get_node_idx(gf, name_l0);
        const int idx_l1 = openshmem_rank > 0 ? ggml_graph_get_node_idx(gf, name_l1) + 1 : gf->n_nodes;

        if (idx_l0 < 0 || idx_l1 < 0) {
            fprintf(stderr, "%s: layer input nodes not found\n", __func__);
            return;
        }

        // attach the input data to all nodes that need it
        // TODO: not great - should be able to do this without modifying the compute graph (see next TODO below)
        for (int i = idx_l0; i < idx_l1; i++) {
            if (gf->nodes[i]->src[0] == gf->nodes[idx_l0]) {
                gf->nodes[i]->src[0] =  inp0;
            }
            if (gf->nodes[i]->src[1] == gf->nodes[idx_l0]) {
                gf->nodes[i]->src[1] =  inp0;
            }
        }

        // TODO: instead of rearranging the nodes, we should be able to execute a subset of the compute graph
        for (int i = 1; i < idx_l1 - idx_l0; i++) {
            gf->nodes[i] = gf->nodes[idx_l0 + i];
            gf->grads[i] = gf->grads[idx_l0 + i];
        }

        // the first node performs the "get_rows" operation, the rest of the nodes get the data from the previous node
        if (openshmem_idx != 0) {
            gf->nodes[0]->op = GGML_OP_NONE;
        }

        gf->n_nodes = idx_l1 - idx_l0;

        //fprintf(stderr, "%s: node %d: processing %d nodes [%d, %d)\n", __func__, openshmem_rank, gf->n_nodes, il0, il1);
    }
}

void ggml_openshmem_graph_compute_post(
        struct ggml_openshmem_context * ctx_openshmem,
             struct ggml_cgraph * gf,
                            int   n_layers) {
    UNUSED(n_layers);

    const int openshmem_rank = ctx_openshmem->pe;
    const int openshmem_size = ctx_openshmem->n_pes;

    // send the output data to the next node
    if (openshmem_rank > 0) {
        ggml_openshmem_tensor_send(ctx_openshmem, gf->nodes[gf->n_nodes - 1], (openshmem_rank + 1) % openshmem_size);
    }
}
